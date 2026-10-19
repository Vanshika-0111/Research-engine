"""LLM judge for cross-paper answers: does every citation support its sentence, is the answer faithful?

Cross-paper questions have no reference answers, so (unlike eval/judge_agent_eval.py) there is no correctness
verdict. One judge call per answered question returns:
    citations   per cited label [E#]: does that passage support the sentence it is attached to? yes/partly/no
    faithful    are ALL claims supported by the cited passages?                                 yes/partly/no
    per_paper   for each paper the answer discusses: is what it says about that paper supported? yes/partly/no
Scores: yes = 1, partly = 0.5, no = 0. Paper coverage is measured without an LLM by run_cross_paper_agent_eval.

Run from the project root (default judge openai/gpt-oss-20b on Groq, a different model from the agent):
    python -m eval.judge_cross_paper results/cross_paper_agent.jsonl
    python -m eval.judge_cross_paper results/cross_paper_agent.jsonl --resume
Writes <input stem>_judge-<model>.jsonl and .json.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.llm import is_daily_limit, is_overloaded, make_llm  # noqa: E402
from eval.judge_agent_eval import _JSON_OBJ, _THINK, SCORE, model_slug  # noqa: E402
from eval.metrics import mean  # noqa: E402
from eval.run_agent_eval import load_rows  # noqa: E402

JUDGE_SYSTEM = """You are a strict evaluator of answers that summarise SEVERAL scientific papers.

You are given a question, a system's answer with citation labels like [E2], and the full text of every passage
the system cited. Each passage is marked with the paper it comes from.

Judge:
1. citations: for EACH cited label, does that passage support the sentence(s) it is attached to?
   "yes" = the passage states it or it follows directly; "partly" = supports only part of the sentence or needs
   an extra assumption; "no" = does not support it (unrelated, about another paper's work, or contradicts it)
2. faithful: are ALL factual claims in the answer supported by the cited passages?
   "yes" = every claim supported; "partly" = some claim unsupported, overstated or uncited; "no" = mostly unsupported
3. per_paper: for each paper the answer discusses (use its paper id), is what the answer says about THAT paper
   supported by passages from that same paper? yes/partly/no

Use only the passages, not your own knowledge.
Respond with JSON only, in this form:
{"citations": {"E1": "yes"}, "faithful": "yes", "per_paper": {"1909.00694": "yes"},
 "unsupported": ["short quote of any unsupported claim"], "note": "one short sentence"}"""

JUDGE_USER = """Question: {question}

System answer:
{answer}

Cited passages:
{passages}"""


_LABEL_KEY = re.compile(r"E\d+", re.IGNORECASE)


def _norm_key(k):
    """'[E4]', ' e4 ', '【E4】' -> 'E4' (judges sometimes copy the brackets into the JSON keys)."""
    m = _LABEL_KEY.search(str(k))
    return m.group(0).upper() if m else str(k).strip()


def parse(text, labels):
    """Returns the verdict dict, or None if the reply can't be used. Since 2026-10-18 a reply that leaves any cited
    label without a yes/partly/no verdict also returns None (it is then saved as a judge error and re-judged by
    --resume): before, such labels were silently dropped from the citation score (XP15: 1 of 8 labels judged)."""
    text = _THINK.sub("", text or "")
    m = _JSON_OBJ.search(text)
    try:
        data = json.loads(m.group(0) if m else text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    norm = lambda v: (str(v).strip().lower() if str(v).strip().lower() in SCORE else None)  # noqa: E731
    cites = data.get("citations") or {}
    if isinstance(cites, list):
        cites = {str(c.get("label")): c.get("supports") for c in cites if isinstance(c, dict)}
    cites = {_norm_key(k): v for k, v in cites.items()} if isinstance(cites, dict) else {}
    per_paper = data.get("per_paper") or {}
    out = {"citations": {l: norm(cites.get(l)) for l in labels}, "faithful": norm(data.get("faithful")),
           "per_paper": {str(k): norm(v) for k, v in per_paper.items()} if isinstance(per_paper, dict) else {},
           "unsupported": [str(u) for u in (data.get("unsupported") or []) if str(u).strip()],
           "note": str(data.get("note", "")).strip()}
    if out["faithful"] is None or any(v is None for v in out["citations"].values()):
        return None
    return out


def summarize(judged):
    answered = [j for j in judged if not j["declined"]]
    return {"judged_citation_precision": mean(j["scores"]["judged_citation_precision"] for j in answered),
            "faithfulness": mean(j["scores"]["faithfulness"] for j in answered),
            "per_paper_support": mean(j["scores"]["per_paper_support"] for j in answered),
            "declined_rate": round(sum(j["declined"] for j in judged) / len(judged), 4),
            "judge_errors": sum(bool(j["judge"].get("error")) for j in judged),
            "avg_judge_prompt_tokens": round(sum(j["judge_tokens"]["prompt"] for j in answered) / max(len(answered), 1)),
            "avg_judge_completion_tokens": round(sum(j["judge_tokens"]["completion"] for j in answered)
                                                 / max(len(answered), 1))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("jsonl", help="results/cross_paper_agent*.jsonl")
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--chars", type=int, default=2500)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--pause", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.pause is None:
        args.pause = 13.0 if args.model.lower().startswith("gemini") else 1.0

    src = Path(args.jsonl)
    rows = load_rows(src)
    if not rows:
        raise SystemExit(f"No rows in {src}")
    stem = src.with_name(f"{src.stem}_judge-{model_slug(args.model)}")
    out_path = stem.with_suffix(".jsonl")
    judged = load_rows(out_path) if out_path.exists() else []
    if judged and not (args.resume or args.overwrite):
        raise SystemExit(f"{out_path} already has {len(judged)} rows. Use --resume or --overwrite.")
    if args.overwrite:
        judged = []
    ids = {r["id"] for r in rows}
    # re-judge errors AND rows saved before the 2026-10-18 fix with a cited label left without a verdict
    incomplete = lambda j: any(v is None for v in j["judge"].get("citations", {}).values())  # noqa: E731
    redo = [j["id"] for j in judged if j["id"] in ids and (j["judge"].get("error") or incomplete(j))]
    if redo:
        print(f"re-judging {len(redo)} rows with an error or a missing citation verdict: {', '.join(redo)}")
    judged = [j for j in judged if j["id"] in ids and not j["judge"].get("error") and not incomplete(j)]
    done = {j["id"] for j in judged}
    todo = [r for r in rows if r["id"] not in done]
    llm = make_llm(args.model, temperature=0.0, reasoning_effort="low") if todo else None
    print(f"{len(rows)} answers | judge={args.model}" + (f" | resuming: {len(done)} judged" if done else ""))

    stopped = None
    with open(out_path, "w", encoding="utf-8") as fh:
        for j in judged:
            fh.write(json.dumps(j, ensure_ascii=False) + "\n")
        fh.flush()
        for r in rows:
            if r["id"] in done:
                continue
            ev = {e["label"]: e for e in r["evidence"]}
            labels = [l for l in r["cited_labels"] if l in ev]
            usage = {"prompt_tokens": 0, "completion_tokens": 0}
            if r["declined"]:
                verdict = {"citations": {}, "faithful": None, "per_paper": {}, "unsupported": [], "note": "declined"}
            else:
                passages = "\n\n".join(f"[{l}] (paper {ev[l]['paper_id']}: {ev[l]['title']}; section: {ev[l]['section']})\n"
                                       + ev[l]["text"][:args.chars] for l in labels) or "(none: the answer cites nothing)"
                msgs = [{"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": JUDGE_USER.format(question=r["question"], answer=r["answer"],
                                                                      passages=passages)}]
                try:
                    reply, usage = llm.chat(msgs, max_tokens=args.max_tokens, json_mode=True)
                except Exception as e:
                    if not (is_daily_limit(e) or is_overloaded(e)):
                        raise
                    stopped = str(e)
                    print(f"\nSTOPPED at {r['id']}: quota or overload for {args.model}. "
                          f"{len(judged)} judged rows saved; run again later with --resume.")
                    break
                verdict = parse(reply, labels)
                if verdict is None:   # one retry: unparseable reply or a cited label left without a verdict
                    time.sleep(args.pause)
                    try:
                        reply, usage = llm.chat(msgs + [{"role": "assistant", "content": reply or ""},
                                                        {"role": "user", "content":
                                                         "Your JSON must give a yes/partly/no verdict for EVERY one of "
                                                         f"these labels: {', '.join(labels)}. Reply with the full JSON "
                                                         "again."}], max_tokens=args.max_tokens, json_mode=True)
                    except Exception as e:
                        if not (is_daily_limit(e) or is_overloaded(e)):
                            raise
                        stopped = str(e)
                        print(f"\nSTOPPED at {r['id']}: quota or overload for {args.model}. "
                              f"{len(judged)} judged rows saved; run again later with --resume.")
                        break
                    verdict = parse(reply, labels)
                verdict = verdict or {"citations": {l: None for l in labels}, "faithful": None,
                                      "per_paper": {}, "unsupported": [], "note": "",
                                      "error": "unparseable judge reply or a cited label without a verdict",
                                      "raw": (reply or "")[:500]}
                if not labels and not verdict.get("error"):
                    verdict["faithful"] = "no"   # an answer that cites nothing is not grounded
                time.sleep(args.pause)
            cs = [SCORE[v] for v in verdict["citations"].values() if v in SCORE]
            ps = [SCORE[v] for v in verdict["per_paper"].values() if v in SCORE]
            row = {"id": r["id"], "type": r["type"], "question": r["question"], "answer": r["answer"],
                   "declined": r["declined"], "judge": verdict, "judge_model": args.model,
                   "scores": {"judged_citation_precision": sum(cs) / len(cs) if cs else None,
                              "faithfulness": SCORE.get(verdict["faithful"]),
                              "per_paper_support": sum(ps) / len(ps) if ps else None},
                   "judge_tokens": {"prompt": usage.get("prompt_tokens", 0),
                                    "completion": usage.get("completion_tokens", 0)}}
            judged.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            s = row["scores"]
            f = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
            print(f"[{r['id']}] cite_support={f(s['judged_citation_precision'])} faithful={verdict['faithful'] or '-'} "
                  f"per_paper={f(s['per_paper_support'])}{'  ERROR' if verdict.get('error') else ''}")

    if not judged:
        raise SystemExit("Nothing judged.")
    order = {r["id"]: k for k, r in enumerate(rows)}
    judged.sort(key=lambda j: order[j["id"]])
    summary = {"all": summarize(judged)}
    for typ in ("dataset", "topic"):
        sub = [j for j in judged if j["type"] == typ]
        if sub:
            summary[typ] = summarize(sub)
    stem.with_suffix(".json").write_text(json.dumps(
        {"source": src.name, "judge_model": args.model, "n": len(judged), "n_planned": len(rows),
         "complete": len(judged) == len(rows), "stopped_reason": stopped, "summary": summary}, indent=2))
    print(f"\njudge={args.model} n={len(judged)}")
    print("| metric | " + " | ".join(summary) + " |\n" + "|---" * (len(summary) + 1) + "|")
    for k in summary["all"]:
        print(f"| {k} | " + " | ".join(str(summary[s].get(k)) for s in summary) + " |")
    flagged = [j for j in judged if j["scores"]["faithfulness"] not in (None, 1.0)]
    if flagged:
        print("\nNot fully faithful:")
        for j in flagged:
            print(f"- {j['id']} faithful={j['judge']['faithful']} | {j['judge'].get('note') or j['judge'].get('error', '')}")
    print(f"\nSaved {stem.with_suffix('.json')} and {out_path}")


if __name__ == "__main__":
    main()
