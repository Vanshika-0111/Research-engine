"""LLM judge for saved agent answers: citation support, faithfulness and answer correctness.

Why: the strict citation metric only counts QASPER's annotated gold passages, so a passage that clearly
supports a claim but wasn't annotated scores 0 (Day 7 review: most "misses" were like that). It also says
nothing about whether the answer itself is right. This script asks a separate LLM to read what the agent
actually cited and judge it.

For every answered question, one judge call returns:
    citations   per cited label [E#]: does that passage support the sentence it is attached to? yes/partly/no
    faithful    are ALL claims in the answer supported by the cited passages?             yes/partly/no
    correct     does the answer agree with QASPER's reference answers?                      yes/partly/no
Scores: yes = 1, partly = 0.5, no = 0.

Summary metrics (averaged over questions):
    strict_citation_precision   the existing metric (cited chunks that are QASPER gold), for comparison
    judged_citation_precision   share of citations the judge says support their sentence   (answered questions)
    faithfulness                answer fully supported by its citations                    (answered questions)
    answer_correctness          agrees with the references (a decline scores 0, or 1 if the reference
                                says "Unanswerable")                                       (all questions)
    + judge tokens and errors

It only reads a saved .jsonl from run_agent_eval, so it never re-runs the agent. Use a DIFFERENT model
from the agent (default openai/gpt-oss-20b on Groq): each model has its own daily token quota, and a judge
that is not the model being graded is less likely to favour its own answers. A model name starting with
"gemini" uses the Google Gemini API instead (needs GEMINI_API_KEY), so the judge can come from a different
model family altogether. Compare two judges with eval.compare_judges.

Every summary is reported twice: on all questions and after the reference-quality filter
(eval/reference_filter.py), which drops questions whose references only point to a citation/figure/table.

Run from the project root:
    python -m eval.judge_agent_eval results/agent_eval_dev_n20_v2.jsonl
    python -m eval.judge_agent_eval results/agent_eval_dev_n20_v2.jsonl --model qwen/qwen3.8-27b
    python -m eval.judge_agent_eval results/agent_eval_dev_n50_v2.jsonl --resume
    python -m eval.judge_agent_eval results/agent_eval_dev_n50_v2.jsonl --model gemini-3.8-flash --resume
Gemini free tier (2026-10): 5 requests/minute and 20/day per model, and failed 503 attempts count too, so a
Gemini run is spread over several days with --resume (13 s between calls; an overloaded model stops the run).
If every row is already judged, --resume makes no LLM calls and just rewrites the summary (with the filter).
Writes <input stem>_judge-<model>.jsonl (one judged row per question) and .json (summary).
Like run_agent_eval: results are saved one question at a time, --resume continues, and Groq's daily token
limit stops the run cleanly.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import mean  # noqa: E402
from eval.reference_filter import RULE, excluded_list, split_rows  # noqa: E402
from agent.llm import is_overloaded  # noqa: E402
from eval.run_agent_eval import is_daily_limit, load_rows, qkey  # noqa: E402

SCORE = {"yes": 1.0, "partly": 0.5, "no": 0.0}
_PREFIX = re.compile(r"^\[[^\]]*\]\s*")
_THINK = re.compile(r"<think>.*?</think>", re.S)
_JSON_OBJ = re.compile(r"\{.*\}", re.S)

JUDGE_SYSTEM = """You are a strict evaluator of answers about a scientific paper.

You are given a question, reference answers written by human annotators, a system's answer with citation labels
like [E2], and the full text of every passage the system cited.

Judge three things:
1. citations: for EACH cited label, does that passage support the sentence(s) it is attached to?
   "yes"    = the passage states it, or it follows directly from the passage
   "partly" = the passage supports only part of the sentence, or support needs an extra assumption
   "no"     = the passage does not support the sentence (unrelated, background only, or contradicts it)
2. faithful: are ALL factual claims in the answer supported by the cited passages?
   "yes" = every claim is supported; "partly" = some claim is unsupported, overstated or not cited;
   "no" = the main claim is unsupported
3. correct: does the answer agree with the reference answers?
   "yes" = same key facts (wording may differ; extra correct detail is fine)
   "partly" = some key facts right but others missing, wrong or about a different quantity
   "no" = contradicts the references or misses the point
   If the references disagree with each other, matching one of them is enough.

Use only the passages and the references, not your own knowledge.
Respond with JSON only, in this form:
{"citations": {"E1": "yes"}, "faithful": "yes", "correct": "yes",
 "unsupported": ["short quote of any claim that is not supported"], "note": "one short sentence"}"""

JUDGE_USER = """Question: {question}

Reference answers:
{references}

System answer:
{answer}

Cited passages:
{passages}"""


def model_slug(model):
    return model.split("/")[-1].replace(".", "")


def parse_verdict(text, labels):
    """Judge JSON -> dict with normalized yes/partly/no values; None if it can't be parsed."""
    text = _THINK.sub("", text or "")
    m = _JSON_OBJ.search(text)
    try:
        data = json.loads(m.group(0) if m else text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    def norm(v):
        v = str(v).strip().lower()
        return v if v in SCORE else None

    cites = data.get("citations") or {}
    if isinstance(cites, list):  # tolerate [{"label": "E1", "supports": "yes"}]
        cites = {str(c.get("label")): c.get("supports") for c in cites if isinstance(c, dict)}
    cites = {lab: norm(cites.get(lab)) for lab in labels}
    out = {"citations": cites, "faithful": norm(data.get("faithful")), "correct": norm(data.get("correct")),
           "unsupported": [str(u) for u in (data.get("unsupported") or []) if str(u).strip()],
           "note": str(data.get("note", "")).strip()}
    if out["correct"] is None or (labels and out["faithful"] is None):
        return None
    return out


def is_unanswerable(refs):
    return bool(refs) and all("unanswerable" in r.lower() for r in refs)


def summarize_judged(judged):
    """Averages over judged rows (also used by eval.reference_filter for the filtered numbers)."""
    n = len(judged)
    answered = [j for j in judged if not j["declined"]]
    return {
        "strict_citation_precision": mean(j["strict_citation_precision"] for j in answered),
        "judged_citation_precision": mean(j["scores"]["judged_citation_precision"] for j in answered),
        "faithfulness": mean(j["scores"]["faithfulness"] for j in answered),
        "answer_correctness": mean(j["scores"]["answer_correctness"] for j in judged),
        "answer_correctness_answered_only": mean(j["scores"]["answer_correctness"] for j in answered),
        "declined_rate": round(sum(j["declined"] for j in judged) / n, 4),
        "judge_errors": sum(bool(j["judge"].get("error")) for j in judged),
        "avg_judge_prompt_tokens": round(sum(j["judge_tokens"]["prompt"] for j in answered) / max(len(answered), 1)),
        "avg_judge_completion_tokens": round(sum(j["judge_tokens"]["completion"] for j in answered) / max(len(answered), 1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="a results/agent_eval_*.jsonl file from run_agent_eval")
    ap.add_argument("--model", default="openai/gpt-oss-20b", help="judge model, not the agent's (gemini-* = Gemini API, else Groq)")
    ap.add_argument("--chars", type=int, default=2500, help="max characters of each cited passage")
    ap.add_argument("--max-tokens", type=int, default=1536, help="judge completion budget (includes reasoning)")
    ap.add_argument("--pause", type=float, default=None,
                    help="seconds between calls (default 1; 13 for gemini-*, whose free tier allows 5 requests/minute)")
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
        raise SystemExit(f"{out_path} already has {len(judged)} judged rows. Use --resume, --overwrite, or another --model.")
    if args.overwrite:
        judged = []
    wanted = {qkey(r) for r in rows}
    judged = [j for j in judged if qkey(j) in wanted and not j["judge"].get("error")]  # retry failed judgements
    done = {qkey(j) for j in judged}
    print(f"{len(rows)} answers in {src.name} | judge={args.model}"
          + (f" | resuming: {len(done)} already judged" if done else ""))

    todo = [r for r in rows if qkey(r) not in done]
    llm = text = section = None
    if todo:   # only load the chunks and the API client if there is something to judge
        from retrieval.data import load_chunks
        chunks = load_chunks()
        text = dict(zip(chunks["chunk_id"].astype(str), chunks["text"]))
        section = dict(zip(chunks["chunk_id"].astype(str), chunks["section"]))

        from agent.llm import make_llm
        llm = make_llm(args.model, temperature=0.0, reasoning_effort="low")

    stopped = None
    with open(out_path, "w", encoding="utf-8") as fh:
        for j in judged:
            fh.write(json.dumps(j, ensure_ascii=False) + "\n")
        fh.flush()
        for i, r in enumerate(rows, 1):
            if qkey(r) in done:
                continue
            refs = r.get("reference_answers") or []
            label_of = {cid: f"E{n}" for n, cid in enumerate(r["evidence_chunk_ids"], start=1)}
            labels = [label_of[c] for c in r["cited_chunk_ids"] if c in label_of]
            base = {"question_id": r.get("question_id"), "paper_id": r["paper_id"], "question": r["question"],
                    "answer": r["answer"], "reference_answers": refs, "cited_labels": labels,
                    "declined": r.get("declined", False),
                    "strict_citation_precision": r["metrics"]["citation_precision"]}

            usage = {"prompt_tokens": 0, "completion_tokens": 0}
            if r.get("declined"):
                # nothing to check for support; a decline is correct only when the reference says Unanswerable
                verdict = {"citations": {}, "faithful": None, "correct": "yes" if is_unanswerable(refs) else "no",
                           "unsupported": [], "note": "declined"}
            else:
                passages = "\n\n".join(
                    f"[{label_of[c]}] (Section: {section.get(c, '?')})\n"
                    + _PREFIX.sub("", text.get(c, "(missing)"), count=1)[:args.chars]
                    for c in r["cited_chunk_ids"] if c in label_of) or "(none: the answer cites nothing)"
                msgs = [{"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": JUDGE_USER.format(
                            question=r["question"], references="\n".join(f"- {x}" for x in refs) or "- (none)",
                            answer=r["answer"], passages=passages)}]
                try:
                    reply, usage = llm.chat(msgs, max_tokens=args.max_tokens, json_mode=True)
                except Exception as e:
                    if not (is_daily_limit(e) or is_overloaded(e)):
                        raise
                    stopped = str(e)
                    why = "daily quota reached" if is_daily_limit(e) else "the model is overloaded (5xx after retries)"
                    print(f"\nSTOPPED at {i}/{len(rows)}: {why} for {args.model}. "
                          f"{len(judged)} judged rows saved; run again later with --resume.")
                    break
                verdict = parse_verdict(reply, labels)
                if verdict is None:
                    verdict = {"citations": {l: None for l in labels}, "faithful": None, "correct": None,
                               "unsupported": [], "note": "", "error": "unparseable judge reply",
                               "raw": (reply or "")[:500]}
                elif not labels:
                    verdict["faithful"] = "no"   # an answer with no valid citation is not grounded
                time.sleep(args.pause)

            cite_scores = [SCORE[v] for v in verdict["citations"].values() if v in SCORE]
            row = {**base, "judge": verdict, "judge_model": args.model,
                   "scores": {"judged_citation_precision": (sum(cite_scores) / len(cite_scores)) if cite_scores else None,
                              "faithfulness": SCORE.get(verdict["faithful"]),
                              "answer_correctness": SCORE.get(verdict["correct"])},
                   "judge_tokens": {"prompt": usage.get("prompt_tokens", 0), "completion": usage.get("completion_tokens", 0)}}
            judged.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            s = row["scores"]
            fmt = lambda v: "-" if v is None else f"{v:.2f}"   # noqa: E731
            strict = base["strict_citation_precision"]
            print(f"[{i:>3}/{len(rows)}] strict={fmt(strict)} judged={fmt(s['judged_citation_precision'])} "
                  f"faithful={verdict['faithful'] or '-'} correct={verdict['correct'] or '-'}"
                  f"{'  ERROR' if verdict.get('error') else ''}  {r['question'][:55]}")

    n = len(judged)
    if n == 0:
        raise SystemExit("Nothing judged.")
    order = {qkey(r): k for k, r in enumerate(rows)}
    judged.sort(key=lambda j: order[qkey(j)])
    summary = summarize_judged(judged)
    kept, _ = split_rows(judged)
    summary_filtered = summarize_judged(kept) if kept else {}
    stem.with_suffix(".json").write_text(json.dumps(
        {"source": src.name, "judge_model": args.model, "n": n, "n_planned": len(rows), "complete": n == len(rows),
         "stopped_reason": stopped, "summary": summary,
         "reference_filter": {"rule": RULE, "n_kept": len(kept), "excluded": excluded_list(judged)},
         "summary_filtered": summary_filtered}, indent=2))

    status = "" if n == len(rows) else f"  (PARTIAL: {n} of {len(rows)}; finish with --resume)"
    print(f"\njudge={args.model}  n={n}{status}")
    print(f"| metric | full (n={n}) | reference-filtered (n={len(kept)}) |\n|---|---|---|")
    for k, v in summary.items():
        print(f"| {k} | {v} | {summary_filtered.get(k)} |")

    flagged = [j for j in judged if j["scores"]["answer_correctness"] not in (None, 1.0) and not j["declined"]
               or j["scores"]["faithfulness"] not in (None, 1.0)]
    if flagged:
        print("\nAnswers the judge marked not fully correct or not fully faithful:")
        for j in flagged:
            print(f"- {j['question'][:70]} | faithful={j['judge']['faithful']} correct={j['judge']['correct']}"
                  f" | {j['judge'].get('note') or j['judge'].get('error', '')}")
    print(f"\nSaved {stem.with_suffix('.json')} and {out_path}")


if __name__ == "__main__":
    main()
