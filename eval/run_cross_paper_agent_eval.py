"""Run the cross-paper agent on the 19 hand-written questions and score it per paper.

Each question in eval/cross_paper_questions.jsonl lists required papers (must be covered) and acceptable papers
(fine to mention). There are no reference answers, so answers are scored on papers and citations here, and
eval/judge_cross_paper.py checks whether each citation supports its sentence.

Metrics (averaged over questions):
    retrieved_paper_recall  required papers among the papers shown to the LLM
    cited_paper_recall      required papers the answer actually cites (coverage of the final answer)
    cited_paper_precision   cited papers that are required or acceptable (None if nothing is cited)
    avg_cited_papers, declined_rate, invalid_citation_rate, tokens and seconds per question

Run from the project root (needs GROQ_API_KEY):
    python -m eval.run_cross_paper_agent_eval                 # all 19 questions, gpt-oss-120b
    python -m eval.run_cross_paper_agent_eval --limit 3       # quick try
    python -m eval.run_cross_paper_agent_eval --resume        # continue after a daily-limit stop
Outputs results/cross_paper_agent[_tag].jsonl (one row per question, written as it goes) and .json (summary).
"""
import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.cross_paper import CrossPaperAgent, CrossPaperAgentConfig  # noqa: E402
from agent.graph import NO_ANSWER  # noqa: E402
from agent.llm import is_daily_limit  # noqa: E402
from eval.metrics import mean  # noqa: E402
from eval.run_agent_eval import load_rows  # noqa: E402
from eval.run_cross_paper_eval import QUESTIONS, load_questions  # noqa: E402
from retrieval.config import RESULTS_DIR  # noqa: E402


def paper_scores(row, q):
    req, ok = set(q["required_papers"]), set(q["required_papers"]) | set(q["acceptable_papers"])
    cited, shown = row["cited_papers"], [p["paper_id"] for p in row["papers"]]
    return {"retrieved_paper_recall": len(req & set(shown)) / len(req),
            "cited_paper_recall": len(req & set(cited)) / len(req),
            "cited_paper_precision": (len(ok & set(cited)) / len(cited)) if cited else None,
            "n_cited_papers": len(cited)}


def summarize(rows):
    n = len(rows)
    out = {k: mean(r["metrics"][k] for r in rows)
           for k in ("retrieved_paper_recall", "cited_paper_recall", "cited_paper_precision")}
    out.update({
        "avg_cited_papers": round(sum(r["metrics"]["n_cited_papers"] for r in rows) / n, 2),
        "declined_rate": round(sum(r["declined"] for r in rows) / n, 4),
        "invalid_citation_rate": round(sum(bool(r["invalid_labels"]) for r in rows) / n, 4),
        "avg_prompt_tokens": round(sum(r["tokens"]["prompt"] for r in rows) / n),
        "avg_completion_tokens": round(sum(r["tokens"]["completion"] for r in rows) / n),
        "avg_secs": round(sum(r["secs"] for r in rows) / n, 2),
    })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--questions", default=str(QUESTIONS))
    ap.add_argument("--model", default=None, help="override the LLM (default openai/gpt-oss-120b)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions")
    ap.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder (not recommended)")
    ap.add_argument("--pause", type=float, default=2.0, help="seconds between questions")
    ap.add_argument("--prompt", default="v1", choices=["v1", "v2"],
                    help="synthesis prompt (v2 = include every relevant paper, cited comparisons); use with --tag v2")
    ap.add_argument("--tag", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    qs = load_questions(args.questions)[:args.limit] if args.limit else load_questions(args.questions)
    if args.prompt != "v1" and not args.tag:
        raise SystemExit("Use --tag with a non-default prompt (e.g. --prompt v2 --tag v2) so the v1 results are kept.")
    cfg = CrossPaperAgentConfig(rerank=not args.no_rerank, prompt=args.prompt)
    if args.model:
        cfg.llm_model = args.model
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = RESULTS_DIR / f"cross_paper_agent{('_' + args.tag) if args.tag else ''}"
    jsonl = stem.with_suffix(".jsonl")
    rows = load_rows(jsonl) if jsonl.exists() else []
    if rows and not (args.resume or args.overwrite):
        raise SystemExit(f"{jsonl} already has {len(rows)} results. Use --resume, --overwrite or another --tag.")
    if args.overwrite:
        rows = []
    wanted = {q["id"] for q in qs}
    rows = [r for r in rows if r["id"] in wanted]
    done = {r["id"] for r in rows}
    agent = CrossPaperAgent(cfg)
    print(f"{len(qs)} cross-paper questions | model={cfg.llm_model} | prompt={cfg.prompt} | rerank={cfg.rerank} | "
          f"{cfg.max_papers} papers x {cfg.chunks_per_paper} passages" + (f" | resuming: {len(done)} done" if done else ""))

    stopped = None
    with open(jsonl, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fh.flush()
        for i, q in enumerate(qs, 1):
            if q["id"] in done:
                continue
            try:
                out = agent.run(q["question"])
            except Exception as e:
                if not is_daily_limit(e):
                    raise
                stopped = str(e)
                print(f"\nSTOPPED at {q['id']}: daily quota reached for {cfg.llm_model}. "
                      f"{len(rows)} results saved; run again later with --resume.")
                break
            row = {"id": q["id"], "type": q["type"], "question": q["question"],
                   "papers": out["papers"], "evidence": out["evidence"], "answer": out["answer"],
                   "cited_labels": out["cited_labels"], "cited_chunk_ids": out["cited_chunk_ids"],
                   "cited_papers": out["cited_papers"], "invalid_labels": out["invalid_labels"],
                   "declined": out["answer"].strip() == NO_ANSWER,
                   "required_papers": q["required_papers"], "acceptable_papers": q["acceptable_papers"],
                   "tokens": out["tokens"], "secs": out["secs"]}
            row["metrics"] = paper_scores(row, q)
            rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            m = row["metrics"]
            prec = "-" if m["cited_paper_precision"] is None else f"{m['cited_paper_precision']:.2f}"
            print(f"[{q['id']}] shown_rec={m['retrieved_paper_recall']:.2f} cited_rec={m['cited_paper_recall']:.2f} "
                  f"cited_prec={prec} papers_cited={m['n_cited_papers']} "
                  f"tokens={sum(out['tokens'].values())} {out['secs']}s  {q['question'][:50]}")
            time.sleep(args.pause)

    if not rows:
        raise SystemExit("No results to summarise.")
    order = {q["id"]: k for k, q in enumerate(qs)}
    rows.sort(key=lambda r: order[r["id"]])
    summary = {"all": summarize(rows)}
    for typ in ("dataset", "topic"):
        sub = [r for r in rows if r["type"] == typ]
        if sub:
            summary[typ] = summarize(sub)
    stem.with_suffix(".json").write_text(json.dumps(
        {"n": len(rows), "n_planned": len(qs), "complete": len(rows) == len(qs), "stopped_reason": stopped,
         "config": asdict(cfg), "summary": summary}, indent=2))
    status = "" if len(rows) == len(qs) else f"  (PARTIAL: {len(rows)} of {len(qs)}; finish with --resume)"
    print(f"\nn={len(rows)}{status}")
    cols = list(summary["all"])
    print("| metric | " + " | ".join(summary) + " |\n" + "|---" * (len(summary) + 1) + "|")
    for k in cols:
        print(f"| {k} | " + " | ".join(str(summary[s].get(k)) for s in summary) + " |")
    print(f"\nSaved {stem.with_suffix('.json')} and {jsonl}")


if __name__ == "__main__":
    main()
