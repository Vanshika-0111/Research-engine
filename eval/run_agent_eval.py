"""Evaluate the Day 6 research agent on a random sample of QASPER questions.

For each question the agent decomposes, retrieves within the paper and writes a cited answer.
Citations are checked against QASPER's gold evidence chunks.

Metrics (averaged over questions):
    citation_precision   share of cited chunks that are gold evidence  (the CV "citation accuracy" metric;
                         strict: a cited chunk can support a claim without being annotated as gold)
    citation_hit         at least one cited chunk is gold
    citation_recall      share of gold chunks the answer cited
    evidence_recall      share of gold chunks that reached the LLM (agent's merged evidence)
    single_query_recall  same, for plain retrieve(question, k=max_evidence) with no decomposition
    no_citation_rate     answers with no valid [E#] citation
    declined_rate        answers saying the evidence does not answer the question
    tokens / latency     per question

Run from the project root (needs a GROQ_API_KEY env var or Colab secret):
    python -m eval.run_agent_eval --n 5        # smoke test
    python -m eval.run_agent_eval --n 20       # Day 6 number (dev)
    python -m eval.run_agent_eval --n 20 --tag v2   # same questions, saved as agent_eval_dev_n20_v2.*
    python -m eval.run_agent_eval --n 50 --tag v2 --resume   # continue a run that stopped part-way
    python -m eval.run_agent_eval --n 50 --split test   # once, when prompts are final
Outputs results/agent_eval_<split>_n<N>.json (summary) and .jsonl (every question, answer and citation).

The .jsonl is written one question at a time, so a run that stops (e.g. Groq's daily token limit) keeps
everything done so far. --resume skips questions already in the .jsonl and appends the rest; the summary
always covers every row in the file. Without --resume an existing non-empty .jsonl is never overwritten
unless --overwrite is given. If Groq's daily token limit is hit, the run stops cleanly and saves a partial
summary ("complete": false).
"""
import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python eval/run_agent_eval.py`

from agent.config import AgentConfig  # noqa: E402
from agent.graph import NO_ANSWER, ResearchAgent  # noqa: E402
from eval.metrics import citation_metrics, mean  # noqa: E402
from retrieval.config import RESULTS_DIR  # noqa: E402
from retrieval.data import load_questions  # noqa: E402
from retrieval.pipeline import get_pipeline  # noqa: E402


def qkey(q):
    """Identifies a question in the sample and in saved rows."""
    return q.get("question_id") or f'{q["paper_id"]}|{q["question"]}'


def load_rows(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a line cut off mid-write when a run was interrupted
    return rows


def is_daily_limit(e):
    """Groq's per-day token/request limit (waiting a few seconds won't help, so stop and save)."""
    msg = str(e).lower()
    return "rate_limit" in msg and ("per day" in msg or "tpd" in msg or "rpd" in msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", help="tune on dev; run test once at the end")
    ap.add_argument("--n", type=int, default=20, help="number of questions to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None, help="override the Groq model ID")
    ap.add_argument("--pause", type=float, default=2.0, help="seconds between questions (free-tier rate limits)")
    ap.add_argument("--tag", default="", help="suffix for the output files, e.g. v2, so earlier runs are kept")
    ap.add_argument("--resume", action="store_true", help="skip questions already in the .jsonl and append")
    ap.add_argument("--overwrite", action="store_true", help="start again even if the .jsonl has results")
    args = ap.parse_args()

    cfg = AgentConfig()
    if args.model:
        cfg.llm_model = args.model
    pipe = get_pipeline()
    questions = load_questions(args.split, set(pipe.chunk_ids))
    sample = random.Random(args.seed).sample(questions, min(args.n, len(questions)))
    agent = ResearchAgent(cfg, pipeline=pipe)
    print(f"{len(sample)} {args.split} questions | model={cfg.llm_model}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = RESULTS_DIR / f"agent_eval_{args.split}_n{len(sample)}{'_' + args.tag if args.tag else ''}"
    jsonl = stem.with_suffix(".jsonl")
    rows = load_rows(jsonl) if jsonl.exists() else []
    if rows and not (args.resume or args.overwrite):
        raise SystemExit(f"{jsonl} already has {len(rows)} results. Use --resume to continue it, "
                         f"--overwrite to start again, or a different --tag.")
    if args.overwrite:
        rows = []
    wanted = {qkey(q) for q in sample}
    rows = [r for r in rows if qkey(r) in wanted]
    done = {qkey(r) for r in rows}
    todo = [q for q in sample if qkey(q) not in done]
    if rows:
        print(f"resuming: {len(rows)} already done, {len(todo)} to go")
    stopped = None
    with open(jsonl, "w", encoding="utf-8") as fh:
        for r in rows:   # rewrite kept rows (drops any line cut off by an interrupted run)
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fh.flush()
        for i, q in enumerate(sample, 1):
            if qkey(q) in done:
                continue
            try:
                out = agent.run(q["question"], q["paper_id"])
            except Exception as e:
                if not is_daily_limit(e):
                    raise
                stopped = str(e)
                print(f"\nSTOPPED at question {i}/{len(sample)}: Groq daily token limit reached for {cfg.llm_model}.")
                print(f"{len(rows)} results are saved. Later (when the limit has reset) run the same command with --resume.")
                break
            ev_ids = [e["chunk_id"] for e in out["evidence"]]
            single = [h["chunk_id"] for h in pipe.retrieve(q["question"], q["paper_id"], k=cfg.max_evidence)]
            m = citation_metrics(out["cited_chunk_ids"], q["gold"], ev_ids)
            m["single_query_recall"] = len(set(single) & set(q["gold"])) / len(q["gold"])
            row = {
                "question_id": q.get("question_id"), "paper_id": q["paper_id"], "question": q["question"],
                "sub_questions": out["sub_questions"], "evidence_chunk_ids": ev_ids,
                "answer": out["answer"], "cited_chunk_ids": out["cited_chunk_ids"],
                "invalid_labels": out.get("invalid_labels", []), "gold_chunk_ids": q["gold"],
                "reference_answers": q.get("reference_answers", []),
                "declined": out["answer"].strip() == NO_ANSWER,
                "tokens": out["tokens"], "secs": out["secs"], "metrics": m,
            }
            rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            p = m["citation_precision"]
            print(f"[{i:>3}/{len(sample)}] cite_prec={'-' if p is None else f'{p:.2f}'} "
                  f"ev_rec={m['evidence_recall']:.2f} tokens={sum(out['tokens'].values())} {out['secs']}s"
                  f"  {q['question'][:60]}")
            time.sleep(args.pause)

    n = len(rows)
    if n == 0:
        raise SystemExit("No results to summarise.")
    order = {qkey(q): j for j, q in enumerate(sample)}
    rows.sort(key=lambda r: order[qkey(r)])
    summary = {k: mean(r["metrics"][k] for r in rows)
               for k in ("citation_precision", "citation_hit", "citation_recall", "evidence_recall",
                         "single_query_recall")}
    summary.update({
        "no_citation_rate": round(sum(not r["cited_chunk_ids"] for r in rows) / n, 4),
        "declined_rate": round(sum(r["declined"] for r in rows) / n, 4),
        "invalid_citation_rate": round(sum(bool(r["invalid_labels"]) for r in rows) / n, 4),
        "avg_sub_questions": round(sum(len(r["sub_questions"]) for r in rows) / n, 2),
        "avg_prompt_tokens": round(sum(r["tokens"]["prompt"] for r in rows) / n),
        "avg_completion_tokens": round(sum(r["tokens"]["completion"] for r in rows) / n),
        "avg_secs": round(sum(r["secs"] for r in rows) / n, 2),
    })
    stem.with_suffix(".json").write_text(json.dumps(
        {"split": args.split, "n": n, "n_planned": len(sample), "complete": n == len(sample),
         "stopped_reason": stopped, "seed": args.seed,
         "tag": args.tag, "config": asdict(cfg), "summary": summary}, indent=2))

    status = "" if n == len(sample) else f"  (PARTIAL: {n} of {len(sample)} done; finish with --resume)"
    print(f"\nsplit={args.split}  n={n}  model={cfg.llm_model}{status}")
    print("| metric | value |\n|---|---|")
    for k, v in summary.items():
        print(f"| {k} | {v} |")
    print(f"\nSaved {stem.with_suffix('.json')} and {stem.with_suffix('.jsonl')}")


if __name__ == "__main__":
    main()
