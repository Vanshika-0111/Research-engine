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
    python -m eval.run_agent_eval --n 50 --split test   # once, when prompts are final
Outputs results/agent_eval_<split>_n<N>.json (summary) and .jsonl (every question, answer and citation).
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", help="tune on dev; run test once at the end")
    ap.add_argument("--n", type=int, default=20, help="number of questions to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None, help="override the Groq model ID")
    ap.add_argument("--pause", type=float, default=2.0, help="seconds between questions (free-tier rate limits)")
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
    stem = RESULTS_DIR / f"agent_eval_{args.split}_n{len(sample)}"
    rows = []
    with open(stem.with_suffix(".jsonl"), "w", encoding="utf-8") as fh:
        for i, q in enumerate(sample, 1):
            out = agent.run(q["question"], q["paper_id"])
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
        {"split": args.split, "n": n, "seed": args.seed, "config": asdict(cfg), "summary": summary}, indent=2))

    print(f"\nsplit={args.split}  n={n}  model={cfg.llm_model}")
    print("| metric | value |\n|---|---|")
    for k, v in summary.items():
        print(f"| {k} | {v} |")
    print(f"\nSaved {stem.with_suffix('.json')} and {stem.with_suffix('.jsonl')}")


if __name__ == "__main__":
    main()
