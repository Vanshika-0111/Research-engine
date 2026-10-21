"""Token efficiency across every evaluated system, read from the saved summary JSONs (no LLM calls).

    python -m eval.token_report        # prints the table and writes results/token_efficiency.md

For each run: LLM calls per question (by design), agent prompt + completion tokens per question, judge tokens per
question, the agent's total tokens for the run, the quality the tokens bought (judged answer correctness or
citation support) and agent tokens per correct answer. Tokens include the reasoning model's hidden reasoning.
Latency is listed only where the whole run was on a GPU (the test run's first 19 questions ran on a CPU).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval.config import RESULTS_DIR  # noqa: E402

RUNS = [  # (results stem, label, split, LLM calls per question, latency comparable?)
    ("agent_eval_test_n100_v2", "one-paper agent (frozen, Day 7 v2)", "test, n=100", "2 (decompose, answer)", False),
    ("agent_eval_dev_n50_v2", "one-paper agent (frozen, Day 7 v2)", "dev, n=50", "2", True),
    ("agent_eval_dev_n50_v3hop", "multi-hop agent (Day 14)", "dev, n=50", "3 (+ sufficiency check)", True),
    ("cross_paper_agent", "cross-paper agent v1", "19 questions", "1 (synthesize)", True),
    ("cross_paper_agent_v2", "cross-paper agent v2", "19 questions", "1", True),
]


def _summary(path):
    if not path.exists():
        return None, None
    d = json.loads(path.read_text())
    s = d.get("summary", {})
    return d.get("n"), s.get("all", s)


def rows(results_dir=RESULTS_DIR):
    out = []
    for stem, label, split, calls, latency_ok in RUNS:
        n, a = _summary(Path(results_dir) / f"{stem}.json")
        _, j = _summary(Path(results_dir) / f"{stem}_judge-gpt-oss-20b.json")
        if not a:
            continue
        j = j or {}
        per_q = a["avg_prompt_tokens"] + a["avg_completion_tokens"]
        quality = j.get("answer_correctness")
        qname = "answer correctness"
        if quality is None:
            quality, qname = j.get("judged_citation_precision"), "citation support"
        out.append({
            "run": label, "split": split, "calls": calls, "n": n,
            "prompt": a["avg_prompt_tokens"], "completion": a["avg_completion_tokens"], "per_q": per_q,
            "judge_per_q": (j.get("avg_judge_prompt_tokens") or 0) + (j.get("avg_judge_completion_tokens") or 0),
            "run_total": per_q * n, "quality": quality, "quality_name": qname,
            "per_correct": round(per_q / quality) if qname == "answer correctness" and quality else None,
            "secs": a.get("avg_secs") if latency_ok else None,
        })
    return out


def markdown(rs):
    lines = ["| system | split | LLM calls / q | agent tokens / q (prompt + completion) | judge tokens / q | "
             "agent tokens, whole run | quality | agent tokens per correct answer | s / q |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rs:
        q = "–" if r["quality"] is None else f"{r['quality']:.3f} {r['quality_name']}"
        lines.append(f"| {r['run']} | {r['split']} | {r['calls']} | {r['per_q']:,} ({r['prompt']:,} + {r['completion']:,}) | "
                     f"{r['judge_per_q']:,} | {r['run_total']:,} | {q} | "
                     f"{'–' if r['per_correct'] is None else format(r['per_correct'], ',')} | "
                     f"{'–' if r['secs'] is None else r['secs']} |")
    return "\n".join(lines)


def main():
    rs = rows()
    text = ("# Token efficiency (from the saved summaries; `python -m eval.token_report`)\n\n" + markdown(rs) +
            "\n\nTokens include gpt-oss-120b's hidden reasoning (reasoning effort low). Seconds per question are mostly "
            "Groq free-tier pacing; the test run's latency is omitted because its first 19 questions ran on a CPU.\n")
    (RESULTS_DIR / "token_efficiency.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
