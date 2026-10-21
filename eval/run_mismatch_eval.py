"""Robustness to a false premise: ask a question about the WRONG paper and see whether the agent declines.

Found while testing the app: "Why are current ELS's not sufficiently effective?" asked about a paper that never
mentions ELS made the agent invent a meaning ("early language systems") instead of declining. QASPER never asks a
question about the wrong paper, so the earlier evaluations could not see this.

Design (fixed before running):
  * Many QASPER questions are generic ("What datasets do they use?") and are fine to answer for ANY paper, so only
    questions with a SPECIFIC PREMISE TERM are used: a capitalised term / acronym from the question (e.g. "ELS",
    "SQuAD", "PARENT") that occurs in the question's own paper but nowhere in the wrong paper.
  * Each such dev question is paired with a random other paper (seed 0) that does not contain any of its premise
    terms. The FROZEN agent (Day 7 v2) is run on (question, wrong paper).
  * The correct behaviour is to decline, or to say the term is not in this paper. A judge (gpt-oss-20b) labels each
    answer: declined / flagged (answers but says the premise is missing) / hallucinated (answers as if the premise
    held). Exact declines (the agent's fixed decline sentence) are also counted without any LLM.

    python -m eval.run_mismatch_eval --n 30            # ~3k gpt-oss-120b + ~1k gpt-oss-20b tokens per question
    python -m eval.run_mismatch_eval --n 30 --resume   # after a daily-limit stop
Outputs results/mismatch_eval_dev_n<N>.jsonl (one row per question) and .json (summary).
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]*[A-Za-z0-9]")
_JSON_OBJ = re.compile(r"\{.*\}", re.S)
_THINK = re.compile(r"<think>.*?</think>", re.S)
COMMON = frozenset("""I A An The What Which How Why Do Does Did Is Are Was Were Who Whom Whose Where When In On For Of
To By With And Or From At As Be Can Could Should Would Will Their They Them This That These Those It Its""".split())

JUDGE_SYSTEM = """You check answers from a question-answering system about scientific papers.

The question was deliberately asked about the WRONG paper: the term(s) listed below come from a different paper and
do NOT occur anywhere in the paper the system searched. A correct system should notice that the question's premise
is not in this paper.

Classify the system's answer:
- "declined": it says the evidence does not answer the question / does not mention the term
- "flagged": it gives some information but clearly says the term or premise is not in this paper
- "hallucinated": it answers as if the premise held in this paper (e.g. invents what the term means, or attributes
  the term's properties to this paper)

Respond with JSON only: {"verdict": "declined" | "flagged" | "hallucinated", "note": "one short sentence"}"""

JUDGE_USER = """Question: {question}
Term(s) not in the searched paper: {terms}

System answer:
{answer}

Passages the system cited:
{passages}"""


def premise_terms(question):
    """Capitalised terms / acronyms of a question: 'ELS', 'SQuAD', 'PARENT'; ignores sentence-initial words."""
    out = []
    for i, m in enumerate(_WORD.finditer(question)):
        t = re.sub(r"'s$", "", m.group(0))
        if t in COMMON or len(t) < 2:
            continue
        upper = sum(c.isupper() for c in t)
        if upper >= 2 or (i > 0 and t[0].isupper()):
            out.append(t)
    return list(dict.fromkeys(out))


def make_pairs(questions, paper_text, n, seed=0, tries=30):
    """Questions with premise terms found in their own paper, each paired with a random wrong paper lacking them."""
    rng = random.Random(seed)
    papers = sorted(paper_text)
    pool = list(questions)
    rng.shuffle(pool)
    pairs = []
    for q in pool:
        own = paper_text.get(str(q["paper_id"]), "")
        terms = [t for t in premise_terms(q["question"]) if t.lower() in own]
        if not terms:
            continue
        for _ in range(tries):
            wrong = rng.choice(papers)
            if wrong == str(q["paper_id"]):
                continue
            text = paper_text[wrong]
            if all(not re.search(r"\b" + re.escape(t.lower()) + r"\b", text) for t in terms):
                pairs.append({**q, "own_paper_id": str(q["paper_id"]), "wrong_paper_id": wrong, "terms": terms})
                break
        if len(pairs) >= n:
            break
    return pairs


def parse_verdict(text):
    text = _THINK.sub("", text or "")
    m = _JSON_OBJ.search(text)
    try:
        d = json.loads(m.group(0) if m else text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    v = str(d.get("verdict", "")).strip().lower() if isinstance(d, dict) else ""
    return {"verdict": v, "note": str(d.get("note", "")).strip()} if v in ("declined", "flagged", "hallucinated") else None


def summarize(rows):
    n = len(rows)
    judged = [r for r in rows if r.get("judge")]
    count = lambda v: sum(r["judge"]["verdict"] == v for r in judged)  # noqa: E731
    return {
        "n": n, "n_judged": len(judged),
        "exact_decline_rate": round(sum(r["declined"] for r in rows) / n, 4) if n else None,
        "declined_rate": round(count("declined") / len(judged), 4) if judged else None,
        "flagged_rate": round(count("flagged") / len(judged), 4) if judged else None,
        "hallucinated_rate": round(count("hallucinated") / len(judged), 4) if judged else None,
        "safe_rate": round((count("declined") + count("flagged")) / len(judged), 4) if judged else None,
        "avg_agent_tokens": round(sum(r["tokens"]["prompt"] + r["tokens"]["completion"] for r in rows) / n) if n else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--judge-model", default="openai/gpt-oss-20b")
    ap.add_argument("--pause", type=float, default=2.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from agent.graph import NO_ANSWER, ResearchAgent
    from agent.llm import is_daily_limit, is_overloaded, make_llm
    from eval.run_agent_eval import load_rows
    from retrieval.config import RESULTS_DIR
    from retrieval.data import load_questions
    from retrieval.pipeline import get_pipeline

    pipe = get_pipeline()
    paper_text = {}
    for pid, text in zip(pipe.chunks["paper_id"].astype(str), pipe.chunks["text"]):
        paper_text[pid] = paper_text.get(pid, "") + " " + text.lower()
    questions = load_questions("dev", set(pipe.chunk_ids))
    pairs = make_pairs(questions, paper_text, args.n, args.seed)
    print(f"{len(pairs)} premise-mismatch pairs from {len(questions)} dev questions (seed {args.seed})")

    stem = RESULTS_DIR / f"mismatch_eval_dev_n{args.n}"
    jsonl = stem.with_suffix(".jsonl")
    rows = load_rows(jsonl) if jsonl.exists() else []
    if rows and not (args.resume or args.overwrite):
        raise SystemExit(f"{jsonl} already has {len(rows)} rows. Use --resume or --overwrite.")
    if args.overwrite:
        rows = []
    key = lambda r: (r["question"], r["wrong_paper_id"])  # noqa: E731
    done = {key(r): r for r in rows if r.get("judge")}       # rows without a verdict are redone (judge only)
    agent, judge = ResearchAgent(pipeline=pipe), make_llm(args.judge_model, temperature=0.0, reasoning_effort="low")
    stopped = None
    out_rows = []
    with open(jsonl, "w", encoding="utf-8") as fh:
        for i, p in enumerate(pairs, 1):
            k = (p["question"], p["wrong_paper_id"])
            if k in done:
                out_rows.append(done[k])
                fh.write(json.dumps(done[k], ensure_ascii=False) + "\n")
                continue
            row = next((dict(r) for r in rows if key(r) == k), None)   # agent answer saved, verdict missing
            try:
                if row is None:
                    out = agent.run(p["question"], p["wrong_paper_id"])
                    ev = {e["label"]: e for e in out["evidence"]}
                    row = {"question": p["question"], "own_paper_id": p["own_paper_id"],
                           "wrong_paper_id": p["wrong_paper_id"], "terms": p["terms"], "answer": out["answer"],
                           "declined": out["answer"].strip() == NO_ANSWER, "cited_labels": out["cited_labels"],
                           "cited_passages": [{"label": l, "section": ev[l]["section"], "text": ev[l]["text"][:1500]}
                                              for l in out["cited_labels"] if l in ev],
                           "sub_questions": out["sub_questions"], "tokens": out["tokens"], "judge": None}
                if row["declined"]:
                    row["judge"] = {"verdict": "declined", "note": "exact decline sentence"}
                else:
                    passages = "\n\n".join(f"[{c['label']}] ({c['section']})\n{c['text']}" for c in row["cited_passages"])
                    reply, _ = judge.chat([{"role": "system", "content": JUDGE_SYSTEM},
                                           {"role": "user", "content": JUDGE_USER.format(
                                               question=row["question"], terms=", ".join(row["terms"]),
                                               answer=row["answer"], passages=passages or "(none)")}],
                                          max_tokens=1024, json_mode=True)
                    row["judge"] = parse_verdict(reply)
            except Exception as e:
                if not (is_daily_limit(e) or is_overloaded(e)):
                    raise
                stopped = str(e)
                if row is not None:                       # keep the agent's answer; --resume only re-judges it
                    out_rows.append(row)
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"\nSTOPPED at {i}/{len(pairs)}: quota or overload. Run again later with --resume.")
                break
            out_rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            v = (row.get("judge") or {}).get("verdict", "?")
            print(f"[{i:>2}/{len(pairs)}] {v:<12} terms={','.join(row['terms'])[:30]:<30} {row['question'][:60]}")
            time.sleep(args.pause)
    if not out_rows:
        raise SystemExit("No results.")
    s = summarize(out_rows)
    stem.with_suffix(".json").write_text(json.dumps({"n_planned": len(pairs), "complete": len(out_rows) == len(pairs),
                                                     "stopped_reason": stopped, "seed": args.seed, "summary": s}, indent=2))
    print("\n" + "\n".join(f"{k:<22} {v}" for k, v in s.items()))
    for r in out_rows:
        if (r.get("judge") or {}).get("verdict") == "hallucinated":
            print(f"- hallucinated: {r['question'][:70]} | {r['answer'][:120]}")
    print(f"\nSaved {stem.with_suffix('.json')} and {jsonl}")


if __name__ == "__main__":
    main()
