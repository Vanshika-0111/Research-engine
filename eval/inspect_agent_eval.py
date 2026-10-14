"""Side-by-side review of agent answers whose citations missed the gold evidence (or that declined).

For each such question it shows the question, QASPER's reference answers, the agent's answer, the passages it
cited and the gold passages, so you can judge by eye whether a "miss" is a real error or valid support that
QASPER simply didn't annotate.

Run from the project root after run_agent_eval:
    python -m eval.inspect_agent_eval results/agent_eval_dev_n20.jsonl
    python -m eval.inspect_agent_eval results/agent_eval_dev_n20.jsonl --all     # every question
Also writes the same review as a Markdown file next to the .jsonl (open it in Drive to read comfortably).
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval.data import load_chunks  # noqa: E402

_PREFIX = re.compile(r"^\[[^\]]*\]\s*")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--all", action="store_true", help="show every question, not just misses")
    ap.add_argument("--chars", type=int, default=500, help="characters of each passage to show")
    args = ap.parse_args()

    chunks = load_chunks()
    text = dict(zip(chunks["chunk_id"].astype(str), chunks["text"]))
    section = dict(zip(chunks["chunk_id"].astype(str), chunks["section"]))

    def passage(cid):
        body = _PREFIX.sub("", text.get(cid, "(missing)"), count=1)
        return f"`{cid}` ({section.get(cid, '?')})\n> {body[:args.chars]}{'…' if len(body) > args.chars else ''}"

    rows = [json.loads(l) for l in open(args.jsonl, encoding="utf-8")]
    out = [f"# Agent review: {Path(args.jsonl).name}\n"]
    shown = 0
    for i, r in enumerate(rows, 1):
        p = r["metrics"]["citation_precision"]
        if not args.all and p == 1.0 and not r.get("declined"):
            continue
        shown += 1
        gold, cited, ev = r["gold_chunk_ids"], r["cited_chunk_ids"], set(r["evidence_chunk_ids"])
        out.append(f"## Q{i}. {r['question']}")
        out.append(f"citation precision: {'-' if p is None else round(p, 2)} | "
                   f"gold reached the model: {sum(g in ev for g in gold)}/{len(gold)} | "
                   f"declined: {r.get('declined')}")
        out.append("**Reference answers:** " + " || ".join(r.get("reference_answers") or ["(none)"]))
        out.append(f"**Agent answer:** {r['answer']}")
        out.append("**Cited passages:**")
        out += [("✅ " if c in gold else "❌ ") + passage(c) for c in cited] or ["(none)"]
        out.append("**Gold passages:**")
        out += [("(in evidence) " if g in ev else "(NOT retrieved) ") + passage(g) for g in gold]
        out.append("**Your verdict** (edit): cited passages support the answer? yes / partly / no\n")
        out.append("---")
    out.insert(1, f"{shown} of {len(rows)} questions shown"
                  + ("" if args.all else " (citation precision < 1 or declined)") + "\n")

    md = "\n\n".join(out)
    md_path = Path(args.jsonl).with_name(Path(args.jsonl).stem + "_review.md")
    md_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nSaved {md_path}")


if __name__ == "__main__":
    main()
