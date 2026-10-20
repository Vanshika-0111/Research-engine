"""Build app/examples.json (instant, precomputed demo answers) from results that are ALREADY saved: no LLM calls.

Within-paper examples: dev answers of the frozen agent that the judge rated correct=yes and faithful=yes
(results/agent_eval_dev_n50_v2.jsonl + its gpt-oss-20b judge), plus multi-hop answers where a follow-up search was
made (results/agent_eval_dev_n50_v3hop.jsonl, if present). Cross-paper examples: v2 answers the judge rated
faithful=yes (results/cross_paper_agent_v2.jsonl + judge). Passage texts come from data/processed/chunks.parquet.

    python -m app.build_examples            # writes app/examples.json
    python -m app.build_examples --n-paper 5 --n-corpus 4
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.run_agent_eval import load_rows, qkey  # noqa: E402
from retrieval.config import RESULTS_DIR  # noqa: E402
from retrieval.data import load_chunks  # noqa: E402

OUT = Path(__file__).resolve().parent / "examples.json"
_SECTION_PREFIX = re.compile(r"^\[[^\]]*\]\s*")   # chunk text starts with "[Section] "


def _judged(path):
    return {qkey(j): j for j in load_rows(path)} if path.exists() else {}


def within_examples(chunks, n, multihop=False):
    tag = "v3hop" if multihop else "v2"
    src = RESULTS_DIR / f"agent_eval_dev_n50_{tag}.jsonl"
    if not src.exists():
        return []
    judged = _judged(src.with_name(f"{src.stem}_judge-gpt-oss-20b.jsonl"))
    text = dict(zip(chunks["chunk_id"].astype(str), chunks["text"]))
    section = dict(zip(chunks["chunk_id"].astype(str), chunks["section"]))
    title = dict(zip(chunks["paper_id"].astype(str), chunks["title"].astype(str)))
    picked, seen_papers = [], set()
    for r in load_rows(src):
        j = judged.get(qkey(r))
        if not j or j["judge"].get("correct") != "yes" or j["judge"].get("faithful") != "yes" or r["declined"]:
            continue
        if not r["cited_chunk_ids"] or r["paper_id"] in seen_papers or len(r["question"]) < 15:
            continue
        if multihop and not r.get("hops"):
            continue     # multi-hop examples should show a follow-up search
        hops = r.get("evidence_hops") or [0] * len(r["evidence_chunk_ids"])
        evidence = []
        for n_, (cid, hop) in enumerate(zip(r["evidence_chunk_ids"], hops), start=1):
            t = _SECTION_PREFIX.sub("", text.get(cid, ""), count=1)
            evidence.append({"label": f"E{n_}", "chunk_id": cid, "section": section.get(cid, ""), "text": t, "hop": hop})
        label_of = {e["chunk_id"]: e["label"] for e in evidence}
        out = {"answer": r["answer"], "evidence": evidence, "cited_chunk_ids": r["cited_chunk_ids"],
               "cited_labels": [label_of[c] for c in r["cited_chunk_ids"] if c in label_of],
               "invalid_labels": r.get("invalid_labels", []), "sub_questions": r["sub_questions"],
               "tokens": r["tokens"], "secs": r["secs"], "llm_calls": [{"model": "openai/gpt-oss-120b"}]}
        if multihop:
            out.update({k: r.get(k) for k in ("sufficient", "missing", "follow_up_query", "hop_log", "hops")})
        picked.append({"question": r["question"], "paper_id": r["paper_id"], "title": title.get(r["paper_id"], ""),
                       "multihop": multihop, "judge": {"correct": "yes", "faithful": "yes"}, "out": out})
        seen_papers.add(r["paper_id"])
        if len(picked) >= n:
            break
    return picked


def corpus_examples(n):
    src = RESULTS_DIR / "cross_paper_agent_v2.jsonl"
    if not src.exists():
        return []
    judged = {j["id"]: j for j in load_rows(src.with_name(f"{src.stem}_judge-gpt-oss-20b.jsonl"))} \
        if src.with_name(f"{src.stem}_judge-gpt-oss-20b.jsonl").exists() else {}
    picked = []
    for r in load_rows(src):
        j = judged.get(r["id"])
        if not j or j["judge"].get("faithful") != "yes" or r["declined"] or len(r["cited_papers"]) < 3:
            continue
        out = {k: r[k] for k in ("answer", "evidence", "cited_labels", "cited_chunk_ids", "cited_papers",
                                  "invalid_labels", "papers", "tokens", "secs")}
        out["llm_calls"] = [{"model": "openai/gpt-oss-120b"}]
        picked.append({"question": r["question"], "id": r["id"], "out": out,
                       "judge": {"faithful": "yes", "citation_support": j["scores"]["judged_citation_precision"]}})
        if len(picked) >= n:
            break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-paper", type=int, default=5)
    ap.add_argument("--n-multihop", type=int, default=2)
    ap.add_argument("--n-corpus", type=int, default=5)
    args = ap.parse_args()
    chunks = load_chunks()
    data = {"paper": within_examples(chunks, args.n_paper) + within_examples(chunks, args.n_multihop, multihop=True),
            "corpus": corpus_examples(args.n_corpus)}
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(data['paper'])} within-paper examples ({sum(e['multihop'] for e in data['paper'])} multi-hop), "
          f"{len(data['corpus'])} cross-paper examples -> {OUT} ({OUT.stat().st_size // 1024} KB)")
    for e in data["paper"]:
        print(f"  [paper{' +hop' if e['multihop'] else ''}] {e['question'][:70]}  ({e['title'][:40]})")
    for e in data["corpus"]:
        print(f"  [corpus] {e['question'][:90]}")


if __name__ == "__main__":
    main()
