"""Paper-level retrieval evaluation on the hand-written cross-paper questions (no LLM calls).

eval/cross_paper_questions.jsonl holds 19 questions (10 about a dataset, 9 about a topic). Each lists
`required_papers` (must be found) and `acceptable_papers` (fine to retrieve, not required). The set was
fixed on 2026-10-15 before any cross-paper system existed; do not edit it after seeing results.

Every system ranks chunks over the whole corpus; papers are then ranked by their first (best) chunk.
Metrics, averaged over questions:
    paper_recall@k   share of required papers among the first k distinct papers (k = 5, 10, 20)
    paper_prec@10    share of the first 10 papers that are required or acceptable
    mrr              1 / rank of the first required paper

Systems: bm25, dense (bge-small via FAISS flat), hybrid (RRF), hybrid+rerank (bge-reranker-v2-m3 over the
top --rerank-top fused chunks; needs a GPU to be quick, skipped with --no-rerank).

Run from the project root:
    python -m eval.run_cross_paper_eval               # all systems
    python -m eval.run_cross_paper_eval --no-rerank   # BM25 / dense / hybrid only (fine on CPU)
Outputs results/cross_paper_retrieval.json and .md. Reranker scores are cached in cache/.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval.config import CACHE_DIR, RESULTS_DIR, RetrievalConfig  # noqa: E402
from retrieval.cross_paper import CrossPaperConfig, CrossPaperRetriever, papers_in_order  # noqa: E402
from retrieval.pipeline import RetrievalPipeline  # noqa: E402

QUESTIONS = Path(__file__).resolve().parent / "cross_paper_questions.jsonl"
KS = (5, 10, 20)


def load_questions(path=QUESTIONS):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def paper_metrics(papers, required, acceptable):
    req, ok = set(required), set(required) | set(acceptable)
    m = {f"paper_recall@{k}": len(req & set(papers[:k])) / len(req) for k in KS}
    m["paper_prec@10"] = len(ok & set(papers[:10])) / 10
    first = next((r for r, p in enumerate(papers, 1) if p in req), None)
    m["mrr"] = 1.0 / first if first else 0.0
    return m


def average(rows):
    return {k: round(float(np.mean([r[k] for r in rows])), 4) for k in rows[0]}


def cached_scores(retr, pairs, refresh=False):
    """Cross-encoder scores for all (question, chunk) pairs, cached by a hash of the pairs."""
    model = retr.pipe.config.reranker_model.replace("/", "_")
    digest = hashlib.sha1("\x1e".join(q + "\x1f" + t for q, t in pairs).encode("utf-8")).hexdigest()[:12]
    path = CACHE_DIR / f"rerank_{model}_crosspaper_{len(pairs)}_{digest}.npy"
    if path.exists() and not refresh:
        print(f"Loading cached reranker scores: {path}")
        return np.load(path), None
    t0 = time.time()
    scores = retr.pipe.reranker.score(pairs, batch_size=retr.pipe.config.reranker_batch_size, show_progress_bar=True)
    secs = time.time() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, scores)
    return scores, secs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--questions", default=str(QUESTIONS))
    ap.add_argument("--pool", type=int, default=200, help="chunks from BM25 and from dense, each")
    ap.add_argument("--rerank-top", type=int, default=100, help="fused chunks the cross-encoder re-scores")
    ap.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder (CPU-friendly)")
    ap.add_argument("--refresh-cache", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    qs = load_questions(args.questions)
    rcfg = RetrievalConfig()   # corpus_index="flat" (FAISS) by default
    pipe = RetrievalPipeline(rcfg)
    retr = CrossPaperRetriever(pipe, CrossPaperConfig(pool=args.pool, rerank_top=args.rerank_top))
    print(f"{len(qs)} cross-paper questions | {len(pipe.chunks)} chunks | dense backend={rcfg.corpus_index}")

    runs = {"bm25": [], "dense": [], "hybrid": []}
    for q in qs:
        b, d = retr.bm25_rows(q["question"]), retr.dense_rows(q["question"])
        runs["bm25"].append(b)
        runs["dense"].append(d)
        runs["hybrid"].append(retr.fused_rows(q["question"], b, d))

    rerank_secs = None
    if not args.no_rerank:
        pairs, spans = [], []
        for q, rows in zip(qs, runs["hybrid"]):
            p = retr.rerank_pairs(q["question"], rows)
            spans.append((len(pairs), len(pairs) + len(p)))
            pairs += p
        scores, rerank_secs = cached_scores(retr, pairs, args.refresh_cache)
        runs["hybrid+rerank"] = [retr.rerank(rows, scores[a:b]) for rows, (a, b) in zip(runs["hybrid"], spans)]

    per_q, results = {}, {}
    for name, rows_list in runs.items():
        ms = []
        for q, rows in zip(qs, rows_list):
            papers = papers_in_order(rows, retr.paper_of)
            m = paper_metrics(papers, q["required_papers"], q["acceptable_papers"])
            ms.append(m)
            per_q.setdefault(q["id"], {})[name] = {**{k: round(v, 3) for k, v in m.items()}, "top10": papers[:10]}
        results[name] = average(ms)
        for typ in ("dataset", "topic"):
            sub = [m for m, q in zip(ms, qs) if q["type"] == typ]
            results[f"{name} [{typ}]"] = average(sub)

    cols = ["paper_recall@5", "paper_recall@10", "paper_recall@20", "paper_prec@10", "mrr"]
    lines = ["| system | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, r in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")
    table = "\n".join(lines)
    print("\n" + table)
    if rerank_secs:
        print(f"\nReranking: {sum(b - a for a, b in spans)} pairs in {rerank_secs:.0f} s")

    names = list(runs)
    qlines = ["| id | type | required | " + " | ".join(f"{n} R@10" for n in names) + " |",
              "|---" * (3 + len(names)) + "|"]
    for q in qs:
        qlines.append(f"| {q['id']} | {q['type']} | {len(q['required_papers'])} | "
                      + " | ".join(f"{per_q[q['id']][n]['paper_recall@10']:.2f}" for n in names) + " |")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = RESULTS_DIR / f"cross_paper_retrieval{('_' + args.tag) if args.tag else ''}"
    stem.with_suffix(".json").write_text(json.dumps(dict(
        n_questions=len(qs), pool=args.pool, rerank_top=None if args.no_rerank else args.rerank_top,
        dense_backend=rcfg.corpus_index, reranker=None if args.no_rerank else rcfg.reranker_model,
        rerank_secs=rerank_secs, results=results, per_question=per_q), indent=2))
    stem.with_suffix(".md").write_text(
        f"# Cross-paper retrieval ({len(qs)} questions, paper-level)\n\n{table}\n\n## Per question (paper recall@10)\n\n"
        + "\n".join(qlines) + "\n")
    print(f"\nSaved {stem.with_suffix('.json')} and {stem.with_suffix('.md')}")


if __name__ == "__main__":
    main()
