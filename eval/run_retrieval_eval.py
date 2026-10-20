"""Retrieval evaluation on QASPER gold evidence — reproduces the results log.

Systems scored (all within the question's paper unless --scope corpus):
    bm25            BM25 top-50
    dense           bge-small-en-v1.5 top-50
    hybrid_rrf      BM25 + dense fused with RRF
    hybrid+rerank   + bge-reranker-v2-m3 over the top-30
    final           + abstract chunks removed           <- the pipeline in retrieval/pipeline.py

Run from the project root:
    python -m eval.run_retrieval_eval --split test      # held-out number: final ev_recall@5 = 0.795
    python -m eval.run_retrieval_eval --split dev       # dev: 0.793
    python -m eval.run_retrieval_eval --limit 100       # quick smoke test
    python -m eval.run_retrieval_eval --split dev --reranker cross-encoder/ms-marco-MiniLM-L-6-v2   # a smaller
        # reranker (e.g. for the CPU demo app); saved as retrieval_eval_dev_paper_rr-<model>.json, frozen files untouched

Reranker scores are cached in cache/ under the same file names Day 4 used, so if the Day 4 cache is
there the test number reproduces in about a minute without running the cross-encoder.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python eval/run_retrieval_eval.py`

from eval.metrics import gold_section_share, markdown_table, rank1_section_share, retrieval_metrics  # noqa: E402
from retrieval.config import CACHE_DIR, RESULTS_DIR, RetrievalConfig  # noqa: E402
from retrieval.data import load_chunks, load_questions  # noqa: E402
from retrieval.hybrid import rrf  # noqa: E402
from retrieval.pipeline import RetrievalPipeline  # noqa: E402

# numbers in claude/results-log.md (full split, paper scope, final pipeline)
EXPECTED_R5 = {"test": 0.795, "dev": 0.793}


def cached_rerank_scores(pipe, pairs, split, scope, n_queries, refresh=False):
    cfg = pipe.config
    length = "" if cfg.reranker_max_length == 512 else f"_len{cfg.reranker_max_length}"
    cache = CACHE_DIR / (f"rerank_{cfg.reranker_model.replace('/', '_')}_{split}_{scope}"
                         f"_top{cfg.rerank_top}_n{n_queries}{length}.npy")
    meta_path = cache.with_suffix(".json")
    digest = hashlib.sha1("\x1e".join(q + "\x1f" + t for q, t in pairs).encode("utf-8")).hexdigest()
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if not refresh and cache.exists() and meta.get("pairs_sha1", digest) == digest:
        scores = np.load(cache)
        if len(scores) == len(pairs):
            print(f"Loading cached reranker scores: {cache}")
            return scores, meta.get("secs", float("nan")), True
    reranker = pipe.reranker
    t0 = time.time()
    scores = reranker.score(pairs, batch_size=cfg.reranker_batch_size, show_progress_bar=True)
    secs = time.time() - t0
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, scores)
    meta_path.write_text(json.dumps({"secs": secs, "fp16": reranker.fp16, "pairs_sha1": digest}))
    return scores, secs, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", help="dev | train | test | all")
    ap.add_argument("--scope", choices=["paper", "corpus"], default="paper")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions")
    ap.add_argument("--refresh-cache", action="store_true", help="re-run the reranker even if scores are cached")
    ap.add_argument("--diagnostics", action="store_true", help="print section type at rank 1 vs gold")
    ap.add_argument("--reranker", default=None, help="another cross-encoder (default: the frozen bge-reranker-v2-m3)")
    ap.add_argument("--max-length", type=int, default=None, help="reranker max tokens per pair (default 512)")
    args = ap.parse_args()

    cfg = RetrievalConfig()
    tag = ""
    if args.reranker:
        cfg.reranker_model = args.reranker
        tag += "_rr-" + args.reranker.split("/")[-1]
    if args.max_length:
        cfg.reranker_max_length = args.max_length
        tag += f"_len{args.max_length}"
    chunks = load_chunks()
    queries = load_questions(args.split, set(chunks["chunk_id"]), args.limit)
    print(f"{len(chunks)} chunks | {chunks.paper_id.nunique()} papers | "
          f"{len(queries)} questions with gold evidence (split={args.split}, scope={args.scope})")

    pipe = RetrievalPipeline(cfg, chunks)
    ids = pipe.chunk_ids

    # ---- stage 1: BM25 and dense separately (scored on their own), then fused
    def scope(q):
        return pipe.by_paper.get(str(q["paper_id"]), []) if args.scope == "paper" else None

    bm25_runs, dense_runs, hybrid_runs = [], [], []
    for q in queries:
        b = pipe.bm25.search(q["question"], cfg.pool, scope(q))
        d = pipe.dense.search(q["question"], cfg.pool, scope(q))
        bm25_runs.append(b)
        dense_runs.append(d)
        hybrid_runs.append(rrf([b, d], k=cfg.rrf_k))

    # ---- stage 2: every (question, candidate) pair scored in one batched pass
    pairs, spans = [], []
    for q, cand in zip(queries, hybrid_runs):
        p = pipe.rerank_pairs(q["question"], cand)
        spans.append((len(pairs), len(pairs) + len(p)))
        pairs += p
    scores, rerank_secs, from_cache = cached_rerank_scores(pipe, pairs, args.split, args.scope,
                                                           len(queries), args.refresh_cache)

    reranked, final = [], []
    for cand, (a, b) in zip(hybrid_runs, spans):
        reranked.append([i for i, _ in pipe.finalize(cand, scores[a:b], exclude=False)])
        final.append([i for i, _ in pipe.finalize(cand, scores[a:b])])

    # ---- metrics
    golds = [q["gold"] for q in queries]
    to_ids = lambda runs: [[ids[i] for i in r] for r in runs]
    systems = {"bm25": bm25_runs, "dense": dense_runs, "hybrid_rrf": hybrid_runs,
               "hybrid+rerank": reranked, "final": final}
    results = {name: retrieval_metrics(to_ids(runs), golds) for name, runs in systems.items()}

    print(f"\nscope={args.scope}  split={args.split}  n={len(queries)}  reranker={cfg.reranker_model}"
          f"  excluded={sorted(pipe.excluded)}")
    print(markdown_table(results))
    ms = 1000 * rerank_secs / max(1, len(queries))
    print(f"\nRerank cost: {len(pairs)} pairs, {ms:.0f} ms/question" + (" (from cache)" if from_cache else ""))

    expected = EXPECTED_R5.get(args.split) if (args.scope == "paper" and not args.limit and not tag) else None
    if expected is not None:
        got = results["final"]["ev_recall@5"]
        ok = abs(got - expected) <= 0.002
        print(f"Results log: final ev_recall@5 = {expected:.3f} | this run: {got:.3f} -> "
              + ("REPRODUCED" if ok else "MISMATCH, check data/processed and the model versions"))

    if args.diagnostics:
        section_of_row = pipe.section_types
        section_of_id = dict(zip(ids, section_of_row))
        gold_share = gold_section_share(golds, section_of_id)
        shares = {n: rank1_section_share(r, section_of_row, len(queries)) for n, r in systems.items()}
        print("\nSection type at rank 1 (share of questions) vs share of gold evidence:")
        for t in sorted(gold_share, key=lambda t: -gold_share[t]):
            print(f"  {t:<14} gold={gold_share[t]:.2f}  "
                  + "  ".join(f"{n}={shares[n].get(t, 0):.2f}" for n in systems))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"retrieval_eval_{args.split}_{args.scope}{f'_n{args.limit}' if args.limit else ''}{tag}.json"
    out.write_text(json.dumps(dict(
        split=args.split, scope=args.scope, n_queries=len(queries), n_chunks=len(chunks),
        config={k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(cfg).items()},
        n_pairs=len(pairs), rerank_ms_per_query=round(ms, 1), results=results), indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
