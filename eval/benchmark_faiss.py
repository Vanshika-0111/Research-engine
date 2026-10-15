"""FAISS vs numpy for corpus-wide dense search (bge-small-en-v1.5, 28,541 chunks). No LLM calls.

What it measures, on the same query vectors for every backend:
  1. Exactness of FAISS IndexFlatIP against the current numpy search (np.argsort(-(emb @ q))):
     identical top-k order / set for k = 10, 50, 100, the largest score gap where they differ,
     and evidence metrics against QASPER gold (they must be equal).
  2. Build time, index size, and per-query search latency (median / p95, one query at a time,
     plus the amortised cost of one batched call).
  3. HNSW (IndexHNSWFlat, inner product) over a grid of M and efSearch: ANN recall@k against the
     exact top-k, gold evidence metrics, latency.
  4. The integrated code path: retrieval.dense.DenseIndex.search(..., ids=None) with corpus_index
     "numpy" vs "flat" on real question strings.

Scope is the whole corpus (the cross-paper setting). Within-paper search does not use FAISS;
check that with `python -m eval.run_retrieval_eval --split test` (must print REPRODUCED).

Run from the project root:
    python -m eval.benchmark_faiss                       # dev + test, default grid
    python -m eval.benchmark_faiss --splits dev --hnsw-m 32 --ef-search 64 128
Outputs: results/faiss_benchmark.json and results/faiss_benchmark.md
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import retrieval_metrics  # noqa: E402
from retrieval.config import CACHE_DIR, RESULTS_DIR, RetrievalConfig  # noqa: E402
from retrieval.data import load_chunks, load_questions  # noqa: E402
from retrieval.dense import DenseIndex  # noqa: E402
from retrieval.faiss_index import FaissIndex  # noqa: E402

# Day 3 corpus-scope dense numbers (results/day3_retrieval_dev_corpus.json), numpy search
DAY3_DEV_CORPUS_DENSE = {"ev_recall@5": 0.1098, "ev_recall@10": 0.155, "hit@1": 0.062}
GOLD_KS = (1, 5, 10, 50)
TIE_EPS = 1e-5  # score gaps below this are float-rounding ties, not real ranking changes


# ---------------------------------------------------------------- helpers
def encode_queries(dense, queries, split):
    """One vector per question, encoded one at a time with the pipeline's own encode_query
    (BGE prefix, normalised), cached so reruns skip the model."""
    texts = [q["question"] for q in queries]
    digest = hashlib.sha1("\x1f".join(texts).encode("utf-8")).hexdigest()[:12]
    path = CACHE_DIR / f"qemb_{dense.model_name.replace('/', '_')}_{split}_{len(texts)}_{digest}.npy"
    if path.exists():
        print(f"Loading cached query embeddings: {path}")
        return np.load(path), None
    t0 = time.perf_counter()
    qv = np.stack([dense.encode_query(t) for t in texts]).astype(np.float32)
    ms = 1000 * (time.perf_counter() - t0) / len(texts)
    np.save(path, qv)
    return qv, ms


def timed_loop(fn, Q, n_latency, warmup=5):
    """Call fn(q) for each query vector; return (results, median ms, p95 ms) over the first n_latency."""
    for q in Q[:warmup]:
        fn(q)
    out, times = [], []
    for i, q in enumerate(Q):
        t0 = time.perf_counter()
        r = fn(q)
        if i < n_latency:
            times.append(time.perf_counter() - t0)
        out.append(r)
    t = 1000 * np.asarray(times)
    return out, float(np.median(t)), float(np.percentile(t, 95))


def compare_rankings(ref_ids, ref_scores, got_ids, got_scores, ks):
    """Exact-order match, set match and the largest score gap at differing positions."""
    out = {}
    for k in ks:
        same_order = same_set = 0
        max_gap = 0.0
        for a, sa, b, sb in zip(ref_ids, ref_scores, got_ids, got_scores):
            a, b = list(a[:k]), list(b[:k])
            same_order += a == b
            same_set += set(a) == set(b)
            if a != b:
                # positions that differ: are they swaps between items with (near-)equal scores?
                diff = [i for i in range(min(len(a), len(b))) if a[i] != b[i]]
                max_gap = max(max_gap, max(abs(float(sa[i]) - float(sb[i])) for i in diff))
        n = len(ref_ids)
        out[k] = {"identical_order": round(same_order / n, 4), "identical_set": round(same_set / n, 4),
                  "max_score_gap_where_different": max_gap}
    return out


def ann_recall(exact_ids, approx_ids, k):
    """Share of the exact top-k that the approximate search also returns in its top-k."""
    return float(np.mean([len(set(e[:k]) & set(a[:k])) / k for e, a in zip(exact_ids, approx_ids)]))


def gold_metrics(runs, golds, chunk_ids):
    return retrieval_metrics([[chunk_ids[i] for i in r] for r in runs], golds, ks=GOLD_KS)


def fmt(x, nd=3):
    return "" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--depth", type=int, default=100, help="top-k depth compared (default 100)")
    ap.add_argument("--hnsw-m", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--ef-search", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    ap.add_argument("--n-latency", type=int, default=500, help="queries timed one at a time per system")
    ap.add_argument("--n-integration", type=int, default=200, help="questions for the DenseIndex code-path check")
    ap.add_argument("--threads", type=int, default=None, help="FAISS OpenMP threads (default: FAISS default)")
    ap.add_argument("--tag", default="", help="suffix for the output file names")
    args = ap.parse_args()

    import faiss
    if args.threads:
        faiss.omp_set_num_threads(args.threads)
    K = args.depth
    cfg = RetrievalConfig()

    chunks = load_chunks()
    texts = chunks["text"].tolist()
    chunk_ids = chunks["chunk_id"].astype(str).tolist()
    known = set(chunk_ids)
    dense = DenseIndex(texts, cfg.dense_model, corpus_index="numpy")
    emb = dense.emb
    env = {"faiss": faiss.__version__, "numpy": np.__version__, "python": platform.python_version(),
           "cpu_count": os.cpu_count(), "faiss_threads": faiss.omp_get_max_threads(),
           "n_chunks": int(emb.shape[0]), "dim": int(emb.shape[1]),
           # inner product = cosine only if the vectors are unit length
           "emb_norm_min_max": [round(float(x), 5) for x in (np.linalg.norm(emb, axis=1).min(),
                                                              np.linalg.norm(emb, axis=1).max())]}
    print(f"{emb.shape[0]} chunk vectors, dim {emb.shape[1]} | faiss {env['faiss']} | "
          f"{env['cpu_count']} CPUs, FAISS threads {env['faiss_threads']}")

    # ---- indexes (built once, shared by all splits)
    t0 = time.perf_counter()
    flat = FaissIndex(emb, kind="flat")
    flat_build = time.perf_counter() - t0
    hnsw = {}
    for m in args.hnsw_m:
        t0 = time.perf_counter()
        idx = FaissIndex(emb, kind="hnsw", hnsw_m=m, ef_construction=args.ef_construction)
        hnsw[m] = (idx, time.perf_counter() - t0, idx.size_mb())
    index_info = {"flat": {"build_s": round(flat_build, 3), "size_mb": round(flat.size_mb(), 1)},
                  "numpy": {"build_s": 0.0, "size_mb": round(emb.nbytes / 1e6, 1)}}
    for m, (_, b, sz) in hnsw.items():
        index_info[f"hnsw_M{m}"] = {"build_s": round(b, 2), "size_mb": round(sz, 1),
                                    "ef_construction": args.ef_construction}

    report = {"env": env, "args": vars(args), "indexes": index_info, "splits": {}}

    for split in args.splits:
        queries = load_questions(split, known)
        golds = [q["gold"] for q in queries]
        Q, enc_ms = encode_queries(dense, queries, split)
        print(f"\n=== {split}: {len(queries)} questions" + (f" | query encoding {enc_ms:.1f} ms/q" if enc_ms else ""))
        nl = min(args.n_latency, len(Q))
        rows = []

        # numpy, exactly as DenseIndex.search(ids=None) did before Day 10
        def np_search(q):
            s = emb @ q
            top = np.argsort(-s)[:K]
            return top, s[top]
        res, med, p95 = timed_loop(np_search, Q, nl)
        np_ids, np_sc = [r[0] for r in res], [r[1] for r in res]
        rows.append(dict(system="numpy argsort (current)", ann_recall_10=1.0, ann_recall_50=1.0,
                         median_ms=med, p95_ms=p95, **gold_metrics(np_ids, golds, chunk_ids)))

        # numpy with argpartition: a cheaper exact numpy baseline, for an honest speed comparison
        def np_part(q):
            s = emb @ q
            top = np.argpartition(-s, K)[:K]
            return top[np.argsort(-s[top])]
        _, med, p95 = timed_loop(np_part, Q, nl)
        rows.append(dict(system="numpy argpartition", ann_recall_10=1.0, ann_recall_50=1.0,
                         median_ms=med, p95_ms=p95))

        # FAISS flat: single queries and one batched call
        res, med, p95 = timed_loop(lambda q: flat.search(q, K), Q, nl)
        fl_ids, fl_sc = [r[1][0] for r in res], [r[0][0] for r in res]
        t0 = time.perf_counter()
        flat.search(Q, K)
        batch_ms = 1000 * (time.perf_counter() - t0) / len(Q)
        fl_gold = gold_metrics(fl_ids, golds, chunk_ids)
        np_gold = rows[0]
        rows.append(dict(system="FAISS IndexFlatIP", ann_recall_10=ann_recall(np_ids, fl_ids, 10),
                         ann_recall_50=ann_recall(np_ids, fl_ids, 50), median_ms=med, p95_ms=p95,
                         batch_ms_per_q=batch_ms, **fl_gold))
        exact = compare_rankings(np_ids, np_sc, fl_ids, fl_sc, ks=(10, 50, K))
        gold_equal = all(fl_gold[k] == np_gold[k] for k in fl_gold)
        flat_ok = gold_equal and all(v["max_score_gap_where_different"] < TIE_EPS for v in exact.values())

        # HNSW grid
        for m, (idx, _, _) in hnsw.items():
            for ef in args.ef_search:
                idx.set_ef_search(ef)
                res, med, p95 = timed_loop(lambda q: idx.search(q, K)[1][0], Q, nl)
                t0 = time.perf_counter()
                idx.search(Q, K)
                bms = 1000 * (time.perf_counter() - t0) / len(Q)
                rows.append(dict(system=f"FAISS HNSW M={m} efSearch={ef}",
                                 ann_recall_10=ann_recall(np_ids, res, 10), ann_recall_50=ann_recall(np_ids, res, 50),
                                 median_ms=med, p95_ms=p95, batch_ms_per_q=bms,
                                 **gold_metrics(res, golds, chunk_ids)))
                print(f"  HNSW M={m:<3} ef={ef:<4} ANN@10 {rows[-1]['ann_recall_10']:.4f}  "
                      f"R@5 {rows[-1]['ev_recall@5']:.4f}  {med:.3f} ms/q")

        day3 = None
        if split == "dev":
            day3 = {k: (v, np_gold[k], abs(np_gold[k] - v) <= 0.002) for k, v in DAY3_DEV_CORPUS_DENSE.items()}

        report["splits"][split] = {"n_queries": len(queries), "query_encode_ms": enc_ms,
                                   "flat_vs_numpy": {str(k): v for k, v in exact.items()},
                                   "flat_gold_metrics_equal": gold_equal, "flat_exact": flat_ok,
                                   "day3_dev_corpus_dense_check": day3, "systems": rows}
        for k, v in exact.items():
            print(f"  Flat vs numpy top-{k}: identical order {v['identical_order']:.4f}, "
                  f"identical set {v['identical_set']:.4f}, max score gap where different "
                  f"{v['max_score_gap_where_different']:.2e}")
        print(f"  Gold metrics numpy == flat: {gold_equal}  ->  " + ("EXACT" if flat_ok else "CHECK DIFFERENCES"))
        if day3:
            print("  Day 3 dev corpus dense (numpy): " + ", ".join(
                f"{k} {ref:.4f} vs {got:.4f}{'' if ok else ' (!)'}" for k, (ref, got, ok) in day3.items()))

    # ---- the integrated code path (real question strings, through DenseIndex.search)
    split = args.splits[0]
    qs = load_questions(split, known)[:args.n_integration]
    dense.corpus_index = "flat"
    flat_runs = [dense.search(q["question"], cfg.pool) for q in qs]
    dense.corpus_index = "numpy"
    np_runs = [[int(i) for i in dense.search(q["question"], cfg.pool)] for q in qs]
    same = float(np.mean([a == b for a, b in zip(np_runs, flat_runs)]))
    report["integration"] = {"split": split, "n": len(qs), "k": cfg.pool, "identical_top_k": same}
    print(f"\nDenseIndex.search(ids=None), numpy vs flat, {len(qs)} {split} questions, top-{cfg.pool}: "
          f"identical {same:.4f}")

    # ---- save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = RESULTS_DIR / f"faiss_benchmark{('_' + args.tag) if args.tag else ''}"
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2, default=float))
    stem.with_suffix(".md").write_text(to_markdown(report))
    print(f"\nSaved {stem.with_suffix('.json')} and {stem.with_suffix('.md')}")
    print(to_markdown(report))


def to_markdown(rep):
    e = rep["env"]
    lines = [f"# FAISS vs numpy, corpus-wide dense search ({e['n_chunks']} chunks, dim {e['dim']})", "",
             f"faiss {e['faiss']}, numpy {e['numpy']}, {e['cpu_count']} CPUs, FAISS threads {e['faiss_threads']}. "
             "Latency = search only (query vectors pre-encoded), one query at a time; "
             "batch = one call for all queries, per query.", "",
             "| index | build s | size MB |", "|---|---|---|"]
    for name, v in rep["indexes"].items():
        lines.append(f"| {name} | {v['build_s']} | {v['size_mb']} |")
    for split, s in rep["splits"].items():
        lines += ["", f"## {split} (n={s['n_queries']})", ""]
        for k, v in s["flat_vs_numpy"].items():
            lines.append(f"- Flat vs numpy top-{k}: identical order {v['identical_order']:.4f}, identical set "
                         f"{v['identical_set']:.4f}, max score gap where different {v['max_score_gap_where_different']:.1e}")
        lines.append(f"- Gold metrics equal: {s['flat_gold_metrics_equal']} -> "
                     + ("**exact**" if s["flat_exact"] else "**check differences**"))
        if s.get("day3_dev_corpus_dense_check"):
            lines.append("- Day 3 dev corpus dense check: " + ", ".join(
                f"{k} {r:.4f} vs {g:.4f}" for k, (r, g, _) in s["day3_dev_corpus_dense_check"].items()))
        lines += ["", "| system | ANN@10 | ANN@50 | R@5 | R@10 | R@50 | hit@1 | median ms | p95 ms | batch ms/q |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in s["systems"]:
            lines.append(f"| {r['system']} | {fmt(r.get('ann_recall_10'), 4)} | {fmt(r.get('ann_recall_50'), 4)} | "
                         f"{fmt(r.get('ev_recall@5'), 4)} | {fmt(r.get('ev_recall@10'), 4)} | "
                         f"{fmt(r.get('ev_recall@50'), 4)} | {fmt(r.get('hit@1'), 4)} | "
                         f"{fmt(r['median_ms'])} | {fmt(r['p95_ms'])} | {fmt(r.get('batch_ms_per_q'))} |")
    if "integration" in rep:
        i = rep["integration"]
        lines += ["", f"DenseIndex.search(ids=None) numpy vs flat on {i['n']} {i['split']} questions, "
                      f"top-{i['k']}: identical {i['identical_top_k']:.4f}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
