"""
Day 4 — Cross-encoder reranking on top of the Day 3 hybrid retriever.

Pipeline:  question -> hybrid (BM25 + dense, RRF) top-N candidates -> cross-encoder re-scores
           each (question, chunk) pair -> re-ordered list -> scored against gold chunk IDs.

Needs day3_hybrid_retrieval.py in the same folder (it reuses its loaders, retrievers and metrics).

Install:  pip install rank_bm25 sentence-transformers pandas pyarrow
Run (from the project root):
  python day4_rerank.py --limit 200                                      # quick check
  python day4_rerank.py --scope paper                                    # main number
  python day4_rerank.py --scope paper --reranker cross-encoder/ms-marco-MiniLM-L-6-v2   # faster, smaller
  python day4_rerank.py --scope corpus
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from day3_hybrid_retrieval import BM25Index, DenseIndex, evaluate, load_data, rrf


class Reranker:
    def __init__(self, model_name, max_length=512, fp16=False):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, max_length=max_length)
        if fp16:
            try:  # half precision: ~2x faster on a T4, negligible quality change
                self.model.model.half()
            except Exception as e:
                print(f"fp16 not applied ({e}); running in fp32")

    def score(self, pairs, batch_size=64):
        return self.model.predict(pairs, batch_size=batch_size, show_progress_bar=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--scope", choices=["paper", "corpus"], default="paper")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5", help="dense embedding model (same as Day 3)")
    ap.add_argument("--reranker", default="BAAI/bge-reranker-base")
    ap.add_argument("--pool", type=int, default=50, help="candidates per retriever before fusion")
    ap.add_argument("--rerank-top", type=int, default=30, help="how many hybrid candidates the cross-encoder re-scores")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--title", action="store_true",
                    help="give the reranker the paper title with the question, so 'they'/'their' has a referent")
    ap.add_argument("--fp16", action="store_true", help="run the reranker in half precision (faster on GPU)")
    ap.add_argument("--exclude-sections", default="",
                    help="comma-separated section_type values to drop from ALL ranked lists, e.g. 'abstract'")
    args = ap.parse_args()

    chunks, queries = load_data(args.split, args.limit)
    texts = chunks["text"].tolist()
    chunk_ids = chunks["chunk_id"].astype(str).tolist()
    by_paper = {p: list(ix) for p, ix in chunks.groupby("paper_id").indices.items()}
    scope = (lambda q: by_paper.get(q["paper_id"], [])) if args.scope == "paper" else (lambda q: None)
    print(f"{len(chunks)} chunks | {len(queries)} questions (split={args.split}, scope={args.scope})")

    # ---- stage 1: hybrid candidates (same as Day 3)
    bm25 = BM25Index(texts)
    dense = DenseIndex(texts, args.model)
    P, R = args.pool, args.rerank_top
    hybrid = []
    for q in queries:
        fused = rrf([bm25.search(q["question"], P, scope(q)), dense.search(q["question"], P, scope(q))])
        hybrid.append(fused)

    # ---- stage 2: cross-encoder re-scores the top-R of each list (all pairs batched in one pass)
    titles = dict(zip(chunks["paper_id"], chunks["title"])) if "title" in chunks else {}
    pairs, owner = [], []
    for qi, (q, cand) in enumerate(zip(queries, hybrid)):
        rq = q["question"]
        if args.title and titles.get(q["paper_id"]):
            rq = f"Paper: {titles[q['paper_id']]}. Question: {rq}"
        for idx in cand[:R]:
            pairs.append((rq, texts[idx]))
            owner.append((qi, idx))
    # reranker scores are cached, so re-running with different post-processing flags costs nothing
    cache = Path("cache") / (f"rerank_{args.reranker.replace('/', '_')}_{args.split}_{args.scope}"
                             f"_top{R}_n{len(queries)}{'_title' if args.title else ''}.npy")
    cache.parent.mkdir(exist_ok=True)
    meta = cache.with_suffix(".json")
    if cache.exists() and len(np.load(cache)) == len(pairs):
        print(f"Loading cached reranker scores: {cache}")
        scores = np.load(cache)
        rerank_secs = json.loads(meta.read_text())["secs"] if meta.exists() else float("nan")
    else:
        reranker = Reranker(args.reranker, fp16=args.fp16)
        t0 = time.time()
        scores = np.asarray(reranker.score(pairs), dtype=np.float32)
        rerank_secs = time.time() - t0
        np.save(cache, scores)
        meta.write_text(json.dumps({"secs": rerank_secs, "fp16": args.fp16}))

    per_q = [[] for _ in queries]
    for (qi, idx), s in zip(owner, scores):
        per_q[qi].append((float(s), idx))
    reranked = []
    for qi, cand in enumerate(hybrid):
        top = [idx for _, idx in sorted(per_q[qi], key=lambda x: -x[0])]
        reranked.append(top + cand[R:])  # anything beyond rerank-top keeps its hybrid order

    # optional: drop section types that never hold gold evidence (applied to every system, so the comparison stays fair)
    excl = {s.strip() for s in args.exclude_sections.split(",") if s.strip()}
    if excl:
        stype_all = chunks["section_type"].astype(str).tolist()
        keep = lambda lst: [i for i in lst if stype_all[i] not in excl]
        hybrid = [keep(l) for l in hybrid]
        reranked = [keep(l) for l in reranked]
        print(f"Excluding section types from rankings: {sorted(excl)}")

    # ---- evaluate
    pos = {id(q): i for i, q in enumerate(queries)}
    systems = {
        "hybrid_rrf": lambda q: hybrid[pos[id(q)]],
        "hybrid+rerank": lambda q: reranked[pos[id(q)]],
        # fuse the reranker's order with the hybrid order instead of trusting the reranker alone
        "hybrid+rerank_rrf": lambda q: rrf([hybrid[pos[id(q)]], reranked[pos[id(q)]]]),
    }
    results = {name: evaluate(fn, queries, chunk_ids) for name, fn in systems.items()}
    for name, r in results.items():
        print(name, r)

    # ---- diagnostic: which kinds of sections does each system put at rank 1, vs where gold evidence lives?
    if "section_type" in chunks:
        from collections import Counter
        stype = chunks["section_type"].astype(str).tolist()
        id2ix = {c: i for i, c in enumerate(chunk_ids)}
        gold_c = Counter(stype[id2ix[g]] for q in queries for g in q["gold"] if g in id2ix)
        n_gold = sum(gold_c.values())
        print("\nSection type at rank 1 (share of questions)  vs  share of gold evidence:")
        rows = {name: Counter(stype[fn(q)[0]] for q in queries) for name, fn in systems.items()}
        for t in sorted(gold_c, key=lambda t: -gold_c[t]):
            line = "  ".join(f"{name}={rows[name][t] / len(queries):.2f}" for name in systems)
            print(f"  {t:<14} gold={gold_c[t] / n_gold:.2f}  {line}")

    Path("results").mkdir(exist_ok=True)
    tag = (args.reranker.split("/")[-1] + ("_title" if args.title else "")
           + ("_no-" + "-".join(sorted(excl)) if excl else ""))
    out_path = Path(f"results/day4_rerank_{args.split}_{args.scope}_{tag}.json")
    out_path.write_text(json.dumps(dict(
        split=args.split, scope=args.scope, dense_model=args.model, reranker=args.reranker, title=args.title,
        excluded_sections=sorted(excl),
        rerank_top=R, n_queries=len(queries), n_pairs=len(pairs),
        rerank_ms_per_query=round(1000 * rerank_secs / max(1, len(queries)), 1),
        results=results), indent=2))

    cols = ["hit@1", "hit@5", "ev_recall@5", "ev_recall@10", "mrr@10"]
    print(f"\nscope={args.scope}  split={args.split}  n={len(queries)}  reranker={args.reranker}")
    print("| system | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for name, r in results.items():
        print(f"| {name} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")
    print(f"\nRerank cost: {len(pairs)} pairs, {1000 * rerank_secs / max(1, len(queries)):.0f} ms/question")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
