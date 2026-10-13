"""
Day 3 — Dense retrieval + Hybrid (BM25 + Dense fused with RRF), scored against Day 2's gold chunk IDs.

Inputs (from Day 2, ingest/build_chunks.py):
  data/processed/chunks.parquet         chunk_id, paper_id, section, text, ...
  data/processed/qasper_questions.jsonl question, paper_id, split, gold_chunk_ids, ...

Outputs:
  cache/emb_<model>_<n>.npy                   chunk embeddings (computed once, reused)
  results/day3_retrieval_<split>_<scope>.json metrics

Install:   pip install rank_bm25 sentence-transformers pandas pyarrow
Run (from the project root):
  python day3_hybrid_retrieval.py --split dev --scope paper     # retrieve within the question's paper
  python day3_hybrid_retrieval.py --split dev --scope corpus    # retrieve across ALL papers (harder)
  python day3_hybrid_retrieval.py --limit 200                   # quick sanity run
"""
import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data/processed")
STOP = set("a an the of to in on for and or is are was were be by with as at from that this these those it its we our".split())


def tok(t: str):
    t = re.sub(r"[^a-z0-9 ]", " ", t.lower())
    return [w for w in t.split() if w not in STOP]


def as_list(v):
    """gold_chunk_ids may come back as a list, a numpy array, or a stringified list."""
    if isinstance(v, str):
        v = ast.literal_eval(v) if v.strip().startswith("[") else [v]
    return [str(x) for x in (v if v is not None else [])]


# ----------------------------------------------------------------- data
def load_data(split, limit=None):
    chunks = pd.read_parquet(DATA / "chunks.parquet").reset_index(drop=True)
    qs = [json.loads(l) for l in open(DATA / "qasper_questions.jsonl", encoding="utf-8")]
    if split != "all":
        qs = [q for q in qs if q.get("split") == split]
    for q in qs:
        q["gold"] = as_list(q.get("gold_chunk_ids"))
    known = set(chunks["chunk_id"])
    qs = [q for q in qs if q["gold"] and any(g in known for g in q["gold"])]  # drop unanswerable / no-text-evidence
    if limit:
        qs = qs[:limit]
    return chunks, qs


# ----------------------------------------------------------------- retrievers
class BM25Index:
    def __init__(self, texts):
        from rank_bm25 import BM25Okapi
        self.bm = BM25Okapi([tok(t) for t in texts])

    def search(self, q, k, ids=None):
        if ids is None:
            s = self.bm.get_scores(tok(q))
            return list(np.argsort(-s)[:k])
        s = np.asarray(self.bm.get_batch_scores(tok(q), ids))
        return [ids[i] for i in np.argsort(-s)[:k]]


class DenseIndex:
    BGE_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, texts, model_name, cache_dir="cache"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        cache = Path(cache_dir) / f"emb_{model_name.replace('/', '_')}_{len(texts)}.npy"
        cache.parent.mkdir(exist_ok=True)
        if cache.exists():
            print(f"Loading cached embeddings: {cache}")
            self.emb = np.load(cache)
        else:
            self.emb = self.model.encode(list(texts), batch_size=64, normalize_embeddings=True,
                                         show_progress_bar=True).astype(np.float32)
            np.save(cache, self.emb)

    def search(self, q, k, ids=None):
        if "bge" in self.model_name.lower():
            q = self.BGE_PREFIX + q
        qv = self.model.encode(q, normalize_embeddings=True).astype(np.float32)
        if ids is None:
            return list(np.argsort(-(self.emb @ qv))[:k])
        s = self.emb[ids] @ qv
        return [ids[i] for i in np.argsort(-s)[:k]]


def rrf(rank_lists, k=60):
    score = {}
    for rl in rank_lists:
        for r, idx in enumerate(rl):
            score[idx] = score.get(idx, 0.0) + 1.0 / (k + r + 1)
    return [i for i, _ in sorted(score.items(), key=lambda x: -x[1])]


# ----------------------------------------------------------------- eval
def evaluate(retrieve, queries, chunk_ids, ks=(1, 3, 5, 10)):
    """hit@k       : at least one gold chunk in top-k
       ev_recall@k : fraction of a question's gold chunks found in top-k, averaged (the CV metric)
       mrr@10      : reciprocal rank of the first gold chunk"""
    hit = {k: 0.0 for k in ks}
    rec = {k: 0.0 for k in ks}
    mrr = 0.0
    K = max(ks)
    for q in queries:
        top = [chunk_ids[i] for i in retrieve(q)[:K]]
        gold = set(q["gold"])
        ranks = [r for r, c in enumerate(top) if c in gold]
        if ranks:
            mrr += 1.0 / (ranks[0] + 1)
        for k in ks:
            n = sum(1 for r in ranks if r < k)
            hit[k] += n > 0
            rec[k] += n / len(gold)
    n = len(queries)
    out = {f"hit@{k}": round(hit[k] / n, 4) for k in ks}
    out.update({f"ev_recall@{k}": round(rec[k] / n, 4) for k in ks})
    out["mrr@10"] = round(mrr / n, 4)
    return out


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", help="dev | train | test | all (the 'split' field in qasper_questions.jsonl)")
    ap.add_argument("--scope", choices=["paper", "corpus"], default="paper",
                    help="paper = search only the question's paper; corpus = search every chunk")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--pool", type=int, default=50, help="candidates per retriever before fusion")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions")
    ap.add_argument("--systems", default="bm25,dense,hybrid_rrf")
    args = ap.parse_args()

    chunks, queries = load_data(args.split, args.limit)
    print(f"{len(chunks)} chunks | {chunks.paper_id.nunique()} papers | "
          f"{len(queries)} questions with gold evidence (split={args.split})")

    # chunk text already starts with "[Section]" from Day 2, so it is indexed as-is
    texts = chunks["text"].tolist()
    chunk_ids = chunks["chunk_id"].astype(str).tolist()
    by_paper = chunks.groupby("paper_id").indices
    by_paper = {p: list(ix) for p, ix in by_paper.items()}
    scope = (lambda q: by_paper.get(q["paper_id"], [])) if args.scope == "paper" else (lambda q: None)

    wanted = args.systems.split(",")
    P = args.pool
    bm25 = BM25Index(texts) if {"bm25", "hybrid_rrf"} & set(wanted) else None
    dense = DenseIndex(texts, args.model) if {"dense", "hybrid_rrf"} & set(wanted) else None

    # memoize per-question rankings so hybrid reuses the BM25/dense runs instead of recomputing them
    memo_b, memo_d = {}, {}

    def run_bm25(q):
        if id(q) not in memo_b:
            memo_b[id(q)] = bm25.search(q["question"], P, scope(q))
        return memo_b[id(q)]

    def run_dense(q):
        if id(q) not in memo_d:
            memo_d[id(q)] = dense.search(q["question"], P, scope(q))
        return memo_d[id(q)]

    systems = {
        "bm25": run_bm25,
        "dense": run_dense,
        "hybrid_rrf": lambda q: rrf([run_bm25(q), run_dense(q)]),
    }

    results = {}
    for name in wanted:
        results[name] = evaluate(systems[name], queries, chunk_ids)
        print(name, results[name])

    Path("results").mkdir(exist_ok=True)
    out_path = Path(f"results/day3_retrieval_{args.split}_{args.scope}.json")
    out_path.write_text(json.dumps(dict(split=args.split, scope=args.scope, model=args.model,
                                        n_chunks=len(chunks), n_queries=len(queries),
                                        results=results), indent=2))

    cols = ["hit@1", "hit@5", "ev_recall@5", "ev_recall@10", "mrr@10"]
    print(f"\nscope={args.scope}  split={args.split}  n={len(queries)}")
    print("| system | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for name, r in results.items():
        print(f"| {name} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
