"""Hybrid retrieval: BM25 and dense rankings fused with reciprocal rank fusion (RRF)."""


def rrf(rank_lists, k=60):
    """Fuse ranked lists of row indices. Each item scores sum(1 / (k + rank)), rank starting at 1."""
    score = {}
    for rl in rank_lists:
        for r, idx in enumerate(rl):
            score[idx] = score.get(idx, 0.0) + 1.0 / (k + r + 1)
    return [i for i, _ in sorted(score.items(), key=lambda x: -x[1])]


class HybridRetriever:
    def __init__(self, bm25, dense, pool=50, rrf_k=60):
        self.bm25, self.dense, self.pool, self.rrf_k = bm25, dense, pool, rrf_k

    def search(self, query, ids=None):
        """Fused ranking (row indices) of the top-`pool` BM25 and top-`pool` dense candidates."""
        return rrf([self.bm25.search(query, self.pool, ids),
                    self.dense.search(query, self.pool, ids)], k=self.rrf_k)
