"""The final Day 4 retrieval pipeline behind one call.

    question
      -> BM25 top-50 + dense (bge-small-en-v1.5) top-50, within the paper
      -> RRF fusion (k=60)
      -> bge-reranker-v2-m3 re-scores the top-30 fused candidates
      -> abstract chunks removed from the ranking
      -> top-k evidence chunks

Usage:
    from retrieval.pipeline import retrieve
    hits = retrieve("What baselines do they compare against?", paper_id="1909.00694")
    for h in hits:
        print(h["rank"], h["chunk_id"], h["section"], h["score"])

The first call loads the chunks, the embedding cache and both models (~1 min on a Colab GPU);
later calls reuse them. Pass paper_id=None to search the whole corpus.
From the shell:  python -m retrieval "What baselines do they use?" --paper-id 1909.00694 -k 5
"""
from .bm25 import BM25Index
from .config import RetrievalConfig
from .data import load_chunks
from .dense import DenseIndex
from .hybrid import HybridRetriever
from .rerank import Reranker, apply_scores


class RetrievalPipeline:
    def __init__(self, config=None, chunks=None):
        self.config = cfg = config or RetrievalConfig()
        self.chunks = load_chunks() if chunks is None else chunks.reset_index(drop=True)
        self.texts = self.chunks["text"].tolist()
        self.chunk_ids = self.chunks["chunk_id"].astype(str).tolist()
        self.section_types = self.chunks["section_type"].astype(str).tolist()
        self.by_paper = {str(p): list(ix) for p, ix in self.chunks.groupby("paper_id").indices.items()}

        self.bm25 = BM25Index(self.texts)
        self.dense = DenseIndex(self.texts, cfg.dense_model)
        self.hybrid = HybridRetriever(self.bm25, self.dense, pool=cfg.pool, rrf_k=cfg.rrf_k)
        self._reranker = None  # loaded on first use, so cached-score evaluation never loads it
        self.excluded = set(cfg.exclude_sections or ())

    @property
    def reranker(self):
        if self._reranker is None:
            cfg = self.config
            self._reranker = Reranker(cfg.reranker_model, max_length=cfg.reranker_max_length, fp16=cfg.fp16)
        return self._reranker

    # ------------------------------------------------------------- stages
    def scope(self, paper_id):
        """Row indices of a paper's chunks, or None for whole-corpus search."""
        if paper_id is None:
            return None
        if str(paper_id) not in self.by_paper:
            raise KeyError(f"Unknown paper_id {paper_id!r}")
        return self.by_paper[str(paper_id)]

    def candidates(self, question, paper_id=None):
        """Stage 1: hybrid (BM25 + dense, RRF) ranking as row indices."""
        return self.hybrid.search(question, self.scope(paper_id))

    def rerank_pairs(self, question, candidates):
        """Stage 2 input: the (question, chunk text) pairs the cross-encoder scores."""
        return [(question, self.texts[i]) for i in candidates[:self.config.rerank_top]]

    def finalize(self, candidates, scores, exclude=True):
        """Stage 2 output + 3: order by reranker score, then drop excluded section types.
        Exclusion happens after reranking, exactly as in the Day 4 run that produced 0.795."""
        ranked = apply_scores(candidates, scores, self.config.rerank_top)
        if not exclude:
            return ranked
        return [(i, s) for i, s in ranked if self.section_types[i] not in self.excluded]

    # ------------------------------------------------------------- one call
    def retrieve(self, question, paper_id=None, k=10):
        cand = self.candidates(question, paper_id)
        scores = self.reranker.score(self.rerank_pairs(question, cand), batch_size=self.config.reranker_batch_size)
        out = []
        for rank, (i, s) in enumerate(self.finalize(cand, scores)[:k], start=1):
            row = self.chunks.iloc[i]
            out.append({
                "rank": rank, "chunk_id": self.chunk_ids[i], "paper_id": str(row["paper_id"]),
                "title": row.get("title", ""), "section": row["section"],
                "section_type": self.section_types[i], "score": s, "text": self.texts[i],
            })
        return out


_default = None


def get_pipeline():
    global _default
    if _default is None:
        _default = RetrievalPipeline()
    return _default


def retrieve(question, paper_id=None, k=10):
    """Top-k evidence chunks for `question`, searched within `paper_id` (or the whole corpus if None)."""
    return get_pipeline().retrieve(question, paper_id, k)

