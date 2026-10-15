"""Dense retrieval with a sentence-transformers bi-encoder. Chunk embeddings are cached to disk.

Search scopes:
  * within a paper (`ids` given): exact numpy dot products over that paper's rows. This is the
    frozen Day 4 path behind R@5 = 0.795 and does not depend on `corpus_index`.
  * whole corpus (`ids=None`): `corpus_index` picks the backend:
        "numpy"  np.argsort(-(emb @ q))   (Day 3 behaviour)
        "flat"   FAISS IndexFlatIP        (exact; same rankings, see eval/benchmark_faiss.py)
        "hnsw"   FAISS IndexHNSWFlat      (approximate, faster)
    The FAISS index is built on the first corpus-wide search, so paper-scoped runs never load it.
"""
from pathlib import Path

import numpy as np

from .config import CACHE_DIR

CORPUS_BACKENDS = ("numpy", "flat", "hnsw")


class DenseIndex:
    BGE_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, texts, model_name, cache_dir=CACHE_DIR, batch_size=64,
                 corpus_index="numpy", hnsw_m=32, hnsw_ef_construction=200, hnsw_ef_search=128):
        from sentence_transformers import SentenceTransformer
        if corpus_index not in CORPUS_BACKENDS:
            raise ValueError(f"corpus_index must be one of {CORPUS_BACKENDS}, got {corpus_index!r}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        # same file name as Day 3, so the existing cache/emb_*.npy is reused
        cache = Path(cache_dir) / f"emb_{model_name.replace('/', '_')}_{len(texts)}.npy"
        cache.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache
        if cache.exists():
            print(f"Loading cached embeddings: {cache}")
            self.emb = np.load(cache)
        else:
            self.emb = self.model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                                         show_progress_bar=True).astype(np.float32)
            np.save(cache, self.emb)
        self.corpus_index = corpus_index
        self.hnsw_params = dict(hnsw_m=hnsw_m, ef_construction=hnsw_ef_construction, ef_search=hnsw_ef_search)
        self._faiss = None

    @property
    def faiss(self):
        """The FAISS index for corpus-wide search (built or loaded on first use), or None for numpy."""
        if self._faiss is None and self.corpus_index != "numpy":
            from .faiss_index import FaissIndex, cache_name
            p = self.hnsw_params
            path = (cache_name(self.cache_path, "hnsw", p["hnsw_m"], p["ef_construction"])
                    if self.corpus_index == "hnsw" else None)  # flat builds instantly; not cached
            self._faiss = FaissIndex(self.emb, kind=self.corpus_index, cache_path=path, **p)
        return self._faiss

    def encode_query(self, query):
        if "bge" in self.model_name.lower():
            query = self.BGE_PREFIX + query  # BGE models expect this instruction on queries only
        return self.model.encode(query, normalize_embeddings=True).astype(np.float32)

    def search_vector(self, qv, k, ids=None):
        """Top-k row indices for an already-encoded query vector. `ids` restricts the search to those rows."""
        if ids is None:
            if self.corpus_index != "numpy":
                return self.faiss.search_one(qv, k)
            return list(np.argsort(-(self.emb @ qv))[:k])
        s = self.emb[ids] @ qv
        return [ids[i] for i in np.argsort(-s)[:k]]

    def search(self, query, k, ids=None):
        """Top-k row indices by cosine similarity. `ids` restricts the search to those rows."""
        return self.search_vector(self.encode_query(query), k, ids)
