"""FAISS indexes over the cached chunk embeddings, for corpus-wide dense search.

The chunk embeddings are L2-normalised (bge-small-en-v1.5, normalize_embeddings=True), so inner
product = cosine similarity, the same score the numpy search uses.

    flat  IndexFlatIP      exact search; same scores/rankings as numpy `emb @ q` (up to float
                           rounding on near-ties). Builds in well under a second; not cached.
    hnsw  IndexHNSWFlat    approximate graph search (inner-product metric). Faster per query,
                           recall < 1 depending on efSearch. Build takes seconds, so it is saved
                           to cache/ and reloaded.

Within-paper search does NOT use FAISS: a paper has ~18 chunks, and the frozen Day 4 pipeline
(R@5 = 0.795) keeps its exact numpy path.
"""
import time
from pathlib import Path

import numpy as np

KINDS = ("flat", "hnsw")


def _faiss():
    try:
        import faiss
    except ImportError as e:  # pragma: no cover
        raise ImportError("faiss is not installed: pip install faiss-cpu") from e
    return faiss


class FaissIndex:
    def __init__(self, emb, kind="flat", hnsw_m=32, ef_construction=200, ef_search=128,
                 cache_path=None, verbose=True):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        faiss = _faiss()
        self.kind, self.hnsw_m, self.ef_construction = kind, hnsw_m, ef_construction
        emb = np.ascontiguousarray(emb, dtype=np.float32)
        self.n, self.dim = emb.shape
        self.loaded_from_cache = False

        t0 = time.perf_counter()
        cache_path = Path(cache_path) if cache_path else None
        if cache_path is not None and cache_path.exists():
            self.index = faiss.read_index(str(cache_path))
            if self.index.ntotal != self.n or self.index.d != self.dim:
                raise ValueError(f"{cache_path} holds {self.index.ntotal}x{self.index.d} vectors, "
                                 f"expected {self.n}x{self.dim}; delete it to rebuild")
            self.loaded_from_cache = True
        else:
            if kind == "flat":
                self.index = faiss.IndexFlatIP(self.dim)
            else:
                self.index = faiss.IndexHNSWFlat(self.dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
                self.index.hnsw.efConstruction = ef_construction
            self.index.add(emb)
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                faiss.write_index(self.index, str(cache_path))
        self.build_secs = time.perf_counter() - t0
        self.set_ef_search(ef_search)
        if verbose:
            how = "loaded" if self.loaded_from_cache else "built"
            print(f"FAISS {self.describe()} {how} in {self.build_secs:.2f} s ({self.n} vectors, dim {self.dim})")

    def set_ef_search(self, ef_search):
        """HNSW only: candidate list size at query time (higher = better recall, slower)."""
        self.ef_search = ef_search
        if self.kind == "hnsw":
            self.index.hnsw.efSearch = int(ef_search)

    def describe(self):
        if self.kind == "flat":
            return "IndexFlatIP"
        return f"IndexHNSWFlat(M={self.hnsw_m}, efConstruction={self.ef_construction}, efSearch={self.ef_search})"

    def size_mb(self):
        return len(_faiss().serialize_index(self.index)) / 1e6

    def search(self, qv, k):
        """qv: one query vector (dim,) or a batch (n, dim). Returns (scores, row_indices) as 2-D arrays.
        Rows with fewer than k results are padded by FAISS with index -1."""
        q = np.ascontiguousarray(np.atleast_2d(qv), dtype=np.float32)
        return self.index.search(q, int(min(k, self.n)))

    def search_one(self, qv, k):
        """Top-k row indices for one query vector, as a Python list (same format as numpy search)."""
        _, idx = self.search(qv, k)
        return [int(i) for i in idx[0] if i >= 0]


def cache_name(emb_cache_path, kind, hnsw_m, ef_construction):
    """File name of a saved index, derived from the embedding cache it was built from."""
    stem = Path(emb_cache_path).stem  # e.g. emb_BAAI_bge-small-en-v1.5_28541
    return Path(emb_cache_path).parent / f"faiss_{kind}_M{hnsw_m}_efc{ef_construction}_{stem}.index"
