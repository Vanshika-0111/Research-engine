"""Dense retrieval with a sentence-transformers bi-encoder. Chunk embeddings are cached to disk."""
from pathlib import Path

import numpy as np

from .config import CACHE_DIR


class DenseIndex:
    BGE_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, texts, model_name, cache_dir=CACHE_DIR, batch_size=64):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        # same file name as Day 3, so the existing cache/emb_*.npy is reused
        cache = Path(cache_dir) / f"emb_{model_name.replace('/', '_')}_{len(texts)}.npy"
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            print(f"Loading cached embeddings: {cache}")
            self.emb = np.load(cache)
        else:
            self.emb = self.model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                                         show_progress_bar=True).astype(np.float32)
            np.save(cache, self.emb)

    def encode_query(self, query):
        if "bge" in self.model_name.lower():
            query = self.BGE_PREFIX + query  # BGE models expect this instruction on queries only
        return self.model.encode(query, normalize_embeddings=True).astype(np.float32)

    def search(self, query, k, ids=None):
        """Top-k row indices by cosine similarity. `ids` restricts the search to those rows."""
        qv = self.encode_query(query)
        if ids is None:
            return list(np.argsort(-(self.emb @ qv))[:k])
        s = self.emb[ids] @ qv
        return [ids[i] for i in np.argsort(-s)[:k]]
