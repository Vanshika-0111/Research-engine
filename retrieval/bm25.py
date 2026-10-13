"""BM25 (Okapi) over chunk text. IDF is computed over the whole corpus, even for paper-scoped search."""
import re

import numpy as np

STOP = set("a an the of to in on for and or is are was were be by with as at from that this these those it its we our".split())


def tokenize(text: str):
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    return [w for w in text.split() if w not in STOP]


class BM25Index:
    def __init__(self, texts):
        from rank_bm25 import BM25Okapi
        self.bm = BM25Okapi([tokenize(t) for t in texts])

    def search(self, query, k, ids=None):
        """Top-k row indices. `ids` restricts the search to those rows (e.g. one paper's chunks)."""
        if ids is None:
            s = self.bm.get_scores(tokenize(query))
            return list(np.argsort(-s)[:k])
        s = np.asarray(self.bm.get_batch_scores(tokenize(query), ids))
        return [ids[i] for i in np.argsort(-s)[:k]]
