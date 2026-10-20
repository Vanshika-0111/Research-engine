"""A tiny fake RetrievalPipeline with the attributes the app, the agents and CrossPaperRetriever use."""
from types import SimpleNamespace

import numpy as np
import pandas as pd


class FakeAppPipeline:
    def __init__(self):
        self.chunks = pd.DataFrame({
            "chunk_id": ["a1", "a2", "a3", "b1", "b2"], "paper_id": ["1909.00001"] * 3 + ["1909.00002"] * 2,
            "title": ["Paper A"] * 3 + ["Paper B"] * 2, "section": ["Data", "Results", "Abstract", "Exp", "Intro"],
            "section_type": ["body", "body", "abstract", "body", "body"],
            "text": ["[Data] We use SQuAD.", "[Results] F1 is 90.", "[Abstract] A paper.", "[Exp] HotpotQA runs.",
                     "[Intro] Multi-hop QA."]})
        self.texts = self.chunks["text"].tolist()
        self.chunk_ids = self.chunks["chunk_id"].tolist()
        self.section_types = self.chunks["section_type"].tolist()
        self.config = SimpleNamespace(reranker_batch_size=8)
        self.reranker = SimpleNamespace(score=lambda pairs, batch_size=8: np.linspace(1, 0, len(pairs)))
        self.bm25 = SimpleNamespace(search=lambda q, k, ids=None: [0, 1, 3, 4][:k])
        self.dense = SimpleNamespace(search=lambda q, k, ids=None: [3, 0, 4][:k], faiss=None)

    def retrieve(self, q, paper_id=None, k=10):
        rows = [i for i, p in enumerate(self.chunks["paper_id"]) if paper_id is None or p == paper_id][:k]
        return [{"rank": n + 1, "chunk_id": self.chunk_ids[i], "paper_id": self.chunks["paper_id"][i],
                 "title": self.chunks["title"][i], "section": self.chunks["section"][i],
                 "section_type": self.section_types[i], "score": 1.0, "text": self.texts[i]} for n, i in enumerate(rows)]
