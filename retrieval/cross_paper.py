"""Cross-paper retrieval: search the whole corpus, then group the evidence by paper.

Within-paper questions (QASPER) already know their paper. Cross-paper questions ("Which papers evaluate on
HotpotQA, and what does each propose?") do not, so retrieval has to find the right PAPERS first:

    question
      -> BM25 top-`pool` + dense (FAISS) top-`pool` over all 28.5k chunks
      -> RRF fusion
      -> (optional) cross-encoder re-scores the top `rerank_top` fused chunks
      -> papers ranked by their best chunk (first appearance in the chunk ranking)
      -> top `max_papers` papers, each with its best `chunks_per_paper` chunks as evidence

This module only reuses the frozen pieces of retrieval/ (BM25Index, DenseIndex with corpus_index="flat",
rrf, Reranker); it changes nothing in the within-paper pipeline.
"""
from dataclasses import dataclass

from .hybrid import rrf


@dataclass
class CrossPaperConfig:
    pool: int = 200             # chunks taken from BM25 and from dense, each, before fusion
    rrf_k: int = 60
    rerank_top: int = 100       # fused chunks the cross-encoder re-scores (0 = no reranking)
    max_papers: int = 10        # papers returned
    chunks_per_paper: int = 2   # evidence chunks kept per paper
    exclude_sections: tuple = ()  # section types dropped before grouping (abstracts are kept: they say what a paper is about)


def papers_in_order(rows, paper_of):
    """Distinct paper IDs in the order their first chunk appears in a ranked list of rows."""
    return list(dict.fromkeys(paper_of[i] for i in rows))


class CrossPaperRetriever:
    def __init__(self, pipe, config=None):
        """`pipe` is a retrieval.pipeline.RetrievalPipeline (it provides chunks, BM25, dense and the reranker)."""
        self.pipe = pipe
        self.cfg = config or CrossPaperConfig()
        self.paper_of = [str(p) for p in pipe.chunks["paper_id"]]
        self.titles = dict(zip(self.paper_of, pipe.chunks["title"].astype(str)))

    # ---------------------------------------------------------------- stages (also used by the eval)
    def bm25_rows(self, question):
        return list(self.pipe.bm25.search(question, self.cfg.pool))

    def dense_rows(self, question):
        return list(self.pipe.dense.search(question, self.cfg.pool))

    def fused_rows(self, question, bm25=None, dense=None):
        bm25 = self.bm25_rows(question) if bm25 is None else bm25
        dense = self.dense_rows(question) if dense is None else dense
        return rrf([bm25, dense], k=self.cfg.rrf_k)

    def rerank_pairs(self, question, rows):
        return [(question, self.pipe.texts[i]) for i in rows[:self.cfg.rerank_top]]

    def rerank(self, rows, scores):
        """Re-order the first `rerank_top` rows by cross-encoder score; the rest keep their fused order."""
        top = self.cfg.rerank_top
        head = [i for i, _ in sorted(zip(rows[:top], (float(s) for s in scores)), key=lambda x: -x[1])]
        return head + list(rows[top:])

    def drop_sections(self, rows):
        ex = set(self.cfg.exclude_sections or ())
        return [i for i in rows if self.pipe.section_types[i] not in ex] if ex else list(rows)

    def group(self, rows):
        """[{paper_id, title, rank, chunk_rows}] for the top `max_papers` papers, best chunks first."""
        out, by_paper = [], {}
        for i in rows:
            pid = self.paper_of[i]
            if pid not in by_paper:
                if len(by_paper) == self.cfg.max_papers:
                    continue
                by_paper[pid] = {"paper_id": pid, "title": self.titles.get(pid, ""), "rank": len(by_paper) + 1,
                                 "chunk_rows": []}
                out.append(by_paper[pid])
            if len(by_paper[pid]["chunk_rows"]) < self.cfg.chunks_per_paper:
                by_paper[pid]["chunk_rows"].append(i)
        return out

    # ---------------------------------------------------------------- one call
    def retrieve(self, question, rerank=True):
        """Top papers for `question`, each with its best evidence chunks (dicts like retrieve() hits)."""
        rows = self.fused_rows(question)
        if rerank and self.cfg.rerank_top:
            pairs = self.rerank_pairs(question, rows)
            rows = self.rerank(rows, self.pipe.reranker.score(pairs, batch_size=self.pipe.config.reranker_batch_size))
        papers = self.group(self.drop_sections(rows))
        for p in papers:
            p["chunks"] = [{"chunk_id": self.pipe.chunk_ids[i], "section": self.pipe.chunks.iloc[i]["section"],
                            "section_type": self.pipe.section_types[i], "text": self.pipe.texts[i]}
                           for i in p.pop("chunk_rows")]
        return papers
