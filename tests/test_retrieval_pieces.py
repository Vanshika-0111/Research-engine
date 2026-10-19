"""Pure retrieval pieces: RRF fusion, BM25 tokenisation/search, reranker ordering, cross-paper grouping."""
import pandas as pd

from retrieval.bm25 import BM25Index, tokenize
from retrieval.cross_paper import CrossPaperConfig, CrossPaperRetriever, papers_in_order
from retrieval.hybrid import rrf
from retrieval.rerank import apply_scores


def test_rrf_rewards_agreement():
    # "b" is 2nd in both lists, "a" 1st in one only: b wins
    assert rrf([["a", "b", "c"], ["d", "b", "e"]])[0] == "b"


def test_rrf_scores():
    fused = rrf([[1, 2], [2, 1]], k=60)
    assert set(fused) == {1, 2}                # tie broken by insertion order, both kept


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("The F1-score of BERT, on SQuAD!") == ["f1", "score", "bert", "squad"]


def test_bm25_search_and_paper_scope():
    idx = BM25Index(["neural machine translation", "speech recognition on librispeech", "machine translation bleu"])
    assert idx.search("machine translation", 1)[0] in (0, 2)
    assert idx.search("machine translation", 2, ids=[1, 2]) [0] == 2   # restricted to rows 1-2


def test_apply_scores_reorders_only_the_head():
    out = apply_scores([10, 11, 12, 13], [0.1, 0.9, 0.5], top=3)
    assert [i for i, _ in out] == [11, 12, 10, 13]
    assert out[-1] == (13, None)


class _Pipe:
    """Just enough of RetrievalPipeline for CrossPaperRetriever.group / rerank."""
    def __init__(self):
        self.chunks = pd.DataFrame({"paper_id": ["p1", "p1", "p2", "p3", "p1", "p2"],
                                    "title": ["T1", "T1", "T2", "T3", "T1", "T2"],
                                    "section": ["s"] * 6})
        self.section_types = ["abstract", "body", "body", "body", "body", "abstract"]


def test_cross_paper_grouping_by_best_chunk():
    r = CrossPaperRetriever(_Pipe(), CrossPaperConfig(max_papers=2, chunks_per_paper=2))
    papers = r.group([2, 0, 1, 3, 4, 5])          # p2 first (row 2), then p1; p3 is beyond max_papers
    assert [p["paper_id"] for p in papers] == ["p2", "p1"]
    assert papers[0]["chunk_rows"] == [2, 5] and papers[1]["chunk_rows"] == [0, 1]
    assert [p["rank"] for p in papers] == [1, 2]
    assert papers_in_order([3, 0, 4], r.paper_of) == ["p3", "p1"]


def test_cross_paper_rerank_and_section_filter():
    r = CrossPaperRetriever(_Pipe(), CrossPaperConfig(rerank_top=2, exclude_sections=("abstract",)))
    assert r.rerank([0, 1, 2], [0.2, 0.8]) == [1, 0, 2]
    assert r.drop_sections([0, 1, 5, 2]) == [1, 2]
