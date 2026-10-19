"""Retrieval and citation metrics (eval/metrics.py) on hand-computed examples."""
import pytest

from eval.metrics import citation_metrics, mean, retrieval_metrics


def test_retrieval_metrics_hand_example():
    rankings = [["a", "x", "b", "y", "z"],      # gold a (rank 1), b (rank 3)
                ["x", "y", "z", "w", "c"]]      # gold c (rank 5)
    golds = [{"a", "b"}, {"c"}]
    m = retrieval_metrics(rankings, golds, ks=(1, 3, 5))
    assert m["hit@1"] == 0.5
    assert m["ev_recall@1"] == 0.25             # (1/2 + 0) / 2
    assert m["ev_recall@3"] == 0.5              # (2/2 + 0) / 2
    assert m["ev_recall@5"] == 1.0
    assert m["mrr@5"] == pytest.approx((1 / 1 + 1 / 5) / 2, abs=1e-4)


def test_retrieval_metrics_ignores_ranks_beyond_max_k():
    m = retrieval_metrics([["x", "y", "g"]], [{"g"}], ks=(1, 2))
    assert m["ev_recall@2"] == 0.0 and m["mrr@2"] == 0.0


def test_citation_metrics():
    m = citation_metrics(cited=["g1", "x", "g1"], gold=["g1", "g2"], evidence=["g1", "g2", "x"])
    assert m["citation_precision"] == 0.5       # duplicates count once: g1 good, x not
    assert m["citation_hit"] == 1.0
    assert m["citation_recall"] == 0.5
    assert m["evidence_recall"] == 1.0


def test_citation_metrics_nothing_cited():
    m = citation_metrics(cited=[], gold=["g"], evidence=[])
    assert m["citation_precision"] is None and m["citation_hit"] == 0.0


def test_mean_skips_none():
    assert mean([1.0, None, 0.0]) == 0.5
    assert mean([None, None]) is None
