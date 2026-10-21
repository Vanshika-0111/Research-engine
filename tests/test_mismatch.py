"""Premise-mismatch robustness eval (eval/run_mismatch_eval.py): term extraction, pairing, verdicts, summary."""
from eval.run_mismatch_eval import make_pairs, parse_verdict, premise_terms, summarize


def test_premise_terms():
    assert premise_terms("Why are current ELS's not sufficiently effective?") == ["ELS"]
    assert premise_terms("How does PARENT correlate with BLEU on WikiBio?") == ["PARENT", "BLEU", "WikiBio"]
    assert premise_terms("What datasets do they use?") == []                    # generic: fine for any paper
    assert premise_terms("Which baselines are used?") == []                     # sentence-initial word ignored


def test_make_pairs_wrong_paper_lacks_every_term():
    papers = {"p1": " we study els models and bleu", "p2": " a paper about bleu only", "p3": " nothing relevant here"}
    qs = [{"question": "Why are ELS models weak?", "paper_id": "p1"},
          {"question": "What datasets do they use?", "paper_id": "p2"},          # no premise term -> skipped
          {"question": "Is ROUGE used?", "paper_id": "p3"}]                      # term not in its own paper -> skipped
    pairs = make_pairs(qs, papers, n=5, seed=0)
    assert len(pairs) == 1 and pairs[0]["terms"] == ["ELS"]
    assert pairs[0]["wrong_paper_id"] in ("p2", "p3")                            # never p1, never a paper with "els"


def test_parse_verdict():
    assert parse_verdict('```json\n{"verdict": "Hallucinated", "note": "x"}\n```') == {"verdict": "hallucinated", "note": "x"}
    assert parse_verdict('{"verdict": "maybe"}') is None and parse_verdict("no json") is None


def test_summarize():
    t = {"prompt": 100, "completion": 10}
    rows = [{"declined": True, "judge": {"verdict": "declined"}, "tokens": t},
            {"declined": False, "judge": {"verdict": "flagged"}, "tokens": t},
            {"declined": False, "judge": {"verdict": "hallucinated"}, "tokens": t},
            {"declined": False, "judge": None, "tokens": t}]                    # verdict missing: not judged
    s = summarize(rows)
    assert s["n"] == 4 and s["n_judged"] == 3 and s["exact_decline_rate"] == 0.25
    assert s["safe_rate"] == round(2 / 3, 4) and s["hallucinated_rate"] == round(1 / 3, 4)
