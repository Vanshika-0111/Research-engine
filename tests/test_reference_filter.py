"""The reference-quality filter rule (fixed before the test run, must not drift)."""
from eval.reference_filter import check_references, reference_problem, split_rows


def test_marker_only_references_are_unusable():
    assert reference_problem("BIBREF9 ; BIBREF8") == "only a citation/figure/table marker"
    assert reference_problem("the same datasets as BIBREF7") == "only a citation/figure/table marker"
    assert reference_problem("Table TABREF3") == "only a citation/figure/table marker"
    assert reference_problem("Answer with content missing: (Table 2) ...") == "content missing"


def test_real_answers_are_kept():
    assert reference_problem("Farasa BIBREF31; MADAMIRA BIBREF29") is None   # names + citations
    assert reference_problem("Yes") is None
    assert reference_problem("1913 signals") is None


def test_question_excluded_only_if_all_references_unusable():
    assert check_references(["BIBREF2", "BERT and ELMo"]) == (True, None)
    assert check_references(["BIBREF2", "TABREF1"])[0] is False
    assert check_references([]) == (True, None)


def test_split_rows():
    kept, excluded = split_rows([{"reference_answers": ["Yes"]}, {"reference_answers": ["BIBREF1"]}])
    assert len(kept) == 1 and excluded[0]["reference_filter_reason"]
