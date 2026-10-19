"""Citation handling in agent/graph.py: what the LLM writes -> which evidence chunks were cited."""
from agent.graph import normalize_citations, parse_citations, parse_sub_questions


def test_gpt_oss_lenticular_brackets_become_ascii():
    # Day 6 bug: gpt-oss writes 【E2】, which parsed as "no citation" for 3/5 answers
    assert normalize_citations("It uses BERT 【E2】.") == "It uses BERT [E2]."


def test_line_range_suffix_is_dropped():
    assert normalize_citations("Accuracy is 91% 【E3†L4-L9】.") == "Accuracy is 91% [E3]."


def test_grouped_and_fullwidth_brackets():
    assert normalize_citations("Two datasets [E1, E3] and ［E4］.") == "Two datasets [E1][E3] and [E4]."
    assert normalize_citations("Shown (E2; E5).") == "Shown [E2][E5]."


def test_ordinary_parentheses_are_kept():
    text = "They use a CNN (see Table 2 for E1 results) on SQuAD."
    assert normalize_citations(text) == text


def test_parse_citations_order_dedupe_and_invalid():
    valid, invalid = parse_citations("A [E2]. B [E1][E2]. C [E9].", {"E1", "E2", "E3"})
    assert valid == ["E2", "E1"]          # first-cited order, no duplicates
    assert invalid == ["E9"]              # a label that was never shown to the LLM


def test_parse_citations_none():
    assert parse_citations("No citations here.", {"E1"}) == ([], [])


def test_sub_questions_from_json_in_code_fence():
    text = '```json\n{"sub_questions": ["What data?", " Which metric? ", ""]}\n```'
    assert parse_sub_questions(text, "orig", 3) == ["What data?", "Which metric?"]


def test_sub_questions_capped_and_fallback():
    assert parse_sub_questions('{"sub_questions": ["a", "b", "c", "d"]}', "orig", 3) == ["a", "b", "c"]
    assert parse_sub_questions("not json at all", "orig", 3) == ["orig"]
    assert parse_sub_questions('["x", "y"]', "orig", 3) == ["x", "y"]   # a bare JSON list is accepted
