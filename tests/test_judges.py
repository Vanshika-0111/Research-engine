"""Judge reply parsing, including the 2026-10-19 missing-verdict fix (XP15: 1 of 8 labels judged)."""
import json

from eval import judge_agent_eval as within
from eval import judge_cross_paper as cross


def test_within_paper_judge_parses_fenced_json_with_think_tags():
    reply = '<think>hmm</think>```json\n{"citations": {"E1": "Yes", "E2": "partly"}, "faithful": "yes", "correct": "no"}\n```'
    v = within.parse_verdict(reply, ["E1", "E2"])
    assert v["citations"] == {"E1": "yes", "E2": "partly"} and v["correct"] == "no"


def test_within_paper_judge_list_form_and_bracket_keys():
    reply = json.dumps({"citations": [{"label": "[E1]", "supports": "yes"}], "faithful": "yes", "correct": "yes"})
    assert within.parse_verdict(reply, ["E1"])["citations"] == {"E1": "yes"}


def test_within_paper_judge_rejects_missing_label_verdict():
    reply = json.dumps({"citations": {"E1": "yes"}, "faithful": "yes", "correct": "yes"})
    assert within.parse_verdict(reply, ["E1", "E2"]) is None       # E2 has no verdict -> judge error


def test_within_paper_judge_rejects_unparseable_or_missing_correct():
    assert within.parse_verdict("no json here", ["E1"]) is None
    assert within.parse_verdict(json.dumps({"citations": {"E1": "yes"}, "faithful": "yes"}), ["E1"]) is None


def test_cross_paper_judge_bracket_keys_and_missing_verdict():
    ok = json.dumps({"citations": {"[E1]": "yes", " e2 ": "no"}, "faithful": "partly", "per_paper": {"p1": "yes"}})
    v = cross.parse(ok, ["E1", "E2"])
    assert v["citations"] == {"E1": "yes", "E2": "no"} and v["per_paper"] == {"p1": "yes"}
    missing = json.dumps({"citations": {"E2": "yes"}, "faithful": "no"})
    assert cross.parse(missing, ["E2", "E4", "E7"]) is None        # the XP15 case


def test_cross_paper_summary_ignores_declines():
    judged = [
        {"declined": False, "judge": {}, "judge_tokens": {"prompt": 10, "completion": 1},
         "scores": {"judged_citation_precision": 1.0, "faithfulness": 0.5, "per_paper_support": 1.0}},
        {"declined": True, "judge": {}, "judge_tokens": {"prompt": 0, "completion": 0},
         "scores": {"judged_citation_precision": None, "faithfulness": None, "per_paper_support": None}},
    ]
    s = cross.summarize(judged)
    assert s["faithfulness"] == 0.5 and s["declined_rate"] == 0.5 and s["avg_judge_prompt_tokens"] == 10
