"""Multi-hop agent (agent/multihop.py): sufficiency check, follow-up search, hop budget, citation mapping."""
import pytest

from agent.multihop import MultiHopConfig, build_multihop_agent, check_preview, parse_check
from test_agents import FakePipeline

RUNS = {"Q": ["q1", "q2", "q3", "q4", "q5", "q6", "q7"],
        "table scores": ["q2", "t1", "t2", "t3", "t4"],          # q2 is already evidence -> skipped
        "dataset size": ["t1", "d1"]}


def run(llm, **cfg):
    graph = build_multihop_agent(FakePipeline(RUNS), llm, MultiHopConfig(**cfg))
    return graph.invoke({"question": "Q", "paper_id": "p", "llm_calls": []})


def test_sufficient_evidence_goes_straight_to_answer(fake_llm):
    llm = fake_llm('{"sub_questions": []}', '{"sufficient": true}', "It is X [E1].")
    out = run(llm)
    assert out["hops"] == 0 and len(out["evidence"]) == 7
    assert [c["messages"][0]["content"][:30] for c in llm.calls][1].startswith("You check whether")
    assert out["cited_chunk_ids"] == ["q1"] and len(llm.calls) == 3


def test_missing_info_triggers_one_follow_up_hop(fake_llm):
    llm = fake_llm('{"sub_questions": []}',
                   '{"sufficient": false, "missing": "scores", "follow_up_query": "table scores"}',
                   "Scores are in [E8] and [E10].")
    out = run(llm)                                   # max_hops=1, max_extra_evidence=3
    assert out["hops"] == 1
    assert [e["chunk_id"] for e in out["evidence"][7:]] == ["t1", "t2", "t3"]
    assert [e["label"] for e in out["evidence"][7:]] == ["E8", "E9", "E10"]
    assert [e["hop"] for e in out["evidence"]] == [0] * 7 + [1] * 3
    assert out["cited_chunk_ids"] == ["t1", "t3"]
    assert out["hop_log"] == [{"hop": 1, "missing": "scores", "query": "table scores",
                               "added_chunk_ids": ["t1", "t2", "t3"]}]
    assert len(llm.calls) == 3                       # decompose, check, answer (no second check)
    assert "E10" in llm.calls[-1]["messages"][1]["content"]   # the answer sees the new evidence


def test_hop_budget_loops_back_to_check(fake_llm):
    llm = fake_llm('{"sub_questions": []}',
                   '{"sufficient": false, "follow_up_query": "table scores"}',
                   '{"sufficient": false, "follow_up_query": "dataset size"}',
                   "Done [E9].")
    out = run(llm, max_hops=2, max_extra_evidence=1)
    assert out["hops"] == 2 and [e["chunk_id"] for e in out["evidence"][7:]] == ["t1", "d1"]
    assert out["cited_chunk_ids"] == ["d1"]


def test_max_hops_zero_skips_the_check(fake_llm):
    llm = fake_llm('{"sub_questions": []}', "A [E1].")
    out = run(llm, max_hops=0)
    assert len(llm.calls) == 2 and out["hops"] == 0


@pytest.mark.parametrize("reply,expected", [
    ('{"sufficient": false, "missing": "m", "follow_up_query": "q"}', (False, "m", "q")),
    ('{"sufficient": "false", "follow_up_query": "q"}', (False, "", "q")),
    ('{"sufficient": false, "follow_up_query": ""}', (True, "", "")),      # no query -> nothing to search
    ('```json\n{"sufficient": true}\n```', (True, "", "")),
    ("garbage", (True, "", "")),
])
def test_parse_check(reply, expected):
    assert parse_check(reply) == expected


def test_check_preview_cuts_long_passages():
    ev = [{"label": "E1", "section": "S", "text": "word " * 300}]
    assert check_preview(ev, 50).endswith(" ...") and len(check_preview(ev, 50)) < 90
