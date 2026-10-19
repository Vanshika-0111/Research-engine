"""Both LangGraph graphs end to end with a fake retriever and a fake LLM (no network, no models)."""
import pandas as pd
import pytest

from agent.config import AgentConfig
from agent.cross_paper import CrossPaperAgentConfig, build_cross_paper_agent
from agent.graph import NO_ANSWER, build_agent
from retrieval.cross_paper import CrossPaperConfig


class FakePipeline:
    """retrieve(q, paper, k) -> hits; each query maps to a fixed ranked list of chunk IDs."""
    def __init__(self, runs):
        self.runs = runs

    def retrieve(self, q, paper_id, k=5):
        return [{"chunk_id": c, "section": "Method", "section_type": "body", "text": f"[Method] text of {c}"}
                for c in self.runs.get(q, [])[:k]]


def run_within(llm, runs, **cfg):
    graph = build_agent(FakePipeline(runs), llm, AgentConfig(**cfg))
    return graph.invoke({"question": "Q", "paper_id": "p", "llm_calls": []})


def test_anchored_merge_keeps_question_hits_first(fake_llm):
    llm = fake_llm('{"sub_questions": ["S1"]}', "Answer [E1] and [E6].")
    runs = {"Q": ["q1", "q2", "q3", "q4", "q5", "q6", "q7"], "S1": ["s1", "q2", "s2"]}
    out = run_within(llm, runs, anchor_k=5, max_evidence=7, k_per_query=4)
    ids = [e["chunk_id"] for e in out["evidence"]]
    assert ids[:5] == ["q1", "q2", "q3", "q4", "q5"]      # anchors always first
    assert ids[5:] == ["s1", "s2"]                          # sub-question finds fill the rest (q2 not repeated)
    assert out["cited_chunk_ids"] == ["q1", "s1"]           # [E1] -> q1, [E6] -> s1
    assert out["evidence"][0]["text"] == "text of q1"       # "[Section] " prefix stripped
    assert out["queries"] == ["Q", "S1"]


def test_no_evidence_gives_the_decline_sentence(fake_llm):
    llm = fake_llm('{"sub_questions": []}')
    out = run_within(llm, {})
    assert out["answer"] == NO_ANSWER and len(llm.calls) == 1   # answer node makes no LLM call


class FakeCrossRetriever:
    def __init__(self, papers):
        self.cfg, self.papers = CrossPaperConfig(), papers

    def retrieve(self, question, rerank=True):
        return [dict(p, chunks=list(p["chunks"])) for p in self.papers]


PAPERS = [{"paper_id": "p1", "title": "Paper One", "rank": 1,
           "chunks": [{"chunk_id": "c1", "section": "Exp", "section_type": "body", "text": "[Exp] We use HotpotQA."}]},
          {"paper_id": "p2", "title": "Paper Two", "rank": 2,
           "chunks": [{"chunk_id": "c2", "section": "Intro", "section_type": "body", "text": "Other work."},
                      {"chunk_id": "c3", "section": "Exp", "section_type": "body", "text": "x " * 800}]}]


def test_cross_paper_agent_maps_citations_to_papers(fake_llm):
    llm = fake_llm("- Paper One: uses HotpotQA 【E1】.\n- Paper Two: also [E3][E9].")
    graph = build_cross_paper_agent(FakeCrossRetriever(PAPERS), llm, CrossPaperAgentConfig(passage_chars=100))
    out = graph.invoke({"question": "Which papers use HotpotQA?", "llm_calls": []})
    assert [e["label"] for e in out["evidence"]] == ["E1", "E2", "E3"]
    assert out["evidence"][0]["text"] == "We use HotpotQA."           # section prefix stripped
    assert len(out["evidence"][2]["text"]) <= 104                      # cut to passage_chars + " ..."
    assert out["cited_papers"] == ["p1", "p2"] and out["invalid_labels"] == ["E9"]
    assert "### P1: Paper One (paper p1)" in llm.calls[0]["messages"][1]["content"]


@pytest.mark.parametrize("prompt,marker", [("v1", "First decide which papers"), ("v2", "Check every paper")])
def test_cross_paper_prompt_versions(fake_llm, prompt, marker):
    llm = fake_llm("- Paper One: x [E1].")
    graph = build_cross_paper_agent(FakeCrossRetriever(PAPERS), llm, CrossPaperAgentConfig(prompt=prompt))
    graph.invoke({"question": "q", "llm_calls": []})
    assert marker in llm.calls[0]["messages"][0]["content"]


def test_cross_paper_unknown_prompt_and_nothing_found(fake_llm):
    with pytest.raises(ValueError):
        build_cross_paper_agent(FakeCrossRetriever(PAPERS), fake_llm(), CrossPaperAgentConfig(prompt="v9"))
    llm = fake_llm()
    out = build_cross_paper_agent(FakeCrossRetriever([]), llm).invoke({"question": "q", "llm_calls": []})
    assert out["answer"] == NO_ANSWER and llm.calls == []
