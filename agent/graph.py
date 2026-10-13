"""Day 6 research agent as a LangGraph graph.

    START -> decompose -> retrieve -> answer -> END

decompose : the LLM splits the question into up to 3 focused search queries (JSON)
retrieve  : retrieve() runs for the original question and each sub-question within the paper;
            the ranked lists are fused with RRF and the top 6 chunks become evidence E1..E6
answer    : the LLM answers from the evidence only, citing [E#] after each claim;
            labels are mapped back to chunk IDs so citations can be checked against gold evidence

Every LLM call's tokens and latency are appended to state["llm_calls"] (for token-efficiency metrics).
"""
import json
import operator
import re
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from retrieval.hybrid import rrf

from .config import AgentConfig
from .prompts import ANSWER_SYSTEM, ANSWER_USER, DECOMPOSE_SYSTEM, format_evidence

NO_ANSWER = "The provided evidence does not answer this question."
_LABEL = re.compile(r"\bE(\d+)\b")
_CITE_GROUP = re.compile(r"\[([^\[\]]*\bE\d+\b[^\[\]]*)\]")   # [E1], [E1, E3], [E1; E2]
_SECTION_PREFIX = re.compile(r"^\[[^\]]*\]\s*")              # chunk text starts with "[Section] "
_JSON_OBJ = re.compile(r"\{.*\}", re.S)
# citation brackets models actually produce: [E2], 【E2】 (gpt-oss), ［E2］, 〔E2〕, (E2), 【E2†L3-L5】
_ANY_CITE = re.compile(r"[\[【［〔(]([^\[\]【】［］〔〕()]*?\bE\d+\b[^\[\]【】［］〔〕()]*?)[\]】］〕)]")


def normalize_citations(answer):
    """Rewrite every citation group to plain [E#][E#] form, e.g. '【E2†L1】' -> '[E2]', '(E1, E3)' -> '[E1][E3]'."""
    def fix(m):
        inner = m.group(1)
        # keep ordinary parentheses that merely mention a label inside prose
        if m.group(0)[0] == "(" and re.sub(r"[\sE\d,;&and†L\-:]", "", inner):
            return m.group(0)
        return "".join(f"[E{n}]" for n in dict.fromkeys(_LABEL.findall(inner)))
    return _ANY_CITE.sub(fix, answer)


class AgentState(TypedDict, total=False):
    question: str
    paper_id: str
    sub_questions: list
    queries: list
    evidence: list
    answer: str
    cited_labels: list
    cited_chunk_ids: list
    invalid_labels: list
    llm_calls: Annotated[list, operator.add]


def parse_sub_questions(text, question, max_n):
    """Sub-questions from the LLM's JSON; falls back to the original question if parsing fails."""
    m = _JSON_OBJ.search(text or "")  # tolerate code fences or text around the JSON
    try:
        data = json.loads(m.group(0) if m else text)
        subs = data.get("sub_questions", []) if isinstance(data, dict) else data
        subs = [s.strip() for s in subs if isinstance(s, str) and s.strip()]
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
        subs = []
    return subs[:max_n] or [question]


def parse_citations(answer, valid_labels):
    """[E#] labels cited in the answer, in first-cited order, split into valid and invalid."""
    seen = []
    for group in _CITE_GROUP.findall(answer):
        for n in _LABEL.findall(group):
            lab = f"E{n}"
            if lab not in seen:
                seen.append(lab)
    valid = [l for l in seen if l in valid_labels]
    return valid, [l for l in seen if l not in valid_labels]


def build_agent(pipeline, llm, config=None):
    """Compile the graph. `pipeline` is a retrieval.RetrievalPipeline, `llm` anything with .chat()."""
    cfg = config or AgentConfig()

    def decompose(state):
        q = state["question"]
        text, usage = llm.chat(
            [{"role": "system", "content": DECOMPOSE_SYSTEM.format(max_n=cfg.max_sub_questions)},
             {"role": "user", "content": q}],
            max_tokens=cfg.decompose_max_tokens, json_mode=True)
        subs = parse_sub_questions(text, q, cfg.max_sub_questions)
        queries = list(dict.fromkeys([q] + subs))  # always search with the original question too
        return {"sub_questions": subs, "queries": queries, "llm_calls": [{"node": "decompose", **usage}]}

    def retrieve(state):
        t0 = time.time()
        runs, found_by = [], {}
        for q in state["queries"]:
            hits = pipeline.retrieve(q, state.get("paper_id"), k=cfg.k_per_query)
            runs.append([h["chunk_id"] for h in hits])
            for h in hits:
                found_by.setdefault(h["chunk_id"], {"hit": h, "queries": []})["queries"].append(q)
        evidence = []
        for n, cid in enumerate(rrf(runs)[:cfg.max_evidence], start=1):
            h = found_by[cid]["hit"]
            evidence.append({"label": f"E{n}", "chunk_id": cid, "section": h["section"],
                             "section_type": h["section_type"], "found_by": found_by[cid]["queries"],
                             "text": _SECTION_PREFIX.sub("", h["text"], count=1)})
        return {"evidence": evidence,
                "llm_calls": [{"node": "retrieve", "prompt_tokens": 0, "completion_tokens": 0,
                               "secs": round(time.time() - t0, 3), "model": "retrieval"}]}

    def answer(state):
        ev = state["evidence"]
        if not ev:
            return {"answer": NO_ANSWER, "cited_labels": [], "cited_chunk_ids": [], "invalid_labels": []}
        text, usage = llm.chat(
            [{"role": "system", "content": ANSWER_SYSTEM},
             {"role": "user", "content": ANSWER_USER.format(question=state["question"],
                                                            evidence=format_evidence(ev))}],
            max_tokens=cfg.max_answer_tokens)
        text = normalize_citations(text.strip())
        label_to_id = {e["label"]: e["chunk_id"] for e in ev}
        valid, invalid = parse_citations(text, set(label_to_id))
        return {"answer": text, "cited_labels": valid,
                "cited_chunk_ids": [label_to_id[l] for l in valid], "invalid_labels": invalid,
                "llm_calls": [{"node": "answer", **usage}]}

    g = StateGraph(AgentState)
    g.add_node("decompose", decompose)
    g.add_node("retrieve", retrieve)
    g.add_node("answer", answer)
    g.add_edge(START, "decompose")
    g.add_edge("decompose", "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", END)
    return g.compile()


class ResearchAgent:
    """agent = ResearchAgent(); out = agent.run("What datasets do they use?", paper_id="1909.00694")"""

    def __init__(self, config=None, pipeline=None, llm=None):
        self.config = config or AgentConfig()
        if pipeline is None:
            from retrieval.pipeline import get_pipeline
            pipeline = get_pipeline()
        if llm is None:
            from .llm import GroqLLM
            llm = GroqLLM(self.config.llm_model, self.config.temperature, self.config.reasoning_effort)
        self.pipeline, self.llm = pipeline, llm
        self.graph = build_agent(pipeline, llm, self.config)

    def run(self, question, paper_id=None):
        t0 = time.time()
        out = self.graph.invoke({"question": question, "paper_id": paper_id, "llm_calls": []})
        calls = out.get("llm_calls", [])
        out["tokens"] = {"prompt": sum(c["prompt_tokens"] for c in calls),
                         "completion": sum(c["completion_tokens"] for c in calls)}
        out["secs"] = round(time.time() - t0, 2)
        return out
