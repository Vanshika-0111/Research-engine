"""Multi-hop research agent (Day 14): the within-paper agent plus a sufficiency check that can search again.

    START -> decompose -> retrieve -> check --(sufficient)----------------------> answer -> END
                                        \\--(missing info)--> follow_up --(hops < max_hops)--> check
                                                                       \\--(hop budget used)--> answer

decompose, retrieve, answer : the same as the frozen Day 7 v2 agent (agent/graph.py, same prompts and settings),
                              so any difference in the results comes from the extra hop.
check     : the LLM reads the question and short previews of the evidence and says whether it is enough to
            answer; if not, it names what is missing and writes ONE new search query for it (JSON).
follow_up : searches the paper with that query and appends up to `max_extra_evidence` NEW chunks as E8, E9, ...
            (chunks already in the evidence are skipped), then loops back to `check` or goes on to `answer`.

Why: decomposition plans all searches up front from the question alone. When the first evidence reveals that a
specific piece is missing (e.g. the passage names a dataset but the numbers are in another section, or the
question's wording doesn't match the paper's), a second, evidence-informed search can find it. Day 8's dev misses
of this kind: Q30 "what datasets?" (paper says "collected ... corpus"), Q33, Q45, Q47, Q49.

The frozen agent (agent/graph.py) is not changed; this is a separate graph, evaluated on dev with
    python -m eval.run_agent_eval --n 50 --tag v3hop --multihop --resume
"""
import json
import operator
import re
import time
from dataclasses import dataclass
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from retrieval.hybrid import rrf

from .config import AgentConfig
from .graph import NO_ANSWER, ResearchAgent, normalize_citations, parse_citations, parse_sub_questions
from .prompts import ANSWER_SYSTEM, ANSWER_USER, DECOMPOSE_SYSTEM, format_evidence

_SECTION_PREFIX = re.compile(r"^\[[^\]]*\]\s*")
_JSON_OBJ = re.compile(r"\{.*\}", re.S)

CHECK_SYSTEM = """You check whether passages retrieved from ONE scientific paper contain enough information to
answer a question. You do not answer the question.

- "sufficient": true if the passages state the answer (or everything needed to work it out), even if the answer
  would be short. Also true if the question cannot be answered from this paper at all and searching again would
  not help.
- "sufficient": false only if a SPECIFIC piece of information needed for the answer is missing from the passages
  (for example the passages name the baselines but not their scores, or mention a dataset without its size).
- If false, write ONE short search query for the missing piece, using words the paper itself would likely use
  (method names, dataset names, metric names from the passages), not a copy of the question.

Respond with JSON only: {"sufficient": true, "missing": "", "follow_up_query": ""}"""

CHECK_USER = """Question: {question}

Passages (previews):
{evidence}"""


@dataclass
class MultiHopConfig(AgentConfig):
    max_hops: int = 1              # extra retrieval rounds allowed after the first one
    hop_k: int = 4                 # chunks searched per follow-up query
    max_extra_evidence: int = 3    # new chunks a hop may add (on top of max_evidence)
    check_chars: int = 500         # each passage is shown to the check node cut to this length (keeps it cheap)
    check_max_tokens: int = 768


class MultiHopState(TypedDict, total=False):
    question: str
    paper_id: str
    sub_questions: list
    queries: list
    evidence: list
    sufficient: bool
    missing: str
    follow_up_query: str
    hops: int
    hop_log: list
    answer: str
    cited_labels: list
    cited_chunk_ids: list
    invalid_labels: list
    llm_calls: Annotated[list, operator.add]


def parse_check(text):
    """(sufficient, missing, follow_up_query) from the check node's JSON. Unparseable -> sufficient (no hop)."""
    m = _JSON_OBJ.search(text or "")
    try:
        data = json.loads(m.group(0) if m else text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return True, "", ""
    if not isinstance(data, dict):
        return True, "", ""
    suff = data.get("sufficient", True)
    if isinstance(suff, str):
        suff = suff.strip().lower() not in ("false", "no", "0")
    query = str(data.get("follow_up_query") or "").strip()
    if not query:                      # "insufficient" without a query can't be acted on
        suff = True
    return bool(suff), str(data.get("missing") or "").strip(), query


def check_preview(evidence, chars):
    out = []
    for e in evidence:
        t = e["text"] if len(e["text"]) <= chars else e["text"][:chars].rsplit(" ", 1)[0] + " ..."
        out.append(f"[{e['label']}] (Section: {e['section']})\n{t}")
    return "\n\n".join(out)


def build_multihop_agent(pipeline, llm, config=None):
    cfg = config or MultiHopConfig()

    # ---- decompose / retrieve / answer: identical to agent/graph.py (Day 7 v2, frozen) -------------------
    def decompose(state):
        q = state["question"]
        text, usage = llm.chat(
            [{"role": "system", "content": DECOMPOSE_SYSTEM.format(max_n=cfg.max_sub_questions)},
             {"role": "user", "content": q}],
            max_tokens=cfg.decompose_max_tokens, json_mode=True)
        subs = parse_sub_questions(text, q, cfg.max_sub_questions)
        queries = list(dict.fromkeys([q] + subs))
        return {"sub_questions": subs, "queries": queries, "hops": 0, "hop_log": [],
                "llm_calls": [{"node": "decompose", **usage}]}

    def _ev(n, h, hop, found_by):
        return {"label": f"E{n}", "chunk_id": h["chunk_id"], "section": h["section"],
                "section_type": h["section_type"], "found_by": found_by, "hop": hop,
                "text": _SECTION_PREFIX.sub("", h["text"], count=1)}

    def retrieve(state):
        t0 = time.time()
        question, paper = state["question"], state.get("paper_id")
        found_by = {}

        def search(q, k):
            hits = pipeline.retrieve(q, paper, k=k)
            for h in hits:
                found_by.setdefault(h["chunk_id"], {"hit": h, "queries": []})["queries"].append(q)
            return [h["chunk_id"] for h in hits]

        main = search(question, cfg.max_evidence)
        chosen = main[:cfg.anchor_k]
        sub_runs = [search(q, cfg.k_per_query) for q in state["queries"] if q != question]
        for cid in rrf(sub_runs) + main[cfg.anchor_k:]:
            if len(chosen) >= cfg.max_evidence:
                break
            if cid not in chosen:
                chosen.append(cid)
        evidence = [_ev(n, found_by[c]["hit"], 0, found_by[c]["queries"]) for n, c in enumerate(chosen, 1)]
        return {"evidence": evidence,
                "llm_calls": [{"node": "retrieve", "prompt_tokens": 0, "completion_tokens": 0,
                               "secs": round(time.time() - t0, 3), "model": "retrieval"}]}

    # ---- new: sufficiency check + follow-up hop ----------------------------------------------------------
    def check(state):
        text, usage = llm.chat(
            [{"role": "system", "content": CHECK_SYSTEM},
             {"role": "user", "content": CHECK_USER.format(question=state["question"],
                                                           evidence=check_preview(state["evidence"], cfg.check_chars))}],
            max_tokens=cfg.check_max_tokens, json_mode=True)
        suff, missing, query = parse_check(text)
        return {"sufficient": suff, "missing": missing, "follow_up_query": query,
                "llm_calls": [{"node": "check", **usage}]}

    def follow_up(state):
        t0 = time.time()
        ev = list(state["evidence"])
        have = {e["chunk_id"] for e in ev}
        hop = state.get("hops", 0) + 1
        query = state["follow_up_query"]
        hits = pipeline.retrieve(query, state.get("paper_id"), k=cfg.hop_k + len(have))
        added = []
        for h in hits:
            if len(added) >= cfg.max_extra_evidence:
                break
            if h["chunk_id"] not in have:
                ev.append(_ev(len(ev) + 1, h, hop, [query]))
                have.add(h["chunk_id"])
                added.append(h["chunk_id"])
        log = state.get("hop_log", []) + [{"hop": hop, "missing": state.get("missing", ""), "query": query,
                                           "added_chunk_ids": added}]
        return {"evidence": ev, "hops": hop, "hop_log": log,
                "llm_calls": [{"node": "follow_up", "prompt_tokens": 0, "completion_tokens": 0,
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

    def after_retrieve(state):
        return "check" if state["evidence"] and cfg.max_hops > 0 else "answer"

    def after_check(state):
        return "answer" if state["sufficient"] else "follow_up"

    def after_follow_up(state):
        return "check" if state["hops"] < cfg.max_hops else "answer"

    g = StateGraph(MultiHopState)
    for name, fn in [("decompose", decompose), ("retrieve", retrieve), ("check", check),
                     ("follow_up", follow_up), ("answer", answer)]:
        g.add_node(name, fn)
    g.add_edge(START, "decompose")
    g.add_edge("decompose", "retrieve")
    g.add_conditional_edges("retrieve", after_retrieve, {"check": "check", "answer": "answer"})
    g.add_conditional_edges("check", after_check, {"answer": "answer", "follow_up": "follow_up"})
    g.add_conditional_edges("follow_up", after_follow_up, {"check": "check", "answer": "answer"})
    g.add_edge("answer", END)
    return g.compile()


class MultiHopAgent(ResearchAgent):
    """agent = MultiHopAgent(); out = agent.run("What datasets do they use?", paper_id="1909.00694")
    out has everything ResearchAgent returns plus hops, sufficient, missing, follow_up_query, hop_log."""

    def __init__(self, config=None, pipeline=None, llm=None):
        super().__init__(config or MultiHopConfig(), pipeline=pipeline, llm=llm)
        self.graph = build_multihop_agent(self.pipeline, self.llm, self.config)
