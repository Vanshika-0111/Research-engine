"""Cross-paper research agent (LangGraph): questions that need several papers.

    START -> retrieve_papers -> synthesize -> END
                     \\-- (nothing found) --> END

retrieve_papers : corpus-wide hybrid search (BM25 + FAISS dense, RRF) + cross-encoder reranking, then the
                  top `max_papers` papers, each with its best `chunks_per_paper` passages
                  (retrieval/cross_paper.py; the setup chosen on Day 12: paper recall@10 = 0.84).
synthesize      : the LLM decides which retrieved papers actually address the question and writes one answer,
                  a bullet per relevant paper plus an optional short comparison, citing [E#] after each claim.
                  Labels are mapped back to chunk IDs and to papers, so citations can be scored per paper.

This is a separate graph: the frozen within-paper agent (agent/graph.py, Day 7 v2) is not changed.

    from agent.cross_paper import CrossPaperAgent
    out = CrossPaperAgent().run("Which papers run experiments on HotpotQA, and what does each propose?")
    print(out["answer"]); print(out["cited_papers"])
"""
import operator
import re
import time
from dataclasses import dataclass
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from .graph import NO_ANSWER, normalize_citations, parse_citations

_SECTION_PREFIX = re.compile(r"^\[[^\]]*\]\s*")

CROSS_SYSTEM = """You answer research questions that span SEVERAL scientific papers (NLP / ML), using ONLY the
evidence passages provided. Passages are grouped by paper. A paper appears because a search found it; that does
not mean it is relevant.

Rules:
- First decide which papers actually address the question. Skip papers whose passages are only loosely related
  (for example, they only mention the topic while describing other work).
- Answer as a list with one bullet per relevant paper, in this form:
  - <short paper title>: <what this paper does or finds that answers the question> [E#]
- Every factual sentence must end with the label(s) of the passage(s) supporting it, written with plain ASCII
  square brackets exactly like [E3] or [E3][E4]. Do not use 【】 or any other bracket style.
- Cite only labels that appear in the evidence, and only passages that state the fact.
- After the list you may add at most two sentences comparing the papers, only if the passages support the
  comparison (cite them too).
- If no paper addresses the question, reply exactly: "The provided evidence does not answer this question."
- Do not use outside knowledge and do not mention papers that are not in the evidence."""

# v2 (2026-10-18), written after the v1 error analysis on the 19 cross-paper questions (so v2 numbers on those
# questions are not held-out). v1 cited exactly 5 papers in 8/19 answers and skipped ~23 required papers whose
# passages plainly showed the dataset/topic; comparison sentences were often uncited; one bullet used its title.
CROSS_SYSTEM_V2 = """You answer research questions that span SEVERAL scientific papers (NLP / ML), using ONLY the
evidence passages provided. Passages are grouped by paper. A paper appears because a search found it; that does
not mean it is relevant.

Rules:
- Check every paper, one by one. Include EVERY paper whose own passages show that the paper itself addresses the
  question (for example, it runs experiments on the dataset asked about, or its method is about the topic asked
  about). There is no limit on the number of bullets, and the papers are not ordered by relevance.
- Skip a paper only if its passages do not show this (for example, they only mention the topic while describing
  other work).
- Answer as a list with one bullet per relevant paper, in this form:
  - <short paper title>: <what this paper does or finds that answers the question> [E#]
- Describe each paper only with what its own passages state. Do not add details from the paper title or from
  your own knowledge.
- Every factual sentence must end with the label(s) of the passage(s) supporting it, written with plain ASCII
  square brackets exactly like [E3] or [E3][E4]. Do not use 【】 or any other bracket style.
- Cite only labels that appear in the evidence, and only passages that state the fact.
- After the list you may add at most two sentences comparing the papers. Every comparison sentence must also end
  with [E#] labels of passages that support it; if you cannot cite it, leave the comparison out.
- If no paper addresses the question, reply exactly: "The provided evidence does not answer this question."
- Do not mention papers that are not in the evidence."""

PROMPTS = {"v1": CROSS_SYSTEM, "v2": CROSS_SYSTEM_V2}

CROSS_USER = """Question: {question}

Evidence (grouped by paper):
{evidence}

Answer (one bullet per relevant paper, with [E#] citations):"""


@dataclass
class CrossPaperAgentConfig:
    llm_model: str = "openai/gpt-oss-120b"
    temperature: float = 0.0
    reasoning_effort: str = "low"
    max_papers: int = 10         # papers shown to the LLM (Day 12: 84% of required papers are in the top 10)
    chunks_per_paper: int = 2    # passages per paper
    passage_chars: int = 1000    # each passage is cut to this length (keeps the prompt ~6k tokens)
    rerank: bool = True          # cross-encoder over the top fused chunks (Day 12: +10 points paper recall@10)
    max_answer_tokens: int = 3072
    prompt: str = "v1"           # "v1" (first run, 2026-10-17) or "v2" (include every relevant paper; cited comparisons)


class CrossPaperState(TypedDict, total=False):
    question: str
    papers: list
    evidence: list
    answer: str
    cited_labels: list
    cited_chunk_ids: list
    cited_papers: list
    invalid_labels: list
    llm_calls: Annotated[list, operator.add]


def format_paper_evidence(evidence):
    """Evidence grouped under one header per paper: '### P2: <title> (paper 1909.00694)' then its [E#] passages."""
    lines, current = [], None
    for e in evidence:
        if e["paper_id"] != current:
            current = e["paper_id"]
            lines.append(f"\n### {e['paper_label']}: {e['title']} (paper {current})")
        lines.append(f"[{e['label']}] (Section: {e['section']})\n{e['text']}")
    return "\n".join(lines).strip()


def build_cross_paper_agent(retriever, llm, config=None):
    """`retriever` is a retrieval.cross_paper.CrossPaperRetriever, `llm` anything with .chat()."""
    cfg = config or CrossPaperAgentConfig()
    if cfg.prompt not in PROMPTS:
        raise ValueError(f"unknown prompt {cfg.prompt!r}; use one of {sorted(PROMPTS)}")
    system_prompt = PROMPTS[cfg.prompt]
    retriever.cfg.max_papers = cfg.max_papers
    retriever.cfg.chunks_per_paper = cfg.chunks_per_paper

    def retrieve_papers(state):
        t0 = time.time()
        papers = retriever.retrieve(state["question"], rerank=cfg.rerank)
        evidence, n = [], 0
        for p_idx, p in enumerate(papers, start=1):
            for c in p["chunks"]:
                n += 1
                text = _SECTION_PREFIX.sub("", c["text"], count=1)
                if len(text) > cfg.passage_chars:
                    text = text[:cfg.passage_chars].rsplit(" ", 1)[0] + " ..."
                evidence.append({"label": f"E{n}", "chunk_id": c["chunk_id"], "paper_id": p["paper_id"],
                                 "paper_label": f"P{p_idx}", "title": p["title"], "section": c["section"],
                                 "section_type": c["section_type"], "text": text})
        papers_out = [{"paper_id": p["paper_id"], "title": p["title"], "rank": p["rank"]} for p in papers]
        return {"papers": papers_out, "evidence": evidence,
                "llm_calls": [{"node": "retrieve_papers", "prompt_tokens": 0, "completion_tokens": 0,
                               "secs": round(time.time() - t0, 3), "model": "retrieval"}]}

    def synthesize(state):
        ev = state["evidence"]
        text, usage = llm.chat(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": CROSS_USER.format(question=state["question"],
                                                           evidence=format_paper_evidence(ev))}],
            max_tokens=cfg.max_answer_tokens)
        text = normalize_citations(text.strip())
        by_label = {e["label"]: e for e in ev}
        valid, invalid = parse_citations(text, set(by_label))
        return {"answer": text, "cited_labels": valid,
                "cited_chunk_ids": [by_label[l]["chunk_id"] for l in valid],
                "cited_papers": list(dict.fromkeys(by_label[l]["paper_id"] for l in valid)),
                "invalid_labels": invalid, "llm_calls": [{"node": "synthesize", **usage}]}

    def nothing_found(state):
        return {"answer": NO_ANSWER, "cited_labels": [], "cited_chunk_ids": [], "cited_papers": [],
                "invalid_labels": []}

    g = StateGraph(CrossPaperState)
    g.add_node("retrieve_papers", retrieve_papers)
    g.add_node("synthesize", synthesize)
    g.add_node("nothing_found", nothing_found)
    g.add_edge(START, "retrieve_papers")
    g.add_conditional_edges("retrieve_papers", lambda s: "synthesize" if s["evidence"] else "nothing_found",
                            {"synthesize": "synthesize", "nothing_found": "nothing_found"})
    g.add_edge("synthesize", END)
    g.add_edge("nothing_found", END)
    return g.compile()


class CrossPaperAgent:
    """agent = CrossPaperAgent(); out = agent.run("Which papers use the MultiWOZ dataset, and for which task?")"""

    def __init__(self, config=None, pipeline=None, llm=None):
        self.config = cfg = config or CrossPaperAgentConfig()
        if pipeline is None:
            from retrieval.pipeline import get_pipeline
            pipeline = get_pipeline()
        if llm is None:
            from .llm import make_llm
            llm = make_llm(cfg.llm_model, temperature=cfg.temperature, reasoning_effort=cfg.reasoning_effort)
        from retrieval.cross_paper import CrossPaperRetriever
        self.pipeline, self.llm = pipeline, llm
        self.retriever = CrossPaperRetriever(pipeline)
        self.graph = build_cross_paper_agent(self.retriever, llm, cfg)

    def run(self, question):
        t0 = time.time()
        out = self.graph.invoke({"question": question, "llm_calls": []})
        calls = out.get("llm_calls", [])
        out["tokens"] = {"prompt": sum(c["prompt_tokens"] for c in calls),
                         "completion": sum(c["completion_tokens"] for c in calls)}
        out["secs"] = round(time.time() - t0, 2)
        return out
