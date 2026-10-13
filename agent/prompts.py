"""Prompts for the research agent."""

DECOMPOSE_SYSTEM = """You plan searches over ONE scientific paper (NLP / ML).
Split the user's question into short, self-contained search queries that together cover everything needed to answer it.

Rules:
- If the question asks for a single fact, return just one query (a clear rewrite of the question).
- Otherwise return at most {max_n} queries, each about one thing (e.g. the dataset, the baseline, the metric, the result).
- Replace vague references ("they", "the proposed model") with descriptive words when possible.
- Do not answer the question.

Respond with JSON only: {{"sub_questions": ["...", "..."]}}"""

ANSWER_SYSTEM = """You answer questions about a scientific paper using ONLY the evidence passages provided.

Rules:
- Every sentence that states a fact must end with the label(s) of the passage(s) supporting it, written with plain
  ASCII square brackets exactly like [E2] or [E1][E4]. Do not use 【】 or any other bracket style.
- Use only labels that appear in the evidence. Never cite a passage that does not support the sentence.
- Be concise: answer in 1-4 sentences. Short factual answers (names, numbers, yes/no) are fine.
- If the evidence does not contain the answer, reply exactly: "The provided evidence does not answer this question."
- Do not use outside knowledge."""

ANSWER_USER = """Question: {question}

Evidence:
{evidence}

Answer (with [E#] citations):"""


def format_evidence(evidence):
    return "\n\n".join(f"[{e['label']}] (Section: {e['section']})\n{e['text']}" for e in evidence)
