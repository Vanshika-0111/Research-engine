"""Settings for the research agent."""
from dataclasses import dataclass


@dataclass
class AgentConfig:
    # Groq model ID. List what your key can use with: Groq().models.list()
    llm_model: str = "openai/gpt-oss-120b"
    temperature: float = 0.0
    # reasoning models (gpt-oss, qwen3) think before answering; "low" keeps that short, cheap and fast
    reasoning_effort: str = "low"
    max_sub_questions: int = 3     # decomposition produces at most this many sub-questions
    anchor_k: int = 5              # the original question's top-k chunks always become evidence first
    k_per_query: int = 4           # chunks retrieved per sub-question
    max_evidence: int = 7          # total evidence chunks shown to the LLM (anchors + sub-question extras)
    # completion budgets include the model's hidden reasoning, so they are generous
    decompose_max_tokens: int = 1024
    max_answer_tokens: int = 2048
