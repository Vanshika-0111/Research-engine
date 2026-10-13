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
    k_per_query: int = 4           # chunks retrieved per query (original question + each sub-question)
    max_evidence: int = 6          # evidence chunks shown to the LLM after merging (keeps prompts ~3-4k tokens)
    # completion budgets include the model's hidden reasoning, so they are generous
    decompose_max_tokens: int = 1024
    max_answer_tokens: int = 2048
