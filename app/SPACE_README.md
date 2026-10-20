---
title: Research Evidence Engine
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.30.0
python_version: "3.11"
app_file: app/app.py
pinned: false
short_description: Cited answers over 1,585 NLP papers with LangGraph agents
---

# Agentic Research & Evidence Synthesis Engine

Ask questions about 1,585 NLP papers (QASPER). Hybrid retrieval (BM25 + bge-small with FAISS, RRF fusion,
bge-reranker-v2-m3 cross-encoder) feeds LangGraph agents (gpt-oss-120b via Groq, Gemini fallback) that answer
only from retrieved passages and cite every claim.

Held-out results: evidence Recall@5 0.795 (1,309 questions), LLM-judged citation accuracy 95.6% and faithfulness
96.9% (100 questions). Code, evaluation and report: https://github.com/Vanshika-0111/Research-engine

Data: QASPER (Dasigi et al., 2021), CC BY 4.0.
