---
title: Research Evidence Engine
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
short_description: Evaluated answers of an agentic RAG system on NLP papers
---

# Agentic Research & Evidence Synthesis Engine — results explorer

Retrieval (BM25 + bge-small with FAISS, RRF fusion, bge-reranker-v2-m3 cross-encoder) and LangGraph agents
(gpt-oss-120b) over 1,585 NLP papers (QASPER). This page shows every answer from the evaluation runs — 100 held-out
test questions, 19 cross-paper questions and the 50-question multi-hop ablation — with clickable citations, QASPER
gold passages and the LLM judge's verdicts.

Held-out results: evidence Recall@5 0.795 (1,309 questions), LLM-judged citation accuracy 95.6% and faithfulness
96.9% (100 questions). Code, the live Gradio app and the report: https://github.com/Vanshika-0111/Research-engine

Data: QASPER (Dasigi et al., 2021), CC BY 4.0.
