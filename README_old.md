# Agentic Research & Evidence Synthesis Engine

Scientific question answering over NLP papers (QASPER corpus): section-aware chunking,
hybrid BM25 + dense retrieval, cross-encoder reranking, a LangGraph research agent,
and an evaluation harness for retrieval quality, citation accuracy, faithfulness and token use.

## Setup
    python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

## Day 1–2: data
    python ingest/download_qasper.py     # downloads + extracts QASPER v0.3 into data/raw/
    python ingest/build_chunks.py        # writes data/processed/{papers,chunks}.parquet + qasper_questions.jsonl

## Layout
    ingest/     download and chunking
    index/      BM25 and vector indexes
    retrieval/  hybrid fusion and reranking
    agent/      LangGraph state, nodes, graph
    eval/       questions, metrics, judge, runner
    app/        demo
    configs/    model names and parameters
