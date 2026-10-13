# Agentic Research & Evidence Synthesis Engine

Question answering over scientific papers with cited evidence. A retrieval stack built for scientific text
(section-aware chunks, BM25, dense embeddings, cross-encoder reranking) feeds LangGraph research agents that
break a question into parts, retrieve over several hops, and combine evidence across papers.

**Targets:** >85% citation accuracy and >75% evidence Recall@5 over 100 research queries, plus LLMOps
evaluation of faithfulness, citation accuracy, retrieval quality and token use.

## Retrieval results (QASPER, within-paper)

Corpus: 1,585 QASPER papers -> 28,541 section-aware chunks. Gold evidence is QASPER's annotated evidence
paragraphs mapped to chunk IDs. Evidence Recall@k = share of a question's gold chunks found in the top k, averaged.

| split | n | system | R@5 | R@10 | hit@1 | MRR@10 | ms/q |
|---|---|---|---|---|---|---|---|
| dev | 887 | BM25 | 0.628 | 0.875 | 0.226 | 0.429 | |
| dev | 887 | Dense (bge-small-en-v1.5) | 0.711 | 0.897 | 0.356 | 0.537 | |
| dev | 887 | Hybrid RRF | 0.710 | 0.904 | 0.345 | 0.532 | |
| dev | 887 | Hybrid + bge-reranker-v2-m3 | 0.748 | 0.934 | 0.356 | 0.550 | 256 |
| dev | 887 | + abstracts excluded | 0.793 | 0.943 | 0.466 | 0.631 | 256 |
| **test (held out)** | **1309** | **final pipeline** | **0.795** | **0.941** | **0.522** | **0.677** | **262** |

All choices were made on dev; test was run once at the end. Evidence Recall@5 of 0.795 is above the 0.75 target.

**Final pipeline:** BM25 top-50 + dense top-50 (within the paper) -> RRF (k=60) -> bge-reranker-v2-m3 (fp16)
re-scores the top 30 -> abstract chunks removed.

What drove the choices:
- Dense beats BM25 within a paper; BM25 beats dense across the whole corpus, where hybrid is best.
- bge-reranker-base gave no gain; bge-reranker-v2-m3 added +3.8 R@5.
- Adding the paper title to the reranker query hurt both rerankers, so it is not used.
- None of the gold evidence is in abstracts, yet the reranker put an abstract first for 31% of questions.
  Removing abstracts from the ranking added +4.5 R@5.
- fp16 reranking gives the same recall at ~5x the speed (1246 -> 256 ms/question on a T4).

## Layout

```
ingest/
  download_qasper.py      QASPER v0.3 -> data/raw/
  build_chunks.py         section-aware chunks + gold chunk IDs -> data/processed/
retrieval/
  config.py               paths + RetrievalConfig (the frozen Day 4 settings)
  data.py                 load chunks / questions
  bm25.py                 BM25Index
  dense.py                DenseIndex (bi-encoder, embeddings cached in cache/)
  hybrid.py               rrf(), HybridRetriever
  rerank.py               Reranker (cross-encoder), apply_scores()
  pipeline.py             RetrievalPipeline + retrieve(question, paper_id)
eval/
  metrics.py              hit@k, evidence recall@k, MRR, section diagnostics
  run_retrieval_eval.py   scores every stage; reproduces the table above
experiments/              original Day 3 / Day 4 scripts, kept for reference
data/  cache/             generated, not committed
results/                  metric JSONs (committed)
```

## Running in Colab

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/research-engine
!pip install -q -r requirements.txt
```

Build the data once (skip if `data/processed/` is already there):

```
!python ingest/download_qasper.py
!python ingest/build_chunks.py
```

Reproduce the held-out number (use a GPU runtime):

```
!python -m eval.run_retrieval_eval --split test     # final ev_recall@5 = 0.795, prints REPRODUCED
!python -m eval.run_retrieval_eval --split dev --diagnostics
!python -m eval.run_retrieval_eval --limit 100      # quick smoke test
```

Chunk embeddings and reranker scores are cached in `cache/` under the same file names the Day 3/4 scripts used,
so an existing Day 4 cache is picked up and the test run finishes in about a minute. Pass `--refresh-cache` to
re-run the cross-encoder.

Use the retriever from code:

```python
from retrieval import retrieve

hits = retrieve("What baselines do they compare against?", paper_id="1909.00694", k=5)
for h in hits:
    print(h["rank"], round(h["score"], 3), h["chunk_id"], h["section"])
```

or from the shell: `python -m retrieval "What baselines do they compare against?" --paper-id 1909.00694`.
`paper_id=None` searches the whole corpus. Each hit has `rank, chunk_id, paper_id, title, section,
section_type, score, text`.

## Roadmap

- [x] Ingest + section-aware chunking with gold evidence mapping (Days 1-2)
- [x] BM25, dense, hybrid retrieval (Day 3)
- [x] Cross-encoder reranking, section filtering (Day 4)
- [x] Refactor into `retrieval/` and `eval/` packages (Day 5)
- [ ] LangGraph agents: query decomposition, multi-hop retrieval, cross-paper evidence aggregation
- [ ] Cited answer generation + citation accuracy on 100 research queries
- [ ] LLMOps evaluation: faithfulness, citation accuracy, retrieval quality, token efficiency
