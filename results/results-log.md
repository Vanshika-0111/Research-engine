# Retrieval results log

Corpus: QASPER, 1,585 papers, 28,541 section-aware chunks (`data/processed/chunks.parquet`).
Gold: `gold_chunk_ids` in `data/processed/qasper_questions.jsonl`. Scope: within the question's paper.
Metric: evidence Recall@k = fraction of a question's gold chunks found in the top k, averaged.

## Final pipeline (Day 4)
Hybrid (BM25 + bge-small-en-v1.5, RRF k=60, top-50 each) -> bge-reranker-v2-m3 (fp16) re-scores top-30 -> abstract chunks excluded from evidence rankings.

| split | n | system | R@5 | R@10 | hit@1 | MRR@10 | ms/q |
|---|---|---|---|---|---|---|---|
| dev | 887 | BM25 | 0.628 | 0.875 | 0.226 | 0.429 | |
| dev | 887 | Dense | 0.711 | 0.897 | 0.356 | 0.537 | |
| dev | 887 | Hybrid RRF | 0.710 | 0.904 | 0.345 | 0.532 | |
| dev | 887 | Hybrid + v2-m3 rerank | 0.748 | 0.934 | 0.356 | 0.550 | 256 |
| dev | 887 | + abstracts excluded | 0.793 | 0.943 | 0.466 | 0.631 | 256 |
| **test (held out)** | **1309** | **final pipeline** | **0.795** | **0.941** | **0.522** | **0.677** | **262** |

## Decisions and findings
- Dense beats BM25 within a paper; BM25 beats dense across the whole corpus (hybrid best there: R@5 0.156 vs 0.134).
- Corpus-scope numbers are low because QASPER questions don't name their paper ("they"); not used as the benchmark.
- bge-reranker-base gave no gain (and hurt hit@1); bge-reranker-v2-m3 gave +3.8 R@5.
- Prepending the paper title to the reranker query hurt badly with both rerankers -> not used.
- Diagnostic: 0% of gold evidence is in abstracts, but the reranker ranked an abstract first for 31% of questions. Excluding abstracts: hybrid 0.710 -> 0.747, reranker adds a further +4.6 -> 0.793.
- fp16 reranking: identical recall, 1246 -> 256 ms/question (~5x faster).
- Choices were made on dev; test was run once at the end: 0.795 vs 0.793 dev, so no overfitting to dev.
