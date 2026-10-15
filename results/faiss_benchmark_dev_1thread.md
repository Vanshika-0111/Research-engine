# FAISS vs numpy, corpus-wide dense search (28541 chunks, dim 384)

faiss 1.15.1, numpy 2.1.3, 2 CPUs, FAISS threads 1. Latency = search only (query vectors pre-encoded), one query at a time; batch = one call for all queries, per query.

| index | build s | size MB |
|---|---|---|
| flat | 0.03 | 43.8 |
| numpy | 0.0 | 43.8 |
| hnsw_M32 | 15.26 | 51.6 |

## dev (n=887)

- Flat vs numpy top-10: identical order 0.9989, identical set 1.0000, max score gap where different 0.0e+00
- Flat vs numpy top-50: identical order 0.9944, identical set 1.0000, max score gap where different 1.2e-07
- Flat vs numpy top-100: identical order 0.9775, identical set 0.9989, max score gap where different 1.2e-07
- Gold metrics equal: True -> **exact**
- Day 3 dev corpus dense check: ev_recall@5 0.1098 vs 0.1098, ev_recall@10 0.1550 vs 0.1550, hit@1 0.0620 vs 0.0620

| system | ANN@10 | ANN@50 | R@5 | R@10 | R@50 | hit@1 | median ms | p95 ms | batch ms/q |
|---|---|---|---|---|---|---|---|---|---|
| numpy argsort (current) | 1.0000 | 1.0000 | 0.1098 | 0.1550 | 0.2482 | 0.0620 | 2.993 | 4.271 |  |
| numpy argpartition | 1.0000 | 1.0000 |  |  |  |  | 2.245 | 3.305 |  |
| FAISS IndexFlatIP | 1.0000 | 1.0000 | 0.1098 | 0.1550 | 0.2482 | 0.0620 | 2.593 | 3.411 | 0.262 |
| FAISS HNSW M=32 efSearch=32 | 0.9023 | 0.8046 | 0.1105 | 0.1462 | 0.2297 | 0.0575 | 0.174 | 0.242 | 0.163 |
| FAISS HNSW M=32 efSearch=64 | 0.9545 | 0.9021 | 0.1066 | 0.1488 | 0.2389 | 0.0586 | 0.260 | 0.336 | 0.225 |
| FAISS HNSW M=32 efSearch=128 | 0.9835 | 0.9608 | 0.1087 | 0.1520 | 0.2477 | 0.0609 | 0.415 | 0.533 | 0.387 |

DenseIndex.search(ids=None) numpy vs flat on 200 dev questions, top-50: identical 0.9900
