# FAISS vs numpy, corpus-wide dense search (28541 chunks, dim 384)

faiss 1.15.1, numpy 2.1.3, 2 CPUs, FAISS threads 2. Latency = search only (query vectors pre-encoded), one query at a time; batch = one call for all queries, per query.

| index | build s | size MB |
|---|---|---|
| flat | 0.03 | 43.8 |
| numpy | 0.0 | 43.8 |
| hnsw_M16 | 9.29 | 48.0 |
| hnsw_M32 | 10.41 | 51.6 |

## dev (n=887)

- Flat vs numpy top-10: identical order 0.9977, identical set 1.0000, max score gap where different 2.4e-07
- Flat vs numpy top-50: identical order 0.9853, identical set 1.0000, max score gap where different 4.2e-07
- Flat vs numpy top-100: identical order 0.9504, identical set 0.9989, max score gap where different 5.4e-07
- Gold metrics equal: True -> **exact**
- Day 3 dev corpus dense check: ev_recall@5 0.1098 vs 0.1098, ev_recall@10 0.1550 vs 0.1550, hit@1 0.0620 vs 0.0620

| system | ANN@10 | ANN@50 | R@5 | R@10 | R@50 | hit@1 | median ms | p95 ms | batch ms/q |
|---|---|---|---|---|---|---|---|---|---|
| numpy argsort (current) | 1.0000 | 1.0000 | 0.1098 | 0.1550 | 0.2482 | 0.0620 | 3.152 | 4.308 |  |
| numpy argpartition | 1.0000 | 1.0000 |  |  |  |  | 2.325 | 3.815 |  |
| FAISS IndexFlatIP | 1.0000 | 1.0000 | 0.1098 | 0.1550 | 0.2482 | 0.0620 | 6.304 | 10.543 | 0.250 |
| FAISS HNSW M=16 efSearch=16 | 0.7823 | 0.6276 | 0.0995 | 0.1337 | 0.2037 | 0.0519 | 0.099 | 0.148 | 0.058 |
| FAISS HNSW M=16 efSearch=32 | 0.8786 | 0.7653 | 0.1021 | 0.1401 | 0.2184 | 0.0541 | 0.138 | 0.199 | 0.085 |
| FAISS HNSW M=16 efSearch=64 | 0.9408 | 0.8789 | 0.1047 | 0.1474 | 0.2331 | 0.0575 | 0.214 | 0.298 | 0.134 |
| FAISS HNSW M=16 efSearch=128 | 0.9752 | 0.9461 | 0.1070 | 0.1516 | 0.2411 | 0.0586 | 0.354 | 0.541 | 0.222 |
| FAISS HNSW M=16 efSearch=256 | 0.9911 | 0.9798 | 0.1087 | 0.1539 | 0.2471 | 0.0609 | 0.593 | 0.739 | 0.440 |
| FAISS HNSW M=32 efSearch=16 | 0.8101 | 0.6678 | 0.0983 | 0.1354 | 0.2155 | 0.0507 | 0.119 | 0.165 | 0.071 |
| FAISS HNSW M=32 efSearch=32 | 0.9023 | 0.8046 | 0.1105 | 0.1462 | 0.2297 | 0.0575 | 0.167 | 0.242 | 0.104 |
| FAISS HNSW M=32 efSearch=64 | 0.9545 | 0.9021 | 0.1066 | 0.1488 | 0.2389 | 0.0586 | 0.276 | 0.390 | 0.157 |
| FAISS HNSW M=32 efSearch=128 | 0.9835 | 0.9608 | 0.1087 | 0.1520 | 0.2477 | 0.0609 | 0.395 | 0.499 | 0.269 |
| FAISS HNSW M=32 efSearch=256 | 0.9940 | 0.9857 | 0.1087 | 0.1539 | 0.2471 | 0.0609 | 0.717 | 0.946 | 0.511 |

## test (n=1309)

- Flat vs numpy top-10: identical order 0.9985, identical set 1.0000, max score gap where different 3.6e-07
- Flat vs numpy top-50: identical order 0.9763, identical set 1.0000, max score gap where different 3.6e-07
- Flat vs numpy top-100: identical order 0.9458, identical set 0.9992, max score gap where different 4.8e-07
- Gold metrics equal: True -> **exact**

| system | ANN@10 | ANN@50 | R@5 | R@10 | R@50 | hit@1 | median ms | p95 ms | batch ms/q |
|---|---|---|---|---|---|---|---|---|---|
| numpy argsort (current) | 1.0000 | 1.0000 | 0.1246 | 0.1644 | 0.2645 | 0.0749 | 3.004 | 3.857 |  |
| numpy argpartition | 1.0000 | 1.0000 |  |  |  |  | 2.327 | 2.906 |  |
| FAISS IndexFlatIP | 1.0000 | 1.0000 | 0.1246 | 0.1644 | 0.2645 | 0.0749 | 6.309 | 9.706 | 0.232 |
| FAISS HNSW M=16 efSearch=16 | 0.7759 | 0.6218 | 0.1052 | 0.1383 | 0.2243 | 0.0619 | 0.104 | 0.178 | 0.081 |
| FAISS HNSW M=16 efSearch=32 | 0.8769 | 0.7649 | 0.1139 | 0.1487 | 0.2370 | 0.0665 | 0.140 | 0.193 | 0.092 |
| FAISS HNSW M=16 efSearch=64 | 0.9416 | 0.8751 | 0.1179 | 0.1564 | 0.2475 | 0.0703 | 0.210 | 0.282 | 0.134 |
| FAISS HNSW M=16 efSearch=128 | 0.9762 | 0.9438 | 0.1254 | 0.1639 | 0.2596 | 0.0749 | 0.349 | 0.509 | 0.265 |
| FAISS HNSW M=16 efSearch=256 | 0.9904 | 0.9777 | 0.1239 | 0.1636 | 0.2613 | 0.0749 | 0.605 | 0.814 | 0.422 |
| FAISS HNSW M=32 efSearch=16 | 0.8086 | 0.6662 | 0.1100 | 0.1468 | 0.2298 | 0.0611 | 0.120 | 0.169 | 0.074 |
| FAISS HNSW M=32 efSearch=32 | 0.8963 | 0.8004 | 0.1151 | 0.1526 | 0.2418 | 0.0680 | 0.166 | 0.226 | 0.103 |
| FAISS HNSW M=32 efSearch=64 | 0.9532 | 0.9014 | 0.1197 | 0.1585 | 0.2516 | 0.0733 | 0.244 | 0.322 | 0.164 |
| FAISS HNSW M=32 efSearch=128 | 0.9820 | 0.9598 | 0.1233 | 0.1602 | 0.2582 | 0.0733 | 0.407 | 0.543 | 0.287 |
| FAISS HNSW M=32 efSearch=256 | 0.9931 | 0.9857 | 0.1254 | 0.1636 | 0.2632 | 0.0741 | 0.706 | 0.889 | 0.480 |

DenseIndex.search(ids=None) numpy vs flat on 200 dev questions, top-50: identical 0.9850
