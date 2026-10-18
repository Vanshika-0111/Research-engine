# Cross-paper retrieval (19 questions, paper-level)

| system | paper_recall@5 | paper_recall@10 | paper_recall@20 | paper_prec@10 | mrr |
|---|---|---|---|---|---|
| bm25 | 0.487 | 0.713 | 0.860 | 0.679 | 0.882 |
| bm25 [dataset] | 0.596 | 0.761 | 0.857 | 0.600 | 0.950 |
| bm25 [topic] | 0.366 | 0.661 | 0.864 | 0.767 | 0.806 |
| dense | 0.479 | 0.725 | 0.851 | 0.674 | 0.888 |
| dense [dataset] | 0.449 | 0.674 | 0.786 | 0.540 | 0.787 |
| dense [topic] | 0.513 | 0.782 | 0.924 | 0.822 | 1.000 |
| hybrid | 0.499 | 0.735 | 0.906 | 0.705 | 0.895 |
| hybrid [dataset] | 0.511 | 0.711 | 0.873 | 0.590 | 0.867 |
| hybrid [topic] | 0.486 | 0.763 | 0.942 | 0.833 | 0.926 |

## Per question (paper recall@10)

| id | type | required | bm25 R@10 | dense R@10 | hybrid R@10 |
|---|---|---|---|---|---|
| XP01 | dataset | 6 | 0.67 | 1.00 | 1.00 |
| XP02 | dataset | 8 | 0.75 | 0.75 | 0.62 |
| XP03 | dataset | 8 | 0.50 | 0.88 | 0.75 |
| XP04 | dataset | 7 | 0.86 | 0.57 | 0.86 |
| XP05 | dataset | 8 | 0.62 | 0.38 | 0.50 |
| XP06 | dataset | 6 | 0.83 | 0.00 | 0.33 |
| XP07 | dataset | 3 | 1.00 | 0.67 | 0.67 |
| XP08 | dataset | 3 | 1.00 | 1.00 | 1.00 |
| XP09 | dataset | 8 | 0.75 | 0.62 | 0.75 |
| XP10 | dataset | 8 | 0.62 | 0.88 | 0.62 |
| XP11 | topic | 11 | 0.55 | 0.82 | 0.73 |
| XP12 | topic | 10 | 0.60 | 0.60 | 0.60 |
| XP13 | topic | 12 | 0.50 | 0.75 | 0.58 |
| XP14 | topic | 7 | 0.71 | 0.86 | 0.86 |
| XP15 | topic | 7 | 0.57 | 0.71 | 0.71 |
| XP16 | topic | 12 | 0.58 | 0.75 | 0.75 |
| XP17 | topic | 12 | 0.83 | 0.75 | 0.83 |
| XP18 | topic | 4 | 1.00 | 1.00 | 1.00 |
| XP19 | topic | 5 | 0.60 | 0.80 | 0.80 |
