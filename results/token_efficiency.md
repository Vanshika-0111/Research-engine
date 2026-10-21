# Token efficiency (from the saved summaries; `python -m eval.token_report`)

| system | split | LLM calls / q | agent tokens / q (prompt + completion) | judge tokens / q | agent tokens, whole run | quality | agent tokens per correct answer | s / q |
|---|---|---|---|---|---|---|---|---|
| one-paper agent (frozen, Day 7 v2) | test, n=100 | 2 (decompose, answer) | 3,007 (2,826 + 181) | 1,208 | 300,700 | 0.830 answer correctness | 3,623 | – |
| one-paper agent (frozen, Day 7 v2) | dev, n=50 | 2 | 3,070 (2,884 + 186) | 1,279 | 153,500 | 0.680 answer correctness | 4,515 | 18.24 |
| multi-hop agent (Day 14) | dev, n=50 | 3 (+ sufficiency check) | 4,485 (4,216 + 269) | 1,275 | 224,250 | 0.770 answer correctness | 5,825 | 28.45 |
| cross-paper agent v1 | 19 questions | 1 (synthesize) | 4,711 (4,269 + 442) | 2,442 | 89,509 | 0.910 citation support | – | 30.17 |
| cross-paper agent v2 | 19 questions | 1 | 4,909 (4,371 + 538) | 2,653 | 93,271 | 0.990 citation support | – | 32.19 |

Tokens include gpt-oss-120b's hidden reasoning (reasoning effort low). Seconds per question are mostly Groq free-tier pacing; the test run's latency is omitted because its first 19 questions ran on a CPU.
