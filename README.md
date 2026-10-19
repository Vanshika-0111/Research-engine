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
  dense.py                DenseIndex (bi-encoder, embeddings cached in cache/; corpus-wide search via FAISS)
  faiss_index.py          FaissIndex: IndexFlatIP (exact) or IndexHNSWFlat (approximate) over the chunk embeddings
  cross_paper.py          CrossPaperRetriever: corpus-wide hybrid search (+ rerank), evidence grouped by paper
  hybrid.py               rrf(), HybridRetriever
  rerank.py               Reranker (cross-encoder), apply_scores()
  pipeline.py             RetrievalPipeline + retrieve(question, paper_id)
agent/
  config.py               AgentConfig (Groq model, sub-question / evidence limits)
  llm.py                  Groq client that logs tokens + latency per call
  prompts.py              decomposition and cited-answer prompts
  graph.py                LangGraph: decompose -> retrieve -> answer; ResearchAgent.run()
  cross_paper.py          LangGraph cross-paper mode: retrieve_papers -> synthesize; CrossPaperAgent.run()
eval/
  metrics.py              retrieval metrics + citation metrics
  run_retrieval_eval.py   scores every retrieval stage; reproduces the table above
  run_agent_eval.py       agent on sampled QASPER questions: citation accuracy, tokens, latency (--tag keeps runs apart)
  inspect_agent_eval.py   side-by-side review of citation misses vs gold passages
  judge_agent_eval.py     LLM judge on saved answers: citation support, faithfulness, answer correctness
  compare_judges.py       agreement between two judges (e.g. gpt-oss-20b vs Gemini) on the same answers
  reference_filter.py     fixed rule that drops questions whose references only point to a citation/table
  benchmark_faiss.py      FAISS (flat, HNSW grid) vs numpy for corpus-wide dense search: exactness, recall, latency
  cross_paper_questions.jsonl  19 hand-written cross-paper questions with required / acceptable papers (frozen)
  run_cross_paper_eval.py paper-level recall of BM25 / dense / hybrid / hybrid+rerank on those questions
  run_cross_paper_agent_eval.py  cross-paper agent on the 19 questions: papers shown / cited, cited-paper precision
  judge_cross_paper.py    LLM judge for cross-paper answers: citation support, faithfulness, per-paper support
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

## FAISS for corpus-wide dense search (Day 10)

Corpus-wide dense search (`paper_id=None`, the cross-paper setting) goes through a FAISS index over the cached
bge-small-en-v1.5 chunk embeddings. `RetrievalConfig.corpus_index` picks the backend: `"flat"` (default,
`IndexFlatIP`, exact), `"hnsw"` (`IndexHNSWFlat`, approximate; saved to `cache/faiss_hnsw_*.index`), or
`"numpy"` (the Day 3 search). Embeddings are unit length, so inner product = cosine, the same score as before.
Within-paper search (about 18 chunks per paper) stays exact numpy and never builds a FAISS index, so the
frozen within-paper results above are unchanged.

```
!python -m eval.benchmark_faiss            # results/faiss_benchmark.md: flat vs numpy, HNSW M x efSearch grid
```

The benchmark encodes every dev/test question once, then compares all backends on the same vectors: whether
FAISS flat returns the same top-k as numpy (it may swap items whose scores differ only by float rounding,
~1e-7), gold evidence metrics, ANN recall@k of HNSW against exact search, build time, index size, and search
latency per query.


## Cross-paper questions (Day 11)

QASPER questions always belong to one paper. To measure the cross-paper mode, `eval/cross_paper_questions.jsonl`
holds 19 hand-written questions that need several papers: 10 about a dataset ("Which papers run experiments on
HotpotQA, and what does each propose?") and 9 about a topic (sarcasm detection, code-switching, question generation,
fake news, ...). Each question lists `required_papers` (the paper runs experiments on the dataset, or the topic is
its main subject; checked by hand) and `acceptable_papers` (minor mentions: not required, not penalised); 7.6
required papers per question on average. The set was fixed before any cross-paper system was built.

`retrieval/cross_paper.py` searches the whole corpus (BM25 + FAISS dense, RRF, optional cross-encoder) and groups
the ranked chunks by paper. `python -m eval.run_cross_paper_eval` scores paper recall@5/10/20, precision@10 and MRR
(`--no-rerank` skips the cross-encoder, which needs a GPU to be quick).

Results (19 questions, `results/cross_paper_retrieval.md`): paper recall@10 is 0.713 for BM25, 0.725 for dense,
0.735 for hybrid and **0.840 for hybrid + reranker** (recall@20 0.959, MRR 0.974). BM25 is better on dataset
questions (exact names), dense on topic questions (paraphrases); the cross-encoder helps on 18 of 19 questions.

**Cross-paper agent** (`agent/cross_paper.py`, a second LangGraph graph; the within-paper agent is unchanged):
`retrieve_papers` (hybrid + reranker, top 10 papers x 2 passages) -> `synthesize` (one bullet per relevant paper,
`[E#]` citations mapped back to chunks and papers). `python -m eval.run_cross_paper_agent_eval` runs it on the 19
questions (papers shown vs cited, cited-paper precision; `--resume` after a daily-limit stop) and
`python -m eval.judge_cross_paper results/cross_paper_agent.jsonl` checks citation support and faithfulness.

## Research agent (Day 6)

A LangGraph graph over the retriever:

```
question -> decompose (LLM, JSON: up to 3 sub-questions)
         -> retrieve  (question's top 5 first, then sub-question results via RRF, up to 7 = E1..E7)
         -> answer    (LLM answers from the evidence only, citing [E#] after each claim)
```

Citation labels are mapped back to chunk IDs, so every citation can be checked against QASPER's gold evidence.
Each LLM call logs prompt/completion tokens and latency. LLM: Groq `openai/gpt-oss-120b` with low reasoning effort (free API key). The model is one line in `agent/config.py`;
list the models your key can use with `Groq().models.list()`.

Setup in Colab: get a key at console.groq.com -> API Keys, then add a Colab secret named `GROQ_API_KEY`.
Commands run with `!python` can't read Colab secrets, so copy the key into an environment variable once per session:

```python
import os
from google.colab import userdata
os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY").strip()
```

```
!python -m agent "What datasets do they evaluate on?" --paper-id 1909.00694
!python -m eval.run_agent_eval --n 5      # smoke test
!python -m eval.run_agent_eval --n 20     # Day 6 numbers on dev (tune here)
!python -m eval.run_agent_eval --n 50 --tag v2 --resume   # continue a run stopped by Groq's daily token limit
!python -m eval.run_agent_eval --n 50 --split test   # once, at the end
```

Groq's free tier allows about 200k tokens per day per model (about 65 agent questions with gpt-oss-120b). Eval
results are written one question at a time; if the daily limit is hit the run stops cleanly, and `--resume`
continues later. An existing results file is never overwritten without `--resume` or `--overwrite`.

```python
from agent import ResearchAgent
out = ResearchAgent().run("What baselines do they compare against?", paper_id="1909.00694")
print(out["answer"], out["cited_chunk_ids"])
```

Review the misses side by side with the gold passages (also saved as `*_review.md`):

```
!python -m eval.inspect_agent_eval results/agent_eval_dev_n20.jsonl
```

Citation precision (share of cited chunks that are gold evidence) is a strict measure: a passage can support
a claim without being one of QASPER's annotated evidence paragraphs. The LLM judge reads what was actually
cited and scores citation support, faithfulness and answer correctness. It uses a different Groq model from the
agent (own daily quota, no self-grading) and only reads the saved `.jsonl`, so the agent is not re-run:

```
!python -m eval.judge_agent_eval results/agent_eval_dev_n20_v2.jsonl          # judge = openai/gpt-oss-20b
!python -m eval.judge_agent_eval results/agent_eval_dev_n20_v2.jsonl --model qwen/qwen3.8-27b
!python -m eval.judge_agent_eval results/agent_eval_dev_n20_v2.jsonl --model gemini-3.8-flash   # Gemini API
```

`agent/llm.py` has two providers behind one `chat()` interface: Groq (`GroqLLM`) and the Google Gemini API
(`GeminiLLM`, needs a `GEMINI_API_KEY` env var or Colab secret). `make_llm(model)` picks Gemini for model names
starting with `gemini`. A Gemini judge is a different model family from the gpt-oss agent, so it is not
grading its own kind of output. Gemini 3 models are run at their default temperature (Google advises against
lowering it), so Gemini verdicts are not fully deterministic. Per-minute 429s are waited out; a per-day quota
stops the run cleanly and `--resume` continues.

Two judges on the same answers, with exact agreement, Cohen's kappa and every disagreement listed:

```
!python -m eval.compare_judges results/agent_eval_dev_n20_v2_judge-gpt-oss-20b.jsonl \
                               results/agent_eval_dev_n20_v2_judge-gemini-38-flash.jsonl
```

**Reference-quality filter.** Some QASPER reference answers only point elsewhere ("BIBREF9 ; BIBREF8",
"the same datasets as BIBREF7", "Answer with content missing ..."), so no answer can match them. A fixed,
mechanical rule (`eval/reference_filter.py`, decided before the test run) excludes a question when all its
references are like that. Every agent and judge summary reports both the full and the filtered numbers
(`summary` and `summary_filtered`). On dev n=50 it excludes 3 questions (Q27, Q39, Q45). To see it on any
saved results file: `!python -m eval.reference_filter results/agent_eval_dev_n50_v2.jsonl`.

## Roadmap

- [x] Ingest + section-aware chunking with gold evidence mapping (Days 1-2)
- [x] BM25, dense, hybrid retrieval (Day 3)
- [x] Cross-encoder reranking, section filtering (Day 4)
- [x] Refactor into `retrieval/` and `eval/` packages (Day 5)
- [x] First LangGraph agent: query decomposition, fused retrieval, cited answers, citation eval (Day 6)
- [x] FAISS index for corpus-wide dense search, benchmarked against numpy (Day 10)
- [ ] Multi-hop: sufficiency check that loops back to retrieval; cross-paper evidence aggregation
- [ ] Cited answer generation + citation accuracy on 100 research queries
- [ ] LLMOps evaluation: faithfulness, citation accuracy, retrieval quality, token efficiency
