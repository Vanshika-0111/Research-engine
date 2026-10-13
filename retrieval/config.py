"""Paths and the frozen Day 4 retrieval settings.

All paths are resolved from the project root (the folder that contains retrieval/),
so scripts work no matter which directory you launch them from.
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"

CHUNKS_PATH = DATA_DIR / "chunks.parquet"
QUESTIONS_PATH = DATA_DIR / "qasper_questions.jsonl"


@dataclass
class RetrievalConfig:
    """Final Day 4 setup (chosen on dev, scored once on test: evidence R@5 = 0.795)."""
    dense_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    pool: int = 50            # candidates from BM25 and from dense, each, before fusion
    rrf_k: int = 60           # reciprocal rank fusion constant
    rerank_top: int = 30      # how many fused candidates the cross-encoder re-scores
    reranker_max_length: int = 512
    reranker_batch_size: int = 64
    fp16: bool = True         # half-precision reranker; only applied on a GPU
    # section types removed from the final ranking (0% of gold evidence is in abstracts)
    exclude_sections: tuple = field(default_factory=lambda: ("abstract",))
