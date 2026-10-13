"""Scientific retrieval: section-aware chunks -> BM25 + dense -> RRF -> cross-encoder rerank."""
from .config import RetrievalConfig
from .pipeline import RetrievalPipeline, retrieve

__all__ = ["RetrievalConfig", "RetrievalPipeline", "retrieve"]
