"""Loaders for the Day 2 outputs (data/processed/chunks.parquet, qasper_questions.jsonl)."""
import ast
import json

import pandas as pd

from .config import CHUNKS_PATH, QUESTIONS_PATH


def as_list(v):
    """gold_chunk_ids may come back as a list, a numpy array, or a stringified list."""
    if isinstance(v, str):
        v = ast.literal_eval(v) if v.strip().startswith("[") else [v]
    return [str(x) for x in (v if v is not None else [])]


def load_chunks(path=CHUNKS_PATH):
    if not path.exists():
        raise SystemExit(f"{path} not found. Run ingest/download_qasper.py then ingest/build_chunks.py.")
    return pd.read_parquet(path).reset_index(drop=True)


def load_questions(split="dev", known_chunk_ids=None, limit=None, path=QUESTIONS_PATH):
    """Questions for one split ('dev' | 'train' | 'test' | 'all'), each with a 'gold' list of chunk IDs.

    Questions whose gold evidence matches no known chunk are dropped (unanswerable / table-only evidence).
    `limit` is applied after that filter, as in Day 3/4.
    """
    with open(path, encoding="utf-8") as fh:
        qs = [json.loads(line) for line in fh]
    if split != "all":
        qs = [q for q in qs if q.get("split") == split]
    for q in qs:
        q["gold"] = as_list(q.get("gold_chunk_ids"))
    known = set(known_chunk_ids) if known_chunk_ids is not None else None
    qs = [q for q in qs if q["gold"] and (known is None or any(g in known for g in q["gold"]))]
    return qs[:limit] if limit else qs


def load_data(split="dev", limit=None):
    chunks = load_chunks()
    return chunks, load_questions(split, set(chunks["chunk_id"]), limit)
