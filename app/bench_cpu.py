"""How fast is a cross-encoder on CPU (like the free Hugging Face Space)? No LLM calls.

    CUDA_VISIBLE_DEVICES="" python -m app.bench_cpu                       # the frozen bge-reranker-v2-m3
    CUDA_VISIBLE_DEVICES="" python -m app.bench_cpu --model cross-encoder/ms-marco-MiniLM-L-6-v2
    CUDA_VISIBLE_DEVICES="" python -m app.bench_cpu --model BAAI/bge-reranker-base --max-length 256
Times 30 pairs (within-paper mode) and 50 / 100 pairs (cross-paper mode) with real chunks, using ALL CPU cores.
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from retrieval.config import RetrievalConfig  # noqa: E402
from retrieval.data import load_chunks  # noqa: E402
from retrieval.rerank import Reranker  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=RetrievalConfig().reranker_model)
ap.add_argument("--max-length", type=int, default=RetrievalConfig().reranker_max_length)
ap.add_argument("--sizes", default="30,50,100")
args = ap.parse_args()

torch.set_num_threads(os.cpu_count() or 1)
print(f"{args.model} | max_length {args.max_length} | device: {'GPU' if torch.cuda.is_available() else 'CPU'} | "
      f"threads: {torch.get_num_threads()} | cpus: {os.cpu_count()}")
texts = load_chunks()["text"].tolist()
rr = Reranker(args.model, max_length=args.max_length, fp16=False)
rr.score([("warm up", texts[0])])
q = "Which papers run experiments on HotpotQA, and what does each of them propose?"
for n in map(int, args.sizes.split(",")):
    pairs = [(q, t) for t in texts[:n]]
    t0 = time.time()
    rr.score(pairs, batch_size=16)
    print(f"rerank {n:>3} pairs: {time.time() - t0:6.1f} s")
