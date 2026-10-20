"""Create (or update) the Hugging Face Space for the demo app in ONE commit, and set its secrets.

Uploads only what the app needs: agent/, retrieval/, app/ (code + examples.json), the chunks parquet and the cached
bge-small embeddings, plus the Space README (front matter) and requirements. Run from the project root in Colab:
    python -m app.deploy_space                       # -> https://huggingface.co/spaces/<you>/research-evidence-engine
    python -m app.deploy_space --reranker cross-encoder/ms-marco-MiniLM-L-6-v2 --reranker-dev-r5 0.7xx
        # the free CPU Space needs a small reranker (v2-m3: ~100 s per question on 2 CPU cores); give its measured
        # dev Recall@5 (eval.run_retrieval_eval --split dev --reranker ...) so the app states it next to the demo
Needs HF_TOKEN (a Hugging Face token with WRITE access) and GROQ_API_KEY / GEMINI_API_KEY in the environment;
the API keys are stored as Space SECRETS (never in the files).
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval.config import CACHE_DIR, CHUNKS_PATH, ROOT, RetrievalConfig  # noqa: E402

CODE_DIRS = ("agent", "retrieval", "app")


def files_to_upload():
    """[(local path, path in the Space repo)]"""
    out = []
    for d in CODE_DIRS:
        for f in sorted((ROOT / d).glob("*.py")):
            out.append((f, f"{d}/{f.name}"))
    ex = ROOT / "app" / "examples.json"
    if not ex.exists():
        raise SystemExit("app/examples.json is missing: run `python -m app.build_examples` first.")
    out.append((ex, "app/examples.json"))
    out.append((ROOT / "app" / "requirements-space.txt", "requirements.txt"))
    out.append((ROOT / "app" / "SPACE_README.md", "README.md"))
    out.append((CHUNKS_PATH, "data/processed/chunks.parquet"))
    import pandas as pd
    n = len(pd.read_parquet(CHUNKS_PATH, columns=["chunk_id"]))
    emb = Path(CACHE_DIR) / f"emb_{RetrievalConfig().dense_model.replace('/', '_')}_{n}.npy"
    if not emb.exists():
        raise SystemExit(f"{emb} is missing (it is built by the retrieval pipeline on first use).")
    out.append((emb, f"cache/{emb.name}"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default=None, help="<user>/<name>; default <your user>/research-evidence-engine")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--cross-rerank-top", type=int, default=100)
    ap.add_argument("--max-per-day", type=int, default=150)
    ap.add_argument("--reranker", default=None, help="smaller cross-encoder for the CPU Space (default: v2-m3)")
    ap.add_argument("--reranker-max-length", type=int, default=None)
    ap.add_argument("--reranker-dev-r5", default=None, help="that reranker's measured dev evidence Recall@5")
    args = ap.parse_args()

    from huggingface_hub import CommitOperationAdd, HfApi
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN (a Hugging Face token with write access).")
    api = HfApi(token=token)
    repo_id = args.space or f"{api.whoami()['name']}/research-evidence-engine"
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", private=args.private, exist_ok=True)

    files = files_to_upload()
    for local, remote in files:
        print(f"  {remote:<55} {Path(local).stat().st_size / 1e6:8.1f} MB")
    api.create_commit(repo_id=repo_id, repo_type="space", commit_message="Deploy demo app",
                      operations=[CommitOperationAdd(path_in_repo=r, path_or_fileobj=str(l)) for l, r in files])

    for key in ("GROQ_API_KEY", "GEMINI_API_KEY"):
        if os.environ.get(key):
            api.add_space_secret(repo_id, key, os.environ[key].strip())
            print(f"  secret {key} set")
    api.add_space_variable(repo_id, "APP_CROSS_RERANK_TOP", str(args.cross_rerank_top))
    api.add_space_variable(repo_id, "APP_MAX_PER_DAY", str(args.max_per_day))
    for name, val in (("APP_RERANKER_MODEL", args.reranker), ("APP_RERANKER_MAX_LENGTH", args.reranker_max_length),
                      ("APP_RERANKER_DEV_R5", args.reranker_dev_r5)):
        if val:
            api.add_space_variable(repo_id, name, str(val))
            print(f"  variable {name} = {val}")
    print(f"\nDeployed. The Space builds for a few minutes, then runs at https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
