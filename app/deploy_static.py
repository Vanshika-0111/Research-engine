"""Upload space_static/ (built by app.build_static) as a FREE Hugging Face Static Space, in one commit.

    python -m app.build_static
    python -m app.deploy_static          # -> https://huggingface.co/spaces/<you>/research-evidence-engine
Needs HF_TOKEN (a Hugging Face token with WRITE access). No API keys are uploaded: the page only shows saved results.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SRC = Path(__file__).resolve().parents[1] / "space_static"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default=None, help="<user>/<name>; default <your user>/research-evidence-engine")
    args = ap.parse_args()
    for f in ("index.html", "data.json", "README.md"):
        if not (SRC / f).exists():
            raise SystemExit(f"{SRC / f} is missing: run `python -m app.build_static` first.")
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN (a Hugging Face token with write access).")
    api = HfApi(token=token)
    repo_id = args.space or f"{api.whoami()['name']}/research-evidence-engine"
    api.create_repo(repo_id, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(folder_path=str(SRC), repo_id=repo_id, repo_type="space",
                      allow_patterns=["index.html", "data.json", "README.md"], commit_message="Results explorer")
    print(f"Deployed: https://huggingface.co/spaces/{repo_id}  (static: live within a minute)")


if __name__ == "__main__":
    main()
