"""Download and extract the QASPER dataset (v0.3) into data/raw/.

Usage:
    python ingest/download_qasper.py

If the download fails (URLs can move), download the two .tgz files manually
from https://allenai.org/data/qasper and put them in data/raw/, then rerun:
the script skips downloading files that already exist and just extracts.
"""
from pathlib import Path
import tarfile

import requests
from tqdm import tqdm

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
URLS = [
    "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
    "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz",
]


def is_complete(archive: Path) -> bool:
    """True if the .tgz can be read to the end (catches half-finished downloads)."""
    try:
        with tarfile.open(archive) as tar:
            tar.getmembers()
        return True
    except (tarfile.TarError, EOFError, OSError):
        return False


def download(url: str, dest: Path) -> None:
    if dest.exists():
        if is_complete(dest):
            print(f"Already have {dest.name}, skipping download")
            return
        print(f"{dest.name} is incomplete or corrupt, downloading again")
        dest.unlink()
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for block in r.iter_content(chunk_size=1 << 20):
                f.write(block)
                bar.update(len(block))
    if total and tmp.stat().st_size != total:
        tmp.unlink()
        raise SystemExit(f"Download of {dest.name} was cut off. Run the script again.")
    tmp.rename(dest)  # only a finished download gets the real name


def extract(archive: Path, out_dir: Path) -> None:
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            # Only plain files with safe relative paths
            if not member.isfile() or member.name.startswith("/") or ".." in Path(member.name).parts:
                continue
            target = out_dir / Path(member.name).name
            if target.exists():
                continue
            with tar.extractfile(member) as src, open(target, "wb") as dst:
                dst.write(src.read())


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for url in URLS:
        archive = RAW_DIR / url.rsplit("/", 1)[-1]
        download(url, archive)
        extract(archive, RAW_DIR)
    found = sorted(p.name for p in RAW_DIR.glob("qasper-*.json"))
    print("JSON files in data/raw:", found)
    if not found:
        raise SystemExit("No qasper-*.json files found after extraction.")


if __name__ == "__main__":
    main()
