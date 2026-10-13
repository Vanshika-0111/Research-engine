"""python -m retrieval "question" [--paper-id ID] [-k 5]  — print the top evidence chunks."""
import argparse

from .pipeline import retrieve

ap = argparse.ArgumentParser(description="Retrieve evidence chunks for one question.")
ap.add_argument("question")
ap.add_argument("--paper-id", default=None, help="search only this paper (default: whole corpus)")
ap.add_argument("-k", type=int, default=5)
a = ap.parse_args()
for h in retrieve(a.question, a.paper_id, a.k):
    print(f"{h['rank']:>2}  {h['score']:.3f}  {h['chunk_id']}  [{h['section_type']}]  {h['text'][:160]}")
