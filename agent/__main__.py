"""python -m agent "question" --paper-id ID  — run the research agent once and print a cited answer."""
import argparse

from .graph import ResearchAgent

ap = argparse.ArgumentParser(description="Answer one question with the research agent.")
ap.add_argument("question")
ap.add_argument("--paper-id", default=None, help="search only this paper (default: whole corpus)")
a = ap.parse_args()

out = ResearchAgent().run(a.question, a.paper_id)
print("Sub-questions:")
for s in out["sub_questions"]:
    print("  -", s)
print("\nEvidence:")
for e in out["evidence"]:
    print(f"  [{e['label']}] {e['chunk_id']}  ({e['section']})  {e['text'][:110]}")
print("\nAnswer:\n" + out["answer"])
print(f"\nCited: {out['cited_chunk_ids']}  |  tokens in/out: {out['tokens']['prompt']}/{out['tokens']['completion']}"
      f"  |  {out['secs']} s")
