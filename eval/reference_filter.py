"""Reference-quality filter: drop questions whose QASPER reference answers can't be used to grade an answer.

Why: in the n=50 dev review about half of the "wrong" answers were dataset artifacts, where the reference
answer points somewhere instead of answering (e.g. "BIBREF9 ; BIBREF8", "the same datasets as BIBREF7",
"Answer with content missing: ..."). A correct agent answer can't match those, so they lower answer
correctness for reasons that have nothing to do with the agent.

The rule is mechanical and was fixed on 2026-10-15, BEFORE the test run (no hand-picking of questions):
  A reference answer is UNUSABLE if
    (a) it starts with "Answer with content missing" (QASPER's own flag for answers whose content was lost), or
    (b) it contains a BIBREF / FIGREF / TABREF / SECREF / EQREF marker AND no content word is left after
        removing the markers and a fixed list of filler words (the, same, as, see, in, table, figure, dataset...),
        i.e. it only points to a citation, figure or table: "BIBREF9 ; BIBREF8", "the same datasets as BIBREF7",
        "Table TABREF3". A short list of names with citations ("Farasa BIBREF31; MADAMIRA BIBREF29") is kept.
  A question is EXCLUDED if ALL its reference answers are unusable. Short real answers ("Yes", "1913 signals")
  have no marker, so they are kept. The filler list below is part of the rule and is not to be edited after
  the test run.
Every eval reports both the full numbers and the filtered numbers.

Known limit: a reference that describes a figure in words ("Figures FIGREF1 and FIGREF1 contain a summary
of ...") is kept, because it has content words. That is the price of a rule that can't be tuned to the results.

Run on any saved results file (agent eval or judge) to see what is excluded and the full vs filtered metrics:
    python -m eval.reference_filter results/agent_eval_dev_n50_v2.jsonl
    python -m eval.reference_filter results/agent_eval_dev_n50_v2_judge-gpt-oss-20b.jsonl
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_MARKER = re.compile(r"\b(?:BIBREF|FIGREF|TABREF|SECREF|EQREF)\d+\b")
_WORD = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
FILLER = frozenset("""a an the and or of in on at to by for from with as same similar previous prior
see shown given described reported listed used using use uses is are was were be this that these those their its
they we our it paper papers work works table tables figure figures fig section sections appendix
dataset datasets data corpus corpora model models method methods result results e g i etc""".split())
RULE = ("exclude a question if ALL its references either start with 'Answer with content missing' or contain a "
        "BIBREF/FIGREF/TABREF/SECREF/EQREF marker and no content word besides markers and filler words")


def reference_problem(ref):
    """Why a single reference answer is unusable, or None if it is fine."""
    ref = (ref or "").strip()
    if ref.lower().startswith("answer with content missing"):
        return "content missing"
    if _MARKER.search(ref) and not [w for w in _WORD.findall(_MARKER.sub(" ", ref)) if w.lower() not in FILLER]:
        return "only a citation/figure/table marker"
    return None


def check_references(refs):
    """(keep, reason) for a question's list of reference answers."""
    refs = [r for r in (refs or []) if str(r).strip()]
    if not refs:
        return True, None   # nothing to judge against: leave it to the metrics (don't silently drop)
    problems = [reference_problem(r) for r in refs]
    if all(problems):
        return False, "; ".join(sorted(set(problems)))
    return True, None


def split_rows(rows):
    """-> (kept rows, excluded rows); each excluded row gets 'reference_filter_reason'."""
    kept, excluded = [], []
    for r in rows:
        keep, reason = check_references(r.get("reference_answers"))
        if keep:
            kept.append(r)
        else:
            excluded.append({**r, "reference_filter_reason": reason})
    return kept, excluded


def excluded_list(rows):
    """Short record of excluded questions (position in the file, question, reason) for summary JSONs."""
    out = []
    for i, r in enumerate(rows, 1):
        keep, reason = check_references(r.get("reference_answers"))
        if not keep:
            out.append({"q": i, "question": r["question"], "reason": reason})
    return out


def main():
    import argparse
    from eval.run_agent_eval import load_rows
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="an agent eval .jsonl or a judge .jsonl")
    args = ap.parse_args()
    rows = load_rows(Path(args.jsonl))
    kept, excluded = split_rows(rows)
    print(f"Rule: {RULE}")
    print(f"{len(rows)} questions -> {len(kept)} kept, {len(excluded)} excluded")
    for e in excluded_list(rows):
        print(f"  Q{e['q']}: {e['question'][:70]}  ({e['reason']})")

    if rows and "judge" in rows[0]:
        from eval.judge_agent_eval import summarize_judged as summarize
    else:
        from eval.run_agent_eval import summarize_rows as summarize
    full, filt = summarize(rows), summarize(kept) if kept else {}
    print(f"\n| metric | full (n={len(rows)}) | filtered (n={len(kept)}) |\n|---|---|---|")
    for k, v in full.items():
        print(f"| {k} | {v} | {filt.get(k)} |")


if __name__ == "__main__":
    main()
