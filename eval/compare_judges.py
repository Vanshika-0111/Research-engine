"""Agreement between two LLM judges that graded the SAME saved agent answers.

Why: one LLM judge is a measuring instrument with its own bias. If a second judge from a different model
family (e.g. Gemini vs gpt-oss) gives the same verdicts, the judged metrics are more trustworthy; where they
disagree, those are the answers to read by hand.

Compared, question by question (only where both judges gave a verdict):
    correct     answer agrees with the references          (all questions; declines are scored by rule, so
                                                            they agree trivially and are left out here)
    faithful    all claims supported by the cited passages (answered questions)
    citations   each cited [E#] supports its sentence      (one item per cited label)
For each: exact agreement on yes/partly/no, Cohen's kappa (agreement corrected for chance; 1 = perfect,
0 = chance level; above ~0.6 is usually called substantial), and "lenient" agreement where partly counts
as yes-ish (yes/partly vs no). Also the two judges' summary metrics side by side, full and reference-filtered.

Run from the project root:
    python -m eval.compare_judges results/agent_eval_dev_n20_v2_judge-gpt-oss-20b.jsonl \
                                  results/agent_eval_dev_n20_v2_judge-gemini-38-flash.jsonl
Writes results/judge_agreement_<agent stem>_<judgeA>_vs_<judgeB>.json and .md (with every disagreement).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.judge_agent_eval import summarize_judged  # noqa: E402
from eval.reference_filter import split_rows  # noqa: E402
from eval.run_agent_eval import load_rows, qkey  # noqa: E402

LABELS = ("yes", "partly", "no")


def cohen_kappa(pairs):
    """Cohen's kappa for a list of (a, b) label pairs; None if undefined."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[l] * cb[l] for l in set(ca) | set(cb)) / (n * n)
    if pe == 1:
        return 1.0 if po == 1 else None
    return round((po - pe) / (1 - pe), 3)


def agreement(pairs):
    pairs = [(a, b) for a, b in pairs if a in LABELS and b in LABELS]
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    lenient = [(a != "no", b != "no") for a, b in pairs]
    confusion = {a: {b: sum(1 for x, y in pairs if x == a and y == b) for b in LABELS} for a in LABELS}
    return {"n": n,
            "exact_agreement": round(sum(a == b for a, b in pairs) / n, 3),
            "kappa": cohen_kappa(pairs),
            "lenient_agreement": round(sum(a == b for a, b in lenient) / n, 3),
            "confusion_rows_A_cols_B": confusion}


def short(model):
    return model.split("/")[-1].replace(".", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("judge_a", help="judge .jsonl (e.g. ..._judge-gpt-oss-20b.jsonl)")
    ap.add_argument("judge_b", help="judge .jsonl from another model on the same agent .jsonl")
    args = ap.parse_args()

    A, B = load_rows(Path(args.judge_a)), load_rows(Path(args.judge_b))
    ma, mb = A[0]["judge_model"], B[0]["judge_model"]
    bkey = {qkey(r): r for r in B}
    both = [(a, bkey[qkey(a)]) for a in A if qkey(a) in bkey]
    if not both:
        raise SystemExit("No questions in common: were both judges run on the same agent .jsonl?")
    # same answers? (a judge comparison only means something if both judged identical text)
    differ = [a["question"][:60] for a, b in both if a["answer"] != b["answer"]]
    if differ:
        raise SystemExit(f"{len(differ)} answers differ between the files (e.g. {differ[0]!r}); "
                         "the judges must grade the same agent .jsonl.")

    pos = {qkey(a): i for i, a in enumerate(A, 1)}
    answered = [(a, b) for a, b in both if not a["declined"]]
    correct = agreement([(a["judge"]["correct"], b["judge"]["correct"]) for a, b in answered])
    faithful = agreement([(a["judge"]["faithful"], b["judge"]["faithful"]) for a, b in answered])
    cite_pairs = [(a["judge"]["citations"].get(l), b["judge"]["citations"].get(l))
                  for a, b in answered for l in a["cited_labels"]]
    citations = agreement(cite_pairs)

    sums = {}
    for name, rows in ((ma, [a for a, _ in both]), (mb, [b for _, b in both])):
        kept, _ = split_rows(rows)
        sums[name] = {"full": summarize_judged(rows), "filtered": summarize_judged(kept), "n_kept": len(kept)}

    disagreements = []
    for a, b in answered:
        ja, jb = a["judge"], b["judge"]
        cd = [l for l in a["cited_labels"] if ja["citations"].get(l) != jb["citations"].get(l)]
        if ja["correct"] != jb["correct"] or ja["faithful"] != jb["faithful"] or cd:
            disagreements.append({
                "q": pos[qkey(a)], "question": a["question"],
                "correct": [ja["correct"], jb["correct"]], "faithful": [ja["faithful"], jb["faithful"]],
                "citations": {l: [ja["citations"].get(l), jb["citations"].get(l)] for l in cd},
                "note_a": ja.get("note", ""), "note_b": jb.get("note", "")})

    src = Path(args.judge_a)
    agent_stem = src.stem.split("_judge-")[0]
    out = src.with_name(f"judge_agreement_{agent_stem}_{short(ma)}_vs_{short(mb)}")
    report = {"agent_results": agent_stem, "judge_a": ma, "judge_b": mb, "n_questions": len(both),
              "n_answered": len(answered),
              "agreement": {"correct": correct, "faithful": faithful, "citations": citations},
              "summaries": sums, "disagreements": disagreements}
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    lines = [f"# Judge agreement: {ma} (A) vs {mb} (B)", "",
             f"Agent results: `{agent_stem}`; {len(both)} questions, {len(answered)} answered.", "",
             "| verdict | items | exact agreement | Cohen's kappa | lenient agreement (yes/partly vs no) |",
             "|---|---|---|---|---|"]
    for name, ag in (("correct", correct), ("faithful", faithful), ("citations", citations)):
        lines.append(f"| {name} | {ag['n']} | {ag.get('exact_agreement')} | {ag.get('kappa')} | {ag.get('lenient_agreement')} |")
    keys = ["strict_citation_precision", "judged_citation_precision", "faithfulness",
            "answer_correctness", "answer_correctness_answered_only", "judge_errors"]
    lines += ["", "| metric | A full | B full | A filtered | B filtered |", "|---|---|---|---|---|"]
    for k in keys:
        lines.append(f"| {k} | {sums[ma]['full'][k]} | {sums[mb]['full'][k]} | "
                     f"{sums[ma]['filtered'][k]} | {sums[mb]['filtered'][k]} |")
    lines.append(f"| n | {len(both)} | {len(both)} | {sums[ma]['n_kept']} | {sums[mb]['n_kept']} |")
    lines += ["", f"## Disagreements ({len(disagreements)})", ""]
    for d in disagreements:
        cites = ", ".join(f"{l} {v[0]}/{v[1]}" for l, v in d["citations"].items())
        lines.append(f"- **Q{d['q']}** {d['question'][:90]}  \n"
                     f"  correct A/B = {d['correct'][0]}/{d['correct'][1]}, faithful = {d['faithful'][0]}/{d['faithful'][1]}"
                     + (f", citations: {cites}" if cites else "")
                     + f"  \n  A: {d['note_a']}  \n  B: {d['note_b']}")
    md = "\n".join(lines) + "\n"
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Saved {out.with_suffix('.json')} and {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
