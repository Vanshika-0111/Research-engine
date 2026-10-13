"""Turn QASPER papers into section-aware chunks and map QASPER's gold evidence to chunk IDs.

Chunking rules:
    * Chunks never cross a section boundary.
    * Whole paragraphs are packed into a chunk until it would exceed --max-tokens.
    * A single paragraph longer than --max-tokens is split into overlapping word windows.
    * Chunks shorter than --min-tokens are merged into a neighbour in the same section.
    * Acknowledgment sections are dropped (they never answer research questions).
    * QASPER placeholders (BIBREF12, INLINEFORM0, TABREF3, FIGREF1) become readable tokens.
    * The section name is prefixed to the chunk text ("[Results] ...").
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

PLACEHOLDERS = [
    (re.compile(r"\b(Tables?|Tab\.|Figures?|Fig\.|Sections?|Sec\.)\s+(?:TABREF|FIGREF|SECREF)\d+"), r"\1"),
    (re.compile(r"BIBREF\d+"), "[ref]"),
    (re.compile(r"(?:INLINE|DISPLAY)FORM\d+"), "[math]"),
    (re.compile(r"TABREF\d+"), "Table"),
    (re.compile(r"FIGREF\d+"), "Figure"),
    (re.compile(r"SECREF\d+"), "Section"),
]

SECTION_TYPES = [
    ("abstract", ["abstract"]),
    ("acknowledgments", ["acknowledg"]),
    ("related_work", ["related", "background", "previous work", "prior work", "literature"]),
    ("introduction", ["introduction", "motivation"]),
    ("conclusion", ["conclusion", "future work", "summary"]),
    ("results", ["result", "analysis", "discussion", "error", "ablation", "findings"]),
    ("experiments", ["experiment", "evaluation", "setup", "dataset", "data", "baseline", "training"]),
    ("method", ["method", "model", "approach", "architecture", "framework", "system", "proposed"]),
]


def n_tokens(text: str) -> int:
    return len(text.split())


def norm(text: str) -> str:
    return " ".join(text.split())


def clean(text: str) -> str:
    for pattern, repl in PLACEHOLDERS:
        text = pattern.sub(repl, text)
    return text


def section_type(name: str) -> str:
    top = name.split(":::")[0].strip().lower()
    for label, keys in SECTION_TYPES:
        if any(k in top for k in keys):
            return label
    return "other"


def split_long_paragraph(text: str, max_tokens: int, overlap: int) -> list[str]:
    words = text.split()
    step = max(1, max_tokens - overlap)
    return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), step)
            if words[i:i + max_tokens]]


def pack_section(paras, max_tokens, min_tokens, overlap):
    groups, buf, src = [], [], []

    def flush():
        nonlocal buf, src
        if buf:
            groups.append((buf, src))
        buf, src = [], []

    for p in paras:
        if n_tokens(p) > max_tokens:
            flush()
            for piece in split_long_paragraph(p, max_tokens, overlap):
                groups.append(([piece], [p]))
            continue
        if buf and n_tokens(" ".join(buf)) + n_tokens(p) > max_tokens:
            flush()
        buf.append(p)
        src.append(p)
    flush()

    merged = []
    for pieces, srcs in groups:
        if merged and n_tokens(" ".join(pieces)) < min_tokens:
            merged[-1] = (merged[-1][0] + pieces, merged[-1][1] + srcs)
        elif merged and n_tokens(" ".join(merged[-1][0])) < min_tokens:
            merged[-1] = (merged[-1][0] + pieces, merged[-1][1] + srcs)
        else:
            merged.append((pieces, srcs))
    return merged


def chunk_paper(paper_id, paper, max_tokens, min_tokens, overlap):
    sections = [{"section_name": "Abstract", "paragraphs": [paper.get("abstract", "")]}]
    sections += paper.get("full_text", []) or []

    chunks, para_to_chunks, idx = [], {}, 0
    for sec_pos, sec in enumerate(sections):
        name = (sec.get("section_name") or "Untitled").strip()
        stype = section_type(name)
        if stype == "acknowledgments":
            continue
        paras = [p for p in (sec.get("paragraphs") or []) if p and p.strip()]
        for pieces, srcs in pack_section(paras, max_tokens, min_tokens, overlap):
            cid = f"{paper_id}::{idx:03d}"
            body = clean("\n".join(pieces))
            chunks.append({
                "chunk_id": cid, "paper_id": paper_id, "title": paper.get("title", ""),
                "section": name, "section_type": stype, "section_pos": sec_pos,
                "chunk_pos": idx, "n_words": n_tokens(body), "text": f"[{name}] {body}",
            })
            for p in dict.fromkeys(srcs):
                para_to_chunks.setdefault(norm(p), []).append(cid)
            idx += 1
    return chunks, para_to_chunks


def gold_from_qas(paper_id, paper, para_to_chunks, split):
    records, stats = [], Counter()
    for qa in paper.get("qas", []) or []:
        gold, answers, n_annot = set(), [], 0
        for ann in qa.get("answers", []) or []:
            a = ann.get("answer", {})
            if a.get("unanswerable"):
                stats["unanswerable_annotation"] += 1
                continue
            n_annot += 1
            for ev in a.get("evidence", []) or []:
                if ev.startswith("FLOAT SELECTED"):
                    stats["evidence_is_table_or_figure"] += 1
                    continue
                ids = para_to_chunks.get(norm(ev))
                if ids:
                    gold.update(ids)
                    stats["evidence_matched"] += 1
                else:
                    stats["evidence_unmatched"] += 1
            ans_text = (a.get("free_form_answer") or "; ".join(a.get("extractive_spans") or [])
                        or ({True: "Yes", False: "No"}.get(a.get("yes_no")) or ""))
            if ans_text:
                answers.append(ans_text)
        if not gold:
            stats["question_dropped_no_gold"] += 1
            continue
        stats["question_kept"] += 1
        records.append({
            "question_id": qa.get("question_id"), "question": qa.get("question"),
            "paper_id": paper_id, "paper_title": paper.get("title", ""), "split": split,
            "type": "single_paper", "gold_chunk_ids": sorted(gold),
            "reference_answers": answers, "n_annotators": n_annot,
        })
    return records, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--min-tokens", type=int, default=40)
    ap.add_argument("--overlap", type=int, default=50)
    args = ap.parse_args()

    files = sorted(RAW_DIR.glob("qasper-*.json"))
    if not files:
        raise SystemExit("No qasper-*.json in data/raw. Run ingest/download_qasper.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks, papers, questions, stats, seen = [], [], [], Counter(), set()
    for f in files:
        split = f.stem.split("-")[1]
        data = json.loads(f.read_text(encoding="utf-8"))
        for pid, paper in data.items():
            if pid in seen:
                continue
            seen.add(pid)
            chunks, p2c = chunk_paper(pid, paper, args.max_tokens, args.min_tokens, args.overlap)
            recs, s = gold_from_qas(pid, paper, p2c, split)
            all_chunks += chunks
            questions += recs
            stats += s
            papers.append({"paper_id": pid, "title": paper.get("title", ""),
                           "abstract": paper.get("abstract", ""), "split": split,
                           "n_chunks": len(chunks)})

    chunks_df = pd.DataFrame(all_chunks)
    pd.DataFrame(papers).to_parquet(OUT_DIR / "papers.parquet", index=False)
    chunks_df.to_parquet(OUT_DIR / "chunks.parquet", index=False)
    with open(OUT_DIR / "qasper_questions.jsonl", "w", encoding="utf-8") as fh:
        for r in questions:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Papers:    {len(papers)}")
    print(f"Chunks:    {len(chunks_df)}")
    print(chunks_df.n_words.describe().round(0).to_string())
    print(f"Chunks under {args.min_tokens} words: {(chunks_df.n_words < args.min_tokens).sum()}")
    print("\nSection types:\n" + chunks_df.section_type.value_counts().to_string())
    print(f"\nQuestions with gold evidence: {len(questions)}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
