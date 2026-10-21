"""Build the static results explorer (a free Hugging Face *Static* Space) from SAVED results: no LLM calls.

Hugging Face now charges for Gradio/Docker Spaces (PRO), but static Spaces are free. This page shows what the
system actually produced on the evaluation sets, with the judge's verdicts, instead of live answers:
  * one paper   : all 100 held-out TEST answers of the frozen agent (Day 7 v2) + gpt-oss-20b judge verdicts,
                  QASPER gold passages marked, reference answers shown
  * multi-hop   : the 50 DEV answers of the multi-hop ablation (Day 14), follow-up passages marked
  * across papers: the 19 cross-paper answers (Day 13, v2 prompt), required / acceptable papers marked
Headline numbers are read from the summary JSONs, so the page cannot drift from the evaluation.

    python -m app.build_static                     # -> space_static/ (index.html, data.json, README.md)
    python -m app.deploy_static                    # uploads space_static/ as a free static Space
The passage texts come from data/processed/chunks.parquet, so run this where the data is (Colab).
"""
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

APP = Path(__file__).resolve().parent
OUT = APP.parent / "space_static"
_SECTION_PREFIX = re.compile(r"^\[[^\]]*\]\s*")


def _rows(path):
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def _key(r):
    return r.get("question_id") or f'{r.get("paper_id")}|{r["question"]}'


def paper_items(answers, judged, titles, prefix):
    """One item per within-paper answer, with judge verdicts mapped from labels [E#] to chunk IDs."""
    jby = {_key(j): j for j in judged}
    items, used = [], set()
    for i, r in enumerate(answers, 1):
        ev = r["evidence_chunk_ids"]
        j = jby.get(_key(r), {})
        verdict = j.get("judge", {})
        label_cid = {f"E{n}": c for n, c in enumerate(ev, 1)}
        item = {
            "id": f"{prefix}{i}", "q": r["question"], "paper_id": r["paper_id"],
            "title": titles.get(str(r["paper_id"]), ""), "answer": r["answer"],
            "refs": r.get("reference_answers", []), "subs": r.get("sub_questions", []),
            "ev": ev, "cited": r["cited_chunk_ids"], "gold": r.get("gold_chunk_ids", []),
            "declined": bool(r.get("declined")),
            "tokens": r["tokens"]["prompt"] + r["tokens"]["completion"],
            "judge": {"correct": verdict.get("correct"), "faithful": verdict.get("faithful"),
                      "cites": {label_cid[l]: v for l, v in (verdict.get("citations") or {}).items() if l in label_cid},
                      "note": verdict.get("note", ""), "unsupported": verdict.get("unsupported", [])},
        }
        if "hops" in r:
            item.update({"hops": r.get("hops", 0), "ev_hops": r.get("evidence_hops", []),
                         "missing": r.get("missing", ""), "follow_up": r.get("follow_up_query", "")})
        items.append(item)
        used.update(ev)
    return items, used


def corpus_items(answers, judged):
    jby = {j["id"]: j for j in judged}
    items = []
    for r in answers:
        req, ok = set(r["required_papers"]), set(r["acceptable_papers"])
        verdict = jby.get(r["id"], {}).get("judge", {})
        items.append({
            "id": r["id"], "type": r["type"], "q": r["question"], "answer": r["answer"],
            "papers": [{"paper_id": p["paper_id"], "title": p["title"], "rank": p["rank"],
                        "role": "required" if p["paper_id"] in req else "acceptable" if p["paper_id"] in ok else ""}
                       for p in r["papers"]],
            "n_required": len(req),
            "ev": [{"l": e["label"], "pid": e["paper_id"], "pl": e.get("paper_label", ""), "sec": e["section"],
                    "text": e["text"]} for e in r["evidence"]],
            "cited_papers": r["cited_papers"], "cited": r["cited_labels"], "declined": bool(r.get("declined")),
            "metrics": r["metrics"], "tokens": r["tokens"]["prompt"] + r["tokens"]["completion"],
            "judge": {"faithful": verdict.get("faithful"), "cites": verdict.get("citations") or {},
                      "per_paper": verdict.get("per_paper") or {}, "note": verdict.get("note", ""),
                      "unsupported": verdict.get("unsupported", [])},
        })
    return items


def summary(res):
    """Headline numbers, read from the saved summary JSONs."""
    rt = _json(res / "retrieval_eval_test_paper.json").get("results", {})
    agent = _json(res / "agent_eval_test_n100_v2.json").get("summary", {})
    judge = _json(res / "agent_eval_test_n100_v2_judge-gpt-oss-20b.json").get("summary", {})
    xp = _json(res / "cross_paper_agent_v2.json").get("summary", {}).get("all", {})
    xpj = _json(res / "cross_paper_agent_v2_judge-gpt-oss-20b.json").get("summary", {}).get("all", {})
    xr = _json(res / "cross_paper_retrieval.json")
    hop = _json(res / "agent_eval_dev_n50_v3hop.json").get("summary", {})
    hopj = _json(res / "agent_eval_dev_n50_v3hop_judge-gpt-oss-20b.json").get("summary", {})
    v2d = _json(res / "agent_eval_dev_n50_v2.json").get("summary", {})
    v2dj = _json(res / "agent_eval_dev_n50_v2_judge-gpt-oss-20b.json").get("summary", {})
    return {
        "retrieval": {k: rt.get(k, {}).get("ev_recall@5") for k in ("bm25", "dense", "hybrid_rrf", "hybrid+rerank", "final")},
        "retrieval_final": rt.get("final", {}),
        "agent": {k: agent.get(k) for k in ("citation_precision", "evidence_recall", "declined_rate",
                                             "avg_prompt_tokens", "avg_completion_tokens")},
        "judge": {k: judge.get(k) for k in ("judged_citation_precision", "strict_citation_precision", "faithfulness",
                                             "answer_correctness", "answer_correctness_answered_only")},
        "cross": {**{k: xp.get(k) for k in ("retrieved_paper_recall", "cited_paper_recall", "cited_paper_precision")},
                  **{k: xpj.get(k) for k in ("judged_citation_precision", "faithfulness")}},
        "cross_retrieval": {k: (xr.get("results", {}).get(k) or {}).get("paper_recall@10") for k in ("bm25", "dense", "hybrid", "hybrid+rerank")},
        "multihop": {"v2": {**v2d, **v2dj}, "v3hop": {**hop, **hopj}},
    }


def build(chunks, res, out=OUT):
    """chunks: DataFrame with chunk_id, paper_id, title, section, text. Writes out/{index.html,data.json,README.md}."""
    res = Path(res)
    titles = dict(zip(chunks["paper_id"].astype(str), chunks["title"].astype(str)))
    test, used1 = paper_items(_rows(res / "agent_eval_test_n100_v2.jsonl"),
                              _rows(res / "agent_eval_test_n100_v2_judge-gpt-oss-20b.jsonl"), titles, "T")
    hop, used2 = paper_items(_rows(res / "agent_eval_dev_n50_v3hop.jsonl"),
                             _rows(res / "agent_eval_dev_n50_v3hop_judge-gpt-oss-20b.jsonl"), titles, "H")
    cross = corpus_items(_rows(res / "cross_paper_agent_v2.jsonl"),
                         _rows(res / "cross_paper_agent_v2_judge-gpt-oss-20b.jsonl"))
    used = used1 | used2
    sub = chunks[chunks["chunk_id"].astype(str).isin(used)]
    table = {str(c): [s, _SECTION_PREFIX.sub("", t, count=1)]
             for c, s, t in zip(sub["chunk_id"], sub["section"], sub["text"])}
    data = {"summary": summary(res), "paper": test, "multihop": hop, "corpus": cross, "chunks": table}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    shutil.copy(APP / "static" / "index.html", out / "index.html")
    shutil.copy(APP / "STATIC_README.md", out / "README.md")
    return data


def main():
    from retrieval.config import RESULTS_DIR
    from retrieval.data import load_chunks
    data = build(load_chunks(), RESULTS_DIR)
    size = (OUT / "data.json").stat().st_size / 1e6
    print(f"{len(data['paper'])} test answers, {len(data['multihop'])} multi-hop answers, {len(data['corpus'])} cross-paper "
          f"answers, {len(data['chunks'])} passages -> {OUT} (data.json {size:.1f} MB)")
    missing = sum(c not in data["chunks"] for it in data["paper"] + data["multihop"] for c in it["ev"])
    print("passages missing from chunks.parquet:", missing)


if __name__ == "__main__":
    main()
