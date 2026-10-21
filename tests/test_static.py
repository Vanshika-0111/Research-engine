"""The static results explorer builder (app/build_static.py) on tiny fake results: no models, no data files."""
import json

import pandas as pd

from app.build_static import build


def _w(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows))


def test_build_static(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    _w(res / "agent_eval_test_n100_v2.jsonl", [{
        "question_id": "q1", "paper_id": "p1", "question": "What data?", "answer": "SQuAD [E1].",
        "evidence_chunk_ids": ["p1::1", "p1::2"], "cited_chunk_ids": ["p1::1"], "gold_chunk_ids": ["p1::2"],
        "reference_answers": ["SQuAD"], "sub_questions": ["data"], "declined": False,
        "tokens": {"prompt": 100, "completion": 10}}])
    _w(res / "agent_eval_test_n100_v2_judge-gpt-oss-20b.jsonl", [{
        "question_id": "q1", "paper_id": "p1", "question": "What data?",
        "judge": {"citations": {"E1": "yes"}, "faithful": "yes", "correct": "partly", "note": "ok"}}])
    _w(res / "cross_paper_agent_v2.jsonl", [{
        "id": "XP01", "type": "dataset", "question": "Which papers?", "answer": "- A [E1]",
        "papers": [{"paper_id": "p1", "title": "A", "rank": 1}, {"paper_id": "p2", "title": "B", "rank": 2}],
        "evidence": [{"label": "E1", "paper_id": "p1", "paper_label": "P1", "section": "S", "text": "t"}],
        "cited_papers": ["p1"], "cited_labels": ["E1"], "declined": False, "required_papers": ["p1"],
        "acceptable_papers": ["p2"], "metrics": {"cited_paper_recall": 1.0}, "tokens": {"prompt": 5, "completion": 1}}])
    (res / "agent_eval_test_n100_v2_judge-gpt-oss-20b.json").write_text(json.dumps({"summary": {"faithfulness": 0.97}}))
    chunks = pd.DataFrame({"chunk_id": ["p1::1", "p1::2"], "paper_id": ["p1", "p1"], "title": ["Paper One"] * 2,
                           "section": ["Data", "Exp"], "text": ["[Data] We use SQuAD.", "[Exp] Results."]})
    d = build(chunks, res, out=tmp_path / "out")
    it = d["paper"][0]
    assert it["title"] == "Paper One" and it["judge"]["cites"] == {"p1::1": "yes"}     # label E1 -> chunk ID
    assert d["chunks"]["p1::1"] == ["Data", "We use SQuAD."]                         # section prefix stripped
    assert d["corpus"][0]["papers"][0]["role"] == "required" and d["corpus"][0]["papers"][1]["role"] == "acceptable"
    assert d["summary"]["judge"]["faithfulness"] == 0.97 and d["multihop"] == []
    assert {p.name for p in (tmp_path / "out").iterdir()} == {"index.html", "data.json", "README.md"}
