"""eval/token_report.py on fake summary files: per-question tokens, run totals and tokens per correct answer."""
import json

from eval.token_report import markdown, rows


def test_token_report(tmp_path):
    (tmp_path / "agent_eval_test_n100_v2.json").write_text(json.dumps(
        {"n": 100, "summary": {"avg_prompt_tokens": 2800, "avg_completion_tokens": 200, "avg_secs": 60}}))
    (tmp_path / "agent_eval_test_n100_v2_judge-gpt-oss-20b.json").write_text(json.dumps(
        {"summary": {"avg_judge_prompt_tokens": 1000, "avg_judge_completion_tokens": 200, "answer_correctness": 0.75}}))
    (tmp_path / "cross_paper_agent_v2.json").write_text(json.dumps(
        {"n": 19, "summary": {"all": {"avg_prompt_tokens": 4000, "avg_completion_tokens": 500, "avg_secs": 30}}}))
    rs = rows(tmp_path)
    assert [r["run"] for r in rs] == ["one-paper agent (frozen, Day 7 v2)", "cross-paper agent v2"]
    test, cross = rs
    assert test["per_q"] == 3000 and test["run_total"] == 300000 and test["per_correct"] == 4000
    assert test["judge_per_q"] == 1200 and test["secs"] is None          # test latency is not comparable
    assert cross["quality"] is None and cross["per_correct"] is None and cross["secs"] == 30
    assert "| one-paper agent (frozen, Day 7 v2) | test, n=100 | 2 (decompose, answer) | 3,000 (2,800 + 200)" in markdown(rs)
