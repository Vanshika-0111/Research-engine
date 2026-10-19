"""Run bookkeeping: resumable .jsonl files, quota detection, per-paper scoring, Groq client retries."""
from types import SimpleNamespace

from agent.llm import GroqLLM, is_daily_limit, is_overloaded
from eval.run_agent_eval import load_rows, qkey
from eval.run_cross_paper_agent_eval import paper_scores


def test_load_rows_skips_a_line_cut_off_mid_write(tmp_path):
    p = tmp_path / "run.jsonl"
    p.write_text('{"id": 1}\n{"id": 2}\n{"id": 3, "ans', encoding="utf-8")
    assert [r["id"] for r in load_rows(p)] == [1, 2]


def test_qkey():
    assert qkey({"question_id": "abc", "paper_id": "p", "question": "q"}) == "abc"
    assert qkey({"question_id": None, "paper_id": "p", "question": "q"}) == "p|q"


def test_daily_limit_vs_overload():
    groq_tpd = "Error code: 429 - rate_limit_exceeded: Limit 200000, Used 199500 tokens per day (TPD)"
    assert is_daily_limit(Exception(groq_tpd))
    assert not is_daily_limit(Exception("rate_limit_exceeded: tokens per minute (TPM)"))
    assert is_daily_limit(Exception("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel"))
    assert is_overloaded(Exception("503 UNAVAILABLE: high demand"))


def test_paper_scores():
    row = {"cited_papers": ["p1", "p3", "x"], "papers": [{"paper_id": p} for p in ("p1", "p2", "p3", "x")]}
    q = {"required_papers": ["p1", "p2", "p4"], "acceptable_papers": ["p3"]}
    s = paper_scores(row, q)
    assert s["retrieved_paper_recall"] == 2 / 3 and s["cited_paper_recall"] == 1 / 3
    assert s["cited_paper_precision"] == 2 / 3 and s["n_cited_papers"] == 3
    assert paper_scores({"cited_papers": [], "papers": []}, q)["cited_paper_precision"] is None


def _resp(text, finish="stop"):
    return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=5, completion_tokens=7),
                           choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)])


class FakeGroqClient:
    def __init__(self, *outcomes):
        self.outcomes, self.kwargs = list(outcomes), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.kwargs.append(kw)
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o


def test_groq_retries_empty_reply_cut_off_by_reasoning_budget():
    client = FakeGroqClient(_resp("", finish="length"), _resp("answer"))
    text, usage = GroqLLM("openai/gpt-oss-120b", client=client).chat([{"role": "user", "content": "q"}], max_tokens=100)
    assert text == "answer" and client.kwargs[1]["max_completion_tokens"] == 200
    assert usage["prompt_tokens"] == 10 and usage["completion_tokens"] == 14     # both attempts counted
    assert client.kwargs[0]["include_reasoning"] is False and client.kwargs[0]["reasoning_effort"] == "low"


def test_groq_falls_back_when_json_mode_is_rejected():
    client = FakeGroqClient(Exception("response_format json_object not supported"), _resp("{}"))
    llm = GroqLLM("some/model", client=client)
    assert llm.chat([{"role": "user", "content": "q"}], json_mode=True)[0] == "{}"
    assert "response_format" in client.kwargs[0] and "response_format" not in client.kwargs[1]
    assert llm.json_ok is False
