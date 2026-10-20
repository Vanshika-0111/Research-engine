"""The demo app without Gradio: HTML rendering, provider fallback, caching and quota guards of the engine."""
import json

import pytest

from agent.llm import FallbackLLM
from app.engine import DailyBudget, Engine
from app.render import answer_html, evidence_html, papers_html, trace_html
from conftest import FakeLLM
from fakes_app import FakeAppPipeline

EV = [{"label": "E1", "section": "Data", "text": "We use <SQuAD>."},
      {"label": "E2", "section": "Results", "text": "F1 is 90.", "hop": 1}]


def test_answer_html_links_citations_and_escapes():
    h = answer_html("They use **SQuAD** <b>x</b> [E1]. Bad [E9].", EV, "u1")
    assert '<a class="cite" href="#ev-u1-E1">[E1]</a>' in h
    assert "<b>SQuAD</b>" in h and "&lt;b&gt;x&lt;/b&gt;" in h       # model HTML is escaped, ** becomes bold
    assert 'class="cite bad"' in h                                    # E9 was never shown to the LLM


def test_answer_html_bullets():
    h = answer_html("Intro.\n- Paper A: x [E1]\n- Paper B: y [E2]\nDone.", EV, "u")
    assert h.count("<li>") == 2 and "<ul>" in h and h.index("Intro") < h.index("<ul>") < h.index("Done")


def test_evidence_html_marks_cited_and_hop():
    h = evidence_html(EV, ["E1"], "u")
    assert 'id="ev-u-E1" class="ev is-cited" open' in h and "found by follow-up search 1" in h
    assert "&lt;SQuAD&gt;" in h


def test_papers_and_trace():
    p = papers_html([{"paper_id": "1909.00001", "title": "Paper A", "rank": 1}], ["1909.00001"])
    assert "arxiv.org/abs/1909.00001" in p and "cited" in p
    t = trace_html({"sub_questions": ["q1"], "sufficient": False, "missing": "scores", "follow_up_query": "table",
                    "hop_log": [{"added_chunk_ids": ["x", "y"]}], "tokens": {"prompt": 1000, "completion": 50},
                    "secs": 3.2, "llm_calls": [{"model": "retrieval"}, {"model": "openai/gpt-oss-120b"}]}, "paper")
    assert "added 2 passage(s)" in t and "1,000 prompt" in t and "gpt-oss-120b" in t and "retrieval" not in t


class _Quota(Exception):
    pass


def test_fallback_llm_skips_exhausted_provider():
    class Dead:
        calls = 0

        def chat(self, *a, **k):
            Dead.calls += 1
            raise _Quota("rate_limit_exceeded: tokens per day (TPD)")
    fb = FallbackLLM([Dead(), FakeLLM("from backup", "again")])
    assert fb.chat([{"role": "user", "content": "q"}])[0] == "from backup"
    assert fb.chat([{"role": "user", "content": "q"}])[0] == "again"
    assert Dead.calls == 1 and len(fb.available) == 1                 # the exhausted provider is not retried


def test_fallback_llm_raises_other_errors():
    class Broken:
        def chat(self, *a, **k):
            raise ValueError("bad request")
    with pytest.raises(ValueError):
        FallbackLLM([Broken(), FakeLLM("x")]).chat([])


def test_daily_budget():
    b = DailyBudget(2)
    assert b.take() and b.take() and not b.take()


def _engine(tmp_path, llm, examples=None):
    p = tmp_path / "examples.json"
    p.write_text(json.dumps(examples or {"paper": [], "corpus": []}))
    e = Engine(pipeline=FakeAppPipeline(), llm=llm, examples_path=p)
    e.load()
    assert e.error is None, e.error
    return e


def test_engine_live_question_then_cache(tmp_path):
    llm = FakeLLM('{"sub_questions": []}', "They use SQuAD [E1].")
    e = _engine(tmp_path, llm)
    assert [p for _, p in e.papers()] == ["1909.00001", "1909.00002"]
    out, msg, live = e.ask_paper("What data?", "1909.00001", session_count=0)
    assert msg is None and live and out["cited_chunk_ids"] == ["a1"]
    out2, _, live2 = e.ask_paper("  what   DATA? ", "1909.00001", session_count=1)
    assert out2["cached"] and not live2 and len(llm.calls) == 2      # same question -> no new LLM call


def test_engine_cross_paper(tmp_path):
    e = _engine(tmp_path, FakeLLM("- Paper B: HotpotQA [E1]."))
    out, msg, live = e.ask_corpus("Which papers use HotpotQA?")
    assert live and out["cited_papers"] and out["papers"][0]["title"]


def test_engine_guards(tmp_path, monkeypatch):
    e = _engine(tmp_path, None, {"paper": [], "corpus": [{"question": "Cached Q?", "out": {"answer": "A", "evidence": []}}]})
    out, msg, _ = e.ask_paper("Q?", "1909.00001")
    assert out is None and "no API key" in msg
    out, msg, live = e.ask_corpus("cached q?")                       # examples work without any key
    assert out["cached"] and not live
    e2 = _engine(tmp_path, FakeLLM())
    out, msg, _ = e2.ask_paper("Q?", "1909.00001", session_count=e2.max_per_session)
    assert out is None and "this session" in msg


def test_clean_tex_for_display():
    from app.render import clean_tex
    assert clean_tex(r"We discover that $71.0\%$ of the top 100 and $53.6\%$ of random") == \
        "We discover that 71.0% of the top 100 and 53.6% of random"
    assert clean_tex(r"with $k=5$ and $\alpha = 0.1$, 3 $\times$ faster") == "with k=5 and α = 0.1, 3 × faster"
    assert "71.0%" in evidence_html([{"label": "E1", "section": "S", "text": r"$71.0\%$"}], [], "u")


def test_clean_tex_formulas():
    from app.render import clean_tex
    t = r"$R_{i,t} = (\partial x_k / \partial x_{i,t})^2$ and $w_c\in \mathbb {R}^{V} $"
    assert clean_tex(t) == "R_i,t = (∂ x_k / ∂ x_i,t)^2 and w_c∈ ℝ^V"
