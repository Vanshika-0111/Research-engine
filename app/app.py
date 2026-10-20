"""Gradio demo of the Agentic Research & Evidence Synthesis Engine (QASPER, 1,585 NLP papers).

Run locally / in Colab from the project root:
    python -m app.app                 # http://127.0.0.1:7860
    python -m app.app --share         # public link from Colab
On Hugging Face Spaces it is started by the Space's README (app_file: app/app.py).
"""
import argparse
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, so agent/ and retrieval/ import

import gradio as gr  # noqa: E402

from app.engine import Engine  # noqa: E402
from app.render import CSS, answer_html, evidence_html, new_uid, papers_html, trace_html  # noqa: E402

REPO_URL = "https://github.com/Vanshika-0111/Research-engine"

INTRO = f"""
# Agentic Research & Evidence Synthesis Engine
Ask questions about **1,585 NLP research papers** (QASPER). Answers come only from retrieved passages, and every
claim cites the passage it comes from: click a citation like **[E3]** to read it.

**Pipeline:** section-aware chunks → BM25 + dense (bge-small, FAISS) → RRF fusion → cross-encoder reranking
(bge-reranker-v2-m3) → LangGraph agents (gpt-oss-120b via Groq, Gemini as fallback).
**Held-out results:** evidence Recall@5 **0.795** (1,309 questions) · LLM-judged citation accuracy **95.6%**,
faithfulness **96.9%** (100 questions). [Code, evaluation and report on GitHub]({REPO_URL})
"""

HOW = f"""
### Two agents, one retrieval pipeline
**Ask about one paper** — LangGraph: `decompose → retrieve → answer`. The question is split into up to 3 search
queries; the question's own top 5 passages always go in first (anchored merge), sub-query results fill the rest
(7 passages, E1–E7); the LLM answers only from them, citing **[E#]** after every claim.
With **multi-hop** on, a `check` node reads the evidence and, if a specific piece is missing, writes one follow-up
search in the paper's own words (`check → follow_up → answer`, up to 3 new passages E8–E10).

**Ask across papers** — LangGraph: `retrieve_papers → synthesize`. Hybrid search + reranking over all 28.5k chunks,
grouped into the top 10 papers × 2 passages; the LLM writes one cited bullet per relevant paper.

### Evaluation (held-out, chosen on dev, test run once)
| | result |
|---|---|
| evidence Recall@5 (1,309 test questions; BM25 0.612 → final 0.795) | **0.795** |
| citation accuracy, LLM-judged (100 test questions; strict vs QASPER gold 72.6%) | **95.6%** |
| faithfulness / answer correctness (LLM judge, 100 test questions) | **96.9% / 83%** |
| cross-paper: required papers found in top 10 (19 hand-labelled questions; BM25 71%) | **84%** |
| judge validation: agreement gpt-oss-20b vs Gemini (different model family) | **82%** |

Data: [QASPER](https://allenai.org/data/qasper) (Dasigi et al., 2021), CC BY 4.0. Models: BAAI bge-small-en-v1.5,
BAAI bge-reranker-v2-m3, OpenAI gpt-oss-120b (Groq), Google Gemini. Full results and limitations: [GitHub]({REPO_URL}).
"""

_DEMO_RR = os.environ.get("APP_RERANKER_MODEL")
if _DEMO_RR:
    INTRO += (f"\n\n> **Demo note:** this free CPU host reranks with the smaller `{_DEMO_RR}`"
              + (f" (dev evidence Recall@5 {os.environ['APP_RERANKER_DEV_R5']}, vs 0.793 for bge-reranker-v2-m3)"
                 if os.environ.get("APP_RERANKER_DEV_R5") else "")
              + ", because bge-reranker-v2-m3 needs ~100 s per question without a GPU. All reported results use"
                " bge-reranker-v2-m3 on a GPU.")

engine = Engine()


def _example_choices(kind):
    return [ex["question"] if kind == "corpus" else f'{ex["question"]}  —  {ex.get("title", ex["paper_id"])[:60]}'
            for ex in engine.examples.get(kind, [])]


def _render_paper(out):
    uid = new_uid()
    return (answer_html(out["answer"], out["evidence"], uid),
            evidence_html(out["evidence"], out.get("cited_labels"), uid),
            trace_html(out, "paper"))


def _render_corpus(out):
    uid = new_uid()
    return (answer_html(out["answer"], out["evidence"], uid),
            papers_html(out.get("papers", []), out.get("cited_papers")),
            evidence_html(out["evidence"], out.get("cited_labels"), uid, by_paper=True),
            trace_html(out, "corpus"))


def _msg(text):
    return f'<div class="answer"><p><i>{text}</i></p></div>'


def ask_paper(question, paper_id, multihop, count):
    try:
        out, msg, live = engine.ask_paper(question, paper_id, multihop, count)
    except Exception as e:   # provider errors etc.: show, don't crash
        return _msg(f"Something went wrong: {type(e).__name__}: {e}"), "", "", count
    if out is None:
        return _msg(msg), "", "", count
    return (*_render_paper(out), count + int(live))


def ask_corpus(question, count):
    try:
        out, msg, live = engine.ask_corpus(question, count)
    except Exception as e:
        return _msg(f"Something went wrong: {type(e).__name__}: {e}"), "", "", "", count
    if out is None:
        return _msg(msg), "", "", "", count
    return (*_render_corpus(out), count + int(live))


def pick_paper_example(choice):
    for ex, label in zip(engine.examples.get("paper", []), _example_choices("paper")):
        if label == choice:
            return ex["question"], ex["paper_id"], bool(ex.get("multihop"))
    return gr.update(), gr.update(), gr.update()


def build_ui():
    with gr.Blocks(title="Research Evidence Engine") as demo:
        gr.Markdown(INTRO)
        count = gr.State(0)          # live questions asked in this browser session
        with gr.Tabs():
            with gr.Tab("Ask about one paper"):
                with gr.Row():
                    with gr.Column(scale=3):
                        paper = gr.Dropdown(label="Paper (type to search 1,585 titles)", choices=[], filterable=True)
                        q1 = gr.Textbox(label="Question", placeholder="e.g. What datasets do they use?", lines=2)
                        with gr.Row():
                            hop = gr.Checkbox(label="Multi-hop: search again if evidence is missing", value=False)
                            b1 = gr.Button("Ask", variant="primary")
                    with gr.Column(scale=2):
                        ex1 = gr.Dropdown(label="Examples (instant, precomputed)", choices=_example_choices("paper"), value=None)
                a1 = gr.HTML()
                with gr.Accordion("Evidence passages", open=True):
                    e1 = gr.HTML()
                with gr.Accordion("What the agent did", open=False):
                    t1 = gr.HTML()
            with gr.Tab("Ask across papers"):
                with gr.Row():
                    with gr.Column(scale=3):
                        q2 = gr.Textbox(label="Question about several papers",
                                        placeholder="e.g. Which papers run experiments on HotpotQA, and what does each propose?",
                                        lines=2)
                        b2 = gr.Button("Ask", variant="primary")
                    with gr.Column(scale=2):
                        ex2 = gr.Dropdown(label="Examples (instant, precomputed)", choices=_example_choices("corpus"), value=None)
                a2 = gr.HTML()
                with gr.Accordion("Papers found (top 10)", open=True):
                    p2 = gr.HTML()
                with gr.Accordion("Evidence passages", open=False):
                    e2 = gr.HTML()
                with gr.Accordion("What the agent did", open=False):
                    t2 = gr.HTML()
            with gr.Tab("How it works"):
                gr.Markdown(HOW)
        gr.Markdown("<small>Free-tier demo: a few live questions per session; examples are always instant. "
                    "Answers can be wrong — check the cited passages.</small>")

        b1.click(ask_paper, [q1, paper, hop, count], [a1, e1, t1, count])
        q1.submit(ask_paper, [q1, paper, hop, count], [a1, e1, t1, count])
        ex1.change(pick_paper_example, ex1, [q1, paper, hop]).then(
            ask_paper, [q1, paper, hop, count], [a1, e1, t1, count])
        b2.click(ask_corpus, [q2, count], [a2, p2, e2, t2, count])
        q2.submit(ask_corpus, [q2, count], [a2, p2, e2, t2, count])
        ex2.change(lambda c: c, ex2, q2).then(ask_corpus, [q2, count], [a2, p2, e2, t2, count])
        demo.load(lambda: gr.update(choices=engine.papers()), None, paper)
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="public gradio.live link (e.g. from Colab)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    args = ap.parse_args()
    threading.Thread(target=engine.load, daemon=True).start()   # UI comes up while the models load
    build_ui().queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0", server_port=args.port, share=args.share, css=CSS)


if __name__ == "__main__":
    main()
