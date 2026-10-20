"""HTML for the demo app: answers with clickable [E#] citations, evidence passages, papers and an agent trace.

Pure functions (no Gradio, no models), so they are unit-tested in tests/test_app_render.py.
"""
import html
import re
import uuid

_CITE = re.compile(r"\[(E\d+)\]")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


_TEX_MATH = re.compile(r"\$([^$\n]{1,120})\$")       # inline math: $71.0\%$, $k=5$, $R_{i,t}$
_TEX_SYM = [(r"\mathbb{R}", "ℝ"), (r"\mathbb {R}", "ℝ"), (r"\partial", "∂"), (r"\nabla", "∇"), (r"\Vert", "‖"),
            (r"\vert", "|"), (r"\top", "⊤"), (r"\in", "∈"), (r"\times", "×"), (r"\cdot", "·"), (r"\pm", "±"),
            (r"\approx", "≈"), (r"\leq", "≤"), (r"\geq", "≥"), (r"\le", "≤"), (r"\ge", "≥"), (r"\neq", "≠"),
            (r"\sim", "~"), (r"\sum", "Σ"), (r"\prod", "Π"), (r"\infty", "∞"), (r"\rightarrow", "→"),
            (r"\to", "→"), (r"\alpha", "α"), (r"\beta", "β"), (r"\gamma", "γ"), (r"\delta", "δ"),
            (r"\epsilon", "ε"), (r"\theta", "θ"), (r"\lambda", "λ"), (r"\mu", "μ"), (r"\sigma", "σ"),
            (r"\tau", "τ"), (r"\phi", "φ"), (r"\log", "log"), (r"\exp", "exp"), (r"\left", ""), (r"\right", ""),
            (r"\%", "%"), (r"\&", "&"), (r"\_", "_"), (r"\,", " "), (r"\;", " ")]
_TEX_CMD_LEFT = re.compile(r"\\(?:mathrm|mathbf|mathit|textrm|textbf|text|operatorname|mathcal)\s*")


def clean_tex(text):
    r"""QASPER keeps the papers' LaTeX. For display only: '$71.0\%$' -> '71.0%',
    '$R_{i,t} = (\partial x)^2$' -> 'R_i,t = (∂x)^2', '$w\in \mathbb{R}^V$' -> 'w∈ ℝ^V'. The data is unchanged."""
    def inner(m):
        t = m.group(1)
        for k, v in _TEX_SYM:
            t = t.replace(k, v)
        t = _TEX_CMD_LEFT.sub("", t)
        return t.replace("{", "").replace("}", "").strip()
    text = _TEX_MATH.sub(inner, text or "")
    for k in (r"\%", r"\&", r"\_"):
        text = text.replace(k, k[1])
    return text


def new_uid():
    return uuid.uuid4().hex[:8]


def _inline(text, uid, valid):
    """Escape one line, then bold and citation links ([E3] -> link to that passage, if it was shown)."""
    t = _BOLD.sub(r"<b>\1</b>", html.escape(clean_tex(text)))

    def link(m):
        lab = m.group(1)
        if lab not in valid:
            return f'<span class="cite bad" title="label not in the evidence">[{lab}]</span>'
        return f'<a class="cite" href="#ev-{uid}-{lab}">[{lab}]</a>'
    return _CITE.sub(link, t)


def answer_html(answer, evidence, uid):
    """Answer text as HTML: bullet lines become a list, every [E#] links to its passage below."""
    valid = {e["label"] for e in evidence}
    parts, items = [], []
    for line in (answer or "").splitlines():
        if not line.strip():
            continue
        if _BULLET.match(line):
            items.append(f"<li>{_inline(_BULLET.sub('', line, count=1), uid, valid)}</li>")
            continue
        if items:
            parts.append("<ul>" + "".join(items) + "</ul>")
            items = []
        parts.append(f"<p>{_inline(line.strip(), uid, valid)}</p>")
    if items:
        parts.append("<ul>" + "".join(items) + "</ul>")
    return '<div class="answer">' + ("".join(parts) or "<p><i>No answer.</i></p>") + "</div>"


def evidence_html(evidence, cited_labels, uid, by_paper=False):
    """Every evidence passage the LLM saw; cited ones are open and marked. Hop-2 passages get a badge."""
    cited = set(cited_labels or [])
    out, current = ['<div class="evidence">'], None
    for e in evidence:
        if by_paper and e.get("paper_id") != current:
            current = e.get("paper_id")
            out.append(f'<div class="ev-paper">{html.escape(e.get("paper_label", ""))} '
                       f'{html.escape(e.get("title", ""))} <span class="pid">{html.escape(str(current))}</span></div>')
        badges = ""
        if e["label"] in cited:
            badges += '<span class="badge cited">cited</span>'
        if e.get("hop"):
            badges += f'<span class="badge hop">found by follow-up search {e["hop"]}</span>'
        out.append(
            f'<details id="ev-{uid}-{e["label"]}" class="ev{" is-cited" if e["label"] in cited else ""}"'
            f'{" open" if e["label"] in cited else ""}>'
            f'<summary><b>[{e["label"]}]</b> {html.escape(str(e.get("section", "")))} {badges}</summary>'
            f'<div class="ev-text">{html.escape(clean_tex(e.get("text", "")))}</div></details>')
    out.append("</div>")
    return "".join(out)


def papers_html(papers, cited_papers):
    """Papers the cross-paper search found, in rank order, with arXiv links; cited ones highlighted."""
    cited = set(cited_papers or [])
    rows = []
    for p in papers:
        pid = str(p["paper_id"])
        mark = '<span class="badge cited">cited</span>' if pid in cited else '<span class="badge">not used</span>'
        rows.append(f'<li><a href="https://arxiv.org/abs/{html.escape(pid)}" target="_blank">'
                    f'{html.escape(p.get("title", pid))}</a> <span class="pid">{html.escape(pid)}</span> {mark}</li>')
    return f'<div class="papers"><ol>{"".join(rows)}</ol></div>' if rows else ""


def trace_html(out, mode):
    """What the agent did: sub-questions, the multi-hop check, papers found, tokens, time, which LLM answered."""
    li = []
    if out.get("cached"):
        li.append("<li><b>Cached example</b>: precomputed answer (no API call).</li>")
    subs = out.get("sub_questions") or []
    if subs:
        li.append("<li><b>Search queries</b> (decomposition): " + "; ".join(html.escape(s) for s in subs) + "</li>")
    if "sufficient" in out and out.get("sufficient") is not None:
        if out["sufficient"]:
            li.append("<li><b>Sufficiency check</b>: evidence judged enough, no extra search.</li>")
        else:
            added = sum(len(h.get("added_chunk_ids", [])) for h in out.get("hop_log", []))
            li.append(f'<li><b>Sufficiency check</b>: missing "{html.escape(out.get("missing", ""))}" → follow-up '
                      f'search "{html.escape(out.get("follow_up_query", ""))}" added {added} passage(s).</li>')
    if mode == "corpus" and out.get("papers"):
        li.append(f"<li><b>Papers</b>: {len(out['papers'])} found, {len(out.get('cited_papers', []))} cited.</li>")
    tok = out.get("tokens") or {}
    if tok:
        li.append(f"<li><b>Tokens</b>: {tok.get('prompt', 0):,} prompt + {tok.get('completion', 0):,} completion</li>")
    if out.get("secs") is not None:
        li.append(f"<li><b>Time</b>: {out['secs']} s</li>")
    models = sorted({c.get("model") for c in out.get("llm_calls", []) if c.get("model") not in (None, "retrieval")})
    if models:
        li.append("<li><b>LLM</b>: " + ", ".join(html.escape(m) for m in models) + "</li>")
    if out.get("invalid_labels"):
        li.append("<li><b>Invalid citations</b> (not in the evidence): " + ", ".join(out["invalid_labels"]) + "</li>")
    return '<div class="trace"><ul>' + "".join(li) + "</ul></div>"


CSS = """
.answer{font-size:1.02rem;line-height:1.55}
.answer ul{margin:.3em 0 .3em 1.2em}
a.cite{font-weight:600;text-decoration:none;padding:0 2px;border-radius:4px;background:rgba(99,102,241,.12)}
.cite.bad{color:#b91c1c}
.evidence details{border:1px solid rgba(128,128,128,.25);border-radius:8px;padding:.35em .7em;margin:.35em 0}
.evidence details.is-cited{border-color:rgba(99,102,241,.6)}
.evidence details:target{outline:2px solid #6366f1}
.ev-text{white-space:pre-wrap;font-size:.92rem;margin-top:.4em;opacity:.92}
.ev-paper{margin-top:.8em;font-weight:600}
.pid{font-family:monospace;font-size:.82rem;opacity:.7}
.badge{font-size:.72rem;padding:1px 6px;border-radius:999px;margin-left:6px;border:1px solid rgba(128,128,128,.35)}
.badge.cited{background:rgba(99,102,241,.15);border-color:rgba(99,102,241,.5)}
.badge.hop{background:rgba(234,179,8,.18);border-color:rgba(234,179,8,.6)}
.trace ul{margin:0 0 0 1.1em;font-size:.9rem}
"""
