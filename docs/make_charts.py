"""Charts for the README / report, drawn from the saved result JSONs (no models, no LLM calls).

    python docs/make_charts.py        # -> docs/retrieval_stages.png, docs/cross_paper_retrieval.png,
                                      #    docs/reranker_cpu_tradeoff.png
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "docs"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, BLUE_LIGHT = "#2a78d6", "#a9c9ef"     # one series: the highlighted bar in full blue, the rest lighter

plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "font.family": "DejaVu Sans"})


def _style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def hbar(path, labels, values, highlight, title, xlabel, fmt, xmax, percent_axis=False):
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(labels) + 1.3), dpi=160)
    y = range(len(labels))[::-1]
    colors = [BLUE if i == highlight else BLUE_LIGHT for i in range(len(labels))]
    ax.barh(list(y), values, height=0.55, color=colors)
    for yi, v in zip(y, values):
        ax.text(v + xmax * 0.01, yi, fmt(v), va="center", ha="left", color=INK, fontsize=10.5)
    ax.set_yticks(list(y), labels)
    ax.set_xlim(0, xmax)
    if percent_axis:
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", color=INK, fontsize=12.5, pad=12)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    r = json.loads((RES / "retrieval_eval_test_paper.json").read_text())["results"]
    stages = [("BM25", "bm25"), ("Dense (bge-small)", "dense"), ("Hybrid (RRF)", "hybrid_rrf"),
              ("+ cross-encoder rerank", "hybrid+rerank"), ("+ abstracts excluded (final)", "final")]
    hbar(OUT / "retrieval_stages.png", [s for s, _ in stages], [r[k]["ev_recall@5"] for _, k in stages], 4,
         "Evidence Recall@5 by retrieval stage\n1,309 held-out test questions", "evidence Recall@5",
         lambda v: f"{v:.3f}", 1.0)

    x = json.loads((RES / "cross_paper_retrieval.json").read_text())["results"]
    systems = [("BM25", "bm25"), ("Dense (FAISS)", "dense"), ("Hybrid (RRF)", "hybrid"), ("Hybrid + cross-encoder", "hybrid+rerank")]
    hbar(OUT / "cross_paper_retrieval.png", [s for s, _ in systems], [x[k]["paper_recall@10"] for _, k in systems], 3,
         "Required papers found in the top 10\n19 hand-labelled cross-paper questions", "paper Recall@10",
         lambda v: f"{100 * v:.0f}%", 1.0, percent_axis=True)

    b = json.loads((RES / "cpu_reranker_benchmark.json").read_text())
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=160)
    names = {"BAAI/bge-reranker-v2-m3": "bge-reranker-v2-m3 (568M)\nused for all reported results",
             "cross-encoder/ms-marco-MiniLM-L-6-v2": "MiniLM-L-6 (22M)\nCPU demo choice",
             "cross-encoder/ms-marco-MiniLM-L-12-v2": "MiniLM-L-12 (33M)",
             "BAAI/bge-reranker-base": "bge-reranker-base (278M)\n256 tokens"}
    offsets = {"BAAI/bge-reranker-v2-m3": (-12, 10, "right"), "cross-encoder/ms-marco-MiniLM-L-6-v2": (10, -16, "left"),
               "cross-encoder/ms-marco-MiniLM-L-12-v2": (10, 6, "left"), "BAAI/bge-reranker-base": (10, -4, "left")}
    for m in b["rerankers"]:
        hl = m["model"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ax.scatter(m["cpu_s_30"], m["dev_r5"], s=70 if hl else 50, color=BLUE if hl else BLUE_LIGHT,
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        dx, dy, ha = offsets[m["model"]]
        ax.annotate(f"{names[m['model']]}\nR@5 {m['dev_r5']:.3f} · {m['cpu_s_30']:g} s", (m["cpu_s_30"], m["dev_r5"]),
                    textcoords="offset points", xytext=(dx, dy), ha=ha, va="center", fontsize=9.5, color=INK)
    ax.axhline(b["no_reranker_dev_r5"], color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(380, b["no_reranker_dev_r5"] - 0.0045, f"hybrid search, no reranker: {b['no_reranker_dev_r5']:.3f}",
            color=INK2, fontsize=9.5, ha="right")
    ax.set_xscale("log")
    ax.set_xlim(2, 400)
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([3, 10, 30, 100, 300]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} s"))
    ax.set_ylim(0.67, 0.81)
    ax.set_xlabel("time to rerank 30 passages, one question (log scale)")
    ax.set_ylabel("dev evidence Recall@5")
    ax.set_title("Reranker quality vs CPU latency\n887 dev questions; 2-core CPU like a free cloud host", loc="left", color=INK, fontsize=12.5, pad=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "reranker_cpu_tradeoff.png")
    plt.close(fig)
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*.png"))))


if __name__ == "__main__":
    sys.exit(main())
