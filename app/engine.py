"""Everything the demo app needs behind three calls: papers(), ask_paper(), ask_corpus().

* One retrieval pipeline (chunks, BM25, bge-small + FAISS, bge-reranker-v2-m3) shared by three agents:
  the frozen within-paper agent (Day 7 v2), the multi-hop variant (Day 14) and the cross-paper agent (v2 prompt).
* One LLM client with provider fallback: Groq gpt-oss-120b first, Gemini when Groq's daily quota is used up.
* Answers are cached (same question -> no new API call); precomputed examples are loaded from app/examples.json.
* A daily cap on live questions protects the shared free-tier quota.

Settings (environment variables, e.g. Hugging Face Space secrets / variables):
    GROQ_API_KEY, GEMINI_API_KEY          at least one is needed for live questions
    APP_GROQ_MODEL    (openai/gpt-oss-120b)    APP_GEMINI_MODEL (gemini-3.6-flash)
    APP_MAX_PER_DAY   (150)  live questions per day for the whole app
    APP_MAX_PER_SESSION (8)  live questions per browser session
    APP_CROSS_RERANK_TOP (100)  fused chunks the cross-encoder re-scores in cross-paper mode (lower = faster on CPU)
    APP_RERANKER_MODEL  (BAAI/bge-reranker-v2-m3)  a smaller cross-encoder for a CPU-only host (the free HF Space):
                        v2-m3 needs ~100 s for 30 passages on 2 CPU cores. The evaluation numbers are always v2-m3;
                        the demo's own dev Recall@5 is measured with `eval.run_retrieval_eval --reranker ...`.
    APP_RERANKER_MAX_LENGTH (512)  tokens per (question, passage) pair for the reranker
"""
import datetime as dt
import json
import os
import threading
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
EXAMPLES_PATH = APP_DIR / "examples.json"


def env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def demo_pipeline():
    """The frozen pipeline, unless APP_RERANKER_MODEL / APP_RERANKER_MAX_LENGTH ask for a faster demo reranker."""
    try:
        import torch
        torch.set_num_threads(os.cpu_count() or 1)   # Colab/Spaces may start torch with 1 thread
    except Exception:
        pass
    from retrieval.config import RetrievalConfig
    from retrieval.pipeline import RetrievalPipeline, get_pipeline
    model = os.environ.get("APP_RERANKER_MODEL")
    length = os.environ.get("APP_RERANKER_MAX_LENGTH")
    if not model and not length:
        return get_pipeline()
    cfg = RetrievalConfig()
    if model:
        cfg.reranker_model = model
    if length:
        cfg.reranker_max_length = int(length)
    print(f"demo reranker: {cfg.reranker_model} (max_length {cfg.reranker_max_length})")
    return RetrievalPipeline(cfg)


def make_app_llm():
    """FallbackLLM over the providers that have a key. None if there is no key at all (cached examples only)."""
    from agent.llm import FallbackLLM, GeminiLLM, GroqLLM
    llms = []
    if os.environ.get("GROQ_API_KEY"):
        llms.append(GroqLLM(os.environ.get("APP_GROQ_MODEL", "openai/gpt-oss-120b"), max_retries=2))
    if os.environ.get("GEMINI_API_KEY"):
        llms.append(GeminiLLM(os.environ.get("APP_GEMINI_MODEL", "gemini-3.6-flash"), max_retries=2))
    return FallbackLLM(llms) if llms else None


class DailyBudget:
    """Counts live questions per calendar day (UTC); resets itself at midnight."""

    def __init__(self, limit):
        self.limit, self.day, self.used = limit, dt.datetime.now(dt.timezone.utc).date(), 0
        self.lock = threading.Lock()

    def take(self):
        with self.lock:
            today = dt.datetime.now(dt.timezone.utc).date()
            if today != self.day:
                self.day, self.used = today, 0
            if self.used >= self.limit:
                return False
            self.used += 1
            return True


def norm_q(q):
    return " ".join((q or "").lower().split())


class Engine:
    def __init__(self, pipeline=None, llm="auto", examples_path=EXAMPLES_PATH):
        self._pipeline, self._llm_arg = pipeline, llm
        self.ready = threading.Event()
        self.error = None
        self.cache = {}
        self.examples = {"paper": [], "corpus": []}
        self.budget = DailyBudget(env_int("APP_MAX_PER_DAY", 150))
        self.max_per_session = env_int("APP_MAX_PER_SESSION", 8)
        self.lock = threading.Lock()   # one question at a time through the models
        if Path(examples_path).exists():
            self.examples = json.loads(Path(examples_path).read_text(encoding="utf-8"))
            for ex in self.examples.get("paper", []):
                self.cache[("paper", ex["paper_id"], bool(ex.get("multihop")), norm_q(ex["question"]))] = ex["out"]
            for ex in self.examples.get("corpus", []):
                self.cache[("corpus", None, False, norm_q(ex["question"]))] = ex["out"]

    # ------------------------------------------------------------------ loading
    def load(self):
        """Load the pipeline, models and agents (call once at startup, e.g. in a background thread)."""
        try:
            t0 = time.time()
            from agent.cross_paper import CrossPaperAgent, CrossPaperAgentConfig
            from agent.graph import ResearchAgent
            from agent.multihop import MultiHopAgent
            pipe = self._pipeline or demo_pipeline()
            self.pipeline = pipe
            self.llm = make_app_llm() if self._llm_arg == "auto" else self._llm_arg
            # paper list for the dropdown: (label, paper_id), sorted by title
            first = pipe.chunks.drop_duplicates("paper_id")
            self.paper_titles = dict(zip(first["paper_id"].astype(str), first["title"].astype(str)))
            if self.llm is not None:
                self.within = ResearchAgent(pipeline=pipe, llm=self.llm)
                self.multihop = MultiHopAgent(pipeline=pipe, llm=self.llm)
                self.cross = CrossPaperAgent(CrossPaperAgentConfig(prompt="v2"), pipeline=pipe, llm=self.llm)
                self.cross.retriever.cfg.rerank_top = env_int("APP_CROSS_RERANK_TOP", 100)
            _ = pipe.reranker.score([("warm up", "the reranker")])   # load the cross-encoder now, not on question 1
            _ = pipe.dense.faiss                                      # build the FAISS index now
            self.reranker_name = getattr(getattr(pipe, "config", None), "reranker_model", "?")
            print(f"engine ready in {time.time() - t0:.0f} s | live LLM: {self.llm is not None} | reranker: {self.reranker_name}")
        except Exception as e:   # shown in the UI instead of crashing the app
            self.error = f"{type(e).__name__}: {e}"
            print("engine failed to load:", self.error)
        finally:
            self.ready.set()

    def papers(self):
        self.ready.wait()
        if self.error:
            return []
        return sorted(((f"{t} ({p})", p) for p, t in self.paper_titles.items()), key=lambda x: x[0].lower())

    # ------------------------------------------------------------------ asking
    def _guard(self, session_count):
        """None if a live question is allowed, else the reason it is not."""
        if self.error:
            return f"The app could not load its models: {self.error}"
        if getattr(self, "llm", None) is None:
            return "Live questions are off (no API key configured). Pick one of the examples."
        if session_count >= self.max_per_session:
            return (f"You have asked {self.max_per_session} live questions in this session (free-tier limit). "
                    "The examples still work, or reload the page later.")
        if not getattr(self.llm, "available", True):   # FallbackLLM: every provider out of daily quota
            return "Today's free LLM quota is used up. The examples still work; please try again tomorrow."
        if not self.budget.take():
            return "The app's daily question limit is reached. The examples still work; please try again tomorrow."
        return None

    def ask_paper(self, question, paper_id, multihop=False, session_count=0):
        """-> (out dict or None, message or None, used_live_call)"""
        self.ready.wait()
        question = (question or "").strip()
        if not question or not paper_id:
            return None, "Pick a paper and type a question.", False
        key = ("paper", str(paper_id), bool(multihop), norm_q(question))
        if key in self.cache:
            return dict(self.cache[key], cached=True), None, False
        reason = self._guard(session_count)
        if reason:
            return None, reason, False
        with self.lock:
            agent = self.multihop if multihop else self.within
            out = agent.run(question, str(paper_id))
        out = self._slim(out, multihop=multihop)
        self.cache[key] = out
        return out, None, True

    def ask_corpus(self, question, session_count=0):
        self.ready.wait()
        question = (question or "").strip()
        if not question:
            return None, "Type a question.", False
        key = ("corpus", None, False, norm_q(question))
        if key in self.cache:
            return dict(self.cache[key], cached=True), None, False
        reason = self._guard(session_count)
        if reason:
            return None, reason, False
        with self.lock:
            out = self.cross.run(question)
        out = self._slim(out)
        self.cache[key] = out
        return out, None, True

    @staticmethod
    def _slim(out, multihop=False):
        """Only what the UI shows (keeps the cache small and JSON-serialisable)."""
        keep = ("answer", "evidence", "cited_labels", "cited_chunk_ids", "invalid_labels", "sub_questions",
                "papers", "cited_papers", "tokens", "secs", "llm_calls")
        if multihop:
            keep += ("sufficient", "missing", "follow_up_query", "hop_log", "hops")
        return {k: out[k] for k in keep if k in out}
