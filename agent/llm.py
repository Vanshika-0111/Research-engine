"""Thin chat clients (Groq and Gemini) that record token usage and latency for every call.

Both classes have the same interface, so the agent and the judge don't care which provider they use:
    llm = make_llm("openai/gpt-oss-120b")      # Groq
    llm = make_llm("gemini-3.8-flash")         # Google Gemini API
    text, usage = llm.chat(messages, max_tokens=..., json_mode=True)
messages are OpenAI-style dicts: {"role": "system" | "user" | "assistant", "content": "..."}.
usage = {prompt_tokens, completion_tokens, secs, model}; completion_tokens includes hidden reasoning tokens.
"""
import os
import re
import time

REASONING_MODELS = ("gpt-oss", "qwen3")


def _api_key(name, provider):
    key = os.environ.get(name)
    if key:
        return key.strip()
    try:  # in Colab, fall back to the secret of the same name
        from google.colab import userdata
        return userdata.get(name).strip()
    except Exception:
        raise SystemExit(f"No {provider} key. Set the {name} environment variable or add a Colab secret named {name}.")


def _groq_api_key():
    return _api_key("GROQ_API_KEY", "Groq")


def is_gemini(model):
    return model.lower().removeprefix("models/").startswith("gemini")


def make_llm(model, temperature=0.0, reasoning_effort="low"):
    """Picks the provider from the model name: gemini-* -> Gemini API, anything else -> Groq."""
    if is_gemini(model):
        return GeminiLLM(model, reasoning_effort=reasoning_effort)
    return GroqLLM(model, temperature=temperature, reasoning_effort=reasoning_effort)


def is_overloaded(e):
    """The provider is temporarily overloaded (5xx after all retries): stop, save, and --resume later."""
    msg = str(e).lower()
    return getattr(e, "code", None) in (500, 502, 503, 504) or "unavailable" in msg or "high demand" in msg


def is_daily_limit(e):
    """A per-day quota error (Groq TPD/RPD, or Gemini's ...PerDay... quota): waiting a few seconds won't help."""
    msg = str(e).lower()
    groq = "rate_limit" in msg and ("per day" in msg or "tpd" in msg or "rpd" in msg)
    gemini = ("resource_exhausted" in msg or "429" in msg) and ("perday" in msg or "per day" in msg)
    return groq or gemini


class GroqLLM:
    def __init__(self, model, temperature=0.0, reasoning_effort="low", max_retries=6, client=None):
        if client is None:
            from groq import Groq
            # the SDK retries 429 / 5xx itself, waiting as long as Groq's retry-after header says
            client = Groq(api_key=_groq_api_key(), max_retries=max_retries)
        self.client, self.model, self.temperature = client, model, temperature
        self.reasoning_effort = reasoning_effort
        self.is_reasoning = any(t in model.lower() for t in REASONING_MODELS)
        self.json_ok = True  # switched off if the model rejects response_format=json_object

    def _create(self, messages, max_tokens, json_mode):
        kwargs = dict(model=self.model, messages=messages, temperature=self.temperature,
                      max_completion_tokens=max_tokens)
        if self.is_reasoning and self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
            if "gpt-oss" in self.model.lower():
                kwargs["include_reasoning"] = False  # keep the thinking out of the reply text
        if json_mode and self.json_ok:
            kwargs["response_format"] = {"type": "json_object"}
        return self.client.chat.completions.create(**kwargs)

    def chat(self, messages, max_tokens=512, json_mode=False):
        """Returns (text, usage) where usage = {prompt_tokens, completion_tokens, secs, model}."""
        t0 = time.time()
        prompt_tokens = completion_tokens = 0
        for attempt in range(2):
            try:
                resp = self._create(messages, max_tokens, json_mode)
            except Exception as e:
                # a few models refuse JSON mode; retry once in plain text (the parser copes with plain text)
                if json_mode and self.json_ok and "response_format" in str(e):
                    self.json_ok = False
                    resp = self._create(messages, max_tokens, json_mode)
                else:
                    raise
            u = resp.usage
            prompt_tokens += getattr(u, "prompt_tokens", 0) or 0
            completion_tokens += getattr(u, "completion_tokens", 0) or 0
            choice = resp.choices[0]
            text = choice.message.content or ""
            # reasoning can use up the whole budget and leave no answer: retry once with double the budget
            if text.strip() or choice.finish_reason != "length" or attempt == 1:
                break
            max_tokens *= 2
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                 "secs": round(time.time() - t0, 3), "model": self.model}
        return text, usage


class GeminiLLM:
    """Google Gemini API client (google-genai SDK) with the same chat() interface as GroqLLM.

    Notes for Gemini 3.x models:
      - temperature is left at the model default (1.0): Google advises against lowering it for Gemini 3,
        because it can cause looping. So Gemini replies are not fully deterministic.
      - reasoning_effort maps to thinking_level ("low" keeps hidden reasoning short and cheap);
        max_output_tokens includes the thinking tokens, like Groq's reasoning models.
      - per-minute 429s are waited out here; a per-day quota error is raised so the caller can stop and save.
    """

    def __init__(self, model, reasoning_effort="low", max_retries=6, overload_retries=2, client=None):
        if client is None:
            from google import genai
            client = genai.Client(api_key=_api_key("GEMINI_API_KEY", "Gemini"))
        self.client, self.model = client, model
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        # free-tier requests are counted per day even when Google answers 503 "high demand", so only retry
        # an overloaded model a couple of times; the caller then stops cleanly and resumes later
        self.overload_retries = overload_retries
        self.json_ok = True

    def _config(self, system, max_tokens, json_mode):
        from google.genai import types
        kwargs = dict(max_output_tokens=max_tokens)
        if system:
            kwargs["system_instruction"] = system
        if json_mode and self.json_ok:
            kwargs["response_mime_type"] = "application/json"
        if self.reasoning_effort:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=self.reasoning_effort)
        return types.GenerateContentConfig(**kwargs)

    def _create(self, system, contents, max_tokens, json_mode):
        overloads = 0
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.models.generate_content(
                    model=self.model, contents=contents, config=self._config(system, max_tokens, json_mode))
            except Exception as e:
                code = getattr(e, "code", None)
                if is_daily_limit(e) or attempt == self.max_retries or code not in (429, 500, 502, 503, 504):
                    raise
                if code != 429:
                    overloads += 1
                    if overloads > self.overload_retries:
                        raise
                # per-minute limit or a busy server: wait as long as the error says (default 30 s), then retry
                m = re.search(r"retry(?:Delay)?['\"]?\s*[:in ]+\s*['\"]?(\d+(?:\.\d+)?)\s*s", str(e), re.I)
                wait = float(m.group(1)) + 1 if m else min(60, 5 * 2 ** attempt)
                print(f"  (Gemini {code}: waiting {wait:.0f}s, retry {attempt + 1}/{self.max_retries})")
                time.sleep(wait)

    def chat(self, messages, max_tokens=512, json_mode=False):
        """Returns (text, usage) where usage = {prompt_tokens, completion_tokens, secs, model}."""
        t0 = time.time()
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                    for m in messages if m["role"] != "system"]
        prompt_tokens = completion_tokens = 0
        for attempt in range(2):
            resp = self._create(system, contents, max_tokens, json_mode)
            u = resp.usage_metadata
            prompt_tokens += (getattr(u, "prompt_token_count", 0) or 0) if u else 0
            completion_tokens += ((getattr(u, "candidates_token_count", 0) or 0)
                                  + (getattr(u, "thoughts_token_count", 0) or 0)) if u else 0
            try:
                text = resp.text or ""
            except Exception:  # blocked or empty candidate
                text = ""
            finish = str(resp.candidates[0].finish_reason) if resp.candidates else ""
            # thinking can use up the whole budget and leave no answer: retry once with double the budget
            if text.strip() or "MAX_TOKENS" not in finish or attempt == 1:
                break
            max_tokens *= 2
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                 "secs": round(time.time() - t0, 3), "model": self.model}
        return text, usage
