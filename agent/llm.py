"""Thin Groq chat client that records token usage and latency for every call."""
import os
import time

REASONING_MODELS = ("gpt-oss", "qwen3")


def _groq_api_key():
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key.strip()
    try:  # in Colab, fall back to the GROQ_API_KEY secret
        from google.colab import userdata
        return userdata.get("GROQ_API_KEY").strip()
    except Exception:
        raise SystemExit("No Groq key. Set the GROQ_API_KEY environment variable or add a Colab secret named GROQ_API_KEY.")


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
