"""Shared fakes. The tests need no GPU, no model downloads, no data files and no API keys."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeLLM:
    """Returns canned replies in order and records every call (like GroqLLM.chat)."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def chat(self, messages, max_tokens=512, json_mode=False):
        self.calls.append({"messages": messages, "max_tokens": max_tokens, "json_mode": json_mode})
        text = self.replies.pop(0) if self.replies else ""
        return text, {"prompt_tokens": 100, "completion_tokens": 10, "secs": 0.0, "model": "fake"}


@pytest.fixture
def fake_llm():
    return FakeLLM
