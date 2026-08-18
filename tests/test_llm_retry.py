"""Tests for the rate-limit retry logic in pricewars/agents/llm.py.

No real API calls, no real waiting — asyncio.sleep is monkeypatched so retry tests
run instantly regardless of the backoff duration they'd compute.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage

from pricewars.agents.llm import (
    RATE_LIMIT_MAX_RETRIES,
    _invoke_with_retry,
    _seconds_until_rate_limit_reset,
)


def make_rate_limit_error(reset_ms: int | None = None, header_location: str = "body") -> openai.RateLimitError:
    """Build a real openai.RateLimitError, optionally carrying a reset timestamp
    either as a genuine HTTP response header or embedded in the JSON error body —
    OpenRouter was observed doing the latter in practice."""
    body: dict = {"error": {"message": "Rate limit exceeded", "code": 429, "metadata": {}}}
    headers = {"content-type": "application/json"}
    if reset_ms is not None:
        if header_location == "body":
            body["error"]["metadata"]["headers"] = {"X-RateLimit-Reset": str(reset_ms)}
        else:
            headers["X-RateLimit-Reset"] = str(reset_ms)
    response = httpx.Response(
        status_code=429,
        headers=headers,
        json=body,
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
    )
    return openai.RateLimitError("rate limited", response=response, body=body)


class TestSecondsUntilRateLimitReset:
    def test_reads_reset_from_body_metadata(self):
        reset_ms = int((time.time() + 30) * 1000)
        error = make_rate_limit_error(reset_ms, header_location="body")
        wait = _seconds_until_rate_limit_reset(error)
        assert wait is not None
        assert 28 <= wait <= 30

    def test_reads_reset_from_response_headers(self):
        reset_ms = int((time.time() + 15) * 1000)
        error = make_rate_limit_error(reset_ms, header_location="response_headers")
        wait = _seconds_until_rate_limit_reset(error)
        assert wait is not None
        assert 13 <= wait <= 15

    def test_clamps_past_reset_to_zero(self):
        reset_ms = int((time.time() - 30) * 1000)  # already in the past
        error = make_rate_limit_error(reset_ms)
        assert _seconds_until_rate_limit_reset(error) == 0.0

    def test_returns_none_when_no_reset_info(self):
        error = make_rate_limit_error(reset_ms=None)
        assert _seconds_until_rate_limit_reset(error) is None


class FlakyModel:
    """A fake model_with_tools that raises RateLimitError N times before succeeding."""

    def __init__(self, fail_times: int, reset_ms: int | None = None):
        self.fail_times = fail_times
        self.reset_ms = reset_ms
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise make_rate_limit_error(self.reset_ms)
        return AIMessage(content="", tool_calls=[{"name": "set_price", "args": {"price": 5.0}, "id": "1"}])


class TestInvokeWithRetry:
    async def _run(self, model, monkeypatch):
        sleeps: list[float] = []

        async def fake_sleep(seconds):
            sleeps.append(seconds)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        result = await _invoke_with_retry(model, [], vendor_name="test-vendor")
        return result, sleeps

    def test_succeeds_immediately_with_no_errors(self, monkeypatch):
        model = FlakyModel(fail_times=0)
        result, sleeps = asyncio.run(self._run(model, monkeypatch))
        assert result.tool_calls[0]["name"] == "set_price"
        assert sleeps == []
        assert model.calls == 1

    def test_retries_and_eventually_succeeds(self, monkeypatch):
        model = FlakyModel(fail_times=2, reset_ms=int((time.time() + 3) * 1000))
        result, sleeps = asyncio.run(self._run(model, monkeypatch))
        assert result.tool_calls[0]["name"] == "set_price"
        assert model.calls == 3
        assert len(sleeps) == 2

    def test_uses_reset_based_wait_when_available(self, monkeypatch):
        model = FlakyModel(fail_times=1, reset_ms=int((time.time() + 10) * 1000))
        _, sleeps = asyncio.run(self._run(model, monkeypatch))
        assert len(sleeps) == 1
        assert 9 <= sleeps[0] <= 11  # ~10s reset wait + up to 1s jitter

    def test_falls_back_to_exponential_backoff_without_reset_info(self, monkeypatch):
        model = FlakyModel(fail_times=1, reset_ms=None)
        _, sleeps = asyncio.run(self._run(model, monkeypatch))
        assert len(sleeps) == 1
        assert 5 <= sleeps[0] <= 6  # first attempt: 5s base + up to 1s jitter

    def test_raises_after_exhausting_max_retries(self, monkeypatch):
        model = FlakyModel(fail_times=RATE_LIMIT_MAX_RETRIES + 1, reset_ms=None)
        with pytest.raises(openai.RateLimitError):
            asyncio.run(self._run(model, monkeypatch))
        assert model.calls == RATE_LIMIT_MAX_RETRIES + 1

    def test_non_rate_limit_errors_are_not_retried(self, monkeypatch):
        class BrokenModel:
            calls = 0

            async def ainvoke(self, messages):
                self.calls += 1
                raise ValueError("something else entirely")

        model = BrokenModel()
        with pytest.raises(ValueError):
            asyncio.run(self._run(model, monkeypatch))
        assert model.calls == 1
