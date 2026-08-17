"""Tests for pricewars/agents/providers.py. No network calls — just verifies the
client is wired up correctly and fails clearly when the API key is missing."""

from __future__ import annotations

import pytest

from pricewars.agents.llm import LLMVendor
from pricewars.agents.providers import OPENROUTER_BASE_URL, build_openrouter_vendor


def test_raises_clearly_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    # load_dotenv() searches upward from this module's file, not the process cwd —
    # it would find the project's real .env (with a real key) regardless of chdir.
    # Stub it out so this test only sees the environment we control.
    monkeypatch.setattr("pricewars.agents.providers.load_dotenv", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        build_openrouter_vendor("some/model")


def test_builds_configured_vendor(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    vendor = build_openrouter_vendor("anthropic/claude-sonnet-4.5", max_tool_calls=5)
    assert isinstance(vendor, LLMVendor)
    assert vendor.name == "anthropic/claude-sonnet-4.5"
    assert vendor.max_tool_calls == 5
    assert vendor.model.openai_api_base == OPENROUTER_BASE_URL


def test_custom_name_overrides_model_id(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    vendor = build_openrouter_vendor("anthropic/claude-sonnet-4.5", name="claude")
    assert vendor.name == "claude"


def test_default_max_tool_calls(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    vendor = build_openrouter_vendor("some/model")
    from pricewars.agents.llm import DEFAULT_MAX_TOOL_CALLS

    assert vendor.max_tool_calls == DEFAULT_MAX_TOOL_CALLS
