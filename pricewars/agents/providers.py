"""Minimal OpenRouter wiring — phase 3's "one provider, end to end."

The full multi-model roster (config-driven, pinned providers per model, response
caching) is phase 4's "Provider abstraction." This is deliberately small: just
enough to point an `LLMVendor` at a real OpenRouter model.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from pricewars.agents.llm import DEFAULT_MAX_TOOL_CALLS, LLMVendor

__all__ = ["OPENROUTER_BASE_URL", "build_openrouter_vendor"]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def build_openrouter_vendor(
    model_id: str,
    name: str | None = None,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    temperature: float = 0.7,
) -> LLMVendor:
    """Build an `LLMVendor` backed by a specific OpenRouter model id, e.g.
    `"anthropic/claude-sonnet-4.5"`. Model ids churn fast — check
    https://openrouter.ai/models for what's actually available before using one.

    Reads `OPENROUTER_API_KEY` from the environment (loading `.env` if present).
    Raises `RuntimeError` with a clear message if the key isn't set, rather than
    letting the underlying HTTP client fail obscurely on the first real call.
    """
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and paste your "
            "key in, or export it in your shell."
        )
    model = ChatOpenAI(
        model=model_id,
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        temperature=temperature,
    )
    return LLMVendor(model=model, name=name or model_id, max_tool_calls=max_tool_calls)
