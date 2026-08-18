"""Config-driven model roster — per PLAN.md's repo layout and settled decision:
"Model roster is config-driven and resolved at runtime, never hardcoded."

Model ids churn fast; this list is a snapshot, not a promise. Verify against
https://openrouter.ai/models before trusting it months from now. Deliberately one
generation back from each provider's bleeding-edge flagship, not the newest/priciest
tier — cost control for exploratory runs, not a claim about which model is "best."
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ModelSpec", "MODEL_REGISTRY"]


@dataclass(frozen=True)
class ModelSpec:
    key: str  # stable identifier for UI widgets, config files, etc.
    display_name: str
    model_id: str  # OpenRouter model id
    provider: str


MODEL_REGISTRY: list[ModelSpec] = [
    ModelSpec("claude_opus", "Claude Opus 4.5", "anthropic/claude-opus-4.5", "Anthropic"),
    ModelSpec("claude_sonnet", "Claude Sonnet 4.5", "anthropic/claude-sonnet-4.5", "Anthropic"),
    ModelSpec("gpt", "GPT-5.5", "openai/gpt-5.5", "OpenAI"),
    ModelSpec("gemini", "Gemini 3.6 Flash", "google/gemini-3.6-flash", "Google"),
    ModelSpec("grok", "Grok 4.5", "x-ai/grok-4.5", "xAI"),
    ModelSpec("deepseek", "DeepSeek V3.1", "deepseek/deepseek-chat-v3.1", "DeepSeek"),
]
