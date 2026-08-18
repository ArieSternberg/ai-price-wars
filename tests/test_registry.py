"""Tests for pricewars/agents/registry.py."""

from __future__ import annotations

from pricewars.agents.registry import MODEL_REGISTRY


def test_registry_is_nonempty():
    assert len(MODEL_REGISTRY) > 0


def test_keys_are_unique():
    keys = [m.key for m in MODEL_REGISTRY]
    assert len(keys) == len(set(keys))


def test_model_ids_are_unique():
    ids = [m.model_id for m in MODEL_REGISTRY]
    assert len(ids) == len(set(ids))


def test_model_ids_look_like_openrouter_slugs():
    for spec in MODEL_REGISTRY:
        assert "/" in spec.model_id, f"{spec.key}: {spec.model_id!r} missing provider/model slash"


def test_two_anthropic_entries():
    anthropic = [m for m in MODEL_REGISTRY if m.provider == "Anthropic"]
    assert len(anthropic) == 2
