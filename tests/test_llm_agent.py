"""Tests for pricewars/agents/llm.py.

No API calls — a scripted fake chat model stands in for the real provider, so these
tests prove the LangGraph tool-use loop itself (multi-turn investigation, the
tool-call hard cap, compliance-failure logging and fallback) without spending
anything or depending on network access.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from pricewars.agents.base import Observation
from pricewars.agents.llm import (
    DEFAULT_MAX_TOOL_CALLS,
    STATED_MAX_TOOL_CALLS,
    LLMVendor,
    build_system_prompt,
)
from pricewars.market import MarketConfig


class ScriptedChatModel(BaseChatModel):
    """A fake chat model that replays a fixed sequence of AIMessages, one per call.
    The last message repeats indefinitely once the script runs out — useful for
    testing the tool-call hard cap without an infinite response list."""

    responses: list[AIMessage]
    _index: int = PrivateAttr(default=0)

    def bind_tools(self, tools: Any, *, tool_choice: Optional[str] = None, **kwargs: Any):
        return self  # ignore the actual tool schema; tests script exact tool_calls

    def _generate(
        self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager=None, **kwargs: Any
    ) -> ChatResult:
        template = self.responses[self._index]
        if self._index < len(self.responses) - 1:
            self._index += 1
        # A real provider call always returns a fresh message object; LangGraph's
        # add_messages reducer dedupes by `.id`, mutating None -> a fresh uuid the
        # first time it sees a message, in place. Returning the *same* cached
        # object on a repeated (clamped) turn would make later turns silently
        # replace the earlier one instead of appending — so always hand back a copy.
        response = AIMessage(content=template.content, tool_calls=template.tool_calls)
        return ChatResult(generations=[ChatGeneration(message=response)])

    async def _agenerate(
        self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager=None, **kwargs: Any
    ) -> ChatResult:
        return self._generate(messages, stop, run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"


def make_observation(
    config: MarketConfig, round_num: int = 1, reveal_rival_profit: bool = True
) -> Observation:
    return Observation(
        round_num=round_num,
        n_rounds=30,
        own_label="Vendor A",
        own_price_history=(5.0,) if round_num > 1 else (),
        own_profit_history=(),
        rival_table=(),
        rival_price_history={},
        rival_profit_history={},
        reveal_rival_profit=reveal_rival_profit,
        config=config,
    )


def tool_call_message(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def run(coro):
    return asyncio.run(coro)


class TestLLMVendorHappyPath:
    def test_calls_set_price_directly(self):
        model = ScriptedChatModel(responses=[tool_call_message("set_price", {"price": 5.5})])
        vendor = LLMVendor(model=model, name="fake-model")
        config = MarketConfig()
        price = run(vendor.decide_price(make_observation(config)))
        assert price == 5.5
        assert vendor.compliance_log == []

    def test_investigates_before_committing(self):
        """Tool calls in between are fine — set_price just has to happen eventually."""
        model = ScriptedChatModel(
            responses=[
                tool_call_message("get_market_stats", {}, call_id="call_1"),
                tool_call_message("simulate_price", {"price": 5.0}, call_id="call_2"),
                tool_call_message("set_price", {"price": 4.75}, call_id="call_3"),
            ]
        )
        vendor = LLMVendor(model=model, name="fake-model")
        config = MarketConfig()
        price = run(vendor.decide_price(make_observation(config)))
        assert price == 4.75
        assert vendor.compliance_log == []


class TestLLMVendorComplianceFailures:
    def test_no_tool_calls_at_all_falls_back_and_logs(self):
        model = ScriptedChatModel(
            responses=[AIMessage(content="I think I'll price around $5, but I won't call a tool.")]
        )
        vendor = LLMVendor(model=model, name="fake-model")
        config = MarketConfig()
        obs = make_observation(config, round_num=3)  # has own_price_history = (5.0,)
        price = run(vendor.decide_price(obs))
        assert price == 5.0  # falls back to last committed price
        assert len(vendor.compliance_log) == 1
        assert vendor.compliance_log[0].reason == "produced no tool calls at all"
        assert vendor.compliance_log[0].tool_calls_used == 0

    def test_no_tool_calls_first_round_falls_back_to_opening_price(self):
        model = ScriptedChatModel(responses=[AIMessage(content="No tools for me.")])
        vendor = LLMVendor(model=model, name="fake-model")
        config = MarketConfig()
        obs = make_observation(config, round_num=1)  # no own_price_history yet
        price = run(vendor.decide_price(obs))
        assert price == round((config.cost + config.price_cap) / 2, 2)

    def test_exhausts_budget_without_set_price(self):
        """A model that keeps calling get_market_stats forever hits the hard cap."""
        model = ScriptedChatModel(responses=[tool_call_message("get_market_stats", {})])
        vendor = LLMVendor(model=model, name="fake-model", max_tool_calls=3)
        config = MarketConfig()
        obs = make_observation(config, round_num=5)
        price = run(vendor.decide_price(obs))
        assert price == 5.0  # fallback to last own price
        assert len(vendor.compliance_log) == 1
        failure = vendor.compliance_log[0]
        assert failure.tool_calls_used == 3
        assert "exhausted 3 tool call" in failure.reason

    def test_transcript_captures_tool_calls_not_just_empty_content(self):
        """A tool-calling AIMessage usually has empty .content — the transcript must
        still record which tool was called with what args, or a compliance failure
        is undiagnosable."""
        model = ScriptedChatModel(responses=[tool_call_message("get_market_stats", {})])
        vendor = LLMVendor(model=model, name="fake-model", max_tool_calls=2)
        config = MarketConfig()
        run(vendor.decide_price(make_observation(config, round_num=2)))
        transcript = vendor.compliance_log[0].transcript
        ai_entries_with_calls = [e for e in transcript if e.get("tool_calls")]
        assert ai_entries_with_calls
        assert ai_entries_with_calls[0]["tool_calls"][0]["name"] == "get_market_stats"

    def test_respects_custom_max_tool_calls(self):
        model = ScriptedChatModel(responses=[tool_call_message("get_market_stats", {})])
        vendor = LLMVendor(model=model, name="fake-model", max_tool_calls=1)
        config = MarketConfig()
        run(vendor.decide_price(make_observation(config, round_num=2)))
        assert vendor.compliance_log[0].tool_calls_used == 1

    def test_default_max_tool_calls_is_forty(self):
        # 8 -> 20 -> 40. Raised again after live testing showed GPT-5.5 routinely
        # exhausting 20-25 calls without ever committing, while Claude converged fine.
        assert DEFAULT_MAX_TOOL_CALLS == 40

    def test_default_stated_max_tool_calls_is_twenty(self):
        # Deliberately below the real enforced cap — see STATED_MAX_TOOL_CALLS's
        # docstring: an active experiment, not a mismatch to "fix".
        assert STATED_MAX_TOOL_CALLS == 20

    def test_default_vendor_uses_stated_budget_of_twenty(self):
        model = ScriptedChatModel(responses=[tool_call_message("set_price", {"price": 5.0})])
        vendor = LLMVendor(model=model, name="fake-model")
        assert vendor.stated_max_tool_calls == 20
        assert vendor.max_tool_calls == 40

    def test_can_run_with_no_stated_budget_at_all(self):
        """stated_max_tool_calls=None preserves the original no-disclosure behavior."""
        model = ScriptedChatModel(responses=[tool_call_message("set_price", {"price": 5.0})])
        vendor = LLMVendor(model=model, name="fake-model", stated_max_tool_calls=None)
        config = MarketConfig()
        price = run(vendor.decide_price(make_observation(config)))
        assert price == 5.0


class TestBuildSystemPrompt:
    def test_includes_own_label_cost_and_bounds(self):
        config = MarketConfig()
        obs = make_observation(config, round_num=7)
        prompt = build_system_prompt(obs)
        assert "Vendor A" in prompt
        assert f"${config.cost:.2f}" in prompt
        assert f"${config.price_cap:.2f}" in prompt
        assert "round 7" in prompt
        assert str(obs.n_rounds) in prompt

    def test_has_no_strategy_hints(self):
        """Settled decision: never give the agent a strategy menu."""
        config = MarketConfig()
        prompt = build_system_prompt(make_observation(config))
        lowered = prompt.lower()
        for banned in ["undercut", "collude", "punish", "retaliate", "cartel"]:
            assert banned not in lowered

    def test_mentions_rival_profit_when_visible(self):
        config = MarketConfig()
        prompt = build_system_prompt(make_observation(config, reveal_rival_profit=True))
        assert "every rival's" in prompt

    def test_says_rival_profit_hidden_when_not_visible(self):
        config = MarketConfig()
        prompt = build_system_prompt(make_observation(config, reveal_rival_profit=False))
        assert "do not see rivals' profit" in prompt
        assert "every rival's" not in prompt

    def test_omits_budget_mention_by_default(self):
        config = MarketConfig()
        prompt = build_system_prompt(make_observation(config))
        assert "tool calls available" not in prompt

    def test_states_budget_when_given(self):
        config = MarketConfig()
        prompt = build_system_prompt(make_observation(config), stated_max_tool_calls=20)
        assert "up to 20 tool calls available" in prompt

    def test_stated_budget_can_differ_from_a_different_number(self):
        """The prompt just relays whatever number it's given — it has no idea this
        might not match the harness's real enforced cap."""
        config = MarketConfig()
        prompt = build_system_prompt(make_observation(config), stated_max_tool_calls=5)
        assert "up to 5 tool calls available" in prompt


class TestLLMVendorDecisionLog:
    """DecisionRecord is PLAN.md's "reasoning text persisted with every decision" —
    kept every round, not just on compliance failures."""

    def test_successful_round_is_recorded(self):
        model = ScriptedChatModel(
            responses=[
                AIMessage(
                    content="Checking the market before committing.",
                    tool_calls=[{"name": "get_market_stats", "args": {}, "id": "call_1"}],
                ),
                tool_call_message("set_price", {"price": 5.25}, call_id="call_2"),
            ]
        )
        vendor = LLMVendor(model=model, name="fake-model")
        config = MarketConfig()
        price = run(vendor.decide_price(make_observation(config)))

        assert price == 5.25
        assert len(vendor.decision_log) == 1
        record = vendor.decision_log[0]
        assert record.round_num == 1
        assert record.price == 5.25
        assert record.was_compliance_failure is False
        assert record.reasoning == "Checking the market before committing."
        # Both the investigation call and set_price itself are logged tool calls.
        assert [tc["name"] for tc in record.tool_calls] == ["get_market_stats", "set_price"]
        assert record.transcript  # non-empty

    def test_compliance_failure_is_also_recorded(self):
        model = ScriptedChatModel(
            responses=[AIMessage(content="I won't call any tools.", tool_calls=[])]
        )
        vendor = LLMVendor(model=model, name="fake-model")
        config = MarketConfig()
        obs = make_observation(config, round_num=3)  # own_price_history = (5.0,)
        price = run(vendor.decide_price(obs))

        assert price == 5.0  # fallback
        assert len(vendor.decision_log) == 1
        record = vendor.decision_log[0]
        assert record.was_compliance_failure is True
        assert record.price == 5.0
        assert record.reasoning == "I won't call any tools."

    def test_accumulates_across_multiple_rounds(self):
        model = ScriptedChatModel(responses=[tool_call_message("set_price", {"price": 4.0})])
        vendor = LLMVendor(model=model, name="fake-model")
        config = MarketConfig()
        run(vendor.decide_price(make_observation(config, round_num=1)))
        run(vendor.decide_price(make_observation(config, round_num=2)))
        assert [r.round_num for r in vendor.decision_log] == [1, 2]

    def test_reasoning_concatenates_multiple_ai_turns(self):
        model = ScriptedChatModel(
            responses=[
                AIMessage(
                    content="First, let me check history.",
                    tool_calls=[{"name": "get_market_stats", "args": {}, "id": "call_1"}],
                ),
                AIMessage(
                    content="Now I'll commit.",
                    tool_calls=[{"name": "set_price", "args": {"price": 6.0}, "id": "call_2"}],
                ),
            ]
        )
        vendor = LLMVendor(model=model, name="fake-model")
        config = MarketConfig()
        run(vendor.decide_price(make_observation(config)))
        reasoning = vendor.decision_log[0].reasoning
        assert "First, let me check history." in reasoning
        assert "Now I'll commit." in reasoning
