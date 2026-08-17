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
from pricewars.agents.llm import DEFAULT_MAX_TOOL_CALLS, LLMVendor, build_system_prompt
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


def make_observation(config: MarketConfig, round_num: int = 1) -> Observation:
    return Observation(
        round_num=round_num,
        n_rounds=30,
        own_label="Vendor A",
        own_price_history=(5.0,) if round_num > 1 else (),
        own_profit_history=(),
        rival_table=(),
        rival_price_history={},
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

    def test_default_max_tool_calls_is_eight(self):
        assert DEFAULT_MAX_TOOL_CALLS == 8


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
