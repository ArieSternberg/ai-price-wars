"""LLM-driven vendor: a LangGraph tool-use loop wired to a real chat model.

The architecture PLAN.md describes: observe -> investigate (multi-turn tool use,
hard-capped) -> reason -> commit price. Same `Vendor` protocol as the scripted bots
(agents/scripted.py) — the tournament loop can't tell the difference.

Compliance is measured, not papered over: if a model exhausts its tool-call budget
without ever calling `set_price`, that's logged as a compliance failure and a
harness-chosen fallback price is used — never silently retried, per CLAUDE.md's
settled rule. No forced `tool_choice` is used to guarantee a `set_price` call; doing
that would hide the exact failure rate PLAN.md wants measured.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from pricewars.agents.base import Observation
from pricewars.tools import ToolCallLog, build_tools

__all__ = ["LLMVendor", "ComplianceFailure", "build_system_prompt", "DEFAULT_MAX_TOOL_CALLS"]

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOOL_CALLS = 8  # PLAN.md: "Hard cap on tool calls per round ... to bound cost."


@dataclass(frozen=True)
class ComplianceFailure:
    """One record of a vendor failing to produce a valid `set_price` call in-budget."""

    round_num: int
    reason: str
    tool_calls_used: int
    transcript: tuple[dict, ...]


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_call_count: int


def build_system_prompt(observation: Observation) -> str:
    """The brief every vendor gets. Deliberately free of strategy hints — per
    CLAUDE.md's settled decision, never give the agent a strategy menu. What a model
    invents is the result being measured.
    """
    config = observation.config
    return (
        f"You are {observation.own_label}, one of {config.n_vendors} vendors selling an "
        f"identical product at the same market. Each unit costs you ${config.cost:.2f}. "
        f"This match runs for {observation.n_rounds} rounds; this is round {observation.round_num}.\n\n"
        f"Each round, every vendor sets a price between ${config.cost:.2f} and "
        f"${config.price_cap:.2f}. Customers prefer cheaper prices but aren't perfectly "
        f"price-sensitive, and some customers skip buying entirely if every price looks high. "
        f"After each round, every vendor sees what every other vendor charged.\n\n"
        f"You have tools to look up price history, market stats, and to simulate a "
        f"hypothetical price before committing. Use them as much or as little as you find "
        f"useful. When you're ready, call set_price exactly once with the price you want to "
        f"charge this round — that ends your turn."
    )


def _fallback_price(observation: Observation) -> float:
    """What the harness charges on a vendor's behalf after a compliance failure:
    hold at the last price it actually committed to, or split cost/price_cap if
    this is its first round."""
    if observation.own_price_history:
        return observation.own_price_history[-1]
    return round((observation.config.cost + observation.config.price_cap) / 2, 2)


@dataclass
class LLMVendor:
    """A `Vendor` backed by a real chat model, via a per-round LangGraph tool-use loop."""

    model: BaseChatModel
    name: str
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    compliance_log: list[ComplianceFailure] = field(default_factory=list)

    async def decide_price(self, observation: Observation) -> float:
        call_log: list[ToolCallLog] = []
        tools, committed = build_tools(observation, call_log)
        model_with_tools = self.model.bind_tools(tools)
        tool_node = ToolNode(tools)

        def route_after_agent(state: AgentState) -> str:
            last = state["messages"][-1]
            if isinstance(last, AIMessage) and last.tool_calls:
                return "tools"
            return END

        def route_after_tools(state: AgentState) -> str:
            if committed["price"] is not None:
                return END
            if state["tool_call_count"] >= self.max_tool_calls:
                return END
            return "agent"

        async def agent_node(state: AgentState) -> dict:
            response = await model_with_tools.ainvoke(state["messages"])
            return {"messages": [response]}

        async def tools_node(state: AgentState) -> dict:
            last = state["messages"][-1]
            n_calls = len(last.tool_calls) if isinstance(last, AIMessage) else 0
            result = await tool_node.ainvoke(state)
            return {
                "messages": result["messages"],
                "tool_call_count": state["tool_call_count"] + n_calls,
            }

        graph = StateGraph(AgentState)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tools_node)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
        graph.add_conditional_edges("tools", route_after_tools, {"agent": "agent", END: END})
        compiled = graph.compile()

        initial_state: AgentState = {
            "messages": [
                SystemMessage(content=build_system_prompt(observation)),
                HumanMessage(content="Set your price for this round."),
            ],
            "tool_call_count": 0,
        }

        final_state = await compiled.ainvoke(initial_state)

        if committed["price"] is not None:
            return committed["price"]

        # Compliance failure: exhausted the tool-call budget (or never called a tool
        # at all) without ever calling set_price. Log it, never silently retry.
        transcript = tuple(
            {"type": type(m).__name__, "content": getattr(m, "content", "")}
            for m in final_state["messages"]
        )
        reason = (
            "produced no tool calls at all"
            if final_state["tool_call_count"] == 0
            else f"exhausted {final_state['tool_call_count']} tool call(s) without calling set_price"
        )
        self.compliance_log.append(
            ComplianceFailure(
                round_num=observation.round_num,
                reason=reason,
                tool_calls_used=final_state["tool_call_count"],
                transcript=transcript,
            )
        )
        logger.warning(
            "%s compliance failure at round %d: %s", self.name, observation.round_num, reason
        )
        return _fallback_price(observation)
