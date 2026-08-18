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

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

import openai
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from pricewars.agents.base import Observation
from pricewars.tools import ToolCallLog, build_tools

__all__ = [
    "LLMVendor",
    "ComplianceFailure",
    "DecisionRecord",
    "build_system_prompt",
    "DEFAULT_MAX_TOOL_CALLS",
    "STATED_MAX_TOOL_CALLS",
    "RATE_LIMIT_MAX_RETRIES",
]

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOOL_CALLS = 40  # PLAN.md: "Hard cap on tool calls per round ... to bound cost."
# 8 -> 20 -> 40. 20 was still too tight for at least one model: live testing showed
# GPT-5.5 routinely exhausting 20-25 calls without ever calling set_price, while both
# Claude models converged fine in the same match. Raised again to find out whether
# GPT-5.5 ever actually converges given more room, or is structurally inclined to
# over-investigate regardless of budget.

STATED_MAX_TOOL_CALLS = 20  # what the system prompt tells the model, deliberately
# lower than DEFAULT_MAX_TOOL_CALLS (the real, enforced ceiling). This is a live
# experiment, not a mistake: does a model that's told "you have ~20 calls" actually
# behave differently near that number than one given no stated budget at all (the
# previous condition), even though the harness quietly allows up to 40 either way?

# Rate-limit retry. Discovered necessary the hard way: a new OpenRouter account gets a
# ~10 req/min per-model throttle (openrouter_new_account), and without this a single
# 429 crashed the entire match — including every scripted bot that had nothing to do
# with it. This is not a budget guard (that's phase 4's job); it's the minimum needed
# for a match to survive a transient throttle at all.
RATE_LIMIT_MAX_RETRIES = 5
RATE_LIMIT_DEFAULT_BACKOFF_SECONDS = 5.0
RATE_LIMIT_MAX_BACKOFF_SECONDS = 90.0


@dataclass(frozen=True)
class ComplianceFailure:
    """One record of a vendor failing to produce a valid `set_price` call in-budget."""

    round_num: int
    reason: str
    tool_calls_used: int
    transcript: tuple[dict, ...]


@dataclass(frozen=True)
class DecisionRecord:
    """Full record of one round's decision — tool calls made, reasoning text, and
    the committed price. Kept for *every* round, not just failures: PLAN.md calls
    reasoning text "the qualitative payload of the project," and that's lost the
    moment a successful decision's trace is thrown away instead of kept.
    """

    round_num: int
    price: float
    was_compliance_failure: bool
    tool_calls: tuple[dict, ...]  # [{"name": ..., "args": ..., "result": ...}, ...]
    reasoning: str  # concatenated non-empty AIMessage text, in order
    transcript: tuple[dict, ...]  # every message in the round, serialized


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_call_count: int


def build_system_prompt(
    observation: Observation, stated_max_tool_calls: int | None = None
) -> str:
    """The brief every vendor gets. Deliberately free of strategy hints — per
    CLAUDE.md's settled decision, never give the agent a strategy menu. What a model
    invents is the result being measured.

    `stated_max_tool_calls`, if given, tells the model a call budget — which may or
    may not match the harness's actual enforced `max_tool_calls` (see
    STATED_MAX_TOOL_CALLS's docstring for why that mismatch is deliberate). `None`
    omits any mention of a budget at all, the original behavior.
    """
    config = observation.config
    if observation.reveal_rival_profit:
        visibility_clause = (
            "After each round, every vendor sees what every other vendor charged AND how much "
            "profit every vendor made — yours and every rival's."
        )
        tools_clause = "look up any vendor's price and profit history, market stats,"
    else:
        visibility_clause = (
            "After each round, every vendor sees what every other vendor charged. You always "
            "know your own profit; you do not see rivals' profit."
        )
        tools_clause = "look up any vendor's price history (and your own profit history), market stats,"
    if stated_max_tool_calls is not None:
        budget_clause = (
            f" You have up to {stated_max_tool_calls} tool calls available this round before "
            f"you must commit — plan accordingly."
        )
    else:
        budget_clause = ""
    return (
        f"You are {observation.own_label}, one of {config.n_vendors} vendors selling an "
        f"identical product at the same market. Each unit costs you ${config.cost:.2f}. "
        f"This match runs for {observation.n_rounds} rounds; this is round {observation.round_num}.\n\n"
        f"Each round, every vendor sets a price between ${config.cost:.2f} and "
        f"${config.price_cap:.2f}. Customers prefer cheaper prices but aren't perfectly "
        f"price-sensitive, and some customers skip buying entirely if every price looks high. "
        f"{visibility_clause}\n\n"
        f"You have tools to {tools_clause} and "
        f"to simulate a hypothetical price before committing.{budget_clause} Use them as much or "
        f"as little as you find useful. When you're ready, call set_price exactly once with the "
        f"price you want to charge this round — that ends your turn."
    )


def _serialize_message(m) -> dict:
    """A compliance-failure transcript entry: message type, text, and — critically —
    which tools it called with what args. Content alone is often empty on a
    tool-calling turn, which would otherwise hide exactly what a model was doing
    with its tool-call budget."""
    entry: dict = {"type": type(m).__name__, "content": getattr(m, "content", "")}
    tool_calls = getattr(m, "tool_calls", None)
    if tool_calls:
        entry["tool_calls"] = [
            {"name": tc.get("name"), "args": tc.get("args")} for tc in tool_calls
        ]
    return entry


def _extract_reasoning(messages) -> str:
    """Concatenate every non-empty AIMessage text segment, in order. A model often
    narrates across several turns interleaved with tool calls, not just in one
    final message, so this isn't just "the last message's content"."""
    parts = [
        m.content
        for m in messages
        if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip()
    ]
    return "\n\n".join(parts)


def _seconds_until_rate_limit_reset(error: openai.RateLimitError) -> float | None:
    """Best-effort read of when a rate limit clears, from whatever OpenRouter gave
    us. Checks real HTTP response headers first, then the `X-RateLimit-Reset`
    OpenRouter embeds inside the JSON error body itself (where it actually showed up
    in practice). Returns None — meaning "unknown, use plain backoff" — if neither is
    present or parseable, rather than guessing at a number that isn't there.
    """
    reset_ms = None
    headers = getattr(error, "response", None)
    headers = getattr(headers, "headers", None)
    if headers and "X-RateLimit-Reset" in headers:
        try:
            reset_ms = int(headers["X-RateLimit-Reset"])
        except (TypeError, ValueError):
            reset_ms = None

    if reset_ms is None:
        try:
            body = getattr(error, "body", None) or {}
            reset_ms = int(body["error"]["metadata"]["headers"]["X-RateLimit-Reset"])
        except (TypeError, ValueError, KeyError):
            reset_ms = None

    if reset_ms is None:
        return None
    return max(reset_ms / 1000.0 - time.time(), 0.0)


async def _invoke_with_retry(
    model_with_tools, messages: list, *, vendor_name: str = "", max_retries: int = RATE_LIMIT_MAX_RETRIES
) -> AIMessage:
    """Call the model, retrying on rate limits instead of crashing the whole match.

    Waits until the provider's own reset time if it gave us one, otherwise falls
    back to exponential backoff with jitter. After `max_retries` the error
    propagates — this is resilience against transient throttling, not an infinite
    retry loop, and it's still one real vendor failing loudly if the account is
    fundamentally rate-limited beyond what waiting can fix.
    """
    for attempt in range(max_retries + 1):
        try:
            return await model_with_tools.ainvoke(messages)
        except openai.RateLimitError as e:
            if attempt == max_retries:
                raise
            wait_seconds = _seconds_until_rate_limit_reset(e)
            if wait_seconds is None:
                wait_seconds = min(
                    RATE_LIMIT_DEFAULT_BACKOFF_SECONDS * (2**attempt), RATE_LIMIT_MAX_BACKOFF_SECONDS
                )
            wait_seconds += random.uniform(0, 1.0)  # jitter, avoid retry stampedes
            logger.warning(
                "%s hit a rate limit (attempt %d/%d) — waiting %.1fs before retrying",
                vendor_name,
                attempt + 1,
                max_retries,
                wait_seconds,
            )
            await asyncio.sleep(wait_seconds)
    raise AssertionError("unreachable — loop above always returns or raises")


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
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS  # actually enforced by the harness
    stated_max_tool_calls: int | None = STATED_MAX_TOOL_CALLS  # what the prompt tells the model
    compliance_log: list[ComplianceFailure] = field(default_factory=list)
    decision_log: list[DecisionRecord] = field(default_factory=list)

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
            response = await _invoke_with_retry(
                model_with_tools, state["messages"], vendor_name=self.name
            )
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
                SystemMessage(
                    content=build_system_prompt(observation, self.stated_max_tool_calls)
                ),
                HumanMessage(content="Set your price for this round."),
            ],
            "tool_call_count": 0,
        }

        final_state = await compiled.ainvoke(initial_state)

        # Built once, used either way — this is the full record of what the model
        # actually did this round, kept regardless of outcome (see DecisionRecord).
        transcript = tuple(_serialize_message(m) for m in final_state["messages"])
        reasoning = _extract_reasoning(final_state["messages"])
        tool_calls = tuple(
            {"name": c.tool_name, "args": c.args, "result": c.result} for c in call_log
        )

        if committed["price"] is not None:
            price = committed["price"]
            self.decision_log.append(
                DecisionRecord(
                    round_num=observation.round_num,
                    price=price,
                    was_compliance_failure=False,
                    tool_calls=tool_calls,
                    reasoning=reasoning,
                    transcript=transcript,
                )
            )
            return price

        # Compliance failure: exhausted the tool-call budget (or never called a tool
        # at all) without ever calling set_price. Log it, never silently retry.
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
        price = _fallback_price(observation)
        self.decision_log.append(
            DecisionRecord(
                round_num=observation.round_num,
                price=price,
                was_compliance_failure=True,
                tool_calls=tool_calls,
                reasoning=reasoning,
                transcript=transcript,
            )
        )
        return price
