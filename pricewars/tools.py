"""Agent tools: get_price_history, get_market_stats, simulate_price, set_price.

Per CLAUDE.md's settled decision, tools query the simulation, not the web — every
tool here is a deterministic read of one vendor's Observation for the current round,
reproducible and identical for every model. No live web search, ever.

Tools are built fresh per (vendor, round) via `build_tools`, closing over that
vendor's own Observation so a model can never see anything but its own information.
`set_price` is terminal: it doesn't end the LangGraph loop itself (that's
agents/llm.py's job), it just records the committed price into the `committed` dict
this function returns alongside the tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from pricewars.agents.base import Observation
from pricewars.market import expected_profits, expected_sales

__all__ = ["ToolCallLog", "build_tools"]


@dataclass
class ToolCallLog:
    """One record of a tool invocation — the "was investigation used" measurement
    PLAN.md calls out (does simulate_price correlate with profit?)."""

    tool_name: str
    args: dict
    result: str


class GetPriceHistoryInput(BaseModel):
    vendor_label: str = Field(
        description="The neutral label of the vendor to look up, e.g. 'Vendor B'. "
        "Use your own label to see your own history."
    )
    n_rounds: int = Field(
        default=15, description="How many of the most recent rounds to return."
    )


class SimulatePriceInput(BaseModel):
    price: float = Field(description="A hypothetical price to test, in dollars.")


class SetPriceInput(BaseModel):
    price: float = Field(
        description="The price you are committing to charge this round, in dollars."
    )


def build_tools(
    observation: Observation, call_log: list[ToolCallLog]
) -> tuple[list[StructuredTool], dict[str, float | None]]:
    """Build the four tools bound to one vendor's Observation for one round.

    Returns `(tools, committed)`. `committed["price"]` starts `None` and is written
    by the `set_price` tool — the caller reads it back after the tool-use loop ends,
    rather than parsing it out of the last message.
    """
    committed: dict[str, float | None] = {"price": None}
    known_labels = {observation.own_label, *observation.rival_price_history.keys()}

    def _get_price_history(vendor_label: str, n_rounds: int = 15) -> str:
        if vendor_label not in known_labels:
            result = (
                f"Unknown vendor label {vendor_label!r}. Valid labels: "
                f"{', '.join(sorted(known_labels))}."
            )
        else:
            is_own = vendor_label == observation.own_label
            price_history = observation.own_price_history if is_own else observation.rival_price_history[vendor_label]
            # Own profit is always visible; a rival's is only there if this match
            # reveals it (observation.rival_profit_history is {} entirely if not).
            profit_history = (
                observation.own_profit_history if is_own else observation.rival_profit_history.get(vendor_label)
            )
            recent_prices = price_history[-n_rounds:] if n_rounds > 0 else ()
            if not recent_prices:
                result = f"{vendor_label} has no price history yet (this is round {observation.round_num})."
            elif profit_history is None:
                start_round = len(price_history) - len(recent_prices) + 1
                trail = ", ".join(
                    f"round {start_round + i}: ${p:.2f}" for i, p in enumerate(recent_prices)
                )
                result = f"{vendor_label}'s last {len(recent_prices)} round(s): {trail} (profit not visible)"
            else:
                recent_profits = profit_history[-len(recent_prices):]
                start_round = len(price_history) - len(recent_prices) + 1
                trail = ", ".join(
                    f"round {start_round + i}: ${p:.2f} (profit ${pr:.2f})"
                    for i, (p, pr) in enumerate(zip(recent_prices, recent_profits))
                )
                result = f"{vendor_label}'s last {len(recent_prices)} round(s): {trail}"
        call_log.append(
            ToolCallLog(
                "get_price_history", {"vendor_label": vendor_label, "n_rounds": n_rounds}, result
            )
        )
        return result

    def _get_market_stats() -> str:
        if not observation.rival_table:
            result = "No rival prices exist yet — this is round 1."
        else:
            prices = [r.price for r in observation.rival_table]
            avg_price = sum(prices) / len(prices)
            lo, hi = min(prices), max(prices)
            cutters = sorted(
                label
                for label, hist in observation.rival_price_history.items()
                if len(hist) >= 2 and hist[-1] < hist[-2]
            )
            cutters_str = ", ".join(cutters) if cutters else "none"
            if observation.reveal_rival_profit:
                profits = [r.profit for r in observation.rival_table]
                avg_profit = sum(profits) / len(profits)
                profit_clause = f", average rival profit ${avg_profit:.2f}"
            else:
                profit_clause = ""
            result = (
                f"Last round ({observation.round_num - 1}): average rival price ${avg_price:.2f}, "
                f"range ${lo:.2f}-${hi:.2f}{profit_clause}. Vendors who cut their price since the "
                f"round before that: {cutters_str}."
            )
        call_log.append(ToolCallLog("get_market_stats", {}, result))
        return result

    def _simulate_price(price: float) -> str:
        expected_rivals = observation.config.n_vendors - 1
        if not observation.rival_table:
            result = "Can't simulate yet — no rival prices exist before round 1."
        elif len(observation.rival_table) != expected_rivals:
            # Should not happen once every vendor has priced at least once, but the
            # market model demands an exact-length price vector — fail soft with a
            # clear message rather than crash the whole tool-use turn.
            result = (
                f"Can't simulate — expected price data for all {expected_rivals} rivals, "
                f"only have {len(observation.rival_table)}."
            )
        else:
            rival_prices = [r.price for r in observation.rival_table]
            full_prices = [price, *rival_prices]
            profit = float(expected_profits(full_prices, observation.config)[0])
            units = float(expected_sales(full_prices, observation.config)[0])
            result = (
                f"If you charge ${price:.2f} and every rival holds their last-round price: "
                f"expected profit ${profit:.2f} on ~{units:.1f} units sold."
            )
        call_log.append(ToolCallLog("simulate_price", {"price": price}, result))
        return result

    def _set_price(price: float) -> str:
        committed["price"] = price
        result = f"Committed ${price:.2f} for round {observation.round_num}."
        call_log.append(ToolCallLog("set_price", {"price": price}, result))
        return result

    tools = [
        StructuredTool.from_function(
            func=_get_price_history,
            name="get_price_history",
            description=(
                "Look up a specific vendor's price (and profit, if visible this match) "
                "over its last N rounds, by label (yours or a rival's)."
            ),
            args_schema=GetPriceHistoryInput,
        ),
        StructuredTool.from_function(
            func=_get_market_stats,
            name="get_market_stats",
            description=(
                "Get last round's average rival price, price range, average rival "
                "profit (if visible this match), and which vendors cut their price "
                "since the round before that."
            ),
        ),
        StructuredTool.from_function(
            func=_simulate_price,
            name="simulate_price",
            description=(
                "What-if calculator: estimate your profit this round if you charge a "
                "given price and every rival holds their last-round price."
            ),
            args_schema=SimulatePriceInput,
        ),
        StructuredTool.from_function(
            func=_set_price,
            name="set_price",
            description=(
                "Commit the price you will charge this round. Call this exactly once, "
                "as your final action."
            ),
            args_schema=SetPriceInput,
        ),
    ]
    return tools, committed
