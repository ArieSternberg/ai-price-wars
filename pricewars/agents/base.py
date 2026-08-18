"""The Vendor protocol: the interface every pricing strategy implements.

Scripted bots (this phase) and LLM agents (phase 3) are both just something with a
`name` and a `decide_price(observation) -> float` coroutine. The tournament loop
doesn't know or care which kind it's driving.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pricewars.market import MarketConfig

__all__ = ["RivalObservation", "Observation", "Vendor"]


@dataclass(frozen=True)
class RivalObservation:
    """One rival's price (and, if the match reveals it, profit) from last round,
    under its neutral label.

    Full profit transparency is a deliberate, opt-in condition, not the realistic
    default — in a real market you don't usually see a competitor's actual profit,
    only their price. `profit` is `None` when the match is run with
    `reveal_rival_profit=False`; check `Observation.reveal_rival_profit` rather than
    inferring visibility from whether this happens to be `None` on any given round
    (round 1 has no rival data yet either way).
    """

    label: str
    price: float
    profit: float | None


@dataclass(frozen=True)
class Observation:
    """What a vendor sees before setting a price for `round_num`.

    Per PLAN.md's "no position effects, by construction": `own_label` and every
    rival's label are stable for the whole match (so `rival_price_history` and
    `rival_profit_history` mean something across rounds — that's what
    `get_price_history` reads from), but `rival_table`'s *order* is reshuffled every
    round by the tournament loop, so no vendor ever benefits from table position.
    """

    round_num: int  # 1-indexed
    n_rounds: int
    own_label: str
    own_price_history: tuple[float, ...]
    own_profit_history: tuple[float, ...]
    rival_table: tuple[RivalObservation, ...]  # last round's prices+profits, order shuffled fresh
    rival_price_history: dict[str, tuple[float, ...]]  # label -> full price history
    rival_profit_history: dict[str, tuple[float, ...]]  # label -> full profit history; empty if hidden
    reveal_rival_profit: bool  # the single source of truth for whether rivals' profit is visible
    config: MarketConfig


class Vendor(Protocol):
    """A pricing strategy. Structural — nothing needs to subclass this."""

    name: str

    async def decide_price(self, observation: Observation) -> float:
        """Return the price to charge this round."""
        ...
