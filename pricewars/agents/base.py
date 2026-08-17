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
    """One rival's price from last round, under its neutral label."""

    label: str
    price: float


@dataclass(frozen=True)
class Observation:
    """What a vendor sees before setting a price for `round_num`.

    Per PLAN.md's "no position effects, by construction": `own_label` and every
    rival's label are stable for the whole match (so `rival_price_history` means
    something across rounds — that's what `get_price_history` will read from later),
    but `rival_table`'s *order* is reshuffled every round by the tournament loop, so
    no vendor ever benefits from table position.
    """

    round_num: int  # 1-indexed
    n_rounds: int
    own_label: str
    own_price_history: tuple[float, ...]
    own_profit_history: tuple[float, ...]
    rival_table: tuple[RivalObservation, ...]  # last round's prices, order shuffled fresh
    rival_price_history: dict[str, tuple[float, ...]]  # label -> full price history
    config: MarketConfig


class Vendor(Protocol):
    """A pricing strategy. Structural — nothing needs to subclass this."""

    name: str

    async def decide_price(self, observation: Observation) -> float:
        """Return the price to charge this round."""
        ...
