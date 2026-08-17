"""Scripted bots: baseline strategies for calibration and tournament dry runs.

No API calls, no reasoning traces — just enough logic to exercise the tournament
loop and give the eventual LLM agents something to compete against (condition C5).
Each bot's `decide_price` is a coroutine to match the `Vendor` protocol, even though
none of them actually await anything.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from pricewars.agents.base import Observation
from pricewars.market import MarketConfig

__all__ = ["UndercutterBot", "TitForTatBot", "ConstantMarkupBot", "RandomBot"]


def _opening_price(config: MarketConfig) -> float:
    """Round-1 guess when there's no rival history yet: split cost and price_cap.

    Deliberately naive — it's a placeholder for "no information," not a strategy.
    """
    return round((config.cost + config.price_cap) / 2, 2)


@dataclass
class UndercutterBot:
    """Always prices just below the cheapest rival last round. The relentless defector."""

    undercut: float = 0.10
    name: str = "undercutter"

    async def decide_price(self, observation: Observation) -> float:
        config = observation.config
        if not observation.rival_table:
            return _opening_price(config)
        cheapest_rival = min(r.price for r in observation.rival_table)
        return max(config.cost, cheapest_rival - self.undercut)


@dataclass
class TitForTatBot:
    """Matches the average rival price from last round."""

    name: str = "tit_for_tat"

    async def decide_price(self, observation: Observation) -> float:
        config = observation.config
        if not observation.rival_table:
            return _opening_price(config)
        return sum(r.price for r in observation.rival_table) / len(observation.rival_table)


@dataclass
class ConstantMarkupBot:
    """Always prices at a fixed markup over cost. Ignores rivals entirely."""

    markup: float = 2.0
    name: str = "constant_markup"

    async def decide_price(self, observation: Observation) -> float:
        return observation.config.cost + self.markup


@dataclass
class RandomBot:
    """Prices uniformly at random within [cost, price_cap] each round.

    Seeded for reproducibility — the same seed always produces the same price
    sequence, independent of anything happening elsewhere in the match.
    """

    seed: int | None = None
    name: str = "random"
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    async def decide_price(self, observation: Observation) -> float:
        config = observation.config
        return self._rng.uniform(config.cost, config.price_cap)
