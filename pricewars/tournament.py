"""Round-by-round match orchestration: seeding, shuffling, and score-keeping.

Runs any list of `Vendor` implementations — scripted bots now, LLM agents later,
same protocol — through a fixed number of rounds against `pricewars.market`'s
demand model. Every vendor decides its price concurrently each round (this is the
"async orchestration" PLAN.md calls out — it matters once decisions are real API
calls, but costs nothing to build in now while the vendors are scripted bots).
"""

from __future__ import annotations

import asyncio
import dataclasses
import random
from dataclasses import dataclass

import pandas as pd

from pricewars.agents.base import Observation, RivalObservation, Vendor
from pricewars.market import MarketConfig, MarketReferences, simulate_round, solve_references

__all__ = ["MatchConfig", "RoundLog", "MatchResult", "run_match"]


@dataclass(frozen=True)
class MatchConfig:
    """Parameters of one match that aren't about the market itself."""

    n_rounds: int = 30
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_rounds < 1:
            raise ValueError(f"n_rounds must be at least 1, got {self.n_rounds}")


@dataclass(frozen=True)
class RoundLog:
    """One row of the match log: one vendor's outcome in one round."""

    round_num: int
    vendor_label: str
    vendor_name: str
    price_submitted: float
    price_clamped: float
    was_out_of_range: bool
    sold: int
    profit: float
    cumulative_profit: float


@dataclass(frozen=True)
class MatchResult:
    rounds: pd.DataFrame  # one row per (round, vendor) — see RoundLog fields
    labels: dict[str, str]  # stable vendor_label -> vendor_name for the whole match
    market_config: MarketConfig
    match_config: MatchConfig
    references: MarketReferences


def _assign_labels(n_vendors: int, rng: random.Random) -> list[str]:
    """One random label per vendor, fixed for the whole match — see Observation's docstring."""
    letters = [f"Vendor {chr(ord('A') + i)}" for i in range(n_vendors)]
    rng.shuffle(letters)
    return letters


async def run_match(
    vendors: list[Vendor],
    market_config: MarketConfig,
    match_config: MatchConfig | None = None,
) -> MatchResult:
    """Play one full match and return its round-by-round log.

    `vendors[i]` is permanently paired with the i-th assigned label for the whole
    match — that pairing is what "stable label" means. What changes every round is
    only the *order* rivals are listed in each vendor's `rival_table`.
    """
    if match_config is None:
        match_config = MatchConfig()
    if len(vendors) != market_config.n_vendors:
        raise ValueError(
            f"got {len(vendors)} vendors but market_config.n_vendors={market_config.n_vendors}"
        )

    rng = random.Random(match_config.seed)
    labels = _assign_labels(market_config.n_vendors, rng)
    references = solve_references(market_config)

    price_history: dict[str, list[float]] = {label: [] for label in labels}
    profit_history: dict[str, list[float]] = {label: [] for label in labels}
    cumulative_profit: dict[str, float] = {label: 0.0 for label in labels}
    logs: list[RoundLog] = []

    for round_num in range(1, match_config.n_rounds + 1):
        observations = []
        for label in labels:
            rival_labels = [l for l in labels if l != label]
            rng.shuffle(rival_labels)  # fresh row order every round; identity stays put
            rival_table = tuple(
                RivalObservation(label=rl, price=price_history[rl][-1])
                for rl in rival_labels
                if price_history[rl]  # empty in round 1 — no history to show yet
            )
            observations.append(
                Observation(
                    round_num=round_num,
                    n_rounds=match_config.n_rounds,
                    own_label=label,
                    own_price_history=tuple(price_history[label]),
                    own_profit_history=tuple(profit_history[label]),
                    rival_table=rival_table,
                    rival_price_history={l: tuple(price_history[l]) for l in rival_labels},
                    config=market_config,
                )
            )

        raw_prices = await asyncio.gather(
            *(vendor.decide_price(obs) for vendor, obs in zip(vendors, observations))
        )

        # Harness-enforced rule, not prompt-enforced: clamp to [cost, price_cap] and
        # log every clamp as a compliance event. Never silently retried.
        clamped_prices = [
            min(max(float(p), market_config.cost), market_config.price_cap) for p in raw_prices
        ]

        result = simulate_round(clamped_prices, market_config)

        for i, label in enumerate(labels):
            profit = float(result.profits[i])
            cumulative_profit[label] += profit
            logs.append(
                RoundLog(
                    round_num=round_num,
                    vendor_label=label,
                    vendor_name=vendors[i].name,
                    price_submitted=float(raw_prices[i]),
                    price_clamped=float(clamped_prices[i]),
                    was_out_of_range=float(raw_prices[i]) != clamped_prices[i],
                    sold=int(result.sold[i]),
                    profit=profit,
                    cumulative_profit=cumulative_profit[label],
                )
            )
            price_history[label].append(clamped_prices[i])
            profit_history[label].append(profit)

    rounds_df = pd.DataFrame([dataclasses.asdict(r) for r in logs])
    return MatchResult(
        rounds=rounds_df,
        labels={label: vendor.name for label, vendor in zip(labels, vendors)},
        market_config=market_config,
        match_config=match_config,
        references=references,
    )
