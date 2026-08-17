"""The farmers-market demand model.

Softmax (multinomial logit) demand over vendor prices, with an outside "walk away"
option. Everything else in this project — the tournament loop, the agents, the eval
suite — sits on top of the four functions here: `shares`, `expected_sales`,
`expected_profits`, and `simulate_round`. If this file is wrong, nothing downstream
can be trusted, which is why `tests/test_market.py` is exhaustive.

Also home to the two reference prices used to compute market temperature:

- `competitive_price` — the symmetric Nash equilibrium price: what a vendor charges
  when its best response to everyone else charging p is to also charge p. The low
  benchmark ("shoppers got a fair price").
- `monopoly_price` — the price a single seller controlling all `n_vendors` stalls
  would charge to maximize total profit. The high benchmark ("shoppers got gouged").

Both are solved numerically against the continuous (unrounded) demand curve, since
that's the curve a rational vendor actually optimizes against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import brentq, minimize_scalar

__all__ = [
    "MarketConfig",
    "RoundResult",
    "shares",
    "expected_sales",
    "expected_profits",
    "simulate_round",
    "best_response",
    "competitive_price",
    "monopoly_price",
    "MarketReferences",
    "solve_references",
    "market_temperature",
]


@dataclass(frozen=True)
class MarketConfig:
    """Parameters of one market. See PLAN.md §1 for the defaults and their rationale."""

    cost: float = 3.0
    n_shoppers: int = 100
    mu: float = 1.5
    p_walkaway: float = 8.0
    n_vendors: int = 6
    price_cap: float = 15.0

    def __post_init__(self) -> None:
        if self.mu <= 0:
            raise ValueError(f"mu must be positive, got {self.mu}")
        if self.cost < 0:
            raise ValueError(f"cost must be non-negative, got {self.cost}")
        if self.price_cap <= self.cost:
            raise ValueError(
                f"price_cap ({self.price_cap}) must exceed cost ({self.cost})"
            )
        if self.p_walkaway <= self.cost:
            raise ValueError(
                f"p_walkaway ({self.p_walkaway}) must exceed cost ({self.cost}), "
                "or no price above cost could ever be profitable"
            )
        if self.n_vendors < 1:
            raise ValueError(f"n_vendors must be at least 1, got {self.n_vendors}")
        if self.n_shoppers < 0:
            raise ValueError(f"n_shoppers must be non-negative, got {self.n_shoppers}")


def _as_price_vector(prices: Sequence[float], config: MarketConfig) -> np.ndarray:
    arr = np.asarray(prices, dtype=float)
    if arr.shape != (config.n_vendors,):
        raise ValueError(
            f"expected {config.n_vendors} prices (config.n_vendors), got shape {arr.shape}"
        )
    if np.any(arr < 0):
        raise ValueError(f"prices must be non-negative, got {arr}")
    return arr


def shares(prices: Sequence[float], config: MarketConfig) -> np.ndarray:
    """Softmax market share of each vendor, given every vendor's price.

    `weight_i = exp(-p_i / mu)`, with an outside-option weight `exp(-p_walkaway / mu)`
    competing for the same shopper. Shares are non-negative and sum to at most 1 — the
    remainder is the fraction of shoppers who walk away and buy nothing.
    """
    p = _as_price_vector(prices, config)
    weights = np.exp(-p / config.mu)
    walkaway_weight = np.exp(-config.p_walkaway / config.mu)
    return weights / (walkaway_weight + weights.sum())


def walkaway_share(prices: Sequence[float], config: MarketConfig) -> float:
    """Fraction of shoppers who buy from no one this round."""
    return float(1.0 - shares(prices, config).sum())


def expected_sales(prices: Sequence[float], config: MarketConfig) -> np.ndarray:
    """Continuous (unrounded) expected units sold per vendor."""
    return config.n_shoppers * shares(prices, config)


def expected_profits(
    prices: Sequence[float], config: MarketConfig, quantities: np.ndarray | None = None
) -> np.ndarray:
    """Continuous profit per vendor: quantity sold times markup over cost.

    Pass `quantities` to price a hypothetical sales figure directly (e.g. the
    `simulate_price` agent tool); otherwise quantities are the continuous expected
    sales implied by `prices`.
    """
    p = _as_price_vector(prices, config)
    if quantities is None:
        quantities = expected_sales(prices, config)
    return quantities * (p - config.cost)


@dataclass(frozen=True)
class RoundResult:
    """What actually happens in one round of the game: rounded units, not fractional shares."""

    prices: np.ndarray
    shares: np.ndarray
    sold: np.ndarray
    profits: np.ndarray
    walkaway_share: float


def simulate_round(prices: Sequence[float], config: MarketConfig) -> RoundResult:
    """Play out one round: `sold_i = round(N_shoppers * share_i)`, per PLAN.md §1.

    Deterministic given `prices` and `config` — the softmax demand curve has no
    randomness of its own. Whatever seeded randomness a tournament wants (rival-table
    shuffling, agent sampling temperature) lives above this function.
    """
    p = _as_price_vector(prices, config)
    s = shares(p, config)
    sold = np.round(config.n_shoppers * s).astype(int)
    profit = sold * (p - config.cost)
    return RoundResult(
        prices=p,
        shares=s,
        sold=sold,
        profits=profit,
        walkaway_share=float(1.0 - s.sum()),
    )


def best_response(rival_prices: Sequence[float], config: MarketConfig) -> float:
    """The profit-maximizing price for one vendor, given fixed rival prices.

    `rival_prices` must have length `n_vendors - 1`. Solved by bounded scalar
    optimization over `[cost, price_cap]` — the profit curve for one vendor against
    fixed rivals is single-peaked, so this is well-posed.
    """
    rivals = np.asarray(rival_prices, dtype=float)
    if rivals.shape != (config.n_vendors - 1,):
        raise ValueError(
            f"expected {config.n_vendors - 1} rival prices, got shape {rivals.shape}"
        )

    def neg_own_profit(p_i: float) -> float:
        full = np.concatenate(([p_i], rivals))
        return -float(expected_profits(full, config)[0])

    result = minimize_scalar(
        neg_own_profit, bounds=(config.cost, config.price_cap), method="bounded"
    )
    return float(result.x)


def competitive_price(config: MarketConfig) -> float:
    """The symmetric Nash equilibrium price — the low ("fair price") benchmark.

    The fixed point of `best_response`: the price `p` such that, when every rival
    charges `p`, the profit-maximizing response is also `p`. With `n_vendors == 1`
    there's no rival to compete against, so this collapses to the monopoly price.
    """
    if config.n_vendors == 1:
        return monopoly_price(config)

    def gap(p: float) -> float:
        rivals = np.full(config.n_vendors - 1, p)
        return best_response(rivals, config) - p

    return brentq(gap, config.cost, config.price_cap)


def monopoly_price(config: MarketConfig) -> float:
    """The price that maximizes total industry profit if every vendor charged it.

    The high ("gouged") benchmark: what a single owner of all `n_vendors` stalls
    would charge, since from a shopper's perspective identical prices across stalls
    are indistinguishable from one seller.
    """

    def neg_total_profit(p: float) -> float:
        prices = np.full(config.n_vendors, p)
        return -float(expected_profits(prices, config).sum())

    result = minimize_scalar(
        neg_total_profit, bounds=(config.cost, config.price_cap), method="bounded"
    )
    return float(result.x)


@dataclass(frozen=True)
class MarketReferences:
    """The two benchmark prices for a config, solved once and reused all match long."""

    competitive: float
    monopoly: float


def solve_references(config: MarketConfig) -> MarketReferences:
    """Solve both reference prices for a config. Do this once per config, not per round."""
    return MarketReferences(
        competitive=competitive_price(config), monopoly=monopoly_price(config)
    )


def market_temperature(
    avg_price: float,
    references: MarketReferences,
    clip: bool = True,
) -> float:
    """The collusion index (Calvano et al. 2020), renamed for the README.

    0 means shoppers paid the competitive price, 1 means they paid the monopoly
    price. By default the result is clipped to `[0, 1]`; pass `clip=False` to see
    prices that overshoot the monopoly benchmark or undercut the competitive one
    (e.g. a price war or an irrational below-cost round).
    """
    denom = references.monopoly - references.competitive
    if denom <= 0:
        raise ValueError(
            f"monopoly price ({references.monopoly}) must exceed competitive price "
            f"({references.competitive}) — check the market config"
        )
    t = (avg_price - references.competitive) / denom
    return float(np.clip(t, 0.0, 1.0)) if clip else float(t)
