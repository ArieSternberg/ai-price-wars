"""Metrics computed from a match's round log.

Starts minimal — just market temperature over time, enough for the first
price-path chart. Grows with the eval suite in later phases (convergence
statistics, price-war detection, recovery speed — see PLAN.md §5).
"""

from __future__ import annotations

import pandas as pd

from pricewars.market import MarketReferences, market_temperature

__all__ = ["temperature_by_round"]


def temperature_by_round(rounds: pd.DataFrame, references: MarketReferences) -> pd.DataFrame:
    """Mean price across vendors per round, converted to the 0-1 temperature index.

    `rounds` is a match's round log (see `tournament.RoundLog`): one row per
    (round, vendor), with a `price_clamped` column.
    """
    avg_price = rounds.groupby("round_num")["price_clamped"].mean()
    temperature = avg_price.map(lambda p: market_temperature(p, references))
    return pd.DataFrame(
        {
            "round_num": avg_price.index,
            "avg_price": avg_price.values,
            "temperature": temperature.values,
        }
    ).reset_index(drop=True)
