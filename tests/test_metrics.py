"""Tests for pricewars/metrics.py."""

from __future__ import annotations

import asyncio

import pytest

from pricewars.agents.scripted import ConstantMarkupBot, RandomBot, TitForTatBot, UndercutterBot
from pricewars.market import MarketConfig, solve_references
from pricewars.metrics import temperature_by_round
from pricewars.tournament import MatchConfig, run_match


def default_roster() -> list:
    return [
        UndercutterBot(undercut=0.10),
        UndercutterBot(undercut=0.50),
        TitForTatBot(),
        ConstantMarkupBot(markup=1.5),
        ConstantMarkupBot(markup=4.0),
        RandomBot(seed=7),
    ]


def test_one_row_per_round():
    config = MarketConfig()
    match_config = MatchConfig(n_rounds=10, seed=1)
    result = asyncio.run(run_match(default_roster(), config, match_config))
    temp_df = temperature_by_round(result.rounds, result.references)
    assert len(temp_df) == match_config.n_rounds
    assert list(temp_df["round_num"]) == list(range(1, match_config.n_rounds + 1))


def test_temperature_within_clipped_bounds():
    config = MarketConfig()
    result = asyncio.run(run_match(default_roster(), config, MatchConfig(n_rounds=15, seed=1)))
    temp_df = temperature_by_round(result.rounds, result.references)
    assert temp_df["temperature"].between(0.0, 1.0).all()


def test_all_vendors_at_competitive_price_gives_zero_temperature():
    config = MarketConfig()
    references = solve_references(config)
    roster = [ConstantMarkupBot(markup=references.competitive - config.cost) for _ in range(config.n_vendors)]
    result = asyncio.run(run_match(roster, config, MatchConfig(n_rounds=1, seed=1)))
    temp_df = temperature_by_round(result.rounds, result.references)
    assert temp_df["temperature"].iloc[0] == pytest.approx(0.0, abs=1e-4)


def test_all_vendors_at_monopoly_price_gives_one_temperature():
    config = MarketConfig()
    references = solve_references(config)
    roster = [ConstantMarkupBot(markup=references.monopoly - config.cost) for _ in range(config.n_vendors)]
    result = asyncio.run(run_match(roster, config, MatchConfig(n_rounds=1, seed=1)))
    temp_df = temperature_by_round(result.rounds, result.references)
    assert temp_df["temperature"].iloc[0] == pytest.approx(1.0, abs=1e-4)
