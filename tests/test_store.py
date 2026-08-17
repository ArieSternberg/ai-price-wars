"""Tests for pricewars/store.py."""

from __future__ import annotations

import asyncio

from pricewars.agents.scripted import ConstantMarkupBot, RandomBot, TitForTatBot, UndercutterBot
from pricewars.market import MarketConfig
from pricewars.store import load_match, save_match
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


def test_save_and_load_roundtrip(tmp_path):
    config = MarketConfig()
    result = asyncio.run(run_match(default_roster(), config, MatchConfig(n_rounds=5, seed=1)))

    save_match(result, "test_run", results_dir=tmp_path)
    rounds, meta = load_match("test_run", results_dir=tmp_path)

    assert len(rounds) == len(result.rounds)
    assert set(rounds.columns) == set(result.rounds.columns)
    assert meta["run_id"] == "test_run"
    assert meta["market_config"]["mu"] == config.mu
    assert meta["match_config"]["seed"] == 1
    assert meta["labels"] == result.labels


def test_save_creates_expected_files(tmp_path):
    config = MarketConfig()
    result = asyncio.run(run_match(default_roster(), config, MatchConfig(n_rounds=3, seed=2)))
    match_dir = save_match(result, "another_run", results_dir=tmp_path)
    assert (match_dir / "rounds.parquet").exists()
    assert (match_dir / "meta.json").exists()
