"""Tests for pricewars/tournament.py."""

from __future__ import annotations

import asyncio

import pytest

from pricewars.agents.base import Observation
from pricewars.agents.scripted import ConstantMarkupBot, RandomBot, TitForTatBot, UndercutterBot
from pricewars.market import MarketConfig
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


def run(coro):
    return asyncio.run(coro)


class FixedPriceBot:
    """A bot that always returns a hardcoded price — useful for probing the
    clamp/compliance-logging behavior directly."""

    def __init__(self, price: float, name: str = "fixed"):
        self.price = price
        self.name = name

    async def decide_price(self, observation: Observation) -> float:
        return self.price


class TestRunMatch:
    def test_produces_one_row_per_round_per_vendor(self):
        config = MarketConfig()
        match_config = MatchConfig(n_rounds=10, seed=1)
        result = run(run_match(default_roster(), config, match_config))
        assert len(result.rounds) == 10 * config.n_vendors

    def test_rejects_vendor_count_mismatch(self):
        config = MarketConfig()
        with pytest.raises(ValueError):
            run(run_match(default_roster()[:5], config, MatchConfig(n_rounds=5)))

    def test_labels_are_stable_across_rounds(self):
        """Each vendor_label maps to exactly one vendor_name for the whole match."""
        config = MarketConfig()
        result = run(run_match(default_roster(), config, MatchConfig(n_rounds=10, seed=1)))
        for label, group in result.rounds.groupby("vendor_label"):
            assert group["vendor_name"].nunique() == 1

    def test_every_vendor_gets_a_label(self):
        config = MarketConfig()
        result = run(run_match(default_roster(), config, MatchConfig(n_rounds=5, seed=1)))
        assert result.rounds["vendor_label"].nunique() == config.n_vendors
        assert len(result.labels) == config.n_vendors

    def test_prices_are_clamped_to_bounds(self):
        config = MarketConfig()
        roster = default_roster()
        result = run(run_match(roster, config, MatchConfig(n_rounds=15, seed=1)))
        assert result.rounds["price_clamped"].between(config.cost, config.price_cap).all()

    def test_cumulative_profit_is_nondecreasing(self):
        """None of the default bots ever price below cost, so profit is always >= 0."""
        config = MarketConfig()
        result = run(run_match(default_roster(), config, MatchConfig(n_rounds=15, seed=1)))
        for label, group in result.rounds.sort_values("round_num").groupby("vendor_label"):
            assert group["cumulative_profit"].is_monotonic_increasing

    def test_same_seed_is_reproducible(self):
        config = MarketConfig()
        result_a = run(run_match(default_roster(), config, MatchConfig(n_rounds=10, seed=99)))
        result_b = run(run_match(default_roster(), config, MatchConfig(n_rounds=10, seed=99)))
        assert result_a.rounds.equals(result_b.rounds)
        assert result_a.labels == result_b.labels

    def test_different_seeds_shuffle_labels_differently(self):
        config = MarketConfig()
        result_a = run(run_match(default_roster(), config, MatchConfig(n_rounds=3, seed=1)))
        result_b = run(run_match(default_roster(), config, MatchConfig(n_rounds=3, seed=2)))
        assert result_a.labels != result_b.labels

    def test_out_of_range_price_is_clamped_and_logged(self):
        config = MarketConfig()
        roster = [FixedPriceBot(config.price_cap + 5.0, name="over_cap")] + default_roster()[1:]
        result = run(run_match(roster, config, MatchConfig(n_rounds=3, seed=1)))
        over_cap_rows = result.rounds[result.rounds["vendor_name"] == "over_cap"]
        assert (over_cap_rows["price_clamped"] == config.price_cap).all()
        assert over_cap_rows["was_out_of_range"].all()

    def test_in_range_price_is_not_flagged(self):
        config = MarketConfig()
        roster = [FixedPriceBot(5.0, name="steady")] + default_roster()[1:]
        result = run(run_match(roster, config, MatchConfig(n_rounds=3, seed=1)))
        steady_rows = result.rounds[result.rounds["vendor_name"] == "steady"]
        assert not steady_rows["was_out_of_range"].any()

    def test_round_one_has_no_rival_history_influence(self):
        """Round 1 opening prices should be identical for identical strategies —
        no rival price exists yet to differentiate them."""
        config = MarketConfig()
        roster = [ConstantMarkupBot(markup=1.0) for _ in range(config.n_vendors)]
        result = run(run_match(roster, config, MatchConfig(n_rounds=1, seed=1)))
        assert (result.rounds["price_clamped"] == config.cost + 1.0).all()

    def test_references_are_solved_for_the_match_config(self):
        config = MarketConfig()
        result = run(run_match(default_roster(), config, MatchConfig(n_rounds=3, seed=1)))
        assert result.references.monopoly > result.references.competitive
