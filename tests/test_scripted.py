"""Tests for pricewars/agents/scripted.py."""

from __future__ import annotations

import asyncio

import pytest

from pricewars.agents.base import Observation, RivalObservation
from pricewars.agents.scripted import ConstantMarkupBot, RandomBot, TitForTatBot, UndercutterBot
from pricewars.market import MarketConfig


def make_observation(config: MarketConfig, rival_table: tuple[RivalObservation, ...] = ()) -> Observation:
    return Observation(
        round_num=2,
        n_rounds=30,
        own_label="Vendor A",
        own_price_history=(5.0,),
        own_profit_history=(50.0,),
        rival_table=rival_table,
        rival_price_history={},
        config=config,
    )


def run(coro):
    return asyncio.run(coro)


class TestUndercutterBot:
    def test_opening_price_with_no_history(self):
        config = MarketConfig()
        bot = UndercutterBot()
        price = run(bot.decide_price(make_observation(config)))
        assert config.cost < price < config.price_cap

    def test_undercuts_cheapest_rival(self):
        config = MarketConfig()
        bot = UndercutterBot(undercut=0.10)
        rivals = (RivalObservation("Vendor B", 6.0), RivalObservation("Vendor C", 5.0))
        price = run(bot.decide_price(make_observation(config, rivals)))
        assert price == pytest.approx(4.90)

    def test_never_prices_below_cost(self):
        config = MarketConfig()
        bot = UndercutterBot(undercut=1.0)
        rivals = (RivalObservation("Vendor B", config.cost + 0.5),)
        price = run(bot.decide_price(make_observation(config, rivals)))
        assert price == config.cost


class TestTitForTatBot:
    def test_opening_price_with_no_history(self):
        config = MarketConfig()
        bot = TitForTatBot()
        price = run(bot.decide_price(make_observation(config)))
        assert config.cost < price < config.price_cap

    def test_matches_average_rival_price(self):
        config = MarketConfig()
        bot = TitForTatBot()
        rivals = (
            RivalObservation("Vendor B", 4.0),
            RivalObservation("Vendor C", 6.0),
            RivalObservation("Vendor D", 5.0),
        )
        price = run(bot.decide_price(make_observation(config, rivals)))
        assert price == pytest.approx(5.0)


class TestConstantMarkupBot:
    def test_ignores_rivals(self):
        config = MarketConfig()
        bot = ConstantMarkupBot(markup=2.5)
        rivals = (RivalObservation("Vendor B", 3.5),)
        price_no_rivals = run(bot.decide_price(make_observation(config)))
        price_with_rivals = run(bot.decide_price(make_observation(config, rivals)))
        assert price_no_rivals == price_with_rivals == config.cost + 2.5


class TestRandomBot:
    def test_within_bounds(self):
        config = MarketConfig()
        bot = RandomBot(seed=1)
        for _ in range(50):
            price = run(bot.decide_price(make_observation(config)))
            assert config.cost <= price <= config.price_cap

    def test_seeded_reproducibility(self):
        config = MarketConfig()
        bot_a = RandomBot(seed=123)
        bot_b = RandomBot(seed=123)
        prices_a = [run(bot_a.decide_price(make_observation(config))) for _ in range(10)]
        prices_b = [run(bot_b.decide_price(make_observation(config))) for _ in range(10)]
        assert prices_a == prices_b

    def test_different_seeds_diverge(self):
        config = MarketConfig()
        bot_a = RandomBot(seed=1)
        bot_b = RandomBot(seed=2)
        prices_a = [run(bot_a.decide_price(make_observation(config))) for _ in range(10)]
        prices_b = [run(bot_b.decide_price(make_observation(config))) for _ in range(10)]
        assert prices_a != prices_b
