"""Correctness tests for pricewars/market.py.

Per CLAUDE.md and PLAN.md §7: nobody should trust a leaderboard built on an
unvalidated market model. This suite is load-bearing — it runs before any
tournament, agent, or paid API code exists.
"""

from __future__ import annotations

import numpy as np
import pytest

from pricewars.market import (
    MarketConfig,
    MarketReferences,
    best_response,
    competitive_price,
    expected_profits,
    expected_sales,
    market_temperature,
    monopoly_price,
    shares,
    simulate_round,
    solve_references,
    walkaway_share,
)


# ---------------------------------------------------------------------------
# MarketConfig validation
# ---------------------------------------------------------------------------


class TestMarketConfig:
    def test_defaults_match_plan(self):
        cfg = MarketConfig()
        assert cfg.cost == 3.0
        assert cfg.n_shoppers == 100
        assert cfg.mu == 1.5
        assert cfg.p_walkaway == 8.0
        assert cfg.n_vendors == 6
        assert cfg.price_cap == 15.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"mu": 0},
            {"mu": -1},
            {"cost": -1},
            {"price_cap": 3.0, "cost": 3.0},
            {"price_cap": 2.0, "cost": 3.0},
            {"p_walkaway": 3.0, "cost": 3.0},
            {"p_walkaway": 2.0, "cost": 3.0},
            {"n_vendors": 0},
            {"n_vendors": -1},
            {"n_shoppers": -10},
        ],
    )
    def test_rejects_invalid_params(self, kwargs):
        with pytest.raises(ValueError):
            MarketConfig(**kwargs)

    def test_accepts_single_vendor(self):
        # n_vendors=1 is a legal degenerate market (used to sanity-check the solvers).
        MarketConfig(n_vendors=1)


# ---------------------------------------------------------------------------
# Demand: shares, sales, profits
# ---------------------------------------------------------------------------


class TestShares:
    def test_shares_plus_walkaway_sum_to_one(self):
        cfg = MarketConfig()
        rng = np.random.default_rng(0)
        for _ in range(200):
            prices = rng.uniform(cfg.cost, cfg.price_cap, size=cfg.n_vendors)
            s = shares(prices, cfg)
            assert s.sum() + walkaway_share(prices, cfg) == pytest.approx(1.0, abs=1e-9)

    def test_shares_are_nonnegative_and_at_most_one(self):
        cfg = MarketConfig()
        rng = np.random.default_rng(1)
        for _ in range(200):
            prices = rng.uniform(0, cfg.price_cap * 2, size=cfg.n_vendors)
            s = shares(prices, cfg)
            assert np.all(s >= 0)
            assert np.all(s <= 1)

    def test_lower_price_strictly_increases_own_share(self):
        """Holding rivals fixed, undercutting always wins more shoppers."""
        cfg = MarketConfig()
        rivals = [5.0, 5.0, 5.0, 6.0, 7.0]
        rng = np.random.default_rng(2)
        for _ in range(50):
            p_hi = rng.uniform(cfg.cost + 0.01, cfg.price_cap)
            p_lo = rng.uniform(cfg.cost, p_hi)
            share_hi = shares([p_hi, *rivals], cfg)[0]
            share_lo = shares([p_lo, *rivals], cfg)[0]
            assert share_lo > share_hi

    def test_symmetric_prices_give_equal_shares(self):
        cfg = MarketConfig()
        s = shares([5.0] * cfg.n_vendors, cfg)
        assert np.allclose(s, s[0])

    def test_rejects_wrong_length_price_vector(self):
        cfg = MarketConfig()
        with pytest.raises(ValueError):
            shares([5.0, 5.0], cfg)

    def test_rejects_negative_prices(self):
        cfg = MarketConfig()
        prices = [-1.0] + [5.0] * (cfg.n_vendors - 1)
        with pytest.raises(ValueError):
            shares(prices, cfg)


class TestProfits:
    def test_profit_is_zero_at_cost(self):
        cfg = MarketConfig()
        prices = [cfg.cost] * cfg.n_vendors
        assert np.allclose(expected_profits(prices, cfg), 0.0)

    def test_profit_negative_below_cost(self):
        cfg = MarketConfig()
        prices = [cfg.cost - 0.5] + [5.0] * (cfg.n_vendors - 1)
        assert expected_profits(prices, cfg)[0] < 0

    def test_expected_sales_equals_shares_times_shoppers(self):
        cfg = MarketConfig()
        prices = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        assert np.allclose(
            expected_sales(prices, cfg), shares(prices, cfg) * cfg.n_shoppers
        )


class TestSimulateRound:
    def test_sold_is_rounded_expected_sales(self):
        cfg = MarketConfig()
        prices = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
        result = simulate_round(prices, cfg)
        assert np.array_equal(
            result.sold, np.round(expected_sales(prices, cfg)).astype(int)
        )

    def test_is_deterministic(self):
        cfg = MarketConfig()
        prices = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
        a = simulate_round(prices, cfg)
        b = simulate_round(prices, cfg)
        assert np.array_equal(a.sold, b.sold)
        assert np.allclose(a.profits, b.profits)

    def test_profits_match_sold_times_markup(self):
        cfg = MarketConfig()
        prices = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
        result = simulate_round(prices, cfg)
        expected = result.sold * (result.prices - cfg.cost)
        assert np.allclose(result.profits, expected)


# ---------------------------------------------------------------------------
# Best response / reference prices
# ---------------------------------------------------------------------------


class TestBestResponse:
    def test_within_bounds(self):
        cfg = MarketConfig()
        rivals = [5.0] * (cfg.n_vendors - 1)
        br = best_response(rivals, cfg)
        assert cfg.cost <= br <= cfg.price_cap

    def test_rejects_wrong_length_rivals(self):
        cfg = MarketConfig()
        with pytest.raises(ValueError):
            best_response([5.0, 5.0], cfg)

    def test_undercutting_rivals_pulls_best_response_down(self):
        cfg = MarketConfig()
        br_high_rivals = best_response([8.0] * (cfg.n_vendors - 1), cfg)
        br_low_rivals = best_response([4.0] * (cfg.n_vendors - 1), cfg)
        assert br_low_rivals < br_high_rivals


class TestReferencePrices:
    def test_competitive_is_a_fixed_point_of_best_response(self):
        cfg = MarketConfig()
        cp = competitive_price(cfg)
        br = best_response([cp] * (cfg.n_vendors - 1), cfg)
        assert br == pytest.approx(cp, abs=1e-4)

    def test_monopoly_exceeds_competitive(self):
        cfg = MarketConfig()
        assert monopoly_price(cfg) > competitive_price(cfg)

    def test_monopoly_joint_profit_exceeds_competitive_joint_profit(self):
        cfg = MarketConfig()
        cp, mp = competitive_price(cfg), monopoly_price(cfg)
        joint_competitive = expected_profits([cp] * cfg.n_vendors, cfg).sum()
        joint_monopoly = expected_profits([mp] * cfg.n_vendors, cfg).sum()
        assert joint_monopoly > joint_competitive

    def test_both_prices_within_bounds(self):
        cfg = MarketConfig()
        cp, mp = competitive_price(cfg), monopoly_price(cfg)
        assert cfg.cost < cp < cfg.price_cap
        assert cfg.cost < mp < cfg.price_cap

    def test_single_vendor_collapses_competitive_to_monopoly(self):
        """With no rivals, 'competing hard' and 'being a monopolist' are the same thing."""
        cfg = MarketConfig(n_vendors=1)
        assert competitive_price(cfg) == pytest.approx(monopoly_price(cfg), abs=1e-6)

    def test_more_vendors_pushes_competitive_price_down(self):
        """More competition should squeeze the equilibrium price toward cost."""
        cp_2 = competitive_price(MarketConfig(n_vendors=2))
        cp_6 = competitive_price(MarketConfig(n_vendors=6))
        cp_12 = competitive_price(MarketConfig(n_vendors=12))
        assert cp_2 > cp_6 > cp_12


class TestMarketTemperature:
    def test_zero_at_competitive_price(self):
        cfg = MarketConfig()
        refs = solve_references(cfg)
        assert market_temperature(refs.competitive, refs) == pytest.approx(0.0, abs=1e-6)

    def test_one_at_monopoly_price(self):
        cfg = MarketConfig()
        refs = solve_references(cfg)
        assert market_temperature(refs.monopoly, refs) == pytest.approx(1.0, abs=1e-6)

    def test_midpoint_is_half(self):
        cfg = MarketConfig()
        refs = solve_references(cfg)
        midpoint = (refs.competitive + refs.monopoly) / 2
        assert market_temperature(midpoint, refs) == pytest.approx(0.5, abs=1e-6)

    def test_clips_by_default(self):
        refs = MarketReferences(competitive=5.0, monopoly=10.0)
        assert market_temperature(0.0, refs) == 0.0
        assert market_temperature(100.0, refs) == 1.0

    def test_unclipped_can_exceed_bounds(self):
        refs = MarketReferences(competitive=5.0, monopoly=10.0)
        assert market_temperature(0.0, refs, clip=False) < 0.0
        assert market_temperature(100.0, refs, clip=False) > 1.0

    def test_rejects_degenerate_references(self):
        refs = MarketReferences(competitive=10.0, monopoly=10.0)
        with pytest.raises(ValueError):
            market_temperature(10.0, refs)


# ---------------------------------------------------------------------------
# Golden-file regression test.
#
# These numbers were computed once from this implementation and are pinned so a
# future refactor can't silently move results out from under committed findings.
# If this test ever needs to change, the change must be deliberate and called out
# in the commit message, not an accidental side effect.
# ---------------------------------------------------------------------------


class TestGoldenValues:
    CFG = MarketConfig()
    PRICES = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

    def test_shares_golden(self):
        expected = np.array(
            [0.32257152, 0.23113260, 0.16561374, 0.11866743, 0.08502893, 0.06092589]
        )
        assert np.allclose(shares(self.PRICES, self.CFG), expected, atol=1e-7)

    def test_simulate_round_golden(self):
        result = simulate_round(self.PRICES, self.CFG)
        assert np.array_equal(result.sold, np.array([32, 23, 17, 12, 9, 6]))
        assert np.allclose(
            result.profits, np.array([16.0, 23.0, 25.5, 24.0, 22.5, 18.0])
        )

    def test_reference_prices_golden(self):
        refs = solve_references(self.CFG)
        assert refs.competitive == pytest.approx(4.793089, abs=1e-5)
        assert refs.monopoly == pytest.approx(9.029815, abs=1e-5)
