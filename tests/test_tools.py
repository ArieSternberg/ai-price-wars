"""Tests for pricewars/tools.py. No API calls — these are pure functions wrapped
as LangChain tools, tested by invoking them directly."""

from __future__ import annotations

import pytest

from pricewars.agents.base import Observation, RivalObservation
from pricewars.market import MarketConfig, expected_profits, expected_sales
from pricewars.tools import ToolCallLog, build_tools


def make_observation(
    config: MarketConfig,
    round_num: int = 4,
    own_history: tuple[float, ...] = (5.0, 4.8, 4.6),
    rival_table: tuple[RivalObservation, ...] = (
        RivalObservation("Vendor B", 4.0),
        RivalObservation("Vendor C", 6.0),
    ),
    rival_price_history: dict[str, tuple[float, ...]] | None = None,
) -> Observation:
    if rival_price_history is None:
        rival_price_history = {
            "Vendor B": (5.0, 4.5, 4.0),
            "Vendor C": (5.0, 6.5, 6.0),
        }
    return Observation(
        round_num=round_num,
        n_rounds=30,
        own_label="Vendor A",
        own_price_history=own_history,
        own_profit_history=(20.0, 18.0, 16.0),
        rival_table=rival_table,
        rival_price_history=rival_price_history,
        config=config,
    )


def get_tool(tools, name: str):
    return next(t for t in tools if t.name == name)


class TestGetPriceHistory:
    def test_own_history(self):
        config = MarketConfig()
        obs = make_observation(config)
        call_log: list[ToolCallLog] = []
        tools, _ = build_tools(obs, call_log)
        result = get_tool(tools, "get_price_history").invoke(
            {"vendor_label": "Vendor A", "n_rounds": 15}
        )
        assert "round 1: $5.00" in result
        assert "round 2: $4.80" in result
        assert "round 3: $4.60" in result

    def test_rival_history(self):
        config = MarketConfig()
        obs = make_observation(config)
        call_log: list[ToolCallLog] = []
        tools, _ = build_tools(obs, call_log)
        result = get_tool(tools, "get_price_history").invoke(
            {"vendor_label": "Vendor B", "n_rounds": 15}
        )
        assert "round 1: $5.00" in result
        assert "round 3: $4.00" in result

    def test_respects_n_rounds(self):
        config = MarketConfig()
        obs = make_observation(config)
        call_log: list[ToolCallLog] = []
        tools, _ = build_tools(obs, call_log)
        result = get_tool(tools, "get_price_history").invoke(
            {"vendor_label": "Vendor A", "n_rounds": 1}
        )
        assert "round 3: $4.60" in result
        assert "round 1" not in result

    def test_unknown_label(self):
        config = MarketConfig()
        obs = make_observation(config)
        call_log: list[ToolCallLog] = []
        tools, _ = build_tools(obs, call_log)
        result = get_tool(tools, "get_price_history").invoke(
            {"vendor_label": "Vendor Z", "n_rounds": 15}
        )
        assert "Unknown vendor label" in result
        assert "Vendor A" in result and "Vendor B" in result

    def test_round_one_has_no_history(self):
        config = MarketConfig()
        obs = make_observation(config, round_num=1, own_history=(), rival_table=(), rival_price_history={})
        call_log: list[ToolCallLog] = []
        tools, _ = build_tools(obs, call_log)
        result = get_tool(tools, "get_price_history").invoke(
            {"vendor_label": "Vendor A", "n_rounds": 15}
        )
        assert "no price history yet" in result

    def test_logs_the_call(self):
        config = MarketConfig()
        obs = make_observation(config)
        call_log: list[ToolCallLog] = []
        tools, _ = build_tools(obs, call_log)
        get_tool(tools, "get_price_history").invoke({"vendor_label": "Vendor A", "n_rounds": 5})
        assert len(call_log) == 1
        assert call_log[0].tool_name == "get_price_history"
        assert call_log[0].args == {"vendor_label": "Vendor A", "n_rounds": 5}


class TestGetMarketStats:
    def test_reports_avg_and_range(self):
        config = MarketConfig()
        obs = make_observation(config)  # rivals last round: 4.0, 6.0
        call_log: list[ToolCallLog] = []
        tools, _ = build_tools(obs, call_log)
        result = get_tool(tools, "get_market_stats").invoke({})
        assert "$5.00" in result  # average
        assert "$4.00-$6.00" in result

    def test_identifies_undercutters(self):
        config = MarketConfig()
        # Vendor B cut price (5.0 -> 4.5 -> 4.0), Vendor C raised then held (5.0 -> 6.5 -> 6.0 is still a cut on the last step)
        obs = make_observation(
            config,
            rival_price_history={
                "Vendor B": (5.0, 4.5, 4.0),  # cut last step
                "Vendor C": (5.0, 5.5, 6.0),  # raised last step
            },
        )
        call_log: list[ToolCallLog] = []
        tools, _ = build_tools(obs, call_log)
        result = get_tool(tools, "get_market_stats").invoke({})
        assert "Vendor B" in result
        assert "Vendor C" not in result.split("cut their price")[1]

    def test_round_one_has_no_stats(self):
        config = MarketConfig()
        obs = make_observation(config, round_num=1, rival_table=(), rival_price_history={})
        call_log: list[ToolCallLog] = []
        tools, _ = build_tools(obs, call_log)
        result = get_tool(tools, "get_market_stats").invoke({})
        assert "round 1" in result.lower()


class TestSimulatePrice:
    def test_matches_market_module_directly(self):
        # rival_table must have exactly config.n_vendors - 1 entries, matching how the
        # tournament loop always populates it from round 2 onward.
        config = MarketConfig(n_vendors=3)
        obs = make_observation(
            config,
            rival_table=(RivalObservation("Vendor B", 4.0), RivalObservation("Vendor C", 6.0)),
        )
        call_log: list[ToolCallLog] = []
        tools, _ = build_tools(obs, call_log)
        result = get_tool(tools, "simulate_price").invoke({"price": 5.0})

        full_prices = [5.0, 4.0, 6.0]
        expected_profit = float(expected_profits(full_prices, config)[0])
        expected_units = float(expected_sales(full_prices, config)[0])
        assert f"${expected_profit:.2f}" in result
        assert f"{expected_units:.1f}" in result

    def test_round_one_cannot_simulate(self):
        config = MarketConfig()
        obs = make_observation(config, round_num=1, rival_table=(), rival_price_history={})
        call_log: list[ToolCallLog] = []
        tools, _ = build_tools(obs, call_log)
        result = get_tool(tools, "simulate_price").invoke({"price": 5.0})
        assert "Can't simulate" in result

    def test_mismatched_rival_table_length_fails_soft(self):
        """If rival_table doesn't have exactly n_vendors - 1 entries (shouldn't happen
        in a real match, but tools must never crash the agent loop on a bad input),
        return a clear message instead of propagating market.py's ValueError."""
        config = MarketConfig()  # n_vendors=6, but the default fixture only has 2 rivals
        obs = make_observation(config)
        call_log: list[ToolCallLog] = []
        tools, _ = build_tools(obs, call_log)
        result = get_tool(tools, "simulate_price").invoke({"price": 5.0})
        assert "Can't simulate" in result
        assert "5" in result  # expected 5 rivals for a 6-vendor market


class TestSetPrice:
    def test_commits_price(self):
        config = MarketConfig()
        obs = make_observation(config)
        call_log: list[ToolCallLog] = []
        tools, committed = build_tools(obs, call_log)
        assert committed["price"] is None
        get_tool(tools, "set_price").invoke({"price": 5.5})
        assert committed["price"] == 5.5

    def test_logs_the_commit(self):
        config = MarketConfig()
        obs = make_observation(config)
        call_log: list[ToolCallLog] = []
        tools, _ = build_tools(obs, call_log)
        get_tool(tools, "set_price").invoke({"price": 5.5})
        assert any(c.tool_name == "set_price" and c.args == {"price": 5.5} for c in call_log)


def test_all_four_tools_present():
    config = MarketConfig()
    obs = make_observation(config)
    tools, _ = build_tools(obs, [])
    names = {t.name for t in tools}
    assert names == {"get_price_history", "get_market_stats", "simulate_price", "set_price"}
