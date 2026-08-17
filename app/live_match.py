"""Streamlit live-match demo: watch a tournament play out round by round.

Usage:
    streamlit run app/live_match.py

Defaults to an all-scripted-bot roster — free, instant, unlimited reruns. Opting
into a real LLM vendor makes real, billed OpenRouter calls every round; off by
default, and the sidebar says so before you can turn it on.
"""

from __future__ import annotations

import asyncio

import pandas as pd
import streamlit as st

from pricewars.agents.scripted import ConstantMarkupBot, RandomBot, TitForTatBot, UndercutterBot
from pricewars.market import MarketConfig, solve_references
from pricewars.tournament import MatchConfig, RoundLog, run_match

st.set_page_config(page_title="AI Price Wars — live match", layout="wide")
st.title("🍅 AI Price Wars — live match")
st.caption(
    "Six vendors, same brief, same costs, same customers. Prices update round by round as the match runs."
)

with st.sidebar:
    st.header("Match setup")
    n_rounds = st.slider("Rounds", 1, 30, 10)
    seed = st.number_input("Seed", value=1, step=1)

    st.divider()
    use_llm = st.checkbox("Include one real LLM vendor", value=False)
    model_id = "anthropic/claude-sonnet-4.5"
    if use_llm:
        model_id = st.text_input("OpenRouter model id", value=model_id)
        st.warning(
            "Each round this vendor plays makes real, billed OpenRouter API calls. "
            "Check current model ids/pricing at openrouter.ai/models before running.",
            icon="⚠️",
        )

    run_clicked = st.button("▶ Run match", type="primary", use_container_width=True)

config = MarketConfig()
references = solve_references(config)

st.caption(
    f"Reference prices for this market: competitive ≈ **\\${references.competitive:.2f}**, "
    f"monopoly ≈ **\\${references.monopoly:.2f}** (cost \\${config.cost:.2f}, {config.n_vendors} vendors)."
)

status_placeholder = st.empty()
progress_placeholder = st.empty()
col_price, col_temp = st.columns([2, 1])
price_chart_placeholder = col_price.empty()
temp_chart_placeholder = col_temp.empty()
standings_placeholder = st.empty()
compliance_placeholder = st.empty()


def build_roster(use_llm: bool, model_id: str) -> list:
    bots = [
        UndercutterBot(undercut=0.10),
        UndercutterBot(undercut=0.50),
        TitForTatBot(),
        ConstantMarkupBot(markup=1.5),
        ConstantMarkupBot(markup=4.0),
    ]
    if use_llm:
        from pricewars.agents.providers import build_openrouter_vendor

        llm_vendor = build_openrouter_vendor(model_id, max_tool_calls=8)
        return [llm_vendor, *bots]
    return [*bots, RandomBot(seed=7)]


def run_live_match(n_rounds: int, seed: int, use_llm: bool, model_id: str) -> None:
    try:
        roster = build_roster(use_llm, model_id)
    except RuntimeError as e:
        status_placeholder.error(str(e))
        return

    match_config = MatchConfig(n_rounds=n_rounds, seed=seed)
    price_rows: list[dict] = []
    temp_rows: list[dict] = []
    llm_vendor = roster[0] if use_llm else None

    def market_temperature_of(round_logs: tuple[RoundLog, ...]) -> float:
        from pricewars.market import market_temperature

        avg_price = sum(r.price_clamped for r in round_logs) / len(round_logs)
        return market_temperature(avg_price, references)

    def on_round(round_num: int, round_logs: tuple[RoundLog, ...]) -> None:
        progress_placeholder.progress(round_num / n_rounds, text=f"Round {round_num} / {n_rounds}")

        for log in round_logs:
            price_rows.append(
                {"round": round_num, "vendor": f"{log.vendor_label} ({log.vendor_name})", "price": log.price_clamped}
            )
        price_df = pd.DataFrame(price_rows).pivot(index="round", columns="vendor", values="price")
        price_chart_placeholder.line_chart(price_df, height=360)

        temp_rows.append({"round": round_num, "temperature": market_temperature_of(round_logs)})
        temp_df = pd.DataFrame(temp_rows).set_index("round")
        temp_chart_placeholder.line_chart(temp_df, height=360, y_label="market temperature")

        standings = sorted(round_logs, key=lambda r: -r.cumulative_profit)
        standings_df = pd.DataFrame(
            [
                {
                    "vendor": f"{s.vendor_label} ({s.vendor_name})",
                    "price this round": f"${s.price_clamped:.2f}",
                    "profit this round": f"${s.profit:.2f}",
                    "cumulative profit": f"${s.cumulative_profit:.2f}",
                    "flag": "⚠️ compliance fallback"
                    if llm_vendor is not None
                    and s.vendor_name == llm_vendor.name
                    and any(f.round_num == round_num for f in llm_vendor.compliance_log)
                    else "",
                }
                for s in standings
            ]
        )
        standings_placeholder.dataframe(standings_df, use_container_width=True, hide_index=True)

        if llm_vendor is not None and llm_vendor.compliance_log:
            compliance_placeholder.warning(
                f"{llm_vendor.name}: {len(llm_vendor.compliance_log)} compliance failure(s) so far — "
                "see the fallback flags in the table above.",
                icon="⚠️",
            )

    status_placeholder.info("Running...")
    asyncio.run(run_match(roster, config, match_config, on_round_complete=on_round))
    status_placeholder.success(f"Match complete — {n_rounds} rounds played.")


if run_clicked:
    run_live_match(n_rounds, int(seed), use_llm, model_id)
else:
    status_placeholder.info("Configure a match in the sidebar and click **Run match**.")
