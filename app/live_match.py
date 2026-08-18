"""Streamlit live-match demo: watch a tournament play out round by round.

Usage:
    streamlit run app/live_match.py

Every seat is independently configurable — a scripted bot or a specific real LLM
model, picked per slot in the sidebar. Real models are opt-in per seat, never on by
default, and the sidebar totals up exactly how many will make billed API calls
before you hit Run.
"""

from __future__ import annotations

import asyncio

import pandas as pd
import streamlit as st

from pricewars.agents.llm import LLMVendor
from pricewars.agents.registry import MODEL_REGISTRY
from pricewars.agents.scripted import ConstantMarkupBot, RandomBot, TitForTatBot, UndercutterBot
from pricewars.market import MarketConfig, market_temperature, solve_references
from pricewars.tournament import MatchConfig, RoundLog, run_match

st.set_page_config(page_title="AI Price Wars — live match", layout="wide")
st.title("🍅 AI Price Wars — live match")
st.caption(
    "Same brief, same costs, same customers. Prices update round by round as the match runs."
)

MIN_VENDORS, MAX_VENDORS = 2, 6

# Bot slot options: key -> (label, zero-argument factory). Fresh instance per slot,
# even if the same bot type is picked twice.
BOT_OPTIONS: dict[str, tuple[str, object]] = {
    "bot_undercutter_mild": ("📜 Undercutter (mild, -$0.10)", lambda: UndercutterBot(undercut=0.10)),
    "bot_undercutter_aggressive": (
        "📜 Undercutter (aggressive, -$0.50)",
        lambda: UndercutterBot(undercut=0.50),
    ),
    "bot_tit_for_tat": ("📜 Tit-for-Tat", lambda: TitForTatBot()),
    "bot_constant_low": ("📜 Constant Markup (low, +$1.50)", lambda: ConstantMarkupBot(markup=1.5)),
    "bot_constant_high": ("📜 Constant Markup (high, +$4.00)", lambda: ConstantMarkupBot(markup=4.0)),
    "bot_random": ("📜 Random", lambda: RandomBot(seed=7)),
}

# Model slot options: key -> (label, ModelSpec). Built from the registry so adding a
# model there is the only edit needed to add it here too.
MODEL_OPTIONS: dict[str, tuple[str, object]] = {
    f"model_{spec.key}": (f"🤖 {spec.display_name} ({spec.provider})", spec) for spec in MODEL_REGISTRY
}

ALL_OPTIONS = {**BOT_OPTIONS, **MODEL_OPTIONS}
DEFAULT_SLOT_ORDER = list(BOT_OPTIONS.keys())  # matches the old all-scripted default exactly

with st.sidebar:
    st.header("Match setup")
    n_rounds = st.slider("Rounds", 1, 30, 10)
    seed = st.number_input("Seed", value=1, step=1)
    n_vendors = st.slider("Number of competitors", MIN_VENDORS, MAX_VENDORS, MAX_VENDORS)

    st.divider()
    st.subheader("Competitors")
    slot_keys: list[str] = []
    for i in range(n_vendors):
        default_key = DEFAULT_SLOT_ORDER[i % len(DEFAULT_SLOT_ORDER)]
        default_index = list(ALL_OPTIONS.keys()).index(default_key)
        chosen = st.selectbox(
            f"Seat {i + 1}",
            options=list(ALL_OPTIONS.keys()),
            format_func=lambda k: ALL_OPTIONS[k][0],
            index=default_index,
            key=f"slot_{i}",
        )
        slot_keys.append(chosen)

    model_slots = [k for k in slot_keys if k.startswith("model_")]
    if model_slots:
        names = ", ".join(ALL_OPTIONS[k][0].removeprefix("🤖 ") for k in model_slots)
        st.warning(
            f"{len(model_slots)} seat(s) will make real, billed OpenRouter API calls every "
            f"round: {names}. Check current pricing at openrouter.ai/models.",
            icon="⚠️",
        )

    st.divider()
    profit_visibility = st.radio(
        "Profit visibility",
        options=["Everyone's (full transparency)", "Own only"],
        index=0,
    )
    reveal_rival_profit = profit_visibility.startswith("Everyone's")

    run_clicked = st.button("▶ Run match", type="primary", use_container_width=True)

config = MarketConfig(n_vendors=n_vendors)
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
decision_log_header_placeholder = st.empty()
decision_log_container = st.container()


def build_roster(slot_keys: list[str]) -> list:
    from pricewars.agents.providers import build_openrouter_vendor

    roster = []
    for key in slot_keys:
        if key in BOT_OPTIONS:
            _, factory = BOT_OPTIONS[key]
            roster.append(factory())
        else:
            _, spec = MODEL_OPTIONS[key]
            roster.append(build_openrouter_vendor(spec.model_id, name=spec.display_name))
    return roster


def run_live_match(
    n_rounds: int, seed: int, slot_keys: list[str], reveal_rival_profit: bool
) -> None:
    try:
        roster = build_roster(slot_keys)
    except RuntimeError as e:
        status_placeholder.error(str(e))
        return

    match_config = MatchConfig(n_rounds=n_rounds, seed=seed)
    price_rows: list[dict] = []
    temp_rows: list[dict] = []
    llm_vendors = [v for v in roster if isinstance(v, LLMVendor)]
    if llm_vendors:
        decision_log_header_placeholder.subheader("🧠 Agent decision log")

    def market_temperature_of(round_logs: tuple[RoundLog, ...]) -> float:
        avg_price = sum(r.price_clamped for r in round_logs) / len(round_logs)
        return market_temperature(avg_price, references)

    def on_round(round_num: int, round_logs: tuple[RoundLog, ...]) -> None:
        progress_placeholder.progress(round_num / n_rounds, text=f"Round {round_num} / {n_rounds}")

        # round_logs[i] always corresponds to roster[i] — tournament.py builds each
        # round's log by iterating the same fixed vendor order every round.
        vendor_by_label = {log.vendor_label: vendor for vendor, log in zip(roster, round_logs)}

        for log in round_logs:
            price_rows.append(
                {"round": round_num, "vendor": f"{log.vendor_label} ({log.vendor_name})", "price": log.price_clamped}
            )
        price_df = pd.DataFrame(price_rows).pivot(index="round", columns="vendor", values="price")
        price_chart_placeholder.line_chart(price_df, height=360)

        temp_rows.append({"round": round_num, "temperature": market_temperature_of(round_logs)})
        temp_df = pd.DataFrame(temp_rows).set_index("round")
        temp_chart_placeholder.line_chart(temp_df, height=360, y_label="market temperature")

        def flag_for(log: RoundLog) -> str:
            vendor = vendor_by_label[log.vendor_label]
            if isinstance(vendor, LLMVendor) and any(
                f.round_num == round_num for f in vendor.compliance_log
            ):
                return "⚠️ compliance fallback"
            return ""

        standings = sorted(round_logs, key=lambda r: -r.cumulative_profit)
        standings_df = pd.DataFrame(
            [
                {
                    "vendor": f"{s.vendor_label} ({s.vendor_name})",
                    "price this round": f"${s.price_clamped:.2f}",
                    "profit this round": f"${s.profit:.2f}",
                    "cumulative profit": f"${s.cumulative_profit:.2f}",
                    "flag": flag_for(s),
                }
                for s in standings
            ]
        )
        standings_placeholder.dataframe(standings_df, use_container_width=True, hide_index=True)

        total_failures = sum(len(v.compliance_log) for v in llm_vendors)
        if total_failures:
            per_vendor = ", ".join(
                f"{v.name}: {len(v.compliance_log)}" for v in llm_vendors if v.compliance_log
            )
            compliance_placeholder.warning(
                f"{total_failures} compliance failure(s) so far ({per_vendor}) — see the "
                "fallback flags in the table above.",
                icon="⚠️",
            )

        for vendor, log in zip(roster, round_logs):
            if not isinstance(vendor, LLMVendor):
                continue
            record = next((r for r in vendor.decision_log if r.round_num == round_num), None)
            if record is None:
                continue
            icon = "⚠️" if record.was_compliance_failure else "✅"
            title = f"{icon} Round {round_num} — {log.vendor_label} ({vendor.name}) priced ${record.price:.2f}"
            if record.was_compliance_failure:
                title += " (compliance fallback, not a real decision)"
            with decision_log_container.expander(title, expanded=False):
                st.markdown("**Reasoning:**")
                st.markdown(
                    record.reasoning
                    if record.reasoning
                    else "*(no reasoning text — went straight to tool calls)*"
                )
                st.markdown(f"**Tool calls ({len(record.tool_calls)}):**")
                if record.tool_calls:
                    st.dataframe(
                        pd.DataFrame(record.tool_calls), use_container_width=True, hide_index=True
                    )
                else:
                    st.caption("None.")

    status_placeholder.info("Running...")
    asyncio.run(
        run_match(
            roster,
            config,
            match_config,
            on_round_complete=on_round,
            reveal_rival_profit=reveal_rival_profit,
        )
    )
    status_placeholder.success(f"Match complete — {n_rounds} rounds played.")


if run_clicked:
    run_live_match(n_rounds, int(seed), slot_keys, reveal_rival_profit)
else:
    status_placeholder.info("Configure a match in the sidebar and click **Run match**.")
