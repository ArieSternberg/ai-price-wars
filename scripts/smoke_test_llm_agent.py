"""A tiny live smoke test: one real model, a few rounds, against scripted bots.

This is the first script in the repo that spends real money. It is not run
automatically by anything — run it deliberately, and only after checking the model
id against https://openrouter.ai/models (ids and pricing churn fast).

No caching, no budget guard yet (that's phase 4) — this is intentionally tiny
(3 rounds, 1 LLM vendor) to keep the blast radius small: a handful of API calls,
worth cents, not dollars.

Usage:
    python scripts/smoke_test_llm_agent.py [model_id]
    python scripts/smoke_test_llm_agent.py anthropic/claude-sonnet-4.5
"""

from __future__ import annotations

import asyncio
import sys

from pricewars.agents.llm import DEFAULT_MAX_TOOL_CALLS
from pricewars.agents.providers import build_openrouter_vendor
from pricewars.agents.scripted import ConstantMarkupBot, RandomBot, TitForTatBot, UndercutterBot
from pricewars.market import MarketConfig
from pricewars.tournament import MatchConfig, RoundLog, run_match

DEFAULT_MODEL_ID = "anthropic/claude-sonnet-4.5"  # check openrouter.ai/models before trusting this
N_ROUNDS = 3


def make_print_round_live(llm_vendor):
    """Live progress: print each vendor's outcome as soon as a round finishes, while
    the next round's API calls are still in flight. Closes over llm_vendor so it can
    flag rounds where the printed price is actually the harness's compliance-failure
    fallback, not what the model decided — otherwise a fallback silently looks like
    a real (if repeated) decision."""

    def print_round_live(round_num: int, round_logs: tuple[RoundLog, ...]) -> None:
        failed_this_round = {f.round_num for f in llm_vendor.compliance_log} & {round_num}
        print(f"\n--- Round {round_num} ---")
        for log in sorted(round_logs, key=lambda r: -r.cumulative_profit):
            flags = []
            if log.was_out_of_range:
                flags.append("CLAMPED/OUT-OF-RANGE")
            if log.vendor_name == llm_vendor.name and round_num in failed_this_round:
                flags.append("COMPLIANCE FALLBACK — not a real decision")
            flag_str = f"  [{', '.join(flags)}]" if flags else ""
            print(
                f"  {log.vendor_label:>10} ({log.vendor_name:<24}) "
                f"price=${log.price_clamped:6.2f}  sold={log.sold:>3}  "
                f"profit=${log.profit:8.2f}  cum=${log.cumulative_profit:9.2f}{flag_str}"
            )

    return print_round_live


async def main(model_id: str) -> None:
    llm_vendor = build_openrouter_vendor(model_id, max_tool_calls=DEFAULT_MAX_TOOL_CALLS)
    roster = [
        llm_vendor,
        UndercutterBot(),
        TitForTatBot(),
        ConstantMarkupBot(markup=1.5),
        ConstantMarkupBot(markup=4.0),
        RandomBot(seed=1),
    ]
    config = MarketConfig()
    match_config = MatchConfig(n_rounds=N_ROUNDS, seed=1)

    print(f"Running {N_ROUNDS} rounds with {model_id} (max_tool_calls={DEFAULT_MAX_TOOL_CALLS}) against 5 scripted bots...")
    result = await run_match(
        roster, config, match_config, on_round_complete=make_print_round_live(llm_vendor)
    )

    llm_rows = result.rounds[result.rounds["vendor_name"] == llm_vendor.name].copy()
    failed_rounds = {f.round_num for f in llm_vendor.compliance_log}
    llm_rows["compliance_fallback"] = llm_rows["round_num"].isin(failed_rounds)
    print("\n--- LLM vendor's rounds ---")
    print(
        llm_rows[
            ["round_num", "price_submitted", "price_clamped", "was_out_of_range", "compliance_fallback", "profit"]
        ].to_string(index=False)
    )

    print(f"\n--- Compliance failures: {len(llm_vendor.compliance_log)} / {N_ROUNDS} rounds ---")
    for failure in llm_vendor.compliance_log:
        print(f"  round {failure.round_num}: {failure.reason}")

    print(
        "\nWiring works end to end if at least one round shows a real (non-fallback) "
        "set_price call. Compliance failures above are a real measurement, not a bug — "
        "they're exactly what PLAN.md's compliance metrics are meant to catch."
    )


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_ID
    asyncio.run(main(model))
