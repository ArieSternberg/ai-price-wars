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

from pricewars.agents.providers import build_openrouter_vendor
from pricewars.agents.scripted import ConstantMarkupBot, RandomBot, TitForTatBot, UndercutterBot
from pricewars.market import MarketConfig
from pricewars.tournament import MatchConfig, run_match

DEFAULT_MODEL_ID = "anthropic/claude-sonnet-4.5"  # check openrouter.ai/models before trusting this
N_ROUNDS = 3


async def main(model_id: str) -> None:
    llm_vendor = build_openrouter_vendor(model_id, max_tool_calls=6)
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

    print(f"Running {N_ROUNDS} rounds with {model_id} against 5 scripted bots...")
    result = await run_match(roster, config, match_config)

    llm_rows = result.rounds[result.rounds["vendor_name"] == llm_vendor.name]
    print("\n--- LLM vendor's rounds ---")
    print(
        llm_rows[
            ["round_num", "price_submitted", "price_clamped", "was_out_of_range", "profit"]
        ].to_string(index=False)
    )

    print(f"\n--- Compliance failures: {len(llm_vendor.compliance_log)} ---")
    for failure in llm_vendor.compliance_log:
        print(f"  round {failure.round_num}: {failure.reason}")

    print("\nIf prices look sane and there are no compliance failures, the wiring works end to end.")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_ID
    asyncio.run(main(model))
