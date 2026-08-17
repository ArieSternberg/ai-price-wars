"""Run a scripted-bot tournament and produce the first price-path chart.

No API calls — this is the phase-2 milestone from CLAUDE.md's build order: prove the
round loop, seeding, shuffling, storage, and scripted bots all work end to end, in
one committable, demoable chart.

Usage:
    python scripts/run_scripted_tournament.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import matplotlib.pyplot as plt

from pricewars.agents.scripted import ConstantMarkupBot, RandomBot, TitForTatBot, UndercutterBot
from pricewars.market import MarketConfig
from pricewars.metrics import temperature_by_round
from pricewars.store import RESULTS_DIR, save_match
from pricewars.tournament import MatchConfig, MatchResult, run_match

RUN_ID = "scripted_demo"


def build_roster() -> list:
    """Six scripted bots, deliberately mixed: two undercutters (aggressive), one
    tit-for-tat (reactive), two constant-markups at different markups (indifferent),
    one random (noise). No two bots share a strategy+parameters — identical
    deterministic bots converge into exact lockstep (same inputs, same logic), which
    is correct but makes for a redundant, overlapping line in the chart."""
    return [
        UndercutterBot(undercut=0.10),
        UndercutterBot(undercut=0.50),
        TitForTatBot(),
        ConstantMarkupBot(markup=1.5),
        ConstantMarkupBot(markup=4.0),
        RandomBot(seed=7),
    ]


def make_chart(result: MatchResult, temp_df, path: Path) -> None:
    fig, (ax_price, ax_temp) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    for label, group in result.rounds.groupby("vendor_label"):
        vendor_name = result.labels[label]
        ax_price.plot(
            group["round_num"],
            group["price_clamped"],
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=f"{label} ({vendor_name})",
        )

    ax_price.axhline(
        result.references.competitive, color="tab:green", linestyle="--", linewidth=1,
        label=f"competitive (${result.references.competitive:.2f})",
    )
    ax_price.axhline(
        result.references.monopoly, color="tab:red", linestyle="--", linewidth=1,
        label=f"monopoly (${result.references.monopoly:.2f})",
    )
    ax_price.set_ylabel("price ($)")
    ax_price.set_title(f"Scripted-bot tournament — price path (seed={result.match_config.seed})")
    ax_price.legend(fontsize=7, ncol=2, loc="center left", bbox_to_anchor=(1.0, 0.5))

    ax_temp.plot(temp_df["round_num"], temp_df["temperature"], color="black", linewidth=1.5)
    ax_temp.set_ylim(-0.05, 1.05)
    ax_temp.set_xlabel("round")
    ax_temp.set_ylabel("market\ntemperature")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    config = MarketConfig()
    vendors = build_roster()
    match_config = MatchConfig(n_rounds=30, seed=42)

    result = asyncio.run(run_match(vendors, config, match_config))

    match_dir = save_match(result, RUN_ID)
    print(f"Wrote {match_dir}")

    temp_df = temperature_by_round(result.rounds, result.references)
    chart_path = RESULTS_DIR / "figures" / f"{RUN_ID}_price_path.png"
    make_chart(result, temp_df, chart_path)
    print(f"Wrote {chart_path}")

    final_round = result.rounds[result.rounds["round_num"] == match_config.n_rounds]
    print("\nFinal-round standings (cumulative profit):")
    standings = final_round.sort_values("cumulative_profit", ascending=False)
    for _, row in standings.iterrows():
        print(f"  {row['vendor_label']:>10} ({row['vendor_name']:<16}) ${row['cumulative_profit']:.2f}")


if __name__ == "__main__":
    main()
