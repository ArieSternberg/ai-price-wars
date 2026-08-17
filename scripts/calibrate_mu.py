"""Calibrate `mu` (price sensitivity) before spending a dollar on API calls.

Per CLAUDE.md: "If price sensitivity is too high the market is degenerate — everyone
races to $3.01 and there is no game. This is the single most likely way the project
fails." This script is the free check.

For each candidate `mu`, it:

1. Solves the competitive and monopoly reference prices.
2. Runs naive simultaneous best-response dynamics (every vendor best-responds to
   last round's rivals, starting from the monopoly price) for a fixed horizon, and
   checks whether/how fast the market settles near the competitive price.
3. Classifies the market as degenerate (undercutting is "fatal" — price collapses to
   ~cost and there's no game), flat (mu so high that price barely affects demand, so
   undercutting isn't "rewarded"), or well-behaved.

This does not use LLM agents or the real scripted-bot roster (that lands with
tournament.py) — best-response dynamics is a standalone, deterministic proxy for
"what happens if vendors compete hard," which is exactly the degenerate case we're
screening for.

Usage:
    python scripts/calibrate_mu.py
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from pricewars.market import MarketConfig, best_response, solve_references

MU_SWEEP = [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
HORIZON = 30
CONVERGENCE_TOL = 0.01  # dollars

# Degeneracy thresholds, as a fraction of the cost-to-price_cap range.
DEGENERATE_MARKUP_FRACTION = 0.05  # competitive price within 5% of cost -> race to the bottom
FLAT_SPREAD_FRACTION = 0.10  # monopoly - competitive within 10% of range -> price barely matters

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


@dataclass
class CalibrationRow:
    mu: float
    competitive: float
    monopoly: float
    spread: float
    competitive_markup_pct: float
    rounds_to_converge: int | None
    classification: str


def best_response_dynamics(config: MarketConfig, horizon: int) -> list[list[float]]:
    """Simultaneous naive best-response dynamics, starting from the monopoly price.

    Every vendor starts at the monopoly price (the most "collusive" starting point)
    and, each round, best-responds to what every rival charged last round. Returns
    the full price path, one list of n_vendors prices per round.
    """
    start = solve_references(config).monopoly
    prices = [start] * config.n_vendors
    path = [prices]
    for _ in range(horizon):
        next_prices = []
        for i in range(config.n_vendors):
            rivals = prices[:i] + prices[i + 1 :]
            next_prices.append(best_response(rivals, config))
        prices = next_prices
        path.append(prices)
    return path


def rounds_to_converge(path: list[list[float]], target: float, tol: float) -> int | None:
    for round_idx, prices in enumerate(path):
        if max(abs(p - target) for p in prices) <= tol:
            return round_idx
    return None


def classify(config: MarketConfig, competitive: float, monopoly: float) -> str:
    price_range = config.price_cap - config.cost
    markup_fraction = (competitive - config.cost) / price_range
    spread_fraction = (monopoly - competitive) / price_range
    if markup_fraction < DEGENERATE_MARKUP_FRACTION:
        return "degenerate (race to cost)"
    if spread_fraction < FLAT_SPREAD_FRACTION:
        return "flat (price barely matters)"
    return "well-behaved"


def calibrate(mu_values: list[float] = MU_SWEEP, horizon: int = HORIZON) -> list[CalibrationRow]:
    rows = []
    for mu in mu_values:
        config = MarketConfig(mu=mu)
        refs = solve_references(config)
        path = best_response_dynamics(config, horizon)
        converge_round = rounds_to_converge(path, refs.competitive, CONVERGENCE_TOL)
        rows.append(
            CalibrationRow(
                mu=mu,
                competitive=refs.competitive,
                monopoly=refs.monopoly,
                spread=refs.monopoly - refs.competitive,
                competitive_markup_pct=(refs.competitive - config.cost) / config.cost * 100,
                rounds_to_converge=converge_round,
                classification=classify(config, refs.competitive, refs.monopoly),
            )
        )
    return rows


def print_table(rows: list[CalibrationRow]) -> None:
    header = (
        f"{'mu':>5} | {'competitive':>11} | {'monopoly':>9} | {'spread':>7} | "
        f"{'markup %':>8} | {'converges by':>12} | classification"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        converge_str = str(r.rounds_to_converge) if r.rounds_to_converge is not None else ">horizon"
        print(
            f"{r.mu:>5.2f} | {r.competitive:>11.2f} | {r.monopoly:>9.2f} | {r.spread:>7.2f} | "
            f"{r.competitive_markup_pct:>7.1f}% | {converge_str:>12} | {r.classification}"
        )


def write_csv(rows: list[CalibrationRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "mu",
                "competitive_price",
                "monopoly_price",
                "spread",
                "competitive_markup_pct",
                "rounds_to_converge",
                "classification",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.mu,
                    round(r.competitive, 4),
                    round(r.monopoly, 4),
                    round(r.spread, 4),
                    round(r.competitive_markup_pct, 2),
                    r.rounds_to_converge if r.rounds_to_converge is not None else "",
                    r.classification,
                ]
            )


def write_markdown_summary(rows: list[CalibrationRow], default_mu: float, path: Path) -> None:
    default_row = next((r for r in rows if r.mu == default_mu), None)
    lines = [
        "# mu calibration sweep",
        "",
        "Generated by `scripts/calibrate_mu.py`. No API calls — pure market-model math, "
        "screening for a degenerate market (everyone races to cost) before any paid run.",
        "",
        f"Default config: `cost=3.00`, `n_vendors=6`, `price_cap=15.00`, horizon={HORIZON} rounds "
        "of naive best-response dynamics starting from the monopoly price.",
        "",
        "| mu | competitive | monopoly | spread | markup % | converges by | classification |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        converge_str = str(r.rounds_to_converge) if r.rounds_to_converge is not None else f">{HORIZON}"
        lines.append(
            f"| {r.mu:.2f} | {r.competitive:.2f} | {r.monopoly:.2f} | {r.spread:.2f} | "
            f"{r.competitive_markup_pct:.1f}% | {converge_str} | {r.classification} |"
        )
    lines.append("")
    if default_row is not None:
        lines.append(
            f"**Default `mu={default_mu}`** classifies as *{default_row.classification}* — "
            f"competitive price ${default_row.competitive:.2f} ({default_row.competitive_markup_pct:.0f}% "
            f"markup over cost), monopoly price ${default_row.monopoly:.2f}, best-response dynamics "
            f"converge within {default_row.rounds_to_converge} rounds. "
            + (
                "Kept as the default — clear separation between the two benchmarks with room "
                "for real strategic behavior in between."
                if default_row.classification == "well-behaved"
                else "**Not well-behaved — reconsider the default before running any paid agents.**"
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    rows = calibrate()
    print_table(rows)
    csv_path = RESULTS_DIR / "mu_calibration.csv"
    md_path = RESULTS_DIR / "mu_calibration.md"
    write_csv(rows, csv_path)
    write_markdown_summary(rows, default_mu=MarketConfig().mu, path=md_path)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
