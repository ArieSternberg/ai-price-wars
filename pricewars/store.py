"""Persistence: every match writes its round log to Parquet plus a metadata sidecar.

Results are committed to the repo (see CLAUDE.md) so the analysis notebook reruns
without re-spending API money. Parquet files under `results/matches/` are the whole
database — the analysis notebook queries them straight with DuckDB later; nothing
here depends on DuckDB itself, since Parquet is the actual interchange format.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd

from pricewars.tournament import MatchResult

__all__ = ["RESULTS_DIR", "save_match", "load_match"]

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def save_match(result: MatchResult, run_id: str, results_dir: Path = RESULTS_DIR) -> Path:
    """Write one match's round log + metadata under results/matches/<run_id>/."""
    match_dir = results_dir / "matches" / run_id
    match_dir.mkdir(parents=True, exist_ok=True)

    result.rounds.to_parquet(match_dir / "rounds.parquet", index=False)

    meta = {
        "run_id": run_id,
        "market_config": dataclasses.asdict(result.market_config),
        "match_config": dataclasses.asdict(result.match_config),
        "references": dataclasses.asdict(result.references),
        "labels": result.labels,
    }
    (match_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return match_dir


def load_match(run_id: str, results_dir: Path = RESULTS_DIR) -> tuple[pd.DataFrame, dict]:
    """Read a match back: (rounds DataFrame, metadata dict)."""
    match_dir = results_dir / "matches" / run_id
    rounds = pd.read_parquet(match_dir / "rounds.parquet")
    meta = json.loads((match_dir / "meta.json").read_text())
    return rounds, meta
