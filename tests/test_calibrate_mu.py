"""Tests for scripts/calibrate_mu.py — mostly a guard that the default `mu` stays
well-behaved if MarketConfig's other defaults ever change.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from calibrate_mu import calibrate, classify  # noqa: E402
from pricewars.market import MarketConfig  # noqa: E402


def test_default_mu_is_well_behaved():
    """The market default (mu=1.5) must not be degenerate — this is the guard CLAUDE.md
    asks for before any paid API run."""
    default_mu = MarketConfig().mu
    rows = calibrate(mu_values=[default_mu], horizon=10)
    assert rows[0].classification == "well-behaved"


def test_low_mu_is_flagged_degenerate():
    """Sanity check the classifier itself: a very low mu (hair-trigger undercutting)
    should be caught, not silently pass calibration."""
    config = MarketConfig(mu=0.05)
    classification = classify(config, competitive=3.05, monopoly=7.0)
    assert classification == "degenerate (race to cost)"


def test_high_mu_with_no_spread_is_flagged_flat():
    config = MarketConfig()
    classification = classify(config, competitive=8.0, monopoly=8.1)
    assert classification == "flat (price barely matters)"


@pytest.mark.parametrize("mu", [0.75, 1.0, 1.25, 1.5, 2.0])
def test_reasonable_mu_range_is_well_behaved(mu):
    rows = calibrate(mu_values=[mu], horizon=10)
    assert rows[0].classification == "well-behaved"
