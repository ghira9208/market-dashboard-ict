"""run_event_study's two new discovery-gate conditions (2026-10-06):
MIN_EVENTS_FOR_EDGE (a sample-size floor) and bootstrap_ci_lower_5pct (a
Monte Carlo robustness floor) — added after an external review of this
pipeline's BH/FDR methodology confirmed both were real gaps: the old
verdict was pure p-value + sign, so a 3-event lucky streak cleared
EDGE_FOUND exactly as easily as a 500-event real edge. These pin the new
behavior directly, not just "verdict changed somehow"."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.evidence import MIN_EVENTS_FOR_EDGE, _bootstrap_mean_ci_lower, run_event_study


def _events(n, raw_return, direction="bullish"):
    """n real-shaped events, each with the same raw_return/direction —
    enough to drive the gates deterministically without needing n_permutations
    worth of real market data."""
    return pd.DataFrame({
        "entry_time": pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC"),
        "raw_return": [raw_return] * n,
        "fvg_type": [direction] * n,
    })


def test_empty_events_is_no_data_not_a_crash():
    result = run_event_study(pd.DataFrame(columns=["entry_time", "raw_return", "fvg_type"]))
    assert result["verdict"] == "NO_DATA"


def test_a_tiny_but_perfect_sample_is_blocked_by_the_min_events_gate():
    # Every single event wins by a wide, consistent margin (p-value would
    # clear 0.05 easily) but there are only 10 of them — below
    # MIN_EVENTS_FOR_EDGE. This is exactly the "3-event lucky streak"
    # case the gate exists to catch.
    events = _events(10, raw_return=0.05)
    result = run_event_study(events, cost_bps=1.0, n_permutations=500)
    assert result["n_events"] == 10
    assert result["n_events"] < MIN_EVENTS_FOR_EDGE
    assert result["verdict"] == "NO_EDGE"
    assert result["bootstrap_ci_lower_5pct"] is None  # never spent the resamples on a sample this small


def test_a_large_perfectly_consistent_sample_clears_every_gate():
    # Every event has the identical positive return — p-value, bootstrap
    # CI, and N all pass cleanly. The real sanity check: this new, stricter
    # verdict logic must still say EDGE_FOUND for an unambiguous real edge,
    # not just get stricter in general.
    events = _events(MIN_EVENTS_FOR_EDGE + 20, raw_return=0.05)
    result = run_event_study(events, cost_bps=1.0, n_permutations=500, n_bootstrap=500)
    assert result["verdict"] == "EDGE_FOUND"
    assert result["bootstrap_ci_lower_5pct"] is not None
    assert result["bootstrap_ci_lower_5pct"] > 0
    assert result["min_events_required"] == MIN_EVENTS_FOR_EDGE


def test_a_skewed_few_big_winners_distribution_fails_the_bootstrap_floor():
    # Most events are flat small losers, a rare couple of huge winners
    # carry the point-estimate mean positive. Documented honestly: for
    # this particular shape the permutation p-value ALSO fails (the big-
    # magnitude events inflate the null's own variance when randomly
    # flipped, so the real mean doesn't look extreme against it either) —
    # this test isn't claiming to isolate the bootstrap gate from the
    # p-value gate, just confirming bootstrap_ci_lower_5pct itself goes
    # negative for a genuinely fragile, lucky-outlier-driven distribution,
    # which is the real mechanism the next test isolates directly.
    n = MIN_EVENTS_FOR_EDGE + 20
    rng = np.random.default_rng(1)
    raw = np.concatenate([
        np.full(n - 2, -0.001),   # almost every event: a small real loss
        np.array([0.5, 0.5]),    # two huge winners carrying the mean
    ])
    rng.shuffle(raw)
    events = pd.DataFrame({
        "entry_time": pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC"),
        "raw_return": raw,
        "fvg_type": ["bullish"] * n,
    })
    result = run_event_study(events, cost_bps=0.1, n_permutations=500, n_bootstrap=500)
    assert result["n_events"] >= MIN_EVENTS_FOR_EDGE
    assert result["mean_return_after_cost"] > 0  # the point estimate alone looks fine
    assert result["bootstrap_ci_lower_5pct"] < 0  # but it isn't robust to resampling
    assert result["verdict"] == "NO_EDGE"


def test_bootstrap_gate_alone_blocks_a_sample_that_clears_p_value_and_n(monkeypatch):
    # Isolates the bootstrap gate's own marginal effect, surgically: a
    # real dataset (seed=6, n=35) that genuinely clears p<0.05 and
    # n>=MIN_EVENTS_FOR_EDGE on its own (confirmed directly beforehand),
    # then forces _bootstrap_mean_ci_lower to return a negative value and
    # confirms THAT alone flips EDGE_FOUND to NO_EDGE — proof the gate is
    # load-bearing, not just a field that happens to agree with the other
    # two most of the time.
    rng = np.random.default_rng(6)
    raw = rng.normal(0.004, 0.02, 35)
    events = pd.DataFrame({
        "entry_time": pd.date_range("2026-01-01", periods=35, freq="1h", tz="UTC"),
        "raw_return": raw,
        "fvg_type": ["bullish"] * 35,
    })
    baseline = run_event_study(events, cost_bps=0.1, n_permutations=1000, n_bootstrap=1000, seed=6)
    assert baseline["p_value_vs_random_direction"] < 0.05
    assert baseline["n_events"] >= MIN_EVENTS_FOR_EDGE
    assert baseline["verdict"] == "EDGE_FOUND"

    import research.evidence as evidence
    monkeypatch.setattr(evidence, "_bootstrap_mean_ci_lower", lambda *a, **k: -0.01)
    forced = run_event_study(events, cost_bps=0.1, n_permutations=1000, n_bootstrap=1000, seed=6)
    assert forced["bootstrap_ci_lower_5pct"] == -0.01
    assert forced["verdict"] == "NO_EDGE"


def test_bootstrap_ci_lower_is_deterministic_for_a_fixed_seed():
    signed = np.array([0.01, 0.02, -0.01, 0.03, -0.02] * 10)
    a = _bootstrap_mean_ci_lower(signed, n_bootstrap=300, seed=7)
    b = _bootstrap_mean_ci_lower(signed, n_bootstrap=300, seed=7)
    assert a == b


def test_bootstrap_ci_lower_is_always_at_most_the_observed_mean():
    # The 5th percentile of a resampled distribution centered on the real
    # mean must sit at or below that mean — a basic sanity property any
    # real implementation of this must satisfy.
    signed = np.array([0.01, 0.02, -0.01, 0.03, -0.02, 0.015, -0.005] * 10)
    lower = _bootstrap_mean_ci_lower(signed, n_bootstrap=1000, seed=0)
    assert lower <= signed.mean() + 1e-9


def test_existing_consumers_still_get_a_dict_they_can_get_from():
    # experiments.py/institutional_backtest.py both read this via .get(...)
    # on keys that already existed — confirm those keys are all still here
    # unchanged, only new ones added.
    events = _events(MIN_EVENTS_FOR_EDGE + 5, raw_return=0.02)
    result = run_event_study(events, cost_bps=0.5, n_permutations=200, n_bootstrap=200)
    for key in ("n_events", "cost_bps", "mean_return_after_cost", "win_rate_after_cost",
                "p_value_vs_random_direction", "by_type", "verdict"):
        assert key in result
