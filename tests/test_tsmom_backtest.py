"""tsmom_backtest.py — the properties that make a momentum backtest honest: no
lookahead, right-signed positions, costs only on change, and a null test that
neither invents edge in a random walk nor misses a real one."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tsmom_backtest as tm


def _series(returns, start="2010-01-04"):
    idx = pd.bdate_range(start, periods=len(returns))
    return pd.Series(100 * np.cumprod(1 + np.asarray(returns)), index=idx)


def _random_walk(n=3200, seed=0, vol=0.01):
    return _series(np.random.default_rng(seed).normal(0, vol, n))


def _regime_momentum(n=3200, seed=0):
    """Drift flips sign every ~100 days: genuine, persistent momentum."""
    rng = np.random.default_rng(seed)
    drift = np.repeat(rng.choice([-0.0015, 0.0015], size=n // 100 + 1), 100)[:n]
    return _series(drift + rng.normal(0, 0.008, n))


def test_signs_follow_the_trend():
    up = tm.weekly_arrays(_series(np.full(900, 0.001)), 60)
    down = tm.weekly_arrays(_series(np.full(900, -0.001)), 60)
    assert (up["pos"] == 1).all() and (down["pos"] == -1).all()
    assert up["r"].mean() > 0 and down["r"].mean() > 0  # both are right about their trend


def test_a_decision_never_uses_the_future():
    base = _random_walk(1500, seed=1)
    wa = tm.weekly_arrays(base, 60)
    t = wa.index[150]
    # corrupt everything more than ~2 weeks after decision week t
    corrupted = base.copy()
    corrupted.loc[corrupted.index > t + pd.Timedelta(days=16)] *= 3.0
    wb = tm.weekly_arrays(corrupted, 60)
    row_a, row_b = wa.loc[t], wb.loc[t]
    assert row_a["pos"] == row_b["pos"] and row_a["u"] == pytest.approx(row_b["u"])


def test_the_position_does_not_depend_on_the_week_it_is_held():
    base = _random_walk(1500, seed=2)
    wa = tm.weekly_arrays(base, 60)
    t = wa.index[150]
    altered = base.copy()
    nxt = (altered.index > t) & (altered.index <= t + pd.Timedelta(days=7))
    altered.loc[nxt] *= 1.05  # change the holding week's prices only
    wb = tm.weekly_arrays(altered, 60)
    assert wa.loc[t, "pos"] == wb.loc[t, "pos"]       # decided before it
    assert wa.loc[t, "u"] != pytest.approx(wb.loc[t, "u"])  # but it is what the position earns


def test_costs_are_charged_only_when_the_position_changes():
    steady = tm.weekly_arrays(_series(np.full(900, 0.001)), 60)
    assert steady["cost"].iloc[1:].eq(0).all()
    choppy = tm.weekly_arrays(_random_walk(3000, seed=3), 30)
    assert choppy["cost"].sum() > 0
    flips = choppy["pos"].diff().abs().fillna(0)
    assert (choppy["cost"][flips == 0] == 0).all()


def test_slices_are_60_20_20():
    assert tm.slice_bounds(100) == {"research": (0, 60), "validation": (60, 80), "locked": (80, 100)}


def test_p_value_is_never_exactly_zero():
    w = tm.weekly_arrays(_regime_momentum(seed=5), 60)
    _, p = tm.shift_p(w["pos"].to_numpy(), w["u"].to_numpy(), w["cost"].to_numpy(), 0, len(w), np.random.default_rng(0), n_shifts=300)
    assert 0 < p <= 1


def test_the_null_does_not_invent_edge_in_a_random_walk():
    rng = np.random.default_rng(10)
    small = 0
    for seed in range(30):
        w = tm.weekly_arrays(_random_walk(3200, seed=seed + 100), 90)
        _, p = tm.shift_p(w["pos"].to_numpy(), w["u"].to_numpy(), w["cost"].to_numpy(), 0, len(w), rng, n_shifts=300)
        small += p < 0.05
    assert small <= 6, f"{small}/30 random walks looked 'significant' — the null is too lenient"


def test_the_null_finds_genuine_regime_momentum():
    rng = np.random.default_rng(12)
    hits = 0
    for seed in range(10):
        w = tm.weekly_arrays(_regime_momentum(seed=seed), 60)
        _, p = tm.shift_p(w["pos"].to_numpy(), w["u"].to_numpy(), w["cost"].to_numpy(), 0, len(w), rng, n_shifts=300)
        hits += p < 0.05
    assert hits >= 7


def test_run_writes_schema_compatible_trials_and_a_portfolio_per_lookback():
    series = {f"T{i}": _regime_momentum(seed=i) for i in range(6)}
    written = []
    trials = tm.run(list(series), series.get, np.random.default_rng(1), written.append)
    assert written == trials
    per_asset = [t for t in trials if t["ticker"] != "PORTFOLIO"]
    portfolio = [t for t in trials if t["ticker"] == "PORTFOLIO"]
    assert len(per_asset) == 6 * len(tm.LOOKBACKS) and len(portfolio) == len(tm.LOOKBACKS)
    needed = {"ticker", "tf_label", "label", "verdict", "p_value_train", "mean_return_train", "mean_return_holdout",
              "p_value_holdout", "holdout_verdict", "same_sign", "locked_mean_return", "locked_n", "strategy_id"}
    assert needed <= set(trials[0])
    assert all(t["verdict"] == "SCORED" and 0 < t["p_value_train"] <= 1 for t in trials)
    assert all(t["n_assets"] == 6 for t in portfolio)


def test_run_skips_assets_without_enough_history():
    short = _random_walk(400, seed=1)  # < MIN_WEEKS of weekly data
    trials = tm.run(["S"], {"S": short}.get, np.random.default_rng(1), lambda t: None)
    assert trials == []


def test_stale_prices_cannot_create_infinite_or_huge_results():
    # an asset that prints the same price for months, then wakes up (ARB-USD's early history)
    rng = np.random.default_rng(3)
    rets = np.concatenate([np.zeros(150), rng.normal(0, 0.03, 100), np.zeros(80), rng.normal(0, 0.03, 600)])
    w = tm.weekly_arrays(_series(rets), 30)
    assert np.isfinite(w[["pos", "u", "cost", "r"]].to_numpy()).all()
    assert w["u"].abs().max() <= tm.U_CLIP + 1e-9


def test_the_vol_floor_stops_a_pegged_asset_being_levered_to_infinity():
    peg = _series(np.random.default_rng(1).normal(0, 0.0001, 900))  # a 1bp-a-day "peg"
    w = tm.weekly_arrays(peg, 30)
    assert w["u"].abs().max() <= tm.U_CLIP + 1e-9


def test_one_genuine_fat_tail_week_is_capped_not_deleted():
    rets = np.random.default_rng(2).normal(0, 0.01, 900)
    rets[500] = 0.9  # a +90% week
    w = tm.weekly_arrays(_series(rets), 30)
    assert w["u"].abs().max() == pytest.approx(tm.U_CLIP)
    assert len(w) > 150  # the week stays in the sample
