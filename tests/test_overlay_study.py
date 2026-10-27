"""overlay_study.py — the pieces where a subtle bug would quietly make a filter
look better than it is: HTF alignment lookahead, the selection-aware hour test,
and the within-combo permutation."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import overlay_study as ov


def test_split_labels_are_60_20_20_in_time_order():
    labels = ov.split_labels(10)
    assert labels == ["research"] * 6 + ["validation"] * 2 + ["locked"] * 2
    assert ov.split_labels(0) == []


def _htf(closes, start="2026-01-01", freq="1h"):
    idx = pd.date_range(start, periods=len(closes), freq=freq, tz="UTC")
    return pd.DataFrame({"Close": closes}, index=idx)


def test_htf_trend_is_only_available_after_the_bar_has_completed():
    closes = np.concatenate([np.full(120, 100.0), np.linspace(100, 160, 40)])
    frame = ov.htf_trend_frame(_htf(closes), 5, 10, pd.Timedelta(hours=1))
    first_bar_open = pd.Timestamp("2026-01-01", tz="UTC")
    assert frame["avail"].iloc[0] == first_bar_open + pd.Timedelta(hours=1)  # bar 0 completes an hour after it opens


def test_a_trade_cannot_see_the_htf_bar_that_is_still_forming():
    # rising market: by bar ~125 the fast EMA is above the slow one
    closes = np.concatenate([np.full(120, 100.0), np.linspace(100, 200, 40)])
    frame = ov.htf_trend_frame(_htf(closes), 5, 10, pd.Timedelta(hours=1))
    bar_open = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=130)
    trade_inside_bar = bar_open + pd.Timedelta(minutes=30)       # bar 130 is still forming
    trade_after_bar = bar_open + pd.Timedelta(hours=1, minutes=1)  # bar 130 has completed
    trades = pd.DataFrame({"entry_utc": [trade_inside_bar, trade_after_bar], "direction": ["bullish", "bullish"],
                           "r": [1.0, 1.0], "combo": ["x", "x"]})
    got = ov.attach_trend(trades, frame)
    used = got["avail"].tolist()
    assert used[0] == bar_open                       # sees bar 129 (completed at bar_open), never bar 130
    assert used[1] == bar_open + pd.Timedelta(hours=1)  # now bar 130 has completed
    assert all(u <= e for u, e in zip(used, got["entry_utc"]))


def test_alignment_follows_direction_and_trend_and_warmup_is_excluded():
    up = np.concatenate([np.full(120, 100.0), np.linspace(100, 200, 60)])
    frame = ov.htf_trend_frame(_htf(up), 5, 10, pd.Timedelta(hours=1))
    late = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=170)
    early = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=5)  # inside the EMA warm-up
    trades = pd.DataFrame({"entry_utc": [late, late, early], "direction": ["bullish", "bearish", "bullish"],
                           "r": [1.0, 1.0, 1.0], "combo": ["x"] * 3})
    got = ov.attach_trend(trades, frame).sort_values("entry_utc").reset_index(drop=True)
    by = {(row.direction, row.entry_utc == late): row.aligned for row in got.itertuples()}
    assert by[("bullish", True)] == 1.0   # long with the uptrend
    assert by[("bearish", True)] == 0.0   # short against it
    assert np.isnan(got[got["entry_utc"] == early]["aligned"].iloc[0])  # warm-up: no verdict


def test_worst_hour_test_is_not_fooled_by_picking_the_minimum_of_noise():
    # pure noise: the worst of 24 hours always looks bad, but the max-statistic p-value must say so
    rng = np.random.default_rng(0)
    n = 24 * 200
    df = pd.DataFrame({"r": rng.normal(0, 1, n), "hour": np.tile(np.arange(24), 200)})
    _, _, p_worst, p_best, _ = ov.worst_hour_test(df, np.random.default_rng(1), b=400)
    assert p_worst > 0.05 and p_best > 0.05


def test_worst_hour_test_does_find_a_real_hour_effect():
    rng = np.random.default_rng(2)
    n = 24 * 200
    hour = np.tile(np.arange(24), 200)
    r = rng.normal(0, 1, n) - np.where(hour == 16, 0.6, 0.0)  # a genuinely bad hour
    worst, _, p_worst, _, _ = ov.worst_hour_test(pd.DataFrame({"r": r, "hour": hour}), np.random.default_rng(3), b=400)
    assert worst == 16 and p_worst < 0.05


def test_stratified_permutation_ignores_differences_between_combos():
    # combo A is great, combo B is terrible; within each combo alignment is irrelevant
    rng = np.random.default_rng(4)
    rows = []
    for combo, base in (("A", 2.0), ("B", -2.0)):
        aligned = rng.random(200) < (0.9 if combo == "A" else 0.1)   # A is mostly aligned, B mostly counter
        rows += [{"combo": combo, "r": base + rng.normal(0, 1), "aligned": float(a)} for a in aligned]
    d, p = ov.stratified_diff_p(pd.DataFrame(rows), np.random.default_rng(5), b=500)
    assert d > 1.0       # naively it looks like alignment is hugely valuable...
    assert p > 0.05      # ...but within each combo it makes no difference, and the test knows


def test_stratified_permutation_detects_a_real_within_combo_effect():
    rng = np.random.default_rng(6)
    rows = []
    for combo in "ABC":
        aligned = rng.random(200) < 0.5
        rows += [{"combo": combo, "r": rng.normal(0, 1) + (0.7 if a else 0.0), "aligned": float(a)} for a in aligned]
    d, p = ov.stratified_diff_p(pd.DataFrame(rows), np.random.default_rng(7), b=500)
    assert d > 0.4 and p < 0.05
