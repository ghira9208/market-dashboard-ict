"""mfe_r/mae_r/vol_regime/trend_regime — the Strategy Observatory's own
per-trade telemetry, added to _simulate_fixed_rr/_classify_regimes
2026-10-08. Purely additive to that function's existing contract (see
test_mechanical_equivalence.py for the pre-existing r_real/r_flip/
direction/entry_time behavior, confirmed unchanged by this file's own
passing run alongside it)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import experiments as ex


def _flat_then_move(n_flat, path):
    """n_flat quiet bars (keeps ATR well-defined and >0), then one row
    per (open, high, low, close) in `path`."""
    rows = [{"Open": 100.0, "High": 100.5, "Low": 99.5, "Close": 100.0} for _ in range(n_flat)]
    rows.extend({"Open": o, "High": h, "Low": l, "Close": c} for o, h, l, c in path)
    return pd.DataFrame(rows)


def test_mfe_mae_track_the_real_excursion_before_a_bullish_target_hit():
    # Entry at 100 (confirm_pos=19 -> entry_pos=20, Open=100). Bar 20 dips
    # to 98.5 (a real 1.5-unit adverse move against a ~1-unit ATR stop —
    # clamped by the stop itself) then bar 21 rallies to 103 before the
    # target is actually touched on bar 22.
    df = _flat_then_move(20, [
        (100.0, 100.2, 99.0, 99.5),    # dips — adverse excursion, no stop hit yet (stop ~= entry - atr)
        (99.5, 102.5, 99.4, 102.0),    # rallies — favorable excursion
        (102.0, 104.0, 101.8, 103.5),  # clears target
    ])
    touches = [{"confirm_pos": 19, "direction": "bullish"}]
    events = ex._simulate_fixed_rr(df, touches, rr_multiple=3.0, atr_mult_stop=1.0, max_bars=10, cost_bps=0.0)
    assert len(events) == 1
    row = events.iloc[0]
    atr_at_entry = ex.atr_series(df).to_numpy()[20]
    risk = ex.compute_risk_unit(atr_at_entry, 1.0)
    # mfe/mae are real price excursions converted to R by the SAME risk
    # unit the trade itself was sized on — recomputed independently here,
    # not copied from the production code path, to actually prove the
    # numbers (not just their existence).
    expected_mfe = max(100.0 - 100.0, 102.5 - 100.0, 104.0 - 100.0) / risk
    expected_mae = max(100.0 - 100.0, 100.0 - 99.0, 100.0 - 99.4, 100.0 - 101.8) / risk
    assert row["mfe_r"] == pytest.approx(expected_mfe)
    assert row["mae_r"] == pytest.approx(expected_mae)
    assert row["mae_r"] > 0  # a real drawdown happened before the eventual win


def test_mfe_mae_are_nonnegative_and_present_for_every_real_trade():
    df = _flat_then_move(20, [(100.0, 100.3, 99.7, 100.1) for _ in range(5)])
    touches = [{"confirm_pos": 19, "direction": "bearish"}]
    events = ex._simulate_fixed_rr(df, touches, rr_multiple=2.0, atr_mult_stop=1.0, max_bars=5, cost_bps=0.0)
    assert len(events) == 1
    row = events.iloc[0]
    assert row["mfe_r"] >= 0
    assert row["mae_r"] >= 0


def test_regime_columns_are_none_without_enough_trailing_history():
    # Only 10 bars exist before the entry bar — well under
    # _classify_regimes' own vol_min_periods=20 floor, so both regime
    # columns must read None rather than a fabricated label.
    df = _flat_then_move(0, [(100.0, 100.5, 99.5, 100.0) for _ in range(13)])
    touches = [{"confirm_pos": 9, "direction": "bullish"}]
    events = ex._simulate_fixed_rr(df, touches, rr_multiple=1.0, atr_mult_stop=1.0, max_bars=3, cost_bps=0.0)
    assert len(events) == 1
    assert events.iloc[0]["trend_regime"] is None
    assert events.iloc[0]["vol_regime"] is None


def test_classify_regimes_labels_a_real_volatility_spike_as_high():
    rng = np.random.default_rng(0)
    quiet = 100.0 + rng.normal(0, 0.01, 150).cumsum() * 0.0 + rng.normal(0, 0.05, 150)
    atr = np.concatenate([np.full(150, 0.1), np.full(20, 5.0)])  # a real, sustained ATR spike
    close = np.concatenate([quiet, quiet[-1] + np.cumsum(rng.normal(0, 2.0, 20))])
    vol_regime, _ = ex._classify_regimes(close, atr)
    # Every spike bar ranks at or near the top of its own trailing 100-bar
    # window of mostly-quiet ATR — must read "high", never silently "low"/"normal".
    assert vol_regime[-1] == "high"


def test_classify_regimes_labels_a_flat_market_as_range():
    close = np.full(120, 100.0)
    atr = np.full(120, 1.0)
    _, trend_regime = ex._classify_regimes(close, atr)
    assert trend_regime[-1] == "range"


def test_classify_regimes_labels_a_real_sustained_move_as_trend():
    close = np.concatenate([np.full(60, 100.0), 100.0 + np.arange(1, 61) * 2.0])
    atr = np.full(120, 1.0)
    _, trend_regime = ex._classify_regimes(close, atr)
    # 60 bars after the breakout, price sits 120 ATRs above its own
    # trailing 50-bar average — unambiguously "trend", not "range".
    assert trend_regime[-1] == "trend"


def test_simulate_fixed_rr_output_still_carries_every_pre_existing_column():
    # Regression pin: adding mfe_r/mae_r/vol_regime/trend_regime must never
    # drop or rename r_real/r_flip/direction/entry_time — every existing
    # consumer (run_deep_backtest_rr's _score_split, in particular) reads
    # those by name.
    df = _flat_then_move(20, [(100.0, 100.5, 99.5, 100.0) for _ in range(5)])
    touches = [{"confirm_pos": 19, "direction": "bullish"}]
    events = ex._simulate_fixed_rr(df, touches, rr_multiple=1.0, atr_mult_stop=1.0, max_bars=3, cost_bps=0.0)
    for col in ("entry_time", "r_real", "r_flip", "direction", "mfe_r", "mae_r", "vol_regime", "trend_regime"):
        assert col in events.columns


# ---- blackout windows: standing aside through scheduled news -------------------------------------

def _flat_df(n=60, price=100.0):
    idx = pd.date_range("2026-10-14 08:00", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"Open": price, "High": price + 0.1, "Low": price - 0.1, "Close": price}, index=idx)


def test_blackout_closes_an_open_position_at_the_open_of_the_blackout_bar_and_blocks_entries_in_it():
    import experiments as ex
    d = _flat_df()
    # the release lands INSIDE bar 40: it opens at 100 and closes at 101 — a long that stayed in enjoys it
    d.iloc[40:, d.columns.get_loc("Close")] = 101.0
    d.iloc[40:, d.columns.get_loc("High")] = 101.1
    d.iloc[40:, d.columns.get_loc("Low")] = 99.95
    d.iloc[41:, d.columns.get_loc("Open")] = 101.0
    touch = [{"confirm_pos": 20, "direction": "bullish"}]
    plain = ex._simulate_fixed_rr(d, touch, rr_multiple=50.0, atr_mult_stop=1.0, max_bars=30)
    flag = np.zeros(len(d), bool)
    flag[40] = True
    out = ex._simulate_fixed_rr(d, touch, rr_multiple=50.0, atr_mult_stop=1.0, max_bars=30, blackout=flag)
    assert len(plain) == len(out) == 1
    assert out["exit_time"].iloc[0] == d.index[40]
    assert out["r_real"].iloc[0] < plain["r_real"].iloc[0]       # stood aside, so no part of the gap
    entry_in_blackout = np.zeros(len(d), bool)
    entry_in_blackout[21] = True                                  # entry bar = confirm_pos + 1
    assert ex._simulate_fixed_rr(d, touch, 2.0, 1.0, 30, blackout=entry_in_blackout).empty


def test_no_blackout_leaves_every_result_unchanged():
    import experiments as ex
    d = _flat_df()
    touch = [{"confirm_pos": 20, "direction": "bullish"}]
    a = ex._simulate_fixed_rr(d, touch, 2.0, 1.0, 30)
    b = ex._simulate_fixed_rr(d, touch, 2.0, 1.0, 30, blackout=None)
    c = ex._simulate_fixed_rr(d, touch, 2.0, 1.0, 30, blackout=np.zeros(len(d), bool))
    assert a.equals(b) and a.equals(c)


# ---- atr_lag: size the stop from information available at the entry bar's open ---------------------

def test_atr_lag_sizes_the_stop_from_the_previous_bar_not_the_entry_bar_itself():
    import experiments as ex
    d = _flat_df()                                   # quiet, equal bars: ATR ~0.2
    d.iloc[21, d.columns.get_loc("High")] = 103.0    # the ENTRY bar (confirm 20 -> entry 21) turns out huge
    d.iloc[21, d.columns.get_loc("Low")] = 97.0
    touch = [{"confirm_pos": 20, "direction": "bullish"}]
    same_bar = ex._simulate_fixed_rr(d, touch, rr_multiple=50.0, atr_mult_stop=1.0, max_bars=5, atr_lag=0)
    causal = ex._simulate_fixed_rr(d, touch, rr_multiple=50.0, atr_mult_stop=1.0, max_bars=5, atr_lag=1)
    # ATR at the entry bar jumps 0.20 -> 0.61 purely because of that bar's own 6-point range; lag 1 keeps 0.20
    assert ex.atr_series(d).iloc[21] > 2.5 * ex.atr_series(d).iloc[20]
    assert causal["r_real"].iloc[0] == pytest.approx(-1.1, abs=0.02)         # stopped at -1R (+0.1R of cost) on the same bar
    assert same_bar["r_real"].iloc[0] == pytest.approx(-1.03, abs=0.02)      # lag 0: the stop was ~3x wider than anyone could know


def test_atr_lag_zero_is_the_unchanged_default():
    import experiments as ex
    d = _flat_df()
    touch = [{"confirm_pos": 20, "direction": "bullish"}]
    assert ex._simulate_fixed_rr(d, touch, 2.0, 1.0, 30).equals(ex._simulate_fixed_rr(d, touch, 2.0, 1.0, 30, atr_lag=0))
