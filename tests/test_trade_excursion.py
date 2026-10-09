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
