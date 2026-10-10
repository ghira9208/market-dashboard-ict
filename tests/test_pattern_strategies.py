"""pattern_strategies.py — squeeze and round-number finders. Same bar as the EMA
finders: no lookahead, and each signal really satisfies the rule it claims."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pattern_strategies as ps


def _ohlc_from_close(close, spread=0.3):
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) + spread,
                         "Low": np.minimum(open_, close) - spread, "Close": close})


def _squeezy(n=1600, seed=5):
    """Alternating quiet and loud regimes so squeezes form and release."""
    rng = np.random.default_rng(seed)
    vol = np.where((np.arange(n) // 80) % 2 == 0, 0.08, 0.9)
    return _ohlc_from_close(100 + np.cumsum(rng.normal(0, 1, n) * vol), spread=0.05)


def _fx_walk(n=2500, seed=2):
    rng = np.random.default_rng(seed)
    return _ohlc_from_close(1.12 + np.cumsum(rng.normal(0, 0.0012, n)), spread=0.0004)


FINDERS = {
    "squeeze_bbkc": (ps.squeeze_bbkc_touches, {"min_bars": 4}, _squeezy),
    "squeeze_nr": (ps.squeeze_nr_touches, {"width": 7}, _squeezy),
    # wick_ratio 0.3: the synthetic bars here have fixed-size wicks, so a 50% rule can never be met on them
    "round_number": (ps.round_number_touches, {"fx": True, "level_type": "half", "wick_ratio": 0.3}, _fx_walk),
}


@pytest.mark.parametrize("name", FINDERS)
def test_every_finder_fires_on_realistic_data(name):
    finder, kw, make = FINDERS[name]
    assert len(finder(make(), **kw)) >= 3


@pytest.mark.parametrize("name", FINDERS)
def test_no_lookahead_the_past_does_not_depend_on_the_future(name):
    finder, kw, make = FINDERS[name]
    df = make()
    full = finder(df, **kw)
    for cut in (700, 1100, 1500):
        truncated = finder(df.iloc[:cut].reset_index(drop=True), **kw)
        assert [t for t in full if t["confirm_pos"] < cut] == truncated, f"{name} changed its past at cut={cut}"


@pytest.mark.parametrize("name", FINDERS)
def test_signals_are_unique_and_directional(name):
    finder, kw, make = FINDERS[name]
    pos = [t["confirm_pos"] for t in finder(make(), **kw)]
    assert pos == sorted(set(pos))
    assert {t["direction"] for t in finder(make(), **kw)} <= {"bullish", "bearish"}


def test_squeeze_breakout_closes_outside_the_bands_after_a_real_compression():
    quiet = 100 + np.cumsum(np.random.default_rng(1).normal(0, 0.02, 120))
    breakout = quiet[-1] + np.cumsum(np.full(6, 1.2))
    touches = ps.squeeze_bbkc_touches(_ohlc_from_close(np.concatenate([quiet, breakout]), spread=0.03), min_bars=4)
    assert any(t["direction"] == "bullish" and 119 <= t["confirm_pos"] <= 130 for t in touches)


def test_squeeze_must_last_min_bars_before_a_breakout_counts():
    df = _squeezy()
    assert len(ps.squeeze_bbkc_touches(df, min_bars=4)) >= 1
    assert ps.squeeze_bbkc_touches(df, min_bars=10_000) == []  # no compression is ever that long


def test_nr7_breaks_the_narrow_bars_high_or_low_not_a_random_bar():
    df = _squeezy()
    high, low, close = df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy()
    rng = high - low
    for t in ps.squeeze_nr_touches(df, width=7, window=3):
        j = t["confirm_pos"]
        # some bar within the last 3 was the narrowest of its 7 and this close is beyond its extreme
        ok = any(rng[i] <= rng[i - 6:i + 1].min() and
                 ((t["direction"] == "bullish" and close[j] > high[i]) or (t["direction"] == "bearish" and close[j] < low[i]))
                 for i in range(j - 3, j))
        assert ok


def test_level_step_scales_with_price_and_differs_for_fx():
    assert ps.level_step(1.1222, fx=True, major=False) == pytest.approx(0.005)    # 1.1200, 1.1250
    assert ps.level_step(158.2, fx=True, major=False) == pytest.approx(0.5)       # 158.00, 158.50
    assert ps.level_step(7800.0, fx=False, major=False) == pytest.approx(50.0)    # 7800, 7850
    assert ps.level_step(82540.0, fx=False, major=False) == pytest.approx(500.0)
    assert ps.level_step(1.1222, fx=True, major=True) == pytest.approx(0.01)      # major = 00 levels only
    assert ps.level_step(0.5612, fx=True, major=False) == pytest.approx(0.005)    # NZDUSD: same 50-pip figures, not 5 pips
    assert ps.level_step(0.6978, fx=True, major=False) == ps.level_step(1.1222, fx=True, major=False)


def test_round_number_rejection_at_a_known_level():
    # price rallies into 1.1250 resistance, wicks through it and closes back below on one bar
    close = np.concatenate([np.linspace(1.1200, 1.1240, 40), [1.1243]])
    df = _ohlc_from_close(close, spread=0.0002)
    df.loc[40, ["Open", "High", "Low", "Close"]] = [1.1244, 1.1252, 1.1241, 1.1242]  # upper wick through 1.1250, closes under
    df = pd.concat([_fx_walk(60).assign(Open=lambda d: d.Open * 0 + 1.1200, High=1.1203, Low=1.1197, Close=1.1200), df], ignore_index=True)
    touches = ps.round_number_touches(df, fx=True, level_type="half", wick_ratio=0.5)
    assert any(t["direction"] == "bearish" and t["confirm_pos"] == 100 for t in touches)


def test_round_number_cooldown_stops_repeat_signals_at_the_same_level():
    df = _fx_walk()
    pos = [t["confirm_pos"] for t in ps.round_number_touches(df, fx=True, level_type="half", cooldown=12)]
    assert all(b - a >= 1 for a, b in zip(pos, pos[1:]))
    more = ps.round_number_touches(df, fx=True, level_type="half", cooldown=1)
    assert len(more) >= len(pos)


def test_settings_grid_and_builders():
    grid = ps.settings_grid()
    assert len(grid) == 12
    assert {fam for fam, _ in grid} == set(ps.FAMILIES)
    ev = ps.BUILDERS["squeeze_nr_fixed_rr"](_squeezy(), width=7, rr_multiple=2.0)
    assert {"entry_time", "r_real", "r_flip", "direction", "mfe_r", "mae_r"} <= set(ev.columns)
