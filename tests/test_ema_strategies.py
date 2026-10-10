"""ema_strategies.py — the four EMA entry families. The invariants that matter
most are (1) no lookahead: a finder must produce the same signals for the
past whether or not the future exists, and (2) each signal really satisfies
the rule it claims."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ema_strategies as es
import experiments as ex


def _ohlc_from_close(close, spread=0.4):
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) + spread,
                         "Low": np.minimum(open_, close) - spread, "Close": close})


def _wavy(n=1500, seed=3):
    """Trending waves with noise — enough crossings/pullbacks to exercise every finder."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return _ohlc_from_close(100 + 8 * np.sin(t / 60) + 3 * np.sin(t / 17) + rng.normal(0, 0.6, n).cumsum() * 0.15)


FINDERS = {
    "ema_cross_pullback": (es.ema_cross_pullback_touches, {"fast": 9, "slow": 21}),
    "ema_cross_anticipation": (es.ema_cross_anticipation_touches, {"fast": 9, "slow": 21, "gap_atr": 0.6}),
    "ema_retrace_bounce": (es.ema_retrace_bounce_touches, {"fast": 9, "slow": 21}),
    "ema_equilibrium": (es.ema_equilibrium_touches, {"spans": (20, 50), "k": 1.5}),
}


@pytest.mark.parametrize("name", FINDERS)
def test_every_finder_fires_on_realistic_data(name):
    finder, kw = FINDERS[name]
    assert len(finder(_wavy(), **kw)) >= 3, f"{name} found almost nothing — the rule may be unreachable"


@pytest.mark.parametrize("name", FINDERS)
def test_no_lookahead_signals_in_the_past_do_not_depend_on_the_future(name):
    finder, kw = FINDERS[name]
    df = _wavy()
    full = finder(df, **kw)
    for cut in (600, 900, 1200):
        truncated = finder(df.iloc[:cut].reset_index(drop=True), **kw)
        assert [t for t in full if t["confirm_pos"] < cut] == truncated, f"{name} changed its past at cut={cut}"


@pytest.mark.parametrize("name", FINDERS)
def test_signals_are_unique_in_time_and_have_a_direction(name):
    finder, kw = FINDERS[name]
    touches = finder(_wavy(), **kw)
    positions = [t["confirm_pos"] for t in touches]
    assert positions == sorted(set(positions))
    assert {t["direction"] for t in touches} <= {"bullish", "bearish"}


def test_pullback_signal_really_touches_fast_ema_while_the_cross_holds():
    df = _wavy()
    _, high, low, close = es._ohlc(df)
    f, s = es._ema(close, 9), es._ema(close, 21)
    for t in es.ema_cross_pullback_touches(df, fast=9, slow=21):
        j = t["confirm_pos"]
        if t["direction"] == "bullish":
            assert low[j] <= f[j] and close[j] > s[j]
        else:
            assert high[j] >= f[j] and close[j] < s[j]


def test_pullback_needs_a_cross_first_a_pure_trend_never_signals():
    df = _ohlc_from_close(np.linspace(100, 200, 600))  # fast is above slow the whole time: no cross
    assert es.ema_cross_pullback_touches(df, fast=9, slow=21) == []


def test_anticipation_signals_only_while_fast_is_closing_in_but_has_not_crossed():
    df = _wavy()
    _, _, _, close = es._ohlc(df)
    f, s = es._ema(close, 9), es._ema(close, 21)
    atr = ex.atr_series(df).to_numpy()
    for t in es.ema_cross_anticipation_touches(df, fast=9, slow=21, gap_atr=0.6, converge_bars=3):
        i = t["confirm_pos"]
        gap = s - f
        assert abs(gap[i]) / atr[i] <= 0.6 + 1e-9
        assert (gap[i] > 0) == (t["direction"] == "bullish")  # still on the pre-cross side
        assert abs(gap[i]) < abs(gap[i - 1]) < abs(gap[i - 2]) < abs(gap[i - 3])


def test_bounce_signal_touches_slow_ema_but_closes_back_on_the_trend_side():
    df = _wavy()
    _, high, low, close = es._ohlc(df)
    s = es._ema(close, 21)
    for t in es.ema_retrace_bounce_touches(df, fast=9, slow=21):
        j = t["confirm_pos"]
        if t["direction"] == "bullish":
            assert low[j] <= s[j] and close[j] > s[j]
        else:
            assert high[j] >= s[j] and close[j] < s[j]


def test_bounce_fires_on_a_clean_uptrend_with_one_dip_to_the_slow_ema():
    t = np.arange(400)
    close = 100 + t * 0.2
    close[300:306] -= np.array([0.5, 2.0, 4.0, 5.0, 3.0, 1.0])  # a dip that reaches the slow EMA
    touches = es.ema_retrace_bounce_touches(_ohlc_from_close(close), fast=9, slow=21)
    assert any(tt["direction"] == "bullish" and 298 <= tt["confirm_pos"] <= 312 for tt in touches)


def test_equilibrium_buys_the_first_up_close_after_a_stretch_below_and_arms_once():
    close = np.concatenate([np.full(300, 100.0), np.linspace(100, 90, 12), [90.5, 91.0, 91.5], np.full(60, 92.0)])
    touches = es.ema_equilibrium_touches(_ohlc_from_close(close, spread=0.2), spans=(20, 50), k=1.5)
    bull = [t for t in touches if t["direction"] == "bullish"]
    assert len(bull) == 1 and 300 <= bull[0]["confirm_pos"] <= 316  # one signal per excursion, not one per bar


def test_equilibrium_three_mas_and_two_mas_define_different_equilibria():
    df = _wavy()
    three = es.ema_equilibrium_touches(df, spans=(20, 50, 100), k=1.5)
    two = es.ema_equilibrium_touches(df, spans=(20, 50), k=1.5)
    assert three != two


def test_settings_grid_is_the_explicit_multiple_testing_burden():
    grid = es.settings_grid()
    assert len(grid) == 26
    assert {fam for fam, _ in grid} == set(es.FAMILIES)
    assert len({(fam, tuple(sorted(s.items()))) for fam, s in grid}) == len(grid)  # no duplicate settings


def test_builders_return_the_shared_event_schema_and_run_through_the_shared_engine():
    df = _wavy()
    for fam, settings in es.settings_grid()[:6]:
        events = es.BUILDERS[f"{fam}_fixed_rr"](df, **settings)
        assert {"entry_time", "r_real", "r_flip", "direction", "mfe_r", "mae_r"} <= set(events.columns)


def test_builders_use_the_shared_stop_target_primitive(monkeypatch):
    calls = []
    real = ex.compute_stop_target
    monkeypatch.setattr(ex, "compute_stop_target", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    es.BUILDERS["ema_cross_pullback_fixed_rr"](_wavy(), fast=9, slow=21, window=20, rr_multiple=2.0)
    assert calls  # exits come from experiments._simulate_fixed_rr, not a private copy
