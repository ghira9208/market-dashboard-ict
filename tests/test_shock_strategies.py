"""shock_strategies.py — the news-shock finder. Same bar as the other finders: no lookahead, and each
signal really satisfies the rule it claims."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import shock_strategies as ss


def _quiet(n=200, start="2026-10-14 00:00", freq="5min", base=0.0004):
    """Perfectly even bars (same range every bar) so 'quiet' is exact, not a coin flip."""
    close = 1.3200 + np.arange(n) * 1e-6
    open_ = np.concatenate([[close[0]], close[:-1]])
    half = (base - 1e-6) / 2
    return pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) + half, "Low": np.minimum(open_, close) - half,
                         "Close": close}, index=pd.date_range(start, periods=n, freq=freq, tz="UTC"))


def _with_shock(df, pos, move=-0.0040):
    d = df.copy()
    o = float(d["Close"].iloc[pos - 1])
    d.iloc[pos, d.columns.get_loc("Open")] = o
    d.iloc[pos, d.columns.get_loc("Close")] = o + move
    d.iloc[pos, d.columns.get_loc("High")] = o + max(0, move) + 0.0001
    d.iloc[pos, d.columns.get_loc("Low")] = o + min(0, move) - 0.0001
    return d


def test_a_big_decisive_bar_after_quiet_is_a_signal_in_its_own_direction():
    d = _with_shock(_quiet(), 100, move=-0.0040)
    sig = ss.shock_touches(d)
    assert [s["confirm_pos"] for s in sig] == [100] and sig[0]["direction"] == "bearish"
    up = ss.shock_touches(_with_shock(_quiet(), 100, move=+0.0040))
    assert up[0]["direction"] == "bullish"


def test_fade_mode_trades_against_the_shock():
    d = _with_shock(_quiet(), 100, move=-0.0040)
    assert ss.shock_touches(d, mode="fade")[0]["direction"] == "bullish"


def test_an_ordinary_bar_or_an_indecisive_one_is_not_a_signal():
    assert ss.shock_touches(_quiet()) == []
    d = _with_shock(_quiet(), 100, move=-0.0040)
    d.iloc[100, d.columns.get_loc("Close")] = float(d["Open"].iloc[100]) - 0.0004   # big range, small body
    assert ss.shock_touches(d) == []


def test_the_baseline_never_includes_the_bar_itself_or_anything_after_it():
    d = _with_shock(_quiet(), 100)
    a = ss.shock_touches(d)
    # change everything AFTER the shock violently: earlier signals must not move
    e = d.copy()
    e.iloc[104:, e.columns.get_loc("High")] += 0.05
    e.iloc[104:, e.columns.get_loc("Low")] -= 0.05
    assert [s for s in ss.shock_touches(e) if s["confirm_pos"] <= 103] == [s for s in a if s["confirm_pos"] <= 103]


def test_quiet_filter_rejects_a_shock_that_follows_an_already_wild_stretch():
    d = _quiet()
    d.iloc[94:100, d.columns.get_loc("High")] += 0.0012
    d.iloc[94:100, d.columns.get_loc("Low")] -= 0.0012
    d = _with_shock(d, 100)
    assert ss.shock_touches(d, quiet=True) == []
    assert len(ss.shock_touches(d, quiet=False)) >= 1


def test_quiet_tolerates_pre_shock_bars_a_little_above_the_baseline_but_not_a_lot():
    d = _quiet()
    d.iloc[94:100, d.columns.get_loc("High")] += 0.00005     # ~25% wider than the baseline bars
    d = _with_shock(d, 100)
    assert len(ss.shock_touches(d, quiet=True)) == 1


def test_scheduled_only_keeps_release_minutes_and_drops_the_rest():
    on = _with_shock(_quiet(start="2026-10-14 04:10"), 100)     # bar 100 is the 12:30 bar
    off = _with_shock(_quiet(start="2026-10-14 04:12"), 100)    # bar 100 is the 12:32 bar
    assert len(ss.shock_touches(on, scheduled_only=True)) == 1
    assert ss.shock_touches(off, scheduled_only=True) == []
    assert len(ss.shock_touches(off, scheduled_only=False)) == 1


def test_cooldown_stops_a_burst_of_big_bars_from_stacking_signals():
    d = _quiet()
    for p in (100, 101, 102):
        d = _with_shock(d, p, move=-0.0040)
    assert len(ss.shock_touches(d, quiet=False, cooldown=6)) == 1
    assert len(ss.shock_touches(d, quiet=False, cooldown=0)) == 3


def test_builders_feed_the_shared_engine_and_return_scored_events():
    d = _with_shock(_quiet(), 100, move=-0.0040)
    ev = ss.BUILDERS["news_shock_fixed_rr"](d, rr_multiple=2.0, quiet=True)
    assert len(ev) >= 1 and {"entry_time", "r_real", "direction"} <= set(ev.columns)


def test_grid_is_pre_registered_with_a_fade_control_and_scheduled_only_on_5m_only():
    g5, g1 = ss.settings_grid("5m"), ss.settings_grid("1h")
    assert len(g1) == 16 and len(g5) == 32
    assert {s["mode"] for _, s in g1} == {"continue", "fade"}
    assert any(s["scheduled_only"] for _, s in g5) and not any(s["scheduled_only"] for _, s in g1)
