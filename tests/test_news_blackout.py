"""news_blackout_study.py — the window arithmetic (the simulation itself is covered in test_trade_excursion)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import news_blackout_study as nb


def _idx(start, periods, freq):
    return pd.date_range(start, periods=periods, freq=freq, tz="UTC")


def test_expand_marks_following_bars_without_wrapping_or_shrinking():
    f = np.zeros(10, bool)
    f[2] = f[8] = True
    out = nb.expand(f, 2)
    assert list(np.flatnonzero(out)) == [2, 3, 4, 8, 9]
    assert nb.expand(f, 0).tolist() == f.tolist()


def test_us_release_slot_follows_new_york_time_through_the_dst_change():
    summer = _idx("2026-10-14 12:00", 12, "5min")          # EDT: 08:30 ET = 12:30 UTC
    winter = _idx("2026-11-04 13:00", 12, "5min")          # EST: 08:30 ET = 13:30 UTC
    fs, fw = nb.calendar_flags(summer, "5m"), nb.calendar_flags(winter, "5m")
    assert summer[fs][0] == pd.Timestamp("2026-10-14 12:30", tz="UTC")
    assert winter[fw][0] == pd.Timestamp("2026-11-04 13:30", tz="UTC")


def test_window_is_the_release_bar_plus_thirty_minutes_on_five_minute_data():
    idx = _idx("2026-10-14 12:00", 24, "5min")
    f = nb.calendar_flags(idx, "5m")
    on = idx[f]
    assert on[0] == pd.Timestamp("2026-10-14 12:30", tz="UTC") and len(on) >= 6
    assert on[5] == pd.Timestamp("2026-10-14 12:55", tz="UTC")


def test_weekends_are_never_release_slots():
    idx = _idx("2026-10-17 12:00", 24, "5min")              # a Saturday
    assert not nb.calendar_flags(idx, "5m").any()


def test_hourly_bar_containing_the_release_is_flagged():
    idx = _idx("2026-10-14 11:00", 5, "1h")                 # 08:30 ET = 12:30 UTC sits inside the 12:00 bar
    f = nb.calendar_flags(idx, "1h")
    assert idx[f][0] == pd.Timestamp("2026-10-14 12:00", tz="UTC")
