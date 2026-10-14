"""market_regime.py — the fingerprint features must mean what they say, never
use a future bar, and analog matching must never match a day to its own future."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import market_regime as mr


def _frame(n=420, seed=0, vol=0.01, start="2024-01-01", common=None):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, vol, n) + (common if common is not None else 0)
    close = 100 * np.cumprod(1 + ret)
    hi, lo = close * (1 + np.abs(rng.normal(0, vol / 2, n))), close * (1 - np.abs(rng.normal(0, vol / 2, n)))
    return pd.DataFrame({"Open": close, "High": hi, "Low": lo, "Close": close}, index=pd.bdate_range(start, periods=n))


def _basket(k=8, n=420, common_strength=0.0, seed=0):
    rng = np.random.default_rng(seed + 99)
    factor = rng.normal(0, 0.01, n) * common_strength
    return {f"A{i}": _frame(n, seed + i, common=factor) for i in range(k)}


def test_features_cover_every_field_and_stay_in_range():
    f = mr.day_features(_basket())
    assert list(f.columns) == mr.FEATURES and len(f) > 100
    assert f["vol_pct"].between(0, 1).all() and f["breadth"].between(0, 1).all()
    assert (f["activity"] > 0).all()


def test_comove_is_high_when_one_factor_drives_everything_and_low_when_independent():
    low = mr.day_features(_basket(common_strength=0.0))["comove"].mean()
    high = mr.day_features(_basket(common_strength=3.0))["comove"].mean()
    assert high > 0.5 and low < 0.2


def test_a_feature_never_uses_a_later_bar():
    frames = _basket()
    full = mr.day_features(frames)
    cut = full.index[100]
    trunc = mr.day_features({t: f[f.index <= cut] for t, f in frames.items()})
    for col in mr.FEATURES:
        assert trunc.loc[cut, col] == pytest.approx(full.loc[cut, col], rel=1e-9, abs=1e-9)


def test_activity_rises_on_a_wide_range_day():
    frames = _basket()
    day = frames["A0"].index[300]
    for f in frames.values():
        f.loc[day, "High"] *= 1.15
        f.loc[day, "Low"] *= 0.85
    f = mr.day_features(frames)
    assert f.loc[day, "activity"] > f["activity"].median() * 2


def test_analogs_exclude_the_target_and_the_recent_buffer():
    f = mr.day_features(_basket())
    target = f.index[-1]
    a = mr.analog_days(f, target, k=20, buffer_days=10)
    assert len(a) == 20 and target not in a.index
    assert (a.index <= target - pd.Timedelta(days=10)).all()


def test_the_nearest_analog_of_a_day_that_is_repeated_is_the_original():
    f = mr.day_features(_basket())
    twin = f.iloc[-1].copy()
    f.loc[f.index[40]] = twin  # plant an exact copy of the target's fingerprint far in the past
    a = mr.analog_days(f, f.index[-1], k=3)
    assert f.index[40] == a.index[0] and a.iloc[0]["distance"] == pytest.approx(0.0, abs=1e-9)


def test_percentile_of_ranks_against_earlier_days_only():
    f = mr.day_features(_basket())
    day = f.index[-1]
    p = mr.percentile_of(f, day, "activity")
    assert 0 <= p <= 1
    f2 = f.copy(); f2.loc[day, "activity"] = f["activity"].max() * 10
    assert mr.percentile_of(f2, day, "activity") == 1.0


def test_assets_on_different_calendars_still_produce_features():
    # crypto-like assets trade every calendar day, FX-like ones skip weekends and a few holidays;
    # a union of those calendars used to leave no clean window for any asset and returned nothing
    frames = {}
    for i in range(4):
        f = _frame(420, seed=i)
        frames[f"FX{i}"] = f.drop(f.index[[30, 31, 200, 201, 202]])        # own holidays
    for i in range(4):
        idx = pd.date_range("2024-01-01", periods=600, freq="D")          # includes weekends
        rng = np.random.default_rng(50 + i)
        close = 100 * np.cumprod(1 + rng.normal(0, 0.02, 600))
        frames[f"CR{i}"] = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close}, index=idx)
    f = mr.day_features(frames)
    assert len(f) > 100 and not f.isna().any().any()
    assert (f.index.dayofweek < 5).all()  # weekday calendar
