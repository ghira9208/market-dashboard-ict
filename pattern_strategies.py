"""Two more entry families for the shared fixed stop/target engine (same
contract as ema_strategies.py: a finder returns {"confirm_pos", "direction"}
for the bar whose CLOSE confirms the setup; entry is the next bar's open, so
no finder can see the bar it trades on).

  squeeze_bbkc   Bollinger bands sitting inside Keltner channels = volatility
                 compression. After at least `min_bars` of it, the first close
                 outside the bands within 3 bars of the squeeze releasing is
                 the breakout.
  squeeze_nr     NR-N: the narrowest bar of the last `width` bars; the first
                 close beyond its high/low within the next 3 bars is the
                 breakout.
  round_number   Rejection at round levels: a wick that reaches a .00/.50 level
                 (half) or a .00 level (major) and closes back on the other
                 side. Level spacing scales with price and differs for FX
                 (4th/5th decimal figures) versus everything else.
"""
import numpy as np
import pandas as pd

import experiments as ex
from ema_strategies import _ohlc

FAMILIES = ("squeeze_bbkc", "squeeze_nr", "round_number")


def squeeze_bbkc_touches(df, min_bars=4, bb_len=20, bb_k=2.0, kc_mult=1.5, release_window=3):
    _, _, _, close = _ohlc(df)
    s = pd.Series(close)
    mid = s.rolling(bb_len).mean().to_numpy()
    sd = s.rolling(bb_len).std(ddof=0).to_numpy()
    bb_up, bb_lo = mid + bb_k * sd, mid - bb_k * sd
    ema = s.ewm(span=bb_len, adjust=False).mean().to_numpy()
    atr = ex.atr_series(df).to_numpy()
    kc_up, kc_lo = ema + kc_mult * atr, ema - kc_mult * atr
    on = (bb_up < kc_up) & (bb_lo > kc_lo)
    out, run, released_at = [], 0, None
    for i in range(2 * bb_len, len(close)):
        if on[i]:
            run += 1
            released_at = None
            continue
        if run >= min_bars:
            released_at = i  # squeeze just ended after a long enough compression
        run = 0
        if released_at is not None and i - released_at < release_window:
            if close[i] > bb_up[i]:
                out.append({"confirm_pos": i, "direction": "bullish"}); released_at = None
            elif close[i] < bb_lo[i]:
                out.append({"confirm_pos": i, "direction": "bearish"}); released_at = None
        elif released_at is not None:
            released_at = None
    return out


def squeeze_nr_touches(df, width=7, window=3):
    _, high, low, close = _ohlc(df)
    rng = high - low
    n, out = len(close), []
    i = width
    while i < n - 1:
        if rng[i] <= rng[i - width + 1:i + 1].min():
            hi, lo = high[i], low[i]
            for j in range(i + 1, min(n, i + 1 + window)):
                if close[j] > hi:
                    out.append({"confirm_pos": j, "direction": "bullish"}); i = j; break
                if close[j] < lo:
                    out.append({"confirm_pos": j, "direction": "bearish"}); i = j; break
        i += 1
    return out


def level_step(price, fx, major):
    """Spacing of 'round' levels at this price. FX trades in figures (1.1250,
    158.50), everything else in 50s/100s of its own magnitude."""
    mag = np.floor(np.log10(price))
    if fx:
        mag = max(mag, 0)  # AUDUSD at 0.69 and EURUSD at 1.12 both trade in 50-pip figures; don't change scale across 1.0
    step = 0.5 * 10 ** (mag - (2 if fx else 1))
    return step * 2 if major else step


def round_number_touches(df, fx=False, level_type="half", wick_ratio=0.5, tol_atr=0.15, cooldown=12):
    open_, high, low, close = _ohlc(df)
    atr = ex.atr_series(df).to_numpy()
    major = level_type == "major"
    out, last_at = [], {}
    for i in range(20, len(close)):
        a, rng = atr[i], high[i] - low[i]
        if not a or np.isnan(a) or a <= 0 or rng <= 0 or close[i] <= 0:
            continue
        step = level_step(close[i], fx, major)
        up_lvl = (np.floor(close[i] / step) + 1) * step
        dn_lvl = np.floor(close[i] / step) * step
        upper_wick, lower_wick = high[i] - max(open_[i], close[i]), min(open_[i], close[i]) - low[i]
        if high[i] >= up_lvl - tol_atr * a and close[i] < up_lvl and upper_wick >= wick_ratio * rng:
            key, direction = round(up_lvl / step), "bearish"
        elif low[i] <= dn_lvl + tol_atr * a and close[i] > dn_lvl and lower_wick >= wick_ratio * rng:
            key, direction = round(dn_lvl / step), "bullish"
        else:
            continue
        if i - last_at.get(key, -10 ** 9) < cooldown:
            continue
        last_at[key] = i
        out.append({"confirm_pos": i, "direction": direction})
    return out


_FINDERS = {"squeeze_bbkc": squeeze_bbkc_touches, "squeeze_nr": squeeze_nr_touches, "round_number": round_number_touches}


def _builder(family):
    finder = _FINDERS[family]

    def build(df, rr_multiple=2.0, atr_mult_stop=1.5, max_bars=60, cost_bps=ex.DEFAULT_COST_BPS, **params):
        return ex._simulate_fixed_rr(df, finder(df, **params), rr_multiple, atr_mult_stop, max_bars, cost_bps)

    build.__name__ = f"build_{family}_fixed_rr_events"
    return build


BUILDERS = {f"{fam}_fixed_rr": _builder(fam) for fam in FAMILIES}
DETECTOR_VERSIONS = {fam: 1 for fam in FAMILIES}
TIMEFRAMES = {"squeeze_bbkc": ("1h", "1D"), "squeeze_nr": ("1h", "1D"), "round_number": ("5m", "1h")}
RR = [1.5, 3.0]


def settings_grid():
    grid = []
    for rr in RR:
        for min_bars in (4, 8):
            grid.append(("squeeze_bbkc", {"min_bars": min_bars, "rr_multiple": rr}))
        for width in (7, 4):
            grid.append(("squeeze_nr", {"width": width, "rr_multiple": rr}))
        for level_type in ("half", "major"):
            grid.append(("round_number", {"level_type": level_type, "rr_multiple": rr}))
    return grid
