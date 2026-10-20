"""News-shock continuation, as an entry family for the shared fixed stop/target engine (same
contract as ema_strategies.py / pattern_strategies.py: a finder returns {"confirm_pos",
"direction"} for the bar whose CLOSE confirms the setup; entry is the next bar's open, so no
finder can see the bar it trades on).

Pattern, from GBPUSD on 2026-10-14: 40 hours of quiet drift (5-10 pip hours), then at 12:30 UTC
(08:30 New York, a US data release) one 5-minute bar of 47 pips on twice the usual volume, every
USD pair, gold, equities and crypto moving together, and then hours of one-way follow-through
(about 130 pips from the high). The idea: after a bar that is several times the recent normal
range AND closes decisively in one direction, price keeps going the same way.

  shock_continuation   bar range >= range_mult x the median range of the previous `lookback`
                       bars (the baseline never includes the bar itself), body >= body_frac of
                       its range, and — when `quiet` — the six bars before it were no wider than
                       `comp_mult` (1.5) x that baseline: not already wild (compression, then
                       release). 1.0 was tried first and rejected today's own GBPUSD event,
                       whose pre-shock bars sat a hair above the 48-bar median. mode="continue"
                       trades the shock's direction; mode="fade" is the pre-registered control
                       that trades against it. `scheduled_only` keeps only bars starting on the
                       :00 or :30 minute (where releases land) so a clock-driven shock can be
                       told apart from an unscheduled spike. `cooldown` bars between signals.
"""
import numpy as np
import pandas as pd

import experiments as ex
from ema_strategies import _ohlc

FAMILIES = ("news_shock",)


def _on_release_minute(ts):
    return ts.minute in (0, 30)


def shock_touches(df, range_mult=4.0, body_frac=0.6, lookback=48, quiet=True, comp_mult=1.5,
                  scheduled_only=False, mode="continue", cooldown=6):
    open_, high, low, close = _ohlc(df)
    rng = high - low
    s = pd.Series(rng)
    baseline = s.rolling(lookback).median().shift(1).to_numpy()   # strictly BEFORE the bar
    pre = s.rolling(6).median().shift(1).to_numpy()
    idx = df.index
    out, last = [], -10 ** 9
    for i in range(lookback + 1, len(close)):
        base = baseline[i]
        if not np.isfinite(base) or base <= 0 or rng[i] < range_mult * base:
            continue
        body = abs(close[i] - open_[i])
        if rng[i] <= 0 or body < body_frac * rng[i]:
            continue
        if quiet and not (pre[i] <= comp_mult * base * (1 + 1e-9)):
            continue
        if scheduled_only and not _on_release_minute(idx[i]):
            continue
        if i - last < cooldown:
            continue
        up = close[i] > open_[i]
        if mode == "fade":
            up = not up
        out.append({"confirm_pos": i, "direction": "bullish" if up else "bearish"})
        last = i
    return out


_FINDERS = {"news_shock": shock_touches}


def _builder(family):
    finder = _FINDERS[family]

    def build(df, rr_multiple=2.0, atr_mult_stop=1.5, max_bars=60, cost_bps=ex.DEFAULT_COST_BPS, **params):
        return ex._simulate_fixed_rr(df, finder(df, **params), rr_multiple, atr_mult_stop, max_bars, cost_bps)

    build.__name__ = f"build_{family}_fixed_rr_events"
    return build


BUILDERS = {f"{fam}_fixed_rr": _builder(fam) for fam in FAMILIES}
DETECTOR_VERSIONS = {fam: 1 for fam in FAMILIES}
TIMEFRAMES = {"news_shock": ("5m", "1h")}
RR = [2.0, 4.0]


def settings_grid(tf_label=None):
    """Fixed before any result was looked at. 'continue' is the hypothesis, 'fade' its control."""
    grid = []
    for mode in ("continue", "fade"):
        for range_mult in (4.0, 6.0):
            for quiet in (True, False):
                for rr in RR:
                    for scheduled in ((False, True) if tf_label == "5m" else (False,)):
                        grid.append(("news_shock", {"mode": mode, "range_mult": range_mult, "quiet": quiet,
                                                    "scheduled_only": scheduled, "rr_multiple": rr}))
    return grid
