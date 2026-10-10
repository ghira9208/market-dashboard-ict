"""Four EMA-based entry families, all exited with the project's one shared
fixed stop/target engine (experiments._simulate_fixed_rr) so anything that
survives research can be executed by the live bot without a second simulator.

Each finder returns a list of {"confirm_pos": int, "direction": "bullish" |
"bearish"} — the bar whose CLOSE confirms the setup; the trade enters at the
NEXT bar's open (experiments.resolve_entry_pos), so no finder can see the
bar it trades on. EMAs are recursive (ewm, adjust=False) and every finder
skips a warm-up of 2x its slowest span, where the average is still mostly
its seed value.

  ema_cross_pullback      after a fast/slow cross, buy the first pullback
                          that touches the FAST EMA while the cross still
                          holds, and ride the trend.
  ema_cross_anticipation  enter on the last candle BEFORE a cross — fast
                          EMA closing in on slow, converging for several
                          bars — betting the cross happens.
  ema_retrace_bounce      in an established trend, buy a retracement that
                          touches the SLOW EMA and holds above it, betting
                          the trend does NOT cross.
  ema_equilibrium         price stretched k ATRs away from the mean of 2 or
                          3 EMAs ("equilibrium") and turning back toward it;
                          bet on the pull to equilibrium (higher timeframes).
"""
import numpy as np
import pandas as pd

import experiments as ex

FAMILIES = ("ema_cross_pullback", "ema_cross_anticipation", "ema_retrace_bounce", "ema_equilibrium")


def _ohlc(df):
    g = lambda up, lo: (df[up] if up in df else df[lo]).to_numpy(dtype=float)
    return g("Open", "open"), g("High", "high"), g("Low", "low"), g("Close", "close")


def _ema(close, span):
    return pd.Series(close).ewm(span=span, adjust=False).mean().to_numpy()


def ema_cross_pullback_touches(df, fast=9, slow=21, window=20):
    _, high, low, close = _ohlc(df)
    f, s = _ema(close, fast), _ema(close, slow)
    n, start = len(close), 2 * slow
    out = []
    for i in range(start + 1, n):
        if f[i - 1] <= s[i - 1] and f[i] > s[i]:
            direction = "bullish"
        elif f[i - 1] >= s[i - 1] and f[i] < s[i]:
            direction = "bearish"
        else:
            continue
        for j in range(i + 1, min(n, i + 1 + window)):
            if direction == "bullish":
                if close[j] < s[j]:  # the cross failed — abandon
                    break
                if low[j] <= f[j] and close[j] > s[j]:
                    out.append({"confirm_pos": j, "direction": direction})
                    break
            else:
                if close[j] > s[j]:
                    break
                if high[j] >= f[j] and close[j] < s[j]:
                    out.append({"confirm_pos": j, "direction": direction})
                    break
    return out


def ema_cross_anticipation_touches(df, fast=9, slow=21, gap_atr=0.3, converge_bars=3):
    _, _, _, close = _ohlc(df)
    f, s = _ema(close, fast), _ema(close, slow)
    atr = ex.atr_series(df).to_numpy()
    n, start = len(close), 2 * slow
    gap = s - f  # >0: fast below slow (bullish cross pending); <0: fast above slow (bearish pending)
    out, armed = [], True
    for i in range(start + converge_bars, n):
        a = atr[i]
        if not a or np.isnan(a) or a <= 0:
            continue
        g = gap[i] / a
        if abs(g) > 2 * gap_atr:
            armed = True  # far apart again — a fresh approach can arm a new signal
        if not armed or gap[i] == 0:
            continue
        pending_bull = gap[i] > 0
        window = gap[i - converge_bars:i + 1]
        shrinking = all(abs(window[k + 1]) < abs(window[k]) for k in range(converge_bars))
        same_side = all((w > 0) == pending_bull for w in window)
        moving = (f[i] > f[i - 1]) if pending_bull else (f[i] < f[i - 1])
        if abs(g) <= gap_atr and shrinking and same_side and moving:
            out.append({"confirm_pos": i, "direction": "bullish" if pending_bull else "bearish"})
            armed = False
    return out


def ema_retrace_bounce_touches(df, fast=9, slow=21, trend_bars=20):
    _, high, low, close = _ohlc(df)
    f, s = _ema(close, fast), _ema(close, slow)
    n, start = len(close), 2 * slow + trend_bars
    out, armed = [], True
    for i in range(start, n):
        up = bool(np.all(f[i - trend_bars:i] > s[i - trend_bars:i])) and s[i] > s[i - 5]
        down = bool(np.all(f[i - trend_bars:i] < s[i - trend_bars:i])) and s[i] < s[i - 5]
        if up and close[i] > f[i]:
            armed = True  # price recovered above the fast EMA — the next dip is a new retracement
        if down and close[i] < f[i]:
            armed = True
        if not armed:
            continue
        if up and low[i] <= s[i] and close[i] > s[i]:
            out.append({"confirm_pos": i, "direction": "bullish"}); armed = False
        elif down and high[i] >= s[i] and close[i] < s[i]:
            out.append({"confirm_pos": i, "direction": "bearish"}); armed = False
    return out


def ema_equilibrium_touches(df, spans=(20, 50, 100), k=2.0):
    _, _, _, close = _ohlc(df)
    eq = np.mean([_ema(close, sp) for sp in spans], axis=0)
    atr = ex.atr_series(df).to_numpy()
    n, start = len(close), 2 * max(spans)
    out, armed = [], True
    for i in range(start, n):
        a = atr[i]
        if not a or np.isnan(a) or a <= 0:
            continue
        dev = (close[i] - eq[i]) / a
        if abs(dev) < k / 2:
            armed = True
        if not armed:
            continue
        if dev <= -k and close[i] > close[i - 1]:
            out.append({"confirm_pos": i, "direction": "bullish"}); armed = False
        elif dev >= k and close[i] < close[i - 1]:
            out.append({"confirm_pos": i, "direction": "bearish"}); armed = False
    return out


_FINDERS = {
    "ema_cross_pullback": ema_cross_pullback_touches,
    "ema_cross_anticipation": ema_cross_anticipation_touches,
    "ema_retrace_bounce": ema_retrace_bounce_touches,
    "ema_equilibrium": ema_equilibrium_touches,
}


def _builder(family):
    finder = _FINDERS[family]

    def build(df, rr_multiple=2.0, atr_mult_stop=1.5, max_bars=60, cost_bps=ex.DEFAULT_COST_BPS, **params):
        return ex._simulate_fixed_rr(df, finder(df, **params), rr_multiple, atr_mult_stop, max_bars, cost_bps)

    build.__name__ = f"build_{family}_fixed_rr_events"
    return build


BUILDERS = {f"{fam}_fixed_rr": _builder(fam) for fam in FAMILIES}
# Not yet in experiments.DETECTOR_LOGIC_VERSIONS — promoting any of these to
# live means adding them there (and to bot_engine's finder dispatch) so the
# strategy_id gate recognises them. The sweep script registers them for its
# own process only.
DETECTOR_VERSIONS = {fam: 1 for fam in FAMILIES}

PAIRS = [(9, 21), (20, 50), (50, 200)]
RR = [1.5, 3.0]


def settings_grid():
    """[(family, settings_dict)] — every setting tried, listed in one place so
    the multiple-testing burden is explicit (len of this list x tickers x
    timeframes = trials charged against the Benjamini-Hochberg correction)."""
    grid = []
    for fast, slow in PAIRS:
        for rr in RR:
            grid.append(("ema_cross_pullback", {"fast": fast, "slow": slow, "window": 20, "rr_multiple": rr}))
            grid.append(("ema_retrace_bounce", {"fast": fast, "slow": slow, "trend_bars": 20, "rr_multiple": rr}))
        for gap in (0.3, 0.6):
            grid.append(("ema_cross_anticipation", {"fast": fast, "slow": slow, "gap_atr": gap,
                                                     "converge_bars": 3, "rr_multiple": 2.0}))
    for spans in ((20, 50, 100), (20, 50)):
        for k in (2.0, 3.0):
            for rr in RR:
                grid.append(("ema_equilibrium", {"spans": spans, "k": k, "rr_multiple": rr}))
    return grid
