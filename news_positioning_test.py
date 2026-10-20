"""What if, instead of standing aside from scheduled news, you POSITION for it?

Fixed release slots only (07:00 London, 08:30 and 10:00 New York, weekdays) — set in advance, no
hindsight about which days turned out to matter. Three ways to be in the market going into a slot:

  with_trend   enter at the start of the release window in the direction of the trend of the
               previous L bars (the move running into the news keeps going)
  against_trend the same entry, opposite direction (the run-up gets faded)
  straddle     leave a buy-stop and a sell-stop `k` ATR either side of the last close; the first one
               touched becomes the trade and the other is cancelled. If one bar touches BOTH, it is
               booked as a full loss (bar data cannot say which came first — the conservative call).
               Fills are assumed AT the bracket price: no slippage, which is the best case, so the
               result is an UPPER bound for this idea.

Every trade: protective stop 1.5 x ATR(14) at entry, time exit after H bars, 2 bps cost, stop assumed
first when a bar could hit both stop and anything else. R = profit / stop distance. Results are
averaged per day across assets (the same news moves every pair) and bootstrapped over days.

    python3 news_positioning_test.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import experiments as ex  # noqa: E402
from news_blackout_test import RELEASES, TF_MIN, _utc_index  # noqa: E402

UNIVERSE = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "NZDUSD=X", "USDCAD=X", "USDCHF=X", "EURJPY=X",
            "GBPJPY=X", "GC=F", "ES=F", "NQ=F", "BTC-USD"]
FETCH = {"1h": dict(period="730d", interval="1h"), "5m": dict(period="60d", interval="5m")}
LOOKBACK = {"1h": 6, "5m": 12}
HOLD = {"1h": 3, "5m": 12}
STRADDLE_WINDOW = {"1h": 1, "5m": 6}
STOP_ATR, K_BRACKET, COST_BPS, B = 1.5, 1.0, 2.0, 4000


def release_positions(index_utc, tf):
    """Bar positions where a release slot starts inside the bar (5m: the bar BEFORE it, so we are in
    first). Returns [(slot_name, pos)]."""
    step, out = TF_MIN[tf], []
    for zone, rel in RELEASES:
        local = index_utc.tz_convert(zone)
        start = local.hour * 60 + local.minute
        hit = (local.weekday < 5) & (start <= rel) & (rel < start + step)
        for pos in np.flatnonzero(hit):
            out.append((f"{zone.split('/')[1]} {rel // 60:02d}:{rel % 60:02d}", int(pos) - (1 if tf == "5m" else 0)))
    return out


def walk(high, low, close, entry_idx, entry_price, sign, risk, hold, n):
    """R of a trade entered at entry_price on bar entry_idx: stop 1.5 ATR, time exit after `hold` bars."""
    stop = entry_price - sign * risk
    cost = entry_price * COST_BPS / 1e4
    last = min(n - 1, entry_idx + hold - 1)
    for j in range(entry_idx, last + 1):
        if (sign > 0 and low[j] <= stop) or (sign < 0 and high[j] >= stop):
            return (-risk - cost) / risk
    return (sign * (close[last] - entry_price) - cost) / risk


def trades_for(df, tf, slots):
    o, h, l, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    atr = ex.atr_series(df).to_numpy()
    n, L, H, W = len(df), LOOKBACK[tf], HOLD[tf], STRADDLE_WINDOW[tf]
    idx = _utc_index(df)
    rows = []
    for slot, p in slots:
        if p < L + 15 or p + H + W + 1 >= n or not np.isfinite(atr[p - 1]) or atr[p - 1] <= 0:
            continue
        a, ref = atr[p - 1], c[p - 1]
        risk = STOP_ATR * a
        day = idx[p].strftime("%Y-%m-%d")
        trend = np.sign(c[p - 1] - c[p - 1 - L])
        if trend != 0:
            for mode, sign in (("with_trend", trend), ("against_trend", -trend)):
                rows.append({"day": day, "slot": slot, "mode": mode, "R": walk(h, l, c, p, o[p], sign, risk, H, n)})
        up, dn = ref + K_BRACKET * a, ref - K_BRACKET * a
        r = None
        for j in range(p, p + W):
            hit_up, hit_dn = h[j] >= up, l[j] <= dn
            if hit_up and hit_dn:
                r = -1.0 - (ref * COST_BPS / 1e4) / risk          # whipsaw: both sides touched in one bar
                break
            if hit_up or hit_dn:
                sign, price = (1, up) if hit_up else (-1, dn)
                # the triggering bar's own remaining path is unknown: stop first if it could have reached it
                stop = price - sign * risk
                if (sign > 0 and l[j] <= stop) or (sign < 0 and h[j] >= stop):
                    r = -1.0 - (price * COST_BPS / 1e4) / risk
                else:
                    r = walk(h, l, c, min(j + 1, n - 1), price, sign, risk, H, n) if j + 1 < n else None
                break
        if r is not None:
            rows.append({"day": day, "slot": slot, "mode": "straddle", "R": r})
    return rows


def summarize(trades, rng, b=B):
    out = []
    for (tf, mode), g in trades.groupby(["tf", "mode"]):
        per_day = g.groupby("day")["R"].mean()
        v = per_day.to_numpy()
        means = v[rng.integers(0, len(v), size=(b, len(v)))].mean(axis=1)
        out.append({"tf": tf, "mode": mode, "trades": len(g), "event_days": len(v), "mean_R": float(v.mean()),
                    "ci90_lo": float(np.percentile(means, 5)), "ci90_hi": float(np.percentile(means, 95)),
                    "win_pct": float((g["R"] > 0).mean() * 100)})
    return pd.DataFrame(out)


def main():
    import data as research_data
    rng = np.random.default_rng(31)
    rows = []
    for tf in ("1h", "5m"):
        for tk in UNIVERSE:
            try:
                df = research_data.get_yf_ohlcv(tk, **FETCH[tf])
            except Exception:
                continue
            if df is None or len(df) < 800:
                continue
            slots = release_positions(_utc_index(df), tf)
            for r in trades_for(df, tf, slots):
                rows.append({"tf": tf, "tk": tk, **r})
    t = pd.DataFrame(rows)
    res = summarize(t, rng)
    by_slot = t.groupby(["tf", "mode", "slot"])["R"].agg(["size", "mean"]).reset_index()
    res.to_json(os.path.join(HERE, ".cache", "news_positioning_test.json"), orient="records", indent=1)
    return res, by_slot, t


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    res, by_slot, t = main()
    print(res.round(3).to_string(index=False))
    print("\nby release slot (mean R, trades):")
    print(by_slot.pivot_table(index=["tf", "slot"], columns="mode", values="mean").round(3).to_string())
