"""What does a very high win rate cost? Take-profit a small fraction of an ATR away, stop many ATRs away.

Entries are NOT a strategy: every 6th hourly bar, in BOTH directions (so any direction edge cancels and only
the exit structure — small target, wide stop — is measured), on 12 liquid assets over ~2 years. Each trade
exits at the first of: target hit (win), stop hit (loss), or the time limit (marked to market at the close).
When one bar could hit both, the stop is assumed first. A round-trip cost is charged on every trade.

    python3 scalp_winrate_study.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import experiments as ex  # noqa: E402

ASSETS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "NZDUSD=X", "USDCAD=X", "USDCHF=X", "EURJPY=X", "GC=F", "CL=F", "BTC-USD", "ETH-USD"]
TARGETS = [0.1, 0.25, 0.5, 1.0]            # ATRs
STOPS = [1.0, 2.0, 5.0, 10.0, 20.0, 1e9]   # ATRs (1e9 = no stop, time limit only)
HOLD = 48                                   # bars (hours): two days
STEP = 6
COST_BPS = 2.0


def entries(df):
    o, h, l, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    atr = ex.atr_series(df).to_numpy()
    idx = np.arange(30, len(df) - HOLD - 1, STEP)
    idx = idx[np.isfinite(atr[idx]) & (atr[idx] > 0)]
    return o, h, l, c, atr, idx


def outcome(o, h, l, c, atr, idx, direction, T, S):
    """R in units of the TARGET distance (+1 = target hit), after cost. Returns (pnl_in_targets, kind)."""
    entry = o[idx + 1]
    a = atr[idx]
    tgt = T * a
    stp = np.minimum(S, 1e6) * a
    n = len(idx)
    pnl = np.full(n, np.nan)
    kind = np.zeros(n, dtype=np.int8)           # 1 win, -1 stop, 0 timed out
    alive = np.ones(n, bool)
    for k in range(1, HOLD + 1):
        hi, lo = h[idx + k], l[idx + k]
        fav = (hi - entry) if direction > 0 else (entry - lo)
        adv = (entry - lo) if direction > 0 else (hi - entry)
        hit_stop = alive & (adv >= stp) & (S < 1e8)
        hit_tgt = alive & ~hit_stop & (fav >= tgt)
        pnl[hit_stop] = -stp[hit_stop] / tgt[hit_stop]; kind[hit_stop] = -1
        pnl[hit_tgt] = 1.0; kind[hit_tgt] = 1
        alive &= ~(hit_stop | hit_tgt)
    last = c[idx + HOLD]
    move = (last - entry) if direction > 0 else (entry - last)
    pnl[alive] = move[alive] / tgt[alive]
    cost = entry * COST_BPS / 1e4 / tgt
    return pnl - cost, kind


def main():
    import data as research_data
    pool = {}
    for tk in ASSETS:
        try:
            df = research_data.get_yf_ohlcv(tk, period="730d", interval="1h")
        except Exception:
            continue
        if df is None or len(df) < 2000:
            continue
        o, h, l, c, atr, idx = entries(df)
        day = pd.DatetimeIndex(df.index).tz_localize(None).normalize()[idx + 1] if pd.DatetimeIndex(df.index).tz is None else pd.DatetimeIndex(df.index).tz_convert(None).normalize()[idx + 1]
        for T in TARGETS:
            for S in STOPS:
                for d in (1, -1):
                    p, k = outcome(o, h, l, c, atr, idx, d, T, S)
                    pool.setdefault((T, S), []).append(pd.DataFrame({"tk": tk, "day": day, "pnl": p, "kind": k, "price_atr": o[idx + 1] / atr[idx]}))
    rows = []
    for (T, S), parts in pool.items():
        d = pd.concat(parts)
        wins = d.pnl > 0
        by_day = d.groupby("day").pnl.sum()
        daily = by_day.to_numpy()
        rows.append({"target_ATR": T, "stop_ATR": "none" if S > 1e8 else S, "trades": len(d), "win_rate_pct": 100 * wins.mean(),
                     "stopped_pct": 100 * (d.kind == -1).mean(), "avg_loss_in_targets": float(d.pnl[d.pnl < 0].mean()) if (d.pnl < 0).any() else 0.0,
                     "mean_pnl_in_targets": d.pnl.mean(), "worst_day_in_targets": daily.min(), "median_price_in_atr": float(d.price_atr.median())})
    return pd.DataFrame(rows), pool


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    res, pool = main()
    print(res.round(3).to_string(index=False))
    res.to_json(os.path.join(HERE, ".cache", "scalp_winrate_study.json"), orient="records", indent=1)
