"""Does acting only in "predictable waters" (quiet volatility, no scheduled news ahead) help?

Calm is defined from what is known at entry, in advance:
  quiet       the ATR at entry is in the lowest third of the previous 250 bars
  no news     no release slot (07:00 London, 08:30 and 10:00 New York, weekdays) in the bar being entered or
              any bar of the holding window
A) the small-target / wide-stop "high win rate" structure (scalp_winrate_study.py) restricted to calm entries,
   with an 8-hour limit so a no-news window can exist at all (a 2-day hold always spans a release);
B) the 82 validated strategies' own trades, split by whether they were entered in quiet conditions and whether
   a release fell inside the trade.

    python3 studies/calm_waters_study.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import scalp_winrate_study as sw  # noqa: E402
from news_blackout_study import _utc_index, calendar_flags  # noqa: E402

HOLD = 8


def causal_quiet(atr, window=250, q=1 / 3):
    s = pd.Series(atr)
    return (s <= s.rolling(window).quantile(q).shift(1)).to_numpy()


def window_has_release(flags, idx, hold):
    out = np.zeros(len(idx), bool)
    for k in range(0, hold + 2):
        out |= flags[np.minimum(idx + 1 + k, len(flags) - 1)]
    return out


def part_a():
    import data as research_data
    sw.HOLD = HOLD
    pool = {}
    for tk in sw.ASSETS:
        try:
            df = research_data.get_yf_ohlcv(tk, period="730d", interval="1h")
        except Exception:
            continue
        if df is None or len(df) < 2000:
            continue
        o, h, l, c, atr, idx = sw.entries(df)
        flags = calendar_flags(_utc_index(df), "1h")
        quiet = causal_quiet(atr)[idx]
        news = window_has_release(flags, idx, HOLD)
        sets = {"all entries": np.ones(len(idx), bool), "quiet only": quiet, "no news ahead only": ~news, "quiet AND no news": quiet & ~news}
        for T, S in [(0.25, 5.0), (0.25, 10.0), (0.5, 5.0), (0.5, 10.0)]:
            for d in (1, -1):
                p, k = sw.outcome(o, h, l, c, atr, idx, d, T, S)
                for name, m in sets.items():
                    pool.setdefault((T, S, name), []).append((p[m], k[m]))
    rows = []
    for (T, S, name), parts in pool.items():
        p = np.concatenate([x[0] for x in parts]); k = np.concatenate([x[1] for x in parts])
        rho = S / T
        rows.append({"target": f"{T}A", "stop": f"{S}A", "entries": name, "trades": len(p), "win_rate": 100 * (p > 0).mean(),
                     "break_even_win_rate": 100 * rho / (1 + rho), "stopped_pct": 100 * (k == -1).mean(), "mean_pnl_targets": p.mean()})
    return pd.DataFrame(rows)


def part_b():
    import functools
    import experiments as ex
    import observatory as obs
    import ticker_behavior as tb
    lb, trials = tb.validated_leaderboard(), ex.load_experiment_trials()
    rows = []
    for _, c in lb.iterrows():
        m = trials[(trials.ticker == c.ticker) & (trials.tf_label == c.tf_label) & (trials.label == c.best_label)]
        if m.empty:
            continue
        eff = obs.effective_settings({k: v for k, v in m.iloc[0]["settings"].items() if not k.startswith("_")}, c.get("best_target_widen_mult"))
        if c.tf_label not in ("5m", "1h"):
            continue                        # daily bars: a release slot is not a meaningful sub-bar event
        try:
            df = obs._fetch_df(c.ticker, c.tf_label)
        except Exception:
            continue
        ev = obs.BUILDER[c.strategy](df, **eff)
        if ev.empty:
            continue
        idx_utc = _utc_index(df)
        flags = calendar_flags(idx_utc, c.tf_label)
        pos = pd.Series(np.arange(len(df)), index=idx_utc)
        e = pos.reindex(pd.to_datetime(ev["entry_time"], utc=True)).to_numpy()
        x = pos.reindex(pd.to_datetime(ev["exit_time"], utc=True)).to_numpy()
        csum = np.concatenate([[0], np.cumsum(flags)])
        has_news = np.array([(csum[int(b) + 1] - csum[int(a)]) > 0 if np.isfinite(a) and np.isfinite(b) else False for a, b in zip(e, x)])
        for (_, r), hn in zip(ev.iterrows(), has_news):
            rows.append({"tf": c.tf_label, "R": r["r_real"], "vol": r["vol_regime"], "news_inside": bool(hn),
                         "day": pd.to_datetime(r["entry_time"], utc=True).strftime("%Y-%m-%d")})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    if "--skip-a" not in sys.argv:
        a = part_a()
        print("A) small target / wide stop, 8-hour limit, 2 bps cost — does calm shift the win rate above break-even?\n")
        print(a.round(2).to_string(index=False))
    b = part_b()
    rng = np.random.default_rng(3)
    print("\nB) the validated strategies' own trades (%d), by entry conditions:" % len(b))
    b["quiet"] = b.vol == "low"
    for name, m in [("quiet entry, no release during the trade", b.quiet & ~b.news_inside), ("quiet entry, release during the trade", b.quiet & b.news_inside),
                    ("normal/high vol, no release", ~b.quiet & ~b.news_inside), ("normal/high vol, release during", ~b.quiet & b.news_inside)]:
        x = b[m]; print(f"  {name:45s} n={len(x):5d}  mean R {x.R.mean():+.2f}  total R {x.R.sum():+8.0f}")
    for name, m in [("quiet vs not (mean R difference)", b.quiet), ("no release vs release inside", ~b.news_inside)]:
        d = b.groupby(["day", m.rename("g")]).R.mean().unstack()
        diff = (d[True] - d[False]).dropna(); bs = [diff.sample(len(diff), replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(2000)]
        print(f"  {name:36s} {diff.mean():+.3f}R per trade-day, 90% CI {np.percentile(bs,5):+.3f}..{np.percentile(bs,95):+.3f}  ({len(diff)} days)")
