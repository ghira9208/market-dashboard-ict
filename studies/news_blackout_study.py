"""Would standing aside through scheduled news have improved the validated strategies?

Each validated combo is re-simulated on the same entries three ways:
  baseline          trades as usual
  calendar blackout flat through fixed release slots every weekday: 07:00 London, 08:30 and 10:00
                    New York (UK data, US data, ISM/sentiment) — the kind of rule you could
                    actually set in advance
  event blackout    flat through the moments a USD-wide shock actually happened (>= 4 of 7 USD
                    majors printing a bar >= 5x its normal range in the same bar). This peeks at
                    which days turned out to matter, so it is an UPPER BOUND on what any calendar
                    could achieve.
Flat means: a position open when the window starts is closed at that bar's open, and no entry is
taken inside the window. Windows last the release bar plus 30 minutes (5m data) / the release
hour (1h data). Costs are charged on the early exit like any other exit.

The answer is judged on per-day total R, bootstrapped over days (the same news hits every combo at
once), and the trades the rule touches are shown separately so the trade-off is visible.

    python3 studies/news_blackout_study.py
"""
import functools
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

MAJORS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "NZDUSD=X", "USDCAD=X", "USDCHF=X"]
RELEASES = [("Europe/London", 7 * 60), ("America/New_York", 8 * 60 + 30), ("America/New_York", 10 * 60)]
TF_MIN = {"5m": 5, "1h": 60}
AFTER_BARS = {"5m": 5, "1h": 0}     # release bar + 5 more five-minute bars = 30 min; one hour bar covers it
B = 4000


def _utc_index(df):
    idx = pd.DatetimeIndex(df.index)
    return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")


def expand(flags, after):
    """Mark `after` extra bars following every flagged bar."""
    out = flags.copy()
    for k in range(1, after + 1):
        out[k:] |= flags[:-k]
    return out


def calendar_flags(index_utc, tf):
    flags = np.zeros(len(index_utc), bool)
    step = TF_MIN[tf]
    for zone, rel in RELEASES:
        local = index_utc.tz_convert(zone)
        start = local.hour * 60 + local.minute
        flags |= (local.weekday < 5) & (start <= rel) & (rel < start + step)
    return expand(flags, AFTER_BARS[tf])


def shock_event_times(tf, fetch, k=4, mult=5.0):
    import data as research_data
    counts = {}
    for tk in MAJORS:
        d = research_data.get_yf_ohlcv(tk, **fetch)
        if d is None or len(d) < 500:
            continue
        rng = (d["High"] - d["Low"]).astype(float)
        base = rng.rolling(48).median().shift(1)
        hit = (rng >= mult * base)
        for ts in _utc_index(d)[hit.to_numpy()]:
            counts[ts] = counts.get(ts, 0) + 1
    return {ts for ts, c in counts.items() if c >= k}


def event_flags(index_utc, tf, times):
    flags = np.asarray(index_utc.isin(list(times)), bool)
    return expand(flags, AFTER_BARS[tf])


def run_variant(ex, builder, df, eff, flags):
    orig = ex._simulate_fixed_rr
    ex._simulate_fixed_rr = functools.partial(orig, blackout=flags) if flags is not None else orig
    try:
        return builder(df, **eff)
    finally:
        ex._simulate_fixed_rr = orig


def daily_r(events):
    if events is None or len(events) == 0:
        return pd.Series(dtype=float)
    t = pd.to_datetime(events["entry_time"], utc=True).dt.strftime("%Y-%m-%d")
    return events.groupby(t.values)["r_real"].sum()


def boot_diff(base_days, var_days, rng, b=B):
    days = sorted(set(base_days.index) | set(var_days.index))
    d = (var_days.reindex(days, fill_value=0.0) - base_days.reindex(days, fill_value=0.0)).to_numpy()
    idx = rng.integers(0, len(d), size=(b, len(d)))
    means = d[idx].sum(axis=1)
    return float(d.sum()), float(np.percentile(means, 5)), float(np.percentile(means, 95)), len(d)


def main():
    import experiments as ex
    import observatory as obs
    import ticker_behavior as tb
    rng = np.random.default_rng(23)
    lb, trials = tb.validated_leaderboard(), ex.load_experiment_trials()
    fetches = {"5m": obs.TF_FETCH["5m"], "1h": obs.TF_FETCH["1h"]}
    shocks = {tf: shock_event_times(tf, fetches[tf]) for tf in ("5m", "1h")}
    print({tf: f"{len(v)} USD-wide shock bars" for tf, v in shocks.items()})
    out = {}
    for tf in ("5m", "1h"):
        base_d, cal_d, evt_d = [], [], []
        n_base = n_cal = n_evt = 0
        touched_cal = touched_evt = 0
        r_touched_base = r_touched_cal = 0.0
        for _, c in lb[lb.tf_label == tf].iterrows():
            m = trials[(trials.ticker == c.ticker) & (trials.tf_label == tf) & (trials.label == c.best_label)]
            if m.empty:
                continue
            eff = obs.effective_settings({k: v for k, v in m.iloc[0]["settings"].items() if not k.startswith("_")},
                                         c.get("best_target_widen_mult"))
            try:
                df = obs._fetch_df(c.ticker, tf)
            except Exception:
                continue
            idx = _utc_index(df)
            builder = obs.BUILDER[c.strategy]
            base = run_variant(ex, builder, df, eff, None)
            cal = run_variant(ex, builder, df, eff, calendar_flags(idx, tf))
            evt = run_variant(ex, builder, df, eff, event_flags(idx, tf, shocks[tf]))
            base_d.append(daily_r(base)); cal_d.append(daily_r(cal)); evt_d.append(daily_r(evt))
            n_base += len(base); n_cal += len(cal); n_evt += len(evt)
        sumd = lambda parts: pd.concat(parts).groupby(level=0).sum() if parts else pd.Series(dtype=float)
        b, cl, ev = sumd(base_d), sumd(cal_d), sumd(evt_d)
        res = {"trades": {"baseline": n_base, "calendar": n_cal, "event": n_evt},
               "total_R": {"baseline": float(b.sum()), "calendar": float(cl.sum()), "event": float(ev.sum())}}
        for name, v in (("calendar", cl), ("event", ev)):
            diff, lo, hi, nd = boot_diff(b, v, rng)
            res[f"{name}_minus_baseline_R"] = {"diff": diff, "ci90": [lo, hi], "days": nd}
        out[tf] = res
    with open(os.path.join(HERE, ".cache", "news_blackout_study.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    res = main()
    for tf, r in res.items():
        print(f"\n{tf}: trades {r['trades']}  total R {{k: round(v,1) for k,v in r['total_R'].items()}}".replace("{k: round(v,1) for k,v in r['total_R'].items()}", str({k: round(v, 1) for k, v in r['total_R'].items()})))
        for name in ("calendar", "event"):
            x = r[f"{name}_minus_baseline_R"]
            print(f"   {name:8s} blackout vs baseline: {x['diff']:+.1f}R   90% CI {x['ci90'][0]:+.1f}..{x['ci90'][1]:+.1f}   ({x['days']} days)")
