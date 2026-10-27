"""Two OVERLAY filters, tested on the trades the validated combos already take
(they add no new trades — they only decide which existing ones to skip):

  1. Time of day — skip the worst entry hour. The hour is chosen using ONLY the
     earliest 60% of every combo's trades (RESEARCH), against a max-statistic
     permutation test (picking the worst of 24 hours is a selection, so the
     null must be the worst hour of shuffled data, not one fixed hour). The
     rule is then judged on the later VALIDATION and LOCKED slices it has never
     seen. The earlier all-history hour scan (16:00 UTC weak, 05:00 strong)
     used future data, so it is reported only as context, never as evidence.

  2. Higher-timeframe trend — take a signal only when the next timeframe up
     (5m -> 1h, 1h -> 1D) trends the same way. The rule is fixed in advance
     (two EMA pairs, no tuning on this data), so nothing is learned from the
     slices at all — each is simply a fresh out-of-sample test. A trade sees
     only HTF bars that had COMPLETED before it entered.

Both are judged by whether skipping the filtered trades would have raised mean
R per remaining trade, with a permutation p-value that respects each combo's
own baseline.

    python3 studies/overlay_study.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from overlay_tools import split_labels  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

HTF_OF = {"5m": ("1h", pd.Timedelta(hours=1)), "1h": ("1D", pd.Timedelta(days=1))}
HTF_FETCH = {"1h": dict(period="730d", interval="1h"), "1D": dict(period="10y", interval="1d")}
HTF_PAIRS = [(20, 50), (50, 200)]
MIN_HOUR_N = 60
B = 5000




def load_trades():
    import experiments as ex
    import observatory as obs
    import ticker_behavior as tb
    lb, trials, rows = tb.validated_leaderboard(), ex.load_experiment_trials(), []
    for _, c in lb.iterrows():
        m = trials[(trials.ticker == c.ticker) & (trials.tf_label == c.tf_label) & (trials.label == c.best_label)]
        if m.empty:
            continue
        settings = obs.effective_settings({k: v for k, v in m.iloc[0]["settings"].items() if not k.startswith("_")},
                                          c.get("best_target_widen_mult"))
        try:
            tr = obs.trade_detail(c.ticker, c.tf_label, c.strategy, settings)
        except Exception:
            continue
        tr = sorted(tr, key=lambda t: t["entry_time"])
        for t, sl in zip(tr, split_labels(len(tr))):
            rows.append({"combo": f"{c.ticker}|{c.tf_label}|{c.strategy}", "ticker": c.ticker, "tf": c.tf_label,
                         "entry_utc": pd.to_datetime(t["entry_time"], utc=True), "direction": t["direction"],
                         "r": t["r_real"], "slice": sl})
    df = pd.DataFrame(rows)
    df["hour"] = df["entry_utc"].dt.hour
    return df


# ------------------------------------------------------------ time of day --

def worst_hour_test(research, rng, b=B):
    """Pooled RESEARCH trades -> (worst_hour, best_hour, p_worst, p_best), p-values
    from the max-statistic null: shuffle hour labels, take the min (max) hour mean."""
    r, h = research["r"].to_numpy(), research["hour"].to_numpy()
    counts = np.bincount(h, minlength=24)
    ok = counts >= MIN_HOUR_N
    means = np.where(ok, np.bincount(h, weights=r, minlength=24) / np.maximum(counts, 1), np.nan)
    worst, best = int(np.nanargmin(means)), int(np.nanargmax(means))
    lo = hi = 0
    for _ in range(b):
        hs = rng.permutation(h)
        m = np.where(ok, np.bincount(hs, weights=r, minlength=24) / np.maximum(counts, 1), np.nan)
        lo += np.nanmin(m) <= means[worst]
        hi += np.nanmax(m) >= means[best]
    return worst, best, (1 + lo) / (1 + b), (1 + hi) / (1 + b), means


def subset_p(values, mask, rng, direction, b=B):
    """P(random subset of the same size is at least as extreme as the masked one)."""
    k = int(mask.sum())
    if k == 0 or k == len(values):
        return float("nan")
    obs = values[mask].mean()
    sims = np.array([values[rng.choice(len(values), k, replace=False)].mean() for _ in range(b)])
    return (1 + (sims <= obs).sum()) / (1 + b) if direction == "low" else (1 + (sims >= obs).sum()) / (1 + b)


def time_of_day(df, rng):
    research = df[df["slice"] == "research"]
    worst, best, p_w, p_b, means = worst_hour_test(research, rng)
    out = {"worst_hour_utc": worst, "best_hour_utc": best, "research_p_worst": round(p_w, 4),
           "research_p_best": round(p_b, 4), "research_mean_worst": round(float(means[worst]), 3),
           "research_mean_best": round(float(means[best]), 3), "research_mean_all": round(float(research["r"].mean()), 3)}
    for sl in ("validation", "locked"):
        s = df[df["slice"] == sl]
        v, hrs = s["r"].to_numpy(), s["hour"].to_numpy()
        w_mask = hrs == worst
        out[sl] = {
            "n": int(len(s)), "n_worst_hour": int(w_mask.sum()),
            "mean_all": round(float(v.mean()), 3),
            "mean_worst_hour": round(float(v[w_mask].mean()), 3) if w_mask.any() else None,
            "mean_without_worst": round(float(v[~w_mask].mean()), 3),
            "p_worst_is_weaker": round(float(subset_p(v, w_mask, rng, "low")), 4),
            "total_R_saved_by_skipping": round(float(-v[w_mask].sum()), 2),
            "mean_best_hour": round(float(v[hrs == best].mean()), 3) if (hrs == best).any() else None,
            "p_best_is_stronger": round(float(subset_p(v, hrs == best, rng, "high")), 4) if (hrs == best).any() else None,
        }
    return out


# ----------------------------------------------------- higher-timeframe trend --

def htf_trend_frame(htf_df, fast, slow, bar_td):
    """Per HTF bar: the trend sign and the moment that bar COMPLETED (open time +
    its length). NaN for the EMA warm-up, where the average is mostly its seed."""
    close = htf_df["Close"].to_numpy(dtype=float)
    f = pd.Series(close).ewm(span=fast, adjust=False).mean().to_numpy()
    s = pd.Series(close).ewm(span=slow, adjust=False).mean().to_numpy()
    trend = np.sign(f - s)
    trend[: 2 * slow] = np.nan
    idx = pd.DatetimeIndex(htf_df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return pd.DataFrame({"avail": idx + bar_td, "trend": trend}).sort_values("avail").reset_index(drop=True)


def attach_trend(trades, frame):
    t = trades.sort_values("entry_utc").copy()
    t["entry_utc"] = t["entry_utc"].astype("datetime64[ns, UTC]")
    frame = frame.copy()
    frame["avail"] = frame["avail"].astype("datetime64[ns, UTC]")
    merged = pd.merge_asof(t, frame, left_on="entry_utc", right_on="avail", direction="backward")
    merged["aligned"] = np.where(merged["trend"].isna(), np.nan,
                                  ((merged["direction"] == "bullish") & (merged["trend"] > 0)
                                   | (merged["direction"] == "bearish") & (merged["trend"] < 0)).astype(float))
    return merged


def stratified_diff_p(df, rng, b=B):
    """aligned - counter mean R, p-value from shuffling the aligned labels WITHIN each combo
    (so a combo that is simply better than another cannot fake an effect)."""
    groups = [(g["r"].to_numpy(), g["aligned"].to_numpy().astype(bool)) for _, g in df.groupby("combo")]
    def diff(shufflers):
        a = np.concatenate([r[m] for (r, _), m in zip(groups, shufflers)])
        c = np.concatenate([r[~m] for (r, _), m in zip(groups, shufflers)])
        return (a.mean() - c.mean()) if len(a) and len(c) else np.nan
    obs = diff([m for _, m in groups])
    sims = np.array([diff([rng.permutation(m) for _, m in groups]) for _ in range(b)])
    return float(obs), float((1 + (sims >= obs).sum()) / (1 + b))


def htf_filter(df, rng, fast, slow):
    import data as research_data
    parts = []
    for (ticker, tf), g in df.groupby(["ticker", "tf"]):
        if tf not in HTF_OF:
            continue
        htf_tf, bar_td = HTF_OF[tf]
        try:
            htf = research_data.get_yf_ohlcv(ticker, **HTF_FETCH[htf_tf])
        except Exception:
            continue
        if htf is None or htf.empty:
            continue
        parts.append(attach_trend(g, htf_trend_frame(htf, fast, slow, bar_td)))
    if not parts:
        return {}
    all_t = pd.concat(parts, ignore_index=True)
    all_t = all_t[all_t["aligned"].notna()]
    out = {"pair": f"{fast}/{slow}", "n_with_htf": int(len(all_t))}
    for sl in ("research", "validation", "locked"):
        s = all_t[all_t["slice"] == sl]
        if len(s) < 40 or s["aligned"].nunique() < 2:
            out[sl] = {"n": int(len(s))}
            continue
        d, p = stratified_diff_p(s, rng, b=2000)
        out[sl] = {"n": int(len(s)), "pct_aligned": round(float(s["aligned"].mean() * 100), 1),
                   "mean_aligned": round(float(s[s.aligned == 1]["r"].mean()), 3),
                   "mean_counter": round(float(s[s.aligned == 0]["r"].mean()), 3),
                   "diff": round(d, 3), "p_aligned_better": round(p, 4),
                   "mean_all": round(float(s["r"].mean()), 3),
                   "mean_if_counter_skipped": round(float(s[s.aligned == 1]["r"].mean()), 3),
                   "n_skipped": int((s.aligned == 0).sum())}
    return out


def main():
    rng = np.random.default_rng(7)
    df = load_trades()
    print(f"{len(df)} trades across {df['combo'].nunique()} validated combos; "
          f"slices: {df['slice'].value_counts().to_dict()}\n")
    tod = time_of_day(df, rng)
    print("TIME OF DAY (rule learned on RESEARCH only)")
    print(json.dumps(tod, indent=2))
    results = {"time_of_day": tod, "htf": []}
    for fast, slow in HTF_PAIRS:
        r = htf_filter(df, rng, fast, slow)
        results["htf"].append(r)
        print(f"\nHTF TREND FILTER, EMA {fast}/{slow} on the next timeframe up (5m->1h, 1h->1D)")
        print(json.dumps(r, indent=2))
    with open(os.path.join(HERE, ".cache", "overlay_study.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
