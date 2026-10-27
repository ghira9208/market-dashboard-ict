"""Small, shared building blocks for overlay studies and the forward test built on them: time-slice labels, the
per-combo touch finders, the EMA state at a bar, and the two statistical tests (within-combo permutation of a binary
feature, calendar-day cluster bootstrap). Kept out of the one-off studies so a scheduled job (ema_lead_forward.py)
never has to import a study."""
import numpy as np
import pandas as pd

import experiments as ex


def split_labels(n):
    a, b = int(n * 0.6), int(n * 0.8)
    return ["research"] * a + ["validation"] * (b - a) + ["locked"] * (n - b)


def _utc(x):
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def ema_state(df):
    close = df["Close"].to_numpy(dtype=float)
    f = pd.Series(close).ewm(span=20, adjust=False).mean().to_numpy()
    s = pd.Series(close).ewm(span=50, adjust=False).mean().to_numpy()
    atr = ex.atr_series(df).to_numpy()
    return close, f, s, atr


FIND = {
    "clean_expansion_fixed_rr": ex._clean_expansion_touches,
    "clean_expansion_liquidity_fixed_rr": ex._clean_expansion_liquidity_touches,
    "clean_retracement_resumption_fixed_rr": ex._clean_retracement_resumption_touches,
}


def strat_diff_p(df, col, rng, b=2000):
    """mean R (feature true) - mean R (feature false); p from shuffling the label within each combo."""
    groups = [(g["r"].to_numpy(), g[col].to_numpy().astype(bool)) for _, g in df.groupby("combo")]
    groups = [(r, m) for r, m in groups if 0 < m.sum() < len(m)]
    if not groups:
        return float("nan"), float("nan")

    def diff(masks):
        t = np.concatenate([r[m] for (r, _), m in zip(groups, masks)])
        f = np.concatenate([r[~m] for (r, _), m in zip(groups, masks)])
        return t.mean() - f.mean()

    obs_d = diff([m for _, m in groups])
    sims = np.array([diff([rng.permutation(m) for _, m in groups]) for _ in range(b)])
    return float(obs_d), float((1 + (sims >= obs_d).sum()) / (1 + b))


def day_cluster_ci(df, col, rng, b=2000):
    """95% CI of the pooled true-minus-false difference, resampling whole calendar days."""
    d = df.assign(day=df["entry"].dt.date)
    days = d["day"].unique()
    g = {k: v for k, v in d.groupby("day")}
    out = []
    for _ in range(b):
        s = pd.concat([g[k] for k in rng.choice(days, len(days))])
        t, f = s[s[col]]["r"], s[~s[col]]["r"]
        if len(t) and len(f):
            out.append(t.mean() - f.mean())
    return [round(float(np.percentile(out, 2.5)), 3), round(float(np.percentile(out, 97.5)), 3)]
