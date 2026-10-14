"""A one-line fingerprint of the whole market for each day, so "what is today
like?" becomes "which past days looked like today, and what did the strategies
do on them?" — instead of reading isolated historical averages.

Per day, across a basket of liquid assets (daily bars only, no lookahead —
every feature at day t uses data up to and including t):

  activity     median over assets of that day's high-low range, in ATR(14) units.
               ~1.0 is a normal day; well below is a sleepy one.
  vol_pct      median over assets of where 20-day realized volatility sits in its
               own trailing-250-day range (0 = calmest of the year, 1 = wildest).
  comove       average pairwise correlation of daily returns over the last 20
               days. High = one market-wide factor drives everything (assets are
               NOT independent bets); low = assets are doing their own thing.
  breadth      share of assets trading above their 50-day average (1 = everything
               trending up, 0 = everything down, 0.5 = mixed/ranging).

analog_days() returns the past days whose fingerprint is closest to a target day
(standardised distance), excluding a buffer of recent days so the future can
never leak into its own comparison.
"""
import numpy as np
import pandas as pd

FEATURES = ["activity", "vol_pct", "comove", "breadth"]


def _atr(df, n=14):
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def _norm_index(df):
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize(None) if idx.tz is not None else idx
    out = df.copy()
    out.index = idx.normalize()
    return out[~out.index.duplicated(keep="last")]


def day_features(frames, corr_window=20, vol_window=20, rank_window=250):
    """frames: {ticker: daily OHLC DataFrame}. -> DataFrame indexed by weekday with FEATURES.

    Assets trade on different calendars (crypto on weekends, FX/indices off on their own
    holidays), so everything is aligned to weekdays; a gap is forward-filled for at most 3
    days (a flat close = a zero return, not a missing one) and the per-day medians skip
    assets that genuinely did not trade."""
    frames = {t: _norm_index(f) for t, f in frames.items() if f is not None and len(f) > rank_window + 30}
    days = pd.bdate_range(min(f.index.min() for f in frames.values()), max(f.index.max() for f in frames.values()))
    rng_atr, vol_rank, above, rets = {}, {}, {}, {}
    for t, f in frames.items():
        close = f["Close"].astype(float)
        rng_atr[t] = ((f["High"] - f["Low"]) / _atr(f).replace(0, np.nan)).reindex(days)
        aligned = close.reindex(days).ffill(limit=3)
        rets[t] = aligned.pct_change()
        rv = rets[t].rolling(vol_window, min_periods=int(vol_window * 0.8)).std()
        vol_rank[t] = rv.rolling(rank_window, min_periods=int(rank_window * 0.8)).apply(
            lambda w: (w[:-1] < w[-1]).mean() if not np.isnan(w[-1]) else np.nan, raw=True)
        above[t] = (aligned > aligned.ewm(span=50, adjust=False, ignore_na=True).mean()).astype(float).where(aligned.notna())
    activity = pd.DataFrame(rng_atr).median(axis=1)
    vol_pct = pd.DataFrame(vol_rank).median(axis=1)
    breadth = pd.DataFrame(above).mean(axis=1)
    R = pd.DataFrame(rets)
    vals = R.to_numpy()
    comove = pd.Series(index=R.index, dtype=float)
    for i in range(corr_window, len(R)):
        w = vals[i - corr_window + 1:i + 1]
        ok = np.isnan(w).mean(axis=0) <= 0.2
        if ok.sum() < 6:
            continue
        w = np.nan_to_num(w[:, ok], nan=0.0)
        if (w.std(axis=0) == 0).any():
            keep = w.std(axis=0) > 0
            w = w[:, keep]
        if w.shape[1] < 6:
            continue
        c = np.corrcoef(w.T)
        comove.iloc[i] = (c.sum() - w.shape[1]) / (w.shape[1] * (w.shape[1] - 1))
    out = pd.DataFrame({"activity": activity, "vol_pct": vol_pct, "comove": comove, "breadth": breadth})
    return out.dropna()


def analog_days(features, target_day, k=30, buffer_days=10):
    """The k past days (at least `buffer_days` before target_day) closest to it in
    z-scored feature space. Returns a DataFrame with the distance."""
    target_day = pd.Timestamp(target_day).normalize()
    f = features[features.index <= target_day]
    mu, sd = f.mean(), f.std(ddof=0).replace(0, 1)
    z = (f - mu) / sd
    t = z.loc[target_day] if target_day in z.index else z.iloc[-1]
    pool = z[z.index <= target_day - pd.Timedelta(days=buffer_days)]
    dist = np.sqrt(((pool - t) ** 2).sum(axis=1))
    near = dist.nsmallest(k)
    return features.loc[near.index].assign(distance=near.values)


def percentile_of(features, day, col):
    """Where `day`'s value of `col` ranks among all earlier days (0..1)."""
    day = pd.Timestamp(day).normalize()
    hist = features[col][features.index < day]
    return float((hist < features.loc[day, col]).mean())
