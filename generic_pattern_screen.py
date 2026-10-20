"""A pre-registered screen of GENERAL market patterns — ones that don't depend on a particular
strategy's settings — asking which of them repeat.

Every statistic is built so it means the same thing on every asset (no USD-quote conventions):
the signal's sign times the outcome, in the asset's own standard-deviation units, so + means
"continues" and - means "reverses". Per class the assets are averaged on each day (the same news
moves every USD pair at once, so a day, not an asset-day, is the independent unit), then bootstrapped
over days. Everything is corrected together (Benjamini-Hochberg over all tests) and must repeat in
BOTH halves of the sample to count. Two positive controls (volatility clustering, the hour-of-day
volatility profile) are included: if the screen cannot find those, the machinery is broken.

    python3 generic_pattern_screen.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CLASSES = {
    "fx": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "NZDUSD=X", "USDCAD=X", "USDCHF=X", "EURGBP=X",
           "EURJPY=X", "GBPJPY=X", "AUDJPY=X"],
    "metals_oil": ["GC=F", "SI=F", "CL=F", "PL=F", "PA=F"],
    "index": ["ES=F", "NQ=F", "YM=F", "^GDAXI", "^FTSE", "^N225"],
    "crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "LTC-USD"],
}
SESSION_CLASSES = ("fx", "metals_oil", "crypto")           # trade (nearly) around the clock
ASIA, LONDON, NY = range(0, 7), range(7, 12), range(12, 21)  # UTC hours
B = 3000


def _utc(df):
    idx = pd.DatetimeIndex(df.index)
    out = df.copy()
    out.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return out


def session_frame(h):
    """One row per UTC date: log returns of the Asia / London / NY windows, plus the Asian range and
    the London-breakout reading at 09:00."""
    h = _utc(h)
    h = h[~h.index.duplicated(keep="last")]
    rows = {}
    for day, g in h.groupby(h.index.date):
        hrs = g.index.hour
        def win(r):
            m = np.isin(hrs, list(r))
            return g[m] if m.sum() >= max(2, len(list(r)) // 2) else None
        a, l, n = win(ASIA), win(LONDON), win(NY)
        if a is None or l is None or n is None:
            continue
        ret = lambda w: float(np.log(w["Close"].iloc[-1] / w["Open"].iloc[0]))
        row = {"asia": ret(a), "london": ret(l), "ny": ret(n)}
        first = g[g.index.hour == 12]
        rest = g[(g.index.hour >= 13) & (g.index.hour <= 20)]
        if len(first) and len(rest) >= 4:
            row["ny_first"] = float(np.log(first["Close"].iloc[0] / first["Open"].iloc[0]))
            row["ny_rest"] = float(np.log(rest["Close"].iloc[-1] / rest["Open"].iloc[0]))
        brk = g[(g.index.hour >= 7) & (g.index.hour <= 8)]
        late = g[(g.index.hour >= 9) & (g.index.hour <= 15)]
        if len(brk) == 2 and len(late) >= 4:
            hi, lo = a["High"].max(), a["Low"].min()
            c9 = brk["Close"].iloc[-1]
            row["brk_sign"] = 1.0 if c9 > hi else -1.0 if c9 < lo else 0.0
            row["brk_out"] = float(np.log(late["Close"].iloc[-1] / c9))
        rows[pd.Timestamp(day)] = row
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def _scaled(x):
    s = np.nanstd(x)
    return x / s if s and np.isfinite(s) else x * np.nan


def session_stats(f):
    """Per-asset daily series (already sign x outcome in std units) for the four session patterns."""
    out = {}
    out["S1 Asia -> London"] = np.sign(f["asia"]) * _scaled(f["london"])
    out["S2 London -> NY"] = np.sign(f["london"]) * _scaled(f["ny"])
    if "ny_first" in f:
        out["S3 first NY hour -> rest of NY"] = np.sign(f["ny_first"]) * _scaled(f["ny_rest"])
    if "brk_sign" in f:
        m = f["brk_sign"] != 0
        out["S4 London breakout of the Asian range"] = (f["brk_sign"] * _scaled(f["brk_out"])).where(m)
    return out


def daily_stats(d):
    """Per-asset daily series for the daily patterns (d: daily OHLC)."""
    c = _utc(d)["Close"].astype(float)
    c.index = c.index.tz_localize(None).normalize()
    c = c[~c.index.duplicated(keep="last")]
    r = np.log(c / c.shift(1)).dropna()
    z = _scaled(r)
    prev = z.shift(1)
    out = {"D1 yesterday -> today": np.sign(prev) * z,
           "D2 after a >2 sd day -> today": (np.sign(prev) * z).where(prev.abs() > 2)}
    w = np.log(c.resample("W-FRI").last()).diff().dropna()
    wz = _scaled(w)
    out["D3 last week -> this week"] = np.sign(wz.shift(1)) * wz
    a = z.abs()
    out["V1 volatility clustering (corr |ret|)"] = (a - a.mean()) * (a.shift(1) - a.mean()) / (a.std() ** 2)
    ext = z - z.mean()
    for i, name in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"]):
        out[f"C2 {name} return vs other days"] = ext.where(ext.index.dayofweek == i)
    month_end = ext.index.to_series().groupby([ext.index.year, ext.index.month]).transform("max")
    pos_from_end = ext.index.to_series().groupby([ext.index.year, ext.index.month]).cumcount(ascending=False)
    pos_from_start = ext.index.to_series().groupby([ext.index.year, ext.index.month]).cumcount()
    turn = (pos_from_end == 0) | (pos_from_start <= 2)
    out["C1 turn of month (last day + first 3) vs other days"] = ext.where(turn)
    return out


def pooled_by_day(per_asset):
    """{pattern: {asset: Series}} -> {pattern: Series averaged over assets per day}."""
    out = {}
    for pat, d in per_asset.items():
        df = pd.DataFrame(d)
        out[pat] = df.mean(axis=1, skipna=True).dropna()
    return out


def test_series(x, rng, b=B):
    x = x.dropna()
    n = len(x)
    if n < 40:
        return None
    v = x.to_numpy()
    idx = rng.integers(0, n, size=(b, n))
    means = v[idx].mean(axis=1)
    mean = float(v.mean())
    p = float(min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean()) + 1.0 / (b + 1)))
    half = n // 2
    return {"n_days": n, "mean": mean, "ci_lo": float(np.percentile(means, 5)), "ci_hi": float(np.percentile(means, 95)),
            "p": p, "first_half": float(v[:half].mean()), "second_half": float(v[half:].mean()),
            "continues_pct": float((v > 0).mean() * 100)}


def bh(ps, q=0.10):
    order = np.argsort(ps)
    m = len(ps)
    passed = np.zeros(m, bool)
    thresh = -1
    for rank, i in enumerate(order, 1):
        if ps[i] <= q * rank / m:
            thresh = rank
    for rank, i in enumerate(order, 1):
        passed[i] = rank <= thresh
    return passed


def hourly_vol_profile(h):
    """Share of the day's total |return| in each UTC hour, per asset -> positive control."""
    h = _utc(h)
    r = np.log(h["Close"] / h["Open"]).abs()
    by = r.groupby(h.index.hour).mean()
    return (by / by.sum()).reindex(range(24))


def main():
    import data as research_data
    rng = np.random.default_rng(11)
    pools = {}
    for cls, tickers in CLASSES.items():
        per = {}
        prof = []
        for tk in tickers:
            try:
                dd = research_data.get_yf_ohlcv(tk, period="10y", interval="1d")
            except Exception:
                dd = None
            if dd is not None and len(dd) > 500:
                for pat, s in daily_stats(dd).items():
                    per.setdefault(pat, {})[tk] = s
            if cls in SESSION_CLASSES:
                try:
                    hh = research_data.get_yf_ohlcv(tk, period="730d", interval="1h")
                except Exception:
                    hh = None
                if hh is not None and len(hh) > 2000:
                    for pat, s in session_stats(session_frame(hh)).items():
                        per.setdefault(pat, {})[tk] = s
                    prof.append(hourly_vol_profile(hh))
        pools[cls] = (pooled_by_day(per), (pd.concat(prof, axis=1).mean(axis=1) if prof else None), {p: len(v) for p, v in per.items()})
    rows = []
    for cls, (series, prof, n_assets) in pools.items():
        for pat, s in series.items():
            if cls == "fx" and pat.startswith("C"):
                continue   # a calendar drift in an FX pair depends on which currency is the quote — no sign convention
            r = test_series(s, rng)
            if r:
                rows.append({"class": cls, "pattern": pat, "assets": n_assets.get(pat, 0), **r})
    df = pd.DataFrame(rows)
    df["bh"] = bh(df["p"].to_numpy())
    df["repeats"] = df["bh"] & (np.sign(df["first_half"]) == np.sign(df["second_half"])) & (np.sign(df["mean"]) == np.sign(df["first_half"]))
    df.to_json(os.path.join(HERE, ".cache", "generic_pattern_screen.json"), orient="records", indent=1)
    profiles = {c: (v[1].round(4).to_dict() if v[1] is not None else None) for c, v in pools.items()}
    with open(os.path.join(HERE, ".cache", "hourly_vol_profile.json"), "w") as f:
        json.dump(profiles, f)
    return df, pools


if __name__ == "__main__":
    df, pools = main()
    pd.set_option("display.width", 220, "display.max_colwidth", 60)
    print(f"{len(df)} tests; BH-significant (q=0.10): {int(df.bh.sum())}; repeat in both halves: {int(df.repeats.sum())}\n")
    show = df.sort_values("p")[["class", "pattern", "assets", "n_days", "mean", "ci_lo", "ci_hi", "p", "first_half", "second_half", "bh", "repeats"]]
    print(show.round(3).head(25).to_string(index=False))
    for cls in ("fx", "crypto"):
        prof = pools[cls][1]
        if prof is not None:
            top = prof.sort_values(ascending=False).head(4)
            print(f"\n{cls}: busiest UTC hours (share of daily movement): " + ", ".join(f"{int(h):02d}:00 {100*v:.1f}%" for h, v in top.items()),
                  f"| quietest: " + ", ".join(f"{int(h):02d}:00 {100*v:.1f}%" for h, v in prof.sort_values().head(3).items()))
