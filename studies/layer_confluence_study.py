"""Does "the more layers confirm a move, the safer it is" hold on the strategies we actually run?

Takes the trades of the 78 validated combos and tags each with four independent, causal "confirming" layers a
top-down trader would stack, then asks whether mean R, loss rate and total R rise with the number of layers that
agree (a dose-response), rather than testing one filter at a time:

  NEWS   no scheduled release slot (London 07:00, NY 08:30 / 10:00) inside the entry bar's window
  HTF    the next timeframe up (5m->1h, 1h->1D) trends the trade's way (EMA 20 vs 50, completed bars only)
  CALM   volatility "quiet": ATR of the last completed bar in the lowest third of the previous 250 bars
  EMA    EMA20 vs EMA50 ordered in the trade's direction at the confirm bar

Judged with R demeaned per combo (so a better combo cannot fake a trend) and a permutation test that shuffles the
layer count WITHIN each combo. Reported on the three time slices.

    python3 studies/layer_confluence_study.py
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd

import data as research_data
import experiments as ex
import observatory as obs
import overlay_study as ot
from news_blackout_study import calendar_flags
from streak_polish_study import _utc

LAYERS = ["NEWS", "HTF", "CALM", "EMA"]


def per_bar_layers(df, tf):
    """Boolean arrays aligned to df's bars: news-clear, calm, ema-trend sign (+1/-1)."""
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    news_clear = ~calendar_flags(idx, tf)
    atr = pd.Series(ex.atr_series(df).to_numpy(), index=idx)
    thr = atr.rolling(250).quantile(1 / 3).shift(1)
    calm = (atr.shift(1) <= thr).to_numpy()
    close = df["Close"].to_numpy(dtype=float)
    f = pd.Series(close).ewm(span=20, adjust=False).mean().to_numpy()
    s = pd.Series(close).ewm(span=50, adjust=False).mean().to_numpy()
    # the EMA read at the bar BEFORE entry == the confirm bar
    ema_sign = np.sign(f - s)
    return pd.DataFrame({"NEWS": news_clear, "CALM": calm, "EMA_SIGN": ema_sign}, index=idx)


def tag(trades):
    parts = []
    for (ticker, tf), g in trades.groupby(["ticker", "tf"]):
        if tf not in ("5m", "1h"):      # the calendar rule is defined for intraday bars only (one daily combo exists)
            continue
        try:
            df = obs._fetch_df(ticker, tf)
        except Exception:
            continue
        bars = per_bar_layers(df, tf)
        e = g["entry_utc"].astype("datetime64[ns, UTC]")
        pos = bars.index.get_indexer(e)
        ok = pos >= 1
        g = g[ok].copy()
        pos = pos[ok]
        g["NEWS"] = bars["NEWS"].to_numpy()[pos]
        g["CALM"] = bars["CALM"].to_numpy()[pos - 1]          # ATR state as of the completed bar before entry
        sign = np.where(g["direction"] == "bullish", 1, -1)
        g["EMA"] = bars["EMA_SIGN"].to_numpy()[pos - 1] * sign > 0
        if tf in ot.HTF_OF:
            htf_tf, bar_td = ot.HTF_OF[tf]
            try:
                htf = research_data.get_yf_ohlcv(ticker, **ot.HTF_FETCH[htf_tf])
                m = ot.attach_trend(g, ot.htf_trend_frame(htf, 20, 50, bar_td))
                g = m.set_index(g.sort_values("entry_utc").index)
                g["HTF"] = g["aligned"]
            except Exception:
                g["HTF"] = np.nan
        else:
            g["HTF"] = np.nan
        parts.append(g)
    df = pd.concat(parts)
    df = df[df["HTF"].notna() & df["CALM"].notna()].copy()
    for c in LAYERS:
        df[c] = df[c].astype(bool)
    df["k"] = df[LAYERS].sum(axis=1)
    return df


def slope_p(df, rng, b=3000):
    """Slope of combo-demeaned R on layer count, p from shuffling the count within each combo."""
    d = df.assign(rc=df["r"] - df.groupby("combo")["r"].transform("mean"))
    groups = [(g["rc"].to_numpy(), g["k"].to_numpy().astype(float)) for _, g in d.groupby("combo") if g["k"].nunique() > 1]

    def slope(ks):
        x = np.concatenate([k - k.mean() for k in ks])
        y = np.concatenate([r for r, _ in groups])
        return float((x * y).sum() / (x * x).sum())

    obs_s = slope([k for _, k in groups])
    sims = np.array([slope([rng.permutation(k) for _, k in groups]) for _ in range(b)])
    return obs_s, float((1 + (np.abs(sims) >= abs(obs_s)).sum()) / (1 + b))


def main():
    rng = np.random.default_rng(13)
    trades = ot.load_trades()
    df = tag(trades)
    print(f"{len(df)} trades, {df['combo'].nunique()} combos; layer shares:",
          {c: round(float(df[c].mean()), 2) for c in LAYERS})
    out = {"by_k": {}, "slope": {}, "single_layer_diff": {}}
    for k, g in df.groupby("k"):
        out["by_k"][int(k)] = {"n": int(len(g)), "mean_R": round(float(g["r"].mean()), 3),
                               "loss_rate": round(float((g["r"] < 0).mean()), 3), "total_R": round(float(g["r"].sum()), 1)}
    for sl in ("research", "validation", "locked", "all"):
        s = df if sl == "all" else df[df["slice"] == sl]
        sl_, p = slope_p(s, rng)
        out["slope"][sl] = {"n": int(len(s)), "R_per_extra_layer": round(sl_, 3), "p": round(p, 4)}
    for c in LAYERS:
        out["single_layer_diff"][c] = {"agree": round(float(df[df[c]]["r"].mean()), 3),
                                       "disagree": round(float(df[~df[c]]["r"].mean()), 3)}
    top = df[df["k"] == df["k"].max()]
    out["all_layers_agree"] = {"n": int(len(top)), "mean_R": round(float(top["r"].mean()), 3),
                               "total_R": round(float(top["r"].sum()), 1), "all_trades_total_R": round(float(df["r"].sum()), 1)}
    print(json.dumps(out, indent=1))
    with open(os.path.join(HERE, ".cache", "layer_confluence.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
