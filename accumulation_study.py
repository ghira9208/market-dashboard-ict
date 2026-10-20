"""Accumulation instead of trading: does ANY rule for how much to buy each week beat plain
dollar-cost averaging (DCA), or is "buy the same amount every week" already about as good as it gets?

Nothing is ever sold. Each week a fixed budget arrives (1.0 unit of money); a rule says how much of
the available cash to spend that week (a multiplier of the budget, 0.25x-3x). Cash not spent waits and
counts at face value at the end, so a rule that sits on cash pays for it. A 0.25% fee is charged on
every purchase. Every possible start week with a full horizon is simulated (3 and 5 years), so the
result is a distribution, not one lucky path, and the same rules are run on BTC, ETH, SOL, gold, the
S&P 500 and the Nasdaq-100 — the long equity/gold histories are the real test of whether a rule
generalises, because BTC's own 12 years are mostly one very strong uptrend.

Rules (fixed in advance, round numbers, no tuning):
  mayer      price vs its 200-day average: <0.8 -> 2x, <1.0 -> 1.5x, <1.5 -> 1x, <2.0 -> 0.5x, else 0.25x
  drawdown   fall from the running all-time high: <10% -> 0.75x, <30% -> 1x, <50% -> 1.5x, else 2.5x
  vol        buy less when volatility is high: median vol so far / 30-day vol, clipped to 0.25x-2x
  combo      geometric mean of mayer and drawdown
  value_avg  buy whatever closes the gap to a portfolio that grows by 1.0 a week (floor 0, cap 3x)

    python3 accumulation_study.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ASSETS = ["BTC-USD", "ETH-USD", "SOL-USD", "GC=F", "^GSPC", "^NDX"]
HORIZONS = {"3y": 156, "5y": 260}
FEE = 0.0025
RULES = ["dca", "mayer", "drawdown", "vol", "combo", "value_avg"]


def weekly_features(daily):
    """Weekly (last close of the week) price plus the causal indicators each rule needs. Everything at week t
    uses data up to and including the close of week t."""
    c = daily["Close"].astype(float)
    c.index = pd.DatetimeIndex(c.index).tz_localize(None) if pd.DatetimeIndex(c.index).tz is None else pd.DatetimeIndex(c.index).tz_convert(None)
    c = c[~c.index.duplicated(keep="last")].dropna()
    ma200 = c.rolling(200).mean()
    ath = c.cummax()
    ret = np.log(c / c.shift(1))
    vol30 = ret.rolling(30).std()
    vol_med = vol30.expanding(min_periods=120).median()
    f = pd.DataFrame({"price": c, "mayer": c / ma200, "dd": 1 - c / ath, "volratio": vol_med / vol30})
    w = f.resample("W-FRI").last().dropna()
    return w


def multipliers(w):
    m = pd.DataFrame(index=w.index)
    m["dca"] = 1.0
    m["mayer"] = np.select([w.mayer < 0.8, w.mayer < 1.0, w.mayer < 1.5, w.mayer < 2.0], [2.0, 1.5, 1.0, 0.5], 0.25)
    m["drawdown"] = np.select([w.dd < 0.10, w.dd < 0.30, w.dd < 0.50], [0.75, 1.0, 1.5], 2.5)
    m["vol"] = w.volratio.clip(0.25, 2.0)
    m["combo"] = np.sqrt(m["mayer"] * m["drawdown"])
    return m


def simulate(price, mult, horizon, rule):
    """All start weeks at once. Returns (final value / DCA final value - 1, share of budget deployed)."""
    n = len(price)
    starts = np.arange(0, n - horizon)
    cash = np.zeros(len(starts))
    units = np.zeros(len(starts))
    spent = np.zeros(len(starts))
    cash_dca = np.zeros(len(starts))
    units_dca = np.zeros(len(starts))
    for t in range(horizon):
        p = price[starts + t]
        cash += 1.0
        cash_dca += 1.0
        if rule == "value_avg":
            want = np.clip((t + 1.0) - units * p, 0.0, 3.0)
        else:
            want = mult[starts + t]
        spend = np.minimum(cash, want)
        units += spend * (1 - FEE) / p
        cash -= spend
        spent += spend
        units_dca += cash_dca * (1 - FEE) / p       # DCA spends everything that arrives
        cash_dca[:] = 0.0
    end = price[starts + horizon - 1]
    v_rule, v_dca = units * end + cash, units_dca * end
    return v_rule / v_dca - 1, spent / horizon


def main():
    import data as research_data
    rows = []
    for tk in ASSETS:
        daily = research_data.get_yf_ohlcv(tk, period="max", interval="1d")
        w = weekly_features(daily)
        m = multipliers(w)
        price = w.price.to_numpy()
        for hname, h in HORIZONS.items():
            if len(w) < h + 60:
                continue
            for rule in RULES[1:]:
                adv, deployed = simulate(price, m[rule].to_numpy() if rule != "value_avg" else None, h, rule)
                half = len(adv) // 2
                rows.append({"asset": tk, "horizon": hname, "rule": rule, "windows": len(adv),
                             "indep_windows": round((len(w) - h) / h, 1), "first": str(w.index[0].date()),
                             "mean_adv_pct": 100 * float(adv.mean()), "median_adv_pct": 100 * float(np.median(adv)),
                             "pct_windows_beating_dca": 100 * float((adv > 0).mean()),
                             "worst_5pct_adv": 100 * float(np.percentile(adv, 5)),
                             "first_half_mean": 100 * float(adv[:half].mean()), "second_half_mean": 100 * float(adv[half:].mean()),
                             "deployed_pct": 100 * float(deployed.mean())})
    df = pd.DataFrame(rows)
    df.to_json(os.path.join(HERE, ".cache", "accumulation_study.json"), orient="records", indent=1)
    return df


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    df = main()
    for h in HORIZONS:
        print(f"\n=== horizon {h}: final value vs plain DCA, % (positive = the rule accumulated more value for the same money) ===")
        x = df[df.horizon == h].pivot_table(index="asset", columns="rule", values="mean_adv_pct")
        print(x.round(1).to_string())
        y = df[df.horizon == h].pivot_table(index="asset", columns="rule", values="pct_windows_beating_dca")
        print("share of start dates where the rule beat DCA, %:"); print(y.round(0).to_string())
