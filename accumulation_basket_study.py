"""Two accumulation questions that are NOT market timing (the first study found no timing rule that beats
plain dollar-cost averaging):

  1. Lump sum vs DCA: if the money exists today, is spreading it over 1-3 years better than investing it
     now? (Same total money; DCA's undeployed cash earns 0.)
  2. One asset vs a basket, and fixed split vs "buy whatever is below its target weight": same weekly
     budget, judged on final value, the worst fall in the portfolio's value on the way, and the bad end
     of the distribution — because the case for a basket is risk, not return.

Baskets (fixed in advance): BTC-only; 50% BTC + 50% gold; 40% BTC + 20% ETH + 20% gold + 20% S&P 500.
Every possible start week with a full horizon, on the weeks all of an asset set has data for. Nothing is
sold; a 0.25% fee is charged on every purchase.

    python3 accumulation_basket_study.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from accumulation_study import FEE, weekly_features  # noqa: E402

BASKETS = {
    "BTC only": {"BTC-USD": 1.0},
    "50 BTC / 50 gold": {"BTC-USD": 0.5, "GC=F": 0.5},
    "40 BTC / 20 ETH / 20 gold / 20 S&P": {"BTC-USD": 0.4, "ETH-USD": 0.2, "GC=F": 0.2, "^GSPC": 0.2},
}
H = {"3y": 156, "5y": 260}


def prices_for(tickers):
    import data as research_data
    cols = {}
    for tk in tickers:
        cols[tk] = weekly_features(research_data.get_yf_ohlcv(tk, period="max", interval="1d")).price
    return pd.DataFrame(cols).dropna()


def run_basket(P, weights, horizon, mode):
    """P: weekly prices (rows x assets). mode 'split' = budget divided by target weights every week;
    'underweight' = the week's budget goes to the assets furthest below target weight. Returns per-start
    arrays: final value per unit of money contributed, and the worst peak-to-trough fall of portfolio value."""
    tickers = list(weights)
    w = np.array([weights[t] for t in tickers])
    X = P[tickers].to_numpy()
    n = len(X)
    starts = np.arange(0, n - horizon)
    units = np.zeros((len(starts), len(tickers)))
    peak = np.zeros(len(starts))
    worst = np.zeros(len(starts))
    for t in range(horizon):
        p = X[starts + t]
        if mode == "split" or len(tickers) == 1:
            alloc = np.tile(w, (len(starts), 1))
        else:
            value = units * p
            total = value.sum(axis=1, keepdims=True) + 1.0                   # including this week's budget
            gap = np.clip(w * total - value, 0.0, None)
            s = gap.sum(axis=1, keepdims=True)
            alloc = np.where(s > 0, gap / np.where(s > 0, s, 1.0), w)
        units += alloc * (1 - FEE) / p
        val = (units * p).sum(axis=1) / (t + 1.0)                            # value per unit of money put in so far
        peak = np.maximum(peak, val)
        worst = np.maximum(worst, 1 - val / peak)
    final = (units * X[starts + horizon - 1]).sum(axis=1) / horizon
    return final, worst


def lump_vs_dca(price, horizon):
    n = len(price)
    starts = np.arange(0, n - horizon)
    end = price[starts + horizon - 1]
    lump = (1 - FEE) * end / price[starts]                                   # all money in at week 0, value per unit invested
    units = np.zeros(len(starts))
    for t in range(horizon):
        units += (1 - FEE) / price[starts + t]
    dca = units * end / horizon
    return lump / dca - 1


def main():
    import data as research_data
    out = {}
    print("=== 1. lump sum now vs DCA over the horizon (same total money; positive = lump sum ended with more) ===")
    for tk in ["BTC-USD", "ETH-USD", "GC=F", "^GSPC", "^NDX"]:
        price = weekly_features(research_data.get_yf_ohlcv(tk, period="max", interval="1d")).price.to_numpy()
        line = []
        for hn, h in H.items():
            a = lump_vs_dca(price, h)
            line.append(f"{hn}: mean {100*a.mean():+5.1f}%, lump sum won {100*(a>0).mean():3.0f}% of start dates")
        print(f"  {tk:8s} " + " | ".join(line))
    print("\n=== 2. baskets (common history), 3-year programs ===")
    allp = prices_for(["BTC-USD", "ETH-USD", "GC=F", "^GSPC"])
    print(f"  common history {allp.index[0].date()} -> {allp.index[-1].date()} ({len(allp)} weeks)\n")
    for hn, h in H.items():
        rows = []
        for name, wts in BASKETS.items():
            for mode in (("split",) if len(wts) == 1 else ("split", "underweight")):
                final, worst = run_basket(allp, wts, h, mode)
                rows.append({"basket": name, "how": mode, "final value per $1 put in (median)": np.median(final),
                             "5th pct (bad luck)": np.percentile(final, 5), "worst fall of value in the program (median)": np.median(worst) * 100,
                             "worst fall, 95th pct": np.percentile(worst, 95) * 100, "windows": len(final)})
        print(f"--- {hn} programs ---")
        print(pd.DataFrame(rows).round(2).to_string(index=False))
        print()


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    main()
