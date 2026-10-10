"""Reads the time-series-momentum ledger: does the diversified book earn
anything, does it hold on data nobody selected on, and does it survive
realistic costs? Per-asset results are shown too, but the portfolio rows are
the headline (see tsmom_backtest.py for why).

    python3 tsmom_report.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import experiments as ex  # noqa: E402
import data as research_data  # noqa: E402
import tsmom_backtest as tm  # noqa: E402

STRESS_BPS = 8.0


def portfolio_slice_stats(cost_bps):
    from run_ema_sweep import universe
    out = {}
    for lb in tm.LOOKBACKS:
        frames = {}
        for t in universe():
            try:
                df = research_data.get_yf_ohlcv(t, period="10y", interval="1d")
            except Exception:
                continue
            if df is None or df.empty or len(df) < 500:
                continue
            w = tm.weekly_arrays(df["Close"], lb, cost_bps=cost_bps)
            if len(w) >= tm.MIN_WEEKS:
                frames[t] = w["r"]
        book = pd.DataFrame(frames).sort_index().mean(axis=1, skipna=True)
        b = tm.slice_bounds(len(book))
        out[lb] = {name: (float(book.iloc[lo:hi].mean()),
                          float(book.iloc[lo:hi].mean() / book.iloc[lo:hi].std(ddof=1) * np.sqrt(52)))
                   for name, (lo, hi) in b.items()}
        out[lb]["n_assets"] = len(frames)
    return out


def main():
    ex._TRIALS_PATH = tm.LEDGER
    t = ex.load_experiment_trials()
    t = t[t["verdict"] == "SCORED"].copy()
    t["lookback"] = t["settings"].map(lambda s: s["lookback_days"])
    port, assets = t[t["ticker"] == "PORTFOLIO"], t[t["ticker"] != "PORTFOLIO"]
    pd.set_option("display.width", 200)
    print(f"{len(t)} trials ({len(assets)} asset-lookbacks + {len(port)} portfolio) | BH-significant: "
          f"{int(t['bh_significant'].fillna(False).sum())} | survived: {int(t['survived'].sum())} | "
          f"asset holdout pass rate {(assets['holdout_verdict'] == 'PASSED').mean() * 100:.1f}% (luck ~5%)\n")

    print("PORTFOLIO (equal-risk across all assets, weekly, one shared shift for the null)")
    cols = ["lookback", "n_assets", "mean_return_train", "p_value_train", "mean_return_holdout", "p_value_holdout",
            "locked_mean_return", "locked_p_value", "same_sign", "bh_significant", "survived"]
    print(port.sort_values("lookback")[cols].round(4).to_string(index=False))

    print("\nBREADTH across assets (does it work in many markets or a few?)")
    rows = []
    for lb, g in assets.groupby("lookback"):
        rows.append({"lookback": lb, "assets": len(g),
                     "pct_val_positive": round((g["mean_return_holdout"] > 0).mean() * 100, 1),
                     "pct_locked_positive": round((g["locked_mean_return"] > 0).mean() * 100, 1),
                     "median_val_R": round(g["mean_return_holdout"].median(), 4),
                     "median_locked_R": round(g["locked_mean_return"].median(), 4),
                     "bh_sig": int(g["bh_significant"].fillna(False).sum()), "survivors": int(g["survived"].sum())})
    print(pd.DataFrame(rows).to_string(index=False))

    s = assets[assets["survived"]].sort_values("mean_return_holdout", ascending=False)
    if len(s):
        print(f"\nASSET SURVIVORS ({len(s)})")
        print(s[["ticker", "lookback", "mean_return_train", "mean_return_holdout", "locked_mean_return"]].head(20).round(4).to_string(index=False))

    print("\nWEEKLY EQUITY-WEIGHTED BOOK, mean R per week and annualised information ratio (mean/std*sqrt(52))")
    print("slice means at 2 bps and at the harsher cost:")
    base, hard = portfolio_slice_stats(tm.COST_BPS), portfolio_slice_stats(STRESS_BPS)
    for lb in tm.LOOKBACKS:
        f = lambda d: " | ".join(f"{k[:3]} {d[k][0]:+.3f} (IR {d[k][1]:+.2f})" for k in ("research", "validation", "locked"))
        print(f"  {lb:3d}d  @2bps:  {f(base[lb])}")
        print(f"        @{STRESS_BPS:g}bps: {f(hard[lb])}   [{base[lb]['n_assets']} assets]")


if __name__ == "__main__":
    main()
