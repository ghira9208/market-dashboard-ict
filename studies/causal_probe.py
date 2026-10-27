"""Is the backtest detector causal? For each full-history touch, re-run the detector on ONLY the bars that existed at
its confirm bar (what live sees) and ask: does the same touch still appear? Result 2026-10-21: expansion 100%,
retracement-resumption 90%, expansion-liquidity 66%; the trades live would NOT see average lower R, so the backtest
is not inflated by the detectors reading ahead.

    python3 studies/causal_probe.py
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import pandas as pd

import experiments as ex
import observatory as obs
import ticker_behavior as tb
from overlay_tools import FIND

WANT = {"clean_retracement_resumption_fixed_rr": 4, "clean_expansion_fixed_rr": 4, "clean_expansion_liquidity_fixed_rr": 2}
MAX_TRADES = 40


def main():
    lb, trials = tb.validated_leaderboard(), ex.load_experiment_trials()
    picked = []
    for strat, k in WANT.items():
        sub = lb[lb.strategy == strat]
        sub = pd.concat([sub[sub.tf_label == "1h"].head(k // 2 + k % 2), sub[sub.tf_label == "5m"].head(k // 2)])
        picked += [c for _, c in sub.iterrows()]
    rows = []
    for c in picked:
        m = trials[(trials.ticker == c.ticker) & (trials.tf_label == c.tf_label) & (trials.label == c.best_label)]
        s = obs.effective_settings({k: v for k, v in m.iloc[0]["settings"].items() if not k.startswith("_")},
                                   c.get("best_target_widen_mult"))
        df = obs._fetch_df(c.ticker, c.tf_label)
        finder = FIND[c.strategy]
        kw = dict(min_streak=s["min_streak"], retr_window_bars=s["retr_window_bars"])
        full = finder(df, **kw)
        ev = ex._simulate_fixed_rr(df, full, s["rr_multiple"], s["atr_mult_stop"], s["max_bars"], ex.DEFAULT_COST_BPS)
        traded = set(ev["entry_time"])
        rmap = dict(zip(ev["entry_time"], ev["r_real"]))
        sel = [t for t in full if t["confirm_pos"] + 1 < len(df) and df.index[t["confirm_pos"] + 1] in traded][-MAX_TRADES:]
        t0 = time.time()
        for t in sel:
            cpos = t["confirm_pos"]
            asof = finder(df.iloc[:cpos + 1], **kw)
            rows.append({"strategy": c.strategy, "ticker": c.ticker, "tf": c.tf_label,
                         "hit": any(a["confirm_pos"] == cpos and a["direction"] == t["direction"] for a in asof),
                         "r": rmap[df.index[cpos + 1]]})
        print(c.strategy[:28], c.ticker, c.tf_label, "touches", len(full), "checked", len(sel), f"{time.time() - t0:.0f}s", flush=True)
    d = pd.DataFrame(rows)
    print("\nshare of backtest trades live would ALSO have seen:", round(d.hit.mean(), 3), "n =", len(d))
    print(d.groupby("strategy").agg(n=("hit", "size"), seen=("hit", "mean"), mean_R_all=("r", "mean")))
    print("mean R of trades live WOULD see   :", round(d[d.hit].r.mean(), 3), "n =", int(d.hit.sum()))
    print("mean R of trades live would NOT see:", round(d[~d.hit].r.mean(), 3), "n =", int((~d.hit).sum()))
    d.to_csv(os.path.join(HERE, ".cache", "causal_probe.csv"), index=False)


if __name__ == "__main__":
    main()
