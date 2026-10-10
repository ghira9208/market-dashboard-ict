"""Reads the EMA sweep's own ledger and reports whether any family earned
anything beyond chance. Three questions, in order of how hard they are to fake:

  1. BREADTH — for each family/setting, what share of tickers show a positive
     out-of-sample (VALIDATION) mean, and how often does the holdout test pass
     versus the ~5% a no-edge strategy passes by luck? A real edge shows up
     across many instruments; a lucky one shows up in a few.
  2. SURVIVORS — trials that cleared Benjamini-Hochberg on the RESEARCH slice,
     passed the VALIDATION holdout, and agreed on sign — then what the LOCKED
     slice (never used for any decision) says about them.
  3. COST STRESS — the locked-positive survivors re-simulated at a harsher
     round-trip cost, because the flat 2 bps used by every backtest here
     understates real futures/CFD costs (see validate_cost_model.py).

    python3 ema_sweep_report.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import argparse  # noqa: E402
import importlib  # noqa: E402

import experiments as ex  # noqa: E402
import data as research_data  # noqa: E402
from run_ema_sweep import TF_FETCH  # noqa: E402

# Generic over any sweep that follows the same contract (a strategies module
# with BUILDERS + DETECTOR_VERSIONS, and its own ledger):
#     python3 ema_sweep_report.py                                  # EMA families
#     python3 ema_sweep_report.py pattern_strategies pattern_sweep_trials.jsonl
_ap = argparse.ArgumentParser()
_ap.add_argument("module", nargs="?", default="ema_strategies")
_ap.add_argument("ledger", nargs="?", default="ema_sweep_trials.jsonl")
_args = _ap.parse_args()
es = importlib.import_module(_args.module)
SWEEP_LEDGER = os.path.join(HERE, ".cache", _args.ledger)
TF_FETCH = {**TF_FETCH, "5m": dict(period="60d", interval="5m")}

ex._TRIALS_PATH = SWEEP_LEDGER
ex.DETECTOR_LOGIC_VERSIONS.update(es.DETECTOR_VERSIONS)
STRESS_BPS = 6.0
MIN_LOCKED_N = 10


def _setting_key(label):
    """Label minus nothing — the label IS the setting (family + params + rr)."""
    return label.split(" · ", 1)[0].replace("_fixed_rr", ""), label.split(" · ", 1)[1] if " · " in label else ""


def load():
    t = ex.load_experiment_trials()
    t = t[t["verdict"] == "SCORED"].copy()
    t["family"] = t["label"].map(lambda l: _setting_key(l)[0])
    t["setting"] = t["label"].map(lambda l: _setting_key(l)[1])
    return t


def breadth(t):
    rows = []
    for (fam, setting), g in t.groupby(["family", "setting"]):
        ho = g["mean_return_holdout"].dropna()
        rows.append({
            "family": fam, "setting": setting, "n": len(g),
            "pct_val_positive": float((ho > 0).mean() * 100) if len(ho) else np.nan,
            "median_val_R": float(ho.median()) if len(ho) else np.nan,
            "median_train_R": float(g["mean_return_train"].median()),
            "holdout_pass_pct": float((g["holdout_verdict"] == "PASSED").mean() * 100),
            "bh_sig": int(g["bh_significant"].fillna(False).sum()),
            "survivors": int(g["survived"].sum()),
        })
    return pd.DataFrame(rows).sort_values(["family", "median_val_R"], ascending=[True, False])


def survivors(t):
    s = t[t["survived"]].copy()
    s["locked_ok"] = (s["locked_mean_return"] > 0) & (s["locked_n"] >= MIN_LOCKED_N)
    return s.sort_values("mean_return_holdout", ascending=False)


def stress(s):
    """Re-simulate each locked-positive survivor at STRESS_BPS and score train/val/locked
    means with the same 60/20/20 time split (events only — no new trial is logged)."""
    out = []
    for _, r in s[s["locked_ok"]].iterrows():
        fam = r["family"]
        settings = {k: (tuple(v) if isinstance(v, list) else v) for k, v in r["settings"].items()}
        try:
            df = research_data.get_yf_ohlcv(r["ticker"], **TF_FETCH[r["tf_label"]])
            ev = es.BUILDERS[f"{fam}_fixed_rr"](df, cost_bps=STRESS_BPS, **settings).sort_values("entry_time")
        except Exception:
            continue
        n = len(ev)
        a, b = int(n * 0.6), int(n * 0.8)
        means = [ev["r_real"].iloc[x:y].mean() for x, y in ((0, a), (a, b), (b, n))]
        out.append({"ticker": r["ticker"], "tf": r["tf_label"], "setting": r["setting"], "family": fam,
                    "R@2bps train/val/locked": f"{r['mean_return_train']:+.2f}/{r['mean_return_holdout']:+.2f}/{r['locked_mean_return']:+.2f}",
                    f"R@{STRESS_BPS:g}bps train/val/locked": f"{means[0]:+.2f}/{means[1]:+.2f}/{means[2]:+.2f}",
                    "survives_stress": bool(means[0] > 0 and means[1] > 0 and means[2] > 0), "n": n})
    return pd.DataFrame(out)


def main():
    t = load()
    all_rows = pd.read_json(SWEEP_LEDGER, lines=True)
    print(f"ledger: {len(all_rows)} trials | scored {len(t)} | insufficient data {int((all_rows['verdict'] != 'SCORED').sum())}")
    print(f"BH-significant on RESEARCH slice: {int(t['bh_significant'].fillna(False).sum())} | "
          f"survived (BH + holdout + same sign): {int(t['survived'].sum())}")
    null_pass = 5.0
    print(f"holdout passed: {(t['holdout_verdict'] == 'PASSED').mean() * 100:.1f}% of scored trials "
          f"(a strategy with NO edge passes ~{null_pass:.0f}% by luck)\n")

    pd.set_option("display.width", 220, "display.max_colwidth", 60, "display.max_rows", 200)
    b = breadth(t)
    print("BREADTH by family/setting (pooled across tickers and timeframes)")
    print(b.round(2).to_string(index=False))

    s = survivors(t)
    print(f"\nSURVIVORS: {len(s)} | of those with a positive LOCKED slice (n>={MIN_LOCKED_N}): {int(s['locked_ok'].sum())}")
    if len(s):
        show = s[["family", "ticker", "tf_label", "setting", "n_train", "mean_return_train", "mean_return_holdout",
                  "locked_n", "locked_mean_return", "locked_ok"]].copy()
        print(show.head(40).round(3).to_string(index=False))
        print("\nSurvivors per family:", s.groupby("family").size().to_dict())
        print("Locked-positive per family:", s[s["locked_ok"]].groupby("family").size().to_dict())

    st = stress(s) if len(s) else pd.DataFrame()
    if len(st):
        print(f"\nCOST STRESS ({STRESS_BPS:g} bps round trip) on locked-positive survivors")
        print(st.to_string(index=False))
        print(f"Survive stress on all three slices: {int(st['survives_stress'].sum())} of {len(st)}")

    out = {"trials": int(len(all_rows)), "scored": int(len(t)), "survivors": int(len(s)),
           "locked_ok": int(s["locked_ok"].sum()) if len(s) else 0,
           "stress_survivors": int(st["survives_stress"].sum()) if len(st) else 0}
    with open(os.path.join(HERE, ".cache", _args.ledger.replace(".jsonl", "_summary.json")), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
