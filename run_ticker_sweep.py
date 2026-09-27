"""One-off sweep driver: fills a real, measured coverage gap in the
_fixed_rr research base.

Confirmed directly by reading .cache/experiment_trials.jsonl (not
guessed): clean_expansion_fixed_rr has already been run across all 15
platform-tracked tickers on 1D/1h/5m (and some 4h/1d-legacy-cased
entries), using a 27-combo settings grid. clean_expansion_liquidity_fixed_rr
and clean_retracement_resumption_fixed_rr were only ever run against 8
tickers (BTC-USD, ES=F, ETH-USD, EURUSD=X, GBPUSD=X, NQ=F, RTY=F, YM=F) on
2 timeframes (1h, 1d) — never on AUDUSD=X, USDCAD=X, USDCHF=X, USDJPY=X,
GC=F, CL=F, SI=F, and never on 1D or 5m for ANY ticker. This script
closes that gap using the EXACT same 27-combo grid already established
for these strategies (reverse-engineered from the trial log's own label
strings, not invented), against the SAME 3 canonical research timeframes
research_stats.py already uses (1D/1h/5m — 4h isn't in the validated
research pipeline yet, it needs 1h->4h resampling that isn't wired into
this path).

Skips any (ticker, tf_label, exact settings) triple already present in
the trial log — safe to re-run, won't duplicate existing trials. Appends
new rows to the same .cache/experiment_trials.jsonl run_deep_backtest_rr
already writes to, then rebuilds ticker_behavior.db so the dashboard
picks up anything newly validated.

Run standalone: python3 run_ticker_sweep.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import experiments as ex
import ticker_behavior
import data as research_data

TICKERS = [
    "GBPUSD=X", "EURUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X",
    "ES=F", "YM=F", "NQ=F", "RTY=F", "GC=F", "SI=F", "CL=F",
    "BTC-USD", "ETH-USD",
]
TF_FETCH = {
    "1D": dict(period="10y", interval="1d"),
    "1h": dict(period="730d", interval="1h"),
    "5m": dict(period="60d", interval="5m"),
}

# Fixed (non-swept) params per strategy, matching what's already in the
# trial log exactly — min_streak=3 and retr_window_bars never varied
# across any of the 1728 existing fixed_rr trials for any of the three
# strategies; sweep_window_bars=10 likewise fixed for the liquidity variant.
STRATEGIES = {
    "clean_expansion_liquidity_fixed_rr": {
        "builder": ex.build_clean_expansion_liquidity_fixed_rr_events,
        "fixed": {"min_streak": 3, "retr_window_bars": 40, "sweep_window_bars": 10},
    },
    "clean_retracement_resumption_fixed_rr": {
        "builder": ex.build_clean_retracement_resumption_fixed_rr_events,
        "fixed": {"min_streak": 3, "retr_window_bars": 40},
    },
}

# The real 27-combo grid, reverse-engineered from the trial log's own
# settings (identical grid shape already used for these two strategies
# AND for clean_expansion_fixed_rr) — not a new invention.
_ATR_RR_MAXBARS = (
    [(atr, rr, 40) for atr in (0.75, 1.0, 1.5) for rr in (0.3, 0.5, 0.75, 1.0, 1.5)]
    + [(atr, rr, 60) for atr in (1.5, 2.0, 2.5) for rr in (1.5, 2.0, 2.5, 3.0)]
)
assert len(_ATR_RR_MAXBARS) == 27


def _label(strategy, atr, rr, win, maxbars):
    base = f"{strategy} · rr={rr} · atr={atr} · entry(win={win}b)"
    return base if maxbars == 40 else f"{base} · maxbars={maxbars}"


def _already_tried():
    """(strategy, ticker, normalized tf_label, label) triples already in
    the trial log — skip these outright rather than re-running them."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "experiment_trials.jsonl")
    seen = set()
    if not os.path.exists(path):
        return seen
    with open(path) as f:
        for line in f:
            try:
                t = json.loads(line)
            except Exception:
                continue
            label = t.get("label") or ""
            strat = next((s for s in STRATEGIES if label.startswith(s)), None)
            if strat is None:
                continue
            tf = t.get("tf_label") or ""
            tf = "1D" if tf.lower() == "1d" else tf
            seen.add((strat, t.get("ticker"), tf, label))
    return seen


def main():
    seen = _already_tried()
    print(f"{len(seen)} (strategy, ticker, tf, settings) combos already in the trial log — skipping those.")

    df_cache = {}
    n_run, n_skipped, n_scored, n_passed, n_fetch_failed = 0, 0, 0, 0, 0
    t0 = time.time()

    for strategy, spec in STRATEGIES.items():
        builder, fixed = spec["builder"], spec["fixed"]
        for ticker in TICKERS:
            for tf_label, fetch_kwargs in TF_FETCH.items():
                cache_key = (ticker, tf_label)
                if cache_key not in df_cache:
                    try:
                        df_cache[cache_key] = research_data.get_yf_ohlcv(ticker, **fetch_kwargs)
                    except Exception as e:
                        print(f"  fetch failed {ticker} {tf_label}: {e}")
                        df_cache[cache_key] = None
                        n_fetch_failed += 1
                df = df_cache[cache_key]
                if df is None or df.empty:
                    continue

                for atr, rr, maxbars in _ATR_RR_MAXBARS:
                    settings = {**fixed, "atr_mult_stop": atr, "rr_multiple": rr, "max_bars": maxbars}
                    label = _label(strategy, atr, rr, fixed["retr_window_bars"], maxbars)
                    if (strategy, ticker, tf_label, label) in seen:
                        n_skipped += 1
                        continue
                    try:
                        events = builder(df, **settings)
                    except Exception as e:
                        print(f"  builder failed {strategy} {ticker} {tf_label} {settings}: {e}")
                        continue
                    trial = ex.run_deep_backtest_rr(ticker, tf_label, events, settings, label)
                    n_run += 1
                    if trial.get("verdict") == "SCORED":
                        n_scored += 1
                        if trial.get("holdout_verdict") == "PASSED" and trial.get("same_sign"):
                            n_passed += 1
                            print(f"  PASSED  {ticker:10s} {tf_label:4s} {label}")

                elapsed = time.time() - t0
                print(f"[{elapsed:6.0f}s] {strategy:38s} {ticker:10s} {tf_label:4s} "
                      f"done (run={n_run} skipped={n_skipped} scored={n_scored} passed={n_passed})")

    print(f"\nDone in {time.time()-t0:.0f}s. New trials run: {n_run}, scored: {n_scored}, "
          f"passed (candidate-validated): {n_passed}. Fetch failures: {n_fetch_failed}.")
    print("Rebuilding ticker_behavior.db ...")
    ticker_behavior.rebuild_db()
    print("Done.")


if __name__ == "__main__":
    main()
