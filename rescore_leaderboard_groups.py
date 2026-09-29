"""One-off: force a fresh, cost-aware trial for EVERY settings combo in
the full 27-combo grid, but only for (ticker, tf_label, strategy)
groups that are CURRENTLY on the validated leaderboard — not the full
15-ticker x 3-tf x 3-strategy universe run_ticker_sweep.py covers.

Why this is needed, not just revalidate_combos.py's narrower recheck:
that script only re-tests each group's own CURRENT BEST settings. The
other 26 settings in the same group's grid are untouched, still
frictionless (from before cost_bps existed in _simulate_fixed_rr) —
ticker_behavior.rebuild_db()'s own best_row = idxmax(mean_return_train)
then compares one freshly cost-adjusted row against 26 optimistic,
never-adjusted ones and can pick a DIFFERENT settings combo purely
because it was never honestly re-priced, not because it's actually
better. Confirmed live: ETH-USD 5m's leaderboard entry changed from
one label to another between the dedup fix and this script, for
exactly that reason.

Scope: only groups (ticker, tf_label, strategy) already on
validated_leaderboard() right now — the ones that actually matter for
what's tradeable, not a full-universe re-sweep (which would be ~3600+
backtests here vs ~1600).

Forces a fresh trial for every settings row regardless of whether that
exact label already exists in the trial log (unlike run_ticker_sweep.py's
own _already_tried skip) — the whole point is to REPLACE stale
frictionless numbers, not skip past them. Appends to the same trial
log; rebuild_db()'s own new dedup (keep the newest trial per exact
label) means these fresh rows simply supersede the old ones.

Run standalone: python3 rescore_leaderboard_groups.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import experiments as ex
import ticker_behavior
import data as research_data

TF_FETCH = {
    "1D": dict(period="10y", interval="1d"),
    "1h": dict(period="730d", interval="1h"),
    "5m": dict(period="60d", interval="5m"),
}

BUILDER = {
    "clean_expansion_fixed_rr": ex.build_clean_expansion_fixed_rr_events,
    "clean_expansion_liquidity_fixed_rr": ex.build_clean_expansion_liquidity_fixed_rr_events,
    "clean_retracement_resumption_fixed_rr": ex.build_clean_retracement_resumption_fixed_rr_events,
}
FIXED = {
    "clean_expansion_fixed_rr": {"min_streak": 3, "retr_window_bars": 60},
    "clean_expansion_liquidity_fixed_rr": {"min_streak": 3, "retr_window_bars": 40, "sweep_window_bars": 10},
    "clean_retracement_resumption_fixed_rr": {"min_streak": 3, "retr_window_bars": 40},
}

_ATR_RR_MAXBARS = (
    [(atr, rr, 40) for atr in (0.75, 1.0, 1.5) for rr in (0.3, 0.5, 0.75, 1.0, 1.5)]
    + [(atr, rr, 60) for atr in (1.5, 2.0, 2.5) for rr in (1.5, 2.0, 2.5, 3.0)]
)
assert len(_ATR_RR_MAXBARS) == 27


def _label(strategy, atr, rr, win, maxbars):
    base = f"{strategy} · rr={rr} · atr={atr} · entry(win={win}b)"
    return base if maxbars == 40 else f"{base} · maxbars={maxbars}"


def main():
    lb = ticker_behavior.validated_leaderboard()
    fixed_rr = lb[lb["strategy"].str.endswith("_fixed_rr")]
    groups = sorted(set(zip(fixed_rr["ticker"], fixed_rr["tf_label"], fixed_rr["strategy"])))
    print(f"{len(groups)} (ticker, tf_label, strategy) groups on the current leaderboard — "
          f"re-scoring all {len(_ATR_RR_MAXBARS)} grid settings for each, cost-aware.")

    df_cache = {}
    t0 = time.time()
    n_run, n_fetch_failed, n_builder_failed = 0, 0, 0
    for i, (ticker, tf_label, strategy) in enumerate(groups):
        builder = BUILDER.get(strategy)
        fixed = FIXED.get(strategy)
        if builder is None or tf_label not in TF_FETCH:
            continue
        cache_key = (ticker, tf_label)
        if cache_key not in df_cache:
            try:
                df_cache[cache_key] = research_data.get_yf_ohlcv(ticker, **TF_FETCH[tf_label])
            except Exception as e:
                print(f"  fetch failed {ticker} {tf_label}: {e}")
                df_cache[cache_key] = None
                n_fetch_failed += 1
        df = df_cache[cache_key]
        if df is None or df.empty:
            continue

        for atr, rr, maxbars in _ATR_RR_MAXBARS:
            win = fixed.get("retr_window_bars", 40)
            settings = {**fixed, "atr_mult_stop": atr, "rr_multiple": rr, "max_bars": maxbars}
            label = _label(strategy, atr, rr, win, maxbars)
            try:
                events = builder(df, **settings)
            except Exception as e:
                print(f"  builder failed {strategy} {ticker} {tf_label} {settings}: {e}")
                n_builder_failed += 1
                continue
            ex.run_deep_backtest_rr(ticker, tf_label, events, settings, label)
            n_run += 1

        elapsed = time.time() - t0
        print(f"[{elapsed:6.0f}s] ({i+1}/{len(groups)}) {ticker:10s} {tf_label:4s} {strategy:38s} done "
              f"(total trials run={n_run})")

    print(f"\nDone in {time.time()-t0:.0f}s. Trials run: {n_run}. "
          f"Fetch failures: {n_fetch_failed}. Builder failures: {n_builder_failed}.")
    print("Rebuilding ticker_behavior.db ...")
    ticker_behavior.rebuild_db()
    print("Done.")


if __name__ == "__main__":
    main()
