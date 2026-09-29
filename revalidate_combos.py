"""Scheduled alpha-decay check — re-runs each CURRENTLY VALIDATED
combo's own best settings against FRESH data, so an edge that quietly
stopped working gets caught instead of trusted forever off one old
backtest. Reuses the exact same validation pipeline that originally
validated it (run_deep_backtest_rr, BH correction via rebuild_db) — this
isn't a new statistical method, it's the same one, re-triggered
periodically.

IMPORTANT, non-obvious correctness point: load_experiment_trials() BH-
corrects across the WHOLE trial log, forever — a trial from weeks ago
never expires or gets superseded just because a fresher one exists for
the same (ticker, tf_label, strategy) group. That means "did this combo
drop out of validated_leaderboard()" is NOT a reliable decay signal: an
old, no-longer-representative trial can keep counting as a survivor
indefinitely, masking real decay in the fresh data. The reliable signal
is the direct result of THIS run's own re-validation trial (its own
holdout_verdict/same_sign), checked immediately, not the pooled
leaderboard after the fact.

Run: python3 revalidate_combos.py (meant for launchd — see
com.edgepipeline.revalidate.plist — weekly, not continuous).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/Users/apple/dev/etoro-dashboard")

import experiments as ex
import ticker_behavior
import data as research_data
from notify import notify

_DECAY_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "decay_flags.json")

BUILDER = {
    "clean_expansion_fixed_rr": ex.build_clean_expansion_fixed_rr_events,
    "clean_expansion_liquidity_fixed_rr": ex.build_clean_expansion_liquidity_fixed_rr_events,
    "clean_retracement_resumption_fixed_rr": ex.build_clean_retracement_resumption_fixed_rr_events,
}
TF_FETCH = {
    "1D": dict(period="10y", interval="1d"),
    "1h": dict(period="730d", interval="1h"),
    "5m": dict(period="60d", interval="5m"),
}


def _decay_verdict(trial):
    """Pure classification of one fresh re-validation trial's own result
    — deliberately NOT "did this combo drop out of validated_leaderboard()"
    (see this module's own docstring for why that pooled signal is
    unreliable). Returns "decayed" (scored, and failed on ITS OWN fresh
    holdout), "ok" (scored and passed), or "inconclusive" (not enough
    fresh events to judge either way — never counted as decay)."""
    if trial.get("verdict") != "SCORED":
        return "inconclusive"
    passed = trial.get("holdout_verdict") == "PASSED" and trial.get("same_sign")
    return "ok" if passed else "decayed"


def main():
    t0 = time.time()
    leaderboard = ticker_behavior.validated_leaderboard()
    fixed_rr = leaderboard[leaderboard["strategy"].str.endswith("_fixed_rr")]
    trials = ex.load_experiment_trials()

    decayed, n_run, n_skipped = [], 0, 0
    for _, row in fixed_rr.iterrows():
        ticker, tf_label, strategy, best_label = row["ticker"], row["tf_label"], row["strategy"], row["best_label"]
        builder = BUILDER.get(strategy)
        if builder is None or tf_label not in TF_FETCH:
            n_skipped += 1
            continue
        match = trials[(trials.ticker == ticker) & (trials.tf_label == tf_label) & (trials.label == best_label)]
        if match.empty:
            n_skipped += 1
            continue
        settings = match.iloc[0]["settings"]

        try:
            df = research_data.get_yf_ohlcv(ticker, **TF_FETCH[tf_label])
            if df.empty:
                n_skipped += 1
                continue
            events = builder(df, **settings)
            if events.empty:
                n_skipped += 1
                continue
            trial = ex.run_deep_backtest_rr(ticker, tf_label, events, settings, best_label)
        except Exception as e:
            print(f"  re-validation errored for {ticker} {tf_label} {strategy}: {e}")
            n_skipped += 1
            continue

        n_run += 1
        verdict = _decay_verdict(trial)
        if verdict == "inconclusive":
            continue
        if verdict == "decayed":
            decayed.append({
                "ticker": ticker, "tf_label": tf_label, "strategy": strategy,
                "mean_return_holdout": trial.get("mean_return_holdout"),
                "was_mean_return_holdout": row.get("best_mean_return_holdout"),
            })
            print(f"  DECAYED: {ticker} {tf_label} {strategy} "
                  f"(fresh mean_R={trial.get('mean_return_holdout')}, was {row.get('best_mean_return_holdout')})")

    print(f"Re-ran {n_run} combos ({n_skipped} skipped — no settings/data/builder), "
          f"{len(decayed)} no longer pass on fresh data, in {time.time()-t0:.0f}s.")

    # Written regardless of whether anything decayed — an EMPTY list with
    # a fresh checked_at is itself real information (dashboard_data.py's
    # own is_stale-style staleness check needs a timestamp to compare
    # against either way). This is what actually closes the loop: without
    # it, a fresh Telegram ping is the ONLY place this finding exists —
    # the dashboard itself would keep showing these cards as indistinguishable
    # from genuinely healthy ones.
    os.makedirs(os.path.dirname(_DECAY_CACHE_PATH), exist_ok=True)
    with open(_DECAY_CACHE_PATH, "w") as f:
        json.dump({"checked_at": time.time(), "decayed": decayed}, f, indent=2)

    print("Rebuilding ticker_behavior.db so the pooled view reflects today's trials too...")
    ticker_behavior.rebuild_db()

    if decayed:
        lines = [f"{d['ticker']} {d['tf_label']} {d['strategy'].replace('_fixed_rr', '')}" for d in decayed]
        notify("Decay check",
               f"{len(decayed)} combo(s) no longer pass their own validation on fresh "
               f"data — {', '.join(lines[:5])}{', ...' if len(lines) > 5 else ''}. Worth a look before "
               f"trusting them the same way going forward.", severity="warning")
    else:
        print("No decay detected — every currently-validated combo still holds up on fresh data.")


if __name__ == "__main__":
    main()
