"""Does live's blanket TARGET_WIDEN_MULT=3.0 actually hold up as its OWN
validated choice, per combo — or was it only ever a pooled, cross-instrument
finding being applied as if it were one? (bot_engine.py's own TARGET_WIDEN_MULT
docstring already says so directly: "a POOLED cross-instrument result, not a
fresh per-instrument backtest.") This is that missing per-instrument backtest.

For every currently-validated _fixed_rr combo, re-simulates its exact same
entries at WIDEN_FACTORS = [1.0, 1.5, 2.0, 3.0, 4.0] — same touches, same
stop, same max_bars, same cost model, only the target distance changes —
through the EXACT SAME statistical pipeline every other trial goes through
(run_deep_backtest_rr: 80/20 time split, 5000-draw permutation test,
BH-corrected across the SAME trial log everything else lives in). Nothing
pooled, nothing cherry-picked after the fact: every variant, pass or fail,
gets appended to the log exactly like every other trial ever has been.

settings passed to run_deep_backtest_rr stays the ORIGINAL (un-widened)
dict — only target_widen_mult varies — matching exactly how bot_engine.py's
own _current_strategy_id computes a combo's live identity (settings
unwidened, TARGET_WIDEN_MULT applied separately on top). This is what lets
compute_strategy_id tell a widen=3.0 trial apart from a widen=1.0 trial of
the identical underlying settings, and what lets bot_engine.py's own
per-combo widen lookup (_live_target_widen_mult) find "the best validated
widen factor for this exact combo" by its own real identity, not a guess.

Run: python3 validate_target_variants.py — NOT scheduled (unlike
revalidate_combos.py's weekly cron); re-run by hand whenever the live
leaderboard changes meaningfully enough to be worth re-checking. Takes
roughly WIDEN_FACTORS-times as long as a single revalidate_combos.py pass
per combo it covers (one real data fetch + simulation per variant).
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/Users/apple/dev/etoro-dashboard")

import numpy as np

import experiments as ex
import ticker_behavior
import data as research_data

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
WIDEN_FACTORS = [1.0, 1.5, 2.0, 3.0, 4.0]

# Strips a trailing " · widen=<number>" this script's own labels add — real
# bug caught live 2026-10-07: without this, re-running this script against
# an ALREADY-widened leaderboard (its own prior run's winning variant is
# now best_label) appended widen=X a SECOND time ("...widen=2.0 · widen=2.0"),
# producing a brand-new label that never matches, let alone supersedes, the
# original — the stale, field-incomplete first-run trial stayed the
# leaderboard's own best_row forever, re-run after re-run. Stripping back
# to the true base label first makes every re-run idempotent: the SAME
# widen factor on the SAME base settings always produces the SAME label
# (and therefore the SAME strategy_id), so dedup's own "newest wins" rule
# correctly supersedes a stale trial with a fresh, complete one instead of
# silently accumulating siblings next to it forever.
_WIDEN_SUFFIX_RE = re.compile(r"\s*·\s*widen=[\d.]+\s*$")


def _base_label(label):
    """REAL REGRESSION caught live 2026-10-08, re-running this script for
    the first time since switching to run_deep_backtest_rr_with_lock: 28
    of 106 validated combos came back with DOUBLE-stacked suffixes
    ("...maxbars=60 · widen=2.0 · widen=4.0") — some already carried a
    historical double-suffixed best_label (itself winning best_row
    selection on mean_return_train, since widening this account's own
    real combos usually inflates the train-side mean), and _sub()'s own
    single pass only ever strips ONE match for an end-anchored pattern —
    stripping the outer suffix left the inner one still attached, and
    the loop below then appended a THIRD layer on top of that. The fix
    above (2026-10-07) made one run idempotent against ONE prior
    widening; it never handled an input that was already stacked twice.
    Looping until stable — not a single .sub() call — is what actually
    guarantees the true base label no matter how many times this bug (or
    any future one shaped like it) has already compounded."""
    while True:
        stripped = _WIDEN_SUFFIX_RE.sub("", label)
        if stripped == label:
            return label
        label = stripped


def _report_stats(r_values):
    """The user-facing reporting stats (event count, mean/median R, win
    rate, profit factor, max drawdown, losing streak) — deliberately NOT
    persisted to the trial log (that's run_deep_backtest_rr's own job, a
    different, already-existing schema everything else relies on); these
    are for THIS SCRIPT's own printed comparison only. Same formulas
    partial_scale_out_backtest.py's own _stats/_longest_losing_streak
    already use elsewhere in this project, reused here rather than
    reinvented."""
    r = np.asarray(r_values, dtype=float)
    if len(r) == 0:
        return None
    wins, losses = r[r > 0], r[r < 0]
    cum = np.cumsum(r)
    running_max = np.maximum.accumulate(cum)
    max_drawdown = float((running_max - cum).max()) if len(cum) else 0.0
    worst_streak = cur = 0
    for v in r:
        cur = cur + 1 if v <= 0 else 0
        worst_streak = max(worst_streak, cur)
    return {
        "n": len(r), "mean_r": round(float(r.mean()), 3), "median_r": round(float(np.median(r)), 3),
        "win_rate_pct": round(float((r > 0).mean() * 100), 1),
        "profit_factor": round(float(wins.sum() / abs(losses.sum())), 2) if len(losses) else None,
        "max_drawdown_r": round(max_drawdown, 2), "worst_losing_streak": worst_streak,
    }


def main():
    t0 = time.time()
    leaderboard = ticker_behavior.validated_leaderboard()
    fixed_rr = leaderboard[leaderboard["strategy"].str.endswith("_fixed_rr")]
    trials = ex.load_experiment_trials()

    n_run = n_skipped = 0
    results = []
    for _, row in fixed_rr.iterrows():
        ticker, tf_label, strategy, best_label = row["ticker"], row["tf_label"], row["strategy"], row["best_label"]
        base_label = _base_label(best_label)
        builder = BUILDER.get(strategy)
        if builder is None or tf_label not in TF_FETCH:
            n_skipped += 1
            continue
        match = trials[(trials.ticker == ticker) & (trials.tf_label == tf_label) & (trials.label == best_label)]
        if match.empty:
            n_skipped += 1
            continue
        # Strip any "_strategy_id" the stored settings dict may already
        # carry (a trial freshly re-run since Phase (a) shipped has one
        # baked in by run_deep_backtest_rr itself) — real bug caught
        # testing this on live data: passed straight through to a builder
        # function as **kwargs, it raised "unexpected keyword argument
        # '_strategy_id'". Harmless for run_deep_backtest_rr itself (it
        # overwrites the key with a freshly computed one either way) but
        # fatal for the builder call below.
        base_settings = {k: v for k, v in match.iloc[0]["settings"].items() if not k.startswith("_")}
        base_rr = base_settings.get("rr_multiple")
        if not base_rr:
            n_skipped += 1
            continue

        try:
            df = research_data.get_yf_ohlcv(ticker, **TF_FETCH[tf_label])
            if df.empty:
                n_skipped += 1
                continue
        except Exception as e:
            print(f"  fetch failed for {ticker} {tf_label}: {e}")
            n_skipped += 1
            continue

        print(f"{ticker} {tf_label} {strategy} (base rr_multiple={base_rr})")
        for widen in WIDEN_FACTORS:
            variant_settings = {**base_settings, "rr_multiple": base_rr * widen}
            try:
                events = builder(df, **variant_settings)
            except Exception as e:
                print(f"  widen={widen}: event build failed — {e}")
                continue
            if events.empty:
                print(f"  widen={widen}: no events")
                continue

            stats = _report_stats(events["r_real"].tolist())
            label = f"{base_label} · widen={widen}"
            # run_deep_backtest_rr_with_lock, not the plain function — this
            # script is genuinely new research surface (target-distance
            # widening), so it's started clean on the 3-way RESEARCH/
            # VALIDATION/LOCKED split from the beginning instead of
            # retrofitting it in later. See that function's own docstring
            # for the written LOCKED-data unlock policy.
            trial = ex.run_deep_backtest_rr_with_lock(ticker, tf_label, events, base_settings, label,
                                                       target_widen_mult=widen)
            n_run += 1

            results.append({
                "ticker": ticker, "tf_label": tf_label, "strategy": strategy, "widen": widen,
                "stats": stats, "verdict": trial.get("verdict"),
                "p_value_train": trial.get("p_value_train"), "holdout_verdict": trial.get("holdout_verdict"),
                "p_value_holdout": trial.get("p_value_holdout"), "same_sign": trial.get("same_sign"),
                "strategy_id": trial.get("strategy_id"),
            })
            pf = f"{stats['profit_factor']:.2f}" if stats and stats["profit_factor"] is not None else "n/a"
            p_train = trial.get("p_value_train")
            p_train_str = f"{p_train:.4f}" if p_train is not None else "n/a"
            print(f"  widen={widen:<4} n={stats['n']:4d}  mean_r={stats['mean_r']:+.3f}  "
                  f"median_r={stats['median_r']:+.3f}  win_rate={stats['win_rate_pct']:5.1f}%  "
                  f"pf={pf}  max_dd={stats['max_drawdown_r']:.2f}R  worst_streak={stats['worst_losing_streak']}  "
                  f"verdict={trial.get('verdict')}  p_train={p_train_str}  holdout={trial.get('holdout_verdict')}  "
                  f"same_sign={trial.get('same_sign')}")

    print(f"\nRan {n_run} variant trials across {len(fixed_rr) - n_skipped} combos "
          f"({n_skipped} skipped — no settings/data/builder), in {time.time()-t0:.0f}s.")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "target_variant_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"checked_at": time.time(), "results": results}, f, indent=2, default=str)
    print(f"Wrote {len(results)} variant results to {out_path}")

    print("Rebuilding ticker_behavior.db so BH correction and the leaderboard reflect every new trial...")
    ticker_behavior.rebuild_db()


if __name__ == "__main__":
    main()
