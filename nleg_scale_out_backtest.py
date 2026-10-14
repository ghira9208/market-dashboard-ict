"""Does adding MORE scale-out legs (3, 4 — matching cTrader's own desktop
"up to 4 TP" ladder) beat the 2-leg 50/50 split already live? Not assumed —
measured, same real historical touches/ATR-stop/cost model as
partial_scale_out_backtest.py, generalized to N legs instead of 2.

Ladder construction: N evenly-spaced R-multiple targets between the
combo's own validated rr_multiple (1x, today's near leg) and
rr_multiple * TARGET_WIDEN_MULT (3x, today's far/live target) — the same
two endpoints the live 2-leg version already uses, just with intermediate
TP levels inserted between them, same shape as cTrader's own manual
multi-TP ladder. Equal position split across legs. Stop rule, kept
IDENTICAL to the live 2-leg design (bot_engine.py's
_check_closed_scale_out_legs): the moment ANY leg closes at its own
target, every remaining still-open leg's stop moves to breakeven — no
trailing beyond that, so this backtest is a true predictor of what the
already-built live mechanism would do if extended to N legs, not a
different (untested) exit rule.

Run standalone: python3 nleg_scale_out_backtest.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/Users/apple/dev/edge-pipeline")

import experiments as ex
import ticker_behavior
import data as research_data
import db
from partial_scale_out_backtest import (
    TARGET_WIDEN_MULT, TF_FETCH, TOUCH_FINDERS, _simulate_partial_scale_out, _stats, _longest_losing_streak,
)


def _simulate_n_leg_scale_out(df, touches, rr_targets, split_fracs, atr_mult_stop, max_bars=40, cost_bps=2.0):
    """N-leg generalization of _simulate_partial_scale_out. rr_targets
    ascending, split_fracs sums to 1.0, same length. Per bar: every
    still-open leg's stop is checked FIRST (stop-first tie convention,
    identical to the 2-leg version), then every still-open leg's target;
    any leg closing at its target this bar immediately moves every OTHER
    still-open leg's stop to breakeven for the next bar — mirrors
    bot_engine.py's real live rule exactly, not a new invention."""
    assert len(rr_targets) == len(split_fracs)
    assert abs(sum(split_fracs) - 1.0) < 1e-9
    high = (df["High"] if "High" in df else df["high"]).to_numpy()
    low = (df["Low"] if "Low" in df else df["low"]).to_numpy()
    close = (df["Close"] if "Close" in df else df["close"]).to_numpy()
    open_ = (df["Open"] if "Open" in df else df["open"]).to_numpy()
    atr = ex.atr_series(df).to_numpy()
    n = len(df)
    n_legs = len(rr_targets)

    def _run(entry_pos, entry_price, risk, direction):
        cost_r = (entry_price * cost_bps / 10_000.0) / risk
        if direction == "bullish":
            targets = [entry_price + risk * rr for rr in rr_targets]
            orig_stop = entry_price - risk
        else:
            targets = [entry_price - risk * rr for rr in rr_targets]
            orig_stop = entry_price + risk

        leg_open = [True] * n_legs
        leg_r = [None] * n_legs
        leg_stop = [orig_stop] * n_legs
        reached_any_target = False

        j = entry_pos
        end = min(n, entry_pos + max_bars)
        for j in range(entry_pos, end):
            h, l = high[j], low[j]
            for i in range(n_legs):
                if not leg_open[i]:
                    continue
                hit_stop = (l <= leg_stop[i]) if direction == "bullish" else (h >= leg_stop[i])
                if hit_stop:
                    leg_r[i] = (leg_stop[i] - entry_price) / risk if direction == "bullish" \
                        else (entry_price - leg_stop[i]) / risk
                    leg_open[i] = False
            newly_closed_target = False
            for i in range(n_legs):
                if not leg_open[i]:
                    continue
                hit_t = (h >= targets[i]) if direction == "bullish" else (l <= targets[i])
                if hit_t:
                    leg_r[i] = rr_targets[i]
                    leg_open[i] = False
                    newly_closed_target = True
                    reached_any_target = True
            if newly_closed_target:
                for i in range(n_legs):
                    if leg_open[i]:
                        leg_stop[i] = entry_price
            if not any(leg_open):
                break

        if any(leg_open):
            exit_price = close[min(j, n - 1)]
            for i in range(n_legs):
                if leg_open[i]:
                    leg_r[i] = ((exit_price - entry_price) if direction == "bullish"
                                else (entry_price - exit_price)) / risk

        blended = sum(f * r for f, r in zip(split_fracs, leg_r)) - cost_r
        return blended, j, reached_any_target

    rows = []
    last_exit_pos = -1
    for t in sorted(touches, key=lambda t: t["confirm_pos"]):
        entry_pos = t["confirm_pos"] + 1
        if entry_pos <= last_exit_pos or entry_pos >= n:
            continue
        a = atr[entry_pos]
        if pd.isna(a) or a <= 0:
            continue
        entry_price = open_[entry_pos]
        risk = a * atr_mult_stop
        r_real, exit_pos, reached = _run(entry_pos, entry_price, risk, t["direction"])
        rows.append({"entry_time": df.index[entry_pos], "r_real": r_real,
                     "reached_any_target": reached, "direction": t["direction"]})
        last_exit_pos = exit_pos
    return pd.DataFrame(rows, columns=["entry_time", "r_real", "reached_any_target", "direction"])


def _ladder(rr_multiple, n_legs):
    rr_targets = [round(rr_multiple * m, 4) for m in np.linspace(1.0, TARGET_WIDEN_MULT, n_legs)]
    split_fracs = [1.0 / n_legs] * n_legs
    return rr_targets, split_fracs


def main():
    tracked = db.tracked_positions("dev_user")
    live_keys = sorted({(t["ticker"], t["tf_label"], t["strategy"]) for t in tracked})
    print(f"Running on {len(live_keys)} real live combos from dev_user's own account:\n")

    trials = ex.load_experiment_trials()
    lb = ticker_behavior.validated_leaderboard()

    configs = [2, 3, 4]
    pooled = {n: {"wr": [], "mean_r": [], "streak": []} for n in configs}
    pooled["base"] = {"wr": [], "mean_r": [], "streak": []}

    for ticker, tf_label, strategy in live_keys:
        row = lb[(lb.ticker == ticker) & (lb.tf_label == tf_label) & (lb.strategy == strategy)]
        if row.empty:
            continue
        best_label = row.iloc[0]["best_label"]
        match = trials[(trials.ticker == ticker) & (trials.tf_label == tf_label) & (trials.label == best_label)]
        if match.empty:
            continue
        settings = match.iloc[0]["settings"]
        rr_multiple, atr_mult_stop = settings.get("rr_multiple"), settings.get("atr_mult_stop")
        if not rr_multiple or not atr_mult_stop:
            continue

        family = strategy[: -len("_fixed_rr")]
        finder = TOUCH_FINDERS.get(family)
        if finder is None:
            continue
        fetch_kwargs = TF_FETCH.get(tf_label)
        if fetch_kwargs is None:
            continue
        try:
            df = research_data.get_yf_ohlcv(ticker, **fetch_kwargs)
        except Exception as e:
            print(f"  {ticker} {tf_label} {strategy}: fetch failed ({e}), skipping")
            continue
        if df is None or df.empty:
            continue
        touches = finder(df, settings)
        if not touches:
            continue

        max_bars = settings.get("max_bars", 40)
        rr_target2 = rr_multiple * TARGET_WIDEN_MULT
        baseline = ex._simulate_fixed_rr(df, touches, rr_multiple=rr_target2,
                                          atr_mult_stop=atr_mult_stop, max_bars=max_bars)
        base_stats = _stats(baseline["r_real"].tolist())
        if base_stats is None:
            continue

        print(f"{ticker:10s} {tf_label:4s} {strategy}")
        print(f"  BASELINE (no scale-out):        n={base_stats['n']:3d}  win_rate={base_stats['win_rate_pct']:5.1f}%  "
              f"mean_r={base_stats['mean_r']:+.3f}  worst_streak={_longest_losing_streak(baseline['r_real'].tolist())}")
        pooled["base"]["wr"].append(base_stats["win_rate_pct"])
        pooled["base"]["mean_r"].append(base_stats["mean_r"])
        pooled["base"]["streak"].append(_longest_losing_streak(baseline["r_real"].tolist()))

        for n_legs in configs:
            if n_legs == 2:
                scaled = _simulate_partial_scale_out(df, touches, rr_target1=rr_multiple, rr_target2=rr_target2,
                                                      atr_mult_stop=atr_mult_stop, max_bars=max_bars)
            else:
                rr_targets, split_fracs = _ladder(rr_multiple, n_legs)
                scaled = _simulate_n_leg_scale_out(df, touches, rr_targets, split_fracs,
                                                    atr_mult_stop=atr_mult_stop, max_bars=max_bars)
            s = _stats(scaled["r_real"].tolist())
            if s is None:
                continue
            streak = _longest_losing_streak(scaled["r_real"].tolist())
            print(f"  {n_legs}-LEG scale-out:               n={s['n']:3d}  win_rate={s['win_rate_pct']:5.1f}%  "
                  f"mean_r={s['mean_r']:+.3f}  worst_streak={streak}")
            pooled[n_legs]["wr"].append(s["win_rate_pct"])
            pooled[n_legs]["mean_r"].append(s["mean_r"])
            pooled[n_legs]["streak"].append(streak)
        print()

    print("=" * 78)
    print("AGGREGATE (mean across all real live combos)")
    print(f"{'config':<22s} {'win rate':>10s} {'mean R':>10s} {'worst streak':>14s}")
    print(f"{'baseline (no scale-out)':<22s} {np.mean(pooled['base']['wr']):9.1f}% "
          f"{np.mean(pooled['base']['mean_r']):+9.3f} {np.mean(pooled['base']['streak']):14.1f}")
    for n_legs in configs:
        if not pooled[n_legs]["wr"]:
            continue
        print(f"{str(n_legs)+'-leg scale-out':<22s} {np.mean(pooled[n_legs]['wr']):9.1f}% "
              f"{np.mean(pooled[n_legs]['mean_r']):+9.3f} {np.mean(pooled[n_legs]['streak']):14.1f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
