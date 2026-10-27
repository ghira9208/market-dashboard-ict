"""Real backtest for a partial scale-out exit, requested specifically to
check whether it's a legitimate way to make the equity curve easier to
sit through psychologically — NOT to chase a higher win-rate number for
its own sake (that trap is already flagged in _simulate_fixed_rr's own
docstring: a smaller target relative to the stop hits more often BY
CONSTRUCTION, which means nothing unless expectancy stays positive).

Mechanic tested: close SCALE_OUT_PCT of the position at the combo's own
ORIGINAL validated rr_multiple (the real, statistically-tested target,
before bot_engine.py's own TARGET_WIDEN_MULT=3.0x live widening), move
the stop on the remainder to breakeven, let it ride to the REAL live
target (rr_multiple * TARGET_WIDEN_MULT) or time out at max_bars. This
is compared against today's REAL live behavior — a single full-size exit
at the already-widened target — on the exact same real historical
touches, same ATR-based risk unit, same 2.0bps real cost assumption
experiments.py's own _simulate_fixed_rr already uses.

Run standalone: python3 studies/partial_scale_out_backtest.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/Users/apple/dev/edge-pipeline")

import experiments as ex
import ticker_behavior
import data as research_data
import db

TARGET_WIDEN_MULT = 3.0  # bot_engine.py's own real live constant, mirrored exactly
SCALE_OUT_PCT = 0.5      # fraction of the position closed at the original (near) target
MOVE_STOP_TO_BREAKEVEN = True

TF_FETCH = {
    "1D": dict(period="10y", interval="1d"),
    "1h": dict(period="730d", interval="1h"),
    "5m": dict(period="60d", interval="5m"),
}

TOUCH_FINDERS = {
    "clean_expansion": lambda df, s: ex._clean_expansion_touches(df, s["min_streak"], s["retr_window_bars"]),
    "clean_expansion_liquidity": lambda df, s: ex._clean_expansion_liquidity_touches(
        df, s["min_streak"], s["retr_window_bars"], s.get("sweep_window_bars", 10)),
    "clean_retracement_resumption": lambda df, s: ex._clean_retracement_resumption_touches(
        df, s["min_streak"], s["retr_window_bars"]),
}


def _simulate_partial_scale_out(df, touches, rr_target1, rr_target2, atr_mult_stop,
                                 scale_out_pct=SCALE_OUT_PCT, move_stop_to_breakeven=MOVE_STOP_TO_BREAKEVEN,
                                 max_bars=40, cost_bps=2.0):
    """Same bar-by-bar walk, same stop-first-on-same-bar tie convention,
    same ATR risk unit and cost model as experiments.py's own
    _simulate_fixed_rr — extended with a second leg. Returns a DataFrame
    with blended r_real (what actually got realized across both legs)
    plus leg1_hit/leg2_outcome for transparency on which trades actually
    used the second leg at all."""
    high = (df["High"] if "High" in df else df["high"]).to_numpy()
    low = (df["Low"] if "Low" in df else df["low"]).to_numpy()
    close = (df["Close"] if "Close" in df else df["close"]).to_numpy()
    open_ = (df["Open"] if "Open" in df else df["open"]).to_numpy()
    atr = ex.atr_series(df).to_numpy()
    n = len(df)

    def _run(entry_pos, entry_price, risk, direction):
        cost_r = (entry_price * cost_bps / 10_000.0) / risk
        if direction == "bullish":
            stop1 = entry_price - risk
            target1 = entry_price + risk * rr_target1
            target2 = entry_price + risk * rr_target2
        else:
            stop1 = entry_price + risk
            target1 = entry_price - risk * rr_target1
            target2 = entry_price - risk * rr_target2

        # ---- leg 1: full position until target1 or stop ----
        leg1_hit = False
        j = entry_pos
        for j in range(entry_pos, min(n, entry_pos + max_bars)):
            h, l = high[j], low[j]
            if direction == "bullish":
                hit_stop, hit_t1 = l <= stop1, h >= target1
            else:
                hit_stop, hit_t1 = h >= stop1, l <= target1
            if hit_stop:
                # Both legs stop out together at the ORIGINAL stop — the
                # whole point of a scale-out is moot if price never even
                # reaches the near target first.
                return -1.0 - cost_r, j, False, "stopped_before_t1"
            if hit_t1:
                leg1_hit = True
                break
        if not leg1_hit:
            # Timed out before ever reaching target1 — same honest
            # time-based close _simulate_fixed_rr itself falls back to,
            # applied to the full (unscaled) position.
            exit_price = close[j]
            realized = (exit_price - entry_price) if direction == "bullish" else (entry_price - exit_price)
            return realized / risk - cost_r, j, False, "timed_out_before_t1"

        leg1_r = rr_target1

        # ---- leg 2: remaining position, stop moved to breakeven (optional) ----
        stop2 = entry_price if move_stop_to_breakeven else stop1
        for k in range(j + 1, min(n, entry_pos + max_bars)):
            h, l = high[k], low[k]
            if direction == "bullish":
                hit_stop2, hit_t2 = l <= stop2, h >= target2
            else:
                hit_stop2, hit_t2 = h >= stop2, l <= target2
            if hit_stop2:
                leg2_r = 0.0 if move_stop_to_breakeven else -1.0
                blended = scale_out_pct * leg1_r + (1 - scale_out_pct) * leg2_r - cost_r
                return blended, k, True, "breakeven_stop" if move_stop_to_breakeven else "stopped_after_t1"
            if hit_t2:
                blended = scale_out_pct * leg1_r + (1 - scale_out_pct) * rr_target2 - cost_r
                return blended, k, True, "full_target"
        # Timed out after t1 but before resolving leg 2 — close the
        # remainder at market, same honest convention as the single-exit
        # version's own fallback.
        k = min(n - 1, entry_pos + max_bars - 1)
        exit_price = close[k]
        leg2_r = ((exit_price - entry_price) if direction == "bullish" else (entry_price - exit_price)) / risk
        blended = scale_out_pct * leg1_r + (1 - scale_out_pct) * leg2_r - cost_r
        return blended, k, True, "timed_out_after_t1"

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
        r_real, exit_pos, reached_t1, outcome = _run(entry_pos, entry_price, risk, t["direction"])
        rows.append({"entry_time": df.index[entry_pos], "r_real": r_real,
                     "reached_t1": reached_t1, "outcome": outcome, "direction": t["direction"]})
        last_exit_pos = exit_pos
    return pd.DataFrame(rows, columns=["entry_time", "r_real", "reached_t1", "outcome", "direction"])


def _stats(r_values):
    r = np.array(r_values)
    if len(r) == 0:
        return None
    wins = r[r > 0]
    return {
        "n": len(r), "win_rate_pct": round(float((r > 0).mean() * 100), 1),
        "mean_r": round(float(r.mean()), 3), "total_r": round(float(r.sum()), 2),
        "max_single_loss_r": round(float(r.min()), 3),
        "profit_factor": round(float(wins.sum() / abs(r[r < 0].sum())), 2) if (r < 0).any() else None,
    }


def _longest_losing_streak(r_values):
    worst = cur = 0
    for v in r_values:
        cur = cur + 1 if v <= 0 else 0
        worst = max(worst, cur)
    return worst


def main():
    tracked = db.tracked_positions("dev_user")
    live_keys = sorted({(t["ticker"], t["tf_label"], t["strategy"]) for t in tracked})
    print(f"Running on {len(live_keys)} real live combos from dev_user's own account:\n")

    trials = ex.load_experiment_trials()
    lb = ticker_behavior.validated_leaderboard()

    results = []
    for ticker, tf_label, strategy in live_keys:
        row = lb[(lb.ticker == ticker) & (lb.tf_label == tf_label) & (lb.strategy == strategy)]
        if row.empty:
            print(f"  {ticker} {tf_label} {strategy}: not on the validated leaderboard, skipping")
            continue
        best_label = row.iloc[0]["best_label"]
        match = trials[(trials.ticker == ticker) & (trials.tf_label == tf_label) & (trials.label == best_label)]
        if match.empty:
            print(f"  {ticker} {tf_label} {strategy}: no matching trial settings, skipping")
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

        rr_target2 = rr_multiple * TARGET_WIDEN_MULT
        baseline = ex._simulate_fixed_rr(df, touches, rr_multiple=rr_target2,
                                          atr_mult_stop=atr_mult_stop, max_bars=settings.get("max_bars", 40))
        scaled = _simulate_partial_scale_out(df, touches, rr_target1=rr_multiple, rr_target2=rr_target2,
                                              atr_mult_stop=atr_mult_stop, max_bars=settings.get("max_bars", 40))

        base_stats = _stats(baseline["r_real"].tolist())
        scaled_stats = _stats(scaled["r_real"].tolist())
        if base_stats is None or scaled_stats is None:
            continue
        results.append({
            "combo": f"{ticker} {tf_label} {strategy}", "base": base_stats, "scaled": scaled_stats,
            "base_losing_streak": _longest_losing_streak(baseline["r_real"].tolist()),
            "scaled_losing_streak": _longest_losing_streak(scaled["r_real"].tolist()),
            "pct_trades_reaching_t1": round(float(scaled["reached_t1"].mean() * 100), 1),
        })

        print(f"{ticker:10s} {tf_label:4s} {strategy}")
        print(f"  CURRENT (single exit, widened target): n={base_stats['n']:3d}  "
              f"win_rate={base_stats['win_rate_pct']:5.1f}%  mean_r={base_stats['mean_r']:+.3f}  "
              f"total_r={base_stats['total_r']:+7.2f}  worst_streak={results[-1]['base_losing_streak']}")
        print(f"  SCALE-OUT ({int(SCALE_OUT_PCT*100)}% @ {rr_multiple:.2f}R, breakeven remainder): "
              f"win_rate={scaled_stats['win_rate_pct']:5.1f}%  mean_r={scaled_stats['mean_r']:+.3f}  "
              f"total_r={scaled_stats['total_r']:+7.2f}  worst_streak={results[-1]['scaled_losing_streak']}  "
              f"({results[-1]['pct_trades_reaching_t1']:.0f}% of trades ever reached the near target)")
        print()

    if not results:
        print("No comparable combos.")
        return

    print("=" * 70)
    print("AGGREGATE (pooled across all real live combos above)")
    all_base_wr = np.mean([r["base"]["win_rate_pct"] for r in results])
    all_scaled_wr = np.mean([r["scaled"]["win_rate_pct"] for r in results])
    all_base_meanr = np.mean([r["base"]["mean_r"] for r in results])
    all_scaled_meanr = np.mean([r["scaled"]["mean_r"] for r in results])
    all_base_streak = np.mean([r["base_losing_streak"] for r in results])
    all_scaled_streak = np.mean([r["scaled_losing_streak"] for r in results])
    print(f"Mean win rate:          {all_base_wr:5.1f}%  ->  {all_scaled_wr:5.1f}%")
    print(f"Mean expectancy (R):    {all_base_meanr:+.3f}  ->  {all_scaled_meanr:+.3f}")
    print(f"Mean worst losing streak: {all_base_streak:.1f}  ->  {all_scaled_streak:.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
