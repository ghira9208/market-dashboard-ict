"""
CLI entry point for the gap-count-tagged Judas swing event study — the
second half of wiring entry_zone in as the real trigger: now that a setup
under tag_gap_count=True only fires on the trader's own rule-mandated entry
(1st gap for 2, 2nd for 3, Fibonacci OTE for 4+), this runs THOSE setups
through the project's existing, trusted run_event_study() permutation test,
sliced by the trader's own taxonomy (gap_count_full_code, and the SLG/
non-SLG split), rather than treating the whole set as one undifferentiated
pool.

    python3 -m research.run_gap_study --ticker "GBPUSD=X" --period histdata-full --interval 1m --provider histdata --resample 15min

Every slice below is run independently and logged independently via
experiment_log.log_run — deliberately, not as a shortcut. A 12-way split of
even a large sample can produce a stray p<0.05 slice by chance alone; the
project's own experiment_log.py exists specifically so "how many things did
we try" stays visible next to any one result that looks good. Nothing here
overrides that discipline — if anything, slicing by 8-12 codes makes it more
important, not less.
"""

import argparse

from research.backtest import run_event_study
from research.data_loader import load_history
from research.events import extract_judas_gap_tagged_events
from research.experiment_log import log_run

# Every code detect_judas_swing_setups(tag_gap_count=True) can produce,
# with and without the -SLG suffix — fixed here (not read off the data) so
# a code with zero observed setups still prints as "n=0", not silently
# disappears from the report.
_BASE_CODES = ["OSG", "TG", "TCG", "3G", "3CG", "MG"]
_ALL_CODES = _BASE_CODES + [f"{c}-SLG" for c in _BASE_CODES]


def _print_result(label, result):
    n = result.get("n_events", 0)
    if not n:
        print(f"  {label:10s} n=0")
        return
    verdict = result["verdict"]
    marker = "***" if verdict == "EDGE_FOUND" else "   "
    print(f"  {label:10s} n={n:4d}  win_rate={result['win_rate_after_cost']*100:5.1f}%  "
          f"mean_ret={result['mean_return_after_cost']*100:+6.3f}%  "
          f"p={result['p_value_vs_random_direction']:.4f}  {marker} {verdict}")


def main():
    p = argparse.ArgumentParser(description="Gap-count-tagged Judas swing event study, sliced by the trader's own taxonomy.")
    p.add_argument("--ticker", default="GBPUSD=X")
    p.add_argument("--period", default="histdata-full")
    p.add_argument("--interval", default="1m")
    p.add_argument("--provider", default="histdata")
    p.add_argument("--resample", default="15min",
                    help="Resample the loaded interval to this pandas offset before detection "
                         "(e.g. 15min, 1h) — sweep/CHoCH/gap detection on raw 1-minute noise isn't "
                         "the same problem as on 15m/1h bars; pass '' to skip resampling.")
    p.add_argument("--forward-bars", type=int, default=10)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--choch-window-bars", type=int, default=20)
    p.add_argument("--retrace-window-bars", type=int, default=30)
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()

    print(f"Loading {args.ticker} {args.interval} ({args.period}, provider={args.provider})...")
    df = load_history(args.ticker, args.period, args.interval, provider=args.provider, refresh=args.refresh)
    print(f"  {len(df)} bars, {df.index[0]} -> {df.index[-1]}")

    if args.resample:
        df = df.resample(args.resample).agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        ).dropna()
        print(f"  resampled to {args.resample}: {len(df)} bars, {df.index[0]} -> {df.index[-1]}")

    print(f"\nDetecting gap-tagged Judas swing setups (choch_window={args.choch_window_bars}, "
          f"retrace_window={args.retrace_window_bars})...")
    events = extract_judas_gap_tagged_events(
        df, forward_bars=args.forward_bars,
        choch_window_bars=args.choch_window_bars, retrace_window_bars=args.retrace_window_bars,
    )
    print(f"  {len(events)} setups with a completed forward window.")

    extra_params = {
        "choch_window_bars": args.choch_window_bars, "retrace_window_bars": args.retrace_window_bars,
        "resample": args.resample,
    }

    print("\n--- Aggregate (all gap-tagged setups pooled) ---")
    agg_result = run_event_study(events, cost_bps=args.cost_bps, direction_col="event_type")
    log_run(args.ticker, f"{args.interval}->{args.resample or args.interval}", args.period, args.provider,
            "Judas Swing (gap-tagged, entry_zone-wired) — aggregate",
            args.forward_bars, args.cost_bps, extra_params, agg_result)
    _print_result("ALL", agg_result)

    print(f"\n--- By gap-count code (each slice its own independent test, n={len(_ALL_CODES)} slices) ---")
    if "gap_count_full_code" not in events.columns or events.empty:
        print("  (no setups at all — nothing to slice)")
    else:
        for code in _ALL_CODES:
            sliced = events[events["gap_count_full_code"] == code]
            result = run_event_study(sliced, cost_bps=args.cost_bps, direction_col="event_type")
            log_run(args.ticker, f"{args.interval}->{args.resample or args.interval}", args.period, args.provider,
                    f"Judas Swing (gap-tagged, entry_zone-wired) — {code}",
                    args.forward_bars, args.cost_bps, extra_params, result)
            _print_result(code, result)

    print("\n--- SLG vs non-SLG (displacement leg 1 vs leg 2) ---")
    if "slg" in events.columns and not events.empty:
        for slg_val, label in ((False, "non-SLG"), (True, "SLG")):
            sliced = events[events["slg"] == slg_val]
            result = run_event_study(sliced, cost_bps=args.cost_bps, direction_col="event_type")
            log_run(args.ticker, f"{args.interval}->{args.resample or args.interval}", args.period, args.provider,
                    f"Judas Swing (gap-tagged, entry_zone-wired) — {label}",
                    args.forward_bars, args.cost_bps, extra_params, result)
            _print_result(label, result)

    print(f"\n{len(_ALL_CODES) + 3} independent tests logged this run (aggregate + {len(_ALL_CODES)} "
          f"gap-count codes + SLG/non-SLG) — see research/.cache/experiment_log.csv for the full, "
          f"permanent record. A single slice clearing p<0.05 here is exactly what you'd expect to see "
          f"by chance alone roughly 1 time in 20, purely from running this many slices — treat any one "
          f"EDGE_FOUND slice as a lead to re-test on fresh/out-of-sample data, not as a standalone result.")


if __name__ == "__main__":
    main()
