"""
CLI entry point for the trader's ACTUAL exit rule, not the flat forward_bars
proxy every other study in this pipeline uses — direct implementation of:
"AT 1:1 RR set Break Even and the end event when price interacts with either
the stop or TP. consider BE a stop at entry point." Plus seven more of the
trader's own stated constraints, layered on top of that: "stop loss minimum
4 pips" (dropped, not widened, if the zone can't support it), "all trades
should end at the before close of the markets," "no asian session start
with london end with ny pm," "no trade in the last hour of market close —
liquidity is thinning," "choch window [drops] from 1h to 20m" (a real-TIME
taper — see --choch-window-minutes below, not a bar count), and "the [trade]
closed at session end, consider the price 5 minutes before [to] calculate
if it was a win or a loss" (see --session-close-lookback-minutes below).

    python3 -m research.run_stoptarget_study --resample 15min
    python3 -m research.run_stoptarget_study --years 3 --resample ''

This is the honest next step flagged in the Field Notes (§11/§12) once
run_gap_study.py's flat 10-bar hold came back NO_EDGE on all 15 slices,
uniformly negative: that read scores every setup off whatever price is
doing exactly 10 bars later, whether or not the trader's own rule would
have already closed the trade (win, loss, or breakeven) long before or
after that point. This script re-runs the SAME gap-tagged setups, same
taxonomy, same permutation test — the only thing that changes is how each
trade's outcome is actually determined: entry_zone's own far edge as the
stop (dropped outright if it's narrower than min_stop_pips — never
widened), a fixed 2R target, moved to breakeven the instant price reaches
1:1, exited the moment price touches whichever of those is still active —
or, failing that, force-closed the moment the trading day's own window
closes (priced off a snapshot a few minutes earlier than the literal final
tick — see --session-close-lookback-minutes). Only setups entered inside
that window, and outside its own last hour (defaults to London Open's
start through one hour before NY PM's end, NY time — see
research/events.py's own extract_judas_stoptarget_events docstring for
exactly how "no Asian session" maps onto that), are scored at all.

--years N restricts the loaded history to just its own last N years
(measured from the data's own last timestamp, not today's date) BEFORE
any resampling or detection runs — a much smaller, faster slice than the
default full ~25-year cache, useful for a quick, fresher-data read (e.g.
"3 years on the 1-minute chart," --resample '' left off entirely so
detection runs on the raw loaded interval instead of resampling it down).

Same multiple-comparisons discipline as run_gap_study.py — every slice
logged independently via experiment_log.log_run, regardless of verdict;
this now includes a slice per entry-hour block and one for the
session-close-forced trades specifically, on top of the gap-count/SLG
slices already logged — see the final summary line's own count.
"""

import argparse
from datetime import time as dtime

import pandas as pd

from research.backtest import run_event_study
from research.data_loader import load_history
from research.events import extract_judas_stoptarget_events
from research.experiment_log import log_run

_BASE_CODES = ["OSG", "TG", "TCG", "3G", "3CG", "MG"]
_ALL_CODES = _BASE_CODES + [f"{c}-SLG" for c in _BASE_CODES]
_EXIT_KINDS = ("win", "be", "loss", "session_close_win", "session_close_loss", "session_close_be")


def _parse_hhmm(s):
    h, m = s.split(":")
    return dtime(int(h), int(m))


def _exit_kind_breakdown(events):
    if events.empty or "exit_kind" not in events.columns:
        return ""
    counts = events["exit_kind"].value_counts()
    n = len(events)
    parts = [f"{k}={counts.get(k, 0)} ({counts.get(k, 0) / n * 100:.1f}%)" for k in _EXIT_KINDS if counts.get(k, 0)]
    return "  [" + ", ".join(parts) + "]"


def _realized_rr(events):
    """The ACTUAL, realized reward:risk ratio for a set of trades — mean
    winning R-multiple divided by mean losing R-multiple (absolute
    value) — as opposed to the fixed 2:1 risk_reward configured for
    trades that hit a real target. Meant for session-close-forced trades
    specifically: those never actually reach the 2R target or the 1R
    stop, they're cut off wherever price happened to be when the day
    ended, so their realized R:R is an empirical question, not the
    configured 2.0. R-multiple = price distance from entry, divided by
    the trade's own original risk (entry-to-stop distance) — win/loss
    read straight off exit_kind, "be" trades excluded from both sides
    (zero by construction, would just dilute the ratio). Returns
    (avg_win_r, avg_loss_r, rr) — rr is None if there are no losers to
    divide by (or no winners, avg_win_r is then 0.0)."""
    if events.empty or "exit_kind" not in events.columns:
        return 0.0, 0.0, None
    risk = (events["entry_price"] - events["stop_price"]).abs()
    r_multiple = (events["exit_price"] - events["entry_price"]).abs() / risk.replace(0, pd.NA)
    wins = r_multiple[events["exit_kind"].str.endswith("win", na=False)].dropna()
    losses = r_multiple[events["exit_kind"].str.endswith("loss", na=False)].dropna()
    avg_win_r = float(wins.mean()) if len(wins) else 0.0
    avg_loss_r = float(losses.mean()) if len(losses) else 0.0
    rr = (avg_win_r / avg_loss_r) if avg_loss_r > 0 else None
    return avg_win_r, avg_loss_r, rr


def _print_result(label, result, events):
    n = result.get("n_events", 0)
    if not n:
        print(f"  {label:10s} n=0")
        return
    verdict = result["verdict"]
    marker = "***" if verdict == "EDGE_FOUND" else "   "
    print(f"  {label:10s} n={n:4d}  win_rate={result['win_rate_after_cost']*100:5.1f}%  "
          f"mean_ret={result['mean_return_after_cost']*100:+6.3f}%  "
          f"p={result['p_value_vs_random_direction']:.4f}  {marker} {verdict}"
          f"{_exit_kind_breakdown(events)}")


def main():
    p = argparse.ArgumentParser(
        description="Gap-count-tagged Judas swing event study, exited on the trader's own "
                    "stop/target/breakeven rule instead of a flat forward-bar hold.")
    p.add_argument("--ticker", default="GBPUSD=X")
    p.add_argument("--period", default="histdata-full")
    p.add_argument("--interval", default="1m")
    p.add_argument("--provider", default="histdata")
    p.add_argument("--resample", default="15min",
                    help="Resample the loaded interval to this pandas offset before detection "
                         "(e.g. 15min, 1h); pass '' to skip resampling and run detection directly "
                         "on the raw loaded interval (e.g. the 1-minute chart itself).")
    p.add_argument("--years", type=float, default=None,
                    help="Restrict to just the loaded history's own last N years (from its own last "
                         "timestamp, not today) before any resampling or detection — e.g. --years 3 "
                         "for a fast, recent-data-only slice instead of the full ~25-year cache.")
    p.add_argument("--risk-reward", type=float, default=2.0, help="Fixed reward:risk target (trader's rule: 2.0).")
    p.add_argument("--be-trigger-r", type=float, default=1.0,
                    help="R-multiple at which the stop moves to breakeven (trader's rule: 1.0, i.e. 1:1).")
    p.add_argument("--max-hold-bars", type=int, default=2000,
                    help="Cap on bars to walk forward per setup before calling it an unresolved timeout "
                         "(dropped, not scored as a loss) — rarely the binding constraint once the daily "
                         "session window below is active, since a trading day is far shorter than this.")
    p.add_argument("--min-stop-pips", type=float, default=4.0,
                    help="Hard floor on stop distance, in pips (trader's rule: 4). A zone-derived stop "
                         "narrower than this is DROPPED entirely, not widened into it — \"don't stretch the "
                         "stop loss to 4 points, drop the trade if it's less.\"")
    p.add_argument("--pip-size", type=float, default=0.0001, help="Price per pip (0.0001 for GBPUSD).")
    p.add_argument("--window-start-et", default="02:00",
                    help="Trading window start, NY time, HH:MM (trader's rule: London Open's own start, "
                         "02:00 ET). A setup not ENTERED inside [window-start-et, window-end-et] is dropped.")
    p.add_argument("--window-end-et", default="16:00",
                    help="Trading window end, NY time, HH:MM (trader's rule: NY PM's own end, 16:00 ET). "
                         "\"All trades should end before close of markets\": a setup still open when its "
                         "own trading day reaches this time is force-closed right there, at that bar's "
                         "own Close — never carried into the next session.")
    p.add_argument("--no-window", action="store_true",
                    help="Disable the trading-window filter and forced same-day close entirely, falling "
                         "back to the old always-24h, max-hold-bars-only behavior (for comparison).")
    p.add_argument("--entry-cutoff-hours", type=float, default=1.0,
                    help="\"No trade in the last hour of market close — liquidity is thinning.\" A setup "
                         "whose entry bar falls inside the trading window but within this many hours of "
                         "window-end-et is dropped and never entered — narrower than the window itself, "
                         "and separate from the forced same-day close: a trade that already entered earlier "
                         "is still held and force-closed at window-end-et exactly as before. Pass "
                         "--no-entry-cutoff to disable just this rule (entries allowed anywhere in the "
                         "window, forced close unchanged).")
    p.add_argument("--no-entry-cutoff", action="store_true",
                    help="Disable the last-hour entry cutoff specifically, keeping the trading window and "
                         "forced same-day close as-is (for comparison). Also automatically off whenever "
                         "--no-window is set, since there's no close to cut off before.")
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--choch-window-minutes", type=float, default=60,
                    help="CHoCH confirmation window, in REAL MINUTES (not bars) — used at/before "
                         "window-start-et and anywhere off-hours (trader's rule: 1 hour, i.e. 60). "
                         "Converted to a bar count internally using the loaded/resampled data's own bar "
                         "spacing, so the same real-world window applies whatever --resample is (or isn't) "
                         "set to — the earlier bars-based version of this flag silently meant a different "
                         "duration per timeframe, which is exactly the bug this fixes. With tapering "
                         "enabled (the default), this is the WIDE end of a linear taper that narrows "
                         "toward --choch-window-taper-to-minutes as the trading day approaches its forced "
                         "close — a sweep with little day left has no realistic use for a wide confirmation "
                         "window. Pass --no-choch-taper to use this value flat, everywhere, with no "
                         "narrowing (the old pre-taper behavior).")
    p.add_argument("--choch-window-taper-to-minutes", type=float, default=20,
                    help="Narrow (TIGHT) end of the CHoCH window taper, in real minutes (trader's rule: 20). "
                         "Reached --choch-window-taper-hours before window-end-et and held flat from there "
                         "through the close. Ignored if --no-choch-taper is set.")
    p.add_argument("--choch-window-taper-hours", type=float, default=1.0,
                    help="How many hours before window-end-et the taper reaches "
                         "--choch-window-taper-to-minutes and flattens out (trader's rule: 1 hour). Ignored "
                         "if --no-choch-taper is set.")
    p.add_argument("--no-choch-taper", action="store_true",
                    help="Disable the CHoCH window taper: use --choch-window-minutes flat, everywhere, all "
                         "day (for comparison against the tapered default). Tapering is also automatically "
                         "disabled whenever --no-window is set, since there's no forced close to taper toward.")
    p.add_argument("--session-close-lookback-minutes", type=float, default=5.0,
                    help="\"The one closed at session end, consider the price 5 minutes before and "
                         "calculate if it was a win or a loss.\" A forced same-day close is priced (and "
                         "classified win/loss/be) off the price this many real minutes before the actual "
                         "close bar, not the literal final tick — avoids the last, noisiest prints right at "
                         "the close deciding the call. Pass 0 to use the literal final tick instead (the "
                         "win/loss/be reclassification itself still applies either way).")
    p.add_argument("--retrace-window-bars", type=int, default=30)
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()

    window_start = None if args.no_window else _parse_hhmm(args.window_start_et)
    window_end = None if args.no_window else _parse_hhmm(args.window_end_et)
    choch_taper_to_minutes = None if args.no_choch_taper else args.choch_window_taper_to_minutes
    entry_cutoff_hours = None if (args.no_entry_cutoff or args.no_window) else args.entry_cutoff_hours

    print(f"Loading {args.ticker} {args.interval} ({args.period}, provider={args.provider})...")
    df = load_history(args.ticker, args.period, args.interval, provider=args.provider, refresh=args.refresh)
    print(f"  {len(df)} bars, {df.index[0]} -> {df.index[-1]}")

    if args.years:
        cutoff = df.index[-1] - pd.Timedelta(days=365.25 * args.years)
        df = df[df.index >= cutoff]
        print(f"  restricted to the last {args.years} years: {len(df)} bars, {df.index[0]} -> {df.index[-1]}")

    if args.resample:
        df = df.resample(args.resample).agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        ).dropna()
        print(f"  resampled to {args.resample}: {len(df)} bars, {df.index[0]} -> {df.index[-1]}")
    else:
        print(f"  no resample -- detection runs directly on the {args.interval} chart, {len(df)} bars.")

    window_desc = "disabled (24h)" if args.no_window else f"{args.window_start_et}-{args.window_end_et} ET"
    if choch_taper_to_minutes is None:
        choch_desc = f"flat {args.choch_window_minutes}min (taper disabled)"
    else:
        choch_desc = (f"{args.choch_window_minutes}min->{choch_taper_to_minutes}min, taper knee "
                      f"{args.choch_window_taper_hours}h before close")
    cutoff_desc = "disabled" if entry_cutoff_hours is None else f"no new entries in the last {entry_cutoff_hours}h"
    print(f"\nDetecting gap-tagged Judas swing setups and walking each forward on the trader's "
          f"stop/target/BE rule (RR={args.risk_reward}, BE at {args.be_trigger_r}R, "
          f"min_stop={args.min_stop_pips} pips [hard floor, drop below it], window={window_desc} "
          f"({cutoff_desc}), max_hold={args.max_hold_bars} bars, choch_window={choch_desc}, "
          f"session_close_lookback={args.session_close_lookback_minutes}min, "
          f"retrace_window={args.retrace_window_bars} bars)...")
    events = extract_judas_stoptarget_events(
        df, choch_window_minutes=args.choch_window_minutes, choch_window_taper_to_minutes=choch_taper_to_minutes,
        choch_window_taper_before_close_hours=args.choch_window_taper_hours,
        retrace_window_bars=args.retrace_window_bars,
        risk_reward=args.risk_reward, be_trigger_r=args.be_trigger_r, max_hold_bars=args.max_hold_bars,
        min_stop_pips=args.min_stop_pips, pip_size=args.pip_size,
        window_start_et=window_start, window_end_et=window_end,
        entry_cutoff_before_close_hours=entry_cutoff_hours,
        session_close_lookback_minutes=args.session_close_lookback_minutes,
    )
    detected = events.attrs.get("n_setups_detected", len(events))
    print(f"  {detected} gap-tagged setups detected -> {len(events)} resolved (scored) within the cap.")
    print(f"  dropped: no_room={events.attrs.get('n_dropped_no_room', 0)}  "
          f"zero_risk={events.attrs.get('n_dropped_zero_risk', 0)}  "
          f"wrong_side={events.attrs.get('n_dropped_wrong_side', 0)}  "
          f"sub_floor_stop={events.attrs.get('n_dropped_sub_floor_stop', 0)}  "
          f"outside_window={events.attrs.get('n_dropped_outside_window', 0)}  "
          f"last_hour={events.attrs.get('n_dropped_last_hour', 0)}  "
          f"data_ended={events.attrs.get('n_dropped_data_ended', 0)}  "
          f"timeout={events.attrs.get('n_dropped_timeout', 0)}")

    extra_params = {
        "choch_window_minutes": args.choch_window_minutes, "choch_window_taper_to_minutes": choch_taper_to_minutes,
        "choch_window_taper_hours": args.choch_window_taper_hours if choch_taper_to_minutes is not None else None,
        "retrace_window_bars": args.retrace_window_bars,
        "resample": args.resample, "years": args.years, "risk_reward": args.risk_reward,
        "be_trigger_r": args.be_trigger_r,
        "min_stop_pips": args.min_stop_pips, "min_stop_pips_mode": "drop", "window": window_desc,
        "entry_cutoff_hours": entry_cutoff_hours,
        "session_close_lookback_minutes": args.session_close_lookback_minutes,
        "bar_interval_minutes": events.attrs.get("bar_interval_minutes"),
        "exit_rule": "stop/target/BE-at-1:1, sub-floor stop dropped, forced same-day close reclassified "
                     "via session_close_lookback_minutes, last-hour entry cutoff (see run_stoptarget_study.py)",
    }

    print("\n--- Aggregate (all gap-tagged setups pooled) ---")
    agg_result = run_event_study(events, cost_bps=args.cost_bps, direction_col="event_type")
    log_run(args.ticker, f"{args.interval}->{args.resample or args.interval}", args.period, args.provider,
            "Judas Swing (gap-tagged, stop/target/BE exit) — aggregate",
            args.max_hold_bars, args.cost_bps, extra_params, agg_result)
    _print_result("ALL", agg_result, events)

    print(f"\n--- By gap-count code (each slice its own independent test, n={len(_ALL_CODES)} slices) ---")
    if "gap_count_full_code" not in events.columns or events.empty:
        print("  (no setups at all — nothing to slice)")
    else:
        for code in _ALL_CODES:
            sliced = events[events["gap_count_full_code"] == code]
            result = run_event_study(sliced, cost_bps=args.cost_bps, direction_col="event_type")
            log_run(args.ticker, f"{args.interval}->{args.resample or args.interval}", args.period, args.provider,
                    f"Judas Swing (gap-tagged, stop/target/BE exit) — {code}",
                    args.max_hold_bars, args.cost_bps, extra_params, result)
            _print_result(code, result, sliced)

    print("\n--- SLG vs non-SLG (displacement leg 1 vs leg 2) ---")
    n_slg_slices = 0
    if "slg" in events.columns and not events.empty:
        for slg_val, label in ((False, "non-SLG"), (True, "SLG")):
            sliced = events[events["slg"] == slg_val]
            result = run_event_study(sliced, cost_bps=args.cost_bps, direction_col="event_type")
            log_run(args.ticker, f"{args.interval}->{args.resample or args.interval}", args.period, args.provider,
                    f"Judas Swing (gap-tagged, stop/target/BE exit) — {label}",
                    args.max_hold_bars, args.cost_bps, extra_params, result)
            _print_result(label, result, sliced)
            n_slg_slices += 1

    print("\n--- By entry hour block (NY-local clock hour, each its own independent test) ---")
    n_hour_slices = 0
    if "entry_hour_block" not in events.columns or events.empty:
        print("  (no setups at all — nothing to slice)")
    else:
        for block in sorted(events["entry_hour_block"].unique()):
            sliced = events[events["entry_hour_block"] == block]
            result = run_event_study(sliced, cost_bps=args.cost_bps, direction_col="event_type")
            log_run(args.ticker, f"{args.interval}->{args.resample or args.interval}", args.period, args.provider,
                    f"Judas Swing (gap-tagged, stop/target/BE exit) — entry {block}",
                    args.max_hold_bars, args.cost_bps, extra_params, result)
            _print_result(block, result, sliced)
            n_hour_slices += 1

    print(f"\n--- Session-close (forced same-day close) trades — reclassified via the price "
          f"{args.session_close_lookback_minutes}min before close ---")
    n_sc_slices = 0
    sc_mask = events["exit_kind"].astype(str).str.startswith("session_close") if "exit_kind" in events.columns else None
    if sc_mask is None or not sc_mask.any():
        print("  (no session-close trades this run — nothing to report)")
    else:
        sc_events = events[sc_mask]
        result = run_event_study(sc_events, cost_bps=args.cost_bps, direction_col="event_type")
        log_run(args.ticker, f"{args.interval}->{args.resample or args.interval}", args.period, args.provider,
                "Judas Swing (gap-tagged, stop/target/BE exit) — session-close forced trades",
                args.max_hold_bars, args.cost_bps, extra_params, result)
        _print_result("SC-ALL", result, sc_events)
        avg_win_r, avg_loss_r, rr = _realized_rr(sc_events)
        rr_str = f"{rr:.2f} : 1" if rr is not None else "n/a (no losers to divide by)"
        print(f"  realized R:R on this subset: avg win = {avg_win_r:.2f}R, avg loss = {avg_loss_r:.2f}R "
              f"-> {rr_str}  (configured risk_reward for a REAL target hit is {args.risk_reward}:1 — "
              f"these never hit it, this is what they actually realized instead)")
        n_sc_slices = 1

    total_slices = 1 + len(_ALL_CODES) + n_slg_slices + n_hour_slices + n_sc_slices
    print(f"\n{total_slices} independent tests logged this run (aggregate + {len(_ALL_CODES)} gap-count "
          f"codes + {n_slg_slices} SLG/non-SLG + {n_hour_slices} entry-hour blocks + {n_sc_slices} "
          f"session-close subset) — see research/.cache/experiment_log.csv for the full, permanent record, "
          f"same file run_gap_study.py logs to (both studies' rows sit side by side, distinguished by "
          f"hypothesis name and the exit_rule/risk_reward/be_trigger_r fields inside each row's "
          f"extra_params). Same caution as before, more so now with more slices: a single slice clearing "
          f"p<0.05 here is exactly what you'd expect by chance alone roughly 1 time in 20, purely from "
          f"running this many slices — treat any one EDGE_FOUND slice as a lead to re-test out-of-sample, "
          f"not a standalone result.")


if __name__ == "__main__":
    main()
