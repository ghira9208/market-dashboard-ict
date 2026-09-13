"""
Rebuilds the trial-runner that used to live here — its own orchestration
code was never committed and is unrecoverable, but the 638 trials it
already produced (edge_lab/.cache/trials.jsonl) survived, and
research/setups.py's load_edge_lab_validation() already reads that file
correctly today. This script is a fresh implementation of "the thing that
adds new rows," matching that existing schema exactly rather than
inventing a new one, so the reader keeps working unchanged.

What it tries: every hypothesis research/events.HYPOTHESES already knows
how to extract (FVG, Order Block, Liquidity Reaction, Equal Highs/Lows,
BOS, CHoCH, Judas Swing) — the same 7 detectors the live dashboards draw
on screen — at a couple of timeframes and a couple of hold lengths, each
one checked three ways: the whole history, one trading session at a time,
and one weekday at a time. Never all three slicing dimensions stacked
together in one trial (session AND weekday AND timeframe all narrowed at
once) — that would just manufacture more distinct ways for pure luck to
produce something that looks real, which is exactly the over-specification
risk a second-opinion review of this project's own architecture flagged.
Extra per-hypothesis parameters (e.g. FVG's min_body_ratio) stay at their
registered default — this v1 doesn't sweep those too, on the same
keep-the-grid-honest reasoning.

Split rule: every ticker/interval's own currently-available history gets
split 80/20 by TIME (not shuffled — shuffling a time series leaks future
bars into "training"), oldest 80% as train, newest 20% held out untouched.
That mirrors the original tool's own edge_lab/.cache/split_config.json
(train_fraction: 0.8), recomputed fresh against today's data each run
rather than reusing its frozen 2021 cutoff date, which no longer sits at
an actual 80% point now that years more history exist.

Every attempt is logged, pass or fail — nothing here decides "this one's
worth keeping" before writing it down. That decision belongs entirely to
research/setups.py's load_edge_lab_validation() (Benjamini-Hochberg
correction over the FULL cumulative log), read fresh every time something
asks "has this been validated," never cached at write time here.

Ticker is always GBPUSD=X (EDGE_LAB_TICKER) — every existing trial in the
log already assumes this market; mixing another ticker's trials into the
same file/correction batch would silently change what "how many things
were tried" means for every trial already logged.

Run directly: python3 -m edge_lab.agent
"""
import json
import os
import time

import numpy as np
import pandas as pd

from research.data_loader import load_history
from research.evidence import run_event_study
from research.events import HYPOTHESES
from research.sessions import KILL_ZONES, filter_events_to_sessions
from research.setups import EDGE_LAB_TICKER
from research.spread import estimate_spread_bps

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
TRIALS_PATH = os.path.join(_CACHE_DIR, "trials.jsonl")

TICKER = EDGE_LAB_TICKER
INTERVALS = ["1h", "1d"]
FORWARD_BARS_OPTIONS = [10, 20]
WEEKDAYS = [0, 1, 2, 3, 4]  # Monday-Friday
TRAIN_FRACTION = 0.8
# Yahoo's own GBPUSD daily/hourly OHLC has a real, confirmed data-quality
# defect: Close sits at or near Open for most candles from roughly 2004
# onward (median body/range ratio ~0.01, vs ~0.46 on the histdata-sourced
# series below) — this silently starves every displacement-gated detector
# (FVG, Order Block) of real candidates, confirmed directly by resampling
# the SAME calendar window from both sources and comparing. Not a bug in
# detectors.py; a known-to-this-project limitation already solved once
# (histdata_import.py exists specifically because of it) and already
# worked around by run_gap_study.py/run_stoptarget_study.py defaulting to
# --provider histdata — this file follows that same established fix
# rather than the generic auto/Yahoo chain research/data_loader.py's
# other callers use by default.
_RESAMPLE_RULE = {"1h": "1h", "1d": "1D"}
# Fewer than run_event_study's own 5000-permutation default — this is a
# broad screen across ~330 trials, not a final confirmation of one; the
# permutation count only affects how finely the p-value is resolved, not
# whether a trial passes the schema's own bh_significant recompute later.
N_PERMUTATIONS = 2000


def _compute_cutoff(index, train_fraction=TRAIN_FRACTION):
    split_pos = int(len(index) * train_fraction)
    return index[split_pos]


def _weekday_mask(entry_time, weekday):
    times = pd.to_datetime(entry_time)
    if times.dt.tz is None:
        times = times.dt.tz_localize("UTC")
    return times.dt.tz_convert("America/New_York").dt.weekday == weekday


def _score_slice(events_df, direction_col, cutoff, cost_bps):
    """Splits events_df by cutoff and runs run_event_study on each side.
    Returns (train_result, holdout_result) — either can be the
    {"verdict": "NO_DATA", ...} shape when that side has no events at all,
    same honest-empty convention run_event_study already uses."""
    entry_times = pd.to_datetime(events_df["entry_time"])
    if entry_times.dt.tz is None:
        entry_times = entry_times.dt.tz_localize("UTC")
    train = events_df[entry_times < cutoff]
    holdout = events_df[entry_times >= cutoff]
    train_result = run_event_study(train, cost_bps=cost_bps, direction_col=direction_col,
                                    n_permutations=N_PERMUTATIONS, ticker=TICKER)
    holdout_result = run_event_study(holdout, cost_bps=cost_bps, direction_col=direction_col,
                                      n_permutations=N_PERMUTATIONS, ticker=TICKER)
    return train_result, holdout_result


def _build_trial_row(hypothesis, interval, session, weekday, forward_bars, extra_params, cost_bps,
                      train_result, holdout_result):
    family = [hypothesis, session, None, weekday]  # vol_regime always None -- see this file's own docstring
    family_key = "|".join(str(x) if x is not None else "" for x in family)
    label_parts = [hypothesis, interval]
    if session:
        label_parts.append(session)
    if weekday is not None:
        label_parts.append(["Mon", "Tue", "Wed", "Thu", "Fri"][weekday])
    label = " · ".join(label_parts)

    n_train = train_result.get("n_events", 0)
    if n_train == 0:
        return {
            "trial_id": int(time.time() * 1000), "logged_at": pd.Timestamp.now("UTC").isoformat(),
            "family": family, "family_key": family_key, "label": label, "hypothesis": hypothesis,
            "extra_params": extra_params, "interval": interval, "session": session, "vol_regime": None,
            "weekday": weekday, "forward_bars": forward_bars, "cost_bps": cost_bps,
            "n_train": 0, "verdict": "INSUFFICIENT_DATA", "train_significant": False,
        }

    row = {
        "trial_id": int(time.time() * 1000), "logged_at": pd.Timestamp.now("UTC").isoformat(),
        "family": family, "family_key": family_key, "label": label, "hypothesis": hypothesis,
        "extra_params": extra_params, "interval": interval, "session": session, "vol_regime": None,
        "weekday": weekday, "forward_bars": forward_bars, "cost_bps": cost_bps,
        "n_train": n_train,
        "mean_return_train": train_result.get("mean_return_after_cost"),
        "win_rate_train": train_result.get("win_rate_after_cost"),
        "p_value_train": train_result.get("p_value_vs_random_direction"),
        "train_significant": bool(train_result.get("p_value_vs_random_direction") is not None
                                   and train_result["p_value_vs_random_direction"] < 0.05),
    }
    n_holdout = holdout_result.get("n_events", 0)
    if n_holdout > 0:
        row.update({
            "n_holdout": n_holdout,
            "mean_return_holdout": holdout_result.get("mean_return_after_cost"),
            "win_rate_holdout": holdout_result.get("win_rate_after_cost"),
            "p_value_holdout": holdout_result.get("p_value_vs_random_direction"),
            "holdout_verdict": "PASSED" if holdout_result.get("verdict") == "EDGE_FOUND" else "FAILED",
            "verdict": "SCORED",
        })
    else:
        row["verdict"] = "INSUFFICIENT_DATA"
    return row


def _load_base_1m():
    """The one real fetch this whole run needs — histdata.com's full 1-minute
    GBPUSD archive (already imported and cached to disk by
    research/histdata_import.py; ~8.6M bars, 2000-2025). Every interval this
    agent tests is resampled from this SAME series, not fetched separately —
    resampling a cached parquet read is fast; a second raw fetch wouldn't be."""
    return load_history(TICKER, "histdata-full", "1m", provider="histdata")


def run_agent(intervals=INTERVALS, forward_bars_options=FORWARD_BARS_OPTIONS):
    """Runs the full grid described in this module's own docstring and
    appends every trial to TRIALS_PATH. Returns the list of rows written,
    for a caller that wants a summary without re-reading the file."""
    written = []
    os.makedirs(_CACHE_DIR, exist_ok=True)

    base_1m = _load_base_1m()

    for interval in intervals:
        rule = _RESAMPLE_RULE.get(interval)
        if rule is None:
            continue
        df = base_1m.resample(rule).agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        ).dropna()
        if len(df) < 200:
            continue
        cutoff = _compute_cutoff(df.index)
        cost_bps = float(estimate_spread_bps(df))

        for hyp_name, hyp in HYPOTHESES.items():
            extract_fn = hyp["extract"]
            direction_col = hyp["direction_col"]
            extra_params = {key: default for key, _label, _lo, _hi, default, _step in hyp["extra_params"]}

            for forward_bars in forward_bars_options:
                try:
                    events_df = extract_fn(df, forward_bars=forward_bars, **extra_params)
                except Exception:
                    continue
                if events_df.empty:
                    continue

                # Slice 1: whole history, no session/weekday restriction.
                slices = [(None, None, events_df)]
                # Slice 2: one trading session at a time — meaningless at
                # daily granularity (every daily bar shares the same UTC
                # clock time once resampled, so it lands in exactly one NY
                # session band always or never, never a real subset;
                # confirmed directly comparing session-sliced vs
                # unrestricted daily FVG counts, byte-identical). Only
                # tested on genuinely intraday intervals.
                if interval != "1d":
                    for session_name in KILL_ZONES:
                        sliced = filter_events_to_sessions(events_df, [session_name])
                        slices.append((session_name, None, sliced))
                # Slice 3: one weekday at a time.
                for wd in WEEKDAYS:
                    mask = _weekday_mask(events_df["entry_time"], wd)
                    slices.append((None, wd, events_df[mask]))

                for session_name, weekday, sliced_df in slices:
                    if sliced_df.empty:
                        continue
                    train_result, holdout_result = _score_slice(sliced_df, direction_col, cutoff, cost_bps)
                    row = _build_trial_row(hyp_name, interval, session_name, weekday, forward_bars,
                                            extra_params, cost_bps, train_result, holdout_result)
                    written.append(row)

    with open(TRIALS_PATH, "a") as f:
        for row in written:
            f.write(json.dumps(row) + "\n")
    return written


if __name__ == "__main__":
    print(f"Running the grid against {TICKER} — {len(INTERVALS)} interval(s) × {len(HYPOTHESES)} "
          f"hypotheses × {len(FORWARD_BARS_OPTIONS)} hold length(s) × (1 overall + {len(KILL_ZONES)} "
          f"sessions + {len(WEEKDAYS)} weekdays) slices...")
    t0 = time.time()
    rows = run_agent()
    elapsed = time.time() - t0
    scored = [r for r in rows if r["verdict"] == "SCORED"]
    passed = [r for r in scored if r.get("holdout_verdict") == "PASSED"]
    print(f"\nDone in {elapsed:.0f}s — {len(rows)} trials attempted, {len(scored)} had enough data to "
          f"score, {len(passed)} beat their holdout (raw, before correcting for how many were tried).")
    print(f"Appended to {TRIALS_PATH} — run research.setups.load_edge_lab_validation() for the "
          f"corrected (Benjamini-Hochberg) verdict across the full cumulative log.")
