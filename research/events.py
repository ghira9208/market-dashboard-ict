"""
Turns fvg.py's FVG detector into a labeled point-in-time event table for
backtesting — the "Feature Generation" stage of the research pipeline,
reusing the exact same detection logic the live ICT chart draws on screen
rather than re-implementing it (see fvg.py's own DISPLACEMENT_MIN_BODY_RATIO
comment for why this specific definition of a gap was chosen).

Hypothesis under test: does price retracing back into a still-open FVG
predict continuation in the gap's original direction — the core discretionary
read the ICT page's FVG layer is built around (a bullish gap read as support
on retest, a bearish gap read as resistance).

Entry timing deliberately avoids lookahead: a gap's `first_touch` bar is the
bar where price is OBSERVED to have wicked into the zone — that's knowable
only once that bar closes. The simulated entry is the NEXT bar's open, not
the touch bar's own close, since a live system reacting to "price just
touched the gap" couldn't have traded the touch bar itself at its own close.
"""

import numpy as np
import pandas as pd

from fvg import (detect_fvgs, detect_order_blocks, detect_liquidity_reactions,
                  detect_equal_highs_lows, detect_structure_breaks, detect_liquidity_sweeps)
from research.sequences import detect_judas_swing_setups
from research.sessions import KILL_ZONES

# GBPUSD (and every other non-JPY major this pipeline studies so far) quotes
# to 4 decimal places — a pip is that 4th decimal, 0.0001. A JPY pair would
# need 0.01 instead; not handled here since every extractor and CLI default
# in this pipeline is GBPUSD-specific (see EDGE_LAB_TICKER in signals.py).
PIP_SIZE = 0.0001

# "start with London end with NY PM" — read as the continuous NY-time window
# from the London Open kill zone's own start through the NY PM kill zone's
# own end, reusing research/sessions.py's already-established constants
# rather than restating them, so this stays in sync if those ever change.
# This is a plain, continuous [start, end] window, not the discrete,
# gapped kill-zone sub-windows KILL_ZONES itself lists — deliberately: the
# ask was a trading DAY boundary (open in London, closed out by NY PM),
# not "only inside these specific narrow kill-zone minutes." It excludes
# the Asian session (19:00-21:00 ET) by construction, since that's well
# outside 02:00-16:00 ET either way.
_LONDON_OPEN_START_ET = KILL_ZONES["London Open (02:00–05:00 ET)"][0][0]
_NY_PM_END_ET = KILL_ZONES["NY PM (13:30–16:00 ET)"][0][1]

# Bounds detect_fvgs'/detect_order_blocks' fill-scan loop — see their own
# max_scan_bars docstrings for why this exists at all (an unbounded scan on
# a multi-decade backtest dataset is a genuine O(n^2) blowup, not just
# slow). 2000 bars is generous relative to any interval this pipeline
# actually studies (weeks of 15m/1h data, months of daily) — confirmed
# directly on 620k rows of GBPUSD 15m that capping at 2000 bars produced
# the EXACT SAME FVG count as an unbounded scan (106,655 either way), just
# 4x faster (6s vs 24s) — the cap isn't losing real gaps here, just
# skipping wasted scanning past the point where a gap was ever going to
# fill. The live chart's own call sites never pass this, so nothing about
# what's drawn on screen changes.
RESEARCH_MAX_SCAN_BARS = 2000


def extract_fvg_retracement_events(df, forward_bars=10, min_body_ratio=None):
    """df: OHLCV with a DatetimeIndex (as returned by research.data_loader).
    Returns a DataFrame, one row per FVG that was ever retraced into, with
    the forward return from the bar after that retracement. min_body_ratio
    overrides fvg.py's own displacement-strictness default when given —
    a sweepable parameter for the research UI, not a live-chart concern."""
    o_col, h_col, l_col, c_col = ("Open", "High", "Low", "Close")
    close = df[c_col].reset_index(drop=True)
    open_ = df[o_col].reset_index(drop=True)
    idx = df.index
    n = len(df)

    # detect_fvgs returns timestamps (start/end/first_touch), not positions —
    # map back to integer bar positions here since forward-return lookups
    # need "N bars after," a positional concept the timestamps alone don't
    # give directly (and gaps in index around illiquid hours make date-math
    # unsafe as a stand-in for bar count).
    pos_by_time = {t: i for i, t in enumerate(idx)}

    ratio = {} if min_body_ratio is None else {"min_body_ratio": min_body_ratio}
    fvgs = detect_fvgs(df, max_scan_bars=RESEARCH_MAX_SCAN_BARS, **ratio)
    rows = []
    for g in fvgs:
        if g["first_touch"] is None:
            continue
        touch_pos = pos_by_time[g["first_touch"]]
        entry_pos = touch_pos + 1
        exit_pos = entry_pos + forward_bars
        if exit_pos >= n:
            continue  # not enough forward bars left in this dataset to score it

        entry_price = float(open_.iloc[entry_pos])
        exit_price = float(close.iloc[exit_pos])
        raw_ret = (exit_price - entry_price) / entry_price
        direction = 1 if g["type"] == "bullish" else -1

        rows.append({
            "fvg_type": g["type"],
            "gap_formed_at": g["start"],
            "touched_at": g["first_touch"],
            "entry_time": idx[entry_pos],
            "exit_time": idx[exit_pos],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "raw_return": raw_ret,
            "signed_return": raw_ret * direction,
            # raw_top/raw_bottom (the zone AS FORMED), not top/bottom (the
            # zone's FINAL, fully-processed state after fill-scanning — see
            # detect_fvgs' own comment on why that's often near-zero-height
            # by the time a fast-filling gap finishes its scan, and wrong
            # for drawing "the gap" as a trader actually saw it at formation).
            "zone_top": g["raw_top"],
            "zone_bottom": g["raw_bottom"],
        })

    return pd.DataFrame(rows)


def _extract_generic(df, events, forward_bars, time_field="end", type_map=None, extra_fields=None):
    """Shared point-in-time event-study logic for every detector below that
    ISN'T FVG — same entry-timing discipline as extract_fvg_retracement_events
    (enter the bar AFTER the event is confirmed, never the confirming bar's
    own close), just generalized over which field marks "confirmed" (a
    retracement's first_touch for zone-style events, or the event's own
    "end" for point-style events that don't have a separate retracement
    step) and over each detector's own "type" vocabulary. type_map remaps
    values that aren't already "bullish"/"bearish" (e.g. equal_highs_lows'
    "equal_high"/"equal_low") — run_event_study only ever compares against
    those two literal strings. extra_fields: optional list of keys to copy
    straight from each event dict into its own row column, unchanged —
    e.g. detect_judas_swing_setups(tag_gap_count=True)'s "gap_count_code"/
    "slg", so a caller can slice the resulting events_df by those tags
    before handing a slice to run_event_study. None (the default) adds no
    columns — every existing call site is untouched by this parameter."""
    close = df["Close"].reset_index(drop=True)
    open_ = df["Open"].reset_index(drop=True)
    idx = df.index
    n = len(df)
    pos_by_time = {t: i for i, t in enumerate(idx)}

    rows = []
    for e in events:
        trigger = e.get(time_field)
        if trigger is None or trigger not in pos_by_time:
            continue
        entry_pos = pos_by_time[trigger] + 1
        exit_pos = entry_pos + forward_bars
        if exit_pos >= n:
            continue

        entry_price = float(open_.iloc[entry_pos])
        exit_price = float(close.iloc[exit_pos])
        raw_ret = (exit_price - entry_price) / entry_price
        mapped_type = (type_map or {}).get(e["type"], e["type"])
        direction = 1 if mapped_type == "bullish" else -1

        row = {
            "event_type": mapped_type,
            "trigger_time": trigger,
            "entry_time": idx[entry_pos],
            "exit_time": idx[exit_pos],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "raw_return": raw_ret,
            "signed_return": raw_ret * direction,
            # top/bottom for zone-style events (order block, equal H/L,
            # Judas swing); BOS/CHoCH have no zone, just a break "level" —
            # both fall back to that so the example chart always has
            # something to shade, even if it's a flat line, not a box.
            "zone_top": e.get("top", e.get("level")),
            "zone_bottom": e.get("bottom", e.get("level")),
        }
        for key in (extra_fields or []):
            row[key] = e.get(key)
        rows.append(row)
    return pd.DataFrame(rows)


def extract_order_block_events(df, forward_bars=10):
    """Same retracement logic as FVG (wait for price to retrace INTO the
    zone, enter the bar after), just against order-block zones instead."""
    return _extract_generic(df, detect_order_blocks(df, max_scan_bars=RESEARCH_MAX_SCAN_BARS), forward_bars, time_field="first_touch")


def extract_liquidity_reaction_events(df, forward_bars=10):
    """A liquidity-reaction zone is itself an order block (the one that
    formed right after a sweep) — same retracement-entry logic applies."""
    return _extract_generic(df, detect_liquidity_reactions(df, max_scan_bars=RESEARCH_MAX_SCAN_BARS),
                             forward_bars, time_field="first_touch")


def extract_equal_highs_lows_events(df, forward_bars=10):
    """No retracement step here (a cluster isn't a zone price re-enters,
    it's a level) — the event's own "end" (the point its 2nd contributing
    swing confirmed the cluster) is the entry trigger. Equal highs are read
    as resting buy-side liquidity — a magnet for a sweep-then-reversal DOWN,
    so mapped to "bearish"; equal lows mirror this as "bullish"."""
    return _extract_generic(df, detect_equal_highs_lows(df), forward_bars, time_field="end",
                             type_map={"equal_high": "bearish", "equal_low": "bullish"})


def extract_bos_events(df, forward_bars=10):
    breaks = [b for b in detect_structure_breaks(df) if b["structure"] == "BOS"]
    return _extract_generic(df, breaks, forward_bars, time_field="end")


def extract_choch_events(df, forward_bars=10):
    breaks = [b for b in detect_structure_breaks(df) if b["structure"] == "CHoCH"]
    return _extract_generic(df, breaks, forward_bars, time_field="end")


def extract_judas_swing_events(df, forward_bars=10, choch_window_bars=20, retrace_window_bars=30):
    setups = detect_judas_swing_setups(df, choch_window_bars=choch_window_bars, retrace_window_bars=retrace_window_bars,
                                        max_scan_bars=RESEARCH_MAX_SCAN_BARS)
    return _extract_generic(df, setups, forward_bars, time_field="end")


def extract_judas_gap_tagged_events(df, forward_bars=10, choch_window_bars=20, retrace_window_bars=30):
    """Same Judas swing sequence as extract_judas_swing_events, but through
    detect_judas_swing_setups(..., tag_gap_count=True) — entries are now
    the trader's own rule-mandated entry_zone (1st gap for 2, 2nd for 3, a
    Fibonacci OTE zone for 4+), not whichever FVG/order-block price
    retraced into first. See sequences.py's own docstring on that function
    for why this can produce a genuinely different NUMBER of setups than
    extract_judas_swing_events on the same data, not just the same setups
    relabeled.

    The returned DataFrame carries "gap_count_code" ("OSG"/"TG"/"TCG"/"3G"/
    "3CG"/"MG"), "gap_count_full_code" (same, with "-SLG" appended when
    displacement fired from a second leg), and "slg" (bool) as their own
    columns alongside everything _extract_generic already returns — slice
    on any of these BEFORE calling run_event_study to test one gap-count
    code (or the SLG/non-SLG split) in isolation, e.g.:

        events = extract_judas_gap_tagged_events(df)
        tcg_slg = events[events["gap_count_full_code"] == "TCG-SLG"]
        result = run_event_study(tcg_slg, direction_col="event_type")

    Every slice is its own independent statistical test — log each one via
    experiment_log.log_run (see run_gap_study.py), since how many slices
    got tried is exactly what makes any single slice's p-value meaningful,
    the same discipline experiment_log.py's own docstring argues for."""
    setups = detect_judas_swing_setups(df, choch_window_bars=choch_window_bars, retrace_window_bars=retrace_window_bars,
                                        max_scan_bars=RESEARCH_MAX_SCAN_BARS, tag_gap_count=True)
    return _extract_generic(df, setups, forward_bars, time_field="end",
                             extra_fields=["gap_count_code", "gap_count_full_code", "slg"])


def _walk_stop_target(direction, entry_price, stop_price, target_price, be_trigger_price,
                       high_arr, low_arr, scan_start, scan_end):
    """Walks forward bar-by-bar from scan_start to scan_end (inclusive,
    clamped to the data) applying the trader's exact exit rule, stated
    directly: at 1:1 RR, move the stop to breakeven — "consider BE a stop
    at entry point" — and the event ends the moment price interacts with
    the CURRENT stop (the original stop, or breakeven once it's been
    moved) or the original target, whichever comes first.

    Per bar, the CURRENT stop and the target are checked against that
    bar's own high/low BEFORE the breakeven trigger is checked — so a bar
    that both arms breakeven and (on the very same bar) would have hit the
    ORIGINAL stop is impossible to get backwards here: arming only ever
    happens after that bar's stop/target check already came back clean.
    A single bar whose range spans both the current stop and the target
    is called against the trade — stop wins — the same conservative same-
    bar tie-break research/signals.py's own compute_setup already uses
    ("an OHLC bar alone can't say which of TP/SL came first... calling it
    against the trade is the conservative read"), reused here rather than
    picked fresh.

    Returns (exit_pos, exit_price, exit_kind): exit_kind is "win" (target
    hit), "loss" (original stop hit, breakeven never armed), or "be"
    (breakeven stop hit — exit_price == entry_price exactly, zero raw
    return before cost). Returns (None, None, None) if nothing resolves
    within [scan_start, scan_end] — a timeout; the caller drops these, the
    same way every other extractor in this module drops events with no
    room left to score."""
    n = len(high_arr)
    scan_end = min(scan_end, n - 1)
    current_stop = stop_price
    at_be = False
    for i in range(max(scan_start, 0), scan_end + 1):
        hi, lo = float(high_arr[i]), float(low_arr[i])
        if direction == "bullish":
            hit_stop = lo <= current_stop
            hit_target = hi >= target_price
        else:
            hit_stop = hi >= current_stop
            hit_target = lo <= target_price

        if hit_stop:
            return i, current_stop, ("be" if at_be else "loss")
        if hit_target:
            return i, target_price, "win"

        if not at_be:
            reached_be = (hi >= be_trigger_price) if direction == "bullish" else (lo <= be_trigger_price)
            if reached_be:
                current_stop = entry_price
                at_be = True

    return None, None, None


def _session_close_positions(idx, window_start_et, window_end_et, entry_cutoff_before_close_hours=None):
    """Vectorized precompute for the day-boundary rules: for every bar
    position, is it inside [window_start_et, window_end_et] NY time
    (in_window, a bool array — governs the forced same-day close: an
    already-open trade is closed out at/by the LAST in-window bar of its
    own NY-local calendar day, close_pos_by_date below, regardless of the
    entry cutoff); is it ALSO early enough for a NEW entry (entry_ok, a
    bool array — "no trade in the last hour of market close... liquidity
    is thinning": entries are cut off entry_cutoff_before_close_hours
    before window_end_et itself, one-sided narrower than in_window, and
    equal to in_window when entry_cutoff_before_close_hours is None, i.e.
    no separate entry cutoff, the old single-window behavior). The cutoff
    excludes its own boundary minute (minutes < cutoff, not <=) — the
    last hour itself starts right there and is fully off-limits to new
    entries, not just the minutes after it.
    Returns (in_window, entry_ok, close_pos_by_date, ny_date).
    (None, None, None, None) if window_start_et or window_end_et is None
    (session filtering off entirely)."""
    if window_start_et is None or window_end_et is None:
        return None, None, None, None
    idx_ny = idx.tz_convert("America/New_York") if idx.tz is not None else idx.tz_localize("UTC").tz_convert("America/New_York")
    minutes = idx_ny.hour * 60 + idx_ny.minute
    start_min = window_start_et.hour * 60 + window_start_et.minute
    end_min = window_end_et.hour * 60 + window_end_et.minute
    in_window = (minutes >= start_min) & (minutes <= end_min)
    if entry_cutoff_before_close_hours is not None:
        cutoff_min = end_min - int(round(entry_cutoff_before_close_hours * 60))
        entry_ok = (minutes >= start_min) & (minutes < cutoff_min)
    else:
        entry_ok = in_window
    ny_date = idx_ny.date

    close_pos_by_date = {}
    if in_window.any():
        positions = np.arange(len(idx))
        close_pos_by_date = (
            pd.Series(positions[in_window], index=pd.Index(ny_date[in_window]))
            .groupby(level=0).max().to_dict()
        )
    return in_window, entry_ok, close_pos_by_date, ny_date


def _bar_interval_minutes(idx):
    """Median spacing between consecutive bars, in minutes — the
    conversion factor for turning a REAL-TIME duration ("1 hour," "20
    minutes," "5 minutes before close") into the bar count detection
    itself has to work with, since detect_judas_swing_setups and
    _walk_stop_target both operate on bar POSITIONS, not timestamps.
    Median, not mean: robust to the occasional large gap (weekend close,
    a holiday) that would otherwise drag a mean well above the bar
    spacing that actually applies during normal trading hours — exactly
    the kind of skew a 5-minute forex weekend gap would introduce into a
    1-minute dataset's average. Falls back to 1.0 (never divides by
    zero/None) if there are fewer than 2 bars to measure a gap from."""
    if len(idx) < 2:
        return 1.0
    diffs = pd.Series(idx).diff().dropna().dt.total_seconds() / 60.0
    med = diffs.median()
    return float(med) if med and med > 0 else 1.0


def _choch_window_schedule(idx, window_start_et, window_end_et, start_bars, end_bars, taper_before_close_hours):
    """"choch window 60 bars and shortens down to 20 at 1 hour before ny
    close," turned into a callable: bar position -> effective
    choch_window_bars for detect_judas_swing_setups' own sweep->CHoCH
    pairing check (via its choch_window_fn parameter), evaluated at the
    SWEEP's own NY-local time-of-day — the clock that's actually running
    out as the trading day approaches its own forced close (see
    _session_close_positions above).

    start_bars holds flat OUTSIDE today's trading window entirely — before
    window_start_et (the prior evening/overnight — the whole day is still
    ahead) AND after window_end_et (today's close already happened, so
    there's no closer deadline pressing on a sweep that late either) —
    then TAPERS LINEARLY from window_start_et down to end_bars by
    (window_end_et - taper_before_close_hours), holding flat at end_bars
    from that knee through window_end_et itself. Getting the "after
    window_end_et" half of this right matters: hour-of-day alone (e.g.
    "20:00") can't tell a sweep that's already past TODAY's close from one
    inside the taper — both would read as "1200 minutes since midnight" —
    so this checks the actual [window_start_et, window_end_et] band first,
    not just "how late in the day is it." Detection itself still scans
    the full continuous history regardless of any of this — this only
    ever governs how long a CANDIDATE sweep->CHoCH pairing is allowed to
    wait, for a sweep whose own trading day is running out."""
    idx_ny = idx.tz_convert("America/New_York") if idx.tz is not None else idx.tz_localize("UTC").tz_convert("America/New_York")
    minutes = idx_ny.hour * 60 + idx_ny.minute
    start_min = window_start_et.hour * 60 + window_start_et.minute
    end_min = window_end_et.hour * 60 + window_end_et.minute
    taper_start_min = end_min - int(round(taper_before_close_hours * 60))
    span = max(taper_start_min - start_min, 1)

    def fn(pos):
        m = minutes[pos]
        if m < start_min or m > end_min:
            return start_bars  # off-hours entirely -- before today's open, or already past today's close
        if m >= taper_start_min:
            return end_bars
        frac = (m - start_min) / span
        return int(round(start_bars - frac * (start_bars - end_bars)))
    return fn


def extract_judas_stoptarget_events(df, choch_window_minutes=60, choch_window_taper_to_minutes=20,
                                     choch_window_taper_before_close_hours=1.0, retrace_window_bars=30,
                                     risk_reward=2.0, be_trigger_r=1.0, max_hold_bars=RESEARCH_MAX_SCAN_BARS,
                                     min_stop_pips=4.0, pip_size=PIP_SIZE,
                                     window_start_et=_LONDON_OPEN_START_ET, window_end_et=_NY_PM_END_ET,
                                     entry_cutoff_before_close_hours=1.0,
                                     session_close_lookback_minutes=5.0):
    """The trader's ACTUAL exit rule, not a flat bar-count proxy for it —
    the direct answer to "at 1:1 RR set Break Even and the end event when
    price interacts with either the stop or TP; consider BE a stop at
    entry point," now with seven more of the trader's own stated
    constraints: "stop loss minimum 4 pips" (a hard floor — drop the
    trade, don't widen into it), "all trades should end at the before
    close of the markets," "no asian session — start with London end
    with NY PM," "no trade in the last hour of market close, liquidity is
    thinning," "choch window [drops] from 1h to 20m" (a real-TIME taper,
    not a bar count — see choch_window_minutes below), and "the [trade]
    closed at session end, consider the price 5 minutes before [to]
    calculate if it was a win or a loss" (see session_close_lookback_minutes
    below). Every other extractor in this module scores an event off a
    flat forward_bars-later close, because that's the only definition
    simple enough for a first, hard-to-fool-yourself read (see
    backtest.py's own docstring). This extractor is what §11-§15 of the
    Field Notes flagged as the honest next step: the same gap-tagged
    setups, exited on the trader's own actual rule instead.

    Entry price and direction are otherwise identical to
    extract_judas_gap_tagged_events — the trader's own rule-mandated
    entry_zone, entered at the NEXT bar's open after the zone is first
    touched (no lookahead; see this module's own docstring). Detection
    itself still runs over the FULL continuous history, unrestricted —
    same discipline research/sessions.py's own filter_events_to_sessions()
    already documents and follows (trimming raw bars first would corrupt
    FVG/swing continuity); only which setups get ENTERED, when each one
    gets force-closed, and how long a sweep is willing to wait for its
    own confirming CHoCH, are session/time-of-day-aware.

    choch_window_minutes=60/choch_window_taper_to_minutes=20/
    choch_window_taper_before_close_hours=1.0: the sweep->CHoCH pairing
    window (see sequences.py's detect_judas_swing_setups) is no longer
    flat all day — it's 1 HOUR of real time for a sweep early in the
    trading day, tapering LINEARLY down to 20 MINUTES by 1 hour before
    the window closes (15:00 ET under the defaults below), and held at
    20 minutes for that last hour. The reasoning, direct from the
    trader: a sweep with only 20 minutes of trading day left has no
    realistic use for a full 1-hour-wide confirmation window, since even
    a confirmed setup then would barely have time to reach its own entry
    before getting force-closed anyway.

    This is deliberately expressed in MINUTES, not bars, and converted
    to a bar count internally (via _bar_interval_minutes, using this
    df's own actual bar spacing) right before handing it to
    detect_judas_swing_setups — an earlier revision of this taper took a
    raw bar count directly (choch_window_bars=60), which silently meant
    something different depending on what timeframe the caller happened
    to be running: 60 bars is 15 HOURS on a 15-minute chart (nowhere
    near what "60 bars and shortens to 20 at 1 hour before close" was
    ever meant to describe — the trader meant 1 hour, full stop), but
    only 60 MINUTES on a 1-minute chart, by pure coincidence of that
    default number. Expressing it in minutes makes the same real-world
    window apply correctly regardless of --resample: 60 minutes is 4
    bars at 15-minute resolution and 60 bars at 1-minute resolution, and
    this function computes the right one either way rather than the
    caller having to remember to re-derive it by hand. See
    _choch_window_schedule's own docstring for the exact taper shape
    (that helper is still bar-count based internally — a private detail,
    since detection genuinely does operate on bar positions — this
    function is just the minutes->bars conversion layer in front of it).
    Pass choch_window_taper_to_minutes=None to disable tapering entirely
    and use a flat choch_window_minutes all day (the old, single-number
    behavior — pass choch_window_minutes equivalent to 20 bars at
    whatever resolution you're running, to match every other extractor's
    own bar-count default, if you want a true apples-to-apples
    comparison).
    retrace_window_bars is unaffected — still a flat 30 BARS (not
    minutes), unchanged; this is the one window in this function still
    expressed in bar count rather than real time, so its real-world
    duration DOES shift with --resample (7.5 hours at 15-minute
    resolution, 30 minutes at 1-minute resolution) — worth knowing
    before switching resample intervals, since nothing here converts it
    for you the way choch_window_minutes now is.

    Stop and target still aren't reinvented — same formula already
    implemented independently in research_app.py/edge_lab_app.py and
    consolidated in research/signals.py's own compute_setup: stop sits at
    the far edge of the entry zone itself (zone bottom for a bullish
    setup, zone top for a bearish one), target is a fixed reward:risk
    (risk_reward=2.0, the trader's stated 2:1). be_trigger_r=1.0 is the
    1:1 breakeven trigger. min_stop_pips=4.0 (pip_size=0.0001 for GBPUSD)
    is a HARD FLOOR, not a rescue: "don't stretch the stop loss to 4
    points, drop the trade if it's less" — a zone-derived stop narrower
    than this is real spread/slippage territory, essentially untradeable,
    and the setup is DROPPED entirely (n_dropped_sub_floor_stop below),
    never widened into a position size the zone itself never actually
    supported. (Earlier revisions of this extractor widened to the floor
    instead — that behavior is gone; a caller wanting the old widen-
    instead-of-drop read has to reconstruct it externally, it's no longer
    a mode this function offers.) This check only ever applies to a stop
    that's already on the correct side of entry; a zone-derived stop
    that's already wrong-side (see n_dropped_wrong_side below) is dropped
    before the floor is even considered.

    window_start_et/window_end_et ("start with London end with NY PM," no
    Asian session): a continuous NY-time trading window, defaulting to
    London Open's own start (02:00 ET) through NY PM's own end (16:00
    ET) — reusing research/sessions.py's KILL_ZONES constants rather than
    restating them. A setup only counts if its entry bar's own NY-local
    time falls inside that window (checked on entry_time, the same field
    filter_events_to_sessions() already filters on elsewhere in this
    project). And — "all trades should end before close of markets" — a
    setup that's still open when its own NY-local calendar day's window
    closes gets FORCE-CLOSED right there, priced and classified per
    session_close_lookback_minutes below (the new "session_close_win"/
    "session_close_loss"/"session_close_be" exit_kind values), rather
    than carried into the next session. Pass window_start_et=None (or
    window_end_et=None) to disable this entirely and fall back to the
    old always-24h, max_hold_bars-only behavior.

    session_close_lookback_minutes=5.0: "the one closed at session end,
    consider the price 5 minutes before and calculate if it was a win or
    a loss." A forced same-day close no longer just reports whatever the
    exact final in-window bar's Close happens to be and calls it
    "session_close" with an unspecified sign — the price actually used,
    for both the reported exit_price and the win/loss call, is taken
    session_close_lookback_minutes EARLIER than the window-close bar
    (clamped to no earlier than the trade's own entry, converted to a
    bar count the same way choch_window_minutes is, via this df's own
    bar spacing) — the reasoning being the same "liquidity is thinning"
    logic entry_cutoff_before_close_hours is already built on: the very
    last prints right at the close are exactly where spreads widen and a
    single noisy tick is most likely to flip an otherwise-clear win into
    a technical loss (or vice versa) on a trade that was never actually
    live-managed against that tick — nothing hit a real stop or target
    here, it's a forced close, so which side of entry a slightly earlier,
    cleaner print landed on is the more honest read of "was this working
    or not." exit_time still reports the actual close bar (that's still
    literally when the position stops being held); only exit_price (and
    therefore raw_return/signed_return) is taken from the earlier bar.
    The resulting exit_kind is "session_close_win" (that reference price
    beyond entry in the trade's own direction), "session_close_loss"
    (beyond entry against it), or "session_close_be" (exactly at entry —
    the same rare tie condition "be" already handles for a real
    breakeven stop). Counts of each land on .attrs too
    ("n_session_close_win"/"n_session_close_loss"/"n_session_close_be"),
    same reconciliation spirit as the drop reasons below. Pass
    session_close_lookback_minutes=0 to price the forced close at the
    literal final bar instead (the old exact-tick behavior) — the
    win/loss/be reclassification itself still applies either way; there
    is no way back to the old single ambiguous "session_close" bucket,
    only to whether its price is the exact final print or an earlier one.

    entry_cutoff_before_close_hours=1.0: "no trade in the last hour of
    market close — we expect to close all positions, liquidity is
    thinning." A NARROWER cutoff than window_end_et itself, applied only
    to NEW entries — a setup whose entry bar falls inside the trading
    window but within this many hours of window_end_et (the last hour,
    under the default) is dropped and never entered at all
    (n_dropped_last_hour below), same as if it were outside the window
    entirely. This is separate from the forced same-day close above: an
    ALREADY-OPEN trade (entered earlier, before its own last hour) is
    still held and force-closed at window_end_et itself, exactly as
    before — this parameter only ever blocks new entries from starting
    that late, it doesn't pull anything's close time earlier. Tied to the
    trading window the same way tapering is: auto-disabled whenever
    window_start_et/window_end_et are None, since there's no close to cut
    off before. Pass entry_cutoff_before_close_hours=None to keep the
    forced-close rule but drop this specific cutoff (entries allowed
    anywhere in the window, same as before this rule existed).

    A setup is dropped (not scored, not counted as a loss) rather than
    guessed at, for six distinct reasons — each tracked separately on
    the returned DataFrame's .attrs so a caller can report them, not just
    silently lose sample size:
      "n_dropped_no_room"          — no bar left after the entry_zone
                                       touch to even fill an entry (end of
                                       dataset).
      "n_dropped_zero_risk"        — entry price landed exactly on the
                                       stop (zero-width risk) — vanishingly
                                       rare, checked rather than assumed.
      "n_dropped_wrong_side"       — entry price already traded through
                                       the stop's side of the zone by fill
                                       time — already-invalidated at fill,
                                       not something the stop floor can
                                       fix (see min_stop_pips above).
      "n_dropped_sub_floor_stop"   — the zone-derived stop was narrower
                                       than min_stop_pips — real
                                       spread/slippage territory, dropped
                                       rather than widened into a position
                                       the zone itself never supported
                                       (see min_stop_pips above).
      "n_dropped_outside_window"   — the entry bar itself falls outside
                                       window_start_et/window_end_et (or,
                                       with the Asian session specifically,
                                       just isn't inside 02:00-16:00 ET at
                                       all) — never entered in the first
                                       place under the trader's own rule.
      "n_dropped_last_hour"        — the entry bar falls INSIDE the
                                       trading window but within
                                       entry_cutoff_before_close_hours of
                                       window_end_et — never entered, same
                                       reasoning as n_dropped_outside_window,
                                       just a narrower cutoff (see
                                       entry_cutoff_before_close_hours
                                       above).
      "n_dropped_data_ended"       — the trade was still open when the
                                       CACHED DATA ITSELF ran out before
                                       that day's own session-close bar
                                       was reached — genuinely can't know
                                       what the force-close price would
                                       have been, not the same thing as a
                                       real session_close exit.
      "n_dropped_timeout"          — (session filtering off, or
                                       max_hold_bars itself the binding
                                       cap) max_hold_bars ran out before
                                       price interacted with the stop,
                                       target, or a session close —
                                       genuinely unresolved, not a 6th
                                       outcome to invent a return for.
    "n_setups_detected" (the gap-tagged setup count before any of the
    above) is also on .attrs, so "detected - dropped = scored" always
    reconciles for whoever's reading a report off this.

    The returned columns are exactly _extract_generic's shape (event_type,
    trigger_time, entry_time, exit_time, entry_price, exit_price,
    raw_return, signed_return, zone_top, zone_bottom) plus gap_count_code/
    gap_count_full_code/slg plus stop_price/target_price/exit_kind plus
    entry_hour_block (a plain label like "09:00-10:00 ET" — the NY-local
    clock hour entry_time falls in, purely for slicing/reporting; see
    run_stoptarget_study.py's own "by entry hour block" section) — so
    this drops straight into the existing, unmodified run_event_study(
    events, direction_col="event_type") with zero changes to backtest.py.
    exit_kind is one of "win" (target), "loss" (original stop, breakeven
    never armed), "be" (breakeven stop — exit_price == entry_price
    exactly), "session_close_win"/"session_close_loss"/"session_close_be"
    (forced same-day close, classified off the price
    session_close_lookback_minutes before the actual close bar — see
    above; unlike a real win/loss/be these never hit an actual stop or
    target, they're a judgment call about which side of entry the trade
    ended up on when the day forced it shut). zone_top/zone_bottom always
    report the RAW zone edges (for reference); stop_price reports the
    ACTUAL stop used, which always equals the zone edge now that
    min_stop_pips drops sub-floor stops instead of widening them."""
    idx = df.index
    n = len(df)
    bar_minutes = _bar_interval_minutes(idx)

    # Every *_minutes parameter below is real time, converted to THIS
    # dataframe's own bar count once, here -- so the same minutes value
    # means the same wall-clock window regardless of --resample (see
    # choch_window_minutes' own docstring above for why that matters).
    choch_window_bars = max(1, round(choch_window_minutes / bar_minutes))
    choch_taper_to_bars = (None if choch_window_taper_to_minutes is None
                            else max(1, round(choch_window_taper_to_minutes / bar_minutes)))
    session_close_lookback_bars = max(0, round(session_close_lookback_minutes / bar_minutes))

    # Tapering needs a defined day-close to taper TOWARD — if the trading
    # window itself is off (window_start_et/window_end_et both None), there's
    # no forced close for the schedule to be shaped around, so it's off too,
    # regardless of choch_taper_to_bars; no separate flag needed to keep
    # both consistent.
    choch_window_fn = None
    if choch_taper_to_bars is not None and window_start_et is not None and window_end_et is not None:
        choch_window_fn = _choch_window_schedule(
            idx, window_start_et, window_end_et,
            choch_window_bars, choch_taper_to_bars, choch_window_taper_before_close_hours)

    setups = detect_judas_swing_setups(df, choch_window_bars=choch_window_bars, retrace_window_bars=retrace_window_bars,
                                        max_scan_bars=RESEARCH_MAX_SCAN_BARS, tag_gap_count=True,
                                        choch_window_fn=choch_window_fn)
    high_arr = df["High"].to_numpy()
    low_arr = df["Low"].to_numpy()
    close_arr = df["Close"].to_numpy()
    open_ = df["Open"].reset_index(drop=True)
    pos_by_time = {t: i for i, t in enumerate(idx)}

    in_window, entry_ok, close_pos_by_date, ny_date = _session_close_positions(
        idx, window_start_et, window_end_et, entry_cutoff_before_close_hours)
    session_filtering = in_window is not None

    # NY-local clock hour per bar, purely for the entry_hour_block report
    # column below — independent of session_filtering, since it's just a
    # display label, not a filter.
    idx_ny_hour = (idx.tz_convert("America/New_York") if idx.tz is not None
                   else idx.tz_localize("UTC").tz_convert("America/New_York")).hour

    rows = []
    n_no_room = n_zero_risk = n_wrong_side = n_timeout = 0
    n_outside_window = n_data_ended = n_dropped_sub_floor_stop = n_dropped_last_hour = 0
    n_session_close_win = n_session_close_loss = n_session_close_be = 0
    for e in setups:
        trigger = e.get("end")
        if trigger is None or trigger not in pos_by_time:
            continue
        entry_pos = pos_by_time[trigger] + 1
        if entry_pos >= n:
            n_no_room += 1
            continue

        session_close_pos = None
        if session_filtering:
            if not in_window[entry_pos]:
                n_outside_window += 1
                continue
            if not entry_ok[entry_pos]:
                n_dropped_last_hour += 1
                continue
            session_close_pos = close_pos_by_date.get(ny_date[entry_pos])

        entry_price = float(open_.iloc[entry_pos])
        direction = e["type"]
        stop_price = float(e["bottom"]) if direction == "bullish" else float(e["top"])
        risk = abs(entry_price - stop_price)
        if risk <= 0:
            n_zero_risk += 1
            continue
        wrong_side = (stop_price >= entry_price) if direction == "bullish" else (stop_price <= entry_price)
        if wrong_side:
            n_wrong_side += 1
            continue

        floor = min_stop_pips * pip_size
        if risk < floor:
            n_dropped_sub_floor_stop += 1
            continue

        if direction == "bullish":
            target_price = entry_price + risk * risk_reward
            be_trigger_price = entry_price + risk * be_trigger_r
        else:
            target_price = entry_price - risk * risk_reward
            be_trigger_price = entry_price - risk * be_trigger_r

        scan_start = entry_pos + 1
        scan_end = min(entry_pos + max_hold_bars, n - 1)
        if session_close_pos is not None:
            scan_end = min(scan_end, session_close_pos)
        exit_pos, exit_price, exit_kind = _walk_stop_target(
            direction, entry_price, stop_price, target_price, be_trigger_price,
            high_arr, low_arr, scan_start, scan_end)
        if exit_pos is None:
            if session_close_pos is not None and scan_end == session_close_pos:
                # "consider the price 5 minutes before and calculate if it
                # was a win or a loss" -- price the forced close off an
                # earlier, less noisy bar than the literal final print,
                # clamped so it never reaches back before this trade's own
                # entry (a trade that entered less than 5 minutes before
                # its own forced close just uses whatever's available).
                ref_pos = max(scan_start - 1, session_close_pos - session_close_lookback_bars)
                ref_price = float(close_arr[ref_pos])
                exit_pos, exit_price = session_close_pos, ref_price
                if direction == "bullish":
                    beat_entry, missed_entry = ref_price > entry_price, ref_price < entry_price
                else:
                    beat_entry, missed_entry = ref_price < entry_price, ref_price > entry_price
                if beat_entry:
                    exit_kind = "session_close_win"
                    n_session_close_win += 1
                elif missed_entry:
                    exit_kind = "session_close_loss"
                    n_session_close_loss += 1
                else:
                    exit_kind = "session_close_be"
                    n_session_close_be += 1
            elif scan_end == n - 1:
                n_data_ended += 1
                continue
            else:
                n_timeout += 1
                continue

        raw_ret = (exit_price - entry_price) / entry_price
        dir_sign = 1 if direction == "bullish" else -1
        entry_hr = int(idx_ny_hour[entry_pos])
        rows.append({
            "event_type": direction,
            "trigger_time": trigger,
            "entry_time": idx[entry_pos],
            "exit_time": idx[exit_pos],
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "exit_price": exit_price,
            "exit_kind": exit_kind,
            "raw_return": raw_ret,
            "signed_return": raw_ret * dir_sign,
            "zone_top": e.get("top"),
            "zone_bottom": e.get("bottom"),
            "gap_count_code": e.get("gap_count_code"),
            "gap_count_full_code": e.get("gap_count_full_code"),
            "slg": e.get("slg"),
            "entry_hour_block": f"{entry_hr:02d}:00-{(entry_hr + 1) % 24:02d}:00 ET",
        })

    events_df = pd.DataFrame(rows)
    events_df.attrs["n_setups_detected"] = len(setups)
    events_df.attrs["n_dropped_no_room"] = n_no_room
    events_df.attrs["n_dropped_zero_risk"] = n_zero_risk
    events_df.attrs["n_dropped_wrong_side"] = n_wrong_side
    events_df.attrs["n_dropped_sub_floor_stop"] = n_dropped_sub_floor_stop
    events_df.attrs["n_dropped_outside_window"] = n_outside_window
    events_df.attrs["n_dropped_last_hour"] = n_dropped_last_hour
    events_df.attrs["n_dropped_data_ended"] = n_data_ended
    events_df.attrs["n_dropped_timeout"] = n_timeout
    events_df.attrs["n_session_close_win"] = n_session_close_win
    events_df.attrs["n_session_close_loss"] = n_session_close_loss
    events_df.attrs["n_session_close_be"] = n_session_close_be
    events_df.attrs["bar_interval_minutes"] = bar_minutes
    return events_df


# Registry the research UI reads to populate its hypothesis picker without
# needing new UI code per hypothesis — every extractor takes (df,
# forward_bars, **extra_params) and returns a DataFrame with a direction
# column (bullish/bearish) and a "raw_return" column (backtest.
# run_event_study derives everything else from those two). One entry per
# detector live_scan.py's agent exposes — same 7, so every detector you can
# toggle on the live agent has a matching backtest page.
HYPOTHESES = {
    "FVG retracement -> continuation": {
        "extract": extract_fvg_retracement_events,
        "direction_col": "fvg_type",
        "extra_params": [
            # (key, label, min, max, default, step)
            ("min_body_ratio", "Min displacement body ratio", 0.1, 0.9, 0.5, 0.05),
        ],
    },
    "Order Block retracement -> continuation": {
        "extract": extract_order_block_events,
        "direction_col": "event_type",
        "extra_params": [],
    },
    "Liquidity Reaction retracement -> continuation": {
        "extract": extract_liquidity_reaction_events,
        "direction_col": "event_type",
        "extra_params": [],
    },
    "Equal Highs/Lows -> sweep reversal": {
        "extract": extract_equal_highs_lows_events,
        "direction_col": "event_type",
        "extra_params": [],
    },
    "BOS -> continuation": {
        "extract": extract_bos_events,
        "direction_col": "event_type",
        "extra_params": [],
    },
    "CHoCH -> reversal": {
        "extract": extract_choch_events,
        "direction_col": "event_type",
        "extra_params": [],
    },
    "Judas Swing -> reversal": {
        "extract": extract_judas_swing_events,
        "direction_col": "event_type",
        "extra_params": [
            ("choch_window_bars", "CHoCH window (bars after sweep)", 5, 50, 20, 1),
            ("retrace_window_bars", "Retrace window (bars after CHoCH)", 5, 80, 30, 1),
        ],
    },
}
