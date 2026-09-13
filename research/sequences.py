"""
Multi-step ICT setups — an alert fires only once every step in a defined
sequence completes IN ORDER, not on any single detector by itself. Built on
top of detectors.py's primitives (liquidity sweep, structure break, FVG/OB
retracement); this is where they get chained into something an ICT trader
would actually call a "setup," not just a list of independent events.

First (and so far only) sequence: the "Judas swing" — the canonical ICT
reversal playbook:
  1. Liquidity sweep — a recent swing high/low gets wicked through (stops
     hunted).
  2. CHoCH in the direction the sweep implies (a HIGH sweep hunts buy-side
     stops and typically precedes a BEARISH reversal, so it's waiting for a
     bearish CHoCH; a LOW sweep waits for bullish) within choch_window_bars
     of the sweep — confirms the reversal is actually underway, not just a
     wick.
  3. Price retraces into a fresh FVG or order block matching that same new
     direction, within retrace_window_bars of the CHoCH. THIS is the alert
     moment — everything upstream already happened, this zone is where a
     trader following the playbook would actually act.

Implemented as a single chronological pass, one pending slot per direction
per stage (latest-wins, consumed exactly once) — NOT "does any sweep in the
last N bars precede this CHoCH," which double-, triple-, quadruple-counts
every later stage against every earlier candidate that happens to fall in
its window. Confirmed directly: the naive any-candidate-in-window version
produced 710 "setups" on 60 days of BTC-USD 15m data — the same single
CHoCH and retracement zone getting reused by every stale prior sweep still
sitting in range, not 710 independent playbook completions.
"""

from detectors import (detect_fvgs, detect_liquidity_levels, detect_liquidity_sweeps, detect_order_blocks,
                        detect_structure_breaks)

# Gap-count -> code, per the trader's own taxonomy: OSG (One Simple Gap),
# TG/TCG (Two Gaps / Two Consecutive Gaps), 3G/3CG (Three / Three
# Consecutive), MG (Multiple, 4+). "Consecutive" here means every gap in the
# leg sits back-to-back with the next (no ordinary candle in between) —
# gaps scattered through the leg with plain candles separating them get the
# non-consecutive code instead.
_GAP_COUNT_CODES = {1: "OSG", 2: ("TG", "TCG"), 3: ("3G", "3CG")}


def _classify_gap_count(gaps):
    """gaps: chronologically-sorted list of {start_pos, end_pos} dicts, all
    of the matching direction, that fell within the impulse leg being
    scored. Returns the code string, or None if the leg had no gaps at all
    (nothing to classify)."""
    n = len(gaps)
    if n == 0:
        return None
    if n >= 4:
        return "MG"
    consecutive = n >= 2 and all(
        gaps[i + 1]["start_pos"] - gaps[i]["end_pos"] <= 1 for i in range(n - 1)
    )
    entry = _GAP_COUNT_CODES[n]
    return entry if n == 1 else (entry[1] if consecutive else entry[0])


def _is_slg(leg_gaps, choch_pos):
    """-SLG (Second Leg Gap), per the trader directly: "slow means no gap,
    no displacement." Leg 1 is the stretch of bars immediately after CHoCH;
    if it produces NO gap (no real displacement) before the leg's first
    gap finally shows up, that first gap didn't come from leg 1 at all —
    it's the start of leg 2, arriving after leg 1 stalled out.

    Operationalized the same way _classify_gap_count already reads
    "consecutive" (adjacent in candle succession, no plain candle breaking
    the streak): if leg_gaps' first gap starts immediately/adjacently
    after the CHoCH bar (within 1 bar — displacement fired right away,
    leg 1 WAS the real leg), this is NOT -SLG. If there's a bigger stretch
    of plain candles between CHoCH and the first gap (leg 1 sat there
    gapless — "slow"), this IS -SLG: the counted gaps came from the leg
    that started after that slow stretch, i.e. leg 2.

    False for a leg with no gaps at all (gap_count_code is already None
    there — nothing to mark as 2nd-leg since nothing was counted)."""
    return bool(leg_gaps) and (leg_gaps[0]["start_pos"] - choch_pos) > 1


def _entry_zone_for_gaps(direction, leg_gaps, high_arr, low_arr):
    """The trader's entry-selection rule, by gap count, stated directly:
      1 gap            -> that gap's own zone (the only candidate).
      2 gaps            -> enter the FIRST gap.
      3 gaps            -> enter the SECOND gap.
      4+ gaps (MG)      -> NOT a specific gap — a Fibonacci retracement
        zone instead, measured from "the beginning of the first gap to
        the end of the last" (the trader's own words): take the actual
        candle extremes across that whole span (not just the gaps'
        own boundaries — the real high/low price action within it), then
        the 0.618-0.79 OTE zone of that range, on the side price would
        pull back to before continuing in the setup's direction (below
        the high for a bullish setup, above the low for a bearish one).
    Returns None if the leg had no gaps at all — nothing to build a zone
    from (see gap_count_code)."""
    n = len(leg_gaps)
    if n == 0:
        return None
    if n <= 2:
        g = leg_gaps[0]
        return {"low": g["bottom"], "high": g["top"], "source": "gap_1"}
    if n == 3:
        g = leg_gaps[1]
        return {"low": g["bottom"], "high": g["top"], "source": "gap_2"}
    # MG: OTE fib zone over [first gap's start bar, last gap's end bar]
    lo_pos, hi_pos = leg_gaps[0]["start_pos"], leg_gaps[-1]["end_pos"]
    span_high = float(high_arr[lo_pos:hi_pos + 1].max())
    span_low = float(low_arr[lo_pos:hi_pos + 1].min())
    rng = span_high - span_low
    if direction == "bullish":
        zone_high, zone_low = span_high - rng * 0.618, span_high - rng * 0.79
    else:
        zone_low, zone_high = span_low + rng * 0.618, span_low + rng * 0.79
    return {"low": zone_low, "high": zone_high, "source": "fib_ote"}


def _first_touch_in_zone(zone, high_arr, low_arr, scan_start, scan_end):
    """First bar position in [scan_start, scan_end] (inclusive, clamped to
    the data) where a candle's range overlaps [zone['low'], zone['high']]
    — same overlap test detect_fvgs/detect_order_blocks use for their own
    first_touch. None if price never trades into the zone within the
    window (the setup this zone belongs to just didn't happen)."""
    n = len(high_arr)
    scan_end = min(scan_end, n - 1)
    for j in range(max(scan_start, 0), scan_end + 1):
        if low_arr[j] <= zone["high"] and high_arr[j] >= zone["low"]:
            return j
    return None


def _scan_start_for_zone(zone, leg_gaps):
    """Earliest bar position price could possibly trade into `zone` — the
    REFERENCE gap's own 3-candle pattern is confirmed formed at
    start_pos + 3 (matching detectors.py's detect_fvgs, which scans for touches
    from i+2 where start_pos == i-1 — see its own comment). NOT the gap's
    end_pos: detectors.py's "end" is when the gap finished getting fully eaten
    (or the last bar of data if it never fills), which is only ever AT OR
    AFTER the first real touch — anchoring the scan there means scanning
    for a touch only after the gap was already touched by definition,
    silently producing zero setups on data that should have several.
    Confirmed directly: this was exactly the bug that zeroed out the
    zigzag end-to-end test's tagged setups the first time this branch was
    written.

    The reference gap is the one that actually DEFINES the zone: the
    single gap for "gap_1"/"gap_2", the LAST gap in the leg for
    "fib_ote" (its own rule — "beginning of first gap to end of last" —
    needs the whole span complete before the zone can be touched at
    all)."""
    if zone["source"] == "fib_ote":
        ref_gap = leg_gaps[-1]
    elif zone["source"] == "gap_2":
        ref_gap = leg_gaps[1]
    else:  # "gap_1"
        ref_gap = leg_gaps[0]
    return ref_gap["start_pos"] + 3


def detect_judas_swing_setups(df, choch_window_bars=20, retrace_window_bars=30, max_scan_bars=None,
                               tag_gap_count=False, choch_window_fn=None):
    """choch_window_fn: None (default, UNCHANGED from before this parameter
    existed) uses the flat choch_window_bars everywhere, exactly as before.
    When given, it's a callable (bar position -> effective choch_window_bars
    for a sweep AT that position) that OVERRIDES choch_window_bars for that
    one sweep->CHoCH pairing check — e.g. research/events.py's
    extract_judas_stoptarget_events passes a schedule that tapers the
    window down as the trading day approaches its own forced close, since
    a sweep with only 20 minutes of trading day left has no realistic use
    for a 60-bar-wide confirmation window. Every other caller (this
    module's own classify_liquidity_tiers, research/events.py's
    extract_judas_swing_events/extract_judas_gap_tagged_events, the live
    agent) never passes this, so they see byte-identical output to before.

    max_scan_bars: passed straight through to the detect_fvgs/
    detect_order_blocks calls below — without it, this recomputes an
    UNBOUNDED fill-scan from scratch even when the standalone FVG/order-
    block hypotheses already computed a capped version moments earlier
    (different max_scan_bars = a different st.cache_data key, so nothing
    gets reused). Confirmed directly: this was the single largest remaining
    cost in a full 7-detector research run (63s of a 126s total on 620k
    rows of GBPUSD 15m) until this call site got the same cap too.

    tag_gap_count=False (default, UNCHANGED from before this parameter
    existed): every existing caller (research/events.py's
    extract_judas_swing_events, the live agent) sees identical output.

    tag_gap_count=True adds four fields to each setup, from the trader's
    own gap-count taxonomy and entry rule, now fully specified:
      "gap_count_code" — classifies the FVGs of matching direction that
        formed between this setup's CHoCH and its retracement entry (the
        Displacement leg) by count and adjacency ("consecutive" = right
        after each other in candle succession, "not" = a plain candle
        breaks the streak — confirmed directly in the trader's own
        words); see _classify_gap_count. None if that leg had no FVGs at
        all (the setup's entry was an order block).
      "slg" — True if Displacement was slow to start after CHoCH (a
        gapless stretch before the first counted gap), meaning the
        counted gaps came from a SECOND leg, not the first — see _is_slg.
        "gap_count_full_code" appends "-SLG" to gap_count_code when this
        is true (e.g. "TCG-SLG"), matching the trader's own code strings.
      "entry_zone" — {"low", "high", "source"} per the trader's own
        entry-by-gap-count rule (1st gap for 2, 2nd gap for 3, a 0.618-
        0.79 Fibonacci OTE zone spanning first-gap-start to last-gap-end
        for MG/4+); see _entry_zone_for_gaps.

    WIRED IN as the actual trigger (not just an annotation): with
    tag_gap_count=True, a setup's "end"/"top"/"bottom" are no longer
    "whichever FVG or order block price happened to retrace into first"
    (that's still exactly what tag_gap_count=False does, unchanged) — they
    ARE entry_zone's own low/high, at the bar price first actually traded
    into that specific zone. Consequences of that switch, deliberately:
      - order-block entries drop out entirely under tag_gap_count=True —
        the trader's rule is about gap counts, so a leg with zero FVGs
        has no defined entry under this rule and produces no setup here,
        even though tag_gap_count=False would have found one via an OB.
      - a CHoCH with gaps that never gets touched at the RULE-mandated
        gap/zone (e.g. price never comes back to the 2nd gap for a 3-gap
        leg, even though it touched the 1st) now produces NO setup for
        that CHoCH — where tag_gap_count=False would have counted the
        1st-gap touch as the setup.
    This means tag_gap_count=True and tag_gap_count=False can legitimately
    return DIFFERENT NUMBERS of setups on the same data, not just the same
    setups with extra fields — by design, since they're now answering two
    different questions ("any retracement happened" vs. "the specific
    rule-mandated entry happened")."""
    idx = df.index
    pos_by_time = {t: i for i, t in enumerate(idx)}

    sweeps = detect_liquidity_sweeps(df)
    chochs = [b for b in detect_structure_breaks(df) if b["structure"] == "CHoCH"]
    fvg_events = detect_fvgs(df, max_scan_bars=max_scan_bars)

    if not tag_gap_count:
        # Exactly the original implementation, byte-for-byte — every
        # existing caller keeps seeing identical output.
        retracements = (
            [{"type": g["type"], "touch": g["first_touch"], "top": g["top"], "bottom": g["bottom"],
              "kind": "fvg", "start": g["start"]}
             for g in fvg_events if g["first_touch"] is not None] +
            [{"type": o["type"], "touch": o["first_touch"], "top": o["top"], "bottom": o["bottom"],
              "kind": "order_block", "start": o["start"]}
             for o in detect_order_blocks(df, max_scan_bars=max_scan_bars) if o["first_touch"] is not None]
        )
        events = (
            [{"stage": "sweep", "pos": pos_by_time[s["end"]], "type": s["type"], "data": s} for s in sweeps] +
            [{"stage": "choch", "pos": pos_by_time[c["end"]], "type": c["type"], "data": c} for c in chochs] +
            [{"stage": "retrace", "pos": pos_by_time[r["touch"]], "type": r["type"], "data": r}
             for r in retracements]
        )
        events.sort(key=lambda e: e["pos"])

        pending_sweep = {"bullish": None, "bearish": None}
        pending_choch = {"bullish": None, "bearish": None}  # (choch_data, choch_pos, sweep_data)
        setups = []

        for e in events:
            d = e["type"]
            if e["stage"] == "sweep":
                pending_sweep[d] = (e["data"], e["pos"])  # latest sweep always wins over a stale one

            elif e["stage"] == "choch":
                if pending_sweep[d] is not None:
                    sweep_data, sweep_pos = pending_sweep[d]
                    window = choch_window_fn(sweep_pos) if choch_window_fn is not None else choch_window_bars
                    if e["pos"] - sweep_pos <= window:
                        pending_choch[d] = (e["data"], e["pos"], sweep_data)
                    pending_sweep[d] = None  # consumed either way — a CHoCH only gets one shot at the pending sweep

            elif e["stage"] == "retrace":
                if pending_choch[d] is not None:
                    choch_data, choch_pos, sweep_data = pending_choch[d]
                    if e["pos"] - choch_pos <= retrace_window_bars:
                        r = e["data"]
                        setups.append({
                            "type": d, "start": sweep_data["start"], "end": r["touch"],
                            "top": r["top"], "bottom": r["bottom"],
                            "sweep_level": sweep_data["level"], "choch_level": choch_data["level"],
                            "retrace_kind": r["kind"],
                        })
                    pending_choch[d] = None  # consumed either way — same one-shot rule as above

        return setups

    # --- tag_gap_count=True: gap-rule-driven entry, wired as the trigger ---
    high_arr = (df["High"] if "High" in df else df["high"]).to_numpy()
    low_arr = (df["Low"] if "Low" in df else df["low"]).to_numpy()

    # raw_top/raw_bottom, NOT top/bottom — detectors.py's "top"/"bottom" is the
    # gap's FINAL, fully-processed remaining-unfilled height (often clamped
    # to near-zero by the time a gap that fills fast finishes its scan; see
    # detectors.py's own comment on this). The trader's entry-by-gap rule means
    # the zone the gap actually occupied when it formed — raw_top/
    # raw_bottom — not however little of it happened to still be
    # unfilled once the whole dataset finished scanning. Confirmed
    # directly: using top/bottom here collapsed every already-filled gap
    # to a single price, so no candle could ever "enter" it and the rule
    # silently produced zero setups on data that should have several.
    fvgs_by_dir = {"bullish": [], "bearish": []}
    for g in fvg_events:
        fvgs_by_dir[g["type"]].append({
            "start_pos": pos_by_time[g["start"]], "end_pos": pos_by_time[g["end"]],
            "top": g["raw_top"], "bottom": g["raw_bottom"],
        })
    for k in fvgs_by_dir:
        fvgs_by_dir[k].sort(key=lambda g: g["start_pos"])

    # Only sweep/CHoCH events feed this stream — the "retrace" stage isn't
    # a generic stream event anymore, it's resolved directly below once a
    # CHoCH confirms, by looking at the whole gap-count rule at once
    # rather than reacting to whichever retracement event streams by first.
    events = (
        [{"stage": "sweep", "pos": pos_by_time[s["end"]], "type": s["type"], "data": s} for s in sweeps] +
        [{"stage": "choch", "pos": pos_by_time[c["end"]], "type": c["type"], "data": c} for c in chochs]
    )
    events.sort(key=lambda e: e["pos"])

    pending_sweep = {"bullish": None, "bearish": None}
    setups = []

    for e in events:
        d = e["type"]
        if e["stage"] == "sweep":
            pending_sweep[d] = (e["data"], e["pos"])
            continue

        # stage == "choch"
        if pending_sweep[d] is None:
            continue
        sweep_data, sweep_pos = pending_sweep[d]
        pending_sweep[d] = None  # consumed either way, same one-shot rule as the base version
        window = choch_window_fn(sweep_pos) if choch_window_fn is not None else choch_window_bars
        if e["pos"] - sweep_pos > window:
            continue
        choch_data, choch_pos = e["data"], e["pos"]
        window_end = choch_pos + retrace_window_bars

        leg_gaps = [g for g in fvgs_by_dir[d] if choch_pos <= g["start_pos"] <= window_end]
        gap_count_code = _classify_gap_count(leg_gaps)
        zone = _entry_zone_for_gaps(d, leg_gaps, high_arr, low_arr)
        if zone is None:
            continue  # no gaps in the leg at all -> the trader's rule doesn't define an entry here

        scan_start = _scan_start_for_zone(zone, leg_gaps)
        entry_pos = _first_touch_in_zone(zone, high_arr, low_arr, scan_start, window_end)
        if entry_pos is None or entry_pos - choch_pos > retrace_window_bars:
            continue

        slg = _is_slg(leg_gaps, choch_pos)
        setups.append({
            "type": d, "start": sweep_data["start"], "end": idx[entry_pos],
            "top": zone["high"], "bottom": zone["low"],
            "sweep_level": sweep_data["level"], "choch_level": choch_data["level"],
            "retrace_kind": zone["source"],
            "gap_count_code": gap_count_code,
            "slg": slg,
            "gap_count_full_code": f"{gap_count_code}-SLG" if slg and gap_count_code else gap_count_code,
            "entry_zone": zone,
        })

    return setups


def classify_liquidity_tiers(df, choch_window_bars=20, retrace_window_bars=30, max_scan_bars=None):
    """The trader's Major/Minor/Local split, read literally rather than
    approximated. detectors.py's detect_liquidity_sweeps(tier=True) can only see
    market STRUCTURE (was this point still the active swing boundary when
    swept) — it has no way to check the trader's actual definition, which
    depends on MSS + Displacement actually happening, a layer only this
    module (built on top of detectors.py) can see:

      Major — "the last liquidity point that forms an impulsive leg":
        a swept point that is the sweep_level of an ACTUALLY CONFIRMED
        Judas swing setup (real sweep -> CHoCH -> Displacement/Retracement
        chain — see detect_judas_swing_setups), not merely a point nothing
        newer happened to replace yet.
      Minor — swept, but never led to a confirmed setup (no CHoCH
        followed within choch_window_bars, or no Displacement/Retracement
        ever completed the chain within retrace_window_bars) — the level
        got taken without producing the impulsive leg Major requires.
      Local — not swept at all (yet): still a live, untouched swing
        point (the same set detectors.py's detect_liquidity_levels calls
        "external range liquidity") — genuinely too recent to say whether
        it resolves Major or Minor, exactly the trader's own definition
        ("recent, not yet categorized").

    Returns one combined list, chronologically sorted within each group:
    resolved sweeps (major/minor, by their sweep time) followed by
    currently-live local points (by their own formation time, since they
    have no sweep time yet). choch_window_bars/retrace_window_bars/
    max_scan_bars: passed straight through to detect_judas_swing_setups.

    NOT YET VALIDATED against real chart examples — this is a first
    implementation of a definition you gave directly, not something
    checked against your own logged setups yet."""
    sweeps = detect_liquidity_sweeps(df)
    setups = detect_judas_swing_setups(df, choch_window_bars=choch_window_bars,
                                        retrace_window_bars=retrace_window_bars, max_scan_bars=max_scan_bars)
    # Join key: (direction, swept level, the swept point's OWN formation
    # time) — detect_judas_swing_setups carries all three straight through
    # from the sweep it consumed (sweep_data["level"]/["start"]), so this
    # is an exact match, not a fuzzy one.
    confirmed = {(s["type"], s["sweep_level"], s["start"]) for s in setups}

    resolved = [
        {**sw, "tier": "major" if (sw["type"], sw["level"], sw["start"]) in confirmed else "minor"}
        for sw in sweeps
    ]

    # detect_liquidity_levels' own n_above/n_below cap how many of the
    # NEAREST live points it returns (a chart-display concern); pass a
    # large cap here to get every currently-untouched swing point, not
    # just the closest few — Local needs all of them, not a top-N.
    live_highs, live_lows = detect_liquidity_levels(df, n_above=len(df), n_below=len(df))
    local = (
        [{"type": "bearish", "level": h["price"], "start": h["time"], "tier": "local"} for h in live_highs] +
        [{"type": "bullish", "level": l["price"], "start": l["time"], "tier": "local"} for l in live_lows]
    )

    return sorted(resolved, key=lambda s: s["end"]) + sorted(local, key=lambda s: s["start"])
