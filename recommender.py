"""
"Best trade right now" for whatever ticker/timeframe the ICT Terminal is
currently showing — the one thing the rest of this project's live-setup
pipeline never actually does. research/setups.py already turns a detected
zone into a concrete entry/stop/target/status box; nothing ranks
candidates against each other or ties any of it to the ticker actually on
screen right now — that's what this module adds, not a new detection
layer or a new entry/stop/target formula.

Presentation-agnostic like research/setups.py itself — app.py owns all
the Streamlit rendering, this module owns only the candidate-building and
ranking, so it stays usable (and testable) without a running Streamlit
process.

Ranking used to be validation-first (Edge Lab's own Benjamini-Hochberg-
corrected pass/fail, GBPUSD=X only) with confluence as a tiebreaker.
Removed per direct request after Edge Lab's own trial-running code
(edge_lab/agent.py and friends) turned out to be missing from the project
entirely — never committed, unrecoverable from git history — leaving the
whole mechanism permanently stuck at "only ever tested GBPUSD=X, and
can't test anything else even if asked." Ranking is confluence-only now:
the more independently-agreeing ICT reads a candidate has, the higher it
ranks — see TF_PAIRS and _weighted_confluence_score below for how a
higher-timeframe agreement counts for more than the same agreement on the
entry timeframe itself, not just a tiebreak on top of something else.
"""

import numpy as np
import pandas as pd

import news
from detectors import (
    current_dealing_range,
    detect_breaker_blocks,
    detect_equal_highs_lows,
    detect_fvgs,
    detect_ifvgs,
    detect_liquidity_levels,
    detect_liquidity_reactions,
    detect_liquidity_sweeps,
    detect_order_blocks,
    detect_structure_breaks,
    detect_swings,
)
from indicators import ema, macd, rsi
from research.setups import compute_setup

EVENT_TYPE_LABELS = {"fvg": "FVG", "order_block": "Order Block", "liquidity_reaction": "Liquidity Reaction",
                      "ma_fvg": "MA+FVG"}

# The MA+FVG strategy's own fast/slow pairing — a standard, widely
# recognized combination, not tuned per-ticker. Kept as a module constant
# rather than a parameter threaded through best_trade_now/scan_watchlist,
# since every caller wants the same two periods; app.py/crypto_app.py
# import this directly too, so the periods drawn on the chart and the
# periods checked for confluence can never silently drift apart.
MA_FVG_PERIODS = (20, 50)


# Higher-timeframe "context" paired with a lower-timeframe "entry" — the
# standard ICT top-down read (establish bias on the bigger picture, place
# the actual trade on a smaller one) rather than treating every timeframe
# as its own independent, disconnected scan. Per direct request/examples:
# 1W context for a 4h entry (a macro-swing read), 4h context for a 5m
# entry (an intraday-scalp read), with 1D-for-1h bridging the gap between
# them — three profiles (swing/day/scalp trader), not an exhaustive
# cross-product of every timeframe against every other one, which would
# mostly just re-test the same bias against itself at slightly different
# zoom levels. Labels match TIMEFRAMES' own keys (app.py/crypto_app.py).
TF_PAIRS = [("1W", "4h"), ("1D", "1h"), ("4h", "5m")]

# How much more a context-timeframe confluence factor counts than the
# same factor on the entry timeframe itself — per direct request ("HTF
# also weights more than LTF confluences"). 2x is a plain, easy-to-explain
# multiplier: a setup confirmed on both timeframes outscores one confirmed
# on the entry timeframe alone by exactly the context-side factors' own
# weight, not some opaque tuned constant.
CONTEXT_WEIGHT = 2


def _equal_level_divergence_inputs(df):
    """The two expensive, df-level (not candidate-level) inputs
    _equal_level_rsi_divergence needs — split out so best_trade_now can
    compute them ONCE per dataframe and hand them to every candidate's
    own _confluence_score call, instead of each call recomputing them
    fresh. Confirmed directly via cProfile as a real, newly-introduced
    cost otherwise: detect_equal_highs_lows is @st.cache_data-decorated,
    but the cache LOOKUP itself hashes the whole dataframe every call —
    cheap once, expensive when _confluence_score (and so this) runs once
    per candidate, times two for a context+entry pair, across every
    ticker in a watchlist scan. 568 calls on a real 10-ticker scan
    measured ~18s of cumulative time inside Streamlit's own cache-hashing
    machinery alone — not the detector, the repeated hashing to check
    it. Returns (clusters, rsi_series); both direction-agnostic, so one
    computation serves every candidate's bullish AND bearish checks."""
    clusters = detect_equal_highs_lows(df)
    close = df["Close"] if "Close" in df else df["close"]
    return clusters, rsi(close)


def _equal_level_rsi_divergence_clusters(direction, clusters, rsi_series):
    """A 5th confluence factor: which Equal High/Low cluster(s) in
    `direction`'s own favor (equal LOWS for a bullish read, equal HIGHS
    for bearish — see detect_equal_highs_lows) show RSI divergence
    between their first and most recent touch? Price revisiting roughly
    the same level while momentum is already fading between the two
    touches is the classic precursor to a liquidity sweep + reversal —
    equal highs/lows are themselves a known resting-liquidity magnet
    (detect_equal_highs_lows' own docstring), and divergence says the
    move drawing price back into that pool is already running out of
    steam. Same "does this OTHER currently-active ICT read agree" spirit
    as the other four factors below, not a new detector. clusters/
    rsi_series: this df's own _equal_level_divergence_inputs() output —
    always passed in (not computed here) so many calls against the SAME
    df, across many candidates and both directions, share one computation
    rather than each paying for its own.

    Returns the list of matching clusters (empty if none) — every
    genuinely divergent cluster, not just the first found, so a caller
    (see _confluence_score) can show exactly which one(s) actually
    backed this factor rather than only a yes/no."""
    kind = "equal_low" if direction == "bullish" else "equal_high"
    matching = [c for c in clusters if c["type"] == kind]
    r = rsi_series
    out = []
    for c in matching:
        if c["start"] not in r.index or c["end"] not in r.index:
            continue
        rsi_start, rsi_end = r.loc[c["start"]], r.loc[c["end"]]
        if pd.isna(rsi_start) or pd.isna(rsi_end):
            continue
        if direction == "bullish" and rsi_end > rsi_start:
            out.append(c)
        elif direction == "bearish" and rsi_end < rsi_start:
            out.append(c)
    return out


# Plain-language description of each confluence factor, in the SAME
# order _confluence_score checks them — lets a caller show WHICH factors
# actually backed one specific trade (the sidebar scan results' own
# per-pick breakdown, see app.py/crypto_app.py), not just the tally.
# Worded the same plain-words-no-jargon way as this module's own
# _CONFLUENCE_HELP text in app.py/crypto_app.py.
_CONFLUENCE_FACTOR_LABELS = [
    "Priced on the discount/premium side that favors this direction",
    "Most recent structure break agrees with this direction",
    "An unmitigated order block backs this direction",
    "Resting liquidity sits in this trade's favor",
    "An equal-high/low pool shows RSI already diverging into it",
]


def _confluence_base_inputs(df):
    """The four df-level (not candidate-level) inputs the ORIGINAL four
    confluence factors need — split out for the exact same reason
    _equal_level_divergence_inputs was. current_dealing_range/
    detect_structure_breaks/detect_order_blocks/detect_liquidity_levels are
    all @st.cache_data-decorated, but the cache LOOKUP itself hashes the
    whole dataframe every call; _confluence_score used to call all four
    fresh on every candidate. Confirmed directly: fixing only the 5th
    (equal-level/RSI) factor this same way barely moved a real 10-ticker
    scan's total time (7.5s -> 6.663s), because these four PRE-EXISTING
    calls were still paying the identical per-candidate hashing cost —
    re-profiling after that first fix showed _confluence_score's own
    cumulative time still dominated by exactly this pattern. best_trade_now
    computes this ONCE per dataframe and hands it to every candidate's own
    _confluence_score call. Returns (dealing_range, structure_breaks,
    order_blocks, (above, below)) — all direction-agnostic, so one
    computation serves every candidate's bullish AND bearish checks."""
    dr = current_dealing_range(df)
    breaks = detect_structure_breaks(df)
    obs = detect_order_blocks(df)
    above, below = detect_liquidity_levels(df)
    return dr, breaks, obs, (above, below)


def _confluence_score(direction, entry_price, df, equal_level_inputs=None, base_inputs=None):
    """0-5 tally of how many OTHER currently-active ICT reads agree with a
    candidate's own direction — reusing the exact detectors app.py's live
    chart already draws (plus one RSI check, see _equal_level_rsi_divergence),
    not a new indicator layer. The sole ranking signal now (see this
    module's own docstring on why Edge Lab's validation-first ranking was
    removed) — used standalone for a single-timeframe candidate, or as
    the two per-timeframe inputs _weighted_confluence_score combines for
    a context+entry pair.

    equal_level_inputs: this df's own _equal_level_divergence_inputs()
    output. base_inputs: this df's own _confluence_base_inputs() output.
    Both precomputed once by the caller (best_trade_now) and reused across
    every candidate — None (compute fresh, here) only for a standalone/
    one-off call outside that loop; every real hot-path call site passes
    both in.

    Returns (score, factor_details) — factor_details is one entry per
    MATCHED factor: {"label": <plain description>, "zones": [...]}, each
    zone a real price/time rectangle ({"kind": "rect", "top", "bottom",
    "start", "end"}) or level ({"kind": "level", "price", "start",
    "end"}) — "end": None means still open/live, draw through to the
    chart's own future edge. This is the ACTUAL geometry that made the
    factor match, not just its name — see app.py/crypto_app.py's "lock
    trade" isolated view, which draws exactly these zones and nothing
    else. score is always len(factor_details); kept as its own return
    value so every existing ranking/filtering call site reading a plain
    int doesn't need to change."""
    details = []

    dr, breaks, obs, (above, below) = base_inputs if base_inputs is not None else _confluence_base_inputs(df)

    if dr is not None:
        if (direction == "bullish" and entry_price < dr["eq"]) or \
           (direction == "bearish" and entry_price > dr["eq"]):
            details.append({
                "label": _CONFLUENCE_FACTOR_LABELS[0],
                "zones": [{"kind": "rect", "top": dr["top"], "bottom": dr["bottom"],
                           "start": dr["start"], "end": None}],
            })

    if breaks and breaks[-1]["type"] == direction:
        b = breaks[-1]
        details.append({
            "label": _CONFLUENCE_FACTOR_LABELS[1],
            "zones": [{"kind": "level", "price": b["level"], "start": b["start"], "end": None}],
        })

    # Most-recent-first, capped at 2 — detect_order_blocks returns EVERY
    # unmitigated OB across the whole df, unbounded; a deep-history
    # context timeframe can easily match 8+ of them, which would just
    # trade one kind of chart clutter (every layer, all zones) for
    # another (every matching OB, all zones) in the isolated "lock
    # trade" view this feeds. Same reasoning liquidity's own detector
    # already applies (nearest 2 above/below) — the most RECENT
    # same-direction OB is what a trader actually means by "an
    # unmitigated order block backs this," not an exhaustive list.
    matching_obs = sorted((o for o in obs if not o["mitigated"] and o["type"] == direction),
                          key=lambda o: o["start"], reverse=True)[:2]
    if matching_obs:
        details.append({
            "label": _CONFLUENCE_FACTOR_LABELS[2],
            "zones": [{"kind": "rect", "top": o["top"], "bottom": o["bottom"],
                       "start": o["start"], "end": None} for o in matching_obs],
        })

    matching_liq = above if direction == "bullish" else below
    if (direction == "bullish" and above) or (direction == "bearish" and below):
        details.append({
            "label": _CONFLUENCE_FACTOR_LABELS[3],
            "zones": [{"kind": "level", "price": lvl["price"], "start": lvl["time"], "end": None}
                      for lvl in matching_liq],
        })

    clusters, rsi_series = equal_level_inputs if equal_level_inputs is not None else _equal_level_divergence_inputs(df)
    # Same most-recent-first, capped-at-2 reasoning as the order-block
    # branch above — a long-history df can show several divergent
    # clusters at once.
    matching_clusters = sorted(_equal_level_rsi_divergence_clusters(direction, clusters, rsi_series),
                               key=lambda c: c["end"], reverse=True)[:2]
    if matching_clusters:
        details.append({
            "label": _CONFLUENCE_FACTOR_LABELS[4],
            "zones": [{"kind": "rect", "top": c["top"], "bottom": c["bottom"],
                       "start": c["start"], "end": c["end"]} for c in matching_clusters],
        })

    return len(details), details


def _weighted_confluence_score(direction, entry_price, entry_df, context_df, context_weight=CONTEXT_WEIGHT,
                                entry_equal_level_inputs=None, context_equal_level_inputs=None,
                                entry_base_inputs=None, context_base_inputs=None):
    """_confluence_score run on BOTH timeframes of a context+entry pair,
    the context side counting `context_weight`x — checked with the SAME
    entry_price against BOTH dataframes on purpose: "is this exact entry
    still inside the bigger picture's own discount half / does the bigger
    picture's own last structure break agree" is exactly the top-down
    question a context timeframe is FOR, not a mismatched comparison.
    0-5 entry-side + 0-5*context_weight context-side — no fixed maximum
    quoted anywhere downstream (a plain "confluence N" display, not a
    fraction), since the max shifts if context_weight or the factor count
    ever does.

    entry_equal_level_inputs/context_equal_level_inputs: see
    _confluence_score's own equal_level_inputs param. entry_base_inputs/
    context_base_inputs: see its base_inputs param. best_trade_now
    precomputes all of these once per df and passes them straight through.

    Returns (total_score, entry_details, context_details) — see
    _confluence_score's own return value for what the detail lists are."""
    entry_score, entry_details = _confluence_score(direction, entry_price, entry_df,
                                                     equal_level_inputs=entry_equal_level_inputs,
                                                     base_inputs=entry_base_inputs)
    context_score, context_details = _confluence_score(direction, entry_price, context_df,
                                                         equal_level_inputs=context_equal_level_inputs,
                                                         base_inputs=context_base_inputs)
    return entry_score + context_score * context_weight, entry_details, context_details


def _zone_is_open(zone):
    """Whether `zone` (FVG or Order Block shape) is still a genuinely live,
    resolvable candidate: fully unfilled/unmitigated AND, when a
    max_scan_bars cap was in effect on the detect_fvgs/detect_order_blocks
    call that produced it, still within its own fill-tracking window. A
    zone whose fill-scan hit that cap without price ever closing it isn't
    "still open" — it's a setup this particular scan gave up watching, and
    treating it as a valid candidate forever afterward was the actual
    bug reported directly against the backtest engine: a zone whose real
    fill would've landed one bar past max_scan_bars stayed pickable for
    the rest of a walk-forward run, long after any real trader would have
    considered the setup dead. "expired" is absent (not just False) on
    any zone detected with max_scan_bars=None (the live chart's own call
    sites) — .get(..., False) treats that exactly as "never expires",
    zero behavior change there."""
    filled_key = "filled" if "filled" in zone else "mitigated"
    return not zone[filled_key] and not zone.get("expired", False)


def _candidates(df):
    """Every currently-open/unmitigated zone from the three detectors whose
    output shape (type/top/bottom/start/end) already matches what
    compute_setup's `row` needs — see this module's own docstring for why
    BOS/CHoCH/Equal Highs-Lows (flat-level events, no real zone to place a
    stop against) are left for a later pass instead of forced in here."""
    out = []
    for g in detect_fvgs(df):
        if _zone_is_open(g):
            out.append(("fvg", g))
    for o in detect_order_blocks(df):
        if _zone_is_open(o):
            out.append(("order_block", o))
    for r in detect_liquidity_reactions(df):
        if not r["mitigated"]:
            out.append(("liquidity_reaction", r))
    return out


def _ma_fvg_candidates(df):
    """Buy/sell where a moving average overlaps a currently-open FVG — the
    MA reads as dynamic support/resistance, the FVG as an imbalance likely
    to get defended; when both agree, that's the entry. MA_FVG_PERIODS'
    EMAs, checked independently against every open FVG's own [bottom,
    top] range using each EMA's CURRENT (last-bar) value — "is this live
    right now," same spirit as everything else in this module. Emits the
    same zone shape _candidates() does (type/top/bottom/start/end), so it
    flows through the existing compute_setup call unchanged; the only
    addition is `ma_periods`, tracking which EMA(s) actually matched,
    carried through onto the resulting setup dict for on-chart/sidebar
    display (see app.py's use of it to bold the FVG's own border when
    this fired)."""
    close = df["Close"] if "Close" in df else df["close"]
    if len(close) < 2:
        return []
    emas = {period: ema(close, period) for period in MA_FVG_PERIODS}
    out = []
    for g in detect_fvgs(df):
        if not _zone_is_open(g):
            continue
        matched = [period for period, series in emas.items()
                   if pd.notna(series.iloc[-1]) and g["bottom"] <= series.iloc[-1] <= g["top"]]
        if matched:
            out.append(("ma_fvg", {**g, "ma_periods": sorted(matched)}))
    return out


def ma_fvg_starts(df):
    """The `start` timestamps of every FVG _ma_fvg_candidates currently
    matches — a plain lookup for on-chart border highlighting (app.py/
    crypto_app.py), deliberately NOT sourced from best_trade_now's own
    ranked/truncated output. Confirmed directly: on a data-rich ticker
    (e.g. ^GSPC at 1h, 1467 bars) a genuine live MA+FVG overlap can rank
    below best_trade_now's top_n once enough unrelated open FVGs/OBs/
    liquidity reactions are competing for the same slots, silently
    dropping the highlight for a real match. The border isn't a
    recommendation, just "is an EMA sitting inside this zone right now" —
    it must not depend on surviving a ranking cutoff meant for a
    different purpose."""
    return {zone["start"] for _, zone in _ma_fvg_candidates(df)}


def _pattern_zones(df, event_type, direction):
    """Every PAST occurrence (open or long since filled — this is a
    historical count, not a live-candidate one) of `event_type` on `df`,
    filtered to `direction`. Shared helper for pattern_win_rate; the
    same four detectors _candidates()/_ma_fvg_candidates() already draw
    from, so "this many past FVGs" here means the exact same thing it
    means everywhere else in this module.

    order_block/liquidity_reaction specifically ask for record_history=True
    here (NOT how _candidates()'s own live detection calls these same
    detectors — a separate cache entry, zero effect on that path) so each
    zone carries _formation_top/_formation_bottom — see pattern_win_rate's
    own use of _zone_formation_bounds for why a HISTORICAL win-rate can't
    use these detectors' default, fully-encroached top/bottom the way a
    LIVE candidate correctly does."""
    if event_type == "fvg":
        return [g for g in detect_fvgs(df) if g["type"] == direction]
    if event_type == "order_block":
        return [o for o in detect_order_blocks(df, record_history=True) if o["type"] == direction]
    if event_type == "liquidity_reaction":
        return [r for r in detect_liquidity_reactions(df, record_history=True) if r["type"] == direction]
    if event_type == "ma_fvg":
        return [z for t, z in _ma_fvg_candidates(df) if t == "ma_fvg" and z["type"] == direction]
    return []


def _zone_formation_bounds(zone):
    """The zone's own boundaries AS FORMED, before any later consequent
    encroachment — raw_top/raw_bottom for FVG (and ma_fvg, which inherits
    them via its own {**g, ...} spread from a FVG zone), _formation_top/
    _formation_bottom for order_block/liquidity_reaction (only present
    when detected with record_history=True — see _pattern_zones above).
    Falls back to the zone's plain top/bottom for any zone shape that
    truly has neither (defensive; every event_type _pattern_zones can
    produce provides one or the other in practice).

    Why this matters, confirmed directly as a real bug otherwise: `df` in
    pattern_win_rate spans the FULL available history, so a zone's own
    top/bottom (as detect_fvgs/detect_order_blocks compute them) reflect
    EVERY touch that ever happened to it, all the way up to the dataset's
    last row — not what it looked like back when history first_touch
    happened. For an old zone, that's often encroached all the way down
    to a sliver (top==bottom) by "today," even though it was a full-size,
    genuinely tradeable zone at first_touch time. Entry/stop computed off
    that shrunk-by-hindsight size makes risk artificially tiny, and with
    an unbounded forward scan (years of bars) even pure noise eventually
    moves 2x a near-zero risk — every pattern tested came back at a
    suspicious flat 100% hit rate this way, in both directions, which a
    real edge (or lack of one) never would. Formation-time bounds are
    what a trader actually saw when the trade would have triggered."""
    top = zone.get("raw_top", zone.get("_formation_top", zone["top"]))
    bottom = zone.get("raw_bottom", zone.get("_formation_bottom", zone["bottom"]))
    return top, bottom


def pattern_win_rate(df, event_type, direction, min_events=20, news_blackout_mask=None):
    """Historical probability THIS pattern's own entry — snapped to the
    zone's own edge, opposite edge stop, fixed 2R target, the EXACT same
    formula best_trade_now uses for a live pick (see its own docstring)
    — reaches its target before its stop, measured across every PAST
    occurrence of this event_type+direction on `df`. Direct request:
    "aggregate a probability for trades to reach target/stop."

    A different question from app.py's own _historical_win_rates: that
    one asks "did price move favorably before the NEXT pattern formed
    (any pattern, fixed hold, no real stop)" — this one asks "did THIS
    EXACT trade (real entry/sl/tp, as best_trade_now would size it right
    now) actually hit its own target first, historically." Scanned
    forward from each zone's own first_touch — best_trade_now's own live
    "ideal entry" premise is "wait for price to retrace back to the
    zone's own edge," so the historical equivalent has to wait for that
    same retrace, not just count from formation. A zone that never got
    touched again (first_touch is None) never actually triggers a trade,
    so it's excluded here entirely — same convention fvg_event_win_rate
    already uses elsewhere in this file. (An earlier version of this
    function scanned from formation instead, on the reasoning that
    best_trade_now's own premise is "assume the retrace happens" rather
    than "wait for it" — confirmed directly as wrong: every single
    pattern/ticker/direction tested came back at a suspicious flat 100%
    hit rate, because the formation candle's own confirmation bar is
    already riding the displacement's own momentum toward the target,
    making it trivially easy to hit almost immediately. first_touch is
    the point a live trader's resting order would actually fill.)

    Entry/stop use each zone's own FORMATION-time bounds (see
    _zone_formation_bounds), not its final top/bottom — a second,
    separate bug found alongside the first_touch one above: `df` spans
    the FULL available history, so a zone's plain top/bottom reflect
    every touch that ever happened to it up to the dataset's LAST row,
    not what it looked like at first_touch time. An old zone is often
    encroached down to a sliver by "today" even though it was full-size
    when first touched — entry/stop off that shrunk-by-hindsight size
    made risk artificially tiny, and over an unbounded forward scan even
    pure noise eventually moves 2x a near-zero risk. This was the actual
    cause of the still-100%-everywhere result the first_touch fix alone
    didn't resolve.

    news_blackout_mask: None (default, unchanged behavior) or a boolean
    array aligned to df.index (see news.blackout_mask) — a zone whose own
    execution bar (exec_pos) falls inside a blackout window is excluded
    from the count entirely, same convention _score_events uses for the
    event-driven win rates elsewhere in this module.

    Returns (rate, n, sufficient) — rate is None when n == 0 (nothing
    ever resolved either way within the available history, e.g. a
    ticker with almost no data yet)."""
    zones = _pattern_zones(df, event_type, direction)
    if not zones:
        return None, 0, False

    idx = df.index
    pos_by_time = {t: i for i, t in enumerate(idx)}
    high = (df["High"] if "High" in df else df["high"]).to_numpy()
    low = (df["Low"] if "Low" in df else df["low"]).to_numpy()
    n = len(df)

    wins = total = 0
    for zone in zones:
        entry_pos = pos_by_time.get(zone.get("first_touch"))
        if entry_pos is None:
            continue  # never got touched again -- this trade never actually triggers
        # +1, not the touch bar itself: a live system reacting to "price
        # just touched the zone" couldn't have traded the touch bar's own
        # close — same no-lookahead convention _score_events already uses
        # for fvg_event_win_rate's own entries (exec_pos = entry_pos + 1).
        exec_pos = entry_pos + 1
        if exec_pos >= n:
            continue
        if news_blackout_mask is not None and news_blackout_mask[exec_pos]:
            continue
        form_top, form_bottom = _zone_formation_bounds(zone)
        entry = form_top if direction == "bullish" else form_bottom
        sl = form_bottom if direction == "bullish" else form_top
        risk = abs(entry - sl)
        if risk <= 0:
            continue
        tp = entry + risk * 2 if direction == "bullish" else entry - risk * 2

        fwd_high, fwd_low = high[exec_pos:], low[exec_pos:]
        if direction == "bullish":
            hit_tp, hit_sl = fwd_high >= tp, fwd_low <= sl
        else:
            hit_tp, hit_sl = fwd_low <= tp, fwd_high >= sl
        tp_i = int(hit_tp.argmax()) if hit_tp.any() else None
        sl_i = int(hit_sl.argmax()) if hit_sl.any() else None
        if tp_i is None and sl_i is None:
            continue  # never resolved either way within available history

        total += 1
        # SL wins a same-bar tie — same conservative rule compute_setup's
        # own historical resolution already uses elsewhere in this project.
        if tp_i is not None and (sl_i is None or tp_i < sl_i):
            wins += 1

    if total == 0:
        return None, 0, False
    return wins / total, total, total >= min_events


def best_trade_now(df, ticker, interval, provider, top_n=3, event_types=None, direction=None, min_confluence=0,
                    context_df=None, news_blackout=None):
    """The top `top_n` currently-active candidates for `ticker` at `interval`
    (already-fetched `df`, e.g. app.py's own main chart data — this does no
    fetching of its own), ranked by confluence (closest-to-entry breaks
    ties) descending. `interval` is a plain fetch-interval string ("1d",
    "60m", ...) — matched against research.setups.LOOKBACK for how long a
    setup stays "live"; an interval that dict doesn't recognize just falls
    back to its own generic default, not an error. Returns [] when nothing
    currently active resolves — an honest empty result, not a forced pick.

    event_types/direction/min_confluence: optional pre-ranking filters —
    "only consider FVG+MA-FVG candidates," "bullish only," "confluence >=
    2" — for a user who wants to steer WHICH pattern the pick comes from
    rather than just seeing whatever the ranking alone would surface.
    None/0 (the defaults) mean unfiltered.

    context_df: optional — a HIGHER timeframe's own already-fetched df
    (see TF_PAIRS) for a top-down "does the bigger picture agree" read.
    When given, ranking uses _weighted_confluence_score (the context
    side counts CONTEXT_WEIGHT x) instead of `df`'s own confluence alone
    — a candidate confirmed on both timeframes outranks one confirmed on
    just the entry timeframe, which outranks one with no confluence at
    all. None (the default) preserves the original single-timeframe
    behavior exactly.

    news_blackout: None (default, unchanged behavior) or a
    (minutes_before, minutes_after) pair - pattern_win_rate's own hit-rate
    stat then excludes any historical trade that would have opened inside
    a high-impact news window for `ticker`'s own relevant currencies (see
    news.blackout_mask). Computed once here since every candidate below
    shares the same df/ticker."""
    _news_blackout_mask = (news.blackout_mask(df.index, ticker, news_blackout[0], news_blackout[1])
                            if news_blackout else None)
    # Computed ONCE per df here, not once per candidate inside the loop
    # below (a genuine, measured perf regression otherwise — see
    # _equal_level_divergence_inputs' and _confluence_base_inputs' own
    # docstrings: Streamlit's own cache-key hashing, not the detection
    # itself, dominates when these run once per candidate).
    _entry_equal_inputs = _equal_level_divergence_inputs(df)
    _context_equal_inputs = _equal_level_divergence_inputs(context_df) if context_df is not None else None
    _entry_base_inputs = _confluence_base_inputs(df)
    _context_base_inputs = _confluence_base_inputs(context_df) if context_df is not None else None
    _win_rate_cache = {}
    ranked = []
    for event_type, zone in _candidates(df) + _ma_fvg_candidates(df):
        if event_types is not None and event_type not in event_types:
            continue
        if direction is not None and zone["type"] != direction:
            continue
        row = {
            "ticker": ticker, "interval": interval, "event_type": event_type,
            "direction": zone["type"], "start": zone["start"], "end": zone["end"],
            "top": zone["top"], "bottom": zone["bottom"],
        }
        if "ma_periods" in zone:
            row["ma_periods"] = zone["ma_periods"]
        setup = compute_setup(row, df)
        if setup is None or setup["status"] != "active":
            continue
        # Snap entry to the zone's own NEAR edge — the classic ICT "resting
        # limit order into the gap/block" convention — instead of
        # compute_setup's own entry (the close of whichever bar its `end`
        # falls on, which for every still-OPEN candidate here is always the
        # dataframe's own last bar, i.e. current price; see that function's
        # own docstring). Direct request: "I want it to snap, to show me
        # the ideal [entry]." compute_setup's stop already sits at the
        # zone's OPPOSITE edge (bottom for bullish, top for bearish) — using
        # the near edge as entry makes risk exactly the zone's own height,
        # not "however far price currently happens to be from the edge."
        # Both bullish (top=entry > bottom=sl) and bearish (bottom=entry <
        # top=sl) hold by construction for any genuinely open zone (top >
        # bottom), which is the only kind _candidates()/_ma_fvg_candidates()
        # ever emit — kept behind the ordering/positivity guards below
        # anyway, as a safety net, not because either is expected to fire.
        setup["entry_price"] = zone["top"] if setup["direction"] == "bullish" else zone["bottom"]
        _risk = abs(setup["entry_price"] - setup["sl_price"])
        if _risk <= 0:
            continue
        setup["tp_price"] = (setup["entry_price"] + _risk * 2 if setup["direction"] == "bullish"
                              else setup["entry_price"] - _risk * 2)
        # risk/reward_risk/distance_pct all quoted compute_setup's OLD
        # entry — recompute against the snapped one so the R:R and
        # "how far price still has to move to reach entry" both stay
        # honest (distance_pct also feeds the ranking tiebreak's own
        # -abs(distance_pct), closest-to-live-first).
        setup["risk"] = _risk
        setup["reward_risk"] = abs(setup["tp_price"] - setup["entry_price"]) / _risk
        setup["distance_pct"] = (setup["current_price"] - setup["entry_price"]) / setup["entry_price"] * 100
        # compute_setup's own stop formula (sl_price = the zone's own
        # bottom/top) quietly assumes the zone actually BRACKETS current
        # price — true for a genuinely live zone, but a currently-open FVG
        # left over from far earlier in a long-history fetch (price has
        # since moved thousands of points away, the gap just never got
        # revisited to fill it) can still pass every other check and still
        # rank #1 on confluence alone. When that happens the stop ends up
        # on the SAME side as the target instead of the opposite one —
        # confirmed directly: a stale bearish FVG on ^IXIC's 180d/1h
        # history produced sl=22480/tp=18723 against an entry of 26236,
        # both BELOW entry, backwards for a short. A real trade always has
        # the stop and target on opposite sides of entry; anything that
        # doesn't is a broken setup born from a stale zone, not a live
        # recommendation, and gets dropped here rather than shown.
        if setup["direction"] == "bullish":
            if not (setup["sl_price"] < setup["entry_price"] < setup["tp_price"]):
                continue
        else:
            if not (setup["tp_price"] < setup["entry_price"] < setup["sl_price"]):
                continue
        # Same stale-zone family as the ordering check above, one level
        # further: an unusually wide zone (risk far larger than entry
        # itself — confirmed directly scanning multiple timeframes at
        # once, which surfaces coarser TFs' own wider zones far more
        # often than a single-timeframe call ever used to) can pass the
        # ordering check while still landing sl_price or tp_price at or
        # below zero — a real instrument's price never does that, so a
        # target of -10830 is exactly as broken a recommendation as one
        # with sl/tp on the wrong side, just a different way to get there.
        if setup["sl_price"] <= 0 or setup["tp_price"] <= 0:
            continue
        # confluence_entry_details/confluence_context_details: the actual
        # zones/levels behind WHICH factors backed this specific trade —
        # not just the tally already in confluence_score. See
        # app.py/crypto_app.py's own per-result breakdown popover and
        # "lock trade" isolated chart view.
        if context_df is not None:
            _total, _entry_details, _context_details = _weighted_confluence_score(
                setup["direction"], setup["entry_price"], df, context_df,
                entry_equal_level_inputs=_entry_equal_inputs, context_equal_level_inputs=_context_equal_inputs,
                entry_base_inputs=_entry_base_inputs, context_base_inputs=_context_base_inputs)
            setup["confluence_score"] = _total
            setup["confluence_entry_details"] = _entry_details
            setup["confluence_context_details"] = _context_details
        else:
            _score, _details = _confluence_score(setup["direction"], setup["entry_price"], df,
                                                  equal_level_inputs=_entry_equal_inputs,
                                                  base_inputs=_entry_base_inputs)
            setup["confluence_score"] = _score
            setup["confluence_entry_details"] = _details
            setup["confluence_context_details"] = []
        if setup["confluence_score"] < min_confluence:
            continue
        # The trigger zone itself — what this candidate actually IS, not
        # just what backs it (see confluence_entry_details above). Every
        # candidate here came from _candidates()/_ma_fvg_candidates(),
        # which only ever emit currently-OPEN zones, so "end": None
        # (still live, draw through to the chart's own future edge) is
        # always correct here, not a per-zone check.
        setup["source_zone"] = {"kind": "rect", "top": zone["top"], "bottom": zone["bottom"],
                                 "start": zone["start"], "end": None}
        # Direct request: "aggregate a probability for trades to reach
        # target/stop" — historical P(this exact entry/sl/tp formula
        # hits target before stop), see pattern_win_rate's own docstring.
        # Cached per (event_type, direction) within this call, not
        # recomputed per candidate — several candidates sharing the same
        # zone type + direction (e.g. two open bullish FVGs) would
        # otherwise re-scan the same history twice for an identical
        # answer, the same "compute once per df" reasoning as the
        # confluence inputs above.
        _wr_key = (event_type, setup["direction"])
        if _wr_key not in _win_rate_cache:
            _win_rate_cache[_wr_key] = pattern_win_rate(df, event_type, setup["direction"],
                                                         news_blackout_mask=_news_blackout_mask)
        _wr_rate, _wr_n, _wr_sufficient = _win_rate_cache[_wr_key]
        setup["hit_rate"] = _wr_rate
        setup["hit_rate_n"] = _wr_n
        setup["hit_rate_sufficient"] = _wr_sufficient
        ranked.append(setup)

    ranked.sort(key=lambda s: (s["confluence_score"], -abs(s["distance_pct"])), reverse=True)
    return ranked[:top_n]


# ---------------------------------------------------------------------------
# "Where is price more likely to head next" — direct request: rank every
# currently-open level (FVG/IFVG/Order Block/Breaker Block/resting
# liquidity) by an actual historical PROBABILITY OF BEING VISITED, not raw
# proximity. This is a genuinely different question from best_trade_now's
# own hit_rate: that one asks "if this exact trade were taken, does it
# reach target before stop" — this one asks "does price even reach this
# level AT ALL," with no trade construction involved. Deliberately its own
# statistic (_level_touch_rates below), not a repurposing of pattern_win_
# rate/fvg_event_win_rate, which both already assume a trade was entered
# and score its outcome — a level nobody entered a trade at can still be
# "visited" or not, and that's the only thing being measured here.
#
# The core idea: how far a level sat from price WHEN IT FORMED predicts,
# historically, how likely it is to get touched at all — a level that
# formed 0.2% from price is a near-certain eventual touch; one that formed
# 8% away may never get revisited. Bucketing by that formation-time
# distance and asking "of all levels like this one, what fraction
# eventually got touched" turns raw proximity into an actual, ticker-
# specific probability instead of an assumption that "closer = more
# likely" holds the same way for every zone type.
_DISTANCE_BUCKETS = [
    (0.0, 0.005, "0-0.5%"), (0.005, 0.015, "0.5-1.5%"),
    (0.015, 0.03, "1.5-3%"), (0.03, float("inf"), "3%+"),
]


def _distance_bucket(pct):
    for lo, hi, label in _DISTANCE_BUCKETS:
        if lo <= pct < hi:
            return label
    return _DISTANCE_BUCKETS[-1][2]


def _zone_mid(zone):
    if "top" in zone and "bottom" in zone:
        return (zone["top"] + zone["bottom"]) / 2
    return zone["price"]


def _swing_touch_events(df):
    """Every swing high/low ever confirmed (detect_swings), each tagged
    with whether it EVER later got wicked through ("swept" — the same
    untouched-high/low check detect_liquidity_levels already does inline,
    just run against every swing in history instead of only the currently-
    live ones) and formed as a point (price, not a top/bottom range) so it
    slots into the same distance-bucket machinery as FVG/OB/IFVG/Breaker
    below. Returns a plain list of {"start","price","first_touch"} dicts —
    first_touch is a placeholder timestamp (not the real sweep time, which
    this doesn't need) whenever swept=True, None otherwise, matching every
    other zone type's own "was this ever touched" field name."""
    high_col = "High" if "High" in df else "high"
    low_col = "Low" if "Low" in df else "low"
    high_arr = df[high_col].to_numpy()
    low_arr = df[low_col].to_numpy()
    highs, lows = detect_swings(df)
    out = []
    for h in highs:
        after = high_arr[h["pos"] + 1:]
        swept = after.size > 0 and bool((after >= h["price"]).any())
        out.append({"start": h["time"], "price": h["price"], "first_touch": h["time"] if swept else None})
    for l in lows:
        after = low_arr[l["pos"] + 1:]
        swept = after.size > 0 and bool((after <= l["price"]).any())
        out.append({"start": l["time"], "price": l["price"], "first_touch": l["time"] if swept else None})
    return out


_LEVEL_DETECTORS = {
    "FVG": lambda df: detect_fvgs(df),
    "IFVG": lambda df: detect_ifvgs(df),
    "Order Block": lambda df: detect_order_blocks(df),
    "Breaker Block": lambda df: detect_breaker_blocks(df),
    "Liquidity": _swing_touch_events,
}


def _historical_touch_rates(df, min_events=10):
    """{level_type: {distance_bucket: (rate, n)}} — the empirical fraction
    of every historical level of this type, formed at roughly this
    distance from price at the time, that ever got touched. n below
    min_events isn't dropped (a caller decides what to do with a thin
    sample), just flagged via the 3rd tuple element."""
    close = df["Close"] if "Close" in df else df["close"]
    pos_by_time = {t: i for i, t in enumerate(df.index)}
    out = {}
    for level_type, detector in _LEVEL_DETECTORS.items():
        try:
            zones = detector(df)
        except Exception:
            zones = []
        buckets = {}
        for z in zones:
            pos = pos_by_time.get(z["start"])
            if pos is None:
                continue
            price_then = float(close.iloc[pos])
            if price_then <= 0:
                continue
            pct = abs(_zone_mid(z) - price_then) / price_then
            bucket = _distance_bucket(pct)
            counts = buckets.setdefault(bucket, [0, 0])
            counts[0] += 1
            if z.get("first_touch") is not None:
                counts[1] += 1
        out[level_type] = {b: (touched / total, total, total >= min_events)
                            for b, (total, touched) in buckets.items() if total > 0}
    return out


def rank_levels_by_visit_probability(df, current_price, top_n=10, min_events=10):
    """The actual answer to "where is price more likely to head next":
    every currently-open level across FVG/IFVG/Order Block/Breaker Block/
    resting liquidity, ranked by the historical touch-rate for its own
    type + how far it sits from price right now (see this section's own
    module comment for the full reasoning). A level whose bucket has too
    thin a historical sample (< min_events) still appears — ranked below
    every level with a real sample, via the sort key's own ordering — with
    "sufficient": False so a caller can label it "not enough history" the
    same honest way pattern_win_rate's own callers already do, instead of
    silently pretending a 3-event sample is as trustworthy as a 300-event
    one.

    Returns a list of {"type","zone","distance_pct","probability","n",
    "sufficient"}, sorted by (sufficient, probability) descending — a real,
    well-supported probability always outranks an insufficient-data one
    regardless of the raw number, which could be a fluke."""
    hist_rates = _historical_touch_rates(df, min_events=min_events)

    open_by_type = {
        "FVG": [g for g in detect_fvgs(df) if not g["filled"]],
        "IFVG": [z for z in detect_ifvgs(df) if z.get("first_touch") is None],
        "Order Block": [o for o in detect_order_blocks(df) if not o["mitigated"]],
        "Breaker Block": [z for z in detect_breaker_blocks(df) if z.get("first_touch") is None],
    }
    live_highs, live_lows = detect_liquidity_levels(df, n_above=top_n, n_below=top_n)
    open_by_type["Liquidity"] = (
        [{"start": h["time"], "price": h["price"], "kind": "BSL"} for h in live_highs]
        + [{"start": l["time"], "price": l["price"], "kind": "SSL"} for l in live_lows]
    )

    if current_price <= 0:
        return []

    ranked = []
    for level_type, zones in open_by_type.items():
        rates = hist_rates.get(level_type, {})
        for z in zones:
            mid = _zone_mid(z)
            pct = abs(mid - current_price) / current_price
            bucket = _distance_bucket(pct)
            rate, n, sufficient = rates.get(bucket, (None, 0, False))
            ranked.append({
                "type": level_type, "zone": z, "price": mid, "distance_pct": pct * 100,
                "probability": rate, "n": n, "sufficient": sufficient,
            })

    ranked.sort(key=lambda r: (r["sufficient"], r["probability"] if r["probability"] is not None else -1),
                reverse=True)
    return ranked[:top_n]


def scan_watchlist(dfs_by_ticker, interval, top_n=8, context_dfs_by_ticker=None):
    """The multi-symbol version of best_trade_now — each ticker's own single
    best candidate (best_trade_now(..., top_n=1)), combined across every
    ticker in `dfs_by_ticker` ({ticker: already-fetched df}) and ranked
    against each other by confluence. For a sidebar/watchlist view: "which
    symbol has the best-supported setup right now," not "what's the best
    setup on the one symbol already on screen."

    context_dfs_by_ticker: optional {ticker: already-fetched HIGHER-
    timeframe df} — when given, each ticker's candidate is scored with
    that ticker's own context_df (see best_trade_now's own context_df
    param / _weighted_confluence_score), the same top-down context+entry
    read scan_timeframes uses, just one fixed pair (TF_PAIRS[0], "1W"
    context for the "4h" entry every caller here already uses) instead of
    trying every pair — trying all of TF_PAIRS per ticker would mean
    fetching 2 dataframes x 3 pairs for every ticker in the watchlist at
    once, a real fetch-cost multiplier this sidebar scan doesn't need to
    pay for a single representative top-down read. None (the default)
    preserves the original single-timeframe behavior exactly — a ticker
    missing from this dict (or whose context df is None/empty) just falls
    back to plain confluence, not an error.

    Fetching stays the caller's job — same reason best_trade_now takes an
    already-fetched df instead of a ticker to fetch itself: this module
    does no I/O of its own. app.py/crypto_app.py already have a parallel
    fetch helper (_prefetch) built for warming several tickers at once;
    reuse that rather than duplicating a second fetch-orchestration path
    here. A ticker whose fetch failed or came back empty is just skipped,
    not an error — a partial scan is still a useful scan."""
    ranked = []
    for ticker, df in dfs_by_ticker.items():
        if df is None or df.empty:
            continue
        context_df = (context_dfs_by_ticker or {}).get(ticker)
        if context_df is not None and context_df.empty:
            context_df = None
        ranked.extend(best_trade_now(df, ticker, interval, provider=None, top_n=1, context_df=context_df))
    ranked.sort(key=lambda s: (s["confluence_score"], -abs(s["distance_pct"])), reverse=True)
    return ranked[:top_n]


def scan_timeframes(dfs_by_pair, ticker, top_n=8):
    """The multi-timeframe sibling of scan_watchlist — same idea, turned
    sideways: instead of "which SYMBOL has the best setup right now" (one
    timeframe, every ticker), this is "which of THIS symbol's context+
    entry timeframe PAIRS (see TF_PAIRS) has the best setup right now."
    Each pair's own single best candidate — best_trade_now(entry_df,
    ..., top_n=1, context_df=context_df), the higher timeframe read as
    top-down context, not just another independent timeframe to flatten
    against every other one — combined across every pair in `dfs_by_pair`
    and ranked against each other by confluence (context-side counting
    CONTEXT_WEIGHT x, see _weighted_confluence_score). Not "what's active
    on whatever timeframe happens to already be charted," which is all a
    single best_trade_now call on its own can answer.

    dfs_by_pair: {(context_tf, entry_tf): (context_df, entry_df,
    entry_fetch_interval)} — the two tf labels are this app's own short
    timeframe keys ("4h", "1D", ...matching TIMEFRAMES, and TF_PAIRS'
    own shape), entry_fetch_interval is the yfinance-style interval
    string best_trade_now needs for its own LOOKBACK lookup (see its own
    docstring). Each returned candidate carries its own "timeframe"
    (the ENTRY tf — for jumping the chart there on click, since that's
    where the trade actually sits) and "context_timeframe" keys on top
    of everything best_trade_now already returns. A pair whose fetch
    failed or came back empty on either side is just skipped, not an
    error — a partial scan is still a useful scan."""
    ranked = []
    for (context_tf, entry_tf), (context_df, df, fetch_interval) in dfs_by_pair.items():
        if df is None or df.empty or context_df is None or context_df.empty:
            continue
        picks = best_trade_now(df, ticker, fetch_interval, provider=None, top_n=1, context_df=context_df)
        for p in picks:
            p["timeframe"] = entry_tf
            p["context_timeframe"] = context_tf
        ranked.extend(picks)
    ranked.sort(key=lambda s: (s["confluence_score"], -abs(s["distance_pct"])), reverse=True)
    return ranked[:top_n]


def rebalance_chain(setups, current_price):
    """Reorders `setups` (e.g. scan_timeframes' own output for one ticker)
    into a greedy nearest-price walk instead of confluence rank — direct
    request: "rank them by distance... from that place, which one is the
    closest, and so on... I want to see what the market maker's pattern
    might look like." Each setup's own entry_price is already the IDEAL
    entry (best_trade_now snaps it to the zone's own near edge — see its
    own docstring), i.e. the price that "rebalances"/hunts into that
    zone; the theory this chain visualizes is that price travels to the
    nearest un-hunted zone, rebalances it, then turns toward whichever
    remaining zone is nearest FROM THERE — a plausible step-by-step
    delivery path through the levels, not just a flat list.

    Starting from `current_price`, repeatedly picks whichever remaining
    setup's entry_price is closest to the PREVIOUS step's own price
    (current_price for the first step) and appends it, until every setup
    has been placed exactly once. A pure greedy nearest-neighbor tour —
    it does not attempt to minimize total path length across all setups
    (that's a much harder problem, and not what "closest, then closest
    from there" asks for); it only ever asks "closest to where we just
    got to."

    Returns a list of {"setup": <original dict, untouched>, "from_price",
    "to_price", "distance"} in visiting order — "distance" is SIGNED
    (to_price - from_price), so a positive value means that hop is
    upward, negative downward."""
    remaining = list(setups)
    chain = []
    from_price = current_price
    while remaining:
        nxt = min(remaining, key=lambda s: abs(s["entry_price"] - from_price))
        chain.append({
            "setup": nxt, "from_price": from_price,
            "to_price": nxt["entry_price"], "distance": nxt["entry_price"] - from_price,
        })
        from_price = nxt["entry_price"]
        remaining.remove(nxt)
    return chain


# ---------------------------------------------------------------------------
# Event-driven win rate — a second, more literal event-study methodology
# alongside research/events.py's own fixed-bar-hold one (app.py/crypto_app.py
# still use that one too, for the plain FVG/OB on-chart label). This one
# exists specifically for two things the fixed-bar version can't express:
# (1) a touch point anchored to an actual indicator's own price level (MA+FVG
# has no meaning under "touched anywhere in the raw FVG range" — the EMA IS
# the trigger), and (2) an exit tied to "the next real signal appeared," not
# an arbitrary bar count picked in advance. Deliberately NOT added to
# research/events.py — that module backs Edge Lab's own cached validated-edge
# results, and even an additive change there risks the kind of silent
# meaning-drift a later reader can't detect (the cached leaderboard was
# computed against ONE definition of "the event"; a second one living
# alongside it invites confusing them). This lives here instead: presentation
# -adjacent, used only by the live chart's own on-demand win-rate labels,
# nothing else depends on it.
EXIT_EVENT_LABELS = {"fvg": "FVG", "order_block": "Order Block", "liquidity": "Liquidity/EQH-L"}


def _exit_positions(df, exit_types):
    """Bar positions where a NEW event of any `exit_types` (a subset of
    EXIT_EVENT_LABELS' keys) first confirms — the shared "next signal"
    exit clock every win-rate function below uses instead of a fixed hold
    length. Sorted ascending, deduplicated."""
    idx = df.index
    n = len(idx)
    pos_by_time = {t: i for i, t in enumerate(idx)}
    positions = set()
    if "fvg" in exit_types:
        for g in detect_fvgs(df):
            # start = idx[i-1] (the pre-displacement candle); the gap is
            # actually confirmed once the candle AFTER the displacement
            # candle closes, i.e. pos(start) + 2.
            p = pos_by_time[g["start"]] + 2
            if p < n:
                positions.add(p)
    if "order_block" in exit_types:
        for ob in detect_order_blocks(df):
            p = ob["idx"] + 2
            if p < n:
                positions.add(p)
    if "liquidity" in exit_types:
        for sweep in detect_liquidity_sweeps(df):
            p = pos_by_time[sweep["end"]]
            if p < n:
                positions.add(p)
    return np.array(sorted(positions))


def _next_exit(exit_positions, after_pos):
    """First position in `exit_positions` strictly after `after_pos`, or
    None if the dataset runs out before the next one."""
    if len(exit_positions) == 0:
        return None
    i = np.searchsorted(exit_positions, after_pos, side="right")
    return int(exit_positions[i]) if i < len(exit_positions) else None


def _score_events(df, entries, exit_positions, min_events, news_blackout_mask=None):
    """entries: [(entry_pos, direction), ...]. Executes one bar after each
    entry_pos (that bar's open), holds until the next exit_positions entry
    (that bar's close) — raw return, sign-flipped for bearish so a "win"
    always means price moved the expected way. An entry with no exit event
    left in the dataset (ran off the end) is dropped, same as the fixed-bar
    version dropping an entry with insufficient forward bars — an
    unresolved trade isn't a scoreable one either way.

    news_blackout_mask: None (default, unchanged behavior) or a boolean
    array aligned to df.index (see news.blackout_mask) — an entry whose
    OWN execution bar (exec_pos, not entry_pos) falls inside a blackout
    window is dropped too, same as a real trader skipping a setup that
    would have opened right into a high-impact release rather than taking
    it and hoping the spread/slippage doesn't eat the edge.

    Returns {"bullish": (rate, n, sufficient), "bearish": (...)}."""
    open_ = (df["Open"] if "Open" in df else df["open"]).to_numpy()
    close = (df["Close"] if "Close" in df else df["close"]).to_numpy()
    n_bars = len(df)
    by_dir = {"bullish": [], "bearish": []}
    for entry_pos, direction in entries:
        exec_pos = entry_pos + 1
        if exec_pos >= n_bars:
            continue
        if news_blackout_mask is not None and news_blackout_mask[exec_pos]:
            continue
        exit_pos = _next_exit(exit_positions, exec_pos)
        if exit_pos is None:
            continue
        entry_price = float(open_[exec_pos])
        exit_price = float(close[exit_pos])
        raw_ret = (exit_price - entry_price) / entry_price
        sign = 1 if direction == "bullish" else -1
        by_dir[direction].append(raw_ret * sign)
    out = {}
    for direction, rets in by_dir.items():
        if not rets:
            continue
        arr = np.array(rets)
        out[direction] = (float((arr > 0).mean()), len(arr), len(arr) >= min_events)
    return out


def fvg_event_win_rate(df, exit_types=("fvg", "order_block", "liquidity"), min_events=20,
                        news_blackout_mask=None):
    """FVG win rate under the event-driven exit rule (see module comment
    above) — touch = first_touch (unchanged from the fixed-bar version),
    exit = the next chosen event type forming. news_blackout_mask: see
    _score_events."""
    idx = df.index
    pos_by_time = {t: i for i, t in enumerate(idx)}
    entries = [(pos_by_time[g["first_touch"]], g["type"])
               for g in detect_fvgs(df) if g["first_touch"] is not None]
    return _score_events(df, entries, _exit_positions(df, set(exit_types)), min_events,
                          news_blackout_mask=news_blackout_mask)


def fvg_run_up_stats(df, min_events=20):
    """For every past FVG that got touched again (first_touch is not
    None), measures how far price ran in the zone's own favor between
    when it FORMED and that first touch — the "warm-up" window sitting
    BEFORE fvg_event_win_rate's own entry point (which starts counting
    from first_touch itself, not formation) — i.e. "how much room was
    there before price came back to retest this."

    A bullish FVG's raw_top is the edge nearest to where price already
    was at formation (the zone's own size as a trader actually saw it,
    not the final post-encroachment top/bottom — see detect_fvgs' own
    note on raw_top/raw_bottom); run-up is the highest high reached
    between formation and first_touch, as a % move away from that edge.
    A bearish FVG mirrors this from raw_bottom, tracking the lowest low.

    Only counts zones that actually GOT touched again — one still open
    (no first_touch yet) has an unresolved, still-growing run-up that
    would bias the average toward whatever partial value it's sitting at
    right now if it were included.

    Returns {"bullish": {"avg_pct", "median_pct", "n", "sufficient"}, "bearish": {...}}."""
    high = (df["High"] if "High" in df else df["high"]).to_numpy()
    low = (df["Low"] if "Low" in df else df["low"]).to_numpy()
    idx = df.index
    pos_by_time = {t: i for i, t in enumerate(idx)}
    by_dir = {"bullish": [], "bearish": []}
    for g in detect_fvgs(df):
        if g["first_touch"] is None:
            continue
        start_pos = pos_by_time[g["start"]]
        touch_pos = pos_by_time[g["first_touch"]]
        if touch_pos <= start_pos:
            continue
        if g["type"] == "bullish":
            ref = g["raw_top"]
            pct = (float(high[start_pos:touch_pos + 1].max()) - ref) / ref * 100
        else:
            ref = g["raw_bottom"]
            pct = (ref - float(low[start_pos:touch_pos + 1].min())) / ref * 100
        by_dir[g["type"]].append(pct)
    out = {}
    for direction, pcts in by_dir.items():
        if not pcts:
            continue
        arr = np.array(pcts)
        out[direction] = {
            "avg_pct": float(arr.mean()), "median_pct": float(np.median(arr)),
            "n": len(arr), "sufficient": len(arr) >= min_events,
        }
    return out


def order_block_event_win_rate(df, exit_types=("fvg", "order_block", "liquidity"), min_events=20,
                                news_blackout_mask=None):
    """Order Block win rate under the event-driven exit rule — touch =
    first_touch (unchanged from the fixed-bar version), exit = the next
    chosen event type forming. news_blackout_mask: see _score_events."""
    idx = df.index
    pos_by_time = {t: i for i, t in enumerate(idx)}
    entries = [(pos_by_time[ob["first_touch"]], ob["type"])
               for ob in detect_order_blocks(df) if ob["first_touch"] is not None]
    return _score_events(df, entries, _exit_positions(df, set(exit_types)), min_events,
                          news_blackout_mask=news_blackout_mask)


def liquidity_event_win_rate(df, exit_types=("fvg", "order_block", "liquidity"), min_events=20,
                              news_blackout_mask=None):
    """External-range-liquidity win rate under the event-driven exit rule
    — touch = the sweep itself (direction convention matches
    detect_liquidity_sweeps' own `type`: sweeping a high points bearish,
    sweeping a low points bullish), exit = the next chosen event type
    forming. news_blackout_mask: see _score_events."""
    idx = df.index
    pos_by_time = {t: i for i, t in enumerate(idx)}
    entries = [(pos_by_time[sweep["end"]], sweep["type"]) for sweep in detect_liquidity_sweeps(df)]
    return _score_events(df, entries, _exit_positions(df, set(exit_types)), min_events,
                          news_blackout_mask=news_blackout_mask)


def ma_fvg_event_win_rate(df, exit_types=("fvg", "order_block", "liquidity"), min_events=20, periods=MA_FVG_PERIODS,
                           news_blackout_mask=None):
    """MA+FVG win rate — the one case where "touch" genuinely means
    something different from plain FVG: entry is the first bar an EMA
    (MA_FVG_PERIODS) actually reaches the gap's own ORIGINAL [raw_bottom,
    raw_top] range while that gap is still open (between its confirmed
    formation and its recorded fill point, or the last bar if it never
    filled) — "the indicator's own level," not "price touched anywhere in
    the zone." Exit = the next chosen event type forming, same as every
    other event-driven win rate here. news_blackout_mask: see
    _score_events."""
    close = df["Close"] if "Close" in df else df["close"]
    idx = df.index
    pos_by_time = {t: i for i, t in enumerate(idx)}
    emas = {p: ema(close, p).to_numpy() for p in periods}
    entries = []
    for g in detect_fvgs(df):
        start_pos = pos_by_time[g["start"]]
        end_pos = pos_by_time[g["end"]]
        touch_pos = None
        for p in range(start_pos + 2, end_pos + 1):
            for period in periods:
                v = emas[period][p]
                if v == v and g["raw_bottom"] <= v <= g["raw_top"]:  # v==v: NaN guard
                    touch_pos = p
                    break
            if touch_pos is not None:
                break
        if touch_pos is not None:
            entries.append((touch_pos, g["type"]))
    return _score_events(df, entries, _exit_positions(df, set(exit_types)), min_events,
                          news_blackout_mask=news_blackout_mask)


# ---------------------------------------------------------------------------
# Generalized indicator confluence — MA+FVG (above) was the FIRST case of
# this: "does an indicator agree with a zone's own direction, while it's
# still live." This generalizes it past EMA-only and past FVG-only, without
# touching MA+FVG's own functions (best_trade_now's "recommended trade" pick
# depends on _ma_fvg_candidates specifically meaning EMA20/50 — left exactly
# as-is; this is a SEPARATE, additional capability for the visual
# highlighting + win-rate stats layer, not a replacement).
#
# Two different KINDS of indicator, because "confluence" means something
# different for each:
#   "level"  — a price-level indicator (EMA now; Bollinger/VWAP would slot
#              in the same way later). Confluence = the indicator's own
#              value sits INSIDE the zone's [bottom, top] range (a ZONE) or
#              within `tolerance_pct` of the level's own price (a POINT
#              level like Liquidity/EQH-L, which has no width to be
#              "inside" of).
#   "state"  — an oscillator (RSI, MACD). These have their own 0-100 or
#              unbounded scale, not a price — comparing them against a
#              price range is meaningless. Confluence instead means "is the
#              oscillator in the state that AGREES with this zone/level's
#              own direction" (RSI oversold under a bullish read, overbought
#              under bearish; MACD histogram's sign matching direction),
#              checked at the same bar, independent of price entirely.
INDICATOR_SPECS = {
    "ema20": {"label": "EMA 20", "kind": "level", "compute": lambda close: ema(close, 20)},
    "ema50": {"label": "EMA 50", "kind": "level", "compute": lambda close: ema(close, 50)},
    "rsi": {
        "label": "RSI", "kind": "state", "compute": lambda close: rsi(close, 14),
        # Momentum exhaustion agreeing with the zone's own reversal thesis
        # — oversold under demand, overbought under supply. Standard 30/70.
        "matches": lambda value, direction: value < 30 if direction == "bullish" else value > 70,
    },
    "macd": {
        "label": "MACD", "kind": "state", "compute": lambda close: macd(close)[2],  # histogram
        # The histogram's SIGN already agreeing with the zone's own
        # direction — not a cross event (a cross landing on the exact same
        # bar as the zone/level is a much narrower, harder-to-hit
        # condition; sign is the simpler, more robust "does momentum
        # currently favor this side" read, and reuses data already
        # computed for the chart's own MACD pane).
        "matches": lambda value, direction: value > 0 if direction == "bullish" else value < 0,
    },
}


def _indicator_series(close, indicator_keys):
    return {k: INDICATOR_SPECS[k]["compute"](close) for k in indicator_keys}


def _confluence_at(spec, value, direction, zone_bottom=None, zone_top=None, level_price=None, tolerance=None):
    if value != value:  # NaN guard (indicator warmup)
        return False
    if spec["kind"] == "level":
        if zone_bottom is not None:
            return zone_bottom <= value <= zone_top
        return abs(value - level_price) <= tolerance
    return spec["matches"](value, direction)


def zone_indicator_matches(zones, close, indicator_keys):
    """Live version — which of `zones` (FVG or Order Block shape: dicts
    with type/top/bottom, and "filled" or "mitigated" marking whether
    still open) have ANY of `indicator_keys` showing confluence RIGHT NOW
    (each indicator's LAST value only — "is this true at this exact
    moment," matching how _ma_fvg_candidates itself checks "now," not
    history). Returns {zone["start"]: [matched_key, ...]} for zones with
    at least one match; a zone with none isn't in the dict at all."""
    if len(close) < 2:
        return {}
    series = _indicator_series(close, indicator_keys)
    out = {}
    for z in zones:
        if not _zone_is_open(z):
            continue
        matched = [k for k in indicator_keys
                   if _confluence_at(INDICATOR_SPECS[k], series[k].iloc[-1], z["type"],
                                      zone_bottom=z["bottom"], zone_top=z["top"])]
        if matched:
            out[z["start"]] = matched
    return out


def point_level_indicator_matches(levels, direction, close, indicator_keys, tolerance_pct=0.0015):
    """Live version for POINT levels (Liquidity BSL/SSL — each a plain
    {"time", "price"} dict, no width) — `direction` is the SAME for every
    level in the list (detect_liquidity_levels already separates "above"
    from "below" into two lists, so this takes one call per side).
    tolerance_pct: how close a level-kind indicator's value needs to be to
    the level's own price to count as "at" it (a point has no range to be
    "inside" of, unlike a zone) — 0.15%, the same tolerance
    detect_equal_levels already uses for "equal" prices, reused here for
    the same reason: a sensible, already-established definition of "close
    enough to matter" on this chart. Returns {level["time"]: [matched_key,
    ...]} for levels with at least one match."""
    if len(close) < 2:
        return {}
    series = _indicator_series(close, indicator_keys)
    out = {}
    for lvl in levels:
        matched = [k for k in indicator_keys
                   if _confluence_at(INDICATOR_SPECS[k], series[k].iloc[-1], direction,
                                      level_price=lvl["price"], tolerance=lvl["price"] * tolerance_pct)]
        if matched:
            out[lvl["time"]] = matched
    return out


# ---------------------------------------------------------------------------
# Custom trade rules — lets the user build their OWN entry/exit/stop instead
# of trusting best_trade_now's automatic pick, so "how was this line decided"
# is answered by a rule they wrote themselves, not a ranking formula. Each of
# entry/exit/stop is ONE of:
#   {"anchor": "current_price"}
#   {"anchor": "next_event", "detector": "fvg"|"order_block"|"liquidity",
#    "side": "above"|"below", "n": 1, "zone_point": 0.0-1.0}
# "next" here means NEAREST BY PRICE among zones/levels open RIGHT NOW, not
# "next forward in time" — there's no future data on a live chart to scan
# forward through (that concept only makes sense for the win-rate engine
# above, which has full history to work with). n=1 is the nearest one on
# the chosen side, n=2 skips past it to the 2nd-nearest, and so on.
# zone_point only applies to FVG/Order Block (an actual range) — a
# Liquidity level has no width to pick a point within, so it's ignored
# there (the UI disables that control for that case).
RULE_DETECTOR_LABELS = {"fvg": "FVG", "order_block": "Order Block", "liquidity": "Liquidity"}
# Quarter-step positions within a zone, near (0.0) to far (1.0) — Entry/
# Target/Stop each pick their own independently (e.g. entry at 75% of the
# zone, stop at 25%), rather than one shared point for all three. Kept as
# an ordered list (not just a set of valid values) since both app.py's and
# backtest_ui.py's sliders build their step options directly from it.
#
# "near"/"far" is relative to CURRENT PRICE, not the zone's own raw
# bottom/top — see _zone_point_price's own docstring for why: a "below"
# zone's near edge (closest to where price approaches from) is its TOP,
# while an "above" zone's near edge is its BOTTOM, so a fixed bottom=0%/
# top=100% mapping made the same zone_point value mean "shallow entry" for
# one direction and "deep entry" for the other. Confirmed directly as a
# real, reported issue (not a hypothetical): switching a rule's direction
# didn't change what the SAME zone_point setting actually did to entry/
# stop/target, when it should have.
ZONE_POINT_STEPS = [0.0, 0.25, 0.5, 0.75, 1.0]
# "high"/"mid"/"low" strings are the pre-quarter-step shape zone_point used
# to be saved as — old sweep results in backtest_results.db still have rule
# descriptions built under that shape, so resolving/labeling a zone_point
# accepts either form rather than breaking on replay of old data.
_LEGACY_ZONE_POINT_ALIASES = {"high": 1.0, "mid": 0.5, "low": 0.0}


def zone_point_fraction(zone_point):
    """Normalizes a zone_point value — a plain 0.0-1.0 fraction, or a
    legacy "high"/"mid"/"low" string — to its 0.0-1.0 fraction from the
    zone's bottom to its top."""
    if isinstance(zone_point, str):
        return _LEGACY_ZONE_POINT_ALIASES.get(zone_point, 0.5)
    return zone_point


def zone_point_label(zone_point):
    """A select_slider shows exactly this text at both ends of its own
    track (its min/max option) as well as above the handle for whatever's
    currently selected — labeling the two extremes answers "which end is
    which" right on the widget itself, not just in a help tooltip someone
    has to go hover to find.

    "Near"/"Far" (distance from current price), not "Low"/"High" (the
    zone's own raw bottom/top) — deliberately direction-agnostic labels,
    matching _zone_point_price's own near/far convention: 0% always
    resolves to whichever edge of the zone sits closest to current price,
    100% to whichever sits farthest, for BOTH a "below" and an "above"
    zone. "Low"/"High" would be actively wrong half the time under that
    convention — for a "below" zone (bullish entry/stop, bearish target)
    the near edge is the zone's TOP, not its bottom."""
    frac = zone_point_fraction(zone_point)
    if frac <= 0.0:
        return "Near"
    if frac >= 1.0:
        return "Far"
    return f"{round(frac * 100)}%"


def _zone_bounds(zone):
    """The zone's ORIGINAL position as it first formed, not the
    consequent-encroachment-shrunk one detect_fvgs also tracks (as "top"/
    "bottom" themselves) for its own eaten-away on-chart visualization.
    Confirmed directly as the cause of two related bugs: judging "which
    side"/"how near" off the shrunk bounds, or computing a zone_point
    price from them, put the resolved price inside whatever sliver of the
    gap happened to still be untouched at evaluation time — not the gap a
    trader actually saw at formation. And since that sliver keeps
    shrinking as price approaches, a stop/target built from it could sit
    almost on top of the CURRENT price already, resolving as "hit" within
    a bar or two without price ever genuinely tracing back into anything.
    raw_top/raw_bottom (FVGs only — set once at formation, never
    shrunk) fix both. Order Block/Liquidity zones have no raw_* fields
    (nothing shrinks them the same way detect_fvgs's own consequent-
    encroachment scan does), so they fall back to their own top/bottom,
    unchanged. Staying a genuinely valid candidate for as long as price
    hasn't fully broken through remains the zone's own "filled" flag's
    job, computed from the shrunk bounds internally — untouched by this;
    only the PRICE a resolved rule returns changes here."""
    return zone.get("raw_bottom", zone["bottom"]), zone.get("raw_top", zone["top"])


def _nth_zone_on_side(zones, current_price, side, n):
    """The nth-nearest (1-indexed) zone to current_price on `side`
    ("above" or "below"), sorted so n=1 is always the closest. None if
    fewer than n zones exist on that side."""
    if side == "above":
        candidates = sorted((z for z in zones if _zone_bounds(z)[0] >= current_price),
                             key=lambda z: _zone_bounds(z)[0])
    else:
        candidates = sorted((z for z in zones if _zone_bounds(z)[1] <= current_price),
                             key=lambda z: -_zone_bounds(z)[1])
    return candidates[n - 1] if len(candidates) >= n else None


def _zone_point_price(zone, zone_point, side):
    """side is the zone's OWN side relative to price ("above" or "below" —
    same value a rule's "side" field already carries), needed because a
    zone_point fraction has to mean the same THING (how deep/aggressive an
    entry) regardless of which side of price the zone sits on — it can't
    just be resolved off the zone's own bottom/top the way an earlier
    version of this function did. Confirmed directly as a real bug, not
    just an inconsistency: a "below" zone's TOP edge is the side CLOSEST
    to current price (price approaches from above, into the zone) while
    an "above" zone's BOTTOM edge is the close side (price approaches from
    below) — so a fixed "0%=bottom, 100%=top" mapping made the exact same
    zone_point value mean "shallow, easy-fill entry" for one direction and
    "deep, aggressive entry" for the other, silently, with no way to tell
    from the number alone. near/far below re-centers 0%/100% on distance
    from current price instead of raw bottom/top, so the same zone_point
    means the same thing (0%=nearest to price, 100%=farthest) for both
    bullish and bearish rules — matching zone_point_label's own Near/Far
    naming."""
    bottom, top = _zone_bounds(zone)
    frac = zone_point_fraction(zone_point)
    near, far = (top, bottom) if side == "below" else (bottom, top)
    return near + (far - near) * frac


# --- Cached "as-of-bar" resolution registry --------------------------------
#
# backtest_custom_rule below is the only place in this codebase that resolves
# a rule against a GROWING slice, hundreds/thousands of times per run (once
# per bar of its walk-forward loop). Every detector this file knows how to
# anchor a rule to (fvg, order_block, liquidity) is itself an O(n) full-
# dataframe scan — fine to pay ONCE, a genuine O(n^2) hang risk to pay again
# at every bar. Confirmed directly (a real timing test): re-running
# detect_liquidity_levels fresh per bar cost 33.7x what detecting once and
# projecting onto each bar costs, for a 3000-bar series.
#
# Rather than one hand-rolled cache dict + one duplicated if/elif branch per
# detector (which is how the fvg/order_block/liquidity caches were first
# built, one at a time), each cacheable detector is registered here as a
# (builder, resolver) pair:
#   - builder(full_df, max_scan_bars) -> raw   — the detector's own O(n) scan
#     over the WHOLE dataframe, called at most ONCE per backtest_custom_rule
#     run, no matter how many bars the walk covers.
#   - resolver(raw, rule, current_price, as_of_bar, origin_out) -> price|None
#     — projects that raw, full-history detection onto "what was actually
#     knowable as of this bar," using each detector's own no-lookahead
#     bookkeeping (a zone's formation_i/expiry_bar for fvg/order_block, a
#     swing's confirmed_pos for liquidity).
# Adding a new cacheable rule detector later means writing one builder + one
# resolver function and registering them below — resolve_rule and
# backtest_custom_rule's own loop never need to change, and there's no
# separate per-detector cache dict for a new one to accidentally fall out of
# sync with (see _build_detector_cache's own docstring on why "one lazily-
# filled dict keyed off a single, immutable full-dataframe reference" makes
# that class of bug structurally impossible rather than a discipline you
# have to remember).

def _build_zone_tracker(zones):
    """Wraps a raw detect_fvgs/detect_order_blocks zone list for the
    incremental as_of_bar walk backtest_custom_rule performs — zones
    sorted by formation_i once, with per-zone history-checkpoint state
    advanced forward-only as as_of_bar increases across calls, instead of
    re-scanning EVERY zone's full history from scratch (including zones
    long since closed) on every single resolve_rule call. Confirmed
    directly via cProfile as the dominant cost of a real backtest run:
    the old per-call full-history rescan this replaces (_zone_at, called
    once per zone per resolve_rule call) alone accounted for ~4.8s of a
    real 4.4s BTC-USD/4h/730d/FVG-anchored run.

    Never mutates a raw zone dict (those come straight from detect_fvgs/
    detect_order_blocks, which are @st.cache_data-cached and may be
    shared with other callers) — all tracked state lives in this
    tracker's own side-dict, keyed by index into `zones`.

    Correctness relies on as_of_bar only ever increasing across calls
    within one backtest_custom_rule run (see its own docstring — i only
    ever grows across that loop) — advancing forward from the tracker's
    own last-seen bar each time, never restarting from scratch, is what
    makes this safe rather than a lookahead risk: a zone's state as of
    bar j never depends on anything after j regardless of when it's
    computed, so replaying the same checkpoints in the same order just
    once, instead of once per call, changes nothing about the result."""
    order = sorted(range(len(zones)), key=lambda idx: zones[idx]["formation_i"])
    return {"zones": zones, "order": order, "open_ptr": 0, "active": {}, "last_j": -1}


def _advance_zone_tracker(tracker, j):
    zones = tracker["zones"]
    order = tracker["order"]
    active = tracker["active"]
    ptr = tracker["open_ptr"]
    n = len(order)
    while ptr < n and zones[order[ptr]]["formation_i"] <= j:
        idx = order[ptr]
        zone = zones[idx]
        active[idx] = {
            "hist_ptr": 0,
            "top": zone.get("raw_top", zone.get("_formation_top")),
            "bottom": zone.get("raw_bottom", zone.get("_formation_bottom")),
        }
        ptr += 1
    tracker["open_ptr"] = ptr
    closed = []
    for idx, state in active.items():
        zone = zones[idx]
        hist = zone["history"]
        hp = state["hist_ptr"]
        hn = len(hist)
        while hp < hn and hist[hp][0] <= j:
            _, state["top"], state["bottom"] = hist[hp]
            hp += 1
        state["hist_ptr"] = hp
        filled = state["top"] <= state["bottom"]
        expiry_bar = zone["expiry_bar"]
        if filled or ((expiry_bar is not None) and (j >= expiry_bar)):
            closed.append(idx)
    for idx in closed:
        del active[idx]
    tracker["last_j"] = j


def _resolve_zone_tracker_cached(tracker, rule, current_price, as_of_bar, origin_out):
    if tracker["last_j"] != as_of_bar:
        _advance_zone_tracker(tracker, as_of_bar)
    zones = tracker["zones"]
    side = rule.get("side", "above")
    open_zones = [{**zones[idx], "top": state["top"], "bottom": state["bottom"],
                    "filled": False, "expired": False}
                  for idx, state in tracker["active"].items()]
    z = _nth_zone_on_side(open_zones, current_price, side, rule.get("n", 1))
    if z is not None and origin_out is not None:
        origin_out["time"] = z["start"]
    return _zone_point_price(z, rule.get("zone_point", 0.5), side) if z else None


def _resolve_fvg_cached(raw, rule, current_price, as_of_bar, origin_out):
    return _resolve_zone_tracker_cached(raw, rule, current_price, as_of_bar, origin_out)


def _resolve_order_block_cached(raw, rule, current_price, as_of_bar, origin_out):
    return _resolve_zone_tracker_cached(raw, rule, current_price, as_of_bar, origin_out)


def _first_touch_bar(arr, pos, price, above):
    """First bar index after `pos` where `arr` reaches/crosses `price` —
    None if it never does across the rest of the series. `above=True` for
    a swing high's own resistance level (touched once a later high >=
    price); False for a swing low's support level (touched once a later
    low <= price). Computed ONCE per swing point with a single vectorized
    numpy scan, replacing a fresh array-slice-plus-.all() scan repeated at
    EVERY as_of_bar call in the walk-forward loop (confirmed directly via
    cProfile: ~1.3s of a real 4.4s run, called ~1M times total)."""
    after = arr[pos + 1:]
    mask = (after >= price) if above else (after <= price)
    hits = np.nonzero(mask)[0]
    return int(pos + 1 + hits[0]) if hits.size > 0 else None


def _resolve_liquidity_cached(raw, rule, current_price, as_of_bar, origin_out):
    j = as_of_bar
    # Confirmed directly this is lookahead-safe, not just faster:
    # detect_swings is a single forward-only pass whose state at bar i
    # depends only on bars 0..i, so filtering the FULL run's own output by
    # confirmed_pos <= j is bit-for-bit identical to what a fresh
    # detect_swings(df.iloc[:j+1]) call would have produced — same swings,
    # same confirmation bars, just computed once instead of redundantly at
    # every bar (see detect_swings' own pos/confirmed_pos docstring). Same
    # reasoning covers pre-computed "first touch" bars below: a point's
    # own future price path never depends on which bar you ask "was this
    # touched yet" from.
    live_highs = sorted(
        (h for h, touch in zip(raw["highs"], raw["high_touch"])
         if h["confirmed_pos"] <= j and h["price"] > current_price and (touch is None or touch > j)),
        key=lambda h: h["price"])
    live_lows = sorted(
        (l for l, touch in zip(raw["lows"], raw["low_touch"])
         if l["confirmed_pos"] <= j and l["price"] < current_price and (touch is None or touch > j)),
        key=lambda l: l["price"], reverse=True)
    levels = live_highs if rule.get("side", "above") == "above" else live_lows
    n = rule.get("n", 1)
    if len(levels) < n:
        return None
    if origin_out is not None:
        origin_out["time"] = levels[n - 1]["time"]
    return levels[n - 1]["price"]


def _build_liquidity_tracker(full_df):
    highs, lows = detect_swings(full_df)
    high_arr = (full_df["High"] if "High" in full_df else full_df["high"]).to_numpy()
    low_arr = (full_df["Low"] if "Low" in full_df else full_df["low"]).to_numpy()
    return {
        "highs": highs, "lows": lows,
        "high_touch": [_first_touch_bar(high_arr, h["pos"], h["price"], above=True) for h in highs],
        "low_touch": [_first_touch_bar(low_arr, l["pos"], l["price"], above=False) for l in lows],
    }


_DETECTOR_BUILDERS = {
    "fvg": lambda full_df, max_scan_bars: _build_zone_tracker(
        detect_fvgs(full_df, max_scan_bars=max_scan_bars, record_history=True)),
    "order_block": lambda full_df, max_scan_bars: _build_zone_tracker(
        detect_order_blocks(full_df, max_scan_bars=max_scan_bars, record_history=True)),
    "liquidity": lambda full_df, max_scan_bars: _build_liquidity_tracker(full_df),
}
_DETECTOR_RESOLVERS = {
    "fvg": _resolve_fvg_cached,
    "order_block": _resolve_order_block_cached,
    "liquidity": _resolve_liquidity_cached,
}


def _build_detector_cache(full_df, max_scan_bars=None):
    """A fresh, empty per-detector cache for ONE backtest_custom_rule run —
    pass the result as resolve_rule's `cache` argument (with `as_of_bar=i`
    per call) instead of re-detecting from scratch at every bar.

    Lazily filled in by resolve_rule the first time a rule actually
    references each detector — a rule that never uses, say, "order_block"
    on any of its three legs never pays for detect_order_blocks at all,
    unlike the eager "detect everything up front regardless of whether this
    rule needs it" version this replaced. Safe to be lazy specifically
    because `full_df` is captured HERE, once, at cache-creation time — a
    detector built on first use always scans the TRUE full dataframe
    regardless of which bar's resolve_rule call happens to trigger that
    first build (as opposed to whatever slice that particular call was
    passed), so laziness only changes WHEN a detector's cost is paid, never
    WHAT it computes. This is also what makes it structurally impossible
    for two detectors' caches to disagree about "as of which bar" — neither
    this dict nor anything built into it stores a bar index at all; the
    caller passes as_of_bar explicitly on every resolve_rule call instead,
    so there's no stored "as_of" state for a future third/fourth cached
    detector to forget to keep in sync."""
    return {"_full_df": full_df, "_max_scan_bars": max_scan_bars}


def _detector_cache_get(cache, name):
    if name not in cache:
        cache[name] = _DETECTOR_BUILDERS[name](cache["_full_df"], cache["_max_scan_bars"])
    return cache[name]


def resolve_rule(rule, df, current_price, max_scan_bars=None, origin_out=None, cache=None, as_of_bar=None):
    """A single entry/exit/stop rule (see this module comment above) →
    an actual price, or None when the rule can't be resolved right now
    (e.g. "3rd-nearest FVG below price" but only 2 exist) — an honest
    "not available," not a fabricated number.

    origin_out: None (default — every existing call site) is a no-op.
    When the caller passes a dict, and the rule resolves via a real
    zone/level (fvg, order_block, or liquidity — everything except
    "current_price", which has no origin), this sets
    origin_out["time"] to that zone/level's OWN formation time (its
    "start" for fvg/order_block, the swing point's own "time" for
    liquidity) — where the thing that produced this price first came
    from, as opposed to `current_price`'s return value itself. Left
    untouched (caller sees no "time" key) when the rule doesn't resolve
    or is anchored to current_price.

    max_scan_bars: threaded straight through to detect_fvgs/
    detect_order_blocks (see their own docstrings — an unbounded fill-
    scan is a real O(n^2) risk on a long history). None (every existing
    call site) preserves the original unbounded-scan behavior; only
    backtest_custom_rule below passes a real cap, since IT calls this
    function once per bar of a walk-forward loop — the same unbounded
    scan that's merely slow once becomes a genuine hang risk repeated
    hundreds of times. Ignored when `cache` is given (the cache carries
    its own max_scan_bars, fixed at cache-creation time).

    cache/as_of_bar: None (default — every call site except
    backtest_custom_rule) preserves the original behavior exactly: each
    detector runs fresh against whatever `df` was passed in. When the
    caller instead passes a cache built by _build_detector_cache(full_df)
    plus the current walk-forward bar as as_of_bar, resolution is
    dispatched through _DETECTOR_RESOLVERS instead — the detector's own
    full-history scan happens at most once (see _build_detector_cache's
    own docstring), and this just projects it onto "as of as_of_bar." Both
    must be given together; as_of_bar is meaningless without a cache to
    resolve against, and unused (df resolves fresh) without it."""
    if rule["anchor"] == "current_price":
        return current_price
    detector = rule["detector"]
    if cache is not None:
        raw = _detector_cache_get(cache, detector)
        return _DETECTOR_RESOLVERS[detector](raw, rule, current_price, as_of_bar, origin_out)
    side = rule.get("side", "above")
    n = rule.get("n", 1)
    if detector == "fvg":
        zones = [g for g in detect_fvgs(df, max_scan_bars=max_scan_bars) if _zone_is_open(g)]
        z = _nth_zone_on_side(zones, current_price, side, n)
        if z is not None and origin_out is not None:
            origin_out["time"] = z["start"]
        return _zone_point_price(z, rule.get("zone_point", 0.5), side) if z else None
    if detector == "order_block":
        zones = [o for o in detect_order_blocks(df, max_scan_bars=max_scan_bars) if _zone_is_open(o)]
        z = _nth_zone_on_side(zones, current_price, side, n)
        if z is not None and origin_out is not None:
            origin_out["time"] = z["start"]
        return _zone_point_price(z, rule.get("zone_point", 0.5), side) if z else None
    if detector == "liquidity":
        above, below = detect_liquidity_levels(df, n_above=n, n_below=n)
        levels = above if side == "above" else below
        if len(levels) < n:
            return None
        if origin_out is not None:
            origin_out["time"] = levels[n - 1]["time"]
        return levels[n - 1]["price"]
    return None


def hold_resolved_trade(held, ctx_key, entry_rule, exit_rule, stop_rule, df, current_price):
    """Resolves entry_rule/exit_rule/stop_rule against df/current_price
    exactly like resolve_rule already does — but once resolved, HOLDS
    that exact result fixed instead of re-resolving on every call, so a
    trade drawn on a live-refreshing chart doesn't visibly reshuffle
    every tick just because current_price nudged which zone counts as
    "nearest" (resolve_rule's own side/n selection is current_price-
    relative by design — that's correct for "what does this rule mean
    right now," but wrong for "keep showing me the trade you already
    found," which is what a chart re-rendering every 1-5s actually
    needs).

    held: whatever this function returned last time (None on first
    call, or once the caller's own ctx_key changes). ctx_key: caller-
    built identity for "is this still the same rule setup" (e.g.
    ticker|timeframe|rule JSON) — a change here always forces a fresh
    resolve; holding a stale price across an actual context switch
    would be wrong, not stable.

    A held trade is treated as a real position, not just a cached
    number: once live, it stays exactly as first resolved until price
    actually reaches its own stop or target — the same event that
    would close a real trade (see research/signals.py's own active/
    hit_tp/hit_sl status tracking) — only then does a new one resolve.
    A combo where stop/target don't land on opposite, sane sides of
    entry can't be tracked as a real trade at all, so it's re-resolved
    on every call instead of held — matches backtest_custom_rule's own
    "stop < entry < target for a long" sanity check.

    Returns (new_held_or_None, entry_price, exit_price, stop_price) —
    the caller is responsible for persisting new_held for next time."""
    def _resolve_fresh():
        entry = resolve_rule(entry_rule, df, current_price)
        exit_ = resolve_rule(exit_rule, df, current_price)
        stop = resolve_rule(stop_rule, df, current_price)
        if entry is None or exit_ is None or stop is None:
            return None, entry, exit_, stop
        if stop < entry < exit_:
            direction = "bullish"
        elif exit_ < entry < stop:
            direction = "bearish"
        else:
            return None, entry, exit_, stop
        return ({"key": ctx_key, "entry": entry, "exit": exit_, "stop": stop, "direction": direction},
                entry, exit_, stop)

    if held is None or held.get("key") != ctx_key:
        return _resolve_fresh()

    if held["direction"] == "bullish":
        invalidated = current_price <= held["stop"] or current_price >= held["exit"]
    else:
        invalidated = current_price >= held["stop"] or current_price <= held["exit"]
    if invalidated:
        return _resolve_fresh()
    return held, held["entry"], held["exit"], held["stop"]


def backtest_custom_rule(df, entry_rule, exit_rule, stop_rule, max_signals=100, max_scan_bars=500,
                          session=None, min_rr=None, min_stop_pct=None, min_stop_abs=None,
                          max_trades_per_session=None, cost_pct=None, news_blackout_mask=None):
    """Walks df bar by bar, taking ONE trade at a time — resolves
    entry/exit/stop using ONLY df.iloc[:i+1] (everything up to and
    including the current bar), the same no-lookahead discipline
    detect_swings' own docstring lays out for the rest of this codebase.
    A trade opens the moment all three resolve to a sane setup (stop and
    target genuinely on opposite sides of entry, in the right order —
    e.g. a long needs stop < entry < target), then holds until price
    touches stop or target, whichever comes first. A bar that touches
    both is scored a loss (stop checked first) — the conservative
    assumption when the exact intrabar path isn't known. Only looks for
    the NEXT signal once the current trade has closed — no overlapping
    positions, matching how one trader actually runs one rule.

    This is a real simulation of the EXACT rule as configured (same
    detector/side/nth/zone-point for each of entry/exit/stop), unlike
    the zone-type win rate and run-up stats elsewhere in this module,
    which are read off the underlying pattern type in general.

    max_signals hard-stops the simulation after this many CLOSED trades
    (not just a courtesy cap) and max_scan_bars bounds each individual
    detect_fvgs/detect_order_blocks call's own fill-scan (see
    resolve_rule's own docstring) — both exist because this repeats real
    zone detection on a growing slice of df at EVERY bar without an open
    trade, so an uncapped version of either is a genuine hang risk on a
    long backtest window, not just a slow one.

    session: None (default, unchanged behavior) or a (start, end) pair of
    datetime.time — same shape and same America/New_York wall-clock
    convention as app.py's own KILL_ZONES. When set, a NEW trade can only
    be OPENED on a bar whose NY time falls in [start, end); an already-
    open trade still gets checked for its stop/target on every bar
    regardless — restricting the SEARCH for a setup to a session window
    doesn't mean abandoning risk management the moment the clock rolls
    past it. Weekends are never in-session (no bar ever falls on one for
    an equities/FX feed, but crypto's 24/7 feed does reach Saturday/
    Sunday bars, so this is a real exclusion there, not a no-op).

    min_rr: None (default) or a float — a resolved setup is only taken
    when reward/risk (|target-entry| / |entry-stop|) is at least this.
    Guards against the "technically sane but nearly 1:1" trades that pad
    trade COUNT without adding real edge.

    min_stop_pct: None (default) or a float (e.g. 0.001 = 0.1% of entry
    price) — a resolved setup is only taken when the stop distance is at
    least this fraction of the entry price. Guards against the opposite
    failure mode from min_rr: a stop sitting almost on top of entry
    (two zones that happen to be nearly the same price) produces a
    technically-valid but unrealistically huge R:R — real fills/spread
    would never let a stop that tight hold. Confirmed directly as the
    actual mechanism behind this rule's own R:R 24.19 trades — a compounding
    simulation off those blew up to $400M from $100 in 94 trades, which is
    exactly the kind of result this parameter exists to keep from happening.

    min_stop_abs: None (default) or a float — same guard as min_stop_pct,
    in the instrument's own raw price units (points/dollars) instead of a
    percentage. Independent of min_stop_pct — pass whichever unit the
    caller actually wants; passing both applies both (a setup must clear
    each one that's set).

    max_trades_per_session: None (default) or an int — caps how many NEW
    trades can OPEN per NY calendar day within the `session` window
    (requires session to be set; ignored otherwise, since "per session"
    has no meaning without one). Matches a real trader's own discipline
    of only taking their best N setups a day instead of every one the
    rule finds — once the cap is hit for a day, later bars that same day
    are skipped even if they're still inside the session window, but an
    already-open trade is never cut short by it.

    cost_pct: None (default, unchanged behavior — every trade's raw price
    move is kept whole) or a float (e.g. 0.0005 = 0.05% round-trip) — a
    real fill costs something (spread crossed on both the way in and the
    way out, plus commission); this doesn't change WHETHER a trade hits
    stop or target (that's still the raw resolved prices — a real order
    sits at that exact level regardless of cost), only what's left once it
    does. Each closed trade gets a `net_r` field: its raw R-multiple
    (reward/risk for a win, -1 for a loss — same shape backtest_engine.py
    already derives from entry/stop/target) minus `cost_pct * entry /
    risk`, i.e. the round-trip cost expressed in THIS trade's own R terms
    (a tight-stop trade pays more R for the same cost than a wide-stop
    one, same as in reality). `won`/`win_rate` deliberately stay untouched
    by this — they answer "did price mechanically reach the target,"
    which must mean the same thing whether or not a cost is applied, so
    sweeps at different cost_pct settings stay comparable on that number;
    net_r is where the honest, cost-adjusted economics live instead.

    news_blackout_mask: None (default, unchanged behavior) or a boolean
    array aligned to df.index (see news.blackout_mask) - same gating as
    `session` above (a NEW trade can't OPEN on a blacked-out bar; an
    already-open trade is still managed normally regardless), just keyed
    on high-impact news windows instead of time-of-day. The two compose:
    passing both means a bar must be in-session AND outside every
    blackout window before a new trade can open there.

    Returns {"trades": [{"entry_time","exit_time","entry","stop",
    "target","won","bars_held","is_long","net_r","entry_origin_time",
    "exit_origin_time","stop_origin_time"}], "n", "wins", "win_rate",
    "avg_net_r"}. The three "*_origin_time" fields are None whenever
    that leg's rule was anchored to "current_price" (nothing to trace
    back to) — otherwise the formation time of the zone/level that
    produced that leg's price (see resolve_rule's origin_out)."""
    o_col, h_col, l_col, c_col = (("Open", "High", "Low", "Close") if "Close" in df
                                   else ("open", "high", "low", "close"))
    high = df[h_col].to_numpy()
    low = df[l_col].to_numpy()
    close = df[c_col].to_numpy()
    n_bars = len(df)
    # See _build_detector_cache's own docstring: every detector a rule can
    # anchor to (fvg, order_block, liquidity) is an O(n) full-dataframe
    # scan — fine once, a genuine O(n^2) hang risk if re-run fresh at every
    # bar of the walk-forward loop below (confirmed directly: 11 SECONDS
    # for fvg/order_block on a real 4,334-bar combo before this cache
    # existed; the liquidity branch had the identical problem via
    # detect_swings, confirmed as a 33.7x slowdown on its own). This cache
    # is lazy — a rule that never anchors any of its three legs to, say,
    # "order_block" never pays for detect_order_blocks at all — but always
    # scans the TRUE full `df` regardless of which bar's resolve_rule call
    # first triggers a given detector's build.
    cache = _build_detector_cache(df, max_scan_bars=max_scan_bars)
    if session is not None:
        idx_ny = df.index.tz_convert("America/New_York")
        sess_start, sess_end = session
        # idx_ny.time is already a plain ndarray; idx_ny.weekday is a
        # pandas Index — combining the two via & yields a plain ndarray
        # too (confirmed directly), no further conversion needed.
        in_session = (idx_ny.time >= sess_start) & (idx_ny.time < sess_end) & (idx_ny.weekday < 5)
        day_of_bar = idx_ny.date if max_trades_per_session is not None else None
    else:
        in_session = None
        day_of_bar = None
    trades_today = {}
    trades = []
    in_trade = None
    # A resolved entry/exit/stop is a RESTING limit order, not an instant
    # fill — confirmed directly as a real bug: checking a batch of real
    # trades against their own entry bar's actual high/low showed 14 of 15
    # entries sitting OUTSIDE the bar's traded range entirely (e.g. a
    # "below price" entry resolved from the bar's close, then counted as
    # filled on that same bar even though its low never reached that
    # price). pending holds the current candidate while it waits; it only
    # becomes a real trade once a LATER bar's own high/low actually
    # crosses its entry price.
    pending = None
    i = 1
    while i < n_bars and len(trades) < max_signals:
        if in_trade is None:
            if in_session is not None and not in_session[i]:
                i += 1
                continue
            if news_blackout_mask is not None and news_blackout_mask[i]:
                i += 1
                continue
            if day_of_bar is not None:
                day = day_of_bar[i]
                if trades_today.get(day, 0) >= max_trades_per_session:
                    i += 1
                    continue

            # A candidate can only fill on a bar AFTER the one that
            # resolved it — its own resolving bar's close is what
            # produced this price, so that same bar's high/low has
            # already happened by the time a real trader could act on it.
            # Checked BEFORE today's re-resolve below, not after: a resting
            # order lives in the market regardless of what a fresh
            # recompute off today's own close would conclude. Confirmed
            # directly as a real bug in the other order: a bar whose price
            # action reaches the resting entry price often ALSO fully
            # consumes the zone that produced it (the same displacement
            # candle does both at once) — re-resolving first made that
            # candidate come back None, which wiped `pending` before the
            # fill-check ever ran, silently discarding a fill that
            # genuinely happened.
            if pending is not None and pending["since_i"] < i:
                pe, ps, px, p_is_long = pending["e"], pending["s"], pending["x"], pending["is_long"]
                if low[i] <= pe <= high[i]:
                    in_trade = {
                        "entry": pe, "stop": ps, "target": px, "entry_i": i, "is_long": p_is_long,
                        "entry_origin": pending.get("e_origin"), "exit_origin": pending.get("x_origin"),
                        "stop_origin": pending.get("s_origin"),
                    }
                    pending = None
                    if day_of_bar is not None:
                        trades_today[day_of_bar[i]] = trades_today.get(day_of_bar[i], 0) + 1

            if in_trade is None:
                # Re-resolved every bar (zones/ranking can shift as new
                # bars come in) off data through THIS bar's own close — the
                # exact setup a trader watching this rule would see right
                # now. Only replaces a still-pending order when the price
                # actually changed (a new/closer zone took over) — the
                # common case of the same zone resolving the same price
                # every bar must not keep resetting how long the order's
                # been resting.
                # df itself (not a per-bar slice) is passed below — with
                # `cache` always supplied in this loop, resolve_rule never
                # touches its own `df` argument at all (every detector
                # resolves through the cache, projected onto as_of_bar=i
                # instead); a growing df.iloc[:i+1] slice used to be needed
                # here before the cache existed; see resolve_rule's own
                # docstring on the cache/as_of_bar contract.
                price = float(close[i])
                e_origin, x_origin, s_origin = {}, {}, {}
                e = resolve_rule(entry_rule, df, price, origin_out=e_origin, cache=cache, as_of_bar=i)
                x = resolve_rule(exit_rule, df, price, origin_out=x_origin, cache=cache, as_of_bar=i)
                s = resolve_rule(stop_rule, df, price, origin_out=s_origin, cache=cache, as_of_bar=i)
                candidate = None
                if e is not None and x is not None and s is not None:
                    # bool(...) here (and on hit_stop/hit_target below) —
                    # confirmed directly as the cause of a real bug: high/
                    # low/close are numpy arrays, so a plain `x > e`-style
                    # comparison against one of their scalar elements
                    # yields numpy.bool_, not a native bool. `is_long`/
                    # `won` get stored straight into the trade dict this
                    # function returns, which the chart's Streamlit
                    # component eventually serializes to JSON — and
                    # numpy's bool scalar isn't JSON-serializable (surfaced
                    # as "Could not fetch <ticker>: ... TypeError('Object
                    # of type bool is not JSON serializable')", since
                    # NumPy 2.x's bool scalar type is literally named
                    # "bool", not "bool_" like older versions).
                    is_long = bool(x > e)
                    sane = (s < e < x) if is_long else (s > e > x)
                    if sane:
                        risk = abs(e - s)
                        reward = abs(x - e)
                        rr = (reward / risk) if risk > 0 else float("inf")
                        passes_rr = min_rr is None or rr >= min_rr
                        passes_stop_pct = min_stop_pct is None or (e > 0 and risk / e >= min_stop_pct)
                        passes_stop_abs = min_stop_abs is None or risk >= min_stop_abs
                        if passes_rr and passes_stop_pct and passes_stop_abs:
                            candidate = {
                                "e": e, "x": x, "s": s, "is_long": is_long,
                                "e_origin": e_origin.get("time"), "x_origin": x_origin.get("time"),
                                "s_origin": s_origin.get("time"),
                            }
                if candidate is None:
                    pending = None
                elif pending is None or candidate["e"] != pending["e"] or \
                        candidate["s"] != pending["s"] or candidate["x"] != pending["x"]:
                    pending = {**candidate, "since_i": i}
        else:
            t = in_trade
            if t["is_long"]:
                hit_stop = bool(low[i] <= t["stop"])
                hit_target = bool(high[i] >= t["target"])
            else:
                hit_stop = bool(high[i] >= t["stop"])
                hit_target = bool(low[i] <= t["target"])
            if hit_stop or hit_target:
                won = hit_target and not hit_stop
                risk = abs(t["entry"] - t["stop"])
                reward = abs(t["target"] - t["entry"])
                raw_r = (reward / risk) if won else -1.0
                net_r = raw_r - (cost_pct * t["entry"] / risk) if cost_pct else raw_r
                trades.append({
                    "entry_time": df.index[t["entry_i"]], "exit_time": df.index[i],
                    "entry": t["entry"], "stop": t["stop"], "target": t["target"],
                    "won": won, "bars_held": i - t["entry_i"], "is_long": t["is_long"],
                    "net_r": net_r,
                    "entry_origin_time": t.get("entry_origin"), "exit_origin_time": t.get("exit_origin"),
                    "stop_origin_time": t.get("stop_origin"),
                })
                in_trade = None
        i += 1
    wins = sum(1 for t in trades if t["won"])
    n_trades = len(trades)
    avg_net_r = (sum(t["net_r"] for t in trades) / n_trades) if n_trades else None
    return {"trades": trades, "n": n_trades, "wins": wins,
            "win_rate": (wins / n_trades) if n_trades else None, "avg_net_r": avg_net_r}


def rule_describe(rule):
    """Plain-English description of a rule — e.g. "Current price" or
    "2nd-nearest FVG above price, 50%" — for showing the user exactly
    what a built rule resolves to, in words, not the raw dict."""
    if rule["anchor"] == "current_price":
        return "Current price"
    ordinal = {1: "Nearest", 2: "2nd-nearest", 3: "3rd-nearest"}.get(rule.get("n", 1), f"{rule.get('n', 1)}th-nearest")
    detector_label = RULE_DETECTOR_LABELS[rule["detector"]]
    side_label = rule.get("side", "above")
    desc = f"{ordinal} {detector_label} {side_label} price"
    if rule["detector"] != "liquidity":
        desc += f", {zone_point_label(rule.get('zone_point', 0.5))}"
    return desc
