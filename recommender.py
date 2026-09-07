"""
"Best trade right now" for whatever ticker/timeframe the ICT Terminal is
currently showing — the one thing the rest of this project's live-setup
pipeline never actually does. research/signals.py already turns a detected
zone into a concrete entry/stop/target/status box and looks up whether
Edge Lab has ever proven a pattern family works (Benjamini-Hochberg
correction + held-up holdout, GBPUSD=X only); research/live_scan.py +
signals_app.py already do that across a fixed watchlist on an hourly cron.
Nothing ranks candidates against each other or ties any of it to the
ticker actually on screen right now — that's what this module adds, not a
new detection layer or a new entry/stop/target formula.

Presentation-agnostic like research/signals.py itself — app.py owns all
the Streamlit rendering, this module owns only the candidate-building and
ranking, so it stays usable (and testable) without a running Streamlit
process.

Ranking is deliberately validation-first: Edge Lab's pass/fail is the only
statistically proven signal anywhere in this project, and it only ever
covers GBPUSD=X. A "confluence" score — how many of the OTHER currently-
active ICT reads agree with a candidate's own direction — is a
discretionary tiebreaker on top of that, not a second proof. For most
tickers (anything that isn't GBPUSD=X) there will never be a validated
candidate at all; ranking still works, it just never fools itself into
calling a confluence-only pick "proven."
"""

import numpy as np
import pandas as pd

from fvg import (
    current_dealing_range,
    detect_fvgs,
    detect_liquidity_levels,
    detect_liquidity_reactions,
    detect_liquidity_sweeps,
    detect_order_blocks,
    detect_structure_breaks,
    detect_swings,
)
from indicators import ema, macd, rsi
from research.signals import (
    EDGE_LAB_TICKER,
    EVENT_TYPE_TO_HYPOTHESIS,
    compute_setup,
    load_edge_lab_validation,
)

EVENT_TYPE_LABELS = {"fvg": "FVG", "order_block": "Order Block", "liquidity_reaction": "Liquidity Reaction",
                      "ma_fvg": "MA+FVG"}

# The MA+FVG strategy's own fast/slow pairing — a standard, widely
# recognized combination, not tuned per-ticker. Kept as a module constant
# rather than a parameter threaded through best_trade_now/scan_watchlist,
# since every caller wants the same two periods; app.py/crypto_app.py
# import this directly too, so the periods drawn on the chart and the
# periods checked for confluence can never silently drift apart.
MA_FVG_PERIODS = (20, 50)


def _validation_rank(event_type, ticker):
    """0-3, mirroring research.signals.validation_badge's own lookup (same
    data, same meaning) but as an orderable number instead of display text
    — a small parallel helper rather than changing validation_badge's
    existing (label, tone) return contract that signals_app.py already
    relies on. 3 = validated (BH-significant AND holdout passed), 2 =
    train-significant only, 1 = tested but nothing significant yet, 0 =
    Edge Lab has never touched this ticker at all."""
    if ticker != EDGE_LAB_TICKER:
        return 0
    hyp = EVENT_TYPE_TO_HYPOTHESIS.get(event_type)
    v = load_edge_lab_validation().get(hyp)
    if v is None:
        return 1
    return 3 if v["validated"] else 2


def _confluence_score(direction, entry_price, df):
    """0-4 tally of how many OTHER currently-active ICT reads agree with a
    candidate's own direction — reusing the exact detectors app.py's live
    chart already draws, not a new indicator. A tiebreaker, not a proof;
    see this module's own docstring."""
    score = 0

    dr = current_dealing_range(df)
    if dr is not None:
        if (direction == "bullish" and entry_price < dr["eq"]) or \
           (direction == "bearish" and entry_price > dr["eq"]):
            score += 1

    breaks = detect_structure_breaks(df)
    if breaks and breaks[-1]["type"] == direction:
        score += 1

    obs = detect_order_blocks(df)
    if any(not o["mitigated"] and o["type"] == direction for o in obs):
        score += 1

    above, below = detect_liquidity_levels(df)
    if (direction == "bullish" and above) or (direction == "bearish" and below):
        score += 1

    return score


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


def _zone_at(zone, j):
    """Reconstructs what a fresh detect_fvgs/detect_order_blocks(df.iloc
    [:j+1], ...) call would have said about THIS exact zone at bar j,
    using its recorded "history" (see detect_fvgs' own record_history
    docstring) instead of re-scanning. Returns None when the zone doesn't
    exist yet as of bar j — its own confirming/breakout candle (
    "formation_i") hasn't closed yet, so a live trader watching bar j
    wouldn't have seen it at all.

    Zones only ever shrink (consequent encroachment is monotonic — see
    detect_fvgs' own clamp comment), so "state as of j" is always exactly
    the last recorded checkpoint at or before j, or the untouched raw
    bounds if price hasn't reached it yet by j. This is what turns the
    backtest engine's walk-forward loop from re-detecting every zone from
    scratch at every bar (confirmed directly: 11 seconds on a real 4,334-
    bar combo) into one detection pass plus a cheap per-bar lookup."""
    if j < zone["formation_i"]:
        return None
    # raw_top/raw_bottom (FVG) or _formation_top/_formation_bottom (Order
    # Block, which has no raw_* fields — see detect_order_blocks' own
    # comment) — either way, this zone's bounds before anything ever
    # touched it. Deliberately NOT included in the returned dict below
    # under the raw_top/raw_bottom names for an Order Block: _zone_bounds
    # (pricing) keys specifically off those names, and OB pricing was
    # never changed to use pre-encroachment bounds the way FVG's was —
    # spreading `zone` as-is preserves that distinction automatically,
    # since only FVG zones carry raw_top/raw_bottom under those names.
    active_top = zone.get("raw_top", zone.get("_formation_top"))
    active_bottom = zone.get("raw_bottom", zone.get("_formation_bottom"))
    for pos, top_at, bottom_at in zone["history"]:
        if pos > j:
            break
        active_top, active_bottom = top_at, bottom_at
    filled = active_top <= active_bottom
    expiry_bar = zone["expiry_bar"]
    expired = (not filled) and (expiry_bar is not None) and (j >= expiry_bar)
    return {**zone, "top": active_top, "bottom": active_bottom, "filled": filled, "expired": expired}


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


def best_trade_now(df, ticker, interval, provider, top_n=3, event_types=None, direction=None, min_confluence=0):
    """The top `top_n` currently-active candidates for `ticker` at `interval`
    (already-fetched `df`, e.g. app.py's own main chart data — this does no
    fetching of its own), ranked (validation_rank, confluence_score,
    closest-to-entry) descending. `interval` is a plain fetch-interval
    string ("1d", "60m", ...) — matched against research.signals.LOOKBACK
    for how long a setup stays "live"; an interval that dict doesn't
    recognize just falls back to its own generic default, not an error.
    Returns [] when nothing currently active resolves — an honest empty
    result, not a forced pick.

    event_types/direction/min_confluence: optional pre-ranking filters —
    "only consider FVG+MA-FVG candidates," "bullish only," "confluence >=
    2" — for a user who wants to steer WHICH pattern the pick comes from
    rather than just seeing whatever the ranking alone would surface.
    None/0 (the defaults) mean unfiltered, byte-identical to this
    function's behavior before these params existed — scan_watchlist below
    never passes them, so the sidebar scan stays exactly as unfiltered as
    it's always been."""
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
        setup["confluence_score"] = _confluence_score(setup["direction"], setup["entry_price"], df)
        if setup["confluence_score"] < min_confluence:
            continue
        setup["validation_rank"] = _validation_rank(event_type, ticker)
        ranked.append(setup)

    ranked.sort(key=lambda s: (s["validation_rank"], s["confluence_score"], -abs(s["distance_pct"])), reverse=True)
    return ranked[:top_n]


def scan_watchlist(dfs_by_ticker, interval, top_n=8):
    """The multi-symbol version of best_trade_now — each ticker's own single
    best candidate (best_trade_now(..., top_n=1)), combined across every
    ticker in `dfs_by_ticker` ({ticker: already-fetched df}) and ranked
    against each other the same way (validation_rank, confluence_score,
    closest-to-entry). For a sidebar/watchlist view: "which symbol has the
    best-supported setup right now," not "what's the best setup on the one
    symbol already on screen."

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
        ranked.extend(best_trade_now(df, ticker, interval, provider=None, top_n=1))
    ranked.sort(key=lambda s: (s["validation_rank"], s["confluence_score"], -abs(s["distance_pct"])), reverse=True)
    return ranked[:top_n]


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


def _score_events(df, entries, exit_positions, min_events):
    """entries: [(entry_pos, direction), ...]. Executes one bar after each
    entry_pos (that bar's open), holds until the next exit_positions entry
    (that bar's close) — raw return, sign-flipped for bearish so a "win"
    always means price moved the expected way. An entry with no exit event
    left in the dataset (ran off the end) is dropped, same as the fixed-bar
    version dropping an entry with insufficient forward bars — an
    unresolved trade isn't a scoreable one either way.
    Returns {"bullish": (rate, n, sufficient), "bearish": (...)}."""
    open_ = (df["Open"] if "Open" in df else df["open"]).to_numpy()
    close = (df["Close"] if "Close" in df else df["close"]).to_numpy()
    n_bars = len(df)
    by_dir = {"bullish": [], "bearish": []}
    for entry_pos, direction in entries:
        exec_pos = entry_pos + 1
        if exec_pos >= n_bars:
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


def fvg_event_win_rate(df, exit_types=("fvg", "order_block", "liquidity"), min_events=20):
    """FVG win rate under the event-driven exit rule (see module comment
    above) — touch = first_touch (unchanged from the fixed-bar version),
    exit = the next chosen event type forming."""
    idx = df.index
    pos_by_time = {t: i for i, t in enumerate(idx)}
    entries = [(pos_by_time[g["first_touch"]], g["type"])
               for g in detect_fvgs(df) if g["first_touch"] is not None]
    return _score_events(df, entries, _exit_positions(df, set(exit_types)), min_events)


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


def order_block_event_win_rate(df, exit_types=("fvg", "order_block", "liquidity"), min_events=20):
    """Order Block win rate under the event-driven exit rule — touch =
    first_touch (unchanged from the fixed-bar version), exit = the next
    chosen event type forming."""
    idx = df.index
    pos_by_time = {t: i for i, t in enumerate(idx)}
    entries = [(pos_by_time[ob["first_touch"]], ob["type"])
               for ob in detect_order_blocks(df) if ob["first_touch"] is not None]
    return _score_events(df, entries, _exit_positions(df, set(exit_types)), min_events)


def liquidity_event_win_rate(df, exit_types=("fvg", "order_block", "liquidity"), min_events=20):
    """External-range-liquidity win rate under the event-driven exit rule
    — touch = the sweep itself (direction convention matches
    detect_liquidity_sweeps' own `type`: sweeping a high points bearish,
    sweeping a low points bullish), exit = the next chosen event type
    forming."""
    idx = df.index
    pos_by_time = {t: i for i, t in enumerate(idx)}
    entries = [(pos_by_time[sweep["end"]], sweep["type"]) for sweep in detect_liquidity_sweeps(df)]
    return _score_events(df, entries, _exit_positions(df, set(exit_types)), min_events)


def ma_fvg_event_win_rate(df, exit_types=("fvg", "order_block", "liquidity"), min_events=20, periods=MA_FVG_PERIODS):
    """MA+FVG win rate — the one case where "touch" genuinely means
    something different from plain FVG: entry is the first bar an EMA
    (MA_FVG_PERIODS) actually reaches the gap's own ORIGINAL [raw_bottom,
    raw_top] range while that gap is still open (between its confirmed
    formation and its recorded fill point, or the last bar if it never
    filled) — "the indicator's own level," not "price touched anywhere in
    the zone." Exit = the next chosen event type forming, same as every
    other event-driven win rate here."""
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
    return _score_events(df, entries, _exit_positions(df, set(exit_types)), min_events)


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
# backtest_app.py's sliders build their step options directly from it.
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

def _resolve_fvg_cached(raw, rule, current_price, as_of_bar, origin_out):
    side = rule.get("side", "above")
    zones = [z for z in (_zone_at(g, as_of_bar) for g in raw) if z is not None and _zone_is_open(z)]
    z = _nth_zone_on_side(zones, current_price, side, rule.get("n", 1))
    if z is not None and origin_out is not None:
        origin_out["time"] = z["start"]
    return _zone_point_price(z, rule.get("zone_point", 0.5), side) if z else None


def _resolve_order_block_cached(raw, rule, current_price, as_of_bar, origin_out):
    side = rule.get("side", "above")
    zones = [z for z in (_zone_at(o, as_of_bar) for o in raw) if z is not None and _zone_is_open(z)]
    z = _nth_zone_on_side(zones, current_price, side, rule.get("n", 1))
    if z is not None and origin_out is not None:
        origin_out["time"] = z["start"]
    return _zone_point_price(z, rule.get("zone_point", 0.5), side) if z else None


def _resolve_liquidity_cached(raw, rule, current_price, as_of_bar, origin_out):
    j = as_of_bar
    high_arr, low_arr = raw["high_arr"], raw["low_arr"]

    def _untouched_high(point):
        after = high_arr[point["pos"] + 1: j + 1]
        return after.size == 0 or bool((after < point["price"]).all())

    def _untouched_low(point):
        after = low_arr[point["pos"] + 1: j + 1]
        return after.size == 0 or bool((after > point["price"]).all())

    # Confirmed directly this is lookahead-safe, not just faster:
    # detect_swings is a single forward-only pass whose state at bar i
    # depends only on bars 0..i, so filtering the FULL run's own output by
    # confirmed_pos <= j is bit-for-bit identical to what a fresh
    # detect_swings(df.iloc[:j+1]) call would have produced — same swings,
    # same confirmation bars, just computed once instead of redundantly at
    # every bar (see detect_swings' own pos/confirmed_pos docstring).
    live_highs = sorted(
        (h for h in raw["highs"]
         if h["confirmed_pos"] <= j and h["price"] > current_price and _untouched_high(h)),
        key=lambda h: h["price"])
    live_lows = sorted(
        (l for l in raw["lows"]
         if l["confirmed_pos"] <= j and l["price"] < current_price and _untouched_low(l)),
        key=lambda l: l["price"], reverse=True)
    levels = live_highs if rule.get("side", "above") == "above" else live_lows
    n = rule.get("n", 1)
    if len(levels) < n:
        return None
    if origin_out is not None:
        origin_out["time"] = levels[n - 1]["time"]
    return levels[n - 1]["price"]


_DETECTOR_BUILDERS = {
    "fvg": lambda full_df, max_scan_bars: detect_fvgs(full_df, max_scan_bars=max_scan_bars, record_history=True),
    "order_block": lambda full_df, max_scan_bars: detect_order_blocks(full_df, max_scan_bars=max_scan_bars,
                                                                        record_history=True),
    "liquidity": lambda full_df, max_scan_bars: (lambda highs, lows: {
        "highs": highs, "lows": lows,
        "high_arr": (full_df["High"] if "High" in full_df else full_df["high"]).to_numpy(),
        "low_arr": (full_df["Low"] if "Low" in full_df else full_df["low"]).to_numpy(),
    })(*detect_swings(full_df)),
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


def backtest_custom_rule(df, entry_rule, exit_rule, stop_rule, max_signals=100, max_scan_bars=500,
                          session=None, min_rr=None, min_stop_pct=None, min_stop_abs=None,
                          max_trades_per_session=None, cost_pct=None):
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
                    is_long = x > e
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
                hit_stop = low[i] <= t["stop"]
                hit_target = high[i] >= t["target"]
            else:
                hit_stop = high[i] >= t["stop"]
                hit_target = low[i] <= t["target"]
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
