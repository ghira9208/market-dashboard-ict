"""
This project's whole ICT detection library — every pattern/zone/level
detector the live charts, the recommender, and the research pipeline share,
not just Fair Value Gaps (this file was named fvg.py originally, when FVG
detection was the only thing in it; the name stuck through everything else
that got added alongside it, which stopped being accurate a long time ago —
renamed to detectors.py to actually match what's here):

  - detect_fvgs / detect_order_blocks: the two core imbalance/footprint
    zone detectors (Fair Value Gaps, Order Blocks).
  - detect_swings / detect_structure_breaks: swing-point and market-
    structure-break (BOS/CHoCH) detection.
  - detect_equal_levels / detect_equal_highs_lows / detect_liquidity_levels /
    detect_liquidity_sweeps / detect_liquidity_reactions: resting-liquidity
    and liquidity-sweep detection.
  - detect_naked_pocs / poc_migration / detect_poor_highs_lows: volume-
    profile-derived detectors (built on indicators.volume_profile).
  - current_dealing_range: premium/discount range.
  - recent_zone_tracker / historical_zone_scanner / merge_zone_engines: the
    two-engine zone-selection layer (recency-based vs. distance-based) that
    picks which of the above zones actually get drawn/considered at once.
"""

import numpy as np
import streamlit as st

from indicators import atr, volume_profile

# Detection here is O(n) but with per-row pandas access, not free on a
# multi-thousand-row df — and several of these are called more than once per
# FVG-page render (detect_swings alone up to 4x: Swing Points, Equal Highs/
# Lows, and again inside detect_liquidity_levels/detect_liquidity_reactions).
# Caching means repeat calls with the same df — within one render AND across
# Streamlit reruns triggered by unrelated widgets — are a hash lookup instead
# of a recompute. TTL matches get_yf_ohlcv's, so it tracks fresh data.
_CACHE_TTL = 300

# What this actually measures, no more: body size as a fraction of the
# candle's own high-low range — nothing here observes conviction, intent,
# or who was trading. ICT's own definition of displacement INTERPRETS a
# candle clearing this bar as "real, fast, mostly-one-direction" — body
# dominating its range, not a long-wicked candle that merely closed past a
# level — but that reading lives in the ICT methodology, not in the number
# itself. Without this gate, both FVG and order-block detection below treat
# ANY 3-candle gap or ANY close-beyond-the-prior-high as valid, which in
# practice fires on plenty of choppy candles a body-ratio filter would
# reject. 0.5 is deliberately looser than the ~0.75 "textbook strong
# displacement" bar ICT itself would use — strict enough to drop the
# clearest noise, loose enough not to silently empty out every FVG/OB on
# calmer timeframes. Chosen as a reasonable engineering threshold, not
# something back-tested to correlate with any actual forward outcome — see
# research/evidence.py for that separate, much harder question.
DISPLACEMENT_MIN_BODY_RATIO = 0.5


def _is_displacement(o, h, l, c, min_ratio=DISPLACEMENT_MIN_BODY_RATIO):
    rng = h - l
    if rng <= 0:
        return False
    return abs(c - o) / rng >= min_ratio


@st.cache_data(ttl=_CACHE_TTL)
def detect_fvgs(df, min_body_ratio=DISPLACEMENT_MIN_BODY_RATIO, max_scan_bars=None, record_history=False):
    # Fair Value Gap — the classic 3-candle imbalance:
    #
    #   candle[i-1]        candle[i]        candle[i+1]
    #   (leaves the gap)   (impulse move)   (confirms the gap)
    #
    # Bullish FVG: low of candle[i+1] > high of candle[i-1] — price left a
    #              void between those two candles that candle[i] jumped
    #              straight over.
    # Bearish FVG: high of candle[i+1] < low of candle[i-1] — same thing,
    #              downward.
    #
    # The gap "fills" (mitigates) the first time a later candle trades back
    # into its price zone. Open gaps (never filled) are the ones price
    # hasn't returned to yet — usually the more interesting ones to watch.
    #
    # record_history=False (default, every existing call site) returns the
    # exact same dict shape as always — this parameter is purely additive.
    # =True (recommender.py's backtest engine only) additionally records
    # each zone's own fill trajectory: (bar_pos, active_top, active_bottom)
    # after every bar that touched it, plus "formation_i" (the confirming
    # candle's own position — the earliest bar this zone can be resolved
    # against at all). Together these let a caller reconstruct "what would
    # a fresh detect_fvgs(df.iloc[:j+1]) call have returned for this exact
    # zone" for ANY bar j, by taking the last checkpoint at or before j
    # (or the untouched raw bounds if none yet) — without re-running the
    # scan. See backtest_custom_rule's own docstring for why that matters:
    # walking a backtest bar by bar used to re-detect every zone from
    # scratch on a growing slice at EVERY bar (confirmed directly: 11
    # seconds on a real 4,334-bar combo) — an O(n) scan repeated n times.
    # Computing the trajectory ONCE here and querying it cheaply per bar
    # is the fix; this function's own formation/fill/expiry logic is
    # otherwise completely unchanged; a caller that never asks for history
    # can't tell the difference.
    #
    # min_body_ratio is a parameter (not just the module constant baked in
    # directly) so the research pipeline can sweep displacement strictness
    # to see whether a signal's edge (or lack of one) is sensitive to this
    # threshold — the live chart's own call site never passes it, so it
    # keeps using DISPLACEMENT_MIN_BODY_RATIO exactly as before.
    #
    # max_scan_bars bounds the inner fill-scan loop below — None (the live
    # chart's own call site; unchanged behavior) scans all the way to the
    # end of df like before. Confirmed directly: on a 620k-row backtest
    # dataset (25 years of GBPUSD resampled to 15m) an unbounded scan never
    # finished in a reasonable time — thousands of gaps each potentially
    # rescanning hundreds of thousands of rows is genuinely O(n^2), the
    # same class of blowup as the Plotly add_shape-in-a-loop bug from
    # earlier in this project's history, just one layer down the stack.
    # research/events.py passes a real cap; the live chart's own
    # OVERLAY_LOOKBACK_DAYS window is already small enough never to hit this.
    open_arr = (df["Open"] if "Open" in df else df["open"]).to_numpy()
    high_arr = (df["High"] if "High" in df else df["high"]).to_numpy()
    low_arr = (df["Low"] if "Low" in df else df["low"]).to_numpy()
    close_arr = (df["Close"] if "Close" in df else df["close"]).to_numpy()
    idx = df.index
    n = len(df)
    fvgs = []

    for i in range(1, n - 1):
        prev_high, prev_low = high_arr[i - 1], low_arr[i - 1]
        next_high, next_low = high_arr[i + 1], low_arr[i + 1]

        # The gap is left behind by candle i itself moving through it — that's
        # the displacement candle whose conviction actually matters, not the
        # confirming candle (i+1) that just happens to close beyond it.
        if not _is_displacement(open_arr[i], high_arr[i], low_arr[i], close_arr[i], min_body_ratio):
            continue

        if next_low > prev_high:
            kind, bottom, top = "bullish", prev_high, next_low
        elif next_high < prev_low:
            kind, bottom, top = "bearish", next_high, prev_low
        else:
            continue

        # A bullish gap formed on an upmove, so price re-tests it from ABOVE,
        # moving down — each wick that clips into it eats the zone away from
        # the top, not the whole thing at once ("consequent encroachment": a
        # gap half-filled by one wick still has a live untested half). A
        # bearish gap mirrors this from below. Only once the eaten-into
        # boundary crosses the other side is it actually fully filled.
        active_bottom, active_top = bottom, top
        end_i = n - 1
        first_touch_i = None  # first bar that trades back into the zone at all,
        # distinct from end_i (which only gets set once the zone is FULLY eaten
        # away) — research/events.py needs "price just retraced into this gap"
        # as its own event, separate from "this gap is now fully mitigated".
        history = [] if record_history else None
        scan_end = n if max_scan_bars is None else min(n, i + 2 + max_scan_bars)
        for j in range(i + 2, scan_end):  # i+1 is the confirming candle itself, can't fill its own gap
            lo_j, hi_j = low_arr[j], high_arr[j]
            if lo_j <= active_top and hi_j >= active_bottom:
                if first_touch_i is None:
                    first_touch_i = j
                # Clamp rather than let a deep wick push the eaten-into edge
                # past the opposite side — that's still just "fully filled",
                # not an inverted zone.
                if kind == "bullish":
                    active_top = max(min(active_top, lo_j), active_bottom)
                else:
                    active_bottom = min(max(active_bottom, hi_j), active_top)
                if record_history:
                    history.append((j, active_top, active_bottom))
                if active_top <= active_bottom:
                    end_i = j
                    break
        filled = active_top <= active_bottom
        # scan_end < n only happens when max_scan_bars actually cut the
        # scan short of the real end of the data — a gap near the end of
        # df needs no cap to reach n anyway, so that case still correctly
        # reads as "genuinely still open as of the last available bar,"
        # not expired. Still not filled AND the window ran out before
        # finding out either way = give up calling this one open — a
        # capped scan silently meaning "permanently open forever after"
        # was the actual bug: a zone whose fill would've taken one bar
        # longer than max_scan_bars stayed a valid, resolvable candidate
        # for the rest of a walk-forward backtest, long after any real
        # trader would have considered the setup dead.
        expired = (not filled) and (scan_end < n)

        fvg_dict = {
            "type": kind, "top": active_top, "bottom": active_bottom,
            "start": idx[i - 1], "end": idx[end_i], "filled": filled, "expired": expired,
            "first_touch": idx[first_touch_i] if first_touch_i is not None else None,
            # The zone as it ORIGINALLY formed, before any consequent-
            # encroachment eating — "top"/"bottom" above are the FINAL,
            # fully-processed state (often clamped down to near-zero height
            # by the time a gap that fills fast finishes its scan), which is
            # right for the live chart's own eaten-away visualization but
            # wrong for anything drawing "the gap" as the size a trader
            # actually saw at formation — confirmed directly: research/
            # events.py's example chart was shading a rectangle with a
            # height of 0.00002 for a gap that had visibly filled by first
            # touch, effectively invisible.
            "raw_top": top, "raw_bottom": bottom,
        }
        if record_history:
            fvg_dict["history"] = history
            fvg_dict["formation_i"] = i + 1
            fvg_dict["expiry_bar"] = (i + 2 + max_scan_bars) if max_scan_bars is not None else None
        fvgs.append(fvg_dict)

    return fvgs


@st.cache_data(ttl=_CACHE_TTL)
def detect_ifvgs(df, min_body_ratio=DISPLACEMENT_MIN_BODY_RATIO, max_scan_bars=None):
    """Inversion Fair Value Gap — an FVG that gets fully mitigated WITH
    real conviction (the candle that finishes eating it all the way through
    also CLOSES beyond the gap's own original far edge, not just wicks
    through it) flips role: a bullish FVG broken this way stops acting as
    support and starts acting as resistance; a bearish FVG broken this way
    starts acting as support. Same price range as the original gap (its
    own formation-time raw_top/raw_bottom, not detect_fvgs' own final
    "top"/"bottom", which is often eaten down to a near-zero sliver by the
    time it fully fills — see detect_fvgs' own raw_top/raw_bottom
    docstring note for why that distinction already exists), just a
    flipped interpretation and a fresh forward scan for whether price
    actually comes back to test it FROM THE NEW DIRECTION.

    Deliberately a separate detector, not a flag bolted onto detect_fvgs'
    own output — an IFVG's own lifecycle (does price respect the flip) is
    a different question from the original FVG's (did it get filled at
    all), with its own first_touch meaning relative to the INVERTED
    direction, not the original one. Mirrors detect_breaker_blocks below,
    same mechanism applied to detect_order_blocks instead.

    "Closes beyond," checked on the single bar that completes the fill —
    not "wicks beyond at any later point" — is an engineering choice, not
    something back-tested to be the one true definition; see this file's
    own DISPLACEMENT_MIN_BODY_RATIO comment on the same kind of choice for
    displacement itself."""
    base = detect_fvgs(df, min_body_ratio=min_body_ratio, max_scan_bars=max_scan_bars, record_history=True)
    if not base:
        return []
    high_arr = (df["High"] if "High" in df else df["high"]).to_numpy()
    low_arr = (df["Low"] if "Low" in df else df["low"]).to_numpy()
    close_arr = (df["Close"] if "Close" in df else df["close"]).to_numpy()
    idx = df.index
    n = len(df)
    pos_by_time = {t: i for i, t in enumerate(idx)}

    ifvgs = []
    for g in base:
        if not g["filled"]:
            continue
        end_i = pos_by_time[g["end"]]
        if g["type"] == "bullish":
            # Filled from above (bullish gap re-tested downward) — inverts
            # only if the closing bar actually closes BELOW the gap's own
            # original bottom, not just wicks into/through it.
            if close_arr[end_i] >= g["raw_bottom"]:
                continue
            new_type = "bearish"
        else:
            if close_arr[end_i] <= g["raw_top"]:
                continue
            new_type = "bullish"

        top, bottom = g["raw_top"], g["raw_bottom"]
        first_touch_i = None
        scan_end = n if max_scan_bars is None else min(n, end_i + 1 + max_scan_bars)
        for j in range(end_i + 1, scan_end):
            if low_arr[j] <= top and high_arr[j] >= bottom:
                first_touch_i = j
                break
        ifvgs.append({
            "type": new_type, "top": top, "bottom": bottom,
            "start": g["end"], "origin_start": g["start"],
            "first_touch": idx[first_touch_i] if first_touch_i is not None else None,
        })
    return ifvgs


def _rolling_mean_min1(arr, period):
    """Simple rolling mean over `period`, min_periods=1 (a shorter, noisier
    average at the very start rather than NaN/undefined) — pure numpy via
    the cumsum-difference trick, no pandas Series needed just for this."""
    n = len(arr)
    csum = np.cumsum(arr, dtype=float)
    out = np.empty(n, dtype=float)
    head = min(period, n)
    out[:head] = csum[:head] / np.arange(1, head + 1)
    if n > period:
        out[period:] = (csum[period:] - csum[:n - period]) / period
    return out


@st.cache_data(ttl=_CACHE_TTL)
def detect_swings(df, atr_period=14, atr_mult=1.5):
    """ZigZag-style swing highs/lows: a confirmed swing high (low) is the
    running extreme since the last confirmed pivot, locked in only once
    price reverses against it by at least `atr_mult` times the rolling ATR
    — not a fixed-bar-count fractal. Two consequences every caller of this
    function relies on (detect_structure_breaks, detect_equal_levels,
    detect_liquidity_levels, detect_liquidity_sweeps,
    detect_liquidity_reactions, current_dealing_range, and both Swing
    Points/Equal Highs-Lows rendering in app.py only ever read the
    returned {"time","price","pos"} dicts, so none of them need to change
    for this):

    Every point carries TWO timestamps, and they mean different things —
    `time`/`pos` is WHERE the pivot itself sits (the actual extreme
    candle), `confirmed_time`/`confirmed_pos` is WHEN a reversal of at
    least `atr_mult`×ATR made it knowable as a swing at all. A swing low
    isn't identified until price has already rallied back away from it —
    by definition confirmed_pos is always later than pos, often many bars
    later on a strong move. Display code (drawing a level at the price it
    actually formed) wants `pos`; any code deciding whether a swing was
    ALREADY KNOWN at some earlier bar i — a pending level a break can fire
    against, a dealing-range boundary, a sweep's major/minor tier — must
    key off `confirmed_pos` instead. Using `pos` for that is a lookahead
    bug: it lets bar i react to a swing that, at bar i, hadn't reversed
    enough to be confirmed yet. Confirmed directly this was live in
    detect_structure_breaks, current_dealing_range, and
    detect_liquidity_sweeps' tier classification — each built a
    pos-keyed "pending" lookup, meaning a break/range/tier could act on a
    swing before the market had actually shown enough reversal to
    establish it, which is how a level ends up looking like it's
    referencing a point in the middle of an unrelated trend instead of
    the swing it's actually anchored to.

    1. Swings STRICTLY ALTERNATE — a high is always followed by a low and
       vice versa, by construction (each bar only ever extends the
       CURRENTLY-tracked side's candidate or confirms-and-flips to the
       other, never both in the same bar), not as a post-filter bolted
       onto a noisier detector.
    2. ATR-relative reversal size scales with this ticker's own recent
       volatility AND this timeframe's own bar size — a fixed percentage
       or fixed-bar-count fractal can't do that on its own; what counts as
       "significant" on a 1m chart and a 1D chart are wildly different in
       raw price terms.

    Replaces the old strict 2-bar fractal (a candle whose high/low was the
    exact max/min of its immediate neighbors) — confirmed that one let
    consecutive highs (or lows) with no intervening opposite point through
    constantly, and reset market-structure/dealing-range state almost
    every bar on trending or coarse (1W/1M/1Y) data, which is why
    Premium/Discount's "not enough swing history yet" fired so often
    there — ATR-scaled confirmed swings persist through a trend instead of
    resetting on every new high/low."""
    high = (df["High"] if "High" in df else df["high"]).to_numpy(dtype=float)
    low = (df["Low"] if "Low" in df else df["low"]).to_numpy(dtype=float)
    close = (df["Close"] if "Close" in df else df["close"]).to_numpy(dtype=float)
    idx = df.index
    n = len(df)
    highs, lows = [], []
    if n < 2:
        return highs, lows

    prev_close = np.concatenate(([close[0]], close[:-1]))
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = _rolling_mean_min1(tr, atr_period)

    # direction: 0 = undetermined (before the first pivot ever confirms,
    # tracking both a candidate high AND low at once), 1 = tracking up
    # (looking for the next high), -1 = tracking down (looking for the
    # next low).
    direction = 0
    cand_hi_price, cand_hi_pos = high[0], 0
    cand_lo_price, cand_lo_pos = low[0], 0

    for i in range(1, n):
        threshold = atr_mult * atr[i]
        if direction == 0:
            if high[i] > cand_hi_price:
                cand_hi_price, cand_hi_pos = high[i], i
            if low[i] < cand_lo_price:
                cand_lo_price, cand_lo_pos = low[i], i
            if cand_hi_price - low[i] >= threshold:
                highs.append({"time": idx[cand_hi_pos], "price": float(cand_hi_price), "pos": int(cand_hi_pos),
                              "confirmed_time": idx[i], "confirmed_pos": int(i)})
                direction = -1
                cand_lo_price, cand_lo_pos = low[i], i
            elif high[i] - cand_lo_price >= threshold:
                lows.append({"time": idx[cand_lo_pos], "price": float(cand_lo_price), "pos": int(cand_lo_pos),
                             "confirmed_time": idx[i], "confirmed_pos": int(i)})
                direction = 1
                cand_hi_price, cand_hi_pos = high[i], i
        elif direction == 1:
            if high[i] > cand_hi_price:
                cand_hi_price, cand_hi_pos = high[i], i
            elif cand_hi_price - low[i] >= threshold:
                highs.append({"time": idx[cand_hi_pos], "price": float(cand_hi_price), "pos": int(cand_hi_pos),
                              "confirmed_time": idx[i], "confirmed_pos": int(i)})
                direction = -1
                cand_lo_price, cand_lo_pos = low[i], i
        else:
            if low[i] < cand_lo_price:
                cand_lo_price, cand_lo_pos = low[i], i
            elif high[i] - cand_lo_price >= threshold:
                lows.append({"time": idx[cand_lo_pos], "price": float(cand_lo_price), "pos": int(cand_lo_pos),
                             "confirmed_time": idx[i], "confirmed_pos": int(i)})
                direction = 1
                cand_hi_price, cand_hi_pos = high[i], i

    return highs, lows


@st.cache_data(ttl=_CACHE_TTL)
def detect_structure_breaks(df, mode="close", atr_period=14, atr_mult=1.5):
    """Market structure shifts — BOS (Break of Structure) and CHoCH (Change
    of Character), the primitive most other ICT reads implicitly lean on
    ("is this pullback likely to continue or reverse" needs a trend bias to
    answer against).

    Tracks the most recent not-yet-broken swing high and swing low
    separately. Whether that break is labeled BOS or CHoCH depends on the
    CURRENT bias, not the break's own direction alone:
      - breaking in the direction that matches the current bias = BOS
        (the trend just confirmed itself again)
      - breaking AGAINST the current bias = CHoCH (the first break beyond
        the level that was holding the old trend together — the classic
        early reversal signal) — and bias flips to the new direction
    The very first break of the series has no prior bias to compare
    against, so it's always reported as a BOS establishing that bias.

    Once a pending swing point is broken it's consumed — the next break in
    that direction needs a NEWER swing point to have formed first, exactly
    mirroring how a real chart reader would only care about the latest
    reference, not a stale one three swings back.

    mode="close" (default, UNCHANGED from before this parameter existed): a
    break fires on a candle's CLOSE beyond the pending level — a decisive,
    committed break. This is "Agresiv" confirmation in the trader's own
    taxonomy (body-close-based MSS). mode="wick": a break fires the moment
    a WICK (High/Low) trades beyond the level, without waiting for the
    close — "Normal" confirmation. Every emitted break dict now carries an
    additive "mss_strength" field ("agresiv" or "normal") naming which rule
    fired it, so existing callers that ignore the field see byte-for-byte
    identical output to before (same default mode, same break timing, same
    keys plus one new one)."""
    close = df["Close"] if "Close" in df else df["close"]
    high = df["High"] if "High" in df else df["high"]
    low = df["Low"] if "Low" in df else df["low"]
    idx = df.index
    n = len(df)
    highs, lows = detect_swings(df, atr_period=atr_period, atr_mult=atr_mult)
    # Keyed on confirmed_pos, not pos — a pending level can't fire a break
    # before the market has actually confirmed it as a swing (see
    # detect_swings' own docstring on the pos/confirmed_pos distinction).
    # Keying this on pos instead was a lookahead bug: it let bar i react
    # to a swing whose OWN reversal hadn't happened yet at bar i.
    highs_by_pos = {h["confirmed_pos"]: h for h in highs}
    lows_by_pos = {l["confirmed_pos"]: l for l in lows}

    pending_high, pending_low = None, None
    bias = None
    breaks = []
    mss_strength = "normal" if mode == "wick" else "agresiv"

    for i in range(n):
        if i in highs_by_pos:
            pending_high = highs_by_pos[i]
        if i in lows_by_pos:
            pending_low = lows_by_pos[i]

        # "Normal" (wick) checks the bar's raw extreme; "Agresiv" (close,
        # the original/default behavior) checks only the settled close —
        # everything else about the pending-level bookkeeping is identical.
        trigger_up = float(high.iloc[i]) if mode == "wick" else float(close.iloc[i])
        trigger_down = float(low.iloc[i]) if mode == "wick" else float(close.iloc[i])

        if pending_high is not None and i > pending_high["pos"] and trigger_up > pending_high["price"]:
            kind = "CHoCH" if bias == "bearish" else "BOS"
            breaks.append({"type": "bullish", "structure": kind, "level": pending_high["price"],
                            "start": pending_high["time"], "end": idx[i], "mss_strength": mss_strength})
            bias = "bullish"
            pending_high = None

        if pending_low is not None and i > pending_low["pos"] and trigger_down < pending_low["price"]:
            kind = "CHoCH" if bias == "bullish" else "BOS"
            breaks.append({"type": "bearish", "structure": kind, "level": pending_low["price"],
                            "start": pending_low["time"], "end": idx[i], "mss_strength": mss_strength})
            bias = "bearish"
            pending_low = None

    return breaks


@st.cache_data(ttl=_CACHE_TTL)
def detect_order_blocks(df, max_scan_bars=None, record_history=False, min_body_ratio=DISPLACEMENT_MIN_BODY_RATIO):
    """OBSERVED: the last opposing candle before a displacement move that
    breaks clean through it. Bullish OB = last down-candle before a candle
    that closes above its high; bearish OB = mirror image. That is the
    complete computational definition — nothing here observes who traded
    it or why.

    INTERPRETED (ICT's own reading, not something this function can verify):
    that opposing candle marks "where institutions likely built a position
    before the move." Keep that reading in the methodology and in how a
    human trader uses this zone — treating it as an observed fact rather
    than an interpretation is exactly the gap a research pipeline needs to
    not fall into (see research/evidence.py's own permutation test, which
    checks whether "OB retracement" actually predicts anything, rather than
    assuming the ICT story is true because the geometry matched).

    max_scan_bars: see detect_fvgs' identical parameter — None (the live
    chart's own call site) preserves the original unbounded fill-scan; the
    research pipeline passes a real cap so a large backtest dataset doesn't
    turn the per-block scan into an O(n^2) blowup.

    record_history: see detect_fvgs' identical parameter — additive only
    (default False preserves today's exact dict shape everywhere else).
    =True adds "history" (the fill trajectory), "formation_i" (the
    breakout candle's own position — the earliest bar this OB can be
    resolved against), and "expiry_bar", used by backtest_custom_rule's
    own incremental zone tracking instead of a fresh per-bar rescan."""
    o = (df["Open"] if "Open" in df else df["open"]).to_numpy()
    h = (df["High"] if "High" in df else df["high"]).to_numpy()
    l = (df["Low"] if "Low" in df else df["low"]).to_numpy()
    c = (df["Close"] if "Close" in df else df["close"]).to_numpy()
    idx = df.index
    n = len(df)
    obs = []

    for i in range(n - 1):
        # The breakout candle (i+1) is what's supposed to be the displacement
        # move — a candle that merely closes past the prior high/low on a
        # long wick doesn't clear DISPLACEMENT_MIN_BODY_RATIO's own body-size
        # bar, just noise that technically satisfies the raw price condition.
        if not _is_displacement(o[i + 1], h[i + 1], l[i + 1], c[i + 1], min_ratio=min_body_ratio):
            continue
        bearish_i = c[i] < o[i]
        bullish_i = c[i] > o[i]
        top, bottom = max(o[i], c[i]), min(o[i], c[i])
        if bearish_i and c[i + 1] > h[i]:
            obs.append({"type": "bullish", "top": top, "bottom": bottom, "idx": i})
        elif bullish_i and c[i + 1] < l[i]:
            obs.append({"type": "bearish", "top": top, "bottom": bottom, "idx": i})

    for ob in obs:
        # Same consequent-encroachment treatment as FVGs (see detect_fvgs) —
        # a bullish OB sits below price that's since moved up, so it's
        # re-tested from above and eaten from the top down; a bearish OB
        # mirrors this from below.
        formation_top, formation_bottom = ob["top"], ob["bottom"]
        active_bottom, active_top = ob["bottom"], ob["top"]
        end_i = n - 1
        first_touch_i = None  # first bar price trades back into the zone at
        # all — see detect_fvgs' identical field for why this is distinct
        # from "mitigated" (research/sequences.py's setup chains need "price
        # just retraced here," not "this zone is now fully used up").
        history = [] if record_history else None
        scan_end = n if max_scan_bars is None else min(n, ob["idx"] + 2 + max_scan_bars)
        for j in range(ob["idx"] + 2, scan_end):
            lo_j, hi_j = l[j], h[j]
            if lo_j <= active_top and hi_j >= active_bottom:
                if first_touch_i is None:
                    first_touch_i = j
                # Clamp rather than let a deep wick push past the opposite
                # side — that's still just "fully mitigated", not inverted.
                if ob["type"] == "bullish":
                    active_top = max(min(active_top, lo_j), active_bottom)
                else:
                    active_bottom = min(max(active_bottom, hi_j), active_top)
                if record_history:
                    history.append((j, active_top, active_bottom))
                if active_top <= active_bottom:
                    end_i = j
                    break
        ob["start"] = idx[ob["idx"]]
        ob["end"] = idx[end_i]
        ob["top"], ob["bottom"] = active_top, active_bottom
        ob["mitigated"] = active_top <= active_bottom
        # See detect_fvgs' identical field for why this is a separate
        # concept from "mitigated" — a scan cut short by max_scan_bars
        # without ever reaching a real close-out isn't "still open," it's
        # a setup this scan gave up watching.
        ob["expired"] = (not ob["mitigated"]) and (scan_end < n)
        ob["first_touch"] = idx[first_touch_i] if first_touch_i is not None else None
        if record_history:
            ob["history"] = history
            ob["formation_i"] = ob["idx"] + 1
            ob["expiry_bar"] = (ob["idx"] + 2 + max_scan_bars) if max_scan_bars is not None else None
            # Order Blocks have no raw_top/raw_bottom the way FVGs do (see
            # _zone_bounds' own docstring) since ob["top"]/["bottom"] above
            # get overwritten with the FINAL post-scan state — _zone_at
            # needs the pre-encroachment starting point too, as its own
            # "nothing touched yet" starting value. Deliberately NOT named
            # raw_top/raw_bottom: those specific keys are what
            # _zone_bounds (pricing) looks for, and OB pricing was never
            # changed to use pre-encroachment bounds the way FVG's was —
            # this is purely _zone_at's own private bookkeeping, invisible
            # to anything that reads a zone's public shape.
            ob["_formation_top"], ob["_formation_bottom"] = formation_top, formation_bottom

    return obs


@st.cache_data(ttl=_CACHE_TTL)
def detect_breaker_blocks(df, max_scan_bars=None, min_body_ratio=DISPLACEMENT_MIN_BODY_RATIO):
    """Breaker Block — the Order Block equivalent of detect_ifvgs above,
    same mechanism applied to detect_order_blocks' own output instead of
    detect_fvgs': an order block that gets fully mitigated WITH the
    completing candle actually CLOSING beyond the block's own formation-
    time boundary (not just wicking through it) flips role — a bullish OB
    broken this way stops acting as support and starts acting as
    resistance, a bearish OB broken this way starts acting as support.
    Same price range as the block's own _formation_top/_formation_bottom
    (the size as it originally formed, before any consequent-encroachment
    eating), plus a fresh forward scan for whether price comes back to
    test it from the new direction. See detect_ifvgs' own docstring for
    the full reasoning (separate detector, not a flag; "closes beyond" on
    the completing bar as the engineering choice for confirmation)."""
    base = detect_order_blocks(df, max_scan_bars=max_scan_bars, record_history=True, min_body_ratio=min_body_ratio)
    if not base:
        return []
    high_arr = (df["High"] if "High" in df else df["high"]).to_numpy()
    low_arr = (df["Low"] if "Low" in df else df["low"]).to_numpy()
    close_arr = (df["Close"] if "Close" in df else df["close"]).to_numpy()
    idx = df.index
    n = len(df)
    pos_by_time = {t: i for i, t in enumerate(idx)}

    breakers = []
    for ob in base:
        if not ob["mitigated"]:
            continue
        end_i = pos_by_time[ob["end"]]
        if ob["type"] == "bullish":
            if close_arr[end_i] >= ob["_formation_bottom"]:
                continue
            new_type = "bearish"
        else:
            if close_arr[end_i] <= ob["_formation_top"]:
                continue
            new_type = "bullish"

        top, bottom = ob["_formation_top"], ob["_formation_bottom"]
        first_touch_i = None
        scan_end = n if max_scan_bars is None else min(n, end_i + 1 + max_scan_bars)
        for j in range(end_i + 1, scan_end):
            if low_arr[j] <= top and high_arr[j] >= bottom:
                first_touch_i = j
                break
        breakers.append({
            "type": new_type, "top": top, "bottom": bottom,
            "start": ob["end"], "origin_start": ob["start"],
            "first_touch": idx[first_touch_i] if first_touch_i is not None else None,
        })
    return breakers


def detect_equal_levels(points, tolerance=0.0015):
    """Groups swing highs (or lows) that sit within `tolerance` of each other
    — ICT's "equal highs/lows," read as resting liquidity: a cluster of stops
    sitting just beyond a shared price level, and a magnet for a sweep.

    The inner loop breaks as soon as a candidate exceeds tolerance rather
    than scanning every remaining point — safe because pts is sorted
    ascending by price and the tolerance check is against pts[i] specifically:
    once pts[j] is too far above pts[i], every later (even higher-priced)
    pts[j'] is too far too, so nothing past that point could ever match.
    Confirmed exact-output-preserving, not an approximation; confirmed the
    original nested-scan-without-break was the real cost driver at scale —
    a research backtest with ~77,000 swing points on one side alone made
    the unbounded O(k^2) version of this the single most expensive call in
    a full detector run.

    Deliberately NOT @st.cache_data — `points` is a plain list of dicts,
    which Streamlit can't hash cheaply: EVERY call, hit or miss, has to
    pickle the whole list first just to compute the cache key, which for
    the live chart's own call site (app.py calls this directly on raw
    swing-point lists every ~1s poll, not through detect_equal_highs_lows'
    df-keyed cache below) was pure overhead on top of an already-fast,
    early-break grouping loop — confirmed directly: this decorator was
    costing more than the function it wrapped. detect_equal_highs_lows
    still caches on `df` (a type st.cache_data hashes cheaply), which is
    what protects the larger-scale research path this function's own
    docstring benchmark describes."""
    groups = []
    used = [False] * len(points)
    pts = sorted(points, key=lambda p: p["price"])

    for i in range(len(pts)):
        if used[i]:
            continue
        group = [pts[i]]
        used[i] = True
        for j in range(i + 1, len(pts)):
            if used[j]:
                continue
            if abs(pts[j]["price"] - pts[i]["price"]) / pts[i]["price"] > tolerance:
                break
            group.append(pts[j])
            used[j] = True
        if len(group) >= 2:
            groups.append(group)

    return groups


@st.cache_data(ttl=_CACHE_TTL)
def detect_equal_highs_lows(df):
    """detect_equal_levels() groups raw swing points, not a shared "same
    shape as the other detectors" event — this wraps it into that shape
    (type/top/bottom/start/end) so a caller (research/live_scan.py's
    detector registry) can treat it like FVG/order-block/liquidity-reaction
    without special-casing it. start = the group's EARLIEST point (a
    cluster's identity is fixed once >=2 points exist there; a later third
    point joining the same cluster isn't a NEW cluster), end = its latest."""
    highs, lows = detect_swings(df)
    out = []
    for kind, points in (("equal_high", highs), ("equal_low", lows)):
        for group in detect_equal_levels(points):
            ordered = sorted(group, key=lambda p: p["time"])
            prices = [p["price"] for p in group]
            out.append({
                "type": kind, "top": max(prices), "bottom": min(prices),
                "start": ordered[0]["time"], "end": ordered[-1]["time"],
            })
    return out


@st.cache_data(ttl=_CACHE_TTL)
def detect_liquidity_levels(df, n_above=2, n_below=2, atr_period=14, atr_mult=1.5):
    """ICT's External Range Liquidity: the nearest swing highs/lows that price
    hasn't traded through yet — buy-side liquidity resting above (stops above
    the recent high), sell-side resting below. A swing point drops out the
    moment any later candle wicks through it — it's been swept, no longer
    "resting." Returns the `n_above` nearest still-live highs and `n_below`
    nearest still-live lows, closest to current price first."""
    high_col = "High" if "High" in df else "high"
    low_col = "Low" if "Low" in df else "low"
    close_col = "Close" if "Close" in df else "close"
    high_arr = df[high_col].to_numpy()
    low_arr = df[low_col].to_numpy()
    highs, lows = detect_swings(df, atr_period=atr_period, atr_mult=atr_mult)
    last_price = float(df[close_col].iloc[-1])

    def untouched_high(point):
        after = high_arr[point["pos"] + 1:]
        return after.size == 0 or bool((after < point["price"]).all())

    def untouched_low(point):
        after = low_arr[point["pos"] + 1:]
        return after.size == 0 or bool((after > point["price"]).all())

    live_highs = sorted((h for h in highs if h["price"] > last_price and untouched_high(h)),
                        key=lambda h: h["price"])
    live_lows = sorted((l for l in lows if l["price"] < last_price and untouched_low(l)),
                       key=lambda l: l["price"], reverse=True)

    return live_highs[:n_above], live_lows[:n_below]


@st.cache_data(ttl=_CACHE_TTL)
def _daily_volume_profiles(df, max_days=60):
    """Shared groundwork for detect_naked_pocs and detect_poor_highs_lows
    below — groups df into NY calendar sessions and computes each COMPLETE
    past day's own volume profile once, so two detectors reading the exact
    same (df, max_days) hit one cache entry instead of each re-walking
    history and re-bucketing every day independently.

    Sessions are NY calendar days — df.index must be tz-aware (every
    caller in this project already fetches tz-aware OHLCV, so this is
    never a fresh conversion) — the same wall-clock convention every
    other session-scoped feature here already uses (Kill Zones, session-
    restricted backtests). Only COMPLETE past days are considered; the
    current, still-forming day is excluded since its own profile isn't
    final yet. max_days bounds how far back to look — each day gets its
    own volume_profile() call, so cost is O(max_days), not O(len(df)).

    Returns a list of {"day" (the NY date), "end_pos" (that day's own last
    integer position in df, for "everything after this point" checks),
    "day_high"/"day_low" (that day's own session extremes), "high_time"/
    "low_time" (the exact bar each of those printed on — NOT necessarily
    end_pos; the high/low can land anywhere in the session), "vp"
    (volume_profile's own return dict for that day)}, most recent day
    first. Days volume_profile returned None for are skipped entirely,
    never included here — same as days with fewer than MIN_BARS_PER_DAY
    bars (below), which volume_profile was never even called for.

    MIN_BARS_PER_DAY exists because "day" here means "however many bars
    of df happen to share an NY calendar date" — on an intraday df that's
    a real session with real intraday price PATH to bucket, but on a
    1D+ df, each bar already covers one whole day, so every "day" group
    ends up with exactly 1 bar: volume_profile() still technically runs
    (bucketing that ONE candle's own high-low range) and returns SOME
    poc_price, but it's not a real session profile — there's no actual
    intraday path in a single OHLC bar, just its own four numbers
    reshuffled into 24 buckets. Confirmed directly as the actual cause of
    "Naked POC / Poor High-Low don't look right on 1D+" reports: the
    detectors WERE running, just on degenerate one-bar-a-day input,
    producing technically-computed but meaningless output instead of an
    honest empty result. 4 is deliberately low — even a coarse 4h main
    chart only has 6 bars/day, and that's still a genuinely useful
    profile; this is only meant to catch the "one bar IS the whole day"
    case, not to demand a lot of resolution.
    """
    MIN_BARS_PER_DAY = 4
    high_col = "High" if "High" in df else "high"
    low_col = "Low" if "Low" in df else "low"
    if df.empty:
        return []
    ny_dates = df.index.tz_convert("America/New_York").date
    unique_days = sorted(set(ny_dates))
    if len(unique_days) < 2:
        return []
    # [:-1] drops the current/still-forming day; [-max_days:] keeps only
    # the most recent max_days of what remains.
    past_days = unique_days[:-1][-max_days:]
    high_arr = df[high_col].to_numpy()
    low_arr = df[low_col].to_numpy()
    results = []
    for day in past_days:
        day_positions = np.flatnonzero(ny_dates == day)
        if day_positions.size < MIN_BARS_PER_DAY:
            continue
        vp = volume_profile(df.iloc[day_positions])
        if vp is None:
            continue
        day_highs, day_lows = high_arr[day_positions], low_arr[day_positions]
        hi_pos = int(day_positions[int(np.argmax(day_highs))])
        lo_pos = int(day_positions[int(np.argmin(day_lows))])
        results.append({
            "day": day, "end_pos": int(day_positions[-1]),
            "day_high": float(day_highs.max()), "day_low": float(day_lows.min()),
            "high_time": df.index[hi_pos], "low_time": df.index[lo_pos], "vp": vp,
        })
    return sorted(results, key=lambda r: r["day"], reverse=True)


@st.cache_data(ttl=_CACHE_TTL)
def detect_naked_pocs(df, max_days=60):
    """Naked (a.k.a. virgin) Points of Control — a past DAILY session's own
    busiest traded price (see indicators.volume_profile) that price hasn't
    traded back through since that session closed. Same "resting until
    touched" rule detect_liquidity_levels above already uses for swing
    highs/lows, applied to a session's POC instead of a swing point: a
    level the market auctioned heavily around once and never revisited
    acts as a magnet, one of the most-watched levels on a professional
    order-flow desk — unlike the CURRENT profile's own POC/VAH/VAL
    (computed once, over whatever window is presently loaded), these
    persist across every session until touched, and get dropped the
    instant that happens.

    Returns a list of {"price", "time" (the day's own last bar — when
    this POC became final and eligible to go naked), "day" (ISO date
    string, for a label)}, naked ones only, most recent first."""
    high_col = "High" if "High" in df else "high"
    low_col = "Low" if "Low" in df else "low"
    high_arr = df[high_col].to_numpy()
    low_arr = df[low_col].to_numpy()
    results = []
    for d in _daily_volume_profiles(df, max_days):
        poc = d["vp"]["poc_price"]
        end_pos = d["end_pos"]
        after_high = high_arr[end_pos + 1:]
        after_low = low_arr[end_pos + 1:]
        # Two-sided range check (unlike untouched_high/untouched_low
        # above, which only ever need to check one direction for a
        # directional swing level) — a POC can get traded through from
        # either side, so "touched" means any later bar's own [low, high]
        # span includes this exact price, not just a one-sided breach.
        touched = after_high.size > 0 and bool(((after_low <= poc) & (after_high >= poc)).any())
        if not touched:
            results.append({"price": poc, "time": df.index[end_pos], "day": d["day"].isoformat()})
    return sorted(results, key=lambda r: r["time"], reverse=True)


@st.cache_data(ttl=_CACHE_TTL)
def poc_migration(df, lookback=5, flat_threshold_pct=1.5):
    """Is the daily session POC drifting session to session, or staying
    roughly in place? A POC that keeps printing higher (or lower) day
    after day means value itself is migrating, not just price wicking
    around — the standard professional read for "is a trend actually
    building" versus "the market's still auctioning around the same
    fair price." Same daily-session groundwork as detect_naked_pocs (see
    _daily_volume_profiles' own docstring) — NY calendar sessions,
    current day excluded, capped to the most recent `lookback` of them.

    flat_threshold_pct: the minimum |% change| between the oldest and
    newest POC in the window to call it a real drift rather than noise —
    below this, direction reads "flat" regardless of which way the raw
    number moved (a real market rarely prints the exact same POC twice
    in a row, so SOME nonzero change is normal background noise, not a
    trend).

    Returns None when fewer than 2 sessions are available in the window
    (nothing to compare), else {"days": [{"day", "price"}, ...] (oldest
    first, at most `lookback` entries), "pct_change", "direction"
    ("up"/"down"/"flat")} — pct_change and direction are both measured
    oldest-to-newest across the whole window, not just the last step."""
    daily = sorted(_daily_volume_profiles(df, max_days=lookback), key=lambda d: d["day"])
    if len(daily) < 2:
        return None
    points = [{"day": d["day"].isoformat(), "price": d["vp"]["poc_price"]} for d in daily]
    first_price, last_price = points[0]["price"], points[-1]["price"]
    pct_change = (last_price - first_price) / first_price * 100 if first_price else 0.0
    if abs(pct_change) < flat_threshold_pct:
        direction = "flat"
    else:
        direction = "up" if pct_change > 0 else "down"
    return {"days": points, "pct_change": pct_change, "direction": direction}


@st.cache_data(ttl=_CACHE_TTL)
def detect_poor_highs_lows(df, max_days=60, min_frac=0.25):
    """A session's own high or low that printed on REAL volume, not a thin
    single tap — the auction didn't cleanly reject there. Distinguishes
    "price wicked up here once and reversed hard" (a clean extreme, real
    rejection, likely to hold) from "price kept trading right at this
    level" (a poor one — real unfinished business, more likely to get
    revisited or run straight through than defended) — the standard
    professional read on a session's own high/low beyond just where it
    printed.

    min_frac: how busy (relative to that day's own busiest bucket) the
    bucket AT the extreme needs to be to count as "poor" — default 0.25
    means it carried at least a quarter of the day's busiest bucket's own
    volume, real trading, not a single spike-and-reverse print.

    volume_profile()'s own bucket edges are built from exactly this day's
    [min(low), max(high)] — the TOP bucket's own upper edge is always
    day_high and the BOTTOM bucket's own lower edge is always day_low, by
    construction, so no separate search for "which bucket contains the
    extreme" is needed.

    Same day/history conventions as detect_naked_pocs (see
    _daily_volume_profiles' own docstring) — NY calendar sessions, current
    day excluded, max_days bounds lookback. Returns a list of {"price",
    "time", "day", "kind" ("high" or "low")}, most recent first. Unlike
    naked POCs, there's no persistence/touch tracking here — a poor high/
    low doesn't go away once price returns to it; it's a property of how
    that session itself traded, not a level still being defended."""
    results = []
    for d in _daily_volume_profiles(df, max_days):
        buckets = d["vp"]["buckets"]
        if not buckets:
            continue
        # time = the bar the extreme itself actually printed on, not the
        # day's own last bar — a session's high/low can land anywhere in
        # it, and a marker belongs at the moment it happened, not at
        # end-of-day.
        if buckets[-1]["volume_frac"] >= min_frac:
            results.append({"price": d["day_high"], "time": d["high_time"], "day": d["day"].isoformat(), "kind": "high"})
        if buckets[0]["volume_frac"] >= min_frac:
            results.append({"price": d["day_low"], "time": d["low_time"], "day": d["day"].isoformat(), "kind": "low"})
    return sorted(results, key=lambda r: r["time"], reverse=True)


@st.cache_data(ttl=_CACHE_TTL)
def detect_liquidity_sweeps(df, tier=False):
    """Every swing high/low that later got wicked through, as a standalone
    point-in-time event — the raw "stop hunt" moment on its own, not paired
    with a reaction order block the way detect_liquidity_reactions requires
    (that one only reports at most one bearish + one bullish sweep, the
    MOST RECENT of each, specifically for chart display; this reports every
    sweep in the window, for research/sequences.py's Judas-swing state
    machine to chain against a LATER CHoCH and retracement).

    `type` is the setup direction a sweep points toward, not the sweep's own
    raw direction — sweeping a HIGH hunts buy-side stops and typically
    precedes a BEARISH reversal, so it's tagged "bearish" here (matching the
    same type convention detect_structure_breaks/detect_fvgs/
    detect_order_blocks already use, so a caller can compare `type` fields
    directly across all four without translating conventions per function).

    tier=False (default, UNCHANGED from before this parameter existed):
    return shape and every existing caller's behavior (research/sequences.py,
    research/events.py, the live chart) is identical to before this
    parameter existed.

    tier=True adds a "tier" field to each sweep dict — a STRUCTURAL PROXY
    for the trader's Major/Minor split, not the authoritative version:
      "major" — this swing point was STILL the active external
        dealing-range boundary (the same pending_high/pending_low
        current_dealing_range/detect_structure_breaks track) at the exact
        moment it got swept — nothing newer had already taken over.
      "minor" — a NEWER swing point had already formed and taken over
        that boundary role before THIS point got swept.
    Reuses the exact pending-boundary replay loop detect_structure_breaks/
    current_dealing_range already use, just recorded here per-bar instead
    of collapsed to a single current value, so every sweep — not only the
    most recent one — can be tiered against what was pending immediately
    before it happened.

    CORRECTED definition (from the trader directly): Major = "the last
    liquidity point that forms an impulsive leg" — i.e. a point PROVEN
    major by actually leading to a confirmed sweep→MSS→displacement
    sequence, not just "nothing newer superseded it yet." This function
    can't check that on its own (it doesn't know about MSS/displacement —
    that's research/sequences.py's layer, and detectors.py importing it back
    would be a circular import). Treat this tier as a decent structural
    approximation, not the final answer — research/sequences.py's
    classify_liquidity_tiers() cross-references actual confirmed Judas
    setups and is the more literal implementation of the trader's own
    wording; prefer it once it's available.

    NOTE: this reads market STRUCTURE (swing highs/lows), not calendar
    time. "Local" is NOT LOD/HOD (that was a mistaken assumption last
    pass, now corrected — LOD/HOD is a real, separate, session-windowed
    concept, not implemented in this module). Local is a swing point too
    recent to have resolved as Major or Minor yet — see
    classify_liquidity_tiers() in research/sequences.py."""
    high_col = "High" if "High" in df else "high"
    low_col = "Low" if "Low" in df else "low"
    close_col = "Close" if "Close" in df else "close"
    high_arr = df[high_col].to_numpy()
    low_arr = df[low_col].to_numpy()
    idx = df.index
    n = len(df)
    highs, lows = detect_swings(df)

    def sweep_idx(point, arr, above):
        after = arr[point["pos"] + 1:]
        mask = (after > point["price"]) if above else (after < point["price"])
        hit = mask.argmax() if mask.any() else None
        return None if hit is None else point["pos"] + 1 + int(hit)

    sweeps = []
    for h in highs:
        j = sweep_idx(h, high_arr, True)
        if j is not None:
            sweeps.append({"type": "bearish", "level": h["price"], "start": h["time"], "end": idx[j],
                            "_pos": h["pos"], "_sweep_i": j})
    for l in lows:
        j = sweep_idx(l, low_arr, False)
        if j is not None:
            sweeps.append({"type": "bullish", "level": l["price"], "start": l["time"], "end": idx[j],
                            "_pos": l["pos"], "_sweep_i": j})

    if tier:
        # confirmed_pos, not pos — see detect_swings' own docstring. A
        # sweep's tier depends on whether the level was already CONFIRMED
        # as a swing before the sweep, not merely whether the pivot bar
        # itself had occurred yet.
        highs_by_pos = {h["confirmed_pos"]: h for h in highs}
        lows_by_pos = {l["confirmed_pos"]: l for l in lows}
        close_arr = df[close_col].to_numpy()
        # pending_*_before[i] = the position of whichever swing point was
        # still pending IMMEDIATELY BEFORE bar i is processed — recorded for
        # every bar, not just sweep bars, so two sweeps landing on the same
        # bar both read the correct shared context instead of a dict
        # overwriting itself.
        pending_high_before = [None] * n
        pending_low_before = [None] * n
        pending_high, pending_low = None, None
        for i in range(n):
            pending_high_before[i] = pending_high["pos"] if pending_high is not None else None
            pending_low_before[i] = pending_low["pos"] if pending_low is not None else None
            if i in highs_by_pos:
                pending_high = highs_by_pos[i]
            if i in lows_by_pos:
                pending_low = lows_by_pos[i]
            c = float(close_arr[i])
            if pending_high is not None and i > pending_high["pos"] and c > pending_high["price"]:
                pending_high = None
            if pending_low is not None and i > pending_low["pos"] and c < pending_low["price"]:
                pending_low = None

        for s in sweeps:
            if s["type"] == "bearish":
                s["tier"] = "major" if pending_high_before[s["_sweep_i"]] == s["_pos"] else "minor"
            else:
                s["tier"] = "major" if pending_low_before[s["_sweep_i"]] == s["_pos"] else "minor"

    for s in sweeps:
        s.pop("_pos", None)
        s.pop("_sweep_i", None)

    return sorted(sweeps, key=lambda s: s["end"])


@st.cache_data(ttl=_CACHE_TTL)
def detect_liquidity_reactions(df, max_candles_after=5, max_scan_bars=None, record_history=False,
                                atr_period=14, atr_mult=1.5):
    """Once external liquidity is swept, ICT expects a reaction from the order
    block that formed right at that sweep — the "external area" price reverses
    from. Finds the MOST RECENT high-sweep and MOST RECENT low-sweep, and for
    each, the nearest still-unmitigated opposing order block that formed within
    a few candles after it. Returns at most one bearish + one bullish reaction
    zone — only ever the most recent, so the chart doesn't fill up with stale
    ones once price has moved on.

    max_scan_bars: passed straight through to the detect_order_blocks(df)
    call below — see that function's own docstring. Without this, calling
    THIS function from the research pipeline (which passes a cap) still
    triggered a full UNBOUNDED order-block scan internally (a different
    max_scan_bars value is a different st.cache_data key, so it can't reuse
    an already-capped result computed elsewhere) — confirmed directly, this
    was still hanging after detect_fvgs/detect_order_blocks' own scans were
    already fixed, until this call site got the same treatment.

    record_history: passed straight through to detect_order_blocks — see
    its own docstring on _formation_top/_formation_bottom, the
    pre-encroachment bounds pattern_win_rate needs and this function's
    own {**ob, ...} spread below otherwise silently drops."""
    high_col = "High" if "High" in df else "high"
    low_col = "Low" if "Low" in df else "low"
    high_arr = df[high_col].to_numpy()
    low_arr = df[low_col].to_numpy()
    highs, lows = detect_swings(df, atr_period=atr_period, atr_mult=atr_mult)
    obs = detect_order_blocks(df, max_scan_bars=max_scan_bars, record_history=record_history)
    reactions = []

    def sweep_idx(point, arr, above):
        after = arr[point["pos"] + 1:]
        mask = (after > point["price"]) if above else (after < point["price"])
        hit = mask.argmax() if mask.any() else None
        return None if hit is None else point["pos"] + 1 + int(hit)

    high_sweeps = [(h, sweep_idx(h, high_arr, True)) for h in highs]
    high_sweeps = [(h, i) for h, i in high_sweeps if i is not None]
    if high_sweeps:
        level, sweep_i = max(high_sweeps, key=lambda hs: hs[1])
        for ob in obs:
            if (ob["type"] == "bearish" and not ob["mitigated"] and not ob["expired"]
                    and sweep_i <= ob["idx"] <= sweep_i + max_candles_after):
                reactions.append({**ob, "swept_level": level["price"]})
                break

    low_sweeps = [(l, sweep_idx(l, low_arr, False)) for l in lows]
    low_sweeps = [(l, i) for l, i in low_sweeps if i is not None]
    if low_sweeps:
        level, sweep_i = max(low_sweeps, key=lambda ls: ls[1])
        for ob in obs:
            if (ob["type"] == "bullish" and not ob["mitigated"] and not ob["expired"]
                    and sweep_i <= ob["idx"] <= sweep_i + max_candles_after):
                reactions.append({**ob, "swept_level": level["price"]})
                break

    return reactions


@st.cache_data(ttl=_CACHE_TTL)
def current_dealing_range(df, atr_period=14, atr_mult=1.5):
    """The active premium/discount range: the most recent swing high and
    swing low that price hasn't closed back through yet — the same
    pending-high/pending-low tracking detect_structure_breaks uses to fire
    BOS/CHoCH, reused here for its other half. A dealing range IS exactly
    that pair: a box from the last unbroken swing high to the last unbroken
    swing low, staying put until a close beyond one side invalidates it —
    at which point that side goes back to None and waits for the next swing
    point to re-anchor it, so the box can flip one boundary at a time
    rather than resetting wholesale on every break.

    Returns None until both a pending high and a pending low have formed at
    least once — nothing meaningful to box in before that.

    top_swept_at/bottom_swept_at: the most recent bar (if any) where a wick
    traded beyond the range's own top/bottom without a CLOSE beyond it —
    the exact "liquidity grab, not a break" pattern (wick sweeps a level,
    price rejects, the level survives because only a close counts as a
    genuine break — see this function's own close-based invalidation
    above). None when no such sweep has happened since this boundary was
    set. This is presentation context, not a second definition of the
    range: the top/bottom above are computed exactly as before, untouched
    by this — it only explains, when a boundary looks like it should have
    moved and didn't, that a wick already tried and got rejected there."""
    high = df["High"] if "High" in df else df["high"]
    low = df["Low"] if "Low" in df else df["low"]
    close = df["Close"] if "Close" in df else df["close"]
    idx = df.index
    n = len(df)
    highs, lows = detect_swings(df, atr_period=atr_period, atr_mult=atr_mult)
    # confirmed_pos, not pos — see detect_swings' own docstring. The range
    # can't anchor to a swing before the market has actually confirmed it.
    highs_by_pos = {h["confirmed_pos"]: h for h in highs}
    lows_by_pos = {l["confirmed_pos"]: l for l in lows}

    pending_high, pending_low = None, None
    for i in range(n):
        if i in highs_by_pos:
            pending_high = highs_by_pos[i]
        if i in lows_by_pos:
            pending_low = lows_by_pos[i]
        c = float(close.iloc[i])
        if pending_high is not None and i > pending_high["pos"] and c > pending_high["price"]:
            pending_high = None
        if pending_low is not None and i > pending_low["pos"] and c < pending_low["price"]:
            pending_low = None

    if pending_high is None or pending_low is None:
        return None

    top_swept_at = None
    for i in range(pending_high["confirmed_pos"] + 1, n):
        if float(high.iloc[i]) > pending_high["price"] and float(close.iloc[i]) <= pending_high["price"]:
            top_swept_at = {"time": idx[i], "price": float(high.iloc[i])}
    bottom_swept_at = None
    for i in range(pending_low["confirmed_pos"] + 1, n):
        if float(low.iloc[i]) < pending_low["price"] and float(close.iloc[i]) >= pending_low["price"]:
            bottom_swept_at = {"time": idx[i], "price": float(low.iloc[i])}

    return {
        "top": pending_high["price"], "top_time": pending_high["time"],
        "bottom": pending_low["price"], "bottom_time": pending_low["time"],
        "start": max(pending_high["time"], pending_low["time"]),
        "eq": (pending_high["price"] + pending_low["price"]) / 2,
        "top_swept_at": top_swept_at, "bottom_swept_at": bottom_swept_at,
    }


# Two deliberately different ways to answer "which of these already-
# detected zones should I actually show" — direct request, for the whole
# detection system, not any one caller: "one that tracks previous
# candles, and one that scans historical values." Neither one re-runs
# detection itself — both take whatever detect_fvgs/detect_order_blocks
# already found (fill/mitigation status resolved against the FULL
# history, never truncated) and just filter/rank that output two
# different ways. A caller showing "areas and levels" on a chart
# typically wants BOTH results, merged (a zone can legitimately appear in
# both) — see this project's own app.py/crypto_app.py FVG/Order Blocks
# layers and the Bar Replay chart for how they're combined in practice.
def recent_zone_tracker(zones, df, lookback_bars=50):
    """Engine A — "tracks previous candles": every zone that FORMED within
    the last `lookback_bars` bars of `df`, uncapped. Direct request: "the
    tracker should mark everything in recent past... tells me with
    surgical precision what and why it happened" — a complete recent
    record, not a filtered top-N. Age is the only filter; a zone's own
    distance from current price doesn't matter here at all (that's
    historical_zone_scanner's own job)."""
    if not zones or df is None or len(df) == 0:
        return []
    pos_by_time = {t: i for i, t in enumerate(df.index)}
    cutoff = max(0, len(df) - lookback_bars)
    return [z for z in zones if (pos_by_time.get(z["start"]) or 0) >= cutoff]


def historical_zone_scanner(zones, current_price, n_per_side=2, top_key="top", bottom_key="bottom"):
    """Engine B — "scans historical values": the nearest `n_per_side`
    zones ABOVE current_price and nearest `n_per_side` BELOW it, drawn
    from the FULL zone list regardless of age — an old, still-unmitigated
    zone from far back in history counts exactly the same as one formed
    yesterday. Direct request: "the historical ones should track areas
    above and below price only... shows me where price might go even if
    it exits the range" — read as potential liquidity draws/targets, not
    "what just happened" (that's recent_zone_tracker's own job). Same
    above/below split this project's own _nearest_by_price already uses
    for on-chart zone selection — reimplemented here (rather than
    imported) since that helper lives duplicated in app.py/crypto_app.py,
    not in this shared detection module."""
    above, below = [], []
    for z in zones:
        top, bottom = z[top_key], z[bottom_key]
        if bottom >= current_price:
            above.append((bottom - current_price, z))
        elif top <= current_price:
            below.append((current_price - top, z))
    above.sort(key=lambda pair: pair[0])
    below.sort(key=lambda pair: pair[0])
    return [z for _, z in above[:n_per_side]] + [z for _, z in below[:n_per_side]]


def merge_zone_engines(recent, historical):
    """Combines recent_zone_tracker's and historical_zone_scanner's own
    output into one list, keeping recent-tracker's own ordering first (a
    caller's most-recently-formed-last convention) and appending whichever
    historical picks weren't already included — a zone easily qualifies
    for both (freshly formed AND the nearest one above/below price), and
    should only ever be drawn once."""
    seen = {id(z) for z in recent}
    return list(recent) + [z for z in historical if id(z) not in seen]


def find_clean_respect_streaks(df, zone_types=("FVG", "Order Block"), min_streak=3,
                                min_body_ratio=DISPLACEMENT_MIN_BODY_RATIO):
    """Runs of CONSECUTIVE zones (FVG and/or Order Block, merged into one
    formation-time-ordered timeline) that every single one gets
    genuinely RESPECTED — the exact "clean price action" read a
    discretionary trader means by it: price never CLOSES through the
    zone (a wick tapping it is fine, ICT's own "liquidity grab, still
    respected" case) and the zone never gets 100% wicked/mitigated (its
    own detect_fvgs/detect_order_blocks "filled"/"mitigated" flag — a
    partial eat-into is fine, full consumption is not). A zone that
    fails EITHER test ends the current streak; only runs of at least
    min_streak zones are returned, so isolated one-off "got respected"
    zones (common, not interesting) don't count.

    Returns a list of {"start", "end", "n_zones", "zones", "bullish",
    "bearish"} dicts, oldest streak first. Each zone in "zones" keeps its
    own full detector shape (type/top/bottom/start/end/...) plus
    "layer" ("FVG"/"Order Block") and "closed_through" (always False
    here, kept for symmetry/debugging)."""
    close_arr = (df["Close"] if "Close" in df else df["close"]).to_numpy()
    n = len(df)
    pos_by_time = {t: i for i, t in enumerate(df.index)}

    merged = []
    if "FVG" in zone_types:
        for z in detect_fvgs(df, min_body_ratio=min_body_ratio):
            merged.append({**z, "layer": "FVG"})
    if "Order Block" in zone_types:
        for z in detect_order_blocks(df, min_body_ratio=min_body_ratio):
            merged.append({**z, "layer": "Order Block"})
    merged.sort(key=lambda z: z["start"])

    def _respected(z):
        mitigated_flag = z.get("filled") if z["layer"] == "FVG" else z.get("mitigated")
        if mitigated_flag:
            return False
        start_pos = pos_by_time.get(z["start"])
        if start_pos is None:
            return True
        # Closed-through check uses the zone's OWN raw formation bounds
        # (raw_top/raw_bottom for FVG, top/bottom already IS the
        # formation size for a fresh Order Block) — the boundary a
        # genuine close-through breaches, not the already-eaten-into
        # "active" edge consequent encroachment may have shrunk it to.
        top = z.get("raw_top", z["top"])
        bottom = z.get("raw_bottom", z["bottom"])
        future_closes = close_arr[start_pos + 1:]
        if z["type"] == "bullish":
            return not bool((future_closes < bottom).any())
        else:
            return not bool((future_closes > top).any())

    streaks = []
    current = []
    for z in merged:
        if _respected(z):
            current.append(z)
        else:
            if len(current) >= min_streak:
                streaks.append(current)
            current = []
    if len(current) >= min_streak:
        streaks.append(current)

    out = []
    for s in streaks:
        out.append({
            "start": s[0]["start"], "end": s[-1]["end"], "n_zones": len(s), "zones": s,
            "bullish": sum(1 for z in s if z["type"] == "bullish"),
            "bearish": sum(1 for z in s if z["type"] == "bearish"),
        })
    return out


def describe_streak_conditions(df, streak, atr_period=14):
    """Context around one find_clean_respect_streaks entry, for the
    "what conditions produced this" read a human studying it actually
    wants: volatility regime (this streak's own mean ATR vs. the whole
    df's own median — >1 means it happened in an above-normal-vol
    stretch), net trend (close-to-close % move across the streak, sign
    only meaningful alongside the bullish/bearish zone tally), and
    duration (bars and, for an intraday df, wall-clock span)."""
    idx = df.index
    close = df["Close"] if "Close" in df else df["close"]
    pos_by_time = {t: i for i, t in enumerate(idx)}
    start_pos = pos_by_time.get(streak["start"], 0)
    end_pos = pos_by_time.get(streak["end"], len(df) - 1)
    end_pos = max(end_pos, start_pos)

    atr_series = atr(df, atr_period)
    streak_atr = atr_series.iloc[start_pos:end_pos + 1].mean()
    baseline_atr = atr_series.median()
    vol_ratio = float(streak_atr / baseline_atr) if baseline_atr else float("nan")

    start_close = float(close.iloc[start_pos])
    end_close = float(close.iloc[end_pos])
    net_move_pct = (end_close - start_close) / start_close * 100 if start_close else float("nan")

    return {
        "n_bars": end_pos - start_pos + 1,
        "duration": idx[end_pos] - idx[start_pos],
        "vol_ratio_vs_median": vol_ratio,
        "net_move_pct": net_move_pct,
        "dominant_direction": "bullish" if streak["bullish"] > streak["bearish"]
                               else ("bearish" if streak["bearish"] > streak["bullish"] else "mixed"),
    }
