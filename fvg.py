"""
Fair Value Gap (FVG) detection — the classic 3-candle imbalance:

  candle[i-1]        candle[i]        candle[i+1]
  (leaves the gap)   (impulse move)   (confirms the gap)

Bullish FVG: low of candle[i+1] > high of candle[i-1] — price left a void
             between those two candles that candle[i] jumped straight over.
Bearish FVG: high of candle[i+1] < low of candle[i-1] — same thing, downward.

The gap "fills" (mitigates) the first time a later candle trades back into
its price zone. Open gaps (never filled) are the ones price hasn't returned
to yet — usually the more interesting ones to watch.
"""

import numpy as np
import streamlit as st

# Detection here is O(n) but with per-row pandas access, not free on a
# multi-thousand-row df — and several of these are called more than once per
# FVG-page render (detect_swings alone up to 4x: Swing Points, Equal Highs/
# Lows, and again inside detect_liquidity_levels/detect_liquidity_reactions).
# Caching means repeat calls with the same df — within one render AND across
# Streamlit reruns triggered by unrelated widgets — are a hash lookup instead
# of a recompute. TTL matches get_yf_ohlcv's, so it tracks fresh data.
_CACHE_TTL = 300

# ICT's own definition of displacement is a real, fast, mostly-one-direction
# candle — body dominating its high-low range, not a long-wicked candle that
# merely closed past a level. Without this gate, both FVG and order-block
# detection below treat ANY 3-candle gap or ANY close-beyond-the-prior-high
# as valid, which in practice fires on plenty of choppy, low-conviction
# candles that no one would actually call an imbalance or an institutional
# footprint. 0.5 is deliberately looser than the ~0.75 "textbook strong
# displacement" bar — strict enough to drop indecisive candles, loose enough
# not to silently empty out every FVG/OB on calmer timeframes.
DISPLACEMENT_MIN_BODY_RATIO = 0.5


def _is_displacement(o, h, l, c, min_ratio=DISPLACEMENT_MIN_BODY_RATIO):
    rng = h - l
    if rng <= 0:
        return False
    return abs(c - o) / rng >= min_ratio


@st.cache_data(ttl=_CACHE_TTL)
def detect_fvgs(df, min_body_ratio=DISPLACEMENT_MIN_BODY_RATIO, max_scan_bars=None, record_history=False):
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
def detect_structure_breaks(df, mode="close"):
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
    highs, lows = detect_swings(df)
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
def detect_order_blocks(df, max_scan_bars=None, record_history=False):
    """The last opposing candle before a displacement move that breaks clean
    through it — the classic ICT proxy for "where institutions likely built
    a position before the move." Bullish OB = last down-candle before a
    candle that closes above its high; bearish OB = mirror image.

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
        # long wick isn't the "institutional footprint" ICT means by order
        # block, just noise that technically satisfies the raw price
        # condition. See DISPLACEMENT_MIN_BODY_RATIO.
        if not _is_displacement(o[i + 1], h[i + 1], l[i + 1], c[i + 1]):
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
def detect_liquidity_levels(df, n_above=2, n_below=2):
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
    highs, lows = detect_swings(df)
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
    that's research/sequences.py's layer, and fvg.py importing it back
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
def detect_liquidity_reactions(df, max_candles_after=5, max_scan_bars=None):
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
    already fixed, until this call site got the same treatment."""
    high_col = "High" if "High" in df else "high"
    low_col = "Low" if "Low" in df else "low"
    high_arr = df[high_col].to_numpy()
    low_arr = df[low_col].to_numpy()
    highs, lows = detect_swings(df)
    obs = detect_order_blocks(df, max_scan_bars=max_scan_bars)
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
def current_dealing_range(df):
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
    highs, lows = detect_swings(df)
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
