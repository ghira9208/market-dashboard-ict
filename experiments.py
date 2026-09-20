"""
The experiment builder — bridges tuning detector/regime settings live
against a chart and running those same settings through a real, Edge-Lab-
grade backtest. Built around a specific trading thesis, stated directly
rather than inferred from a request: trade sharp, fast reactions off
known zones (FVG/Order Block), but ONLY when the market has enough
volatility to actually deliver the move being looked for — a
consolidating range offers nothing to react FROM, so entering one is
just paying cost for no edge.

Three ingredients, each a checkable definition, not a vibe:

  - volatility regime: each bar's own ATR ranked against its trailing
    history (a rolling percentile, computed causally — no lookahead).
    High percentile = expansion, low = squeeze/consolidation.
  - reaction quality: how far price closed away from a zone's own
    ORIGINAL bounds (raw_top/raw_bottom — before consequent-encroachment
    eats into it) within a handful of bars of first touching it. Sharp
    and immediate scores high; a slow grind through the zone doesn't
    qualify at all.
  - fast resolution: forward_bars is deliberately short (a handful of
    bars) rather than the 10-20+ used elsewhere in this project — the
    actual thesis is "did this resolve FAST," not "did it eventually
    work out."

Reuses detectors.py's own zone detection (first_touch/raw_top/raw_bottom
already tracked there for exactly this kind of downstream use — nothing
new added to the detectors themselves) and research/evidence.py's
permutation-test engine. No new detection or statistics machinery, just
a new way of combining what already exists.

Every "deep backtest" run — pass or fail — is appended to
.cache/experiment_trials.jsonl, and load_experiment_trials() BH-corrects
across the WHOLE accumulated log, same "one p-value means nothing in
isolation" discipline as edge_lab/agent.py and institutional_backtest.py.
Tuning sliders and re-running until one trial looks good, then reporting
only that one, is exactly the p-hacking this correction step exists to
catch — it's applied here for the same reason.
"""
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

from detectors import (DISPLACEMENT_MIN_BODY_RATIO, detect_breaker_blocks, detect_fvgs, detect_ifvgs,
                        detect_liquidity_sweeps, detect_order_blocks, detect_swings)
from edge_lab.multiple_testing import benjamini_hochberg
from research.evidence import run_event_study

_CACHE_TTL = 300
_TRIALS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "experiment_trials.jsonl")

# detect_ifvgs/detect_breaker_blocks don't take a record_history kwarg (they
# have no "still open" concept of their own to track — an inversion only
# exists once its origin zone already fully mitigated, see each detector's
# own docstring) — the two lambdas below just swallow it so every entry
# here shares one call shape: detect_fn(df, max_scan_bars=..., record_history=...).
# Both already return the exact same zone shape detect_fvgs/detect_order_blocks
# do (type/top/bottom/start/first_touch) — confirmed directly, nothing new
# needed in _zone_reaction/_qualifying_touches to support them.
DETECTOR_FNS = {
    "FVG": detect_fvgs,
    "Order Block": detect_order_blocks,
    "IFVG": lambda full_df, max_scan_bars=None, record_history=False: detect_ifvgs(full_df, max_scan_bars=max_scan_bars),
    "Breaker Block": lambda full_df, max_scan_bars=None, record_history=False: detect_breaker_blocks(full_df, max_scan_bars=max_scan_bars),
}


@st.cache_data(ttl=_CACHE_TTL)
def atr_series(df, period=14):
    high = (df["High"] if "High" in df else df["high"]).to_numpy()
    low = (df["Low"] if "Low" in df else df["low"]).to_numpy()
    close = (df["Close"] if "Close" in df else df["close"]).to_numpy()
    prev_close = np.empty_like(close)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return pd.Series(tr, index=df.index).rolling(period, min_periods=1).mean()


@st.cache_data(ttl=_CACHE_TTL)
def volatility_percentile(df, atr_period=14, lookback=100):
    """Each bar's own ATR, ranked against its trailing `lookback` bars —
    0.0 = the quietest this instrument has recently been, 1.0 = the most
    volatile. A rolling rank per bar, not one current-vs-history number,
    so every historical bar in a backtest gets a regime label computed
    only from data available AT that bar."""
    atr = atr_series(df, atr_period)
    return atr.rolling(lookback, min_periods=max(5, lookback // 2)).apply(
        lambda w: (w.iloc[-1] >= w).mean(), raw=False
    )


def _zone_reaction(df, zone, reaction_window, reaction_mult, atr, close_arr, pos_by_time):
    """Was the reaction off this zone SHARP — did price close away from
    its own ORIGINAL bounds by at least reaction_mult x ATR within
    reaction_window bars of first touching it? None means "nothing to
    judge" (never touched at all, or touched too close to the end of the
    loaded data to see reaction_window bars past it) — never a False
    disguised as a real negative result.

    confirm_pos (the bar where the reaction is fully evaluated, touch_pos
    + reaction_window) is returned alongside touch_pos — see
    build_experiment_events' own comment on why the forward return MUST
    be measured starting from confirm_pos, never from touch_pos: the
    reaction check here already requires price to have moved
    reaction_mult x ATR by confirm_pos, so a "forward return" computed
    FROM touch_pos would re-include that same already-confirmed move
    inside the very thing being tested — confirmed directly as a real
    bug (a wide sweep run on bar-order-SHUFFLED, i.e. genuinely
    information-free, data still "found" large, BH-significant,
    same-sign, holdout-passing edges under the old touch_pos-anchored
    version, because the qualifying criterion and the outcome window
    overlapped almost entirely)."""
    if zone.get("first_touch") is None:
        return None
    touch_pos = pos_by_time.get(zone["first_touch"])
    if touch_pos is None:
        return None
    confirm_pos = touch_pos + reaction_window
    if confirm_pos >= len(close_arr):
        return None
    atr_at_touch = atr.iloc[touch_pos]
    if pd.isna(atr_at_touch) or atr_at_touch <= 0:
        return None
    moved = close_arr[confirm_pos] - close_arr[touch_pos]
    direction = zone["type"]
    expected_sign = 1 if direction == "bullish" else -1
    qualifies = (moved * expected_sign) >= (reaction_mult * atr_at_touch)
    return {"qualifies": bool(qualifies), "direction": direction, "touch_pos": touch_pos, "confirm_pos": confirm_pos}


def _qualifying_touches(df, detector_name, min_volatility_pctile, reaction_window, reaction_mult,
                         max_scan_bars=2000):
    """Every historical zone touch (this detector, over whatever history
    df covers) that got a sharp, fast reaction AND fired while the
    volatility regime was at or above min_volatility_pctile — the shared
    entry-criteria filter behind both build_experiment_events (which
    additionally requires the trade's own forward_bars exit to already
    exist in df, to compute a real backtested return) and
    find_live_validated_signal (which doesn't — a signal that's live
    right now by definition hasn't exited yet). Returns a list of dicts
    (touch_pos, confirm_pos, direction, zone_top, zone_bottom,
    zone_start), in whatever order the detector found the zones —
    callers sort by touch_pos/confirm_pos themselves if order matters."""
    detect_fn = DETECTOR_FNS[detector_name]
    zones = detect_fn(df, max_scan_bars=max_scan_bars, record_history=True)
    if not zones:
        return []

    vol_pctile = volatility_percentile(df)
    atr = atr_series(df)
    close = (df["Close"] if "Close" in df else df["close"])
    close_arr = close.to_numpy()
    pos_by_time = {t: i for i, t in enumerate(df.index)}

    touches = []
    for zone in zones:
        reaction = _zone_reaction(df, zone, reaction_window, reaction_mult, atr, close_arr, pos_by_time)
        if reaction is None or not reaction["qualifies"]:
            continue
        touch_pos = reaction["touch_pos"]
        vp = vol_pctile.iloc[touch_pos]
        if pd.isna(vp) or vp < min_volatility_pctile:
            continue
        touches.append({"touch_pos": touch_pos, "confirm_pos": reaction["confirm_pos"],
                         "direction": reaction["direction"],
                         "zone_top": zone["top"], "zone_bottom": zone["bottom"], "zone_start": zone["start"]})
    return touches


@st.cache_data(ttl=_CACHE_TTL)
def build_experiment_events(df, detector_name, min_volatility_pctile, reaction_window,
                             reaction_mult, forward_bars, max_scan_bars=2000):
    """events_df (entry_time, raw_return, direction, plus the triggering
    zone's own top/bottom/start — carried through for find_live_validated_
    signal's own stop-price lookup, unused by research/evidence.run_event_study
    which only reads entry_time/raw_return/direction). Built by filtering
    EVERY historical zone this detector found (over whatever history `df`
    covers — switch the chart's own period/timeframe for more) down to
    the ones that both got a sharp, fast reaction AND fired while the
    volatility regime was at or above min_volatility_pctile at the
    moment of that reaction. raw_return is the ACTUAL forward return
    over forward_bars bars AFTER the reaction itself already confirmed
    (from confirm_pos, i.e. touch_pos + reaction_window — never from
    touch_pos directly; see _zone_reaction's own comment on why that
    would double-count the qualifying move as part of the return being
    tested) — never adjusted, never assumed."""
    touches = _qualifying_touches(df, detector_name, min_volatility_pctile, reaction_window,
                                   reaction_mult, max_scan_bars)
    cols = ["entry_time", "raw_return", "direction", "zone_top", "zone_bottom", "zone_start"]
    if not touches:
        return pd.DataFrame(columns=cols)

    close = (df["Close"] if "Close" in df else df["close"])
    close_arr = close.to_numpy()
    n = len(df)

    # One trade at a time, same "no overlapping positions" rule
    # backtest_custom_rule already enforces elsewhere in this project —
    # confirmed as a real bug via a wide sweep across 55 tickers: without
    # this, touches from DIFFERENT zones (or different detectors — FVG,
    # Order Block, IFVG, Breaker Block all scan the same candles)
    # routinely land on the same bar or a few bars apart during the same
    # market move, so their forward_bars return windows overlap almost
    # entirely. run_deep_backtest's permutation test treats every row of
    # events_df as an independent trial; a handful of real regime moves
    # each counted 3-5x over as "separate confirmations" makes a p-value
    # look far more significant than the actual independent evidence
    # supports — this is why the wide sweep briefly showed "3800+
    # survivors" across nearly every ticker tested before this fix.
    # Sorting by confirm_pos and skipping any touch confirming inside the
    # PREVIOUS kept touch's own still-open holding window (confirm_pos <=
    # last_exit_pos) keeps only the first touch of each overlapping
    # cluster — restores real independence between rows.
    touches = sorted(touches, key=lambda t: t["confirm_pos"])
    rows = []
    last_exit_pos = -1
    for t in touches:
        confirm_pos = t["confirm_pos"]
        if confirm_pos <= last_exit_pos:
            continue
        exit_pos = confirm_pos + forward_bars
        if exit_pos >= n:
            continue
        # Entry is confirm_pos, not touch_pos: you can only realistically
        # act once the reaction has actually confirmed, and raw_return
        # must be measured from the SAME point the entry criterion
        # stopped looking, or the qualifying move gets counted twice —
        # once as "why this qualified," once as "the return it earned."
        fwd_return = float((close_arr[exit_pos] - close_arr[confirm_pos]) / close_arr[confirm_pos])
        rows.append({"entry_time": df.index[confirm_pos], "raw_return": fwd_return, "direction": t["direction"],
                     "zone_top": t["zone_top"], "zone_bottom": t["zone_bottom"], "zone_start": t["zone_start"]})
        last_exit_pos = exit_pos
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# A second, unrelated strategy family — not zone-based like FVG/OB/IFVG/
# Breaker Block above, an actual trend-following idea: a fast/slow EMA
# cross marks a new trend, then wait for price to pull back and RETEST
# the fast EMA (not chase the cross itself), and only take that retest
# once momentum (RSI) shows it's actually troughing/turning back the
# trend's way — not a random 1-bar wiggle. Direct request: "moving
# average buy at touch after cross... add some momentum indicator... to
# really land a trade near bottom, as trend starts shifting for real."
# Shares run_deep_backtest/load_experiment_trials with the zone-based
# strategies above (same events_df shape, same permutation test, same
# accumulated BH-corrected trial log — "how many things were tried"
# has to count this family's trials too, not a separate pool) but has
# its OWN entry mechanics entirely, so it gets its own event builder
# rather than being bolted onto _qualifying_touches.
MA_PERIOD_PAIRS = [(10, 20), (20, 50), (50, 200)]


@st.cache_data(ttl=_CACHE_TTL)
def build_ma_cross_retest_events(df, fast_period, slow_period, rsi_dip_threshold,
                                  max_bars_after_cross, forward_bars, rsi_period=14):
    """events_df (entry_time, raw_return, direction) — same shape
    build_experiment_events returns, so it flows through run_deep_backtest
    unchanged. Three-part rule, each part checkable causally (no
    lookahead — every value used to decide bar j only reads data up to
    and including bar j):

      1. CROSS: fast EMA crosses from one side of slow EMA to the other
         — the trend just changed (or re-asserted). Direction follows the
         cross (bullish = fast crossed above).
      2. RETEST: scan forward (up to max_bars_after_cross bars) for the
         first bar whose [low, high] range actually touches the fast EMA
         again — a pullback to the new trend's own moving average, not a
         chase of the cross candle itself. Voided if the fast/slow
         relationship flips back before a touch happens (the "trend
         shift" didn't hold, this was noise).
      3. MOMENTUM CONFIRMATION, at the touch bar itself: RSI must have
         genuinely dipped (bullish: RSI the bar BEFORE the touch was at
         or below rsi_dip_threshold; bearish: at or above 100 minus it)
         AND be turning back the trend's own direction exactly ON the
         touch bar (RSI[touch] > RSI[touch-1] for bullish, < for
         bearish) — "land near bottom" as a real, checkable requirement,
         not just "touched the average."

    Entry is the touch bar itself (the LAST bar whose data is used to
    decide the trade qualifies) — raw_return is measured forward from
    THAT bar, over forward_bars bars, never from the earlier cross bar:
    measuring from an earlier point would re-count part of the already-
    used qualifying information as if it were new evidence, the exact
    bug build_experiment_events' own comments document being fixed
    for the zone-based strategies above. One trade at a time (skip any
    touch inside a still-open previous trade's own holding window) —
    same "no overlapping positions" rule as everywhere else in this
    module."""
    cols = ["entry_time", "raw_return", "direction"]
    close = (df["Close"] if "Close" in df else df["close"])
    high = (df["High"] if "High" in df else df["high"])
    low = (df["Low"] if "Low" in df else df["low"])
    n = len(df)
    if n < slow_period + 5:
        return pd.DataFrame(columns=cols)

    from indicators import ema as _ema, rsi as _rsi
    fast_ma = _ema(close, fast_period).to_numpy()
    slow_ma = _ema(close, slow_period).to_numpy()
    rsi_arr = _rsi(close, rsi_period).to_numpy()
    close_arr = close.to_numpy()
    high_arr = high.to_numpy()
    low_arr = low.to_numpy()
    diff = fast_ma - slow_ma

    touches = []  # (touch_pos, direction)
    i = 0
    while i < n - 1:
        if np.isnan(diff[i]) or np.isnan(diff[i + 1]):
            i += 1
            continue
        crossed_bullish = diff[i] <= 0 and diff[i + 1] > 0
        crossed_bearish = diff[i] >= 0 and diff[i + 1] < 0
        if not (crossed_bullish or crossed_bearish):
            i += 1
            continue
        direction = "bullish" if crossed_bullish else "bearish"
        cross_pos = i + 1
        touch_pos = None
        for j in range(cross_pos + 1, min(n, cross_pos + 1 + max_bars_after_cross)):
            if np.isnan(fast_ma[j]) or np.isnan(slow_ma[j]):
                continue
            still_trending = (fast_ma[j] > slow_ma[j]) if direction == "bullish" else (fast_ma[j] < slow_ma[j])
            if not still_trending:
                break  # trend flipped back before ever retesting -- void, not a real shift
            if low_arr[j] <= fast_ma[j] <= high_arr[j]:
                touch_pos = j
                break
        if touch_pos is None:
            i = cross_pos
            continue
        if touch_pos >= 1 and not (np.isnan(rsi_arr[touch_pos]) or np.isnan(rsi_arr[touch_pos - 1])):
            if direction == "bullish":
                momentum_ok = (rsi_arr[touch_pos - 1] <= rsi_dip_threshold) and (rsi_arr[touch_pos] > rsi_arr[touch_pos - 1])
            else:
                momentum_ok = (rsi_arr[touch_pos - 1] >= (100 - rsi_dip_threshold)) and (rsi_arr[touch_pos] < rsi_arr[touch_pos - 1])
            if momentum_ok:
                touches.append((touch_pos, direction))
        i = cross_pos

    if not touches:
        return pd.DataFrame(columns=cols)

    touches = sorted(touches, key=lambda t: t[0])
    rows = []
    last_exit_pos = -1
    for pos, direction in touches:
        if pos <= last_exit_pos:
            continue
        exit_pos = pos + forward_bars
        if exit_pos >= n:
            continue
        fwd_return = float((close_arr[exit_pos] - close_arr[pos]) / close_arr[pos])
        rows.append({"entry_time": df.index[pos], "raw_return": fwd_return, "direction": direction})
        last_exit_pos = exit_pos
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# A third strategy family, deliberately the opposite BET of the two above
# (both were continuation — does the move keep going — and both came up
# empty across 55 tickers): mean reversion off a real Bollinger Band
# divergence. Not "touched the band" alone (a band touch during a strong
# trend is common and means little) — a genuine RSI DIVERGENCE between
# two touches of the same band: price makes a new extreme but momentum
# doesn't confirm it, the classic "the move is running out of gas" tell.
@st.cache_data(ttl=_CACHE_TTL)
def build_bb_divergence_events(df, num_std, lookback_bars, forward_bars, bb_period=20, rsi_period=14):
    """events_df (entry_time, raw_return, direction) — same shape every
    other builder in this module returns. Rule, entirely causal:

      1. A "touch" is any bar whose [low, high] range reaches the lower
         Bollinger Band (candidate bullish) or the upper band (candidate
         bearish) — an extreme by definition, not yet a signal.
      2. For each touch at bar j, look back at the MOST RECENT PRIOR
         touch of the SAME band, bar k (k < j, j-k <= lookback_bars).
         Divergence: price made a WORSE extreme at j than at k (a lower
         close for the lower band, a higher close for the upper band)
         while RSI did NOT — RSI at j is actually better than at k. That
         mismatch (price extreme confirmed, momentum extreme NOT
         confirmed) is the divergence; direction is the reversal a
         real divergence implies (bullish off the lower band, bearish
         off the upper).
      3. Entry is bar j itself — the SECOND touch, the one where the
         divergence actually confirms; k is only ever read for its own
         OLDER close/RSI values, never used to decide anything about
         bar j itors own future. raw_return is measured forward from j,
         over forward_bars bars — never from k, same discipline as
         build_experiment_events/build_ma_cross_retest_events (the
         qualifying comparison and the tested outcome must not overlap).

    One trade at a time (skip any touch inside a still-open previous
    trade's own holding window) — same rule as every other builder
    here."""
    from indicators import bollinger_bands, rsi as _rsi
    cols = ["entry_time", "raw_return", "direction"]
    close = (df["Close"] if "Close" in df else df["close"])
    high = (df["High"] if "High" in df else df["high"])
    low = (df["Low"] if "Low" in df else df["low"])
    n = len(df)
    if n < bb_period + 5:
        return pd.DataFrame(columns=cols)

    upper, _, lower = bollinger_bands(close, bb_period, num_std)
    rsi_arr = _rsi(close, rsi_period).to_numpy()
    upper_arr, lower_arr = upper.to_numpy(), lower.to_numpy()
    close_arr, high_arr, low_arr = close.to_numpy(), high.to_numpy(), low.to_numpy()

    lower_touches = [j for j in range(n) if not np.isnan(lower_arr[j]) and low_arr[j] <= lower_arr[j]]
    upper_touches = [j for j in range(n) if not np.isnan(upper_arr[j]) and high_arr[j] >= upper_arr[j]]

    events = []
    for touches, side in ((lower_touches, "bullish"), (upper_touches, "bearish")):
        for idx in range(1, len(touches)):
            j, k = touches[idx], touches[idx - 1]
            if j - k > lookback_bars or np.isnan(rsi_arr[j]) or np.isnan(rsi_arr[k]):
                continue
            if side == "bullish":
                diverges = close_arr[j] < close_arr[k] and rsi_arr[j] > rsi_arr[k]
            else:
                diverges = close_arr[j] > close_arr[k] and rsi_arr[j] < rsi_arr[k]
            if diverges:
                events.append((j, side))

    if not events:
        return pd.DataFrame(columns=cols)

    events = sorted(events, key=lambda e: e[0])
    rows = []
    last_exit_pos = -1
    for pos, direction in events:
        if pos <= last_exit_pos:
            continue
        exit_pos = pos + forward_bars
        if exit_pos >= n:
            continue
        fwd_return = float((close_arr[exit_pos] - close_arr[pos]) / close_arr[pos])
        rows.append({"entry_time": df.index[pos], "raw_return": fwd_return, "direction": direction})
        last_exit_pos = exit_pos
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# A fourth strategy family, different in kind from the three above: no
# indicator at all (no ATR, no EMA, no RSI, no Bollinger Bands) — just the
# raw sequence of candle closes. Direct request: "a system that analyses
# how each candle close related to the previous one and choose to hold or
# close based on that." Both the ENTRY and the EXIT are driven purely by
# consecutive close-to-close direction, nothing else: enter once N
# straight closes have gone the same way (a raw momentum confirmation),
# stay in as long as that keeps being true, get out once enough
# consecutive closes go the other way. This is a genuinely different BET
# from the three continuation/reversion ideas above too: those all used a
# FIXED holding period (forward_bars) regardless of what happened along
# the way; this one's hold length is itself the thing being tested —
# does managing the exit off live price action beat exiting on a timer.
@st.cache_data(ttl=_CACHE_TTL)
def build_candle_momentum_events(df, entry_confirm_bars, exit_tolerance_bars, max_hold_bars):
    """events_df (entry_time, raw_return, direction) — same shape every
    other builder here returns, but exit_pos is no longer entry_pos +
    forward_bars; it's wherever the rule below actually closes the trade.

    ENTRY: the last entry_confirm_bars candles all closed the same
    direction in a row (all up = bullish, all down = bearish) — a raw,
    no-indicator momentum confirmation. Entry is the bar where the Nth
    straight close just confirmed (using only closes up to and including
    that bar — no lookahead).

    HOLD/EXIT, decided bar by bar going forward, each decision using only
    that bar's own just-revealed close: track a running streak of
    consecutive closes AGAINST the position. Any close that DOESN'T go
    against the position resets that streak to zero (a bar that pauses or
    goes back your way earns the trade another chance, not an instant
    exit on the very next wobble). Once the against-streak reaches
    exit_tolerance_bars, exit AT that bar's close. If neither the
    tolerance nor max_hold_bars is hit, exit at max_hold_bars as a safety
    cap — real trades don't run forever just because the rule never
    happened to trigger.

    One trade at a time is structural here, not a separate dedup pass:
    scanning resumes strictly AFTER the previous trade's own exit_pos, so
    trades can never overlap by construction."""
    cols = ["entry_time", "raw_return", "direction"]
    close = (df["Close"] if "Close" in df else df["close"])
    close_arr = close.to_numpy()
    n = len(df)
    if n < entry_confirm_bars + 2:
        return pd.DataFrame(columns=cols)

    step = np.sign(np.diff(close_arr))  # step[k] = sign(close[k+1] - close[k]), length n-1

    rows = []
    i = entry_confirm_bars
    while i < n:
        window = step[i - entry_confirm_bars:i]
        if len(window) < entry_confirm_bars:
            i += 1
            continue
        if np.all(window == 1):
            direction = "bullish"
        elif np.all(window == -1):
            direction = "bearish"
        else:
            i += 1
            continue

        entry_pos = i
        adverse_run = 0
        exit_pos = None
        scan_end = min(n, entry_pos + 1 + max_hold_bars)
        for k in range(entry_pos + 1, scan_end):
            bar_dir = step[k - 1]  # direction of THIS bar's own close vs the previous one
            is_adverse = (bar_dir == -1) if direction == "bullish" else (bar_dir == 1)
            if is_adverse:
                adverse_run += 1
                if adverse_run >= exit_tolerance_bars:
                    exit_pos = k
                    break
            else:
                adverse_run = 0
        if exit_pos is None:
            exit_pos = min(n - 1, entry_pos + max_hold_bars)
        if exit_pos <= entry_pos or exit_pos >= n:
            i = entry_pos + 1
            continue

        fwd_return = float((close_arr[exit_pos] - close_arr[entry_pos]) / close_arr[entry_pos])
        rows.append({"entry_time": df.index[entry_pos], "raw_return": fwd_return, "direction": direction})
        i = exit_pos + 1
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# A fifth strategy family: the classic, named "trade the session open"
# pattern (Opening Range Breakout) rather than anything built earlier —
# none of the four above anchor to a specific intraday reference range.
# Direct request: NY session, the open specifically, and days with no
# high-impact news filtered out entirely (not just a narrow window around
# the release — a real news DAY is often erratic well beyond any 15/30-
# minute blackout window, and "no news days" was explicit).
@st.cache_data(ttl=_CACHE_TTL)
def build_orb_events(df, ticker, range_minutes, forward_bars, max_bars_after_range=8, no_news_days=True):
    """events_df (entry_time, raw_return, direction) — same shape every
    other builder here returns. Rule, entirely causal, one trade per day:

      1. RANGE: the high/low of NY session bars from 09:30 ET to
         09:30+range_minutes ET — the "opening range."
      2. BREAKOUT: scan forward from the end of the range (up to
         max_bars_after_range bars, never past 11:30 ET) for the first
         bar whose CLOSE clears the range high (bullish) or range low
         (bearish) — a close beyond it, not a wick, same "closes beyond"
         convention detect_ifvgs/detect_breaker_blocks already use
         elsewhere in this project for confirming a break is real.
      3. NO-NEWS DAYS: if no_news_days, any calendar day carrying a High-
         impact release for `ticker`'s own relevant currencies (see
         news.ticker_currencies/high_impact_events_in_range) is skipped
         ENTIRELY — every bar on that day, not just a window around the
         release itself; ticker_currencies returning empty (no curated
         mapping for this ticker) is a silent no-op, not a full exclusion,
         same honest-default convention news.py already documents.

    Entry is the breakout bar itself; raw_return is measured forward from
    THERE over forward_bars bars — never from the range's own bars, same
    no-overlap discipline as every builder above. At most one trade per
    calendar day by construction (one breakout scan per day)."""
    from datetime import time as dtime
    import news as _news

    cols = ["entry_time", "raw_return", "direction"]
    close = (df["Close"] if "Close" in df else df["close"])
    high = (df["High"] if "High" in df else df["high"])
    low = (df["Low"] if "Low" in df else df["low"])
    if df.empty:
        return pd.DataFrame(columns=cols)

    idx_ny = df.index.tz_convert("America/New_York")
    dates = idx_ny.normalize()
    times = idx_ny.time
    close_arr = close.to_numpy()
    high_arr = high.to_numpy()
    low_arr = low.to_numpy()
    n = len(df)

    excluded_dates = set()
    if no_news_days:
        currencies = _news.ticker_currencies(ticker)
        if currencies:
            events = _news.high_impact_events_in_range(idx_ny.min(), idx_ny.max(), currencies=currencies)
            if not events.empty:
                excluded_dates = set(pd.DatetimeIndex(events["time"]).tz_convert("America/New_York").normalize())

    open_t = dtime(9, 30)
    range_end_minute = 9 * 60 + 30 + range_minutes
    range_end_t = dtime(range_end_minute // 60, range_end_minute % 60)
    session_end_t = dtime(11, 30)

    rows = []
    last_exit_pos = -1
    for d in pd.unique(dates):
        if d in excluded_dates:
            continue
        day_positions = np.where(dates == d)[0]
        if len(day_positions) == 0:
            continue
        range_positions = [p for p in day_positions if open_t <= times[p] < range_end_t]
        if not range_positions:
            continue
        range_high = high_arr[range_positions].max()
        range_low = low_arr[range_positions].min()
        after_positions = [p for p in day_positions if range_end_t <= times[p] < session_end_t]

        entry_pos, direction = None, None
        for count, p in enumerate(after_positions):
            if count >= max_bars_after_range:
                break
            if p <= last_exit_pos:
                continue
            if close_arr[p] > range_high:
                entry_pos, direction = p, "bullish"
                break
            if close_arr[p] < range_low:
                entry_pos, direction = p, "bearish"
                break
        if entry_pos is None:
            continue

        exit_pos = entry_pos + forward_bars
        if exit_pos >= n:
            continue
        fwd_return = float((close_arr[exit_pos] - close_arr[entry_pos]) / close_arr[entry_pos])
        rows.append({"entry_time": df.index[entry_pos], "raw_return": fwd_return, "direction": direction})
        last_exit_pos = exit_pos
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def preview_stats(events_df, cost_bps=10.0):
    """Instant, no-permutation-test feedback for tuning sliders live — n /
    win-rate / mean-return only, cheap enough to recompute on every widget
    change. NOT a validated result on its own — see run_deep_backtest for
    that; this only exists to tell you if a setting is worth testing
    properly at all before spending a real permutation test on it."""
    if events_df.empty:
        return {"n": 0, "win_rate": None, "mean_return": None}
    cost = cost_bps / 10_000.0
    direction = np.where(events_df["direction"] == "bullish", 1, -1)
    signed = events_df["raw_return"].to_numpy() * direction - cost
    return {"n": len(events_df), "win_rate": float((signed > 0).mean()), "mean_return": float(signed.mean())}


def run_deep_backtest(ticker, tf_label, events_df, settings, cost_bps=10.0,
                       train_fraction=0.8, n_permutations=2000, label=None):
    """The real version: 80/20 time split, permutation test on each side
    (research/evidence.run_event_study, unmodified), same-sign check
    between train and holdout (the gap institutional_backtest.py found
    and fixed — a trial can clear significance on both splits while they
    disagree on direction; that's a near-constant classifier riding one
    regime in train and the opposite in holdout, not a real edge).
    Logged to the accumulating trials file regardless of outcome — see
    this module's own docstring on why. Returns the trial dict that was
    appended.

    label: None (default) builds the zone-based strategies' own label
    straight from `settings` (detector/min_volatility_pctile/reaction_mult/
    reaction_window/forward_bars) — every existing call site. A caller
    whose settings dict has a different shape (e.g. build_ma_cross_retest_
    events' fast/slow/rsi_dip/max_bars/forward_bars) must pass its own
    pre-built label instead, since there's no one settings shape every
    strategy family shares — only events_df's own (entry_time, raw_return,
    direction) contract is universal."""
    if label is None:
        label = (f"{settings['detector']} · vol≥{settings['min_volatility_pctile']:.0%} · "
                 f"react {settings['reaction_mult']}x/{settings['reaction_window']}b · "
                 f"fwd {settings['forward_bars']}b")
    trial = {
        "logged_at": pd.Timestamp.now("UTC").isoformat(), "ticker": ticker, "tf_label": tf_label,
        "label": label, "settings": settings, "n_events": len(events_df),
    }
    if len(events_df) < 30:
        trial["verdict"] = "INSUFFICIENT_DATA"
        _append_trial(trial)
        return trial

    events_df = events_df.sort_values("entry_time").reset_index(drop=True)
    cutoff_pos = int(len(events_df) * train_fraction)
    cutoff = events_df["entry_time"].iloc[cutoff_pos]
    train = events_df[events_df["entry_time"] < cutoff]
    holdout = events_df[events_df["entry_time"] >= cutoff]
    train_result = run_event_study(train, cost_bps=cost_bps, direction_col="direction",
                                    n_permutations=n_permutations)
    holdout_result = run_event_study(holdout, cost_bps=cost_bps, direction_col="direction",
                                      n_permutations=n_permutations)

    n_train = train_result.get("n_events", 0)
    if n_train == 0:
        trial["verdict"] = "INSUFFICIENT_DATA"
        _append_trial(trial)
        return trial

    p_train = train_result.get("p_value_vs_random_direction")
    mean_train = train_result.get("mean_return_after_cost")
    mean_holdout = holdout_result.get("mean_return_after_cost")
    trial.update({
        "n_train": n_train, "n_holdout": holdout_result.get("n_events", 0),
        "mean_return_train": mean_train, "p_value_train": p_train,
        "mean_return_holdout": mean_holdout,
        "holdout_verdict": "PASSED" if holdout_result.get("verdict") == "EDGE_FOUND" else "FAILED",
        "same_sign": bool(mean_train is not None and mean_holdout is not None
                           and mean_train > 0 and mean_holdout > 0),
        "verdict": "SCORED",
    })
    _append_trial(trial)
    return trial


def _append_trial(trial):
    try:
        os.makedirs(os.path.dirname(_TRIALS_PATH), exist_ok=True)
        with open(_TRIALS_PATH, "a") as f:
            f.write(json.dumps(trial, default=str) + "\n")
    except Exception:
        pass


def load_experiment_trials():
    """Every trial ever run through run_deep_backtest, BH-corrected
    across the FULL log (not just today's session) — same convention as
    research/setups.py's load_edge_lab_validation. Empty DataFrame if
    nothing's been run yet.

    Thin, uncached wrapper around _load_experiment_trials_cached: this
    file has grown to 80k+ lines/30+MB over the course of a single day's
    testing, and re-reading/re-parsing/re-BH-correcting the WHOLE thing
    had no caching at all -- confirmed as a real, live performance bug,
    not a hypothetical one: every caller of this function sits inside a
    Streamlit tab/expander body, and Streamlit executes EVERY tab's body
    on EVERY rerun regardless of which one is actually visible (confirmed
    directly elsewhere in this project). With an auto-refreshing chart
    fragment on intraday timeframes, that meant a ~0.4s+ file parse plus a
    full Benjamini-Hochberg pass across 22,000+ scored rows, repeated on
    close to every single tick, whether or not anyone had the Experiments
    tab open -- exactly the kind of continuous, unnecessary CPU load that
    reads as "the app clogs sometimes." os.stat() is effectively free, so
    checking it fresh every call and only doing the real work when the
    file has genuinely changed (a NEW trial got appended) costs nothing
    while fixing everything."""
    if not os.path.exists(_TRIALS_PATH):
        return pd.DataFrame()
    stat = os.stat(_TRIALS_PATH)
    return _load_experiment_trials_cached(stat.st_mtime, stat.st_size)


# Different sweep scripts across this project's history have logged the
# same real timeframes under different spellings — the ORIGINAL detector-
# reaction family (and app.py's own TIMEFRAMES dict) always writes
# "1D"/"1W"/"1M"; research/data_loader.py's INTERVAL_LABELS (used by a
# same-day all-timeframes sweep for clean_expansion_retracement) spells
# the same three "1d"/"1wk"->"1W"/"1mo"->"1M" (1d specifically
# lowercase). Normalized once here, at the single root every caller of
# load_experiment_trials() reads through (list_validated_pairs,
# find_live_validated_signal, ticker_behavior.py's own rebuild_db), so
# nothing downstream needs its own copy of this map — confirmed as a
# real bug: list_validated_pairs surfacing a lowercase "1d" pair straight
# into app.py's TIMEFRAMES["1d"] (only "1D" exists there) crashed the
# sidebar's "Scan for validated setups" the moment clean_expansion_
# retracement's first 1d survivor showed up.
TF_LABEL_NORMALIZE = {"1d": "1D", "1wk": "1W", "1mo": "1M"}


@st.cache_data(ttl=_CACHE_TTL)
def _load_experiment_trials_cached(_mtime, _size):
    """The actual read+parse+BH-correct, cached on (mtime, size) so
    Streamlit's own cache key changes exactly when the file's contents
    do -- not a blind time-based TTL, which would either serve stale
    results right after a fresh 'Run deep backtest' or still redo the
    full pass on a timer regardless of whether anything changed."""
    trials = []
    with open(_TRIALS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                trials.append(json.loads(line))
    if not trials:
        return pd.DataFrame()

    df = pd.DataFrame(trials)
    df["tf_label"] = df["tf_label"].map(lambda t: TF_LABEL_NORMALIZE.get(t, t))
    scored = df[df["verdict"] == "SCORED"]
    if not scored.empty:
        q_values, significant = benjamini_hochberg(scored["p_value_train"].tolist(), alpha=0.05)
        df.loc[scored.index, "q_value_train"] = q_values
        df.loc[scored.index, "bh_significant"] = significant
        df["survived"] = False
        df.loc[scored.index, "survived"] = (
            df.loc[scored.index, "bh_significant"].fillna(False)
            & (df.loc[scored.index, "holdout_verdict"] == "PASSED")
            & df.loc[scored.index, "same_sign"].fillna(False)
        )
    return df.sort_values("logged_at", ascending=False).reset_index(drop=True)


def list_validated_pairs(tickers):
    """Every (ticker, tf_label) combo among `tickers` that currently has
    at least one BH-corrected, same-sign-confirmed validated trial in the
    accumulated experiment log — drives the sidebar's "Validated edge"
    scan so it only ever checks combos that have actually earned the
    right to be checked live, instead of a hardcoded guess at which
    ticker/timeframe pairs might matter. Grows on its own as more
    Experiments tab backtests get run and survive correction — nothing
    to maintain by hand. Empty list when nothing's survived yet for any
    of these tickers, the honest default most of the time."""
    trials = load_experiment_trials()
    if trials.empty:
        return []
    survived = trials[(trials.get("survived", False) == True) & (trials["ticker"].isin(tickers))]
    return sorted(set(zip(survived["ticker"], survived["tf_label"])))


def find_live_validated_signal(df, ticker, tf_label):
    """Checks whether any of this (ticker, tf_label)'s own BH-corrected,
    same-sign-confirmed validated experiments (see load_experiment_trials)
    has a signal that's LIVE right now on df — touched recently enough
    that this trial's own forward_bars hold hasn't finished yet. Same
    entry criteria as the Experiments tab's own "Run deep backtest"
    button (_qualifying_touches), just checked against the most recent
    bars instead of replayed across the full history — the live-scan
    counterpart to that manual check, driving the sidebar instead.

    Checks the STRONGEST validated trial for this ticker+timeframe first
    (lowest train p-value) and returns the first one with a live touch.
    None when nothing's validated yet for this exact ticker+timeframe, or
    nothing's live right now — the normal case, most of the time, for
    most tickers. An honest empty result, not an error.

    The returned target_price is a PROJECTION of the historical holdout
    mean return, not a real take-profit level — this kind of edge exits
    by TIME (forward_bars), not by price reaching anywhere specific.
    Callers must label it as such rather than presenting it as a hard
    target the way a zone-edge stop genuinely is."""
    trials = load_experiment_trials()
    if trials.empty or df.empty:
        return None
    matches = trials[(trials["ticker"] == ticker) & (trials["tf_label"] == tf_label)
                      & (trials.get("survived", False) == True)]
    if matches.empty:
        return None
    matches = matches.sort_values("p_value_train")
    for _, trial in matches.iterrows():
        touches = _touches_for_settings(df, trial)
        result = _signal_from_touches(df, ticker, tf_label, trial, touches, require_live=True)
        if result is not None:
            return result
    return None


def _touches_for_settings(df, trial):
    """Dispatches to whichever strategy family a trial belongs to.
    Settings-SHAPE alone stopped being unique once clean_expansion_
    retracement and clean_retracement_resumption both landed on the
    identical {min_streak, retr_window_bars, forward_bars} shape (same
    detection core, see _scan_clean_zone_streaks — they only disagree on
    what to DO with the streaks found) — dispatching on the trial's own
    `label` prefix (every build_* function's own family name, always the
    text before the first " · ") is the only thing guaranteed unique per
    strategy, so that's the primary key; settings-shape is kept only as
    the original family's own fallback, from before this needed to be
    unambiguous. Add a new branch here whenever a new strategy family
    earns its own validated survivor, not before."""
    label = trial["label"] if isinstance(trial, (pd.Series, dict)) else ""
    settings = trial["settings"] if isinstance(trial, (pd.Series, dict)) else trial
    family = str(label).split(" · ", 1)[0].strip() if label else ""
    if family == "clean_expansion_retracement":
        return _clean_expansion_touches(df, settings["min_streak"], settings["retr_window_bars"])
    if family == "clean_retracement_resumption":
        return _clean_retracement_resumption_touches(df, settings["min_streak"], settings["retr_window_bars"])
    if family == "clean_structure_trend":
        return _clean_structure_touches(df, settings["min_swings"],
                                         settings.get("atr_period", 14), settings.get("atr_mult", 1.5))
    if family == "clean_expansion_liquidity":
        return _clean_expansion_liquidity_touches(df, settings["min_streak"], settings["retr_window_bars"],
                                                   settings.get("sweep_window_bars", 10))
    if "detector" in settings:
        return _qualifying_touches(df, settings["detector"], settings["min_volatility_pctile"],
                                    settings["reaction_window"], settings["reaction_mult"])
    if "min_streak" in settings:
        return _clean_expansion_touches(df, settings["min_streak"], settings["retr_window_bars"])
    return []


def _signal_from_touches(df, ticker, tf_label, trial, touches, require_live):
    """Shared by find_live_validated_signal (require_live=True — only
    ever returns a signal still inside its own forward_bars hold) and
    latest_validated_example (require_live=False — the Survivors tab's
    "apply to chart" wants to show the most recent instance even after
    its hold period is long over, honestly labeled as such via the
    returned is_live flag). One place computing entry/stop/target from a
    touch, so the two call sites can't drift apart on the math."""
    if not touches:
        return None
    n = len(df)
    close_arr = (df["Close"] if "Close" in df else df["close"]).to_numpy()
    settings = trial["settings"]
    latest = max(touches, key=lambda t: t["confirm_pos"])
    bars_since = n - 1 - latest["confirm_pos"]
    if require_live and bars_since > settings["forward_bars"]:
        return None  # already past this trial's own hold length -- no longer live
    direction = latest["direction"]
    # Entry is confirm_pos, not touch_pos — see build_experiment_events'
    # own comment: you can only realistically act once the reaction has
    # actually confirmed, and this must match how the trial itself
    # measured its own p-value/returns or the live pick would be
    # showing a different (and again look-ahead-biased) entry point
    # than what was actually validated.
    entry_price = float(close_arr[latest["confirm_pos"]])
    # The zone's own far edge is the natural stop for the original
    # detector-reaction family (zone direction == trade direction, entry
    # sits at/near the zone). clean_expansion_retracement's own touches
    # are the OPPOSITE case by construction (see _clean_expansion_touches
    # — the zone here is the retracement leg, trade direction is the
    # ORIGINAL expansion) and its entry (next bar's open after the zone's
    # own confirming candle) can legitimately land already past that
    # zone's edge — confirmed directly on a live BTC-USD 1h signal, where
    # the "stop" would otherwise have sat ABOVE a bullish entry. Clamping
    # to stay strictly on the risk side of entry keeps this correct for
    # both families with no behavior change for the original one (whose
    # zone edge is already on the right side in practice).
    if direction == "bullish":
        stop_price = latest["zone_bottom"] if latest["zone_bottom"] < entry_price else entry_price * 0.999
    else:
        stop_price = latest["zone_top"] if latest["zone_top"] > entry_price else entry_price * 1.001
    sign = 1 if direction == "bullish" else -1
    target_price = entry_price * (1 + sign * abs(trial["mean_return_holdout"]))
    return {
        "ticker": ticker, "tf_label": tf_label, "direction": direction,
        "entry_price": entry_price, "stop_price": float(stop_price), "target_price": float(target_price),
        "label": trial["label"], "p_value_train": trial["p_value_train"],
        "mean_return_train": trial["mean_return_train"], "mean_return_holdout": trial["mean_return_holdout"],
        "n_events": int(trial["n_events"]), "entry_time": df.index[latest["confirm_pos"]],
        "bars_since_touch": bars_since, "forward_bars": settings["forward_bars"],
        "zone_start": latest["zone_start"], "zone_top": latest["zone_top"], "zone_bottom": latest["zone_bottom"],
        "is_live": bars_since <= settings["forward_bars"],
    }


def latest_validated_example(df, ticker, tf_label, trial):
    """Same result shape as find_live_validated_signal, for ONE specific
    trial (not "the strongest one for this ticker/timeframe") and WITHOUT
    requiring it to still be live — the Survivors tab's "apply to chart"
    button uses this so picking a validated strategy always shows
    something (its most recent instance, live or not), rather than
    silently doing nothing on the (usual) day nothing's live right now.
    None only when this trial's own settings never produced a single
    touch anywhere on df."""
    touches = _touches_for_settings(df, trial)
    return _signal_from_touches(df, ticker, tf_label, trial, touches, require_live=False)


# A genuinely different kind of bet from everything above: every prior
# strategy in this module tests DIRECTION (did price go the way the rule
# said). This one tests MAGNITUDE instead — does price move MORE around a
# real catalyst (a high-impact news release) than a synthetic "straddle"
# entered there would have cost to buy? A straddle (long a call + long a
# put, same strike/expiry) pays off on a big move either way and loses
# only if price sits still — so betting on magnitude sidesteps the
# direction question every other strategy here already found no edge in.
#
# No real options data is used — this project has none. The premium a
# straddle would have cost is approximated with the standard at-the-money
# approximation, premium ≈ 0.8 × σ × √T (0.8 ≈ √(2/π)), using the
# instrument's own TRAILING REALIZED volatility as σ. That's a
# deliberately GENEROUS proxy for the buyer: real implied volatility
# usually trades a bit richer than trailing realized vol (the well-
# documented volatility risk premium — option sellers get paid for
# bearing tail risk), so if this still comes up unprofitable against the
# generous proxy, real options would likely have been worse, not better.
def _rolling_realized_vol(close_arr, lookback_bars):
    """Trailing per-bar return stdev at every bar, computed causally —
    bar i's own value uses only returns up to and including bar i, never
    later ones. Not annualized; run_vol_harvest_backtest scales it by
    sqrt(forward_bars) itself, since both are already in the same
    per-bar units."""
    returns = np.diff(close_arr) / close_arr[:-1]
    vol = pd.Series(returns).rolling(lookback_bars, min_periods=lookback_bars).std()
    out = np.full(len(close_arr), np.nan)
    out[1:] = vol.to_numpy()
    return out


@st.cache_data(ttl=_CACHE_TTL)
def build_vol_harvest_events(df, ticker, forward_bars, vol_lookback_bars=100):
    """One row per bar (not just at catalysts — the non-catalyst bars ARE
    this test's own baseline population, see run_vol_harvest_backtest):
    realized_move (the |return| over forward_bars from that bar — what a
    straddle entered there would have needed to clear), premium_est (the
    ATM straddle premium estimate at that bar, as a fraction of price),
    and is_catalyst (True at the bar a High-impact news event for
    ticker's own relevant currencies lands on or just after). ticker
    having no curated currency mapping (news.ticker_currencies) leaves
    is_catalyst all False — an honest no-op, not a full exclusion, same
    convention news.py documents elsewhere."""
    import news as _news

    close = (df["Close"] if "Close" in df else df["close"])
    close_arr = close.to_numpy()
    n = len(df)
    if n < vol_lookback_bars + forward_bars + 2:
        return pd.DataFrame(columns=["entry_time", "realized_move", "premium_est", "is_catalyst"])

    vol = _rolling_realized_vol(close_arr, vol_lookback_bars)
    premium_est = 0.8 * vol * np.sqrt(forward_bars)

    is_catalyst = np.zeros(n, dtype=bool)
    currencies = _news.ticker_currencies(ticker)
    if currencies:
        idx_ny = df.index.tz_convert("America/New_York")
        events = _news.high_impact_events_in_range(idx_ny.min(), idx_ny.max(), currencies=currencies)
        if not events.empty:
            pos_by_time = {t: i for i, t in enumerate(df.index)}
            event_times = pd.DatetimeIndex(events["time"]).tz_convert(df.index.tz)
            for et in event_times:
                after = df.index[df.index >= et]
                if len(after):
                    is_catalyst[pos_by_time[after[0]]] = True

    rows = []
    for i in range(n):
        exit_pos = i + forward_bars
        if exit_pos >= n or np.isnan(premium_est[i]) or close_arr[i] <= 0:
            continue
        realized_move = abs(close_arr[exit_pos] - close_arr[i]) / close_arr[i]
        rows.append({"entry_time": df.index[i], "realized_move": realized_move,
                      "premium_est": float(premium_est[i]), "is_catalyst": bool(is_catalyst[i])})
    return pd.DataFrame(rows)


def run_vol_harvest_backtest(events_df, ticker, tf_label, forward_bars, n_permutations=2000,
                              train_fraction=0.8, min_catalysts=15, label=None):
    """events_df from build_vol_harvest_events (one ticker, or several
    pooled together — every row's own premium_est already reflects that
    row's own local vol, so pooling different instruments is fine, same
    reasoning as the session-timing sweep's own pooling).

    Hypothesis: mean(net_payoff) is HIGHER at catalyst bars than at
    ordinary bars, where net_payoff = realized_move - premium_est —
    i.e. news-driven moves are bigger than what the instrument's own
    trailing volatility would have priced an ATM straddle at, not just
    bigger in some absolute sense. Tested by permutation: repeatedly draw
    a same-size random group of bars from the FULL pool (catalyst bars
    included, matching how they'd be drawn from real data — this asks
    "would a random group of bars this size ever score this high," not
    "how do catalysts compare to non-catalysts specifically") and compare
    the real catalyst group's own mean against that null.

    Same discipline as every other builder here: 80/20 time split on the
    catalyst events themselves, both train and holdout must independently
    clear significance AND agree that net_payoff is positive (same_sign)
    before this counts as anything. Logged through the same accumulating,
    BH-corrected trial file (_append_trial/load_experiment_trials) as
    every directional strategy — "how many things were tried" has to
    include this test too, not a separate pool."""
    if label is None:
        label = f"VOL HARVEST fwd{forward_bars}b"
    trial = {
        "logged_at": pd.Timestamp.now("UTC").isoformat(), "ticker": ticker, "tf_label": tf_label,
        "label": label, "settings": {"strategy": "vol_harvest", "forward_bars": forward_bars},
    }
    events_df = events_df.sort_values("entry_time").reset_index(drop=True)
    catalysts = events_df[events_df["is_catalyst"]]
    trial["n_events"] = int(len(catalysts))
    if len(catalysts) < min_catalysts:
        trial["verdict"] = "INSUFFICIENT_DATA"
        _append_trial(trial)
        return trial

    cutoff_pos = int(len(catalysts) * train_fraction)
    cutoff_time = catalysts["entry_time"].iloc[min(cutoff_pos, len(catalysts) - 1)]
    train_catalysts = catalysts[catalysts["entry_time"] < cutoff_time]
    holdout_catalysts = catalysts[catalysts["entry_time"] >= cutoff_time]
    train_pool = events_df[events_df["entry_time"] < cutoff_time]["realized_move"] \
        - events_df[events_df["entry_time"] < cutoff_time]["premium_est"]
    holdout_pool = events_df[events_df["entry_time"] >= cutoff_time]["realized_move"] \
        - events_df[events_df["entry_time"] >= cutoff_time]["premium_est"]

    def _test_split(catalyst_sub, pool, tag):
        n_cat = len(catalyst_sub)
        if n_cat < min_catalysts // 2 or len(pool) <= n_cat:
            return None
        net = (catalyst_sub["realized_move"] - catalyst_sub["premium_est"]).to_numpy()
        pool_arr = pool.to_numpy()
        observed_mean = float(net.mean())
        rng = np.random.default_rng(0)
        null_means = np.empty(n_permutations)
        for k in range(n_permutations):
            sample = rng.choice(pool_arr, size=n_cat, replace=False)
            null_means[k] = sample.mean()
        p_value = float((null_means >= observed_mean).mean())
        return {"n": n_cat, "mean_net_payoff": observed_mean, "p_value": p_value}

    train_result = _test_split(train_catalysts, train_pool, "train")
    holdout_result = _test_split(holdout_catalysts, holdout_pool, "holdout")
    if train_result is None:
        trial["verdict"] = "INSUFFICIENT_DATA"
        _append_trial(trial)
        return trial

    trial.update({
        "n_train": train_result["n"],
        "n_holdout": holdout_result["n"] if holdout_result else 0,
        "mean_return_train": train_result["mean_net_payoff"], "p_value_train": train_result["p_value"],
        "mean_return_holdout": holdout_result["mean_net_payoff"] if holdout_result else None,
        "holdout_verdict": ("PASSED" if holdout_result and holdout_result["p_value"] < 0.05
                             and holdout_result["mean_net_payoff"] > 0 else "FAILED"),
        "same_sign": bool(holdout_result and train_result["mean_net_payoff"] > 0
                          and holdout_result["mean_net_payoff"] > 0),
        "verdict": "SCORED",
    })
    _append_trial(trial)
    return trial


# A seventh strategy family, an explicit trading concept stated directly:
# "a strong trend is the one which respects all future bullish/bearish
# areas — the moment a gap is closed by a wick, even if trend goes up,
# that's bearish signal. if candle breaks, that's true bearish."
#
# This maps onto fields detectors.py already tracks, not new detection
# logic: a zone's own "filled"/"mitigated" is WICK-based (consequent
# encroachment uses each bar's raw high/low, see detect_fvgs' own
# comment) — a zone can go fully filled purely by wicks, with no candle
# ever closing beyond it. detect_ifvgs/detect_breaker_blocks separately
# require a CLOSE beyond the zone's own original bound to confirm an
# inversion. So: filled but NOT inverted = "closed by a wick" (the
# user's own bearish-even-in-an-uptrend warning); filled AND inverted =
# "candle breaks" (already this project's own IFVG/Breaker Block
# concept, tested on its own earlier — not duplicated here). Only
# zones in the PREVAILING TREND'S OWN direction count (a bullish zone
# during an uptrend, a bearish zone during a downtrend) — a violation
# of a COUNTER-trend zone isn't the "strong trend weakening" signal
# being described here at all.
def _trend_at(fast_arr, slow_arr, pos):
    if pos >= len(fast_arr) or np.isnan(fast_arr[pos]) or np.isnan(slow_arr[pos]):
        return None
    if fast_arr[pos] > slow_arr[pos]:
        return "bullish"
    if fast_arr[pos] < slow_arr[pos]:
        return "bearish"
    return None


@st.cache_data(ttl=_CACHE_TTL)
def build_trend_wick_violation_events(df, detector_name, forward_bars, ma_fast=20, ma_slow=50,
                                       max_scan_bars=2000):
    """events_df (entry_time, raw_return, direction) — same shape every
    other builder here returns. direction is the SIGNAL's own direction
    (opposite the prevailing trend — a wick violation of a bullish zone
    during an uptrend is a BEARISH signal), not the trend's own.

    Entry is the bar the zone actually goes fully filled (detectors.py's
    own "end"/g["end"] — the bar the wick-based consequent-encroachment
    scan already resolved as fully eaten) — the LAST bar whose data is
    used to decide this qualifies (was it filled, did that same bar's own
    close also break through). raw_return is measured forward from
    THERE, never from the zone's own earlier start — same no-overlap
    discipline as every builder above. One trade at a time (skip a
    violation landing inside a still-open previous signal's own holding
    window)."""
    cols = ["entry_time", "raw_return", "direction"]
    close = (df["Close"] if "Close" in df else df["close"])
    close_arr = close.to_numpy()
    n = len(df)
    if n < ma_slow + forward_bars + 2:
        return pd.DataFrame(columns=cols)

    from indicators import ema as _ema
    fast_arr = _ema(close, ma_fast).to_numpy()
    slow_arr = _ema(close, ma_slow).to_numpy()
    pos_by_time = {t: i for i, t in enumerate(df.index)}

    if detector_name == "FVG":
        zones = detect_fvgs(df, max_scan_bars=max_scan_bars, record_history=True)
        inversions = detect_ifvgs(df, max_scan_bars=max_scan_bars)
        filled_key, zone_kind_key = "filled", "type"
    else:
        zones = detect_order_blocks(df, max_scan_bars=max_scan_bars, record_history=True)
        inversions = detect_breaker_blocks(df, max_scan_bars=max_scan_bars)
        filled_key, zone_kind_key = "mitigated", "type"
    inverted_origins = {inv["origin_start"] for inv in inversions}

    violations = []
    for zone in zones:
        if not zone.get(filled_key):
            continue
        if zone["start"] in inverted_origins:
            continue  # that same zone ALSO closed through -- its own IFVG/
            # Breaker Block trial already covers it; this signal is
            # specifically the wick-only case, not a duplicate of that one.
        end_pos = pos_by_time.get(zone["end"])
        if end_pos is None:
            continue
        trend = _trend_at(fast_arr, slow_arr, end_pos)
        if trend is None or trend != zone[zone_kind_key]:
            continue  # only a same-direction-as-trend zone counts
        signal_direction = "bearish" if trend == "bullish" else "bullish"
        violations.append((end_pos, signal_direction))

    if not violations:
        return pd.DataFrame(columns=cols)

    violations = sorted(violations, key=lambda v: v[0])
    rows = []
    last_exit_pos = -1
    for pos, direction in violations:
        if pos <= last_exit_pos:
            continue
        exit_pos = pos + forward_bars
        if exit_pos >= n:
            continue
        fwd_return = float((close_arr[exit_pos] - close_arr[pos]) / close_arr[pos])
        rows.append({"entry_time": df.index[pos], "raw_return": fwd_return, "direction": direction})
        last_exit_pos = exit_pos
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# An eighth strategy family, an explicit classic-ICT trade management
# style, stated directly: "enter gap, stop below it, no target, just
# stop moved below every new bullish/bearish area and never below the
# slower moving average." Genuinely different in KIND from every
# builder above — not a fixed forward_bars hold, a real path-dependent
# trailing stop: enter on a same-direction-as-trend zone touch, initial
# stop at that zone's own ORIGINAL far edge (before consequent
# encroachment eats into it — raw_top/raw_bottom for FVG,
# _formation_top/_formation_bottom for Order Blocks, detectors.py's own
# two different names for the identical concept), then every bar:
# ratchet the stop toward the newest same-direction zone's own far edge
# once that's tighter than the current stop, and never let it sit
# looser than the slow EMA (so the EFFECTIVE stop is
# max(zone-trail, slow EMA) for a long, min(...) for a short) — a real
# trend-follower's "cut losses short, let winners run," not a fixed R:R.
#
# A trade still open when df runs out is NOT counted — there's no
# lookahead-safe way to know its real outcome, and fabricating a close
# at some arbitrary bar cap would systematically UNDERSTATE exactly the
# long, unbounded winners this style of stop exists to catch. That's
# also why there's no max-hold parameter here at all, unlike every
# fixed-horizon builder above.
@st.cache_data(ttl=_CACHE_TTL)
def build_gap_trend_ride_events(df, detector_name, ma_fast=20, ma_slow=50, max_scan_bars=2000):
    """events_df (entry_time, raw_return, direction) — same shape every
    other builder here returns, raw_return now spanning a variable,
    trade-specific holding period instead of a fixed bar count."""
    cols = ["entry_time", "raw_return", "direction"]
    close = (df["Close"] if "Close" in df else df["close"])
    close_arr = close.to_numpy()
    n = len(df)
    if n < ma_slow + 10:
        return pd.DataFrame(columns=cols)

    from indicators import ema as _ema
    fast_arr = _ema(close, ma_fast).to_numpy()
    slow_arr = _ema(close, ma_slow).to_numpy()
    pos_by_time = {t: i for i, t in enumerate(df.index)}

    if detector_name == "FVG":
        zones = detect_fvgs(df, max_scan_bars=max_scan_bars, record_history=True)
        top_key, bottom_key = "raw_top", "raw_bottom"
    else:
        zones = detect_order_blocks(df, max_scan_bars=max_scan_bars, record_history=True)
        top_key, bottom_key = "_formation_top", "_formation_bottom"

    # Precompute, per bar, the tightest newly-CONFIRMED same-direction
    # zone's own far edge — confirmed at zone["start"]'s own position
    # (this project's existing "the instant its own displacement candle
    # closes" convention for both FVG and Order Blocks). Several zones
    # confirming on the same bar: keep only the tightest (highest floor /
    # lowest ceiling), since that's the only one that could ever actually
    # move the ratchet.
    new_bull_floor, new_bear_ceiling = {}, {}
    touches = []  # (touch_pos, direction, initial_far_edge)
    for zone in zones:
        start_pos = pos_by_time.get(zone["start"])
        if start_pos is not None:
            if zone["type"] == "bullish":
                edge = zone.get(bottom_key, zone["bottom"])
                new_bull_floor[start_pos] = max(new_bull_floor.get(start_pos, -float("inf")), edge)
            else:
                edge = zone.get(top_key, zone["top"])
                new_bear_ceiling[start_pos] = min(new_bear_ceiling.get(start_pos, float("inf")), edge)
        if zone.get("first_touch") is not None:
            touch_pos = pos_by_time.get(zone["first_touch"])
            if touch_pos is not None:
                far_edge = zone.get(bottom_key, zone["bottom"]) if zone["type"] == "bullish" \
                    else zone.get(top_key, zone["top"])
                touches.append((touch_pos, zone["type"], far_edge))

    touches.sort(key=lambda t: t[0])
    rows = []
    last_exit_pos = -1
    for touch_pos, direction, far_edge in touches:
        if touch_pos <= last_exit_pos or touch_pos >= n:
            continue
        if np.isnan(fast_arr[touch_pos]) or np.isnan(slow_arr[touch_pos]):
            continue
        trend = ("bullish" if fast_arr[touch_pos] > slow_arr[touch_pos]
                  else "bearish" if fast_arr[touch_pos] < slow_arr[touch_pos] else None)
        if trend is None or trend != direction:
            continue  # only enter WITH the prevailing trend -- a counter-trend
            # gap is a different trade entirely, out of scope here

        entry_pos = touch_pos
        entry_price = float(close_arr[entry_pos])
        stop = max(far_edge, slow_arr[entry_pos]) if direction == "bullish" \
            else min(far_edge, slow_arr[entry_pos])

        exit_pos = None
        for k in range(entry_pos + 1, n):
            if direction == "bullish":
                if not np.isnan(slow_arr[k]):
                    stop = max(stop, slow_arr[k])
                if k in new_bull_floor and new_bull_floor[k] > stop:
                    stop = new_bull_floor[k]
                if close_arr[k] < stop:
                    exit_pos = k
                    break
            else:
                if not np.isnan(slow_arr[k]):
                    stop = min(stop, slow_arr[k])
                if k in new_bear_ceiling and new_bear_ceiling[k] < stop:
                    stop = new_bear_ceiling[k]
                if close_arr[k] > stop:
                    exit_pos = k
                    break
        if exit_pos is None:
            continue  # never actually stopped out within available data

        exit_price = float(close_arr[exit_pos])
        fwd_return = (exit_price - entry_price) / entry_price
        rows.append({"entry_time": df.index[entry_pos], "raw_return": fwd_return, "direction": direction})
        last_exit_pos = exit_pos
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# A ninth strategy family, an explicit classic ICT setup: "test only 1m
# candles on all tickers if during kill zones, you can predict swings
# high and low and travel to the opposite liquidity once trend reversed
# after a sweep." Reuses detect_liquidity_sweeps (already built — every
# swing high/low that later got wicked through, tagged with the
# REVERSAL direction it implies) and detect_swings' own confirmed_pos
# (the exact field this project's own docstring warns must be used, not
# pos, for "was this level already knowable at some earlier bar" — using
# pos there is a documented lookahead bug elsewhere in this codebase,
# avoided here on purpose).
def _in_kill_zone(times_arr):
    from datetime import time as _dtime
    windows = [(_dtime(2, 0), _dtime(5, 0)), (_dtime(9, 30), _dtime(11, 30)), (_dtime(13, 30), _dtime(16, 0))]
    return np.array([any(s <= t < e for s, e in windows) for t in times_arr])


@st.cache_data(ttl=_CACHE_TTL)
def build_liquidity_sweep_reversal_events(df, kill_zones_only=True):
    """events_df (entry_time, raw_return, direction) — same shape every
    other builder here returns. Rule, entirely causal:

      1. SWEEP: a swing high/low gets wicked through (detect_liquidity_
         sweeps) — `type` is already the reversal direction this implies
         (sweeping a high -> bearish, sweeping a low -> bullish), the
         same type convention every detector in this project shares.
      2. CONFIRMATION: scanning forward from the sweep (up to
         confirm_window bars — a real rejection candle rarely lands on
         the exact same bar as the raw wick, it usually takes the
         market a bar or two to actually close back inside), the FIRST
         bar whose CLOSE gets back on the correct side of the swept
         level — that bar, not the raw sweep bar itself, is the real
         entry. A sweep that never gets a close-back within the window
         isn't "trend reversed," it's still a live break in progress —
         excluded, not force-confirmed.
      3. KILL ZONE (if kill_zones_only): the CONFIRMATION bar must fall
         in London Open, NY AM, or NY PM (NY time) — Asian excluded,
         matching this session's own established session-testing
         convention.
      4. TARGET ("the opposite liquidity"): the nearest OPPOSITE-side
         swing point already CONFIRMED as of the entry bar (confirmed_
         pos <= entry bar, never pos — pos alone would let this react to
         a swing the market hadn't actually shown enough reversal to
         establish yet, exactly the lookahead bug detect_swings' own
         docstring warns other callers about).
      5. STOP: the swept level itself — a CLOSE back beyond it means the
         reversal failed.

    Walked forward bar by bar from entry, stop checked before target on
    any bar that would trigger both (the conservative assumption
    backtest_custom_rule already uses elsewhere in this project). A
    trade that hits neither within available data is excluded — no
    lookahead-safe way to know its real outcome, not fabricated.
    One trade at a time."""
    from datetime import time as _dtime
    cols = ["entry_time", "raw_return", "direction"]
    close = (df["Close"] if "Close" in df else df["close"])
    high = (df["High"] if "High" in df else df["high"])
    low = (df["Low"] if "Low" in df else df["low"])
    close_arr = close.to_numpy()
    high_arr = high.to_numpy()
    low_arr = low.to_numpy()
    n = len(df)
    if n < 30:
        return pd.DataFrame(columns=cols)

    confirm_window = 10
    sweeps = detect_liquidity_sweeps(df)
    highs, lows = detect_swings(df)
    pos_by_time = {t: i for i, t in enumerate(df.index)}

    if kill_zones_only:
        idx_ny = df.index.tz_convert("America/New_York")
        in_session = _in_kill_zone(idx_ny.time)
    else:
        in_session = None

    candidates = []
    for sweep in sweeps:
        sweep_pos = pos_by_time.get(sweep["end"])
        if sweep_pos is None or sweep_pos >= n - 1:
            continue
        direction = sweep["type"]
        level = sweep["level"]

        entry_pos = None
        for k in range(sweep_pos, min(n, sweep_pos + 1 + confirm_window)):
            if direction == "bearish" and close_arr[k] < level:
                entry_pos = k
                break
            if direction == "bullish" and close_arr[k] > level:
                entry_pos = k
                break
        if entry_pos is None:
            continue  # never actually closed back inside within the window
        if in_session is not None and not in_session[entry_pos]:
            continue
        entry_price = float(close_arr[entry_pos])

        # "Opposite" liquidity: a bearish reversal (swept a HIGH, buy-side
        # liquidity) travels toward sell-side liquidity resting BELOW a
        # swing LOW; a bullish reversal (swept a LOW) travels toward
        # buy-side liquidity resting ABOVE a swing HIGH. Confirmed
        # directly as a real bug before this fix: swapping these produced
        # a target on the WRONG side of entry, which the exit loop below
        # (high >= target for bullish, low <= target for bearish) would
        # then satisfy almost immediately at a nonsensical price —
        # exactly why every trade came back a loss regardless of
        # direction (a perfect, suspicious 100% inverse pattern, not
        # genuine noise).
        if direction == "bearish":
            opp = [l for l in lows if l["confirmed_pos"] <= entry_pos and l["price"] < entry_price]
            if not opp:
                continue
            target = max(opp, key=lambda l: l["price"])["price"]
        else:
            opp = [h for h in highs if h["confirmed_pos"] <= entry_pos and h["price"] > entry_price]
            if not opp:
                continue
            target = min(opp, key=lambda h: h["price"])["price"]

        exit_pos, exit_price = None, None
        for k in range(entry_pos + 1, n):
            if direction == "bullish":
                if close_arr[k] < level:
                    exit_pos, exit_price = k, float(close_arr[k])
                    break
                if high_arr[k] >= target:
                    exit_pos, exit_price = k, float(target)
                    break
            else:
                if close_arr[k] > level:
                    exit_pos, exit_price = k, float(close_arr[k])
                    break
                if low_arr[k] <= target:
                    exit_pos, exit_price = k, float(target)
                    break
        if exit_pos is None:
            continue

        fwd_return = (exit_price - entry_price) / entry_price
        candidates.append({"entry_time": df.index[entry_pos], "raw_return": fwd_return,
                            "direction": direction, "pos": entry_pos, "exit_pos": exit_pos})

    if not candidates:
        return pd.DataFrame(columns=cols)
    candidates.sort(key=lambda c: c["pos"])
    rows = []
    last_exit_pos = -1
    for c in candidates:
        if c["pos"] <= last_exit_pos:
            continue
        rows.append({"entry_time": c["entry_time"], "raw_return": c["raw_return"], "direction": c["direction"]})
        last_exit_pos = c["exit_pos"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# A tenth strategy family, and a genuinely different KIND of bet from
# every one of the nine above: cross-ASSET, not single-instrument. All
# nine tested one instrument against its own history. This tests
# whether an unusually large, ATR-relative move in a LEADER (BTC-USD —
# crypto's most liquid, most-watched instrument) predicts a same-
# direction follow-through in a FOLLOWER (an altcoin) over the next few
# bars — "gradual information diffusion," a real, documented effect in
# equity/crypto lead-lag literature (large-cap moves reaching smaller,
# less-watched names with a lag), not an ICT concept at all.
@st.cache_data(ttl=_CACHE_TTL)
def build_leader_follower_events(leader_df, follower_df, lead_window, atr_z_threshold, forward_bars,
                                  atr_lookback=100):
    """events_df (entry_time, raw_return, direction) — same shape every
    other builder here returns, but raw_return is the FOLLOWER's own
    forward return, conditioned on the LEADER's own recent move.

    LEADER EVENT: over the last lead_window bars, the leader moved by at
    least atr_z_threshold times its own trailing per-bar volatility
    (scaled by sqrt(lead_window) — same volatility-scaling convention as
    every ATR/vol-based builder above), causally (only the leader's own
    past returns feed its own trailing-vol estimate). direction is the
    leader's own move direction.

    FOLLOWER OUTCOME: raw_return is the FOLLOWER's forward return over
    forward_bars from that SAME bar — never the leader's own return,
    and never using anything past that bar on either series.

    Both dataframes must already share the same timestamps (same
    provider/interval fetch) — aligned here via index intersection, not
    resampled or interpolated, so a mismatched pair just yields fewer
    (or zero) usable bars rather than fabricated alignment.

    One trade at a time (skip a leader event whose own forward_bars
    window would overlap a still-open previous one) — same discipline
    as every builder above, applied here per (leader, follower) pair."""
    cols = ["entry_time", "raw_return", "direction"]
    common_idx = leader_df.index.intersection(follower_df.index)
    if len(common_idx) < atr_lookback + lead_window + forward_bars + 10:
        return pd.DataFrame(columns=cols)
    leader = leader_df.loc[common_idx]
    follower = follower_df.loc[common_idx]

    l_close = (leader["Close"] if "Close" in leader else leader["close"]).to_numpy()
    f_close = (follower["Close"] if "Close" in follower else follower["close"]).to_numpy()
    n = len(common_idx)

    l_returns = np.diff(l_close) / l_close[:-1]
    l_vol = pd.Series(l_returns).rolling(atr_lookback, min_periods=atr_lookback).std().to_numpy()
    l_vol_padded = np.concatenate([[np.nan], l_vol])

    rows = []
    last_exit_pos = -1
    for t in range(lead_window + atr_lookback, n - forward_bars):
        if t <= last_exit_pos:
            continue
        move = (l_close[t] - l_close[t - lead_window]) / l_close[t - lead_window]
        sigma = l_vol_padded[t] * np.sqrt(lead_window)
        if not np.isfinite(sigma) or sigma <= 0:
            continue
        z = move / sigma
        if abs(z) < atr_z_threshold:
            continue
        direction = "bullish" if z > 0 else "bearish"
        entry_price = float(f_close[t])
        exit_price = float(f_close[t + forward_bars])
        fwd_return = (exit_price - entry_price) / entry_price
        rows.append({"entry_time": common_idx[t], "raw_return": fwd_return, "direction": direction})
        last_exit_pos = t + forward_bars
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# An eleventh strategy family: volatility SQUEEZE then BREAKOUT — the
# classic "Bollinger Band squeeze" / volatility-contraction pattern
# (TTM Squeeze and similar retail indicators), genuinely different from
# every one of the ten above: it's not zone-reaction, MA-retest, session
# timing, or cross-asset — it's a pure volatility-CYCLE bet (quiet
# periods precede loud ones, trade the transition), proposed early this
# session and never actually built until now.
@st.cache_data(ttl=_CACHE_TTL)
def build_vol_squeeze_breakout_events(df, squeeze_pctile_max, forward_bars, max_bars_after_squeeze=10,
                                       bb_period=20, num_std=2.0, vol_lookback=100):
    """events_df (entry_time, raw_return, direction) — same shape every
    other builder here returns.

    SQUEEZE: a bar whose own volatility_percentile (this module's own
    causal, trailing-ATR-rank helper, already used by build_experiment_
    events above) is at or below squeeze_pctile_max — among the quietest
    it's recently been.

    BREAKOUT: scanning forward from the squeeze bar (up to max_bars_
    after_squeeze), the first bar whose CLOSE clears the Bollinger Band
    computed AT THAT BAR (upper band -> bullish breakout, lower -> bearish)
    — a real, convicted close beyond the band, not a wick, same "close,
    not wick" confirmation discipline as every ICT-specific builder above
    even though this one isn't ICT at all.

    Entry is the breakout bar; raw_return measured forward from there,
    never from the squeeze bar itself. One trade at a time."""
    from indicators import bollinger_bands
    cols = ["entry_time", "raw_return", "direction"]
    close = (df["Close"] if "Close" in df else df["close"])
    close_arr = close.to_numpy()
    n = len(df)
    if n < max(bb_period, vol_lookback) + max_bars_after_squeeze + forward_bars + 10:
        return pd.DataFrame(columns=cols)

    vol_pctile = volatility_percentile(df, lookback=vol_lookback).to_numpy()
    upper, _, lower = bollinger_bands(close, bb_period, num_std)
    upper_arr, lower_arr = upper.to_numpy(), lower.to_numpy()

    candidates = []
    i = 0
    while i < n:
        if np.isnan(vol_pctile[i]) or vol_pctile[i] > squeeze_pctile_max:
            i += 1
            continue
        breakout_pos, direction = None, None
        for j in range(i + 1, min(n, i + 1 + max_bars_after_squeeze)):
            if np.isnan(upper_arr[j]) or np.isnan(lower_arr[j]):
                continue
            if close_arr[j] > upper_arr[j]:
                breakout_pos, direction = j, "bullish"
                break
            if close_arr[j] < lower_arr[j]:
                breakout_pos, direction = j, "bearish"
                break
        if breakout_pos is not None:
            candidates.append((breakout_pos, direction))
            i = breakout_pos + 1  # resume scanning for the NEXT squeeze after this breakout
        else:
            i += 1

    if not candidates:
        return pd.DataFrame(columns=cols)
    candidates.sort(key=lambda c: c[0])
    rows = []
    last_exit_pos = -1
    for pos, direction in candidates:
        if pos <= last_exit_pos:
            continue
        exit_pos = pos + forward_bars
        if exit_pos >= n:
            continue
        fwd_return = float((close_arr[exit_pos] - close_arr[pos]) / close_arr[pos])
        rows.append({"entry_time": df.index[pos], "raw_return": fwd_return, "direction": direction})
        last_exit_pos = exit_pos
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# A twelfth strategy family, a direct refinement request: "cut the
# losing trades by buying in retracements after crossings and limit
# somehow as the trend exhausts." Same ENTRY as build_ma_cross_retest_
# events (EMA cross, retest the fast MA, RSI-momentum-turn confirmation
# at the touch) — but a genuinely different EXIT, aimed specifically at
# cutting losers short rather than holding a fixed bar count or a
# consecutive-adverse-close streak: hold only as long as price keeps
# making new same-direction extremes that RSI ALSO confirms (a new high
# with a new-high RSI too); exit the FIRST time price makes a new
# extreme WITHOUT RSI confirming it — classic bearish/bullish
# divergence, the standard definition of "the trend is running out of
# steam." The slow EMA is a hard backstop underneath that (same "never
# below the slow MA" floor build_gap_trend_ride_events already uses) —
# whichever trips first ends the trade.
@st.cache_data(ttl=_CACHE_TTL)
def build_ma_retest_exhaustion_events(df, ma_fast=20, ma_slow=50, rsi_dip_threshold=60,
                                       max_bars_after_cross=20, rsi_period=14, max_hold_bars=150):
    """events_df (entry_time, raw_return, direction) — same shape every
    other builder here returns. Entry mechanics are identical to
    build_ma_cross_retest_events (see its own docstring for the cross/
    retest/RSI-turn rule); only what happens AFTER entry differs."""
    from indicators import ema as _ema, rsi as _rsi
    cols = ["entry_time", "raw_return", "direction"]
    close = (df["Close"] if "Close" in df else df["close"])
    high = (df["High"] if "High" in df else df["high"])
    low = (df["Low"] if "Low" in df else df["low"])
    close_arr = close.to_numpy()
    high_arr = high.to_numpy()
    low_arr = low.to_numpy()
    n = len(df)
    if n < ma_slow + 10:
        return pd.DataFrame(columns=cols)

    fast_ma = _ema(close, ma_fast).to_numpy()
    slow_ma = _ema(close, ma_slow).to_numpy()
    rsi_arr = _rsi(close, rsi_period).to_numpy()
    diff = fast_ma - slow_ma

    rows = []
    last_exit_pos = -1
    i = 0
    while i < n - 1:
        if np.isnan(diff[i]) or np.isnan(diff[i + 1]):
            i += 1
            continue
        crossed_bullish = diff[i] <= 0 and diff[i + 1] > 0
        crossed_bearish = diff[i] >= 0 and diff[i + 1] < 0
        if not (crossed_bullish or crossed_bearish):
            i += 1
            continue
        direction = "bullish" if crossed_bullish else "bearish"
        cross_pos = i + 1

        touch_pos = None
        for j in range(cross_pos + 1, min(n, cross_pos + 1 + max_bars_after_cross)):
            if np.isnan(fast_ma[j]) or np.isnan(slow_ma[j]):
                continue
            still_trending = (fast_ma[j] > slow_ma[j]) if direction == "bullish" else (fast_ma[j] < slow_ma[j])
            if not still_trending:
                break
            if low_arr[j] <= fast_ma[j] <= high_arr[j]:
                touch_pos = j
                break
        if touch_pos is None:
            i = cross_pos
            continue
        if touch_pos <= last_exit_pos:
            i = cross_pos
            continue
        if touch_pos < 1 or np.isnan(rsi_arr[touch_pos]) or np.isnan(rsi_arr[touch_pos - 1]):
            i = cross_pos
            continue
        if direction == "bullish":
            momentum_ok = (rsi_arr[touch_pos - 1] <= rsi_dip_threshold) and (rsi_arr[touch_pos] > rsi_arr[touch_pos - 1])
        else:
            momentum_ok = (rsi_arr[touch_pos - 1] >= (100 - rsi_dip_threshold)) and (rsi_arr[touch_pos] < rsi_arr[touch_pos - 1])
        if not momentum_ok:
            i = cross_pos
            continue

        entry_pos = touch_pos
        entry_price = float(close_arr[entry_pos])
        running_extreme_price = entry_price
        running_extreme_rsi = float(rsi_arr[entry_pos])
        exit_pos = None
        for k in range(entry_pos + 1, min(n, entry_pos + 1 + max_hold_bars)):
            if np.isnan(rsi_arr[k]):
                continue
            if direction == "bullish":
                if not np.isnan(slow_ma[k]) and close_arr[k] < slow_ma[k]:
                    exit_pos = k
                    break
                if high_arr[k] > running_extreme_price:
                    if rsi_arr[k] < running_extreme_rsi:
                        exit_pos = k  # new high, RSI didn't confirm -- exhausted
                        break
                    running_extreme_price = float(high_arr[k])
                    running_extreme_rsi = float(rsi_arr[k])
            else:
                if not np.isnan(slow_ma[k]) and close_arr[k] > slow_ma[k]:
                    exit_pos = k
                    break
                if low_arr[k] < running_extreme_price:
                    if rsi_arr[k] > running_extreme_rsi:
                        exit_pos = k
                        break
                    running_extreme_price = float(low_arr[k])
                    running_extreme_rsi = float(rsi_arr[k])
        if exit_pos is None:
            exit_pos = min(n - 1, entry_pos + max_hold_bars)
        if exit_pos <= entry_pos or exit_pos >= n:
            i = cross_pos
            continue

        fwd_return = float((close_arr[exit_pos] - entry_price) / entry_price)
        rows.append({"entry_time": df.index[entry_pos], "raw_return": fwd_return, "direction": direction})
        last_exit_pos = exit_pos
        i = cross_pos
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# A 13th strategy family, from a direct order-flow-flavored request:
# "candles with exponentially increasingly positive volume delta over 3
# candles" -- accelerating buy (or sell) pressure. yfinance OHLCV has no
# real buyer/seller trade-side split (that needs tick data with a
# classified side, which this project doesn't have yet), so "volume
# delta" here is the standard Chaikin-style approximation: split each
# bar's own total volume by where its close landed in its own [low,
# high] range -- close near the high looks buyer-driven, close near the
# low looks seller-driven. A real proxy, not real order flow; reported
# as such.
@st.cache_data(ttl=_CACHE_TTL)
def build_volume_delta_accel_events(df, forward_bars):
    """events_df (entry_time, raw_return, direction). volume_delta[t] =
    volume[t] * (2*close[t] - high[t] - low[t]) / (high[t] - low[t]) --
    zero when high==low (a truly flat bar has no directional volume to
    assign).

    Signal (causal, uses only bars up to and including t): three
    consecutive bars t-2, t-1, t each with delta > 0 AND strictly
    increasing (delta[t-2] < delta[t-1] < delta[t]) -- accelerating
    buying pressure, direction "bullish". The exact mirror -- three
    consecutive bars each < 0 and strictly decreasing (more negative
    each time) -- is direction "bearish", tested the same way for the
    usual direction-balance check every builder here gets.

    Entry/exit are lag-corrected FROM THE START this time (the mistake
    caught and fixed after the fact in every earlier builder today): the
    pattern is confirmed using only bar t's own just-revealed data, but
    the real fill is bar t+1's OPEN, not bar t's own close -- a live
    system only learns the pattern completed after bar t closes, so it
    can't also transact at that same already-gone price. Exit is
    forward_bars bars after that real entry, at that bar's close. One
    trade at a time (skip a signal landing inside a still-open previous
    trade's own holding window)."""
    cols = ["entry_time", "raw_return", "direction"]
    o = (df["Open"] if "Open" in df else df["open"]).to_numpy()
    h = (df["High"] if "High" in df else df["high"]).to_numpy()
    l = (df["Low"] if "Low" in df else df["low"]).to_numpy()
    c = (df["Close"] if "Close" in df else df["close"]).to_numpy()
    vol_col = "Volume" if "Volume" in df else "volume"
    if vol_col not in df:
        return pd.DataFrame(columns=cols)
    v = df[vol_col].to_numpy().astype(float)
    n = len(df)
    if n < forward_bars + 5:
        return pd.DataFrame(columns=cols)

    rng_ = h - l
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(rng_ > 0, v * (2 * c - h - l) / rng_, 0.0)

    rows = []
    last_exit_pos = -1
    for t in range(2, n - 1 - forward_bars):
        d0, d1, d2 = delta[t - 2], delta[t - 1], delta[t]
        bullish = d0 > 0 and d1 > 0 and d2 > 0 and d0 < d1 < d2
        bearish = d0 < 0 and d1 < 0 and d2 < 0 and d0 > d1 > d2
        if not (bullish or bearish):
            continue
        entry_pos = t + 1
        if entry_pos <= last_exit_pos:
            continue
        exit_pos = entry_pos + forward_bars
        if exit_pos >= n:
            continue
        entry_price = o[entry_pos]
        exit_price = c[exit_pos]
        fwd_return = float((exit_price - entry_price) / entry_price)
        direction = "bullish" if bullish else "bearish"
        rows.append({"entry_time": df.index[entry_pos], "raw_return": fwd_return, "direction": direction})
        last_exit_pos = exit_pos
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# A 14th strategy: the "human instinct" request made literal -- not one
# indicator, but several independent price-action tells voting together
# at the exact moment price makes a fresh extreme, the way a discretionary
# trader blends "this looks exhausted" cues by feel. Four cheap, already-
# proven-relevant reads (each one alone was tried today in some other
# builder and didn't survive): RSI failing to confirm the new extreme
# (divergence -- build_bb_divergence_events/build_ma_retest_exhaustion_
# events' own logic), a sweep-and-reject of the prior extreme (build_
# liquidity_sweep_reversal_events' own mechanic), volume pressure
# deserting the move (build_volume_delta_accel_events' own CLV-delta
# proxy), and a real rejection wick. Deliberately a simple VOTE COUNT,
# not a weighted score -- fewer knobs to curve-fit than a regression
# would need, closer to "how many of my usual tells agree right now."
@st.cache_data(ttl=_CACHE_TTL)
def build_price_action_confluence_events(df, forward_bars, extreme_lookback=20, min_votes=3,
                                          rejection_mult=1.5, rsi_period=14):
    """events_df (entry_time, raw_return, direction). At any bar t where
    price makes a fresh extreme_lookback-bar high or low, count up to 4
    independent bearish (at a new high) or bullish (at a new low) votes:

      1. RSI divergence: RSI[t] did NOT also make a new extreme_lookback
         high (bearish case) / low (bullish case) alongside price --
         momentum didn't confirm.
      2. Sweep-and-reject: close[t] already back on the OTHER side of the
         prior extreme_lookback-bar high/low (the level just swept),
         i.e. the breakout failed intrabar.
      3. Volume pressure flip: the CLV volume-delta proxy at t is on the
         OPPOSITE side of zero from the move (negative/selling delta at
         a fresh high, positive/buying delta at a fresh low).
      4. Rejection wick: the wick beyond the body on the extreme's own
         side is at least rejection_mult times the body itself.

    A signal fires only when at least min_votes of these 4 agree, at the
    bar the extreme was made (the LAST bar whose data decides this --
    no lookahead). Entry is lag-corrected from the start (next bar's
    open, not this bar's own close -- the same fix every earlier builder
    needed applied retroactively). Exit is forward_bars bars later at
    that bar's close. One trade at a time."""
    cols = ["entry_time", "raw_return", "direction"]
    o = (df["Open"] if "Open" in df else df["open"]).to_numpy()
    h = (df["High"] if "High" in df else df["high"]).to_numpy()
    l = (df["Low"] if "Low" in df else df["low"]).to_numpy()
    c = (df["Close"] if "Close" in df else df["close"]).to_numpy()
    vol_col = "Volume" if "Volume" in df else "volume"
    n = len(df)
    if n < extreme_lookback + forward_bars + 5:
        return pd.DataFrame(columns=cols)

    from indicators import rsi as _rsi
    close_s = df["Close"] if "Close" in df else df["close"]
    rsi_arr = _rsi(close_s, rsi_period).to_numpy()

    if vol_col in df:
        v = df[vol_col].to_numpy().astype(float)
        rng_ = h - l
        with np.errstate(divide="ignore", invalid="ignore"):
            vol_delta = np.where(rng_ > 0, v * (2 * c - h - l) / rng_, 0.0)
    else:
        vol_delta = np.zeros(n)

    body = np.abs(c - o)
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l

    rows = []
    last_exit_pos = -1
    for t in range(extreme_lookback, n - 1 - forward_bars):
        window_high = h[t - extreme_lookback:t]
        window_low = l[t - extreme_lookback:t]
        window_rsi = rsi_arr[t - extreme_lookback:t]
        if np.isnan(rsi_arr[t]):
            continue

        is_new_high = h[t] > window_high.max()
        is_new_low = l[t] < window_low.min()
        if is_new_high and not is_new_low:
            votes = 0
            if not np.all(np.isnan(window_rsi)) and rsi_arr[t] < np.nanmax(window_rsi):
                votes += 1
            if c[t] < window_high.max():
                votes += 1
            if vol_delta[t] < 0:
                votes += 1
            if body[t] > 0 and upper_wick[t] >= rejection_mult * body[t]:
                votes += 1
            direction = "bearish"
        elif is_new_low and not is_new_high:
            votes = 0
            if not np.all(np.isnan(window_rsi)) and rsi_arr[t] > np.nanmin(window_rsi):
                votes += 1
            if c[t] > window_low.min():
                votes += 1
            if vol_delta[t] > 0:
                votes += 1
            if body[t] > 0 and lower_wick[t] >= rejection_mult * body[t]:
                votes += 1
            direction = "bullish"
        else:
            continue

        if votes < min_votes:
            continue
        entry_pos = t + 1
        if entry_pos <= last_exit_pos:
            continue
        exit_pos = entry_pos + forward_bars
        if exit_pos >= n:
            continue
        entry_price = o[entry_pos]
        exit_price = c[exit_pos]
        fwd_return = float((exit_price - entry_price) / entry_price)
        rows.append({"entry_time": df.index[entry_pos], "raw_return": fwd_return, "direction": direction})
        last_exit_pos = exit_pos
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# A 15th strategy, addressing two things directly: (1) "self-adjusting,
# not static" -- today's 14 builders all fire the same way regardless of
# where price sits in the bigger picture; this one gates its entries on
# HIGHER-TIMEFRAME context, so it's naturally selective rather than
# uniform. (2) "HTF peaks should give off a good bearish trade" -- the
# exact top-down ICT idea: an LTF reversal trigger (build_price_action_
# confluence_events' own 4-vote logic, unmodified) only counts when it
# ALSO lands near a genuine higher-timeframe extreme, not just a noisy
# local one. Two separate dataframes, same pattern build_leader_follower_
# events already uses for this file's only other multi-timeframe input.
@st.cache_data(ttl=_CACHE_TTL)
def build_htf_peak_gated_reversal_events(ltf_df, htf_df, forward_bars, extreme_lookback=20, min_votes=3,
                                          rejection_mult=1.5, htf_lookback=20, htf_proximity_pct=0.0015,
                                          rsi_period=14):
    """events_df (entry_time, raw_return, direction). Reuses build_price_
    action_confluence_events' own 4-vote reversal logic on ltf_df bar for
    bar (RSI divergence / sweep-and-reject / volume-pressure flip /
    rejection wick), but ADDS a hard gate: the vote only counts if the
    LTF extreme is also within htf_proximity_pct of the rolling
    htf_lookback-bar high (bearish gate) or low (bullish gate) computed
    from htf_df -- a genuinely higher-timeframe peak/trough, not just a
    small extreme_lookback-bar wiggle on the LTF series itself.

    HTF alignment is causal by construction: htf_df's own rolling
    high/low is shifted forward one HTF bar before being matched to each
    LTF timestamp (merge_asof, backward), so only an ALREADY-CLOSED HTF
    bar's information gates any given LTF bar -- never the HTF bar still
    forming at that moment.

    Entry lag-corrected from the start (next LTF bar's open). Exit
    forward_bars LTF bars later at that bar's close. One trade at a
    time."""
    cols = ["entry_time", "raw_return", "direction"]
    o = (ltf_df["Open"] if "Open" in ltf_df else ltf_df["open"]).to_numpy()
    h = (ltf_df["High"] if "High" in ltf_df else ltf_df["high"]).to_numpy()
    l = (ltf_df["Low"] if "Low" in ltf_df else ltf_df["low"]).to_numpy()
    c = (ltf_df["Close"] if "Close" in ltf_df else ltf_df["close"]).to_numpy()
    vol_col = "Volume" if "Volume" in ltf_df else "volume"
    n = len(ltf_df)
    if n < extreme_lookback + forward_bars + 5 or htf_df is None or len(htf_df) < htf_lookback + 2:
        return pd.DataFrame(columns=cols)

    from indicators import rsi as _rsi
    close_s = ltf_df["Close"] if "Close" in ltf_df else ltf_df["close"]
    rsi_arr = _rsi(close_s, rsi_period).to_numpy()

    if vol_col in ltf_df:
        v = ltf_df[vol_col].to_numpy().astype(float)
        rng_ = h - l
        with np.errstate(divide="ignore", invalid="ignore"):
            vol_delta = np.where(rng_ > 0, v * (2 * c - h - l) / rng_, 0.0)
    else:
        vol_delta = np.zeros(n)

    body = np.abs(c - o)
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l

    htf_high = (htf_df["High"] if "High" in htf_df else htf_df["high"])
    htf_low = (htf_df["Low"] if "Low" in htf_df else htf_df["low"])
    htf_roll_high = htf_high.rolling(htf_lookback, min_periods=htf_lookback).max().shift(1)
    htf_roll_low = htf_low.rolling(htf_lookback, min_periods=htf_lookback).min().shift(1)
    htf_ref = pd.DataFrame({"htf_high": htf_roll_high, "htf_low": htf_roll_low}, index=htf_df.index)
    htf_ref = htf_ref.dropna()

    ltf_times = pd.DataFrame({"t": ltf_df.index}, index=ltf_df.index)
    merged = pd.merge_asof(ltf_times.sort_index(), htf_ref.sort_index(), left_index=True, right_index=True,
                            direction="backward")
    gate_high = merged["htf_high"].to_numpy()
    gate_low = merged["htf_low"].to_numpy()

    rows = []
    last_exit_pos = -1
    for t in range(extreme_lookback, n - 1 - forward_bars):
        if np.isnan(rsi_arr[t]) or np.isnan(gate_high[t]) or np.isnan(gate_low[t]):
            continue
        window_high = h[t - extreme_lookback:t]
        window_low = l[t - extreme_lookback:t]
        window_rsi = rsi_arr[t - extreme_lookback:t]

        is_new_high = h[t] > window_high.max()
        is_new_low = l[t] < window_low.min()
        if is_new_high and not is_new_low:
            near_htf_peak = h[t] >= gate_high[t] * (1 - htf_proximity_pct)
            if not near_htf_peak:
                continue
            votes = 0
            if not np.all(np.isnan(window_rsi)) and rsi_arr[t] < np.nanmax(window_rsi):
                votes += 1
            if c[t] < window_high.max():
                votes += 1
            if vol_delta[t] < 0:
                votes += 1
            if body[t] > 0 and upper_wick[t] >= rejection_mult * body[t]:
                votes += 1
            direction = "bearish"
        elif is_new_low and not is_new_high:
            near_htf_trough = l[t] <= gate_low[t] * (1 + htf_proximity_pct)
            if not near_htf_trough:
                continue
            votes = 0
            if not np.all(np.isnan(window_rsi)) and rsi_arr[t] > np.nanmin(window_rsi):
                votes += 1
            if c[t] > window_low.min():
                votes += 1
            if vol_delta[t] > 0:
                votes += 1
            if body[t] > 0 and lower_wick[t] >= rejection_mult * body[t]:
                votes += 1
            direction = "bullish"
        else:
            continue

        if votes < min_votes:
            continue
        entry_pos = t + 1
        if entry_pos <= last_exit_pos:
            continue
        exit_pos = entry_pos + forward_bars
        if exit_pos >= n:
            continue
        entry_price = o[entry_pos]
        exit_price = c[exit_pos]
        fwd_return = float((exit_price - entry_price) / entry_price)
        rows.append({"entry_time": ltf_df.index[entry_pos], "raw_return": fwd_return, "direction": direction})
        last_exit_pos = exit_pos
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# A 16th strategy, a direct request: does a MOMENTUM CROSSOVER that
# happens counter to the recent trend (a bullish MACD/MA cross after a
# recent bearish stretch, or the mirror) carry different information
# depending on WHERE it crosses -- MACD's own zero line, or price's
# side of a longer-term MA -- versus just "a cross happened." Two
# separate builders (MACD zero-line position, MA 200-EMA position),
# same trend-precondition logic, each filterable by position so the two
# sides can be compared directly.
@st.cache_data(ttl=_CACHE_TTL)
def build_macd_cross_trend_events(df, forward_bars, trend_lookback=30, trend_majority=0.7,
                                   zero_filter="both", macd_fast=12, macd_slow=26, macd_signal=9,
                                   ema_fast=20, ema_slow=50):
    """events_df (entry_time, raw_return, direction). Precondition: the
    ema_fast/ema_slow relationship over the trend_lookback bars BEFORE
    the cross bar was bearish (fast<slow) at least trend_majority of the
    time, for a BULLISH MACD cross (counter-trend reversal setup) -- or
    the mirror (majority bullish, then a bearish MACD cross) for
    direction="bearish". A cross with no such prior counter-trend
    stretch is skipped entirely -- this only tests the reversal-context
    case, not plain trend-following continuation crosses.

    zero_filter ("below", "above", "both"): at the cross bar itself, is
    the MACD line still on the OLD trend's side of zero ("below" for a
    bullish cross while MACD line < 0 -- hasn't caught up yet) or
    already on the NEW direction's side ("above" -- momentum already
    flipped sign, more confirmed/later signal)? "both" -- no filter,
    every qualifying cross regardless of zero-line side.

    Entry lag-corrected (next bar's open after the cross bar's own
    close confirms it). Exit forward_bars bars later at that bar's
    close. One trade at a time."""
    cols = ["entry_time", "raw_return", "direction"]
    o = (df["Open"] if "Open" in df else df["open"]).to_numpy()
    c = (df["Close"] if "Close" in df else df["close"]).to_numpy()
    n = len(df)
    if n < trend_lookback + ema_slow + forward_bars + 5:
        return pd.DataFrame(columns=cols)

    from indicators import ema as _ema, macd as _macd
    close_s = df["Close"] if "Close" in df else df["close"]
    fast_ma = _ema(close_s, ema_fast).to_numpy()
    slow_ma = _ema(close_s, ema_slow).to_numpy()
    macd_line, signal_line, _ = _macd(close_s, macd_fast, macd_slow, macd_signal)
    macd_arr = macd_line.to_numpy()
    sig_arr = signal_line.to_numpy()

    rows = []
    last_exit_pos = -1
    for t in range(trend_lookback + ema_slow, n - 1 - forward_bars):
        if np.isnan(macd_arr[t]) or np.isnan(sig_arr[t]) or np.isnan(macd_arr[t - 1]) or np.isnan(sig_arr[t - 1]):
            continue
        crossed_bullish = macd_arr[t - 1] <= sig_arr[t - 1] and macd_arr[t] > sig_arr[t]
        crossed_bearish = macd_arr[t - 1] >= sig_arr[t - 1] and macd_arr[t] < sig_arr[t]
        if not (crossed_bullish or crossed_bearish):
            continue

        window_fast = fast_ma[t - trend_lookback:t]
        window_slow = slow_ma[t - trend_lookback:t]
        valid = ~(np.isnan(window_fast) | np.isnan(window_slow))
        if valid.sum() < trend_lookback * 0.5:
            continue
        bearish_frac = (window_fast[valid] < window_slow[valid]).mean()
        bullish_frac = (window_fast[valid] > window_slow[valid]).mean()

        if crossed_bullish and bearish_frac >= trend_majority:
            direction = "bullish"
            zero_side = "below" if macd_arr[t] < 0 else "above"
        elif crossed_bearish and bullish_frac >= trend_majority:
            direction = "bearish"
            zero_side = "above" if macd_arr[t] > 0 else "below"
        else:
            continue

        if zero_filter != "both" and zero_side != zero_filter:
            continue

        entry_pos = t + 1
        if entry_pos <= last_exit_pos:
            continue
        exit_pos = entry_pos + forward_bars
        if exit_pos >= n:
            continue
        entry_price = o[entry_pos]
        exit_price = c[exit_pos]
        fwd_return = float((exit_price - entry_price) / entry_price)
        rows.append({"entry_time": df.index[entry_pos], "raw_return": fwd_return, "direction": direction})
        last_exit_pos = exit_pos
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


@st.cache_data(ttl=_CACHE_TTL)
def build_ma_cross_htf_filtered_events(df, forward_bars, trend_lookback=30, trend_majority=0.7,
                                        fast=20, slow=50, htf_ma=200, position_filter="both"):
    """Same trend-precondition + counter-trend-cross idea as
    build_macd_cross_trend_events, but for a plain fast/slow EMA cross
    instead of MACD, classified by price's side of a longer htf_ma EMA
    at the cross bar instead of a zero line: position_filter "below_htf"
    (price still under the bigger-picture average -- an early,
    unconfirmed-by-the-bigger-trend cross) or "above_htf" (already back
    over it -- later, more trend-aligned) or "both"."""
    cols = ["entry_time", "raw_return", "direction"]
    o = (df["Open"] if "Open" in df else df["open"]).to_numpy()
    c = (df["Close"] if "Close" in df else df["close"]).to_numpy()
    n = len(df)
    if n < trend_lookback + htf_ma + forward_bars + 5:
        return pd.DataFrame(columns=cols)

    from indicators import ema as _ema
    close_s = df["Close"] if "Close" in df else df["close"]
    fast_ma = _ema(close_s, fast).to_numpy()
    slow_ma = _ema(close_s, slow).to_numpy()
    htf_arr = _ema(close_s, htf_ma).to_numpy()

    rows = []
    last_exit_pos = -1
    for t in range(trend_lookback + htf_ma, n - 1 - forward_bars):
        if np.isnan(fast_ma[t]) or np.isnan(slow_ma[t]) or np.isnan(fast_ma[t - 1]) or np.isnan(slow_ma[t - 1]) \
                or np.isnan(htf_arr[t]):
            continue
        crossed_bullish = fast_ma[t - 1] <= slow_ma[t - 1] and fast_ma[t] > slow_ma[t]
        crossed_bearish = fast_ma[t - 1] >= slow_ma[t - 1] and fast_ma[t] < slow_ma[t]
        if not (crossed_bullish or crossed_bearish):
            continue

        window_fast = fast_ma[t - trend_lookback:t]
        window_slow = slow_ma[t - trend_lookback:t]
        valid = ~(np.isnan(window_fast) | np.isnan(window_slow))
        if valid.sum() < trend_lookback * 0.5:
            continue
        bearish_frac = (window_fast[valid] < window_slow[valid]).mean()
        bullish_frac = (window_fast[valid] > window_slow[valid]).mean()

        if crossed_bullish and bearish_frac >= trend_majority:
            direction = "bullish"
            htf_side = "below_htf" if c[t] < htf_arr[t] else "above_htf"
        elif crossed_bearish and bullish_frac >= trend_majority:
            direction = "bearish"
            htf_side = "above_htf" if c[t] > htf_arr[t] else "below_htf"
        else:
            continue

        if position_filter != "both" and htf_side != position_filter:
            continue

        entry_pos = t + 1
        if entry_pos <= last_exit_pos:
            continue
        exit_pos = entry_pos + forward_bars
        if exit_pos >= n:
            continue
        entry_price = o[entry_pos]
        exit_price = c[exit_pos]
        fwd_return = float((exit_price - entry_price) / entry_price)
        rows.append({"entry_time": df.index[entry_pos], "raw_return": fwd_return, "direction": direction})
        last_exit_pos = exit_pos
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# A 17th strategy, genuinely new (not a parameter variant of anything
# above): today's HTF-peak-gated builder deliberately FADES a higher-
# timeframe extreme (bearish near an HTF high, bullish near an HTF low --
# counter-trend by design). This one does the mechanical OPPOSITE: trade
# WITH the higher-timeframe trend, gating a plain volatility-squeeze
# breakout (build_vol_squeeze_breakout_events' own squeeze+BB-breakout
# trigger, unmodified) on whether the breakout direction agrees with
# htf_df's own fast/slow trend at that moment. Same two-dataframe,
# merge_asof('backward') pattern build_htf_peak_gated_reversal_events
# already uses for its own htf_high/htf_low gate -- reused here for a real
# HTF moving-average relationship instead of a rolling high/low, so an
# already-CLOSED htf_df bar's trend state gates the LTF breakout, never
# the still-forming one.
@st.cache_data(ttl=_CACHE_TTL)
def build_squeeze_breakout_htf_trend_events(ltf_df, htf_df, squeeze_pctile_max, forward_bars,
                                             max_bars_after_squeeze=10, bb_period=20, num_std=2.0,
                                             vol_lookback=100, htf_trend_fast=10, htf_trend_slow=30):
    """events_df (entry_time, raw_return, direction). SQUEEZE+BREAKOUT
    trigger identical to build_vol_squeeze_breakout_events: a bar at or
    below squeeze_pctile_max on ltf_df's own causal volatility_percentile,
    then the first bar within max_bars_after_squeeze whose CLOSE clears
    the Bollinger Band computed at that bar (upper -> bullish, lower ->
    bearish breakout candidate).

    HTF TREND GATE (the only new part): htf_df's own fast/slow SMA
    relationship, shifted forward one HTF bar (so only an ALREADY-CLOSED
    HTF bar's trend state is used, never the one still forming), matched
    onto each LTF timestamp via merge_asof(direction="backward"). A
    bullish breakout only counts when the HTF trend is UP (fast>slow) at
    that moment; a bearish breakout only when it's DOWN. A breakout
    fighting the higher-timeframe trend is dropped entirely -- this only
    tests the "ride with the tide" case, the mechanical mirror of today's
    HTF-peak-gated REVERSAL idea (that one deliberately fades HTF
    extremes; this one deliberately requires HTF agreement).

    Entry lag-corrected: next LTF bar's OPEN after the breakout bar's own
    close confirmed it. Exit forward_bars LTF bars later at that bar's
    close. One trade at a time."""
    cols = ["entry_time", "raw_return", "direction"]
    from indicators import bollinger_bands
    o = (ltf_df["Open"] if "Open" in ltf_df else ltf_df["open"]).to_numpy()
    close = (ltf_df["Close"] if "Close" in ltf_df else ltf_df["close"])
    close_arr = close.to_numpy()
    n = len(ltf_df)
    if (n < max(bb_period, vol_lookback) + max_bars_after_squeeze + forward_bars + 10
            or htf_df is None or len(htf_df) < htf_trend_slow + 2):
        return pd.DataFrame(columns=cols)

    vol_pctile = volatility_percentile(ltf_df, lookback=vol_lookback).to_numpy()
    upper, _, lower = bollinger_bands(close, bb_period, num_std)
    upper_arr, lower_arr = upper.to_numpy(), lower.to_numpy()

    htf_close = htf_df["Close"] if "Close" in htf_df else htf_df["close"]
    htf_fast = htf_close.rolling(htf_trend_fast, min_periods=htf_trend_fast).mean().shift(1)
    htf_slow = htf_close.rolling(htf_trend_slow, min_periods=htf_trend_slow).mean().shift(1)
    htf_ref = pd.DataFrame({"htf_fast": htf_fast, "htf_slow": htf_slow}, index=htf_df.index).dropna()

    ltf_times = pd.DataFrame({"t": ltf_df.index}, index=ltf_df.index)
    merged = pd.merge_asof(ltf_times.sort_index(), htf_ref.sort_index(), left_index=True, right_index=True,
                            direction="backward")
    gate_fast = merged["htf_fast"].to_numpy()
    gate_slow = merged["htf_slow"].to_numpy()

    candidates = []
    i = 0
    while i < n:
        if np.isnan(vol_pctile[i]) or vol_pctile[i] > squeeze_pctile_max:
            i += 1
            continue
        breakout_pos, direction = None, None
        for j in range(i + 1, min(n, i + 1 + max_bars_after_squeeze)):
            if np.isnan(upper_arr[j]) or np.isnan(lower_arr[j]) or np.isnan(gate_fast[j]) or np.isnan(gate_slow[j]):
                continue
            if close_arr[j] > upper_arr[j]:
                breakout_pos, direction = j, "bullish"
                break
            if close_arr[j] < lower_arr[j]:
                breakout_pos, direction = j, "bearish"
                break
        if breakout_pos is not None:
            htf_up = gate_fast[breakout_pos] > gate_slow[breakout_pos]
            if (direction == "bullish" and htf_up) or (direction == "bearish" and not htf_up):
                candidates.append((breakout_pos, direction))
            i = breakout_pos + 1
        else:
            i += 1

    if not candidates:
        return pd.DataFrame(columns=cols)
    candidates.sort(key=lambda c: c[0])
    rows = []
    last_exit_pos = -1
    for pos, direction in candidates:
        entry_pos = pos + 1
        if entry_pos <= last_exit_pos or entry_pos >= n:
            continue
        exit_pos = entry_pos + forward_bars
        if exit_pos >= n:
            continue
        entry_price = o[entry_pos]
        exit_price = close_arr[exit_pos]
        fwd_return = float((exit_price - entry_price) / entry_price)
        rows.append({"entry_time": ltf_df.index[entry_pos], "raw_return": fwd_return, "direction": direction})
        last_exit_pos = exit_pos
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# An 18th strategy, also genuinely new: a pure VOLUME-CLIMAX exhaustion
# idea, distinct from build_price_action_confluence_events' 4-vote system
# (that one uses volume-delta SIGN as just one of four equally-weighted
# votes). This is a single, high-conviction combined condition instead --
# the classic "blow-off top / capitulation bottom" pattern: a fresh
# extreme_lookback-bar price extreme made on a volume bar that is a
# genuine statistical OUTLIER against its own trailing history (not just
# "more than average", a z-score against a rolling mean/std computed only
# from bars STRICTLY BEFORE the signal bar, so the spike itself never
# contaminates its own baseline), plus a rejection wick on the extreme's
# own side. Three independent things (price extreme, volume outlier,
# rejection) all required at once, not a vote count.
@st.cache_data(ttl=_CACHE_TTL)
def build_volume_climax_reversal_events(df, forward_bars, extreme_lookback=20, vol_zscore_lookback=50,
                                         vol_zscore_threshold=2.5, rejection_mult=1.5):
    """events_df (entry_time, raw_return, direction). At any bar t making a
    fresh extreme_lookback-bar high or low (same causal test as build_
    price_action_confluence_events): require Volume[t]'s z-score against
    the ROLLING mean/std of Volume over the vol_zscore_lookback bars
    STRICTLY BEFORE t (never including t itself, so the spike can't
    inflate its own baseline) to exceed vol_zscore_threshold, AND a
    rejection wick on the extreme's own side of at least rejection_mult x
    the bar's own body. All three required together -- a fresh extreme on
    merely-elevated volume with no rejection wick does NOT qualify, nor
    does a huge volume bar making no fresh extreme.

    direction: bearish at a fresh high (climax buying exhausting into
    supply), bullish at a fresh low (climax selling exhausting into
    demand) -- the classic blow-off-top/capitulation-bottom read, not a
    continuation bet.

    Entry lag-corrected: next bar's OPEN after the climax bar's own close
    confirmed it. Exit forward_bars bars later at that bar's close. One
    trade at a time."""
    cols = ["entry_time", "raw_return", "direction"]
    o = (df["Open"] if "Open" in df else df["open"]).to_numpy()
    h = (df["High"] if "High" in df else df["high"]).to_numpy()
    l = (df["Low"] if "Low" in df else df["low"]).to_numpy()
    c = (df["Close"] if "Close" in df else df["close"]).to_numpy()
    vol_col = "Volume" if "Volume" in df else "volume"
    n = len(df)
    lookback_needed = max(extreme_lookback, vol_zscore_lookback)
    if n < lookback_needed + forward_bars + 5 or vol_col not in df:
        return pd.DataFrame(columns=cols)

    v = df[vol_col].to_numpy().astype(float)
    vol_s = pd.Series(v)
    roll_mean = vol_s.rolling(vol_zscore_lookback, min_periods=vol_zscore_lookback).mean().shift(1).to_numpy()
    roll_std = vol_s.rolling(vol_zscore_lookback, min_periods=vol_zscore_lookback).std(ddof=0).shift(1).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_z = np.where(roll_std > 0, (v - roll_mean) / roll_std, np.nan)

    body = np.abs(c - o)
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l

    rows = []
    last_exit_pos = -1
    for t in range(lookback_needed, n - 1 - forward_bars):
        if np.isnan(vol_z[t]) or vol_z[t] < vol_zscore_threshold:
            continue
        window_high = h[t - extreme_lookback:t]
        window_low = l[t - extreme_lookback:t]
        is_new_high = h[t] > window_high.max()
        is_new_low = l[t] < window_low.min()
        if is_new_high and not is_new_low:
            if not (body[t] > 0 and upper_wick[t] >= rejection_mult * body[t]):
                continue
            direction = "bearish"
        elif is_new_low and not is_new_high:
            if not (body[t] > 0 and lower_wick[t] >= rejection_mult * body[t]):
                continue
            direction = "bullish"
        else:
            continue

        entry_pos = t + 1
        if entry_pos <= last_exit_pos:
            continue
        exit_pos = entry_pos + forward_bars
        if exit_pos >= n:
            continue
        entry_price = o[entry_pos]
        exit_price = c[exit_pos]
        fwd_return = float((exit_price - entry_price) / entry_price)
        rows.append({"entry_time": df.index[entry_pos], "raw_return": fwd_return, "direction": direction})
        last_exit_pos = exit_pos
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# A 19th strategy, a direct request: a continuous, every-bar directional
# BIAS reading (not a rare, selective setup like every builder above) --
# where the current bar's close lands relative to the PREVIOUS bar's own
# [low, high] range. Explicitly "no matter the timeframe" -- tested here
# across both 1-minute and hourly data for that reason.
@st.cache_data(ttl=_CACHE_TTL)
def build_prior_range_position_events(df, forward_bars, position_class="all"):
    """events_df (entry_time, raw_return, direction). At every bar t
    (using only bar t's own already-closed data and bar t-1's own already-
    closed high/low/close -- no lookahead), classify bar t's close
    relative to bar t-1's own [low, high] range:

      - close[t] > high[t-1]: "above_range" -> bullish (closed clean
        outside the prior bar's own range to the upside -- momentum
        already broke free of that reference).
      - close[t] < low[t-1]: "below_range" -> bearish (mirror).
      - low[t-1] <= close[t] <= high[t-1] and close[t] >= the range's own
        midpoint: "inside_upper" -> bullish ("closed inside but above
        50%, expect the high to be taken or the trend to continue").
      - low[t-1] <= close[t] <= high[t-1] and close[t] < midpoint:
        "inside_lower" -> bearish (mirror read).

    position_class: "all" tests the flat always-in bias exactly as
    described (every bar gets a directional bet per its own class) --
    one of the four class names isolates just that class, to see which
    part of the framework (if any) actually carries signal on its own.

    This is a continuous bias reading, not a rare setup -- classifications
    happen on nearly every bar by construction, unlike every builder
    above. Still only one open position at a time: a new classification
    is skipped while the prior trade from THIS rule is still open (same
    discipline as everywhere else in this file), so overlapping bias
    reads don't stack into simultaneous positions.

    Entry lag-corrected from the start: classified using bar t's own
    close, but the real fill is bar t+1's OPEN (a live system only learns
    the bar-over-bar comparison after bar t closes). Exit forward_bars
    bars later (from that real entry) at that bar's close."""
    cols = ["entry_time", "raw_return", "direction"]
    o = (df["Open"] if "Open" in df else df["open"]).to_numpy()
    h = (df["High"] if "High" in df else df["high"]).to_numpy()
    l = (df["Low"] if "Low" in df else df["low"]).to_numpy()
    c = (df["Close"] if "Close" in df else df["close"]).to_numpy()
    n = len(df)
    if n < forward_bars + 5:
        return pd.DataFrame(columns=cols)

    rows = []
    last_exit_pos = -1
    for t in range(1, n - 1 - forward_bars):
        prev_high, prev_low = h[t - 1], l[t - 1]
        if prev_high <= prev_low:
            continue
        close_t = c[t]
        if close_t > prev_high:
            cls, direction = "above_range", "bullish"
        elif close_t < prev_low:
            cls, direction = "below_range", "bearish"
        else:
            midpoint = (prev_high + prev_low) / 2.0
            if close_t >= midpoint:
                cls, direction = "inside_upper", "bullish"
            else:
                cls, direction = "inside_lower", "bearish"

        if position_class != "all" and cls != position_class:
            continue

        entry_pos = t + 1
        if entry_pos <= last_exit_pos:
            continue
        exit_pos = entry_pos + forward_bars
        if exit_pos >= n:
            continue
        entry_price = o[entry_pos]
        exit_price = c[exit_pos]
        fwd_return = float((exit_price - entry_price) / entry_price)
        rows.append({"entry_time": df.index[entry_pos], "raw_return": fwd_return, "direction": direction})
        last_exit_pos = exit_pos
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# A 20th strategy, a direct request: a 3-MA "ribbon" trend-following
# system. Enter when three EMAs sit in the trend's own correct stacked
# order (bullish: fast>medium>slow; bearish: mirror) and price pulls back
# to touch ONE specific MA in that stack (touch_ma selects which).
# Exit either on the ACTUAL fast/medium cross (the ribbon genuinely
# untangling), or -- the novel part requested -- an EARLIER exit the
# moment the fast/medium gap is not just shrinking but shrinking at an
# ACCELERATING rate (each bar's convergence bigger than the last), on the
# theory that accelerating convergence is a leading tell a cross is
# coming even before it actually happens. Deliberately NOT a real "ML
# prediction engine" -- just a causal, auditable extrapolation of the
# gap's own recent trajectory (same honest-about-its-own-limits spirit as
# this session's Price Projection tool) -- use_early_exit=False gives the
# plain "wait for the real cross" baseline to compare against directly.
@st.cache_data(ttl=_CACHE_TTL)
def build_ma_ribbon_scale_entry_events(df, touch_ma="fast", ma_fast=9, ma_medium=21, ma_slow=50,
                                        use_early_exit=True, accel_confirm_bars=3, max_hold_bars=150):
    """events_df (entry_time, raw_return, direction).

    ALIGNMENT (causal, bar by bar): bullish when fast>medium>slow, bearish
    when fast<medium<slow, "none" otherwise -- a tangled ribbon has
    nothing to trade.

    ENTRY: while aligned, the first bar whose [low, high] range touches
    touch_ma's own current value (same "retest" concept build_ma_cross_
    retest_events already uses, just against a chosen rung of a 3-MA
    ladder instead of a single fast/slow pair) is a candidate. One trade
    at a time -- a touch inside a still-open previous trade's own window
    is skipped, same discipline as everywhere else in this file.

    EXIT, checked bar by bar going forward from entry:
      1. Real cross: (fast-medium)*direction_sign flips to <= 0 -- the
         ribbon's own leading edge actually crossed.
      2. use_early_exit only: accel_confirm_bars CONSECUTIVE bars where
         the (fast-medium)*direction_sign gap both shrank AND shrank by
         MORE than the bar before it (the convergence itself speeding
         up) -- exits before the real cross, betting the acceleration
         itself is the tell.
      3. max_hold_bars safety cap if neither fires.

    Entry AND exit both lag-corrected: whichever bar's own already-closed
    data confirms the touch/cross/acceleration, the real fill is the
    NEXT bar's open, never that confirming bar's own close -- applied to
    the exit here too, not just entry, since a path-dependent exit is
    exactly the same "decide now, can only act next bar" situation."""
    cols = ["entry_time", "raw_return", "direction"]
    o = (df["Open"] if "Open" in df else df["open"]).to_numpy()
    h = (df["High"] if "High" in df else df["high"]).to_numpy()
    l = (df["Low"] if "Low" in df else df["low"]).to_numpy()
    c = (df["Close"] if "Close" in df else df["close"]).to_numpy()
    n = len(df)
    if n < ma_slow + max_hold_bars + 10:
        return pd.DataFrame(columns=cols)

    from indicators import ema as _ema
    close_s = df["Close"] if "Close" in df else df["close"]
    fast_arr = _ema(close_s, ma_fast).to_numpy()
    medium_arr = _ema(close_s, ma_medium).to_numpy()
    slow_arr = _ema(close_s, ma_slow).to_numpy()
    touch_arr = {"fast": fast_arr, "medium": medium_arr, "slow": slow_arr}[touch_ma]

    rows = []
    last_exit_pos = -1
    t = ma_slow
    while t < n - 1 - max_hold_bars:
        if np.isnan(fast_arr[t]) or np.isnan(medium_arr[t]) or np.isnan(slow_arr[t]):
            t += 1
            continue
        if fast_arr[t] > medium_arr[t] > slow_arr[t]:
            direction, dir_sign = "bullish", 1
        elif fast_arr[t] < medium_arr[t] < slow_arr[t]:
            direction, dir_sign = "bearish", -1
        else:
            t += 1
            continue

        if not (l[t] <= touch_arr[t] <= h[t]):
            t += 1
            continue
        touch_pos = t
        entry_pos = touch_pos + 1
        if entry_pos <= last_exit_pos or entry_pos >= n:
            t += 1
            continue

        gap_prev = (fast_arr[entry_pos] - medium_arr[entry_pos]) * dir_sign if not (
            np.isnan(fast_arr[entry_pos]) or np.isnan(medium_arr[entry_pos])) else None
        shrink_prev = None
        accel_streak = 0
        exit_pos = None
        scan_end = min(n, entry_pos + 1 + max_hold_bars)
        for k in range(entry_pos + 1, scan_end):
            if np.isnan(fast_arr[k]) or np.isnan(medium_arr[k]):
                continue
            gap_k = (fast_arr[k] - medium_arr[k]) * dir_sign
            if gap_k <= 0:
                exit_pos = k
                break
            if use_early_exit and gap_prev is not None:
                shrink_k = gap_prev - gap_k  # positive = converging this bar
                if shrink_k > 0 and shrink_prev is not None and shrink_k > shrink_prev:
                    accel_streak += 1
                    if accel_streak >= accel_confirm_bars:
                        exit_pos = k
                        break
                else:
                    accel_streak = 0
                shrink_prev = shrink_k
            gap_prev = gap_k
        if exit_pos is None:
            exit_pos = scan_end - 1
        if exit_pos <= entry_pos or exit_pos >= n - 1:
            t = touch_pos + 1
            continue

        real_entry_pos = entry_pos + 1
        real_exit_pos = exit_pos + 1
        if real_entry_pos >= n or real_exit_pos >= n or real_exit_pos <= real_entry_pos:
            t = touch_pos + 1
            continue
        entry_price = o[real_entry_pos]
        exit_price = o[real_exit_pos]
        fwd_return = float((exit_price - entry_price) / entry_price)
        rows.append({"entry_time": df.index[real_entry_pos], "raw_return": fwd_return, "direction": direction})
        last_exit_pos = real_exit_pos
        t = touch_pos + 1
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# A 21st strategy: classic ICT SMT (Smart Money Technique) divergence --
# two CORRELATED instruments (this session's own ES/NQ/YM/RTY cluster is
# exactly the classic textbook SMT set) where one makes a fresh extreme
# but the other does NOT confirm it. The read: a new high/low nothing
# else correlated is backing isn't a genuine, broad move -- expect the
# unconfirmed instrument to reverse.
@st.cache_data(ttl=_CACHE_TTL)
def build_smt_divergence_events(signal_df, confirm_df, forward_bars, extreme_lookback=20):
    """events_df (entry_time, raw_return, direction) -- on signal_df.

    At any bar t where signal_df makes a fresh extreme_lookback-bar HIGH
    (h[t] > max of the prior extreme_lookback highs), look up confirm_df's
    own state as of the last confirm_df bar AT OR BEFORE t (merge_asof,
    backward -- never a lookahead into confirm_df's future) and check
    whether confirm_df ALSO made its own fresh extreme_lookback-bar high
    at THAT bar. If confirm_df did NOT, that's bearish SMT divergence on
    signal_df. Mirror: signal_df fresh LOW not confirmed by confirm_df's
    own fresh low -> bullish divergence.

    confirm_df's own fresh-extreme check only ever uses confirm_df's OWN
    history up to and including its own matched bar -- no lookahead on
    either side of the pair.

    Entry lag-corrected: divergence confirmed using signal_df's own bar
    t close, but the real fill is bar t+1's OPEN on signal_df. Exit
    forward_bars bars later (on signal_df) at that bar's close. One
    trade at a time."""
    cols = ["entry_time", "raw_return", "direction"]
    s_o = (signal_df["Open"] if "Open" in signal_df else signal_df["open"]).to_numpy()
    s_h = (signal_df["High"] if "High" in signal_df else signal_df["high"]).to_numpy()
    s_l = (signal_df["Low"] if "Low" in signal_df else signal_df["low"]).to_numpy()
    s_c = (signal_df["Close"] if "Close" in signal_df else signal_df["close"]).to_numpy()
    n = len(signal_df)
    if n < extreme_lookback + forward_bars + 5 or confirm_df is None or len(confirm_df) < extreme_lookback + 2:
        return pd.DataFrame(columns=cols)

    c_h = (confirm_df["High"] if "High" in confirm_df else confirm_df["high"]).to_numpy()
    c_l = (confirm_df["Low"] if "Low" in confirm_df else confirm_df["low"]).to_numpy()
    c_n = len(confirm_df)
    c_fresh_high = np.array([c_h[i] > c_h[max(0, i - extreme_lookback):i].max() if i >= extreme_lookback else False
                              for i in range(c_n)])
    c_fresh_low = np.array([c_l[i] < c_l[max(0, i - extreme_lookback):i].min() if i >= extreme_lookback else False
                             for i in range(c_n)])
    confirm_ref = pd.DataFrame({"fresh_high": c_fresh_high, "fresh_low": c_fresh_low}, index=confirm_df.index)

    sig_times = pd.DataFrame({"t": signal_df.index}, index=signal_df.index)
    merged = pd.merge_asof(sig_times.sort_index(), confirm_ref.sort_index(), left_index=True, right_index=True,
                            direction="backward")
    confirm_fresh_high = merged["fresh_high"].to_numpy()
    confirm_fresh_low = merged["fresh_low"].to_numpy()

    rows = []
    last_exit_pos = -1
    for t in range(extreme_lookback, n - 1 - forward_bars):
        window_high = s_h[t - extreme_lookback:t]
        window_low = s_l[t - extreme_lookback:t]
        is_new_high = s_h[t] > window_high.max()
        is_new_low = s_l[t] < window_low.min()
        if is_new_high and not is_new_low:
            if confirm_fresh_high[t]:
                continue  # confirmed by the other instrument -- not a divergence
            direction = "bearish"
        elif is_new_low and not is_new_high:
            if confirm_fresh_low[t]:
                continue
            direction = "bullish"
        else:
            continue

        entry_pos = t + 1
        if entry_pos <= last_exit_pos or entry_pos >= n:
            continue
        exit_pos = entry_pos + forward_bars
        if exit_pos >= n:
            continue
        entry_price = s_o[entry_pos]
        exit_price = s_c[exit_pos]
        fwd_return = float((exit_price - entry_price) / entry_price)
        rows.append({"entry_time": signal_df.index[entry_pos], "raw_return": fwd_return, "direction": direction})
        last_exit_pos = exit_pos
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def build_clean_expansion_retracement_events(df, min_streak=3, retr_window_bars=40, forward_bars=10,
                                              min_body_ratio=DISPLACEMENT_MIN_BODY_RATIO):
    """events_df (entry_time, raw_return, direction).

    Thesis (a same-session finding, not assumed): a run of min_streak+
    consecutive same-direction FVG/Order Block zones that ALL get
    genuinely respected — no close breaks them, none gets 100%
    wick-mitigated, checked over a bounded retr_window_bars-bar window so
    a zone near the end of that window isn't unfairly called "clean" just
    because it hasn't had time to fail yet — measurably raises the odds
    that the FIRST opposite-direction zone forming right after it also
    holds (a same-session study on 8 tickers/1h/2yr found +3.1pp on
    average, same-sign on 7/8, permutation p=0.004 — see this project's
    own session notes; that study only measured zone-hold RATES, this
    function turns it into an actual entry). Trade the ORIGINAL expansion's
    own direction (classic ICT "buy the dip at a respected zone during a
    pullback in an uptrend"), entering once that first opposite-direction
    retracement zone appears — NOT the retracement's own direction: tried
    that first (fade/continue the pullback itself) and it backtested to a
    statistically significant NEGATIVE mean return on ES=F (same trades,
    wrong-signed), i.e. the exact mirror of this rule — confirms the
    retracement zone reads as a continuation entry, not a place to keep
    riding the pullback.

    Zone formation timing (so entries stay causal, no lookahead):
      - FVG: "start" = idx[i-1]; the gap isn't confirmed until candle i+1
        closes (start_pos + 2), so entry fills at (start_pos + 3)'s open.
      - Order Block: "start" = the opposing candle itself; the breakout
        that confirms it is the very next candle (start_pos + 1), so
        entry fills at (start_pos + 2)'s open.
    Same "decide on the confirming candle's own close, fill at the next
    candle's open" rule this whole module already uses elsewhere — these
    two detectors just confirm one bar apart from each other.

    One trade at a time (last_exit_pos), same as every other builder
    here. Shares its actual zone/streak/trigger detection with
    _clean_expansion_touches (used by find_live_validated_signal for the
    sidebar's live scan and the Survivors tab's "apply to chart") — one
    definition of "what counts as a signal," not two that could drift
    apart."""
    cols = ["entry_time", "raw_return", "direction"]
    close_arr = (df["Close"] if "Close" in df else df["close"]).to_numpy()
    open_arr = (df["Open"] if "Open" in df else df["open"]).to_numpy()
    n = len(df)
    touches = _clean_expansion_touches(df, min_streak, retr_window_bars, min_body_ratio=min_body_ratio)

    rows = []
    last_exit_pos = -1
    for t in touches:
        entry_pos = t["confirm_pos"] + 1
        if entry_pos <= last_exit_pos or entry_pos >= n:
            continue
        exit_pos = entry_pos + forward_bars
        if exit_pos >= n:
            continue
        entry_price = open_arr[entry_pos]
        exit_price = close_arr[exit_pos]
        fwd_return = float((exit_price - entry_price) / entry_price)
        rows.append({"entry_time": df.index[entry_pos], "raw_return": fwd_return, "direction": t["direction"]})
        last_exit_pos = exit_pos
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def _confirmed_zone_pos(z):
    """When a zone actually becomes KNOWABLE, not just when it starts —
    FVG's own "start" is idx[i-1], but the gap isn't confirmed until
    candle i+1 closes (start_pos + 2); an Order Block's "start" is the
    opposing candle itself, confirmed one candle later (start_pos + 1).
    Shared by every clean-zone-streak reader below so none of them can
    disagree on this."""
    return z["_pos"] + (2 if z["layer"] == "FVG" else 1)


def _scan_clean_zone_streaks(df, min_streak, retr_window_bars, min_body_ratio=DISPLACEMENT_MIN_BODY_RATIO):
    """Shared detection core behind every clean-zone-streak strategy in
    this module (build_clean_expansion_retracement_events and
    build_clean_retracement_resumption_events, plus find_live_validated_
    signal's own live-touch lookups for both) — one definition of "what
    counts as a streak," so they can't quietly drift apart. Returns
    (merged, qualifying): `merged` is every FVG/Order Block zone in
    formation order, each tagged with its own bar position (_pos) and
    layer; `qualifying` is every run of min_streak+ CONSECUTIVE zones,
    same direction, that ALL get genuinely respected within
    retr_window_bars of their own formation (no close breaks them, none
    gets 100% wick-mitigated) — a single non-respected or opposite-
    direction zone ends the current run. Each qualifying entry is
    {"zones", "direction", "last_pos"} — the full streak, not just where
    it ends, since a caller may want to react to any zone in it (the
    retracement-resumption strategy needs the SECOND streak's own last
    zone, not just its position)."""
    close_arr = (df["Close"] if "Close" in df else df["close"]).to_numpy()
    n = len(df)
    if n < retr_window_bars + 10:
        return [], []
    pos_by_time = {t: i for i, t in enumerate(df.index)}

    merged = []
    for z in detect_fvgs(df, min_body_ratio=min_body_ratio, max_scan_bars=retr_window_bars):
        merged.append({**z, "layer": "FVG"})
    for z in detect_order_blocks(df, min_body_ratio=min_body_ratio, max_scan_bars=retr_window_bars):
        merged.append({**z, "layer": "Order Block"})
    merged.sort(key=lambda z: z["start"])
    for z in merged:
        z["_pos"] = pos_by_time.get(z["start"])
    merged = [z for z in merged if z["_pos"] is not None]

    def _respected(z):
        mitigated_flag = z.get("filled") if z["layer"] == "FVG" else z.get("mitigated")
        if mitigated_flag:
            return False
        top = z.get("raw_top", z["top"])
        bottom = z.get("raw_bottom", z["bottom"])
        end_pos = min(n, z["_pos"] + 1 + retr_window_bars)
        future_closes = close_arr[z["_pos"] + 1:end_pos]
        if len(future_closes) == 0:
            return True
        if z["type"] == "bullish":
            return not bool((future_closes < bottom).any())
        else:
            return not bool((future_closes > top).any())

    qualifying = []
    current = []
    for z in merged:
        ok = _respected(z)
        if current and z["type"] != current[0]["type"]:
            if len(current) >= min_streak:
                qualifying.append({"zones": list(current), "direction": current[0]["type"],
                                    "last_pos": current[-1]["_pos"]})
            current = [z] if ok else []
        elif ok:
            current.append(z)
        else:
            if len(current) >= min_streak:
                qualifying.append({"zones": list(current), "direction": current[0]["type"],
                                    "last_pos": current[-1]["_pos"]})
            current = []
    if len(current) >= min_streak:
        qualifying.append({"zones": list(current), "direction": current[0]["type"],
                            "last_pos": current[-1]["_pos"]})
    return merged, qualifying


def _clean_expansion_touches(df, min_streak, retr_window_bars, min_body_ratio=DISPLACEMENT_MIN_BODY_RATIO):
    """The detection core behind build_clean_expansion_retracement_events,
    factored out so find_live_validated_signal can check "is there a
    signal as of the last bar in df" without needing a completed
    forward_bars hold the way a real backtest event does — a live signal
    by definition hasn't exited yet. Same shape _qualifying_touches
    already returns for the original detector-reaction family (touch_pos,
    confirm_pos, direction, zone_top, zone_bottom, zone_start), same
    field meanings, so find_live_validated_signal's own downstream
    stop-price/live-check logic needs zero changes to accept either.

    "zone_top"/"zone_bottom" here describe the RETRACEMENT trigger zone
    (the one actually being entered on), not the expansion streak itself
    — that's the zone whose far edge is the natural stop, exactly the
    role zone_top/zone_bottom plays for the original family too."""
    merged, qualifying = _scan_clean_zone_streaks(df, min_streak, retr_window_bars, min_body_ratio)
    touches = []
    for streak in qualifying:
        last_pos, expansion_dir = streak["last_pos"], streak["direction"]
        retr_zone = next((z for z in merged
                           if z["_pos"] > last_pos and z["_pos"] <= last_pos + retr_window_bars
                           and z["type"] != expansion_dir), None)
        if retr_zone is None:
            continue
        touches.append({
            "touch_pos": retr_zone["_pos"], "confirm_pos": _confirmed_zone_pos(retr_zone),
            "direction": expansion_dir,
            "zone_top": retr_zone.get("raw_top", retr_zone["top"]),
            "zone_bottom": retr_zone.get("raw_bottom", retr_zone["bottom"]),
            "zone_start": retr_zone["start"],
        })
    return touches


def _clean_retracement_resumption_touches(df, min_streak, retr_window_bars,
                                           min_body_ratio=DISPLACEMENT_MIN_BODY_RATIO):
    """The mirror question to _clean_expansion_touches: instead of "does a
    clean expansion predict a good entry AT the first retracement zone,"
    this asks "does the RETRACEMENT ITSELF being clean (its own qualifying
    min_streak+ run of respected zones, not just one) predict anything."
    Looks for two ADJACENT qualifying streaks of opposite direction —
    streak A (the expansion), streak B immediately after it (the
    retracement, itself clean) — and reads streak B's own completion as
    the signal.

    Trades streak B's OWN direction (the retracement continuing into a
    genuine reversal), NOT a resumption of streak A: tried resumption
    first (entering back in streak A's direction) and it backtested to a
    statistically significant NEGATIVE mean return on BTC-USD 1h (same
    trades, wrong-signed) — the exact mirror of build_clean_expansion_
    retracement_events' own history (that one ALSO needed the opposite
    of its first guess). Read together, the two findings say something
    coherent: a retracement that's merely a lone zone reads as "noise,
    the trend will resume" (that IS the other strategy's own edge), but
    a retracement clean enough to form its OWN multi-zone streak reads
    as real strength changing hands, not a dip to buy.

    A streak that isn't immediately followed by an OPPOSITE qualifying
    streak (the expansion just fizzles, or the "retracement" never
    itself forms a clean run) produces no touch — this is a strictly
    narrower, stricter-filtered condition than _clean_expansion_touches,
    by design.

    zone_top/zone_bottom/zone_start describe streak B's own LAST zone —
    the retracement's own most recent respected level, the natural stop
    (a break through it undoes the very thing that made the retracement
    read as "clean")."""
    _, qualifying = _scan_clean_zone_streaks(df, min_streak, retr_window_bars, min_body_ratio)
    touches = []
    for i in range(len(qualifying) - 1):
        a, b = qualifying[i], qualifying[i + 1]
        if a["direction"] == b["direction"]:
            continue  # both the same direction -- not an expansion/retracement pair
        last_zone = b["zones"][-1]
        touches.append({
            "touch_pos": last_zone["_pos"], "confirm_pos": _confirmed_zone_pos(last_zone),
            "direction": b["direction"],
            "zone_top": last_zone.get("raw_top", last_zone["top"]),
            "zone_bottom": last_zone.get("raw_bottom", last_zone["bottom"]),
            "zone_start": last_zone["start"],
        })
    return touches


def build_clean_retracement_resumption_events(df, min_streak=3, retr_window_bars=40, forward_bars=10,
                                               min_body_ratio=DISPLACEMENT_MIN_BODY_RATIO):
    """events_df (entry_time, raw_return, direction) — see
    _clean_retracement_resumption_touches for the exact rule. Same
    causal-fill (confirm_pos + 1's open) and one-trade-at-a-time
    convention as every other builder here."""
    cols = ["entry_time", "raw_return", "direction"]
    close_arr = (df["Close"] if "Close" in df else df["close"]).to_numpy()
    open_arr = (df["Open"] if "Open" in df else df["open"]).to_numpy()
    n = len(df)
    touches = _clean_retracement_resumption_touches(df, min_streak, retr_window_bars,
                                                     min_body_ratio=min_body_ratio)
    rows = []
    last_exit_pos = -1
    for t in touches:
        entry_pos = t["confirm_pos"] + 1
        if entry_pos <= last_exit_pos or entry_pos >= n:
            continue
        exit_pos = entry_pos + forward_bars
        if exit_pos >= n:
            continue
        entry_price = open_arr[entry_pos]
        exit_price = close_arr[exit_pos]
        fwd_return = float((exit_price - entry_price) / entry_price)
        rows.append({"entry_time": df.index[entry_pos], "raw_return": fwd_return, "direction": t["direction"]})
        last_exit_pos = exit_pos
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def _clean_structure_touches(df, min_swings=3, atr_period=14, atr_mult=1.5):
    """A "clean trend" defined structurally instead of by zones: a run of
    min_swings+ CONSECUTIVE confirmed swing pivots (detect_swings — ATR-
    scaled ZigZag, strictly alternating high/low by construction) each
    one extending the same direction relative to the pivot of the SAME
    kind two positions back — a proper higher-high/higher-low sequence
    for an uptrend, lower-low/lower-high for a downtrend. A single pivot
    that fails to extend ends the run immediately, same "one bad one
    ends the streak" discipline as _clean_expansion_touches' own zone
    streaks, just applied to swing structure instead of FVG/OB zones —
    this project's OTHER natural reading of "clean" for a trend.

    Emits a touch every time the run is AT OR PAST min_swings length (not
    just once) — a trend that keeps extending keeps re-confirming itself,
    each one its own fresh continuation entry; one-trade-at-a-time in the
    caller naturally spaces these out. Same {"touch_pos","confirm_pos",
    "direction","zone_top","zone_bottom","zone_start"} shape every other
    touch-finder in this module returns — zone_top/zone_bottom here are
    both the LAST swing against the trend (the low behind a higher-low
    run, the high behind a lower-high run), the natural structural stop:
    a break back through it invalidates the "clean trend" read.

    confirmed_pos (not pos) is what a caller can actually act on — using
    a swing's own pos would be a lookahead bug (see detect_swings' own
    docstring: a swing isn't KNOWN until price has already reversed away
    from it)."""
    highs, lows = detect_swings(df, atr_period=atr_period, atr_mult=atr_mult)
    tagged = ([{**h, "kind": "high"} for h in highs] + [{**l, "kind": "low"} for l in lows])
    tagged.sort(key=lambda p: p["pos"])

    touches = []
    run_dir = None
    run_len = 0
    last_price = {"high": None, "low": None}
    for piv in tagged:
        kind = piv["kind"]
        prev_price = last_price[kind]
        last_price[kind] = piv["price"]
        if prev_price is None:
            run_dir, run_len = None, 0
            continue
        extends_up = piv["price"] > prev_price
        extends_down = piv["price"] < prev_price
        if run_dir == "up":
            if extends_up:
                run_len += 1
            else:
                run_dir = "down" if extends_down else None
                run_len = 1 if run_dir else 0
        elif run_dir == "down":
            if extends_down:
                run_len += 1
            else:
                run_dir = "up" if extends_up else None
                run_len = 1 if run_dir else 0
        else:
            if extends_up:
                run_dir, run_len = "up", 1
            elif extends_down:
                run_dir, run_len = "down", 1
            else:
                run_dir, run_len = None, 0
        if run_dir and run_len >= min_swings:
            # The stop reference is the LAST swing AGAINST the trend —
            # for an uptrend that's the most recent confirmed LOW (not
            # necessarily THIS pivot, which could itself be the high
            # side of the pair); last_price["low"]/["high"] already
            # holds exactly that, updated in swing-formation order same
            # as everything else here.
            stop_ref = last_price["low"] if run_dir == "up" else last_price["high"]
            touches.append({
                "touch_pos": piv["pos"], "confirm_pos": piv["confirmed_pos"],
                "direction": "bullish" if run_dir == "up" else "bearish",
                "zone_top": stop_ref, "zone_bottom": stop_ref, "zone_start": piv["time"],
            })
    return touches


def build_clean_structure_trend_events(df, min_swings=3, forward_bars=10, atr_period=14, atr_mult=1.5):
    """events_df (entry_time, raw_return, direction) — the structural
    sibling of build_clean_expansion_retracement_events: same "clean ==
    no violation yet" thesis, applied to swing highs/lows instead of
    FVG/OB zones (see _clean_structure_touches' own docstring for the
    exact definition). Trade WITH the confirmed trend direction, entering
    once a run of min_swings+ consecutive higher-highs/higher-lows (or
    the bearish mirror) is established — a different, structure-based
    answer to the same "clean trend" question this session's own zone-
    based strategy already validated an edge for, not a variant of it.

    Same causal-fill convention as every other builder here: a swing's
    own confirmed_pos is when it becomes KNOWABLE (see detect_swings),
    entry fills at confirmed_pos + 1's open — not confirmed_pos's own
    close, which would already be known-in-the-past by the time a real
    trader could act. One trade at a time."""
    cols = ["entry_time", "raw_return", "direction"]
    close_arr = (df["Close"] if "Close" in df else df["close"]).to_numpy()
    open_arr = (df["Open"] if "Open" in df else df["open"]).to_numpy()
    n = len(df)
    touches = _clean_structure_touches(df, min_swings=min_swings, atr_period=atr_period, atr_mult=atr_mult)

    rows = []
    last_exit_pos = -1
    for t in touches:
        entry_pos = t["confirm_pos"] + 1
        if entry_pos <= last_exit_pos or entry_pos >= n:
            continue
        exit_pos = entry_pos + forward_bars
        if exit_pos >= n:
            continue
        entry_price = open_arr[entry_pos]
        exit_price = close_arr[exit_pos]
        fwd_return = float((exit_price - entry_price) / entry_price)
        rows.append({"entry_time": df.index[entry_pos], "raw_return": fwd_return, "direction": t["direction"]})
        last_exit_pos = exit_pos
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def _clean_expansion_liquidity_touches(df, min_streak, retr_window_bars, sweep_window_bars=10,
                                        min_body_ratio=DISPLACEMENT_MIN_BODY_RATIO):
    """A stricter version of _clean_expansion_touches: the same clean-
    expansion-then-first-opposite-zone setup, kept ONLY when a genuine
    liquidity sweep (detect_liquidity_sweeps — a real swing high/low
    actually getting wicked through, not an inferred shape) pointing the
    SAME direction as the continuation trade happened during the
    retracement leg itself (between the expansion streak's own end and
    the retracement zone's confirmation, plus a small sweep_window_bars
    margin on each side for a sweep that lands just outside that exact
    span). Thesis: a clean expansion's retracement zone is a plausible
    entry on its own (already validated); one that ALSO coincides with
    real stops getting run is a stronger version of the same setup — a
    mechanistically grounded reason (actual resting orders triggered),
    not just another inferred pattern. Same touch shape as
    _clean_expansion_touches — this filters that function's own touches
    down to the subset with a matching sweep, it doesn't find new ones."""
    merged, qualifying = _scan_clean_zone_streaks(df, min_streak, retr_window_bars, min_body_ratio)
    if not qualifying:
        return []
    pos_by_time = {t: i for i, t in enumerate(df.index)}
    sweeps = detect_liquidity_sweeps(df)
    for s in sweeps:
        s["_end_pos"] = pos_by_time.get(s["end"])
    sweeps = [s for s in sweeps if s["_end_pos"] is not None]

    touches = []
    for streak in qualifying:
        last_pos, expansion_dir = streak["last_pos"], streak["direction"]
        retr_zone = next((z for z in merged
                           if z["_pos"] > last_pos and z["_pos"] <= last_pos + retr_window_bars
                           and z["type"] != expansion_dir), None)
        if retr_zone is None:
            continue
        confirm_pos = _confirmed_zone_pos(retr_zone)
        has_sweep = any(s["type"] == expansion_dir
                         and (last_pos - sweep_window_bars) <= s["_end_pos"] <= (confirm_pos + sweep_window_bars)
                         for s in sweeps)
        if not has_sweep:
            continue
        touches.append({
            "touch_pos": retr_zone["_pos"], "confirm_pos": confirm_pos,
            "direction": expansion_dir,
            "zone_top": retr_zone.get("raw_top", retr_zone["top"]),
            "zone_bottom": retr_zone.get("raw_bottom", retr_zone["bottom"]),
            "zone_start": retr_zone["start"],
        })
    return touches


def build_clean_expansion_liquidity_events(df, min_streak=3, retr_window_bars=40, forward_bars=10,
                                            sweep_window_bars=10, min_body_ratio=DISPLACEMENT_MIN_BODY_RATIO):
    """events_df (entry_time, raw_return, direction) — see
    _clean_expansion_liquidity_touches for the exact rule. Same causal-
    fill (confirm_pos + 1's open) and one-trade-at-a-time convention as
    every other builder here."""
    cols = ["entry_time", "raw_return", "direction"]
    close_arr = (df["Close"] if "Close" in df else df["close"]).to_numpy()
    open_arr = (df["Open"] if "Open" in df else df["open"]).to_numpy()
    n = len(df)
    touches = _clean_expansion_liquidity_touches(df, min_streak, retr_window_bars, sweep_window_bars,
                                                  min_body_ratio=min_body_ratio)
    rows = []
    last_exit_pos = -1
    for t in touches:
        entry_pos = t["confirm_pos"] + 1
        if entry_pos <= last_exit_pos or entry_pos >= n:
            continue
        exit_pos = entry_pos + forward_bars
        if exit_pos >= n:
            continue
        entry_price = open_arr[entry_pos]
        exit_price = close_arr[exit_pos]
        fwd_return = float((exit_price - entry_price) / entry_price)
        rows.append({"entry_time": df.index[entry_pos], "raw_return": fwd_return, "direction": t["direction"]})
        last_exit_pos = exit_pos
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def _htf_trend_direction(daily_df, ema_daily, touch_time):
    """Which way the daily trend was pointing as of the last COMPLETED
    daily bar strictly before touch_time — never the still-forming bar,
    which would be a lookahead (the daily candle touch_time falls inside
    hasn't closed yet at that moment). None when there's no prior daily
    history at all, or its own 50-EMA hasn't warmed up yet."""
    prior = daily_df[daily_df.index < touch_time]
    if len(prior) < 2:
        return None
    last_close = prior["Close"].iloc[-1] if "Close" in prior else prior["close"].iloc[-1]
    last_ema = ema_daily.reindex(prior.index).iloc[-1]
    if pd.isna(last_ema):
        return None
    return "bullish" if last_close > last_ema else "bearish"


def _filter_touches_by_htf_trend(df, touches, daily_df, ema_period=50):
    """Keeps only the touches whose own direction agrees with the daily
    trend (Close vs its own ema_period-EMA) as of the last daily bar
    already closed at that touch's own moment — see _htf_trend_direction.
    Same-session finding this filters toward: on 5 already-validated
    (ticker, setting) clean-expansion combos, HTF-aligned trades averaged
    +1.50% vs +0.95% for disaligned ones (p=0.01, permutation test) — both
    still positive, so this is a magnitude booster, not a loss filter."""
    ema_daily = (daily_df["Close"] if "Close" in daily_df else daily_df["close"]).ewm(span=ema_period, adjust=False).mean()
    kept = []
    for t in touches:
        touch_time = df.index[t["confirm_pos"]]
        trend = _htf_trend_direction(daily_df, ema_daily, touch_time)
        if trend is not None and trend == t["direction"]:
            kept.append(t)
    return kept


def _events_from_touches(df, touches, forward_bars):
    """Shared entry/exit mechanics (causal fill, one-trade-at-a-time) for
    any already-built touches list — every clean-* builder above
    duplicated this same loop; the HTF-filtered variants below are what
    finally made sharing it worth doing."""
    cols = ["entry_time", "raw_return", "direction"]
    close_arr = (df["Close"] if "Close" in df else df["close"]).to_numpy()
    open_arr = (df["Open"] if "Open" in df else df["open"]).to_numpy()
    n = len(df)
    rows = []
    last_exit_pos = -1
    for t in sorted(touches, key=lambda t: t["confirm_pos"]):
        entry_pos = t["confirm_pos"] + 1
        if entry_pos <= last_exit_pos or entry_pos >= n:
            continue
        exit_pos = entry_pos + forward_bars
        if exit_pos >= n:
            continue
        entry_price = open_arr[entry_pos]
        exit_price = close_arr[exit_pos]
        fwd_return = float((exit_price - entry_price) / entry_price)
        rows.append({"entry_time": df.index[entry_pos], "raw_return": fwd_return, "direction": t["direction"]})
        last_exit_pos = exit_pos
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def build_clean_expansion_retracement_htf_events(df, daily_df, min_streak=3, retr_window_bars=40, forward_bars=10,
                                                  ema_period=50, min_body_ratio=DISPLACEMENT_MIN_BODY_RATIO):
    """build_clean_expansion_retracement_events, gated to only the touches
    where the daily trend agrees with the trade's own direction (see
    _filter_touches_by_htf_trend) — same participants visible at two
    timescales at once, tested as a magnitude booster, not a new entry
    rule. daily_df is a plain argument, not run_deep_backtest's own
    (JSON-logged) settings dict — a dataframe can't serialize into the
    trial log anyway, and every OTHER setting here already fully
    describes the rule that produced a given result."""
    touches = _clean_expansion_touches(df, min_streak, retr_window_bars, min_body_ratio=min_body_ratio)
    touches = _filter_touches_by_htf_trend(df, touches, daily_df, ema_period)
    return _events_from_touches(df, touches, forward_bars)


def build_clean_expansion_liquidity_htf_events(df, daily_df, min_streak=3, retr_window_bars=40, forward_bars=10,
                                                sweep_window_bars=10, ema_period=50,
                                                min_body_ratio=DISPLACEMENT_MIN_BODY_RATIO):
    """build_clean_expansion_liquidity_events, HTF-trend-gated the same
    way build_clean_expansion_retracement_htf_events is — see that
    function's own docstring."""
    touches = _clean_expansion_liquidity_touches(df, min_streak, retr_window_bars, sweep_window_bars,
                                                  min_body_ratio=min_body_ratio)
    touches = _filter_touches_by_htf_trend(df, touches, daily_df, ema_period)
    return _events_from_touches(df, touches, forward_bars)


def _simulate_fixed_rr(df, touches, rr_multiple=0.5, atr_mult_stop=1.0, max_bars=40):
    """A genuinely different EXIT mechanic on the SAME already-validated
    entries (touches, from any of the _clean_*_touches finders above) —
    a fixed stop (atr_mult_stop x ATR at entry) and a fixed target
    (rr_multiple x that same distance), walked forward bar-by-bar until
    one is touched, instead of the time-based forward_bars close-to-close
    exit every OTHER builder here uses. This is the honest way to chase
    a higher win rate: a smaller target relative to the stop hits more
    often BY CONSTRUCTION, so testing this only means something if the
    resulting expectancy (mean R) is ALSO still positive and beats
    random direction-calling — see run_deep_backtest_rr's own real-vs-
    flipped permutation test for that check.

    ATR at entry, not the zone's own edge: the zone edge (used as the
    stop for the time-exit builders) is sometimes clamped to an
    artificial 0.1%-of-price distance when the zone sits on the wrong
    side of entry (see _signal_from_touches' own comment) — using that
    as a FIXED R-multiple's own risk unit was tried first and produced
    nonsense (46% win rate at RR=0.3, which should be far higher for a
    target that much closer than the stop) because "risk" was sometimes
    a near-zero distance the very next candle's ordinary noise clears
    either way. ATR is a stable, always-real distance regardless of
    which side of entry the zone landed on.

    On a bar where both stop and target fall inside its own high-low
    range, this assumes the stop hit first — the standard conservative
    convention for reconstructing intrabar order from OHLC alone, which
    has no real path between the open and close to check.

    Returns a DataFrame (entry_time, r_real, r_flip, direction) —
    r_real is the realized R multiple trading the touch's own real
    direction; r_flip is what the SAME touch would have realized had the
    direction been the opposite call, computed from the same entry bar
    with mirrored stop/target — the pairing run_deep_backtest_rr's own
    permutation test needs to ask "did calling the real direction
    actually beat a coin flip on this exact set of entries," not just
    "was the mean positive," which a merely lucky exit rule could
    produce even from uninformed entries."""
    high = (df["High"] if "High" in df else df["high"]).to_numpy()
    low = (df["Low"] if "Low" in df else df["low"]).to_numpy()
    close = (df["Close"] if "Close" in df else df["close"]).to_numpy()
    open_ = (df["Open"] if "Open" in df else df["open"]).to_numpy()
    atr = atr_series(df).to_numpy()
    n = len(df)

    def _run(entry_pos, entry_price, risk, direction):
        if direction == "bullish":
            stop_price = entry_price - risk
            target_price = entry_price + risk * rr_multiple
        else:
            stop_price = entry_price + risk
            target_price = entry_price - risk * rr_multiple
        for j in range(entry_pos, min(n, entry_pos + max_bars)):
            h, l = high[j], low[j]
            if direction == "bullish":
                hit_stop, hit_target = l <= stop_price, h >= target_price
            else:
                hit_stop, hit_target = h >= stop_price, l <= target_price
            if hit_stop:
                return -1.0, j
            if hit_target:
                return rr_multiple, j
        j = min(n - 1, entry_pos + max_bars - 1)
        exit_price = close[j]
        realized = (exit_price - entry_price) if direction == "bullish" else (entry_price - exit_price)
        return realized / risk, j

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
        real_dir = t["direction"]
        opp_dir = "bearish" if real_dir == "bullish" else "bullish"
        r_real, exit_pos = _run(entry_pos, entry_price, risk, real_dir)
        r_flip, _ = _run(entry_pos, entry_price, risk, opp_dir)
        rows.append({"entry_time": df.index[entry_pos], "r_real": r_real, "r_flip": r_flip, "direction": real_dir})
        last_exit_pos = exit_pos
    return pd.DataFrame(rows, columns=["entry_time", "r_real", "r_flip", "direction"])


def run_deep_backtest_rr(ticker, tf_label, events_df, settings, label, train_fraction=0.8, n_permutations=5000,
                          seed=0):
    """The R-multiple-metric sibling of run_deep_backtest — same 80/20
    time split, same same-sign-across-splits discipline, same accumulating
    trial log (_append_trial, so load_experiment_trials' own BH-correction
    and ticker_behavior.py both pick this up with zero changes — they
    only ever read p_value_train/mean_return_train/mean_return_holdout/
    holdout_verdict/same_sign, never caring what UNITS mean_return is in).

    The significance test itself has to be different: run_event_study's
    own permutation shuffles which events get which DIRECTION LABEL
    across the whole sample, which only makes sense when direction is
    independent of the outcome computation — for a fixed-R:R exit, the
    stop/target placement itself depends on direction, so relabeling
    would silently paste one touch's real outcome onto a different
    touch's own entry price/ATR. Instead, events_df carries BOTH r_real
    (the touch's own real-direction outcome) and r_flip (the SAME touch,
    same entry bar, mirrored stop/target) — the null is "for each touch
    independently, a coin flip decides whether you get r_real or r_flip,"
    which tests the right thing (does calling the ACTUAL direction beat
    chance on these exact entries) without ever mixing outcomes across
    touches."""
    trial = {
        "logged_at": pd.Timestamp.now("UTC").isoformat(), "ticker": ticker, "tf_label": tf_label,
        "label": label, "settings": settings, "n_events": len(events_df),
    }
    if len(events_df) < 30:
        trial["verdict"] = "INSUFFICIENT_DATA"
        _append_trial(trial)
        return trial

    events_df = events_df.sort_values("entry_time").reset_index(drop=True)
    cutoff_pos = int(len(events_df) * train_fraction)
    train = events_df.iloc[:cutoff_pos]
    holdout = events_df.iloc[cutoff_pos:]
    rng = np.random.default_rng(seed)

    def _score(split):
        n = len(split)
        if n == 0:
            return {"n": 0, "mean": None, "p": None}
        real = split["r_real"].to_numpy()
        flip = split["r_flip"].to_numpy()
        observed = real.mean()
        null = np.empty(n_permutations)
        for i in range(n_permutations):
            coin = rng.integers(0, 2, size=n).astype(bool)
            null[i] = np.where(coin, real, flip).mean()
        p = float((null >= observed).mean())
        return {"n": n, "mean": float(observed), "p": p}

    train_score = _score(train)
    holdout_score = _score(holdout)
    if train_score["n"] == 0:
        trial["verdict"] = "INSUFFICIENT_DATA"
        _append_trial(trial)
        return trial

    mean_train, mean_holdout = train_score["mean"], holdout_score["mean"]
    trial.update({
        "n_train": train_score["n"], "n_holdout": holdout_score["n"],
        "mean_return_train": mean_train, "p_value_train": train_score["p"],
        "mean_return_holdout": mean_holdout,
        "holdout_verdict": "PASSED" if (holdout_score["p"] is not None and holdout_score["p"] < 0.05
                                         and mean_holdout is not None and mean_holdout > 0) else "FAILED",
        "same_sign": bool(mean_train is not None and mean_holdout is not None
                           and mean_train > 0 and mean_holdout > 0),
        "verdict": "SCORED",
    })
    _append_trial(trial)
    return trial


def build_clean_expansion_fixed_rr_events(df, min_streak=3, retr_window_bars=40, rr_multiple=0.5,
                                           atr_mult_stop=1.0, max_bars=40, min_body_ratio=DISPLACEMENT_MIN_BODY_RATIO):
    """clean_expansion_retracement's own entries, exited with a fixed
    stop/target (see _simulate_fixed_rr) instead of a time-based hold —
    the high-win-rate sibling of build_clean_expansion_retracement_events,
    same entries, different question."""
    touches = _clean_expansion_touches(df, min_streak, retr_window_bars, min_body_ratio=min_body_ratio)
    return _simulate_fixed_rr(df, touches, rr_multiple, atr_mult_stop, max_bars)


def build_clean_expansion_liquidity_fixed_rr_events(df, min_streak=3, retr_window_bars=40, sweep_window_bars=10,
                                                     rr_multiple=0.5, atr_mult_stop=1.0, max_bars=40,
                                                     min_body_ratio=DISPLACEMENT_MIN_BODY_RATIO):
    """clean_expansion_liquidity's own entries, fixed-R:R exit — see
    build_clean_expansion_fixed_rr_events's own docstring."""
    touches = _clean_expansion_liquidity_touches(df, min_streak, retr_window_bars, sweep_window_bars,
                                                  min_body_ratio=min_body_ratio)
    return _simulate_fixed_rr(df, touches, rr_multiple, atr_mult_stop, max_bars)


def build_clean_retracement_resumption_fixed_rr_events(df, min_streak=3, retr_window_bars=40, rr_multiple=0.5,
                                                        atr_mult_stop=1.0, max_bars=40,
                                                        min_body_ratio=DISPLACEMENT_MIN_BODY_RATIO):
    """clean_retracement_resumption's own entries, fixed-R:R exit — see
    build_clean_expansion_fixed_rr_events's own docstring."""
    touches = _clean_retracement_resumption_touches(df, min_streak, retr_window_bars, min_body_ratio=min_body_ratio)
    return _simulate_fixed_rr(df, touches, rr_multiple, atr_mult_stop, max_bars)
