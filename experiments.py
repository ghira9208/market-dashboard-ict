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

from detectors import detect_breaker_blocks, detect_fvgs, detect_ifvgs, detect_order_blocks
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
    nothing's been run yet."""
    if not os.path.exists(_TRIALS_PATH):
        return pd.DataFrame()
    trials = []
    with open(_TRIALS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                trials.append(json.loads(line))
    if not trials:
        return pd.DataFrame()

    df = pd.DataFrame(trials)
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
    n = len(df)
    close = (df["Close"] if "Close" in df else df["close"])
    close_arr = close.to_numpy()

    for _, trial in matches.iterrows():
        settings = trial["settings"]
        touches = _qualifying_touches(df, settings["detector"], settings["min_volatility_pctile"],
                                       settings["reaction_window"], settings["reaction_mult"])
        if not touches:
            continue
        latest = max(touches, key=lambda t: t["confirm_pos"])
        bars_since = n - 1 - latest["confirm_pos"]
        if bars_since > settings["forward_bars"]:
            continue  # already past this trial's own hold length -- no longer live
        direction = latest["direction"]
        # Entry is confirm_pos, not touch_pos — see build_experiment_events'
        # own comment: you can only realistically act once the reaction has
        # actually confirmed, and this must match how the trial itself
        # measured its own p-value/returns or the live pick would be
        # showing a different (and again look-ahead-biased) entry point
        # than what was actually validated.
        entry_price = float(close_arr[latest["confirm_pos"]])
        stop_price = latest["zone_bottom"] if direction == "bullish" else latest["zone_top"]
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
        }
    return None
