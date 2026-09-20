"""
Markets — the ICT terminal scoped to highly liquid markets only (forex
majors, major indices, commodities), not a universal ticker search. See
crypto_app.py for the separate Bitcoin/crypto counterpart; the two share
every underlying module (detectors.py, theme.py, data.py, recommender.py,
ict_chart) and differ only in which curated symbol lists their own ticker
picker shows. Independent of market-dashboard — see README.md for why.
"""

import concurrent.futures
import json
import math
import os
import threading
import time
from datetime import time as dtime

import altair as alt
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

import backtest_ui
import experiments
import news
import survivors_ui
import theme
from data import get_latest_bars, get_yf_ohlcv, is_ticker_alive, resample_ohlc, warm_in_background
from detectors import (
    current_dealing_range,
    detect_breaker_blocks,
    detect_equal_levels,
    detect_fvgs,
    detect_ifvgs,
    detect_liquidity_levels,
    detect_liquidity_reactions,
    detect_naked_pocs,
    detect_order_blocks,
    detect_poor_highs_lows,
    detect_structure_breaks,
    detect_swings,
    describe_streak_conditions,
    find_clean_respect_streaks,
    historical_zone_scanner,
    merge_zone_engines,
    poc_migration,
    recent_zone_tracker,
)
from ict_chart import ict_chart
from indicators import atr, bollinger_bands, ema, macd, rsi, volume_profile
from recommender import (
    CONTEXT_WEIGHT,
    EVENT_TYPE_LABELS,
    EXIT_EVENT_LABELS,
    INDICATOR_SPECS,
    MA_FVG_PERIODS,
    RULE_DETECTOR_LABELS,
    TF_PAIRS,
    _confluence_score,
    backtest_custom_rule,
    best_trade_now,
    fvg_event_win_rate,
    fvg_run_up_stats,
    hold_resolved_trade,
    liquidity_event_win_rate,
    ma_fvg_event_win_rate,
    ma_fvg_starts,
    order_block_event_win_rate,
    point_level_indicator_matches,
    rank_levels_by_visit_probability,
    rebalance_chain,
    resolve_rule,
    rule_describe,
    scan_timeframes,
    scan_watchlist,
    zone_indicator_matches,
)
from research.data_loader import INTERVAL_MAX_PERIOD, load_history
from research.setups import EVENT_TYPE_TO_HYPOTHESIS, validation_badge


st.set_page_config(page_title="Markets", layout="wide", initial_sidebar_state="collapsed")
theme.inject(chart_layout=True)


def tiny(text):
    """Genuinely optional text — small and muted, never competing with real content."""
    st.markdown(f'<div class="tiny-note">{text}</div>', unsafe_allow_html=True)


def _hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _price_decimals(price):
    """How many decimals a price needs to stay meaningful, based on its
    own magnitude rather than a fixed assumption — a 2-decimal default
    (fine for a $65,000 BTC print) rounds a $1.35 forex pair's real
    movement away entirely (1.3465 -> 1.3546 all reads as "1.35"), and
    would show "0.00" outright for a fraction-of-a-cent altcoin. Same
    log-scale reasoning as the chart's own JS-side computePriceDecimals
    (ict_chart/frontend/index.html) — kept independent rather than shared
    since one's Python and one's JS, but the formula must stay identical:
    5 significant digits above the decimal point's own order of
    magnitude, clamped to [2, 10]."""
    if price is None or price <= 0:
        return 2
    decimals = 5 - math.floor(math.log10(price))
    return max(2, min(10, decimals))


PRICE_PROJECTION_BARS = 20


def _compute_price_projection(df, c_col, n_bars=PRICE_PROJECTION_BARS, vol_lookback=100, ema_period=20, slope_lookback=5):
    """A volatility cone (±1σ/±2σ, sqrt-time scaling off this instrument's
    own trailing realized volatility — the exact same magnitude-only model
    this session's own vol-harvest experiment used) plus a labeled trend-
    extrapolation center line (the current EMA's own recent slope,
    projected forward linearly). Direct request, with an explicit
    constraint carried over from a full day of rigorous backtesting: none
    of the six directional/magnitude strategies tried in the Experiments
    tab this session survived correction (see experiments.py's own
    docstring) — so this overlay is deliberately NOT presented as a
    prediction anywhere it renders (see the render call site's own
    "NOT a validated prediction" label on the center line), just an
    honest "here's what current trend + volatility implies," nothing
    more. Returns None when there isn't enough history yet to compute a
    trailing volatility estimate (an honest empty state, not a fabricated
    cone) — an early int/1h fetch right after a ticker/timeframe switch
    can legitimately be this thin.

    Returns {"center", "upper1", "lower1", "upper2", "lower2"}, each a
    list of n_bars floats (one per future bar, nearest first)."""
    close = df[c_col]
    n = len(close)
    if n < vol_lookback + slope_lookback + 2:
        return None
    returns = close.pct_change().dropna()
    sigma = float(returns.tail(vol_lookback).std())
    if not math.isfinite(sigma) or sigma <= 0:
        return None

    ema_series = ema(close, ema_period)
    if len(ema_series) < slope_lookback + 1 or pd.isna(ema_series.iloc[-1]) or pd.isna(ema_series.iloc[-1 - slope_lookback]):
        slope = 0.0
    else:
        slope = float(ema_series.iloc[-1] - ema_series.iloc[-1 - slope_lookback]) / slope_lookback

    current_price = float(close.iloc[-1])
    result = {"center": [], "upper1": [], "lower1": [], "upper2": [], "lower2": []}
    for i in range(1, n_bars + 1):
        center_i = current_price + slope * i
        sigma_i = sigma * math.sqrt(i) * current_price
        result["center"].append(center_i)
        result["upper1"].append(center_i + sigma_i)
        result["lower1"].append(center_i - sigma_i)
        result["upper2"].append(center_i + 2 * sigma_i)
        result["lower2"].append(center_i - 2 * sigma_i)
    return result


def _win_rate_label(wr, precision=0):
    """wr: a _historical_win_rates value, (rate, n, sufficient), or None.
    A thin sample shows as "low data (n=X)" rather than disappearing —
    the label always renders when there's SOME data, it just says
    plainly when there isn't enough to trust the percentage. None (zero
    qualifying events at all) still returns None — nothing to show any
    number, low-data or otherwise, for."""
    if not wr:
        return None
    rate, n, sufficient = wr
    return f"{rate*100:.{precision}f}% (n={n})" if sufficient else f"low data (n={n})"


def _nearest_by_price(zones, current_price, n, top_key="top", bottom_key="bottom"):
    """The `n` zones closest to current_price on each side (above AND
    below) — "active" as in "immediately relevant to where price actually
    is right now," not "most recently detected in time," which on a chart
    with a lot of history loaded can mean a zone nowhere near current
    price. A zone straddling current price (its own top/bottom span across
    it — rare, but the far edge of a zone mid-fill can do this) always
    shows, since price sitting inside it is the most obviously live case
    there is. Returns the picks in their ORIGINAL relative order (not
    sorted by distance), so legend counts/detail-table ordering downstream
    don't need to know this reordered anything."""
    above, below, straddle = [], [], []
    for z in zones:
        top, bottom = z[top_key], z[bottom_key]
        if bottom >= current_price:
            above.append((bottom - current_price, z))
        elif top <= current_price:
            below.append((current_price - top, z))
        else:
            straddle.append(z)
    above.sort(key=lambda pair: pair[0])
    below.sort(key=lambda pair: pair[0])
    picked_ids = {id(z) for _, z in above[:n]} | {id(z) for _, z in below[:n]} | {id(z) for z in straddle}
    return [z for z in zones if id(z) in picked_ids]


# Bridges the exact gap flagged in this project's own known-limitations
# list: a drawn zone meant "matches the ICT geometric definition," nothing
# about whether that definition has actually worked on THIS ticker. Reuses
# Research Lab's own event extractors (research/events.py) rather than
# reimplementing FVG/OB backtesting a second time — same detection logic
# the chart itself draws, same retracement-entry timing discipline.
#
# Deliberately NOT the full permutation test research/evidence.py runs for
# Research Lab/Edge Lab — that's 5000 random-direction trials, built for a
# considered, one-at-a-time verdict, not something to re-run on every chart
# rerun for every zone. This is a plain raw-return win rate: enough for a
# quick on-chart reference number, not a claim of statistical significance.
# Long TTL (1h) because it's backward-looking history, not live price — it
# has no reason to change minute to minute the way the chart itself does.
_WIN_RATE_MIN_EVENTS = 20


@st.cache_data(ttl=3600, show_spinner=False)
def _historical_win_rates(ticker, tf_label, layer, provider, exit_types=tuple(EXIT_EVENT_LABELS.keys()),
                           news_blackout=None):
    """{"bullish": (win_rate, n, sufficient), "bearish": (...)} for this
    ticker's own available history at tf_label. `sufficient` is False when
    fewer than _WIN_RATE_MIN_EVENTS qualifying events were found — the win
    rate is still returned (not hidden), callers label it "low data"
    instead of a percentage rather than omitting it outright, so a thin
    sample reads as a visible state, not silent absence. A missing
    direction key means ZERO qualifying events at all (nothing to show any
    number for, low-data or otherwise). Uses the LONGEST history the
    provider will serve at this interval (research/data_loader.
    INTERVAL_MAX_PERIOD), not the live chart's own shorter TIMEFRAMES
    period — that period exists to keep the chart responsive, not to have
    enough samples for a stat worth showing.

    Event-driven, not a fixed hold length — see recommender.py's own
    module comment on fvg_event_win_rate/order_block_event_win_rate/
    ma_fvg_event_win_rate for why this is a SEPARATE methodology from
    research/events.py's fixed-bar one (which still backs Edge Lab's own
    cached validated-edge results, untouched here). `exit_types` (a subset
    of EXIT_EVENT_LABELS' keys) controls which pattern forming next closes
    the hold — user-controlled via the Settings popover's own "Win rate"
    section. Raw return, no spread/cost deduction — this is "did price
    move in the expected direction over the hold," not a claim about what
    a real fill would have net after cost.

    news_blackout: None (default, unchanged behavior) or a (minutes_before,
    minutes_after) pair - a plain hashable tuple, not the mask itself, so
    st.cache_data's own argument-hashing keys different settings into
    different cache entries correctly; the mask is computed below, after
    `df` is loaded, from `ticker` + this pair (see news.blackout_mask)."""
    conf = TIMEFRAMES[tf_label]
    fetch_interval = conf["fetch_interval"]
    period = INTERVAL_MAX_PERIOD.get(fetch_interval, "60d")
    try:
        df = load_history(ticker, period, fetch_interval, provider=provider)
    except Exception:
        return {}
    if conf["resample"]:
        df = resample_ohlc(df, conf["resample"])
    if len(df) < 50:
        return {}

    nb_mask = news.blackout_mask(df.index, ticker, news_blackout[0], news_blackout[1]) if news_blackout else None
    fn = {"FVG": fvg_event_win_rate, "Order Block": order_block_event_win_rate,
          "MA+FVG": ma_fvg_event_win_rate, "Liquidity": liquidity_event_win_rate}[layer]
    return fn(df, exit_types=exit_types, min_events=_WIN_RATE_MIN_EVENTS, news_blackout_mask=nb_mask)


_EPOCH = pd.Timestamp("1970-01-01")


def _ny_fake_utc_seconds(ts):
    """lightweight-charts has no timezone setting — it always renders a
    UTCTimestamp's UTC wall-clock digits. The standard workaround: convert
    to the timezone the user's own theme.render_top_bar() selector has
    picked (theme.get_display_tz(), America/New_York by default — the
    original, still-matching-ICT-kill-zone-convention choice before that
    selector existed), strip the tz label, and encode THOSE digits as if
    they were UTC. Name kept as "_ny_..." since America/New_York is still
    the default/most common case and every call site already uses this
    name — only the body changed, not what callers need to know."""
    disp = ts.tz_convert(theme.get_display_tz()).tz_localize(None)
    return int((disp - _EPOCH).total_seconds())


def _ny_fake_utc_seconds_vec(idx):
    """Same conversion as _ny_fake_utc_seconds, vectorized over a whole
    DatetimeIndex instead of one Timestamp's own .tz_convert/.tz_localize
    per call in a Python-level loop. Confirmed directly as a real, live
    performance bug, not a style nit: a scalar per-timestamp loop pays
    pandas/pytz's own per-call DST-lookup overhead (tz_convert_from_utc_single
    -> Localizer.__new__ -> get_dst_info -> tz_cache_key) on EVERY bar —
    caught live via lldb thread backtrace during a genuine multi-minute
    stall, both times sampled sitting inside exactly this call chain. A
    17,322-row backfill fetch (1h's own backfill tier: 730 days of native
    60m bars) measured 0.30s via the scalar list-comprehension form this
    replaces vs. 0.002s here — 148x — and that cost was being paid on
    EVERY fragment auto-refresh tick (run_every=1 on an intraday
    timeframe), not once, which is what turned a sub-second cost into a
    self-reinforcing pile-up: each tick's own render taking longer than
    the 1s interval between ticks means the NEXT tick fires before the
    current one finishes, so overlapping/queued reruns accumulate rather
    than settle. Returns a plain list of ints, same shape
    [_ny_fake_utc_seconds(ts) for ts in idx] already produced everywhere
    this replaces it, so no caller downstream of the list itself needs to
    change."""
    disp_idx = idx.tz_convert(theme.get_display_tz()).tz_localize(None)
    return ((disp_idx - _EPOCH) // pd.Timedelta(seconds=1)).tolist()


def _series_to_points(series):
    """A pandas Series (e.g. an EMA/RSI/MACD line) -> the {"time","value"}
    point list ict_chart's own indicators= param expects, dropping NaNs
    (the warm-up period every rolling/EWM indicator has at its start).
    Same exact shape [{"time": _ny_fake_utc_seconds(ts), "value": float(v)}
    for ts, v in series.items() if pd.notna(v)] already produced
    everywhere this replaces it, just vectorized — that per-point form is
    the SAME scalar-timezone-conversion bug _ny_fake_utc_seconds_vec's own
    docstring already documents fixing for the candle axis, just never
    caught here too since indicator lines are a separate code path.
    Confirmed directly on a real 21k-row 5m dataset: ~0.3s per indicator
    series in that form — with MA+RSI+MACD+Bollinger all enabled at once
    (4 checkboxes, 8 total lines), that added up to ~2.2s of pure Python
    overhead on EVERY render. This does the identical output in ~6ms per
    series."""
    mask = series.notna().to_numpy()
    secs = _ny_fake_utc_seconds_vec(series.index[mask])
    return [{"time": t, "value": float(v)} for t, v in zip(secs, series.to_numpy()[mask])]


def _hist_to_points(series, pos_color, neg_color):
    """Same as _series_to_points, plus a per-point "color" field (MACD's
    histogram bars are colored by sign) — kept as its own function rather
    than an optional param so the common (no color) case stays a plain
    2-key dict, matching every other indicator line's own shape exactly."""
    mask = series.notna().to_numpy()
    secs = _ny_fake_utc_seconds_vec(series.index[mask])
    vals = series.to_numpy()[mask]
    return [{"time": t, "value": float(v), "color": pos_color if v >= 0 else neg_color}
            for t, v in zip(secs, vals)]


def _line_stop_time(df, start_ts, price, direction=None, exclude=None):
    """First candle strictly after start_ts that invalidates a level at
    `price`. With `direction` given ('above' for a resistance-type level
    like an equal high, 'below' for a support-type level like an equal
    low), that means a decisive CLOSE beyond it — the same close-based
    break rule detect_structure_breaks uses for BOS/CHoCH, not a wick that
    merely grazes the level without actually confirming a break through it.
    Without `direction` (Liquidity's own use), any wick TOUCH at all counts
    — harmless there specifically because detect_liquidity_levels already
    only ever returns levels no later wick has touched, so this branch
    never actually fires for them; kept non-directional rather than forced
    to pick a side for a caller that never reaches it.

    `exclude` skips a given set of candle timestamps when checking for a
    break — Equal Highs/Lows passes its own cluster's member times here:
    those candles sit right at the shared price BY CONSTRUCTION and can
    occasionally close a hair beyond the group's averaged price, which
    would otherwise make a freshly-formed cluster appear to break itself
    the instant its own second point forms.

    Falls back to the last loaded candle when never invalidated (still
    live)."""
    if direction is not None:
        c = df["Close"] if "Close" in df else df["close"]
        after = (df.index > start_ts) & ((c > price) if direction == "above" else (c < price))
    else:
        h = df["High"] if "High" in df else df["high"]
        l = df["Low"] if "Low" in df else df["low"]
        after = (df.index > start_ts) & (l <= price) & (h >= price)
    if exclude:
        after &= ~df.index.isin(exclude)
    hit = df.index[after]
    return hit[0] if len(hit) else df.index[-1]


def _to_ghost_candles(gdf, opacity=0.16):
    """An OHLCV frame → the ghost-candle primitive's box list (see
    GhostCandlePrimitive in the frontend). Each candle is its own drawn
    primitive with real start/end timestamps rather than being tied to the
    main series' own bar spacing — what lets the Overlay timeframe
    picker's own different-timeframe candles sit correctly on the main
    chart's shared time axis, laid over the SAME period the real series
    already covers. Not used for backfilling history further back than the
    series' own first bar — lightweight-charts won't compute a coordinate
    for (or scroll to) a time before that, so a primitive positioned there
    would just never draw; see the backfill block in _render_chart, which
    prepends real series bars instead."""
    if gdf.empty:
        return []
    o_col, h_col, l_col, c_col = (("Open", "High", "Low", "Close") if "Close" in gdf
                                   else ("open", "high", "low", "close"))
    axis = _ny_fake_utc_seconds_vec(gdf.index)
    step = (axis[-1] - axis[-2]) if len(axis) > 1 else 1
    # .to_numpy() once up front instead of gdf[col].iloc[i] per cell per row —
    # pandas' scalar .iloc has real per-call overhead (index alignment, block
    # lookup); confirmed directly (see the identical pattern's own fix in
    # _render_chart's own bars list) this is ~100x+ faster at the row counts
    # this project's own large timeframes (e.g. 5m, 16k+ rows) actually hit.
    _o, _h, _l, _c = (gdf[o_col].to_numpy(), gdf[h_col].to_numpy(),
                       gdf[l_col].to_numpy(), gdf[c_col].to_numpy())
    out = []
    for i in range(len(gdf)):
        is_bull = _c[i] >= _o[i]
        out.append({
            "t0": axis[i], "t1": axis[i + 1] if i + 1 < len(axis) else axis[i] + step,
            "open": float(_o[i]), "high": float(_h[i]),
            "low": float(_l[i]), "close": float(_c[i]),
            "color": _hex_to_rgba(theme.NEON_CYAN if is_bull else theme.NEON_MAGENTA, opacity),
        })
    return out


# "Exchange" here is a coarse venue tag for filtering, not a precise listing
# record — ETFs get lumped under "ETF" regardless of which venue they
# technically print on (NYSE Arca vs Cboe BZX isn't a distinction anyone
# filtering this list actually cares about); individual stocks get their
# real primary listing (NASDAQ vs NYSE).
#
# ETFs stay hand-picked on purpose — Twelve Data's own /etf reference list
# runs to 11,000+ US-listed funds (mostly obscure single-security or
# leveraged/inverse products nobody searching this dashboard is looking
# for), so pulling it wholesale would trade "clean, compact" for noise. The
# 40-ish here are the ones actually liquid enough to matter for ICT-style
# trading.
# No comprehensive free reference-data source exists for any of these three
# the way Twelve Data covers stocks or Binance covers crypto — hand-curated,
# same reasoning either way (a search box that never returns an obscure-but-
# real symbol beats one that tries to be exhaustive and ends up mostly
# noise). All confirmed live via Yahoo's own ticker conventions (^ indices,
# =F futures, =X forex) before being added here. This app is scoped to
# highly liquid markets only — Markets, not a universal ticker search — so
# Stocks/ETF/Crypto (its own separate dashboard, see crypto_app.py) aren't
# here at all.
CURATED_INDICES = {
    "^GSPC": ("S&P 500", "Indices"), "^DJI": ("Dow Jones Industrial Average", "Indices"),
    "^IXIC": ("Nasdaq Composite", "Indices"), "^RUT": ("Russell 2000", "Indices"),
    "^VIX": ("CBOE Volatility Index", "Indices"), "^FTSE": ("FTSE 100", "Indices"),
    "^N225": ("Nikkei 225", "Indices"), "^GDAXI": ("DAX", "Indices"),
}
# Deliberately just the three commodities themselves (via their front-month
# futures contract, the standard way to chart spot-adjacent commodity price
# on Yahoo) — not the full index-futures chain, which is its own category
# below for a real reason, not a duplicate of it.
CURATED_COMMODITIES = {
    "GC=F": ("Gold Futures", "Commodities"), "SI=F": ("Silver Futures", "Commodities"),
    "CL=F": ("Crude Oil Futures", "Commodities"),
}
# Index futures (their own category, not folded into Commodities) — the
# direct futures counterpart of four of CURATED_INDICES's own cash tickers
# (NQ<->^IXIC, ES<->^GSPC, YM<->^DJI, RTY<->^RUT). Genuinely NOT a duplicate
# of the cash index despite tracking the same underlying: Yahoo's cash
# index feed only carries the regular 9:30-16:00 ET session, while these
# trade nearly 24/5 on CME Globex — confirmed directly, 1m bars for all
# four span 00:00 ET onward, not just cash-session hours. That's the whole
# point of adding them: real overnight/pre-market price action a cash
# index never shows at all, at 1-minute resolution (the finest free data
# that exists for a CME-listed future — true tick data is a licensed
# product, not available here or anywhere for free).
CURATED_INDEX_FUTURES = {
    "NQ=F": ("Nasdaq-100 Futures", "Futures"), "ES=F": ("S&P 500 Futures", "Futures"),
    "YM=F": ("Dow Jones Futures", "Futures"), "RTY=F": ("Russell 2000 Futures", "Futures"),
}
CURATED_FOREX = {
    "EURUSD=X": ("Euro / US Dollar", "Forex"), "GBPUSD=X": ("British Pound / US Dollar", "Forex"),
    "USDJPY=X": ("US Dollar / Japanese Yen", "Forex"), "USDCHF=X": ("US Dollar / Swiss Franc", "Forex"),
    "AUDUSD=X": ("Australian Dollar / US Dollar", "Forex"), "USDCAD=X": ("US Dollar / Canadian Dollar", "Forex"),
    "NZDUSD=X": ("New Zealand Dollar / US Dollar", "Forex"), "EURJPY=X": ("Euro / Japanese Yen", "Forex"),
    "GBPJPY=X": ("British Pound / Japanese Yen", "Forex"), "EURGBP=X": ("Euro / British Pound", "Forex"),
}


@st.cache_data(ttl=86400)
def _build_ticker_info():
    """Just the curated Forex/Indices/Commodities lists, merged — no live
    stock/crypto universe fetch here, this dashboard never shows either
    (see crypto_app.py for the crypto-only counterpart)."""
    info = dict(CURATED_INDICES)
    info.update(CURATED_COMMODITIES)
    info.update(CURATED_INDEX_FUTURES)
    info.update(CURATED_FOREX)
    return info


TICKER_INFO = _build_ticker_info()
TICKER_NAMES = {t: info[0] for t, info in TICKER_INFO.items()}
TICKER_UNIVERSE = list(TICKER_INFO.keys())

# The ticker menu's top-level category buttons. Order here is the order
# the buttons render in.
TICKER_CATEGORIES = {
    "Forex": ["Forex"], "Indices": ["Indices"], "Commodities": ["Commodities"], "Futures": ["Futures"],
}

# Display label <-> internal id for data.py's provider chain (see its own
# _all_providers). "Auto" is Yahoo-first-with-fallback, today's default
# behavior; picking one of the other four FORCES that single provider, with
# no fallback if it fails — a way to actually audit/compare a specific
# source rather than just hoping the right one served a given render.

# No "Binance" entry here (unlike crypto_app.py's own copy of this dict) —
# this dashboard never charts a -USD ticker, and _provider_chain() only
# ever adds Binance to the fallback chain for crypto tickers, so picking
# it here would force an empty chain and silently show "no data" for
# every ticker on this page.
DATA_SOURCES = {"Auto": "auto", "Yahoo Finance": "yahoo",
                 "Twelve Data": "twelvedata", "Tiingo": "tiingo"}
DATA_SOURCE_LABELS = {v: k for k, v in DATA_SOURCES.items()}


def _is_open(ticker):
    """24/7 for crypto; NYSE/Nasdaq hours (Mon-Fri 9:30-16:00 ET) for
    everything else — a simplification (no holiday calendar), but enough to
    flag "don't bother, it's closed" at a glance."""
    if ticker.endswith("-USD"):
        return True
    now = pd.Timestamp.now(tz="America/New_York")
    return now.weekday() < 5 and dtime(9, 30) <= now.time() < dtime(16, 0)


def _ticker_label(t):
    name = TICKER_NAMES.get(t)
    label = f"{t} — {name}" if name else t
    return label if _is_open(t) else f"{label}  ·  closed"

# Core interval bar, minutes -> hours -> day/week/month/year. Yahoo natively
# supports 1m/5m/15m/30m/60m/1d/1wk/1mo/3mo; 4h/1Y aren't native, so those are
# built by resampling the nearest native interval rather than silently
# swapping in a different timeframe than what was actually selected.
TIMEFRAMES = {
    "1m":  {"fetch_interval": "1m",  "period": "7d",   "resample": None, "sub_hour": True},
    "5m":  {"fetch_interval": "5m",  "period": "60d",  "resample": None, "sub_hour": True},
    "15m": {"fetch_interval": "15m", "period": "60d",  "resample": None, "sub_hour": True},
    "30m": {"fetch_interval": "30m", "period": "60d",  "resample": None, "sub_hour": True},
    # 1h used to share 4h's 730d fetch un-resampled — ~17,300 raw bars, far
    # more than any other timeframe ever sends the chart (5m/15m/30m cap at
    # 60d; 4h resamples the same 730d fetch down to a quarter as many bars).
    # Confirmed directly: that bar count reliably left the candlestick series
    # unable to paint at all (blank chart, "Value is null" thrown deep in
    # lightweight-charts' own renderer) — reproduced on a fresh page load
    # with zero app-side customization involved, so it's a real limit in the
    # library at this data volume, not a bug in anything this app does with
    # it. 180d keeps a solid ~6 months of native hourly history — in the
    # same ballpark as what 4h already shows post-resample — well clear of
    # whatever threshold triggers the crash.
    "1h":  {"fetch_interval": "60m", "period": "180d", "resample": None, "sub_hour": False},
    "4h":  {"fetch_interval": "60m", "period": "730d", "resample": "4h", "sub_hour": False},
    "1D":  {"fetch_interval": "1d",  "period": "2y",   "resample": None, "sub_hour": False},
    "1W":  {"fetch_interval": "1wk", "period": "10y",  "resample": None, "sub_hour": False},
    "1M":  {"fetch_interval": "1mo", "period": "max",  "resample": None, "sub_hour": False},
    "1Y":  {"fetch_interval": "3mo", "period": "max",  "resample": "1YE", "sub_hour": False},
}


def _period_days(period):
    """Rough real-world day count for a TIMEFRAMES period string ("7d",
    "2y", "max"), so periods can be compared across units — used to find a
    genuinely longer-history backfill timeframe rather than just the next
    entry in the dict (several adjacent tiers share the same period: 5m/
    15m/30m are all 60d, 1M/1Y are both "max", so "next" alone can mean
    zero extra history)."""
    if period == "max":
        return float("inf")
    n = int(period[:-1])
    return n * 365 if period[-1] == "y" else n


_TF_ORDER = list(TIMEFRAMES.keys())
_TF_PERIOD_DAYS = {name: _period_days(conf["period"]) for name, conf in TIMEFRAMES.items()}
# Real seconds per candle for each timeframe — used to tile deep-history
# backfill candles (see the backfill block in _render_chart): the chart
# spaces every bar by uniform INDEX, not by real elapsed time, so a single
# coarse candle sitting next to native bars needs to occupy as many
# native-duration slots as it actually spans, or the time axis's pace
# breaks right at that boundary. 1M/1Y use the calendar-average month/year
# (30.44d / 365.24d) since actual months vary.
_TF_BAR_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400,
    "1D": 86400, "1W": 604800, "1M": 2629746, "1Y": 31556952,
}
# Hard ceiling on the SUM of tiled slots across the whole backfill range
# (see the two-pass proportional-scaling block in _render_chart) — an
# active LAYER can be set to a timeframe far coarser than the main chart
# itself (e.g. a "1M" layer's own dataframe backfilling a "1m" main
# chart), and the naive slot count is the ratio of the two durations,
# unbounded: 1M into 1m alone is ~43,800 slots for ONE row, times up to
# 1500 rows — confirmed directly, this sent a 385MB payload and blew past
# Streamlit's message-size limit outright. Each tiled slot is also its own
# individually-drawn canvas primitive (see GhostCandlePrimitive) with no
# batching, so even after that payload-size crash was capped, the
# resulting ~75,000 of them redrawn every frame during pan/zoom pegged a
# real machine's CPU hard enough to make the tab unresponsive — confirmed
# directly. 5000 stays comfortably clear of that (confirmed a real
# machine handles it fine — the working ^IXIC/1W case needed ~1500-2700
# total, no issue) while still capping the worst case regardless of how
# extreme the timeframe mismatch is.
#
# Lowered from 5000 to 1500 after a SEPARATE finding, once the batched-
# primitive + viewport-culling fix existed (see GhostCandleBatchPaneView's
# own comment in ict_chart/frontend/index.html): that fix only helps once
# the user has manually zoomed in — a freshly loaded/reloaded chart calls
# fitContentWhenReady(), which fits the WHOLE series including every
# backfill tile, so the "visible range" filter is a no-op right at load
# time, the single most common moment. That same profiling measured 4,500
# tiles (this cap's old ceiling, and confirmed live via __ictDebug on a
# real multi-layer session) at 143ms p99 frame time — every sub-30fps
# frame eliminated only after dropping to 1,479 (21.5ms p99). 1500 lands
# at that already-measured-smooth point; backfill DEPTH is unaffected
# (governed by bfdf.tail(1500) above, a different cap) — only the
# staircase's own granularity coarsens somewhat at extreme zoom-out.
_MAX_TOTAL_TILES = 1500


def _backfill_timeframe(tf_label):
    """The nearest timeframe further up the ladder whose fetched period
    actually covers MORE real history than `tf_label`'s own — for
    splicing older, coarser candles in where the main timeframe's own data
    runs out. None once nothing further up covers more (1Y past 1M, or
    already-"max" 1M/1Y themselves)."""
    cur_days = _TF_PERIOD_DAYS[tf_label]
    for t in _TF_ORDER[_TF_ORDER.index(tf_label) + 1:]:
        if _TF_PERIOD_DAYS[t] > cur_days:
            return t
    return None


ICT_LAYERS = ["FVG", "IFVG", "Order Blocks", "Breaker Block", "Swing Points", "Equal Highs/Lows",
              "Market Structure", "Premium/Discount", "Liquidity", "Naked POC", "Poor High/Low",
              "Price Projection"]
# All off for a brand-new session — a first-time load shouldn't dump every
# layer onto the chart at once. Returning users don't see this default at
# all: their own on/off choice per layer is persisted (see .last_state.json
# below) and restored ahead of it.
#
# Premium/Discount is the one exception, on by default even on a first
# load — it's the layer that answers "is this actually a good price to buy
# at all," which turned out to be easy to forget even exists: confirmed
# directly trying to plan a real trade off this chart with it off, the
# question it answers never came up because there was nothing on screen
# prompting it.
LAYER_DEFAULTS = {name: False for name in ICT_LAYERS}
LAYER_DEFAULTS["Premium/Discount"] = True

# How many candles late each layer's own read typically is, for the same
# reason every other hover-help in this app spells out real numbers
# instead of vague reassurance — confirmed directly by measuring
# detect_swings' own confirmed_pos - pos gap on real data (DOT-USD, 5m/
# 1h/4h all landed within the same range): median 2 candles, 90% confirm
# within 4, occasionally into double digits on a slow, grinding reversal.
# It's ATR-relative, not a fixed bar count (see detect_swings' own
# docstring) — "typically" is doing real work in that number, not a
# rounding of something exact. FVG/Order Blocks are a different KIND of
# lag entirely: a fixed, ~1-candle structural delay (confirmed the
# instant their own defining candle closes), not a reversal to wait out.
# Naked POC/Poor High-Low are lagged by construction, not by detection —
# neither can exist before the session that produces them has fully
# closed.
_SWING_LAG = ("ATR-based, not a fixed bar count — a swing isn't confirmed until price reverses far "
              "enough away from it. Measured directly on real data: median 2 candles late, 90% confirm "
              "within 4, occasionally more on a slow, grinding reversal.")
LAYER_LAG_HELP = {
    "FVG": "Confirmed the instant its own 3rd candle closes — a fixed, ~1-candle structural delay, "
           "not a reversal to wait out.",
    "IFVG": "An FVG that gets fully broken through WITH a close beyond it, not just a wick — confirmed "
            "the instant that breaking candle closes, same fixed structural delay as plain FVG.",
    "Order Blocks": "Confirmed the instant the displacement candle that breaks through it closes — "
                     "same fixed, small structural delay as FVG, not reversal-based.",
    "Breaker Block": "An Order Block that gets fully broken through WITH a close beyond it, not just a "
                      "wick — confirmed the instant that breaking candle closes, same fixed structural "
                      "delay as plain Order Blocks.",
    "Swing Points": _SWING_LAG,
    "Equal Highs/Lows": f"Built on Swing Points, so it inherits the same lag. {_SWING_LAG}",
    "Market Structure": ("The break itself fires the instant a candle CLOSES beyond the level — but "
                          "that level is a confirmed Swing Point, so it isn't even a pending level to "
                          "break until Swing Points' own lag has passed. " + _SWING_LAG),
    "Premium/Discount": ("The box's own top/bottom are confirmed Swing Points, so it carries the same "
                          "lag before either boundary is even set. " + _SWING_LAG),
    "Liquidity": ("Resting levels are confirmed Swing Points too, so the same lag applies before a "
                  "high/low even becomes a tracked level. " + _SWING_LAG),
    "Naked POC": "Lagged by construction, not by detection — a session's own POC isn't final until "
                 "that full session closes, so this can only ever speak about YESTERDAY's session "
                 "at the earliest, never today's still-forming one.",
    "Poor High/Low": "Same as Naked POC — only knowable once the full session that produced it has "
                      "closed, so today's own high/low can't be judged poor or clean until today ends.",
    "Price Projection": "Not lagged — the opposite problem: this is a forward guess, not a confirmed "
                         "read of what already happened. The shaded band is how far price COULD move "
                         "(this instrument's own historical volatility, widening the further out you "
                         "look) — the dashed line is just the current trend extended in a straight line, "
                         "explicitly NOT a validated prediction. A full day of rigorous backtesting this "
                         "session found no statistically confirmed directional edge in any single-"
                         "indicator pattern tried (see the Experiments tab) — this overlay doesn't change "
                         "that, it just visualizes what the current numbers imply.",
}

# Kill zones apply only below 1h — a 1h+ candle either barely fits inside a
# 2-hour session window (leaving a nearly-empty, visually broken chart) or
# spans past it entirely, so filtering to one would do nothing meaningful.
INTRADAY_TFS = {"1m", "5m", "15m", "30m"}
# Each entry is a LIST of (start, end) ranges rather than a single one —
# most sessions are one contiguous window, but ETH (Electronic/Extended
# Trading Hours) isn't: it's pre-market AND after-hours, wrapping around
# RTH rather than sitting next to it. Modeling every entry as a list keeps
# the render loop below uniform instead of special-casing the one
# discontiguous session.
KILL_ZONES = {
    "NY AM (9:30–11:30 ET)": [(dtime(9, 30), dtime(11, 30))],
    "NY PM (13:30–16:00 ET)": [(dtime(13, 30), dtime(16, 0))],
    "London Open (02:00–05:00 ET)": [(dtime(2, 0), dtime(5, 0))],
    "Asian Session (19:00–21:00 ET)": [(dtime(19, 0), dtime(21, 0))],
    "RTH (9:30–16:00 ET)": [(dtime(9, 30), dtime(16, 0))],
    "ETH (pre + after hours)": [(dtime(4, 0), dtime(9, 30)), (dtime(16, 0), dtime(20, 0))],
}
# Multiple sessions can be highlighted at once now (see the Settings
# multiselect below) — each needs its own color so overlapping/adjacent
# bands stay distinguishable rather than reading as one blob.
# Deliberately NOT theme.NEON_CYAN/MAGENTA/GREEN/AMBER — those are already
# claimed by FVG (cyan/magenta), order blocks (green/amber), and liquidity.
# A low-opacity kill-zone band sharing a hue with a much more opaque FVG/OB
# box sitting in the same area is effectively invisible against it — confirmed
# directly (the NY PM band was present in the data, correctly positioned, but
# unreadable next to a bearish FVG using the same magenta). A distinct cool
# blue/violet family reads as its own "session marker" layer instead.
KILL_ZONE_COLORS = {
    "NY AM (9:30–11:30 ET)": "#60a5fa",
    "NY PM (13:30–16:00 ET)": "#c084fc",
    "London Open (02:00–05:00 ET)": "#38bdf8",
    "Asian Session (19:00–21:00 ET)": "#818cf8",
    "RTH (9:30–16:00 ET)": "#2dd4bf",
    "ETH (pre + after hours)": "#94a3b8",
}
# Two low-opacity fills in neighboring hues (this session's blue vs.
# another's violet) turned out to read as "the same color" at a glance —
# confirmed directly against the live app. A short text label drawn right
# on the band (see the frontend's RectangleRenderer) removes the ambiguity
# outright instead of chasing more color separation.
KILL_ZONE_LABELS = {
    "NY AM (9:30–11:30 ET)": "AM",
    "NY PM (13:30–16:00 ET)": "PM",
    "London Open (02:00–05:00 ET)": "LDN",
    "Asian Session (19:00–21:00 ET)": "ASIA",
    "RTH (9:30–16:00 ET)": "RTH",
    "ETH (pre + after hours)": "ETH",
}

# How far back (from the most recent loaded candle) auto-detected structure
# stays visible — "the next/current session and the previous 5 days," not
# the whole loaded history. Keeps the chart from turning into a wall of
# stacked FVGs/order blocks on anything with real history loaded, without
# throwing that history away — candles still go back as far as the
# timeframe's own period; only the overlays are recency-windowed.
OVERLAY_LOOKBACK_DAYS = 5

# How often the chart re-fetches/redraws on its own, with no click needed —
# matched to each timeframe's own candle duration so it wakes up right when
# a new candle would have closed, not on some arbitrary fixed clock. Daily+
# timeframes get no auto-refresh at all: the source data itself only turns
# over once a day, so polling would just re-run detection against the exact
# same bars for no reason — the definition of the wasteful recomputation
# this is meant to avoid.
CANDLE_SECONDS = {
    "1m": 10, "5m": 20, "15m": 30, "30m": 45, "1h": 60, "4h": 120,
    "1D": None, "1W": None, "1M": None, "1Y": None,
}

# The mini reference charts are for "what's the immediate context on this
# timeframe," not a full history browse — their timeframe is now user-
# adjustable (see CHART_IDS below), so every TIMEFRAMES key needs a count,
# not just the original 15m/30 and 4h/10 defaults. Smaller candles get more
# of them (a wider recent window covering comparable real time); larger
# candles get fewer (each one already covers a lot of ground). Raised
# across the board from the original values (1m 60, 5m 40, 15m 30, 30m 24,
# 1h 24, 4h 10, 1D 20, 1W 20, 1M 12, 1Y 10) — these panels now render at a
# real 380px height in a proper-width panel (see the Charts tab), and that
# original count looked sparse/empty at that size, confirmed directly.
MINI_CHART_CANDLES = {
    "1m": 150, "5m": 120, "15m": 100, "30m": 80, "1h": 80, "4h": 60,
    "1D": 60, "1W": 52, "1M": 24, "1Y": 15,
}

# Three chart panels on the page, each independently timeframe-able: the
# main analysis chart (full ICT overlays) plus two plain-candle reference
# panels, higher-timeframe on top and lower-timeframe below. Clicking any
# one of them (see ict_chart's click-return-value) points the shared TF bar
# at it — "selected_chart" tracks which.
CHART_IDS = ["main", "htf", "ltf"]
TF_KEY_BY_CHART = {"main": "fvg_tf", "htf": "mini_tf_htf", "ltf": "mini_tf_ltf"}
DEFAULT_TF_BY_CHART = {"main": "4h", "htf": "4h", "ltf": "15m"}


def fvg_legend(items):
    """Color-coded pills instead of one dense caption line — items is a list
    of (label, value_text, color) tuples."""
    pills = "".join(
        f'<div class="fvg-legend-pill" style="border-color:{color}66;">'
        f'<span class="fvg-legend-dot" style="background:{color};"></span>'
        f'<span class="fvg-legend-label">{label}</span>'
        f'<span class="fvg-legend-value" style="color:{color};">{value}</span>'
        f'</div>'
        for label, value, color in items
    )
    st.markdown(f'<div class="fvg-legend-strip">{pills}</div>', unsafe_allow_html=True)


# Sent back by ict_chart/frontend/index.html's own handleRender in place of
# a real click value — see its own comment on neededResyncBeforeThisRender —
# when THIS frontend instance's candle cache is empty but Python believes
# (from st.session_state, which survives an iframe remount that resets the
# frontend's own in-memory state) it already sent this fingerprint's full
# data. Confirmed as a REAL failure mode, not a hypothetical: the mini HTF/
# LTF panels hit this exact bug first (see their own "tried that first"
# comment a few hundred lines down) and were fixed by simply always sending
# their full array every tick — affordable there (60-150 bars). The main
# chart can't take that same fix (it can carry 10,000+ bars; resending that
# every ~10s tick would be a real bandwidth/rerun-cost regression), so it
# needs the frontend to flag the mismatch instead of Python guessing it away.
_CHART_NEEDS_FULL_RELOAD = "__ICT_CHART_NEEDS_FULL_RELOAD__"


def _reverse_ny_fake_utc_seconds(secs):
    """Inverse of _ny_fake_utc_seconds — given the fake-UTC epoch seconds
    a chart click reports, recovers the real, tz-aware timestamp it
    corresponds to. _ny_fake_utc_seconds took a real ts, converted to the
    display tz, stripped the tz label, and encoded those wall-clock digits
    as if they were UTC; this just runs that exact chain backwards:
    rebuild the naive wall-clock digits, then genuinely localize them to
    the display tz (not UTC) to get back a real instant. Comparisons
    against a df index in any other tz still work correctly afterward —
    pandas Timestamp comparisons normalize by real instant, not by label."""
    naive = _EPOCH + pd.Timedelta(seconds=secs)
    return naive.tz_localize(theme.get_display_tz())


def _hit_test_zone(clicked, clickable_zones):
    """clicked: {"time": <fake-utc seconds>, "price": <float>} from the
    chart's click handler. Returns the first clickable_zones entry (see
    its own construction comment, next to `clickable_zones = []` in
    _render_chart) whose box/level/point contains the click, searching in
    REVERSE append order so a zone drawn later (visually on top, same
    z-order every layer already draws in) wins on overlap. None on a
    miss — an honest "didn't land on anything," not a guess."""
    if not clickable_zones or clicked.get("price") is None or clicked.get("time") is None:
        return None
    click_price = clicked["price"]
    click_time = _reverse_ny_fake_utc_seconds(clicked["time"])
    for zone in reversed(clickable_zones):
        df_ref = zone.get("df_ref")
        bar_step = pd.Timedelta(seconds=60)
        if df_ref is not None and len(df_ref) > 1:
            bar_step = df_ref.index[-1] - df_ref.index[-2]
        if zone["kind"] == "rect":
            if zone["bottom"] <= click_price <= zone["top"] and zone["start"] <= click_time <= zone["end"]:
                return zone
        elif zone["kind"] == "level":
            tolerance = abs(zone["price"]) * 0.0015
            if abs(click_price - zone["price"]) <= tolerance and \
                    zone["start"] - bar_step * 10 <= click_time <= zone["end"] + bar_step * 10:
                return zone
        else:  # "point"
            tolerance = abs(zone["price"]) * 0.0015
            if abs(click_price - zone["price"]) <= tolerance and \
                    abs((click_time - zone["time"]).total_seconds()) <= bar_step.total_seconds() * 10:
                return zone
    return None


def _select_chart(chart_id, click_value):
    """Reacts to ict_chart's click-return-value — None/repeated means no new
    click since last read, a fresh value means the user just clicked this
    panel. A full st.rerun() (the default scope) is what's needed here, not
    a fragment-scoped one: the TF bar and the OTHER two panels' highlight
    state all live outside this fragment and need to see the new selection
    too, not just this one panel.

    Never called with click_value == _CHART_NEEDS_FULL_RELOAD — every call
    site that can receive it (currently just the main chart) intercepts and
    handles that sentinel itself before reaching here, since it needs
    different handling (a session-state fix-up + fragment rerun) from a
    genuine click (cross-fragment chart-selection state + a full rerun)."""
    if click_value is None:
        return
    marker_key = f"_last_click_{chart_id}"
    if st.session_state.get(marker_key) != click_value:
        st.session_state[marker_key] = click_value
        if st.session_state.get("selected_chart") != chart_id:
            st.session_state["selected_chart"] = chart_id
            st.rerun()


def _prefetch(specs):
    """The main chart, both mini charts, and the 4h/15m reference frames
    each call get_yf_ohlcv/get_latest_bars independently further down —
    fine when cached, but on a full st.rerun() (switching ticker/timeframe)
    every one of those is a cache MISS, and run one after another that's
    several sequential blocking Yahoo Finance round trips stacked up.
    Firing them all at once here first means each real call site below
    just hits an already-warm st.cache_data cache instead. get_yf_ohlcv
    itself is plain, Streamlit-API-free Python (see data.py) — but
    st.cache_data's own bookkeeping still touches the script run context
    on a slow call, hence add_script_run_ctx below rather than assuming
    these threads need nothing from Streamlit at all."""
    if len(specs) < 2:
        for fn, args, kwargs in specs:
            try:
                fn(*args, **kwargs)
            except Exception:
                pass  # the real call site below will retry and surface the error
        return
    ctx = get_script_run_ctx()
    def _run(fn, args, kwargs):
        add_script_run_ctx(threading.current_thread(), ctx)
        return fn(*args, **kwargs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(specs)) as pool:
        futures = [pool.submit(_run, fn, args, kwargs) for fn, args, kwargs in specs]
        concurrent.futures.wait(futures)


def _build_mini_overlays(chart_id, df_own, current_price, axis_secs_own, future_edge_own, df_cross):
    """A deliberately minimal ICT overlay set for the HTF/LTF mini
    reference panels — no win-rate labels, no indicator confluence, a
    fixed one-zone-per-side instead of a configurable count, unlike the
    main chart's own Layers system. These panels exist for fast "where
    does this sit in the bigger picture" context, not a second full
    analysis surface, so they show the minimum that actually answers that
    question.

    HTF panel (chart_id == "htf"):
      - "current state of delivery": the most recent confirmed structure
        break (BOS/CHoCH, via detect_structure_breaks) — the last time
        price actually confirmed a directional shift, which IS the
        current bias/"state of delivery" until the next one fires. Drawn
        extended out to future_edge (still the live read, not a closed
        historical event), same CHoCH-always-amber / BOS-by-direction
        color convention the main chart's own Market Structure layer uses.
      - "one gap per side of current price": the nearest OPEN FVG on each
        side of price.
      - "one liquidity": the single nearest liquidity level overall
        (whichever side is actually closer, not one of each — every
        other pick here is "per side," this one isn't).
      - "15 min gaps plus PD array": the nearest FVG on each side from
        df_cross (the LTF panel's own timeframe) at reduced opacity, PLUS
        this timeframe's own current dealing range — Premium/Discount,
        "PD array" — using the exact same box/EQ-line convention as the
        main chart's own Premium/Discount layer.

    LTF panel (chart_id == "ltf"): df_cross here is the HTF panel's own
    timeframe instead — its current dealing range (clipped to start no
    earlier than this panel's own visible window, so the box doesn't
    imply meaning for a stretch of time this panel doesn't even show) —
    "the move inside the current state of delivery on 4h" — plus this
    panel's own nearest FVG on each side.

    Every detector runs on the CONFIRMED slice (drops a still-forming
    last bar, same as the main chart's own tf_frames["confirmed"]) —
    an in-progress candle shouldn't get to define a zone edge yet.
    Returns {"rectangles": [], "price_lines": []} (both possibly empty)
    if df_own is too short to detect anything from at all."""
    rectangles, price_lines = [], []
    if len(df_own) < 3:
        return {"rectangles": rectangles, "price_lines": price_lines}
    own_confirmed = df_own.iloc[:-1] if len(df_own) > 1 else df_own

    def _dealing_range_overlay(dr, t0_floor=None):
        t0 = _ny_fake_utc_seconds(dr["start"])
        if t0_floor is not None:
            t0 = max(t0, t0_floor)
        rectangles.append({"t0": t0, "t1": future_edge_own, "p0": dr["eq"], "p1": dr["top"],
                            "fill": _hex_to_rgba(theme.NEON_MAGENTA, 0.14), "border": None})
        rectangles.append({"t0": t0, "t1": future_edge_own, "p0": dr["bottom"], "p1": dr["eq"],
                            "fill": _hex_to_rgba(theme.NEON_GREEN, 0.14), "border": None})
        price_lines.append({"t0": t0, "t1": future_edge_own, "price": dr["eq"],
                             "color": _hex_to_rgba(theme.NEON_AMBER, 1.0), "title": "EQ 50%", "above": True})

    def _fvg_overlay(zones, opacity):
        # Every zone passed in has already been filtered to open-only
        # (see the two call sites below) — always projecting to
        # future_edge_own rather than branching on g["filled"] the way
        # the main chart's own richer FVG block does is correct here,
        # not a shortcut, since a filled zone never reaches this function.
        for g in zones:
            color = theme.NEON_CYAN if g["type"] == "bullish" else theme.NEON_MAGENTA
            rectangles.append({"t0": _ny_fake_utc_seconds(g["start"]), "t1": future_edge_own,
                                "p0": g["bottom"], "p1": g["top"],
                                "fill": _hex_to_rgba(color, opacity), "border": color})

    if chart_id == "htf":
        breaks = detect_structure_breaks(own_confirmed)
        if breaks:
            b = breaks[-1]
            is_choch = b["structure"] == "CHoCH"
            color = theme.NEON_AMBER if is_choch else (theme.NEON_GREEN if b["type"] == "bullish" else theme.NEON_MAGENTA)
            price_lines.append({"t0": _ny_fake_utc_seconds(b["start"]), "t1": future_edge_own,
                                 "price": b["level"], "color": _hex_to_rgba(color, 1.0),
                                 "title": f"{b['type'].capitalize()} {b['structure']}", "above": b["type"] == "bullish"})

        dr = current_dealing_range(own_confirmed)
        if dr is not None:
            _dealing_range_overlay(dr)

        open_fvgs = [g for g in detect_fvgs(own_confirmed) if not g["filled"]]
        _fvg_overlay(_nearest_by_price(open_fvgs, current_price, 1), 0.35)

        above, below = detect_liquidity_levels(own_confirmed, n_above=1, n_below=1)
        nearest_liq = None
        if above and below:
            nearest_liq = ("above", above[0]) if (above[0]["price"] - current_price) \
                <= (current_price - below[0]["price"]) else ("below", below[0])
        elif above:
            nearest_liq = ("above", above[0])
        elif below:
            nearest_liq = ("below", below[0])
        if nearest_liq is not None:
            side, lvl = nearest_liq
            color = theme.NEON_MAGENTA if side == "above" else theme.NEON_AMBER
            price_lines.append({"t0": _ny_fake_utc_seconds(lvl["time"]), "t1": future_edge_own,
                                 "price": lvl["price"], "color": _hex_to_rgba(color, 1.0),
                                 "title": "BSL" if side == "above" else "SSL", "above": side == "above"})

        if len(df_cross) >= 3:
            cross_confirmed = df_cross.iloc[:-1] if len(df_cross) > 1 else df_cross
            cross_fvgs = [g for g in detect_fvgs(cross_confirmed) if not g["filled"]]
            _fvg_overlay(_nearest_by_price(cross_fvgs, current_price, 1), 0.18)

    else:  # "ltf"
        if len(df_cross) >= 3:
            cross_confirmed = df_cross.iloc[:-1] if len(df_cross) > 1 else df_cross
            dr = current_dealing_range(cross_confirmed)
            if dr is not None:
                _dealing_range_overlay(dr, t0_floor=axis_secs_own[0] if axis_secs_own else None)

        open_fvgs = [g for g in detect_fvgs(own_confirmed) if not g["filled"]]
        _fvg_overlay(_nearest_by_price(open_fvgs, current_price, 1), 0.35)

    return {"rectangles": rectangles, "price_lines": price_lines}


def _render_mini_chart(ticker, chart_id):
    """A small reference chart with a light ICT overlay layer (see
    _build_mini_overlays) — deliberately much less detection work than
    the main chart's full Layers system, so glancing at HTF/LTF context
    next to whatever the main chart is zoomed into stays cheap. Its own
    timeframe is independently selectable (via the shared TF bar, once
    clicked to select it) rather than fixed."""
    tf_key = st.session_state.get(TF_KEY_BY_CHART[chart_id], DEFAULT_TF_BY_CHART[chart_id])
    tf_conf = TIMEFRAMES[tf_key]
    refresh_interval = CANDLE_SECONDS.get(tf_key)
    # Every intraday panel polls at 1s regardless of which one is selected.
    # This used to only apply to the selected panel (10-120s otherwise) on
    # the theory that nobody's watching the other two closely — but the main
    # chart's price ticks live every 1s while an unselected HTF/LTF panel
    # sat on its slow cadence (up to 120s for 4h), so the same live ticker
    # visibly showed a different last price/candle across panels for up to
    # two minutes at a time — reads as the charts being "out of sync" even
    # though they're the same instrument. get_yf_ohlcv's own ttl=60 cache
    # already bounds the real fetch cost, and mini panels do no detection
    # work (plain candles only), so polling all three at 1s is cheap — most
    # reruns are just a cache hit plus one incremental series.update() call.
    # Daily+ TFs stay un-polled (refresh_interval already None) — a once-a-
    # day candle has nothing meaningful to tick every second.
    if refresh_interval is not None:
        refresh_interval = 1
    # Daily+ TFs (refresh_interval still None here) don't need per-second
    # CANDLE redraws, but the panel's own "current price" readout is a
    # different thing — that still needs to eventually catch up to the
    # live price. Confirmed directly as a real bug: a daily+ mini panel
    # left alone (nothing ELSE on the page happening to force a rerun)
    # showed a genuinely stale last price — a large, real gap from the
    # true live one, not a rounding nitpick. fragment_interval is what
    # actually goes to @st.fragment below; refresh_interval keeps its
    # EXACT original meaning (None vs 1) for the splice-logic gate just
    # below, untouched — this only adds a slow floor under the "never
    # reruns on its own at all" case, it doesn't change intraday behavior.
    fragment_interval = refresh_interval if refresh_interval is not None else 45
    # Read from session_state rather than a passed-in arg — this function is
    # a plain module-level def (not nested inside the settings popover's own
    # scope the way _render_chart is), same reason fvg_show_volume below is
    # read the same way.
    data_source = DATA_SOURCES.get(st.session_state.get("fvg_data_source", "Auto"), "auto")

    @st.fragment(run_every=fragment_interval)
    def _inner():
        try:
            df = get_yf_ohlcv(ticker, period=tf_conf["period"], interval=tf_conf["fetch_interval"], provider=data_source)
            # Checked BEFORE resample, not after — resample_ohlc's own
            # df.resample(rule) call raises on an empty df's default
            # RangeIndex instead of just handing back another empty df
            # (same bug, same fix, as the main chart's version of this).
            if df.empty:
                st.caption(f"{tf_key}: no data for {ticker}")
                return
            if tf_conf["resample"] is None and refresh_interval is not None:
                # Same fast-splice pattern as the main chart's _render_chart
                # (see its own comment above accum_key there for the full
                # reasoning). Without this, a mini panel's price only moved
                # once per get_yf_ohlcv's own 60s cache TTL no matter how
                # often its fragment polled — the panel LOOKED like it was
                # ticking every 1s (the fragment really did rerun) but the
                # underlying df was the same stale object until the cache
                # actually expired, so the candle just sat there. Splicing
                # get_latest_bars' cheap, fast-ttl fetch on top makes the
                # visible price move every real tick instead of every 60th.
                df_latest = get_latest_bars(ticker, tf_conf["fetch_interval"], provider=data_source)
                if not df_latest.empty:
                    df = pd.concat([df[df.index < df_latest.index[0]], df_latest])
                    last_ts = df.index[-1]
                    accum_key = f"_forming_bar_mini_{chart_id}_{ticker}_{tf_key}"
                    accum = st.session_state.get(accum_key)
                    last_open = float(df["Open"].iloc[-1])
                    last_high = float(df["High"].iloc[-1])
                    last_low = float(df["Low"].iloc[-1])
                    if accum is None or accum["ts"] != last_ts:
                        accum = {"ts": last_ts, "open": last_open, "high": last_high, "low": last_low}
                    else:
                        accum["high"] = max(accum["high"], last_high)
                        accum["low"] = min(accum["low"], last_low)
                    st.session_state[accum_key] = accum
                    df.loc[last_ts, "Open"] = accum["open"]
                    df.loc[last_ts, "High"] = accum["high"]
                    df.loc[last_ts, "Low"] = accum["low"]
            if tf_conf["resample"]:
                df = resample_ohlc(df, tf_conf["resample"])
            df = df.tail(MINI_CHART_CANDLES[tf_key])

            # The OTHER mini panel's own timeframe — HTF's overlays need
            # LTF's df for its "15m gaps" context, LTF's overlays need
            # HTF's df for its "current state of delivery" box (see
            # _build_mini_overlays' own docstring). Same (ticker, period,
            # interval) combo the other panel's own _inner() ALSO fetches,
            # so get_yf_ohlcv's cache makes this a real network round trip
            # only on the very first render of either panel, a cache hit
            # every time after.
            other_id = "ltf" if chart_id == "htf" else "htf"
            other_tf_key = st.session_state.get(TF_KEY_BY_CHART[other_id], DEFAULT_TF_BY_CHART[other_id])
            other_tf_conf = TIMEFRAMES[other_tf_key]
            df_cross = get_yf_ohlcv(ticker, period=other_tf_conf["period"],
                                     interval=other_tf_conf["fetch_interval"], provider=data_source)
            if not df_cross.empty and other_tf_conf["resample"]:
                df_cross = resample_ohlc(df_cross, other_tf_conf["resample"])
            if not df_cross.empty:
                df_cross = df_cross.tail(MINI_CHART_CANDLES[other_tf_key])

            o_col, h_col, l_col, c_col = (("Open", "High", "Low", "Close") if "Close" in df
                                           else ("open", "high", "low", "close"))
            v_col = "Volume" if "Volume" in df else ("volume" if "volume" in df else None)
            axis_secs = _ny_fake_utc_seconds_vec(df.index)
            # Same 3-candle future extension the main chart uses, so a
            # dealing-range box/liquidity line drawn out to "still active"
            # visibly continues past the last real candle instead of
            # stopping exactly where price has printed so far.
            _mini_bar_step = (axis_secs[-1] - axis_secs[-2]) if len(axis_secs) > 1 else 1
            future_edge = axis_secs[-1] + 3 * _mini_bar_step if axis_secs else 0
            # .to_numpy() once, not .iloc[i] per cell per row — see the main
            # chart's own identical fix below for the measured cost of that.
            _o, _h, _l, _c = (df[o_col].to_numpy(), df[h_col].to_numpy(),
                              df[l_col].to_numpy(), df[c_col].to_numpy())
            _v = df[v_col].to_numpy() if v_col is not None else None
            bars = [
                {
                    "time": axis_secs[i],
                    "open": float(_o[i]), "high": float(_h[i]),
                    "low": float(_l[i]), "close": float(_c[i]),
                    "volume": float(_v[i]) if _v is not None else 0.0,
                }
                for i in range(len(df))
            ]
            # Just ticker+timeframe — no axis_secs[0] either. Unlike the
            # main chart (where the oldest loaded bar only moves when the
            # whole multi-day window genuinely rolls forward), these mini
            # panels are ALWAYS a fixed-size trailing tail(N): the oldest
            # bar shifts by design on literally every new candle, so using
            # it as a "did something structural change" signal would force
            # a full refit on every ordinary tick — the exact bug this is
            # fixing, just via a different field. Nothing here should ever
            # warrant a reset except actually switching ticker or timeframe.
            fingerprint = f"{ticker}|{tf_key}|mini"

            # Always the FULL array here, not the main chart's own
            # send-only-the-tail-on-an-unchanged-fingerprint trimming —
            # tried that first (a session_state flag remembering "already
            # sent this fingerprint's full data") and confirmed directly
            # it breaks for this panel specifically: living inside the
            # side panel's Charts tab, its own iframe can get freshly
            # remounted (a brand-new client with no cached data at all)
            # without the fingerprint itself changing — e.g. clicking
            # into the Charts tab after it wasn't the active one. The
            # session_state flag, having already been set from an
            # EARLIER mount, told this fresh one "just send the 2-bar
            # tail," leaving its series holding only those 2 bars —
            # confirmed live: a chart rendering almost empty, visible
            # range going negative. Mini panels only hold 60-150 bars
            # (see MINI_CHART_CANDLES) — resending the full array every
            # tick here is a small, affordable price for not depending on
            # an assumption (this component's own client-side state
            # persists across every render) that doesn't actually always
            # hold.
            bars_payload = bars
            current_price = float(df[c_col].iloc[-1])
            # Overlays only get RECOMPUTED when the underlying CONFIRMED
            # data actually changed (a new bar closed, on either this
            # panel's own timeframe or the cross-referenced one) — not
            # every tick. Confirmed directly as a real, sustained CPU
            # cost otherwise: this fragment polls every 1s on an
            # intraday timeframe, and each of detect_fvgs/detect_swings/
            # detect_structure_breaks/current_dealing_range/
            # detect_liquidity_levels is individually @st.cache_data-
            # decorated, but every call still pays real hashing/lookup
            # overhead regardless of a hit — roughly 5 calls x 2 (own +
            # cross) x 2 panels x once a second measured as a sustained
            # 55-75% CPU load, running continuously regardless of
            # whether the Charts tab is even the one currently visible
            # (Streamlit executes a tab's own body every rerun
            # regardless of which tab is shown — a well-known framework
            # behavior, not specific to this code). A dealing range or a
            # structure break genuinely changes on the order of minutes
            # to hours, never every second, so recomputing either at 1s
            # cadence was pure waste — the "nearest zone" picks below
            # only ever refresh on a bar close as a result, not
            # continuously, which is the right tradeoff: correctness
            # (which zones exist) is untouched, only the update
            # frequency of a "reference glance" panel's own emphasis is.
            _mini_own_last = df.index[-2] if len(df) > 1 else (df.index[-1] if len(df) else None)
            _mini_cross_last = df_cross.index[-2] if len(df_cross) > 1 else (df_cross.index[-1] if len(df_cross) else None)
            _mini_overlay_key = (chart_id, _mini_own_last, _mini_cross_last)
            _mini_overlay_state_key = f"_mini_overlay_cache_{chart_id}"
            _mini_cached = st.session_state.get(_mini_overlay_state_key)
            if _mini_cached is not None and _mini_cached["key"] == _mini_overlay_key:
                overlays = _mini_cached["overlays"]
            else:
                overlays = _build_mini_overlays(chart_id, df, current_price, axis_secs, future_edge, df_cross)
                st.session_state[_mini_overlay_state_key] = {"key": _mini_overlay_key, "overlays": overlays}

            # key stays keyed to chart_id ("htf"/"ltf"), NOT tf_key — tf_key
            # now changes at runtime, and a key that changes when the data
            # changes forces a full iframe remount (see ict_chart's own
            # docstring warning), which would silently break every time this
            # panel's timeframe gets switched.
            clicked = ict_chart(
                bars_payload, fingerprint,
                overlays=overlays,
                options={"log_scale": st.session_state.get("fvg_log_scale", False),
                          "volume": st.session_state.get("fvg_show_volume", False),
                          "selected": st.session_state.get("selected_chart") == chart_id,
                          # Plain candles at a glance — no O/H/L/C/source
                          # heading competing with the main chart's own
                          # for space in this smaller panel.
                          "hide_ohlc": True,
                          # Always show the whole (small, fixed tail(N))
                          # window fitted on every timeframe switch — this
                          # panel's own iframe persists across a TF switch
                          # (key stays chart_id, not tf_key; see the
                          # comment above), so without this it inherited
                          # the main chart's calendar-window-preservation
                          # behavior instead, which reads as "not fitted"
                          # for a small reference panel that should always
                          # just show everything it's currently holding.
                          "always_fit": True,
                          # A bit more breathing room after the last candle
                          # than the main chart's own 8 (see createChart's
                          # rightOffset) — requested directly, this panel's
                          # smaller width made the default margin read as
                          # too tight.
                          "right_offset": 14},
                ohlc={"ticker": "", "source": tf_key, "symbol": ticker, "interval": tf_key},
                height=380,
                key=f"ict_chart_mini_{chart_id}",
                display_tz=theme.get_display_tz(),
            )
            _select_chart(chart_id, clicked)
        except Exception as e:
            st.error(f"{tf_key}: {e}")

    _inner()


# Remembers the last ticker/main-timeframe across page reloads AND server
# restarts (session_state alone only survives within one browser tab's
# session) — a small local file, read once at startup and rewritten
# whenever either actually changes. Deliberately just ticker + the MAIN
# chart's timeframe, not the HTF/LTF panels' — those have fixed roles
# (higher/lower timeframe reference), not a "last used" concept.
_LAST_STATE_FILE = os.path.join(os.path.dirname(__file__), ".last_state.json")


def _load_last_state():
    try:
        with open(_LAST_STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_last_state(ticker, tf_label, layers_state):
    try:
        with open(_LAST_STATE_FILE, "w") as f:
            json.dump({"ticker": ticker, "tf": tf_label, "layers": layers_state}, f)
    except OSError:
        pass


_last_state = _load_last_state()
st.session_state.setdefault("fvg_ticker", _last_state.get("ticker", "BTC-USD"))

# Lets an external link/bookmark/script drive the ticker via ?ticker=AAPL —
# only applied once per distinct query value (tracked in _last_qp_ticker) so
# it seeds the widget on load without fighting the user's own later picks;
# once they change the ticker by hand, the URL is stale and stays ignored
# until it actually changes to something new. An explicit link deliberately
# outranks the remembered-from-last-time ticker above.
qp_ticker = st.query_params.get("ticker")
if qp_ticker and st.session_state.get("_last_qp_ticker") != qp_ticker:
    st.session_state["fvg_ticker"] = qp_ticker.strip().upper()
    st.session_state["_last_qp_ticker"] = qp_ticker

for _cid in CHART_IDS:
    default_tf = _last_state.get("tf", DEFAULT_TF_BY_CHART[_cid]) if _cid == "main" else DEFAULT_TF_BY_CHART[_cid]
    st.session_state.setdefault(TF_KEY_BY_CHART[_cid], default_tf)

# Same seed-before-widget-creation trick as ticker/tf above, so each layer's
# on/off + own detection timeframe survives a real browser refresh (a new
# Streamlit session) instead of resetting every time. Each layer picks ONE
# timeframe independently via its own dropdown (see the Layers popover) —
# no more fixed 4h/15m pair with confluence filtering between them. Default
# is "Chart TF" — a live-follow of whatever the main chart is currently
# showing (resolved in _render_chart, same as the MA indicator's own
# "Chart TF" option), not a one-time snapshot of the TF at load time —
# picking a concrete timeframe instead is a deliberate per-layer override
# that persists on its own from then on, same as everything else in
# .last_state.json.
_last_layers = _last_state.get("layers", {})
for _name in ICT_LAYERS:
    _saved_layer = _last_layers.get(_name, {})
    st.session_state.setdefault(f"fvg_layer_{_name}", _saved_layer.get("on", LAYER_DEFAULTS[_name]))
    st.session_state.setdefault(f"fvg_tf_{_name}", _saved_layer.get("tf", "Chart TF"))

theme.render_top_bar("Markets")

st.session_state.setdefault("selected_chart", "main")
selected_chart = st.session_state["selected_chart"]

# fvg_ticker is plain session_state now, not a selectbox's own value — the
# market-menu buttons below set it directly (see "Markets" expander), so
# there's no widget to read it FROM at this point in the script.
ticker = st.session_state.get("fvg_ticker", "GBPUSD=X").strip().upper()

# Warms all three panels' data caches IN PARALLEL before any of them render.
# Without this, a full script rerun (ticker change, tf change, or even an
# unrelated widget like a layer checkbox) renders main/htf/ltf sequentially,
# one Python call after another — whichever panel's fetch is biggest (HTF's
# 4h is resampled from a 730-day/60m pull, far larger than main's 7d/1m or
# LTF's 60d/15m) blocks for noticeably longer, so it's still showing its
# PREVIOUS price/candle for a second or more after the other two have
# already repainted with the new data. Confirmed directly: right after a
# full reload, HTF sat several seconds behind main/LTF before catching up —
# exactly this effect, not a per-tick sync bug (that one's fixed above via
# the splice pattern). Firing all three's get_yf_ohlcv + get_latest_bars
# through _prefetch's ThreadPoolExecutor means each panel's own render call
# below just hits an already-warm st.cache_data entry, so they finish and
# repaint together instead of staggered by fetch size.
_prefetch_data_source = DATA_SOURCES.get(st.session_state.get("fvg_data_source", "Auto"), "auto")
_prefetch_specs = []
for _cid in CHART_IDS:
    _tf_key = st.session_state.get(TF_KEY_BY_CHART[_cid], DEFAULT_TF_BY_CHART[_cid])
    _tf_conf = TIMEFRAMES[_tf_key]
    _prefetch_specs.append((get_yf_ohlcv, (ticker, _tf_conf["period"], _tf_conf["fetch_interval"], _prefetch_data_source), {}))
    if _tf_conf["resample"] is None and CANDLE_SECONDS.get(_tf_key) is not None:
        _prefetch_specs.append((get_latest_bars, (ticker, _tf_conf["fetch_interval"]), {"provider": _prefetch_data_source}))
_prefetch(_prefetch_specs)

def _confluence_breakdown_md(setup, entry_tf, context_tf=None):
    """Plain-language rundown of exactly which confluence factors backed
    THIS ONE trade. Shown directly as the scan-result button's own
    `help=` tooltip (see both sidebar scan loops below) — a fixed,
    identical-on-every-row explanation of the general methodology used to
    live there instead, which meant hovering any of 3+ results in a scan
    showed the exact same text three times over; per-result content is
    what's actually worth surfacing per row."""
    entry_labels = [d["label"] for d in (setup.get("confluence_entry_details") or [])]
    context_labels = [d["label"] for d in (setup.get("confluence_context_details") or [])]
    lines = [f"**{entry_tf} entry** — {len(entry_labels)} of 5 factors agree:"]
    lines += [f"- {f}" for f in entry_labels] if entry_labels else ["- none currently agree"]
    if context_tf:
        lines.append("")
        lines.append(f"**{context_tf} context** (counts {CONTEXT_WEIGHT}x) — {len(context_labels)} of 5 factors agree:")
        lines += [f"- {f}" for f in context_labels] if context_labels else ["- none currently agree"]
    return "\n".join(lines)


def _confluence_body_lines(entity):
    """Compact on-chart confluence COUNT for a trade's own box fill (see
    _trade_box_overlays/_rebalance_chain_overlays' own body_lines param)
    — direct request: "I just want amount of confluences for live feed,"
    walking back the earlier full-sentence-per-factor version (still
    available in full via the sidebar's own scan-result hover — see
    _confluence_breakdown_md — this is only the on-chart headline
    number). Works for either a best_trade_now/scan_timeframes setup
    dict or an _active_scan_pick dict — both carry the same two field
    names. A "Validated edge" sidebar pick (see validated_stats) carries
    real permutation-test evidence instead of a confluence tally — a
    stronger claim than confluence, so it's shown in place of the count
    rather than alongside "0 confluences" (which this kind of pick
    always has, having nothing to do with best_trade_now's scoring)."""
    validated = entity.get("validated_stats")
    if validated:
        return [f"Validated: p={validated['p_value_train']:.4f}, n={validated['n_events']}, "
                f"holdout {validated['mean_return_holdout']:+.2%}"]
    entry_n = len(entity.get("confluence_entry_details") or [])
    context_details = entity.get("confluence_context_details")
    if context_details:
        return [f"{entry_n} entry + {len(context_details)} HTF confluences"]
    return [f"{entry_n} confluences"]


def _scan_pick_overlays(scan_pick, future_edge, ts_to_x):
    """The isolated "lock trade" view: draw JUST this one trade's own
    trigger zone plus the specific zones that backed each matched
    confluence factor — nothing else, no other layer's clutter (direct
    request: "I want to see all the confluences ... with those exact
    zones, nothing else"). Replaces the normal rectangles/price_lines
    entirely at the call site below rather than adding to them.

    ts_to_x: the caller's own _ny_fake_utc_seconds — converts a real
    timestamp into THIS chart's x-axis regardless of which timeframe is
    currently displayed. Real timestamps are absolute, so this still
    places a zone correctly even when it came from a DIFFERENT
    timeframe's own dataframe than what's on screen right now (the whole
    point of the lock checkbox: pin the trade, then browse timeframes
    freely). Returns (rectangles, price_lines)."""
    direction = scan_pick["direction"]
    trigger_color = theme.NEON_GREEN if direction == "bullish" else theme.NEON_MAGENTA
    rectangles, price_lines = [], []

    def _end_x(end_ts):
        return future_edge if end_ts is None else ts_to_x(end_ts)

    src = scan_pick.get("source_zone")
    if src:
        # Gold border — the same "premium, confluence-backed zone" visual
        # language every other layer on this chart already uses (see
        # theme.CONFLUENCE_GOLD's own other call sites), reused here for
        # consistency rather than inventing a new meaning for gold.
        rectangles.append({
            "t0": ts_to_x(src["start"]), "t1": _end_x(src["end"]), "p0": src["bottom"], "p1": src["top"],
            "fill": _hex_to_rgba(trigger_color, 0.18), "fill_to": None,
            "border": theme.CONFLUENCE_GOLD, "border_width": 2, "label": scan_pick.get("label"),
        })

    # Supporting confluence zones — thin, muted cyan outlines so they
    # read as corroborating evidence, not competing primary zones next
    # to the trigger's own gold-bordered one.
    for details_key in ("confluence_entry_details", "confluence_context_details"):
        for factor in scan_pick.get(details_key) or []:
            for z in factor["zones"]:
                if z["kind"] == "rect":
                    rectangles.append({
                        "t0": ts_to_x(z["start"]), "t1": _end_x(z["end"]), "p0": z["bottom"], "p1": z["top"],
                        "fill": _hex_to_rgba(theme.NEON_CYAN, 0.05), "fill_to": None,
                        "border": _hex_to_rgba(theme.NEON_CYAN, 0.5), "border_width": 1, "label": factor["label"],
                    })
                else:  # "level"
                    price_lines.append({
                        "t0": ts_to_x(z["start"]), "t1": _end_x(z["end"]), "price": z["price"],
                        "color": _hex_to_rgba(theme.NEON_CYAN, 0.6), "title": "", "line_width": 1, "dashed": True,
                        "above": bool(z["price"] >= scan_pick["entry"]),
                    })
    return rectangles, price_lines


def _rebalance_chain_overlays(chain, chart_edge_x, slice_seconds):
    """Draws each rebalance_chain step as its OWN forward-projected risk/
    reward box — the EXACT SAME solid flat fill + visible border the
    Backtest tab's own "Show these trades on the chart" already draws
    for a resolved trade (see _render_chart's own backtest-trades block),
    not a gradient. Direct request: "During the backtest, the areas are
    rendered a specific way. I want exactly that for the active trade...
    the box should be printed from present to future a bit. Then the
    next trade chained from that future point to the next" — each step
    starts exactly where the previous one's box ends (chart_edge_x for
    the first is the ACTIVE trade's own box end, see _active_box_t1),
    forming one continuous chained sequence rather than independent
    slices. Numbered so the sequence reads left to right exactly as the
    sidebar's own chain list does: closest hunted level first, then
    whichever's closest from there.

    chart_edge_x: where the first step's box starts. slice_seconds:
    width of each step's box; NOT real elapsed time (this is a
    projection, not a forecast timestamp) — just wide enough to read
    comfortably at whatever timeframe is currently charted.

    The step's own "N. ctx→entry Pattern" identifier is drawn as the
    REWARD rectangle's own right-anchored label (not a separate price-
    line title) — reusing the exact same convention every other zone on
    this chart already uses, and the exact same width-based hiding (see
    RectangleRenderer's own comment) instead of a bespoke placement.
    Reward, not risk: best_trade_now's fixed 2R target means the reward
    box is always twice the risk box's own height, so it has more room
    for both this label and the confluence body_lines below. The
    confluence factors behind THIS step (same ones the sidebar's own ℹ️
    popover breaks down) are drawn as body_lines inside that same
    rectangle's fill — direct request: "within the fill area, should
    contain text with all confluences to consider about it." Returns
    (rectangles, price_lines)."""
    rectangles, price_lines = [], []
    t = chart_edge_x
    for i, step in enumerate(chain):
        t0, t1 = t, t + slice_seconds
        t = t1
        setup = step["setup"]
        entry, sl, tp = setup["entry_price"], setup["sl_price"], setup["tp_price"]
        _pattern = EVENT_TYPE_LABELS.get(setup["event_type"], setup["event_type"])
        rectangles.append({"t0": t0, "t1": t1, "p0": entry, "p1": sl,
                            "fill": _hex_to_rgba(theme.NEON_MAGENTA, 0.10),
                            "border": _hex_to_rgba(theme.NEON_MAGENTA, 0.9)})
        rectangles.append({"t0": t0, "t1": t1, "p0": entry, "p1": tp,
                            "fill": _hex_to_rgba(theme.NEON_GREEN, 0.10),
                            "border": _hex_to_rgba(theme.NEON_GREEN, 0.9),
                            "label": f"{i + 1}. {setup['context_timeframe']}→{setup['timeframe']} {_pattern}",
                            "body_lines": _confluence_body_lines(setup)})
        # A plain, bounded entry marker — dashed (not the solid/glow
        # style the always-open active trade uses), since a NON-dashed
        # line stretches to the pane's own right edge regardless of t1
        # (see HLineRenderer's own comment): every step's line would
        # overshoot into every later box instead of staying inside its
        # own, exactly the overlap the "avoid overlapping text... don't
        # want a circus" request is about.
        price_lines.append({
            "t0": t0, "t1": t1, "price": entry, "color": _hex_to_rgba(theme.NEON_AMBER, 0.7),
            "title": "", "line_width": 1, "dashed": True, "above": True,
        })
    return rectangles, price_lines


# Sidebar: a multi-symbol scan across the whole watchlist — collapsed by
# default (initial_sidebar_state="collapsed" above), so the chart stays
# the primary view. This is deliberately where the "advanced, scan
# everything" functionality lives instead of a separate page: tucked out
# of the way until asked for, one click from the chart it feeds (picking
# a result jumps the main ticker straight to it), not a second surface to
# split attention with.
with st.sidebar:
    # The strongest tier of signal this page can show: unlike "Scan
    # watchlist" below (confluence — several ICT reads agreeing, no
    # statistical proof behind it), a "Validated edge" hit is backed by
    # an actual permutation test that already cleared Benjamini-Hochberg
    # correction across the WHOLE accumulated Experiments trial log and
    # held up on untouched holdout data (see experiments.py's own
    # run_deep_backtest/load_experiment_trials) — the same bar Edge Lab
    # used to hold before its own trial-running code went missing.
    # list_validated_pairs reads that log directly rather than a
    # hardcoded ticker/timeframe list, so this grows on its own as more
    # Experiments tab backtests get run and survive correction, with
    # nothing to maintain here.
    st.subheader("🔬 Validated edge")
    if st.button("🔬 Scan for validated setups", key="validated_scan_btn", width="stretch",
                 help="Checks every ticker/timeframe on this watchlist that has ever cleared a real "
                      "permutation test (Experiments tab \"Run deep backtest\", BH-corrected across "
                      "everything ever tried) for a signal that's live right now — not just confluence, "
                      "actual statistical evidence. Usually empty; that's honest, not broken."):
        _val_pairs = experiments.list_validated_pairs(TICKER_UNIVERSE)
        _val_specs = [(get_yf_ohlcv, (t, TIMEFRAMES[tf]["period"], TIMEFRAMES[tf]["fetch_interval"],
                                       _prefetch_data_source), {}) for t, tf in _val_pairs]
        _prefetch(_val_specs)
        _val_hits = []
        for _val_ticker, _val_tf in _val_pairs:
            _val_conf = TIMEFRAMES[_val_tf]
            try:
                _val_df = get_yf_ohlcv(_val_ticker, period=_val_conf["period"], interval=_val_conf["fetch_interval"],
                                        provider=_prefetch_data_source)
                if _val_conf["resample"] and not _val_df.empty:
                    _val_df = resample_ohlc(_val_df, _val_conf["resample"])
                _val_hit = experiments.find_live_validated_signal(_val_df, _val_ticker, _val_tf)
            except Exception:
                _val_hit = None
            if _val_hit:
                _val_hits.append(_val_hit)
        st.session_state["_validated_scan"] = _val_hits
        st.session_state["_validated_scan_checked"] = len(_val_pairs)

    _val_results = st.session_state.get("_validated_scan")
    if _val_results is None:
        st.caption("Not scanned yet this session.")
    elif not _val_results:
        _val_n_checked = st.session_state.get("_validated_scan_checked", 0)
        if _val_n_checked == 0:
            st.caption("Nothing on this watchlist has cleared a validated Experiments backtest yet — "
                       "run \"Run deep backtest\" in a chart's own 🧪 Experiments tab to build one.")
        else:
            st.caption(f"Scanned {_val_n_checked} validated ticker/timeframe combo(s) — nothing live right now.")
    else:
        for _v in _val_results:
            _v_label = f"{_v['ticker']} · {_v['tf_label']} · {_v['label']} ({_v['direction']})"
            _v_help = (f"p={_v['p_value_train']:.4f} on train · train mean {_v['mean_return_train']:+.3%} · "
                       f"holdout mean {_v['mean_return_holdout']:+.3%} (same direction, {_v['n_events']} events, "
                       f"touched {_v['bars_since_touch']} bar(s) ago). Target line is a PROJECTION of the "
                       f"historical holdout mean move, not a real take-profit — this edge actually exits after "
                       f"{_v['forward_bars']} bars, not at a price level.")
            if st.button(_v_label, key=f"val_jump_{_v['ticker']}_{_v['tf_label']}_{_v['entry_time']}",
                         width="stretch", help=_v_help):
                st.session_state["fvg_ticker"] = _v["ticker"]
                st.session_state[TF_KEY_BY_CHART["main"]] = _v["tf_label"]
                st.session_state["_active_scan_pick"] = {
                    "ticker": _v["ticker"], "timeframe": _v["tf_label"], "direction": _v["direction"],
                    "entry": _v["entry_price"], "stop": _v["stop_price"], "target": _v["target_price"],
                    "label": f"Validated: {_v['label']}",
                    "context_timeframe": None,
                    "source_zone": {"start": _v["zone_start"], "end": None,
                                     "top": _v["zone_top"], "bottom": _v["zone_bottom"]},
                    "confluence_entry_details": [], "confluence_context_details": [],
                    "validated_stats": {"p_value_train": _v["p_value_train"], "n_events": _v["n_events"],
                                         "mean_return_holdout": _v["mean_return_holdout"]},
                }
                st.session_state["_scan_pick_locked"] = False
                st.session_state.pop("_active_chart_rule", None)
                st.rerun()

    st.divider()
    st.subheader("🎯 Signals")
    # Explanations live in each button's own hover (help=), not as a
    # standing caption -- direct request: don't put explanatory text in my
    # face every time I open this, spoon-feed the actual results instead.
    if st.button("🔍 Scan watchlist", key="signals_scan_btn", width="stretch",
                 help="Scans every symbol on this watchlist for its own single best currently-active setup on "
                      "4h, with 1W context weighted in (see recommender.py's own TF_PAIRS[0]) — see which "
                      "symbol has the strongest read right now, not just whatever's already charted."):
        _scan_conf = TIMEFRAMES["4h"]
        # TF_PAIRS[0] ("1W" context for the "4h" entry this scan already
        # uses) — one representative top-down read per ticker, not every
        # pair (see scan_watchlist's own docstring on why trying all of
        # TF_PAIRS here would multiply fetch cost across the whole
        # watchlist for little extra signal).
        _scan_context_tf, _ = TF_PAIRS[0]
        _scan_context_conf = TIMEFRAMES[_scan_context_tf]
        _scan_specs = [(get_yf_ohlcv, (t, _scan_conf["period"], _scan_conf["fetch_interval"], _prefetch_data_source), {})
                       for t in TICKER_UNIVERSE]
        _scan_specs += [(get_yf_ohlcv, (t, _scan_context_conf["period"], _scan_context_conf["fetch_interval"],
                                         _prefetch_data_source), {}) for t in TICKER_UNIVERSE]
        _prefetch(_scan_specs)
        _scan_dfs = {}
        _scan_context_dfs = {}
        for _t in TICKER_UNIVERSE:
            try:
                _scan_dfs[_t] = get_yf_ohlcv(_t, period=_scan_conf["period"], interval=_scan_conf["fetch_interval"],
                                              provider=_prefetch_data_source)
            except Exception:
                _scan_dfs[_t] = None
            try:
                _scan_context_dfs[_t] = get_yf_ohlcv(_t, period=_scan_context_conf["period"],
                                                       interval=_scan_context_conf["fetch_interval"],
                                                       provider=_prefetch_data_source)
            except Exception:
                _scan_context_dfs[_t] = None
        st.session_state["_signals_scan"] = scan_watchlist(
            _scan_dfs, _scan_conf["fetch_interval"], top_n=8, context_dfs_by_ticker=_scan_context_dfs)

    _scan_results = st.session_state.get("_signals_scan")
    if _scan_results is None:
        st.caption("Not scanned yet this session.")
    elif not _scan_results:
        st.caption("Scanned — nothing currently active anywhere on the watchlist right now.")
    else:
        _scan_context_tf_label, _scan_entry_tf_label = TF_PAIRS[0]
        for _s in _scan_results:
            _pattern = EVENT_TYPE_LABELS.get(_s["event_type"], _s["event_type"])
            _label = f"{_s['ticker']} · {_pattern} ({_s['direction']}) · confluence {_s['confluence_score']}"
            if st.button(_label, key=f"sig_jump_{_s['ticker']}_{_s['event_type']}_{_s['start']}", width="stretch",
                         help=_confluence_breakdown_md(_s, _scan_entry_tf_label, _scan_context_tf_label)):
                # Every scan_watchlist candidate is on the fixed "4h" entry
                # timeframe (_scan_conf above) — jump the chart there too,
                # not just the ticker, and snapshot the exact entry/stop/
                # target like the "Scan timeframes" section below already
                # does. Without this, clicking a result switched ticker but
                # left whatever timeframe/rule was already active driving
                # the Entry/SL/TP lines — showing something the scan never
                # actually picked, or nothing at all.
                st.session_state["fvg_ticker"] = _s["ticker"]
                st.session_state[TF_KEY_BY_CHART["main"]] = "4h"
                st.session_state["_active_scan_pick"] = {
                    "ticker": _s["ticker"], "timeframe": "4h", "direction": _s["direction"],
                    "entry": _s["entry_price"], "stop": _s["sl_price"], "target": _s["tp_price"],
                    "label": f"{_pattern} ({_s['direction']})",
                    "context_timeframe": _scan_context_tf_label,
                    "source_zone": _s["source_zone"],
                    "confluence_entry_details": _s["confluence_entry_details"],
                    "confluence_context_details": _s["confluence_context_details"],
                }
                st.session_state["_scan_pick_locked"] = False
                st.session_state.pop("_active_chart_rule", None)
                st.rerun()

    st.divider()
    if st.button("🔍 Scan timeframes", key="tf_scan_btn", width="stretch",
                 help=f"Scans every context+entry timeframe pair ({ticker}, see recommender.py's own TF_PAIRS "
                      "— e.g. 1W context for a 4h entry, 4h context for a 5m entry) for its own single best "
                      "currently-active setup, ranked by confluence — the higher (context) timeframe's own "
                      "agreement counts more than the entry timeframe's own. Picking a result jumps the chart "
                      "to the ENTRY timeframe and shows that exact entry/stop/target."):
        _tf_pair_tfs = sorted({tf for _pair in TF_PAIRS for tf in _pair})
        _tf_scan_specs = [
            (get_yf_ohlcv, (ticker, TIMEFRAMES[_tf]["period"], TIMEFRAMES[_tf]["fetch_interval"],
                            _prefetch_data_source), {})
            for _tf in _tf_pair_tfs
        ]
        _prefetch(_tf_scan_specs)
        _tf_dfs = {}
        for _tf_label in _tf_pair_tfs:
            _tf_conf = TIMEFRAMES[_tf_label]
            try:
                _tf_df = get_yf_ohlcv(ticker, period=_tf_conf["period"], interval=_tf_conf["fetch_interval"],
                                       provider=_prefetch_data_source)
                if _tf_conf["resample"] and not _tf_df.empty:
                    _tf_df = resample_ohlc(_tf_df, _tf_conf["resample"])
                _tf_dfs[_tf_label] = (_tf_df, _tf_conf["fetch_interval"])
            except Exception:
                _tf_dfs[_tf_label] = (None, None)
        _dfs_by_pair = {}
        for _context_tf, _entry_tf in TF_PAIRS:
            _context_df, _ = _tf_dfs[_context_tf]
            _entry_df, _entry_interval = _tf_dfs[_entry_tf]
            _dfs_by_pair[(_context_tf, _entry_tf)] = (_context_df, _entry_df, _entry_interval)
        _tf_scan_results = scan_timeframes(_dfs_by_pair, ticker, top_n=8)
        st.session_state["_tf_scan"] = _tf_scan_results
        # rebalance_chain: direct request — "rank them by distance...
        # which one is the closest? and so on" — reordered by nearest-
        # price walk instead of confluence rank, starting from current
        # price. Needs a real "right now" price, independent of whichever
        # timeframe happens to be charted — the FINEST timeframe among
        # this scan's own already-fetched dfs is the most current read
        # available without a fresh fetch.
        _finest_tf = min(_tf_pair_tfs, key=lambda tf: _TF_BAR_SECONDS[tf])
        _finest_df, _ = _tf_dfs[_finest_tf]
        if _tf_scan_results and _finest_df is not None and not _finest_df.empty:
            _chain_c_col = "Close" if "Close" in _finest_df else "close"
            _current_price_for_chain = float(_finest_df[_chain_c_col].iloc[-1])
            st.session_state["_tf_scan_chain"] = rebalance_chain(_tf_scan_results, _current_price_for_chain)
        else:
            st.session_state["_tf_scan_chain"] = []

    _tf_results = st.session_state.get("_tf_scan")
    if _tf_results is None:
        st.caption("Not scanned yet this session.")
    elif not _tf_results:
        st.caption("Scanned — nothing currently active on any timeframe pair right now.")
    else:
        for _s in _tf_results:
            _pattern = EVENT_TYPE_LABELS.get(_s["event_type"], _s["event_type"])
            _label = (f"{_s['context_timeframe']}→{_s['timeframe']} · {_pattern} ({_s['direction']}) · "
                      f"confluence {_s['confluence_score']}")
            _tf_jump_clicked = st.button(
                _label, key=f"tfsig_jump_{_s['timeframe']}_{_s['event_type']}_{_s['start']}", width="stretch",
                help=_confluence_breakdown_md(_s, _s["timeframe"], _s["context_timeframe"]))
            if _tf_jump_clicked:
                st.session_state[TF_KEY_BY_CHART["main"]] = _s["timeframe"]
                # A static snapshot (fixed prices, see _active_scan_pick's
                # own read-site comment), NOT a resolve_rule-style rule —
                # this candidate came from one specific zone at scan time;
                # re-resolving "nearest FVG below price" live could land on
                # a DIFFERENT zone entirely if price has since moved,
                # silently swapping what's shown for something the scan
                # never actually picked. Clearing _active_chart_rule keeps
                # the two mechanisms from fighting over the same three
                # lines — see backtest_ui.py's own row-selection code for
                # the mirrored clear in the other direction.
                st.session_state["_active_scan_pick"] = {
                    "ticker": ticker, "timeframe": _s["timeframe"], "direction": _s["direction"],
                    "entry": _s["entry_price"], "stop": _s["sl_price"], "target": _s["tp_price"],
                    "label": f"{_pattern} ({_s['direction']})",
                    "context_timeframe": _s["context_timeframe"],
                    "source_zone": _s["source_zone"],
                    "confluence_entry_details": _s["confluence_entry_details"],
                    "confluence_context_details": _s["confluence_context_details"],
                }
                st.session_state["_scan_pick_locked"] = False
                st.session_state.pop("_active_chart_rule", None)
                st.rerun()

    # Rebalance chain — direct request: rank these same results by
    # DISTANCE instead of confluence, walking from current price to
    # whichever entry is closest, then from THERE to whichever remaining
    # one is closest, and so on. Each entry_price is already the IDEAL
    # (zone-edge) entry, i.e. the price that "hunts"/rebalances that
    # zone — chaining them this way sketches one plausible step-by-step
    # path a market maker's own delivery might take through the levels,
    # not just a flat confluence-ranked list.
    _chain = st.session_state.get("_tf_scan_chain")
    if _chain:
        st.divider()
        st.caption("⛓️ Rebalance chain — same results as above, walked by nearest price instead of "
                   "confluence: closest entry to current price first, then whichever remaining entry "
                   "is closest FROM there, and so on. One guess at the market's own step-by-step path "
                   "through these levels.")
        for _i, _step in enumerate(_chain, start=1):
            _s = _step["setup"]
            _pattern = EVENT_TYPE_LABELS.get(_s["event_type"], _s["event_type"])
            _arrow = "▲" if _step["distance"] > 0 else "▼"
            st.markdown(
                f"**{_i}.** {_arrow} {abs(_step['distance']):,.2f} pts → **{_step['to_price']:,.2f}** "
                f"— {_s['context_timeframe']}→{_s['timeframe']} {_pattern} ({_s['direction']})")
        st.checkbox(
            "Show chain on chart", key="_show_rebalance_chain",
            help="Draws each step above as its own risk/reward box — same green-above-entry/"
                 "red-below-entry shading the live Entry/SL/TP already gets — stacked side by side "
                 "moving forward, numbered in the same order as the list above. Replaces every other "
                 "layer on the chart with just this, same as the lock-trade view.")

# A popover, not an expander — opening it floats the menu over the page
# instead of pushing the chart down (the earlier expander-based version
# did that, same mechanic as the Legend expander at the bottom; moved off
# it once the menu itself grew past a quick glance-and-pick). Click-based
# throughout, not hover — a hover-to-reveal category menu would need
# Streamlit's own internal DOM/CSS, not a stable public API, so it can
# silently break on a version update; a plain click is guaranteed to keep
# working. Confirmed as the wanted tradeoff directly. The button's own
# label always shows the current pick, so which market is loaded stays
# visible even collapsed.
st.session_state.setdefault("fvg_active_category", "Forex")
with st.popover(f"🔍 {_ticker_label(ticker)} · change market", width="stretch"):
    _cat_cols = st.columns(len(TICKER_CATEGORIES))
    for _cat_col, _cat_name in zip(_cat_cols, TICKER_CATEGORIES.keys()):
        with _cat_col:
            _is_active_cat = st.session_state["fvg_active_category"] == _cat_name
            if st.button(_cat_name, key=f"fvg_cat_{_cat_name}", use_container_width=True,
                         type="primary" if _is_active_cat else "secondary"):
                st.session_state["fvg_active_category"] = _cat_name
                st.session_state["fvg_ticker_search"] = ""
                st.rerun()

    _cat_tags = TICKER_CATEGORIES[st.session_state["fvg_active_category"]]
    _cat_symbols = [t for t in TICKER_UNIVERSE if TICKER_INFO[t][1] in _cat_tags]
    _search_query = st.text_input(
        "Search", key="fvg_ticker_search", label_visibility="collapsed",
        placeholder=f"Search {st.session_state['fvg_active_category']} ({len(_cat_symbols)} available)…",
    )
    if _search_query.strip():
        _q = _search_query.strip().upper()
        _matches = [t for t in _cat_symbols if _q in t.upper() or _q in TICKER_INFO[t][0].upper()]
    else:
        # Every category here (Forex/Indices/Commodities) is a small,
        # hand-curated list — show the whole thing immediately, nothing to
        # gain from forcing a search first when there are only a dozen-odd
        # options total.
        _matches = list(_cat_symbols)
    _matches = sorted(_matches)[:40]

    if _matches:
        # Live-checked so a briefly-unavailable symbol shows red rather
        # than silently failing when picked. Parallelized the same way
        # chart-switching's own fetches are (see _prefetch).
        _prefetch([(is_ticker_alive, (t,), {}) for t in _matches])
        for t in _matches:
            _dot = "🟢" if is_ticker_alive(t) else "🔴"
            if st.button(f"{_dot}  {_ticker_label(t)}", key=f"fvg_pick_{t}", use_container_width=True):
                st.session_state["fvg_ticker"] = t
                st.rerun()
    elif _search_query.strip():
        st.caption("No matches.")

# mtf_col used to carry the FULL Layers/Strategy/Backtest/Settings/Charts panel at a
# fixed 3/8 width, always reserved even while a panel was "collapsed" to
# None (that state only ever hid the panel's own content via CSS
# display:none — the column's own width allocation from st.columns is
# fixed regardless of what's inside it, so the chart never actually got
# that space back). Per direct request, that panel is now a genuine
# retracting menu (a popover, opened/closed on demand) instead. Reserving
# even a slim dedicated st.columns() split for its own trigger button was
# STILL wasted space, though — confirmed directly per direct follow-up
# request: that column's own width sits empty to the button's left (the
# button doesn't fill its column), and its FULL remaining height below
# the button (the entire rest of the page) is reserved and empty too,
# since nothing else is ever placed there. Fixed by dropping that column
# split entirely — main_col alone now spans the full width — and instead
# absolutely-positioning the trigger button into the main_col's own
# top-left corner (see the CSS + "_menu_trigger_wrap" container further
# down), which removes it from normal document flow so it reserves NO
# layout space of its own, just floats over whatever's already there.
#
# main_col MUST stay a real st.columns()-produced column (not a plain
# st.container()) — confirmed directly as a real regression: swapping it
# for st.container(key=...) collapsed the whole chart to Streamlit's
# ~150px component fallback height. theme.py's CHART_LAYOUT_CSS (the
# "fullscreen, non-scrolling terminal" flex chain that makes the chart
# fill all remaining vertical space) is entirely built on
# [data-testid="stColumn"] ancestor selectors — a bare st.container()
# renders as stVerticalBlock instead, with no stColumn anywhere above the
# iframe, so every one of those rules silently stopped matching and the
# chart lost its flex-grow-to-fill-height behavior entirely. st.columns(1)
# (a single-column "split") still produces a real stColumn, so it's used
# here purely to keep that CSS chain intact — the "1" isn't a meaningful
# ratio since there's only one column.
main_col = st.columns(1)[0]

with main_col:
    # Invisible marker + :has() (same pattern as .topnav-marker/.sec-chip/
    # .ref-tf-toggle elsewhere in theme.py) — scopes "position: relative"
    # to THIS specific stColumn only, so the menu-trigger button further
    # down (position: absolute; top:0; left:0) anchors to main_col's own
    # top-left corner, not the page's or some other column's. Tried
    # anchoring to a nested st.container(key=...) instead first and
    # confirmed directly that lands the button at the top of whichever
    # block that container wraps — if that's the block containing the
    # chart itself (needed separately so main_col stays a real stColumn;
    # see the comment above main_col's own definition), the button then
    # sits on TOP of the chart's own header instead of beside the TF bar.
    # Anchoring to main_col itself (spanning both the TF bar AND the
    # chart, since both live inside it) puts top:0 at the actual top —
    # the TF bar's own row — regardless of which of main_col's two
    # separate `with` blocks the trigger's markup happens to sit in.
    st.markdown('<span class="main-chart-col-marker"></span>', unsafe_allow_html=True)
    # TF bar + Layers/Settings live in one row, scoped to the chart's own
    # width — not a page-wide row, and no separate sidebar column either
    # (that freed-up width goes straight to the chart below). Unlike
    # TradingView (one chart, no ambiguity), this page has 3 independent
    # panels, so which chart the TF bar edits depends on selected_chart —
    # see the square highlight border on the chart itself.
    #
    # tf_bar_col used to be 5.2 of 5.8 (~90%) — st.columns() gives each
    # column a fixed proportional width regardless of its content's own
    # size, and the TF pill row (width: fit-content, left-aligned) only
    # ever fills a fraction of that, so the icon columns after it inherited
    # whatever was left over as dead space before them. Confirmed directly:
    # a 356px empty gap between the pills' own right edge and the ☰ icon.
    #
    # A 4th, unrendered spacer column soaks up the actual leftover space —
    # confirmed directly that tf_bar_col and the icon columns can't be
    # tuned independently without it: with only 3 columns their ratios are
    # forced to sum to 100% of the row between just the two of them, so
    # shrinking the icon columns to close the gap AFTER them only ever
    # reopened a gap BEFORE them (tf_bar_col's own share grew to fill
    # whatever the icons gave up). Pushing the true remainder into a 4th
    # column lets tf_bar sit close behind the pills AND the two icons sit
    # close behind each other, independently.
    # One permanently-keyed widget rather than swapping `key=` per
    # selected chart — confirmed directly that swapping keys is buggy
    # here: Streamlit/React doesn't repaint the checked pill when the
    # active key changes via an st.rerun() from ANOTHER component's
    # click handler (DOM inspection showed "1m" still checked while the
    # actual session_state value was already "4h", until a real click
    # on the widget forced a resync). Instead the bar's OWN value gets
    # explicitly synced to/from whichever chart currently owns it, so
    # the widget itself never changes identity. Stays OUTSIDE
    # _render_layer_controls (below) on purpose — it drives
    # refresh_interval, which _render_chart's own @st.fragment(run_
    # every=...) decorator needs at OUTER-script scope, so it has to
    # keep triggering a full rerun when changed, unlike the controls.
    st.session_state.setdefault("_tf_bar_owner", selected_chart)
    if st.session_state["_tf_bar_owner"] != selected_chart:
        st.session_state["tf_bar_value"] = st.session_state[TF_KEY_BY_CHART[selected_chart]]
        st.session_state["_tf_bar_owner"] = selected_chart
    else:
        st.session_state.setdefault("tf_bar_value", st.session_state[TF_KEY_BY_CHART[selected_chart]])
    st.radio("TF", list(TIMEFRAMES.keys()), key="tf_bar_value", horizontal=True, label_visibility="collapsed")
    st.session_state[TF_KEY_BY_CHART[selected_chart]] = st.session_state["tf_bar_value"]

    # MA/RSI/MACD/Bollinger Bands all now live in the Layers tab's own
    # "Indicators" section (see _render_layer_controls) — moved off this
    # row and out of Settings per direct request, kept as one group
    # separate from the ICT "Detectors" section rather than merged in
    # among them, since these are plain technical indicators, not ICT
    # pattern detectors. show_indicators/indicator_tf and show_rsi/
    # show_macd/show_bb are read back out of _chart_controls below,
    # alongside everything else that fragment writes.

    # Layers/Settings/Trade Rules used to sit directly in the outer script,
    # meaning EVERY click inside any of them — toggling one layer checkbox,
    # nudging one slider — forced a full top-to-bottom rerun: re-prefetching
    # every ticker's mini-chart data, rebuilding the sidebar, the works.
    # Confirmed directly this was the real cause of "changes take a while
    # and I lose the chart" — _render_chart below was ALREADY its own
    # @st.fragment, but these controls, living outside it, couldn't benefit.
    # A separate fragment (no run_every — it only needs to react to its OWN
    # widgets, not auto-tick) isolates their reruns from the rest of the
    # page. It can't return its computed values as plain Python locals the
    # way a normal function call could, though — a fragment-only rerun
    # doesn't re-execute the surrounding script, so those locals would
    # never update. Writing them into session_state instead, and having
    # _render_chart re-read that dict at the TOP of its own body (see
    # below), is what actually closes the loop: _render_chart is a
    # SEPARATE fragment that keeps auto-ticking every 1-5s regardless of
    # whether these controls just changed, so it picks up the fresh values
    # on its own very next tick even though this rerun never touched it
    # directly — a beat of catch-up latency instead of a full page reload.
    # The panels (Layers, Strategy, Backtest, Settings, Charts) used to be 3 separate
    # top-row popovers plus the mini HTF/LTF charts sitting in their own
    # column — consolidated into one side area you cycle through instead,
    # per direct request. st.tabs with on_change="rerun" is what makes the
    # ◀/▶ buttons work at all: passing key= lets Python read AND WRITE the
    # active tab via session_state (confirmed in the local Streamlit
    # docs — "setting a key lets you read or update the active tab label
    # via st.session_state[key]"), which the default "ignore" mode doesn't
    # support. Every tab's content still renders on every rerun regardless
    # of which one is showing (the docs are explicit that on_change="rerun"
    # only EXPOSES which tab is active via .open, it doesn't skip the
    # others unless you gate on that yourself) — load-bearing here, since
    # _new_cc below reads every widget's value regardless of which tab was
    # last visible, and skipping a tab's own widgets would leave those
    # values stale.
    # Real st.tabs() again, not the TradingView-style collapsed icon rail —
    # per direct follow-up request. The icon rail existed specifically
    # because "collapse the currently-open panel" used to be the ONLY way
    # to give the chart its width back (this whole panel lived in a
    # permanently-reserved column next to it). Now that the panel is a
    # popover/drawer that floats OVER the chart (see the "☰" trigger
    # above) and the chart's own width no longer depends on this panel's
    # state at all, that collapse-to-nothing behavior has nowhere useful
    # left to do its job — closing the drawer itself already does it — so
    # a real 4th state (None, "no tab active") isn't needed and st.tabs()
    # (always exactly one active tab) fits again. Confirmed directly (see
    # the docs excerpt this comment used to quote) that st.tabs() here
    # still computes every tab's own body on every rerun regardless of
    # which one is showing — only the DOM visibility differs — so
    # _new_cc's own unconditional read of every panel's local variables
    # further down needs no change at all.
    @st.fragment
    def _render_layer_controls():
        _prev_cc = st.session_state.get("_chart_controls")
        _tab_layers, _tab_strategy, _tab_survivors, _tab_backtest, _tab_settings, _tab_charts = st.tabs([
            ":material/layers: Layers", ":material/rule: Strategy",
            ":material/emoji_events: Survivors",
            ":material/monitoring: Backtest", ":material/tune: Settings",
            ":material/candlestick_chart: Charts",
        ])
        with _tab_layers, st.container(key="_panel_layers"):
            # "Detectors" (the ICT pattern layers below) and "Indicators"
            # (MA/RSI/MACD/Bollinger, further down) are kept as two visibly
            # separate groups in this one tab per direct request — both
            # used to live scattered (MA next to the TF bar, RSI/MACD/BB in
            # Settings), consolidated here since they're all "what draws on
            # the chart" controls, but NOT merged into one list: a detector
            # (FVG, Order Block, ...) finds this project's own ICT patterns,
            # while an indicator is a plain, well-known technical formula —
            # different kinds of things, kept visually distinct.
            st.markdown("**Detectors**")
            # Sensitivity knobs, tucked into a collapsed expander rather than
            # inline with the per-layer checkboxes below — these are "how
            # strict is a real signal" tuning, not day-to-day on/off choices,
            # and most of the 12 layers share just THREE underlying knobs
            # (detect_swings' own ATR sensitivity alone drives five of them),
            # so one shared block beats one duplicate slider per layer.
            with st.expander(":material/tune: Detector sensitivity (advanced)", expanded=False):
                displacement_ratio = st.slider(
                    "Displacement strictness", min_value=0.1, max_value=0.9, value=0.5, step=0.05,
                    key="fvg_displacement_ratio",
                    help="How much of a candle's own range has to be real body (not wick) before "
                         "it counts as a genuine breakout/displacement move. Higher = stricter, fewer "
                         "zones. Drives FVG, IFVG, Order Blocks, and Breaker Block — all four key off "
                         "this same 'was that a real move' test.")
                _swing_col1, _swing_col2 = st.columns(2)
                with _swing_col1:
                    swing_atr_period = st.number_input(
                        "Swing ATR period", min_value=2, max_value=100, value=14, step=1,
                        key="fvg_swing_atr_period",
                        help="How many candles of typical range to average when deciding what counts "
                             "as a big enough reversal to confirm a swing high/low.")
                with _swing_col2:
                    swing_atr_mult = st.number_input(
                        "Swing reversal size (x ATR)", min_value=0.5, max_value=5.0, value=1.5, step=0.1,
                        key="fvg_swing_atr_mult",
                        help="How many ATRs price has to reverse before a swing point is confirmed. "
                             "Higher = fewer, more significant swings; lower = more, smaller ones. "
                             "This one pair of numbers drives Swing Points, Equal Highs/Lows, Market "
                             "Structure, Premium/Discount, and Liquidity — all five ultimately read "
                             "the same underlying swing detector.")
                session_lookback_days = st.number_input(
                    "Session lookback (days)", min_value=5, max_value=365, value=60, step=5,
                    key="fvg_session_lookback_days",
                    help="How many days back Naked POC and Poor High/Low look when bucketing "
                         "session volume.")
            # One timeframe dropdown per layer, not a fixed 4h/15m pair — each
            # layer detects against exactly the timeframe its own dropdown
            # says, independent of every other layer and of the main chart's
            # own candle timeframe. Defaults to whatever the main chart is
            # showing (see the seeding block above), so "follow the main
            # selector" is the out-of-the-box behavior; picking something else
            # here is a deliberate per-layer override, not a mistake to warn
            # about — there's no more "both toggles off" dead state to fall
            # into, a dropdown always has exactly one value.
            # A few layers carry ONE extra knob beyond the shared sensitivity
            # block above — not shared with any other layer, so it lives
            # right on that layer's own row instead of the expander.
            _PER_LAYER_EXTRA = {"Equal Highs/Lows", "Market Structure", "Liquidity", "Price Projection"}
            layers = []
            layer_tf = {}
            eq_tolerance = 0.0015
            ms_mode = "close"
            liquidity_reaction_window = 5
            price_projection_bars = PRICE_PROJECTION_BARS
            for name in ICT_LAYERS:
                if name in _PER_LAYER_EXTRA:
                    name_col, tf_col, extra_col = st.columns([3, 2, 2])
                else:
                    name_col, tf_col = st.columns([3, 2])
                    extra_col = None
                with name_col:
                    # No value= here — session_state is already seeded (see
                    # setdefault block above), from the persisted last-used
                    # state or LAYER_DEFAULTS, before this widget is created.
                    on = st.checkbox(name, key=f"fvg_layer_{name}", help=LAYER_LAG_HELP.get(name))
                with tf_col:
                    chosen_tf = st.selectbox(f"{name} timeframe", ["Chart TF"] + list(TIMEFRAMES.keys()),
                                              key=f"fvg_tf_{name}", label_visibility="collapsed",
                                              help="'Chart TF' (default) follows whatever timeframe the "
                                                   "main chart itself is showing. Pick a specific one to "
                                                   "detect this layer on its own fixed timeframe instead, "
                                                   "independent of the main chart selector.")
                if extra_col is not None:
                    with extra_col:
                        if name == "Equal Highs/Lows":
                            eq_tolerance = st.number_input(
                                "Tolerance", min_value=0.0002, max_value=0.02, value=0.0015, step=0.0001,
                                format="%.4f", key="fvg_eq_tolerance", label_visibility="collapsed",
                                help="How close two swing prices must be, as a fraction of price, to "
                                     "count as 'equal' highs/lows.")
                        elif name == "Market Structure":
                            ms_mode = st.selectbox(
                                "Confirmation", ["Close", "Wick"], key="fvg_ms_mode",
                                label_visibility="collapsed",
                                help="'Close' (default): a break only counts once a candle CLOSES "
                                     "beyond the level — decisive, fewer false breaks. 'Wick': counts "
                                     "the moment a wick trades beyond it, without waiting for the "
                                     "close — earlier, more of them.").lower()
                        elif name == "Liquidity":
                            liquidity_reaction_window = st.number_input(
                                "Reaction window", min_value=1, max_value=30, value=5, step=1,
                                key="fvg_liquidity_reaction_window", label_visibility="collapsed",
                                help="How many candles after a liquidity sweep an order block can "
                                     "still count as 'the reaction' to it.")
                        elif name == "Price Projection":
                            price_projection_bars = st.number_input(
                                "Bars ahead", min_value=5, max_value=100, value=PRICE_PROJECTION_BARS,
                                step=5, key="fvg_price_projection_bars", label_visibility="collapsed",
                                help="How many bars into the future the projection cone/trend line "
                                     "extends.")
                if on:
                    layers.append(name)
                    layer_tf[name] = chosen_tf
            st.markdown("---")
            # Lives in the Layers tab, not Settings — it's a per-layer cap, and
            # this is where you're already looking when a layer you just
            # enabled is slow or cluttered, not a settings-menu afterthought.
            # For FVG/Order Blocks/Liquidity this is a PER-SIDE count (see
            # _nearest_by_price) — "active areas" means the ones immediately
            # above/below current price, not just whatever was detected most
            # recently in time (which can sit anywhere on a chart with a lot
            # of history loaded). Swings/Equal H-L/Structure don't have a
            # clean above/below-price split the same way and still just cap
            # to the N most recent overall.
            max_items_per_layer = st.number_input(
                "Areas per side", min_value=1, max_value=200, value=5, step=1,
                key="fvg_max_items", help="FVG/Order Blocks/Liquidity: shows the N zones "
                "closest to price on each side (above + below) — the ones actually active "
                "right now, not just recently detected. Swings/Equal H-L/Structure: still "
                "caps to the N most recent overall.",
            )
            st.markdown("---")
            st.markdown("**Indicators**")
            # Plain technical indicators — not ICT-specific, see indicators.py.
            # MA drives the MA+FVG strategy elsewhere on this chart, so it
            # defaults ON; RSI/MACD/Bollinger are just optional reading aids
            # with no strategy behind them here, so they default OFF.
            show_indicators = st.checkbox("MA", value=True, key="fvg_show_indicators",
                                           help="Show/hide two EMA lines on the main chart. Updates the "
                                                "instant a candle closes — no separate confirmation "
                                                "delay the way Swing Points has above — but as a moving "
                                                "average it's a DIFFERENT kind of lag: it's always "
                                                "reacting to price that already happened, smoothed over "
                                                "its own period, not confirming a specific past pivot "
                                                "the way the ICT layers do.")
            _ind_tf_options = ["Chart TF"] + list(TIMEFRAMES.keys())
            indicator_tf = st.selectbox(
                "MA timeframe", _ind_tf_options, index=0, key="fvg_indicator_tf",
                label_visibility="collapsed", disabled=not show_indicators,
                help="Which timeframe the EMA lines are computed on. 'Chart TF' (default) "
                     "matches whatever timeframe the main chart itself is showing. Pick a higher "
                     "one (e.g. viewing 15m candles but computing the average on 1h closes) to "
                     "see a steadier, less noisy line laid over a more detailed chart.",
            )
            _ma_col1, _ma_col2 = st.columns(2)
            with _ma_col1:
                ma_fast_period = st.number_input(
                    "Fast EMA", min_value=2, max_value=200, value=20, step=1,
                    key="fvg_ma_fast_period", disabled=not show_indicators,
                    help="Chart-only — this line is for reading the chart. The MA+FVG strategy's "
                         "own confluence scoring and win-rate stats still use the fixed 20/50 pair "
                         "regardless of what's set here.")
            with _ma_col2:
                ma_slow_period = st.number_input(
                    "Slow EMA", min_value=2, max_value=400, value=50, step=1,
                    key="fvg_ma_slow_period", disabled=not show_indicators,
                    help="Chart-only, same as Fast EMA — doesn't affect the MA+FVG strategy's own "
                         "fixed 20/50 pair.")
            show_rsi = st.checkbox("Show RSI", value=False, key="fvg_show_rsi",
                                    help="Adds an RSI pane below the main chart. No confirmation "
                                         "delay — updates on every closed candle — but as a smoothed "
                                         "oscillator it's a lagging read of momentum that already "
                                         "happened, not a leading signal.")
            rsi_period = st.number_input(
                "RSI period", min_value=2, max_value=100, value=14, step=1,
                key="fvg_rsi_period", disabled=not show_rsi, label_visibility="collapsed",
                help="How many candles RSI averages over. Lower = twitchier, higher = smoother.")
            show_macd = st.checkbox("Show MACD", value=False, key="fvg_show_macd",
                                     help="Adds a MACD pane below the main chart. Same "
                                          "'updates instantly, but reads what already happened' "
                                          "character as RSI above — built from two EMAs, so its own "
                                          "lag is the smoothing kind, not a confirmation delay.")
            _macd_col1, _macd_col2, _macd_col3 = st.columns(3)
            with _macd_col1:
                macd_fast = st.number_input("MACD fast", min_value=2, max_value=100, value=12, step=1,
                                             key="fvg_macd_fast", disabled=not show_macd,
                                             label_visibility="collapsed", help="Fast EMA period.")
            with _macd_col2:
                macd_slow = st.number_input("MACD slow", min_value=2, max_value=200, value=26, step=1,
                                             key="fvg_macd_slow", disabled=not show_macd,
                                             label_visibility="collapsed", help="Slow EMA period.")
            with _macd_col3:
                macd_signal = st.number_input("MACD signal", min_value=2, max_value=100, value=9, step=1,
                                               key="fvg_macd_signal", disabled=not show_macd,
                                               label_visibility="collapsed",
                                               help="Signal-line smoothing period.")
            show_bb = st.checkbox("Show Bollinger Bands", value=False, key="fvg_show_bb",
                                   help="Adds Bollinger Bands over the candles on the main chart. "
                                        "Same smoothing-lag character as RSI/MACD above — the basis "
                                        "line is a moving average.")
            _bb_col1, _bb_col2 = st.columns(2)
            with _bb_col1:
                bb_period = st.number_input("BB period", min_value=2, max_value=200, value=20, step=1,
                                             key="fvg_bb_period", disabled=not show_bb,
                                             label_visibility="collapsed",
                                             help="How many candles the basis (middle) line averages.")
            with _bb_col2:
                bb_std = st.number_input("BB std dev", min_value=0.5, max_value=5.0, value=2.0, step=0.1,
                                          key="fvg_bb_std", disabled=not show_bb,
                                          label_visibility="collapsed",
                                          help="How many standard deviations the upper/lower bands sit "
                                               "from the basis line. Higher = wider bands.")
            show_atr = st.checkbox("Show ATR", value=False, key="fvg_show_atr",
                                    help="Adds an ATR pane below the main chart — how many "
                                         "price units (not a percentage) this ticker has typically "
                                         "moved per candle lately. Rising = volatility expanding, "
                                         "falling = contracting. Useful for sizing a stop to the "
                                         "market's own current noise level instead of a fixed "
                                         "number, and for judging whether a big-looking candle was "
                                         "actually unusual or just normal for this ticker right now. "
                                         "Same smoothing-lag character as RSI/MACD/BB above, not a "
                                         "confirmation delay — it's reacting to recent moves, not "
                                         "waiting to confirm a specific one.")
            atr_period_ind = st.number_input(
                "ATR period", min_value=2, max_value=100, value=14, step=1,
                key="fvg_atr_period_ind", disabled=not show_atr, label_visibility="collapsed",
                help="How many candles ATR averages over.")
            # Volume used to be permanently-on base chart furniture (no toggle) —
            # back to user-controlled, and defaulting off this time, on all three
            # panels (main + both mini charts) sharing this one setting rather
            # than three separate checkboxes for what's really one preference.
            # Lives here next to Volume Profile (what draws on the chart), not
            # in Settings (how it's configured) — moved per direct request.
            show_volume = st.checkbox("Show volume", value=False, key="fvg_show_volume")
            show_log_scale = st.checkbox(
                "Log scale", value=False, key="fvg_log_scale",
                help="Equal on-screen distance means equal PERCENT move instead of equal dollar move — "
                     "a real $50k Bitcoin range and a $2k one look proportionally the same, instead of "
                     "the small move getting flattened to nothing next to the big one.")
            show_volume_profile = st.checkbox(
                "Volume Profile", value=False, key="fvg_show_volume_profile",
                help="How much volume traded at each PRICE level (not each candle) over this "
                     "chart's own loaded history — drawn as horizontal bars growing from the left "
                     "edge, brightest at the busiest price (the Point of Control, dashed line), "
                     "with a second dashed band marking the Value Area (the tightest price range "
                     "holding 70% of that volume). Approximated from each candle's own high/low/"
                     "volume, not individual trades — real trade-level volume only exists for "
                     "Binance crypto pairs here (see the Footprint chart), and Yahoo Forex tickers "
                     "report no real volume at all, so this stays off automatically wherever there's "
                     "nothing honest to compute it from.")
            _vp_anchor_options = ["Full history", "Today", "This week", "This month", "Custom date"]
            vp_anchor = st.selectbox(
                "Volume Profile anchor", _vp_anchor_options, index=0, key="fvg_vp_anchor",
                label_visibility="collapsed", disabled=not show_volume_profile,
                help="Where the Volume Profile starts counting from. 'Full history' (default) uses "
                     "everything currently loaded for this timeframe — honest, but on a long window "
                     "(months of history) the busiest price can sit far from where price is trading "
                     "today. Anchoring to 'Today'/'This week'/'This month' — or a specific 'Custom "
                     "date' — recomputes it from just that point forward instead, the same idea as "
                     "an Anchored Volume Profile on other platforms.")
            vp_anchor_date = None
            if vp_anchor == "Custom date":
                vp_anchor_date = st.date_input(
                    "Volume Profile anchor date", value=None, key="fvg_vp_anchor_date",
                    label_visibility="collapsed",
                    help="Recomputes the Volume Profile using only bars from this date forward.")
        with _tab_settings, st.container(key="_panel_settings"):
            selected_kill_zones = st.multiselect("Kill zones", list(KILL_ZONES.keys()),
                                                  default=["NY AM (9:30–11:30 ET)"], key="fvg_kill_zones")
            show_mitigated = st.checkbox("Show mitigated/filled areas", value=False, key="fvg_show_mitigated")
            zone_opacity = st.slider("Zone opacity", 0.05, 0.6, 0.25, 0.05, key="fvg_zone_opacity")
            # "Auto" is Yahoo-first-with-fallback (unaffected by this control at
            # all). Picking a specific provider forces every panel + reference
            # frame onto just that one, with no fallback — e.g. "Binance" on a
            # non-crypto ticker will show no data rather than quietly serving
            # Yahoo's instead, since the whole point here is to see exactly what
            # one source gives you.
            data_source_label = st.selectbox("Data source", list(DATA_SOURCES.keys()),
                                              index=0, key="fvg_data_source")
            data_source = DATA_SOURCES[data_source_label]
            # A second timeframe's candles, faded, drawn on the main chart's own
            # axis (see the ghost_candles block below _render_chart). Off by
            # default per direct request; pick a timeframe here to turn it
            # on. Picking whatever the main chart is already showing is a
            # no-op (nothing to overlay on itself).
            _overlay_tf_options = ["None"] + list(TIMEFRAMES.keys())
            overlay_tf = st.selectbox("Overlay timeframe", _overlay_tf_options,
                                       index=_overlay_tf_options.index("None"), key="fvg_overlay_tf")
            # "Max items per layer" moved to the Layers (☰) tab — same control,
            # just living where it's actually relevant instead of a settings
            # afterthought. See its definition there for the full rationale.
            st.markdown("---")
            # Controls the historical win-rate stats (FVG/OB/MA+FVG labels and
            # the detail table below the chart) — see recommender.py's
            # fvg_event_win_rate & friends. A past trade closes when the NEXT
            # one of these forms, not after a fixed bar count — "let me choose"
            # what counts as the exit signal, instead of an arbitrary hold
            # length picked in advance.
            st.caption("Win rate")
            _win_rate_exit_labels = st.multiselect(
                "Exit trigger", list(EXIT_EVENT_LABELS.keys()), default=list(EXIT_EVENT_LABELS.keys()),
                format_func=lambda k: EXIT_EVENT_LABELS[k], key="fvg_win_rate_exit_types",
                help="How the win-rate numbers decide a past trade is 'done': as soon as the next one of "
                     "these forms after entry, that's the exit point, and whether price is above or below "
                     "the entry price at that moment decides win or loss. All three checked by default, so "
                     "whichever comes first ends the trade. No fixed number of candles held either way.",
            )
            _win_rate_exit_types = tuple(_win_rate_exit_labels) or tuple(EXIT_EVENT_LABELS.keys())
            _use_news_blackout = st.checkbox(
                "Exclude high-impact news windows", value=False, key="fvg_use_news_blackout",
                help="Drops any past trade the win-rate/backtest numbers above would otherwise count if it "
                     "opened on a bar sitting inside a red-folder news window for this ticker's own "
                     "currencies (e.g. USD news for GBPUSD=X) — real trades wouldn't take that entry either, "
                     "spread and slippage blow out right around the release. Coverage: this week's real "
                     "ForexFactory calendar, everything this app has itself observed live over time (grows "
                     "automatically, starts empty), and US Non-Farm Payrolls (first Friday of the month, "
                     "exact for any date). Other recurring events (FOMC, CPI) are only covered for weeks "
                     "this app was actually running to see them live.")
            if _use_news_blackout:
                _nbc1, _nbc2 = st.columns(2)
                with _nbc1:
                    _news_blackout_before = st.selectbox("Minutes before", [5, 10, 15, 30], index=2,
                                                          key="fvg_news_blackout_before")
                with _nbc2:
                    _news_blackout_after = st.selectbox("Minutes after", [5, 10, 15, 30], index=2,
                                                         key="fvg_news_blackout_after")
                _news_blackout = (_news_blackout_before, _news_blackout_after)
            else:
                _news_blackout = None
            _confluence_keys = st.multiselect(
                "Indicator confluence", list(INDICATOR_SPECS.keys()), default=[],
                format_func=lambda k: INDICATOR_SPECS[k]["label"], key="fvg_confluence_indicators",
                help="Highlights zones/levels where one of these ALSO lines up, and shows a separate win "
                     "rate for that combination. "
                     "EMA 20 / EMA 50: the average closing price of the last 20 or 50 candles — a moving "
                     "line that traces the recent trend. A match means that line is sitting right inside "
                     "the zone. "
                     "RSI: a 0-100 gauge of how far and how fast price has moved recently. Under 30 means "
                     "it's dropped hard and may be due to bounce back up (oversold); over 70 means it's "
                     "risen hard and may be due to pull back (overbought). A match means RSI is reading "
                     "oversold under a bullish zone, or overbought under a bearish one. "
                     "MACD: compares a fast and a slow moving average to gauge whether momentum currently "
                     "favors buyers or sellers. A match means that momentum reading agrees with the zone's "
                     "own direction.",
            )
        with _tab_strategy, st.container(key="_panel_strategy"):
            # Sweep IS the rule builder now — a single rule is just a sweep
            # with every range narrowed to one value (same engine, same
            # results table), so there's exactly one way to build/test a
            # rule instead of a separate hand-built-widget tool plus a
            # sweep tool sitting next to each other. Pick any result row
            # below and "Use this rule on the chart" to drive Entry/Stop/
            # Target and the Backtest tab next door — merged in per
            # direct request. backtest_ui.py owns the actual render logic,
            # shared with Crypto's own Strategy tab — same backtest_results.db
            # underneath regardless of which of the two called it.
            st.caption(
                "Build a rule as either 'current price' or the Nth-nearest FVG, Order Block, or Liquidity "
                "level above or below it. Test one exact combo (narrow every range below to a single value) "
                "or a whole grid of variations at once — then pick any result to drive the chart's Entry/"
                "Stop/Target and see its full trade history in the Backtest tab next door."
            )
            _sweep_tab, _heatmap_tab = st.tabs(["Sweep", "Heatmap"])
            with _sweep_tab:
                backtest_ui.render_sweep_tab(TICKER_INFO)
            with _heatmap_tab:
                backtest_ui.render_heatmap_tab()

        with _tab_survivors, st.container(key="_panel_survivors"):
            # Everything that's ever cleared a REAL statistical bar
            # (survived BH-correction across the whole accumulated
            # experiments.py trial log, held up on untouched holdout
            # data) — see ticker_behavior.py/survivors_ui.py. Distinct
            # from the Strategy tab's own sweep above: that engine builds
            # generic anchor-based rules on demand, this browses what's
            # already been proven and lets you jump straight to it.
            survivors_ui.render_survivors_tab(
                ticker, TF_KEY_BY_CHART["main"], TIMEFRAMES, TICKER_INFO, data_source)

        with _tab_backtest, st.container(key="_panel_backtest"):
            # The currently active rule — whatever result row was last sent
            # here via "Use this rule on the chart" (backtest_ui.py writes
            # this exact key). None until a first selection is made, same
            # honest empty state the old flow had before its own first
            # "Run backtest" click.
            _active = st.session_state.get("_active_chart_rule")
            if _active is None:
                _rule_direction = "Bullish"
                _entry_rule = _exit_rule = _stop_rule = {"anchor": "current_price"}
            else:
                _rule_direction = "Bullish" if _active["direction"] == "bullish" else "Bearish"
                _entry_rule, _exit_rule, _stop_rule = _active["entry_rule"], _active["exit_rule"], _active["stop_rule"]

            # Resolved right here, using its own cheap df fetch (a cache
            # hit off the main chart's own identical fetch, not a real new
            # request) — NOT read back from a value the main chart's OWN
            # fragment computed on its own separate schedule. That cross-
            # fragment version lagged behind whatever rule was actually
            # just picked until the main chart's next tick, which could be
            # seconds away or, on a non-auto-ticking timeframe (1D/1W/1M/
            # 1Y), never — confirmed directly as the cause of "R:R doesn't
            # update / looks wrong after switching a rule."
            _rr_tf_label = st.session_state.get(TF_KEY_BY_CHART["main"], DEFAULT_TF_BY_CHART["main"])
            _rr_conf = TIMEFRAMES[_rr_tf_label]
            _rr_df = get_yf_ohlcv(ticker, period=_rr_conf["period"], interval=_rr_conf["fetch_interval"],
                                   provider=data_source)
            if _rr_conf["resample"] and not _rr_df.empty:
                _rr_df = resample_ohlc(_rr_df, _rr_conf["resample"])

            with st.container(border=True):
                st.markdown("**Backtest**")
                if _active is None:
                    st.caption("Select a result in the Strategy tab's Sweep and click \"Use this rule on "
                               "the chart\" — nothing active yet.")
                else:
                    st.caption(
                        f"Active rule (found sweeping {_active['ticker']} / {_active['timeframe']}) — "
                        f"**{_rule_direction}**: Entry {rule_describe(_entry_rule)} · "
                        f"Exit {rule_describe(_exit_rule)} · Stop {rule_describe(_stop_rule)}"
                    )
                    st.caption(
                        "Simulates this EXACT rule against THIS chart's own history — whatever ticker/"
                        "timeframe is on screen right now, not necessarily where it was originally found — "
                        "bar by bar, resolving Entry/Exit/Stop using only data that existed at that point "
                        "(no lookahead), one trade at a time, held until price touches Stop or Target. Raw "
                        "price move only, no spread/slippage/fees deducted unless a round-trip cost was set "
                        "in the sweep's own Quality filters."
                    )
                    if _rr_df.empty:
                        st.caption("No data loaded for this chart yet.")
                    else:
                        _rr_close_col = _rr_df["Close"] if "Close" in _rr_df else _rr_df["close"]
                        _rr_price = float(_rr_close_col.iloc[-1])
                        _rr_entry = resolve_rule(_entry_rule, _rr_df, _rr_price)
                        _rr_exit = resolve_rule(_exit_rule, _rr_df, _rr_price)
                        _rr_stop = resolve_rule(_stop_rule, _rr_df, _rr_price)
                        if _rr_entry is not None and _rr_exit is not None and _rr_stop is not None:
                            _rr_risk = abs(_rr_entry - _rr_stop)
                            _rr_reward = abs(_rr_exit - _rr_entry)
                            if _rr_risk > 0:
                                st.caption(f"R:R — 1 : {_rr_reward / _rr_risk:.2g}  "
                                           f"(risk {_rr_risk:.5g}, reward {_rr_reward:.5g})")
                            else:
                                st.caption("R:R — stop is at entry, can't divide by zero risk.")
                        else:
                            st.caption("R:R — not available until Entry, Stop, and Target all resolve to a "
                                       "price on this chart.")
                        # Not a backtest of this exact rule (that would need
                        # to walk every past FVG/OB/Liquidity zone that
                        # matched this rule's own side/nth/point and
                        # simulate forward to TP/SL — real future work) —
                        # this is the underlying zone TYPE's own historical
                        # win rate, the same event-driven number already
                        # shown elsewhere on the chart for FVG/OB/
                        # Liquidity. Close enough for "does this kind of
                        # setup tend to work" without pretending it's more
                        # precise than it is. Only meaningful for a
                        # detector-based Entry — "at current price" has no
                        # zone type to look a win rate up for.
                        if _entry_rule.get("anchor") == "next_event":
                            _rule_wr_layer = RULE_DETECTOR_LABELS[_entry_rule["detector"]]
                            _rule_wr_direction = "bullish" if _rule_direction == "Bullish" else "bearish"
                            _rule_wrs = _historical_win_rates(ticker, _rr_tf_label, _rule_wr_layer, data_source,
                                                               exit_types=_win_rate_exit_types,
                                                               news_blackout=_news_blackout)
                            _rule_wr_label = _win_rate_label(_rule_wrs.get(_rule_wr_direction))
                            _rule_wr_exit_names = "/".join(EXIT_EVENT_LABELS[k] for k in _win_rate_exit_types) \
                                or "any of the exit triggers"
                            _rule_wr_help = (
                                f"Looks back through this ticker's own history for every past {_rule_wr_layer} "
                                f"that formed in the {_rule_wr_direction.lower()} direction — each one counts "
                                f"as one trial. A trial is a WIN if price was still on the favorable side of "
                                f"where the zone formed by the time the next {_rule_wr_exit_names} appeared "
                                f"(whichever forms first ends the trial) — a LOSS if price had already moved "
                                f"the other way by then. Not a fixed number of candles held, and not a "
                                f"simulation of this exact Entry/Exit/Stop rule — just 'when a {_rule_wr_layer} "
                                f"like this formed before, did price tend to keep going the right way.' Change "
                                f"which events count as the exit trigger in the Settings tab's own 'Win rate' "
                                f"section."
                            )
                            if _rule_wr_label:
                                st.caption(f"{_rule_wr_layer} {_rule_direction.lower()} win rate — "
                                           f"{_rule_wr_label} (the zone type, not this exact rule)",
                                           help=_rule_wr_help)
                            else:
                                st.caption(f"{_rule_wr_layer} {_rule_direction.lower()} win rate — no "
                                           f"qualifying history found", help=_rule_wr_help)
                            # FVG-only for now — Order Block's own detector
                            # doesn't keep the zone's PRE-encroachment size
                            # the way FVG's raw_top/raw_bottom does, so this
                            # can't be computed the same way for it yet
                            # without adding that first.
                            if _entry_rule["detector"] == "fvg":
                                _rr_wr_conf = TIMEFRAMES[_rr_tf_label]
                                _runup_period = INTERVAL_MAX_PERIOD.get(_rr_wr_conf["fetch_interval"], "60d")
                                _runup_df = load_history(ticker, _runup_period, _rr_wr_conf["fetch_interval"],
                                                          provider=data_source)
                                if _rr_wr_conf["resample"] and not _runup_df.empty:
                                    _runup_df = resample_ohlc(_runup_df, _rr_wr_conf["resample"])
                                _runup_stats = fvg_run_up_stats(_runup_df) if not _runup_df.empty else {}
                                _runup = _runup_stats.get(_rule_wr_direction)
                                _runup_help = (
                                    f"For every past bullish/bearish FVG that eventually got touched again, "
                                    f"this measures the best price reached between when the gap FIRST formed "
                                    f"and that touch — the run a trader would have caught by entering right "
                                    f"at formation and holding until price came back. A bullish FVG is "
                                    f"measured from its own top edge (where price already was when the gap "
                                    f"printed) to the highest high reached before the retest; a bearish FVG "
                                    f"mirrors this from its bottom edge to the lowest low. Zones still open "
                                    f"(never touched again yet) aren't counted — their run is still ongoing, "
                                    f"so including it would understate what a resolved trade actually "
                                    f"captured."
                                )
                                if _runup and _runup["n"] > 0:
                                    _runup_n_note = "" if _runup["sufficient"] else " — low data"
                                    st.caption(f"Avg. run-up before retest — {_runup['avg_pct']:.2g}% "
                                               f"(median {_runup['median_pct']:.2g}%, n={_runup['n']}"
                                               f"{_runup_n_note})", help=_runup_help)
                                else:
                                    st.caption("Avg. run-up before retest — no qualifying history found",
                                               help=_runup_help)

                        # Cached on (ticker, timeframe, rule) — NOT recomputed
                        # on every rerun. backtest_custom_rule walks the
                        # chart's own history bar by bar (see its own
                        # docstring on why that's not O(n) — full zone
                        # redetection on a growing slice, every bar); on a
                        # real 730-day/4h BTC-USD combo this measured 11
                        # SECONDS. This whole block reruns on ANY interaction
                        # anywhere in Layers/Strategy/Backtest/Settings/Charts (one
                        # shared fragment) — recomputing unconditionally
                        # meant clicking an unrelated Layers checkbox paid
                        # that same 11s, every time, confirmed directly as
                        # the actual mechanism behind "the trades don't show
                        # up" reports (nothing was broken; a still-running
                        # recompute just looks identical to a stuck one with
                        # no spinner). Still written into the SAME session-
                        # state shape _render_chart already reads for its
                        # trade-box overlay, so that fragment needs no
                        # changes at all — only what feeds it, here, changed.
                        _bt_key = f"{ticker}|{_rr_tf_label}|{_entry_rule}|{_exit_rule}|{_stop_rule}"
                        _bt_cached = st.session_state.get("_backtest_result")
                        if _bt_cached is not None and _bt_cached["key"] == _bt_key:
                            _bt_result = _bt_cached["result"]
                        else:
                            with st.spinner("Running backtest — walking the chart's history bar by bar…"):
                                _bt_result = backtest_custom_rule(_rr_df, _entry_rule, _exit_rule, _stop_rule)
                            st.session_state["_backtest_result"] = {"key": _bt_key, "result": _bt_result}

                        if _bt_result["n"] == 0:
                            st.caption("No trades triggered over this chart's own history — try a different "
                                       "rule, a different ticker/timeframe, or a longer period on the main "
                                       "chart.")
                        else:
                            _bt = _bt_result
                            st.markdown(f"**{_bt['n']} trades · {_bt['win_rate'] * 100:.0f}% win rate** "
                                        f"({_bt['wins']}W / {_bt['n'] - _bt['wins']}L)")
                            _trades_df = pd.DataFrame(_bt["trades"])
                            _trades_df["result"] = _trades_df["won"].map({True: "Win", False: "Loss"})
                            st.dataframe(_trades_df[["entry_time", "exit_time", "bars_held", "result"]]
                                         .rename(columns={"entry_time": "Entry", "exit_time": "Exit",
                                                           "bars_held": "Bars held", "result": "Result"}),
                                         width="stretch", hide_index=True)

                            st.checkbox(
                                "Show these trades on the chart", key="_bt_visualize_trades",
                                help="Draws each trade above as a risk/reward box over its own entry-to-exit "
                                     "span on the main chart — shaded = risk (red, toward Stop) / reward "
                                     "(green, toward Target) as the rule defined it, bright outline = "
                                     "whichever side actually got hit. Turns off automatically if you change "
                                     "the rule, ticker, or timeframe.")
                            # Written by _render_chart (a separate, auto-
                            # ticking fragment) each time it redraws —
                            # visible proof of what actually happened
                            # instead of a silent yes/no, since "nothing
                            # showed up" alone can't tell you whether the
                            # rule/ticker/timeframe just didn't match, or
                            # trades matched but some fell outside the
                            # chart's own loaded range (see the off_axis
                            # count below).
                            _viz_status = st.session_state.get("_bt_viz_status")
                            if _viz_status and _viz_status.get("checked"):
                                if not _viz_status.get("matched"):
                                    st.caption("⏳ Waiting for the chart to catch up to this rule/ticker/"
                                               "timeframe…")
                                elif _viz_status.get("off_axis"):
                                    st.caption(f"Drew {_viz_status['drawn']} of {_viz_status['total']} "
                                               f"trades — {_viz_status['off_axis']} fell before the chart's "
                                               f"own loaded history and can't be drawn (their entry predates "
                                               f"the earliest bar this chart currently has loaded).")
                                else:
                                    st.caption(f"Drew all {_viz_status['drawn']} trades on the chart.")

                            # R-multiples, not dollars — this app talks in
                            # R:R everywhere else, so a win is +reward/risk
                            # and a loss is a flat -1R (the definition of
                            # "lost," regardless of this rule's own R:R)
                            # rather than inventing an arbitrary stake/
                            # starting balance nothing else here uses.
                            _risk = (_trades_df["entry"] - _trades_df["stop"]).abs()
                            _reward = (_trades_df["target"] - _trades_df["entry"]).abs()
                            _r_multiple = (_reward / _risk).where(_trades_df["won"], -1.0)
                            _cum_r = _r_multiple.cumsum()
                            st.caption(f"Cumulative R — ended at {_cum_r.iloc[-1]:+.1f}R over {_bt['n']} trades")
                            st.line_chart(pd.DataFrame({"Cumulative R": _cum_r.values}), width="stretch")

        with _tab_charts, st.container(key="_panel_charts"):
            # The HTF/LTF mini charts used to sit in their own dedicated
            # column next to the main chart, always visible — now the 4th
            # cyclable panel instead. Each is still its own independently
            # auto-ticking fragment (see _render_mini_chart), so switching
            # away and back doesn't lose their own live-refresh cadence.
            _render_mini_chart(ticker, "htf")
            _render_mini_chart(ticker, "ltf")

        # Written once, at the end, after every widget above has had its
        # chance to update it this rerun — _render_chart (a separate,
        # auto-ticking fragment) reads this back out at the top of its own
        # body rather than relying on closure-captured locals from the
        # outer script, which is what actually lets it pick up a change
        # made in here without the outer script — and this fragment's own
        # popovers — needing to rerun at all.
        _new_cc = {
            "layers": layers, "layer_tf": layer_tf, "max_items_per_layer": max_items_per_layer,
            "selected_kill_zones": selected_kill_zones, "show_mitigated": show_mitigated,
            "show_volume": show_volume, "show_rsi": show_rsi, "show_macd": show_macd, "show_bb": show_bb,
            "show_atr": show_atr, "log_scale": show_log_scale,
            "show_ma": show_indicators, "ma_tf": indicator_tf, "show_volume_profile": show_volume_profile,
            "vp_anchor": vp_anchor, "vp_anchor_date": vp_anchor_date,
            "zone_opacity": zone_opacity, "data_source": data_source,
            "overlay_tf": overlay_tf, "win_rate_exit_types": _win_rate_exit_types,
            "news_blackout": _news_blackout,
            "confluence_keys": _confluence_keys,
            "entry_rule": _entry_rule, "exit_rule": _exit_rule, "stop_rule": _stop_rule,
            # Feature A — adjustable indicator periods (all chart-only, see
            # each control's own help text on why they don't touch MA_FVG's
            # own fixed 20/50 strategy pair).
            "ma_fast_period": ma_fast_period, "ma_slow_period": ma_slow_period,
            "rsi_period": rsi_period, "macd_fast": macd_fast, "macd_slow": macd_slow,
            "macd_signal": macd_signal, "bb_period": bb_period, "bb_std": bb_std,
            "atr_period_ind": atr_period_ind,
            # Feature B — detector sensitivity (shared blocks + the few
            # per-layer knobs), read back the same way as every other
            # chart-affecting control here.
            "displacement_ratio": displacement_ratio, "swing_atr_period": swing_atr_period,
            "swing_atr_mult": swing_atr_mult, "session_lookback_days": session_lookback_days,
            "eq_tolerance": eq_tolerance, "ms_mode": ms_mode,
            "liquidity_reaction_window": liquidity_reaction_window,
            "price_projection_bars": price_projection_bars,
            # Not a chart-drawing input itself, but its own toggle needs to
            # reach _render_chart the same forced way a rule change does on
            # 1D/1W/1M/1Y (see the comment just below) — omitting it here
            # was a real bug: on those timeframes, checking "Show these
            # trades on the chart" did nothing until some UNRELATED control
            # happened to change too, since this dict never reflected the
            # checkbox's own state changing.
            "visualize_trades": st.session_state.get("_bt_visualize_trades", False),
        }
        st.session_state["_chart_controls"] = _new_cc
        # 1D/1W/1M/1Y have no auto-tick to catch up on later, so a change
        # made in here needs a full rerun to actually reach the chart. Only
        # when something in here actually changed, though — reacting to
        # the timeframe alone (with no comparison against the previous
        # value) reruns this same fragment as part of every full rerun it
        # triggers, which re-hits this same unconditional branch again —
        # confirmed directly: an infinite rerun loop pegging the process
        # at ~100% CPU any time the main chart sits on 1D/1W/1M/1Y, since
        # that condition never stops being true on its own.
        #
        # On an auto-ticking timeframe, picking a new sweep result (or
        # toggling "Show these trades on the chart") is the one interaction
        # worth forcing immediately too — _render_chart lives in its own
        # SEPARATELY auto-ticking fragment, so without this the chart only
        # caught up on that fragment's own next tick (up to
        # refresh_interval seconds away), which read as "nothing happened"
        # right after a selection that clearly worked. Narrowly scoped to
        # just these fields (not the broader _new_cc != _prev_cc used for
        # the 1D+ case) so an unrelated Layers/Settings tweak on a live
        # timeframe keeps its existing cheap fragment-only rerun instead of
        # paying for a full-page one every time. Tried forcing this rerun
        # from INSIDE backtest_ui.py's own row-selection handler instead
        # (mid-fragment, before this same fragment run ever reached the
        # checkbox below) and confirmed directly that desyncs the
        # checkbox's own frontend widget — this end-of-fragment call site,
        # AFTER everything above has already rendered once normally, is the
        # same one the 1D+ case already relies on safely.
        _rule_or_viz_changed = _prev_cc is not None and any(
            _new_cc.get(_k) != _prev_cc.get(_k)
            for _k in ("entry_rule", "exit_rule", "stop_rule", "visualize_trades")
        )
        if _new_cc != _prev_cc and (
            _rule_or_viz_changed or CANDLE_SECONDS.get(st.session_state.get(TF_KEY_BY_CHART["main"])) is None
        ):
            st.rerun()

    with main_col:
        # A genuine retracting menu — opens as a floating panel over the
        # chart instead of a permanently-reserved column. Same content as
        # before (the icon-row-driven Layers/Strategy/Backtest/Settings/Charts cycle
        # inside _render_layer_controls, untouched), just no longer nailed
        # open. `_render_layer_controls` still calls st.rerun() in a
        # couple of places (the 1D+ forced-catch-up, and the sweep-
        # selection one) — confirmed directly those don't close this
        # popover: Streamlit tracks a popover's open/closed state the same
        # way it tracks any other widget's value, surviving a full-app
        # rerun exactly like the checkbox/dataframe selections inside it
        # already did.
        #
        # The trigger button itself is absolutely-positioned into
        # main_col's own top-left corner (a plain st.container(key=...)
        # doesn't reserve any layout space once its CSS position is
        # "absolute" — it's pulled clean out of the normal document flow,
        # floating over whatever would otherwise render there, confirmed
        # directly this leaves neither a horizontal gap beside it nor a
        # reserved column beneath it the way a dedicated st.columns()
        # split did before). main_col itself needs `position: relative`
        # so "top: 0; left: 0" resolves against ITS OWN box (the whole
        # chart column) rather than the page/viewport. Left, not right, so
        # this menu trigger and the Signals sidebar's own expand chevron
        # (now flipped to the right, see theme.py's stSidebar/
        # stExpandSidebarButton rules) don't compete for the same corner.
        #
        # st.popover's own default look is a small card anchored right
        # under its trigger button — per earlier direct request, restyled
        # into an actual left-edge drawer instead: full viewport height,
        # pinned to the left edge, sized to what its content actually
        # needs instead of a cramped fixed popover box. st.popover has no
        # built-in way to ask for this shape, so this overrides its
        # floating-ui-computed inline position/size via a CSS rule with
        # !important (confirmed directly: !important in a stylesheet DOES
        # beat a plain, non-!important inline style, which is exactly what
        # Streamlit sets here). Scoped to THIS popover only, via its own
        # aria-label (confirmed directly: Streamlit stamps the popover's
        # own `label` argument onto its floating body as aria-label) — the
        # ticker-search popover elsewhere on this page uses a completely
        # different label, so it's untouched by this rule. Label is the
        # plain "☰" glyph, not a ":material/xxx:" shortcode string —
        # confirmed directly that embedding a shortcode like
        # ":material/tune:" inside a <style> block gets it silently
        # mangled (the "/" rewritten to "_") by Streamlit's own markdown
        # icon-shortcode processing, which scans st.markdown bodies
        # generally, not just button labels — so a selector built from the
        # literal shortcode text never actually matched the real
        # (unmangled) aria-label on the live element.
        st.markdown(
            """<style>
            [data-testid="stColumn"]:has(> [data-testid="stElementContainer"] .main-chart-col-marker) {
                position: relative;
            }
            .st-key-_menu_trigger_wrap {
                position: absolute !important;
                top: 0 !important;
                left: 0 !important;
                z-index: 100 !important;
                width: auto !important;
                /* Flexbox stretches an abspos flex item to the full cross
                size of its original flex row unless told otherwise -
                confirmed directly: this box's own rendered height was
                991px (main_col's FULL height) versus its 40px content,
                turning the entire right edge of the page into an
                invisible hit-area that ate clicks meant for the chart's
                own Fit button underneath. height:auto alone doesn't
                override flex stretch sizing; align-self does. */
                height: auto !important;
                align-self: flex-start !important;
                /* Belt-and-suspenders: even if some future change makes
                this box tall again, clicks should fall through empty
                space to whatever's under it rather than getting eaten. */
                pointer-events: none !important;
            }
            .st-key-_menu_trigger_wrap * {
                pointer-events: auto !important;
            }
            /* _render_layer_controls (what this popover opens into) is a
               real @st.fragment, but its own content is expensive enough
               that opening the drawer still measures ~1.7s from click to
               the popover body actually mounting (confirmed directly via
               performance.now() around a synthetic click) — a real
               backend-render cost, not something this CSS file can fix.
               Without any visual feedback in that window, a click reads
               as "did nothing," and a second, impatient click during the
               same window toggles the popover shut again before it ever
               finished opening (confirmed as the mechanism behind a
               direct report of "click too fast, doesn't open"). This is
               a stopgap, not a fix for the underlying latency: an instant
               pressed-state so the click itself is never in doubt, even
               while the drawer is still on its way. */
            .st-key-_menu_trigger_wrap button:active {
                background: rgba(255,255,255,0.15) !important;
            }
            div[data-testid="stPopoverBody"][aria-label="☰"] {
                position: fixed !important;
                inset: 0 auto 0 0 !important;
                transform: none !important;
                height: 100vh !important;
                max-height: 100vh !important;
                /* 560px default (up from 460px, "extend it more"), further
                   adjustable via the drag handle below. min/max keep a
                   manual drag from collapsing it unusably small or past
                   the viewport. */
                width: min(560px, 92vw) !important;
                min-width: 320px !important;
                max-width: 92vw !important;
                border-radius: 0 !important;
                border-right: 1px solid rgba(255,255,255,0.15) !important;
                /* NOT overflow-y:auto here anymore — confirmed directly as
                   the cause of "can't resize after scrolling down": the
                   handle below is a plain child of THIS box, so scrolling
                   IT scrolled the handle away along with everything else.
                   Scrolling now happens per-tab instead (the stTabPanel
                   rule further down) — this outer box's own content
                   (the tab bar + whichever panel) always fits exactly
                   100vh, so it has nothing of its own left to scroll.
                   overflow:visible (the default, so not stated) is
                   required for the handle's own -14px negative left offset
                   to actually render outside this box instead of being
                   clipped by it. */
                display: flex !important;
                flex-direction: column !important;
            }
            /* Streamlit's st.tabs() renders as stTabs > (a template, empty)
               + one wrapper div, itself holding [the tab-button row] then
               one stTabPanel per tab (only the active one visible). Turning
               that wrapper AND stTabs into flex columns, letting the
               button row size to its own content (flex:0) and each
               stTabPanel fill and independently scroll the remaining space
               (flex:1 + its own overflow-y) is what pins the tab bar in
               place while its content scrolls beneath it — confirmed
               directly against the real DOM (see the childCount/testid
               values read straight off the live page) rather than guessed
               at, since Streamlit's own emotion-hash classes aren't stable
               enough to target directly. */
            div[data-testid="stPopoverBody"][aria-label="☰"] [data-testid="stTabs"],
            div[data-testid="stPopoverBody"][aria-label="☰"] [data-testid="stTabs"] > div {
                display: flex !important;
                flex-direction: column !important;
                flex: 1 1 auto !important;
                min-height: 0 !important;
            }
            /* Streamlit wraps the outer st.tabs() call in a chain of
               generic containers (stLayoutWrapper, then ANOTHER
               stVerticalBlock) between the already-bounded outer
               stVerticalBlock and stTabs itself — confirmed directly by
               walking the live DOM chain (each level's own
               clientHeight/computed flex/min-height) after the
               flex-sizing still wasn't taking effect. Two distinct
               problems found stacked on top of each other:
               1) stLayoutWrapper measured 2001px (its own natural
                  content height) instead of the ~950px actually
                  available — it had no flex-item sizing rule applied to
                  it at all, so it defaulted to flex:0 1 auto (size to
                  content, don't grow).
               2) The stVerticalBlock one level below it already had
                  flex:1 1 0% (Streamlit's own default — correctly set up
                  to grow), but its min-height computed to "auto", not
                  "0" — the classic flexbox min-height:auto trap, where a
                  flex item with flex-basis:0% still refuses to shrink
                  below its OWN content's intrinsic height unless
                  min-height:0 is set explicitly. This is why its content
                  (2001px) leaked through even with the right flex value.
               :has() scopes both fixes to only the wrapper/block that
               actually contains the tabs component, leaving any other
               stLayoutWrapper/stVerticalBlock elsewhere in the popover
               (e.g. around plain widget rows) untouched. */
            div[data-testid="stPopoverBody"][aria-label="☰"] [data-testid="stLayoutWrapper"]:has([data-testid="stTabs"]),
            div[data-testid="stPopoverBody"][aria-label="☰"] [data-testid="stVerticalBlock"]:has(> [data-testid="stTabs"]) {
                flex: 1 1 auto !important;
                min-height: 0 !important;
            }
            div[data-testid="stPopoverBody"][aria-label="☰"] [data-testid="stTabs"] > div > div:not([data-testid="stTabPanel"]) {
                flex: 0 0 auto !important;
            }
            div[data-testid="stPopoverBody"][aria-label="☰"] [data-testid="stTabPanel"] {
                /* Deliberately NOT display:flex here — confirmed directly
                   that forcing it broke tab switching entirely: Streamlit
                   hides an inactive stTabPanel via a plain (non-
                   !important) display:none on a conditionally-assigned
                   emotion-hash class, and any !important display value
                   here always wins over that regardless of which state
                   Streamlit intends, making every tab's panel visible at
                   once. flex/min-height/overflow-y don't have this
                   problem — they're flex-ITEM properties (how THIS
                   element is sized by ITS OWN parent) and an overflow
                   rule (how it scrolls ITS OWN children), neither of
                   which touches this element's own display value, so
                   they apply cleanly whichever panel Streamlit shows. */
                flex: 1 1 auto !important;
                min-height: 0 !important;
                overflow-y: auto !important;
            }
            /* The Strategy tab nests its OWN Sweep/Heatmap/Footprint
               st.tabs() inside this same panel. Rather than making
               stTabPanel itself display:flex (which broke tab-switching,
               see above), the nested stTabs' own wrapper div is already
               forced flex by the rule above — it simply grows to its
               natural content height instead of being height-capped,
               since its flex-sizing has no flex-container ancestor
               between it and the (non-flex) stTabPanel. That's fine: the
               OUTER stTabPanel is still correctly height-bounded (as a
               flex item of the stTabs wrapper) with overflow-y:auto, so
               it scrolls whatever tall content sits inside it, nested
               tabs included — the nested Sweep/Heatmap/Footprint buttons
               just aren't independently sticky while scrolling within
               Rules, which was never separately requested. */
            /* Deliberately NOT native CSS resize:horizontal (tried first) —
               confirmed directly this is a structurally bad fit for a
               pinned panel: native resize assumes a fixed top-left anchor
               with a free bottom-right corner the handle drags to a new
               position, but this panel has ITS OWN edge pinned (left,
               below) and the opposite edge free — the handle itself sits
               glued to the screen's own pinned edge no matter how far you
               drag, since resizing never actually moves it, so the cursor
               and the handle immediately disconnect. This is a real
               element instead: a full-height strip at the panel's own
               free edge (where the moving edge actually is), dragged via
               JS (see the st.html script below — plain st.markdown can't
               run <script>, confirmed directly; unsafe_allow_javascript=True
               is what actually executes it).
               ::after is the visible grip bar — a 4px accent line that
               brightens on hover. Sits ENTIRELY outside the panel's own
               padding box now (right:-14px, width:14px — was -5px/10px on
               the left, before the drawer moved to the screen's left edge
               and its free edge flipped to the right) — confirmed
               directly the old, narrower offset let the grip bar's own
               paint area overlap the first few pixels of real content
               (checkbox/label text right at the drawer's free edge);
               fully clearing the padding needs more than half the old hit
               area's width of clearance. */
            .ict-drawer-resize-handle {
                /* height:100vh instead of top:0;bottom:0 — confirmed
                   directly that bottom:0 resolves against the nearest
                   intermediate wrapper Streamlit puts around this
                   st.markdown call (itself height:auto, i.e. no definite
                   height), not the drawer's own fixed/100vh box, and
                   collapses to 0 rather than stretching. A fixed 100vh
                   sidesteps needing that containing-block chain to
                   resolve correctly at all. */
                position: absolute; right: -14px; top: 0; height: 100vh; width: 14px;
                cursor: ew-resize; z-index: 50;
                display: flex; align-items: center; justify-content: center;
            }
            .ict-drawer-resize-handle::after {
                content: ''; width: 4px; height: 64px; border-radius: 2px;
                background: rgba(255,255,255,0.25); transition: background 0.15s ease;
            }
            .ict-drawer-resize-handle:hover::after,
            .ict-drawer-resize-handle.ict-dragging::after {
                background: #0A84FF;
            }
            </style>""",
            unsafe_allow_html=True,
        )
        # Idempotent setup guarded by a window flag — this script tag can
        # re-execute every time _render_layer_controls' own fragment
        # reruns (any interaction inside the drawer), but the actual
        # mousedown/move/up listeners must only ever be attached ONCE
        # (duplicates would multiply the width delta per pixel dragged).
        # Delegated on document rather than the handle element directly
        # since the popover's own DOM node (and everything inside it,
        # including the handle) gets torn down and rebuilt each time it
        # opens/closes — a direct listener would be attached to a node
        # that's already gone by the next open.
        st.html(
            """<script>
            if (!window.__ictDrawerResizeSetup) {
                window.__ictDrawerResizeSetup = true;
                let dragging = false, panel = null, handleEl = null, framesDisabled = [];
                document.addEventListener('mousedown', (e) => {
                    const handle = e.target.closest('.ict-drawer-resize-handle');
                    if (!handle) return;
                    panel = handle.closest('div[data-testid="stPopoverBody"]');
                    if (!panel) return;
                    handleEl = handle;
                    dragging = true;
                    handle.classList.add('ict-dragging');
                    document.body.style.cursor = 'ew-resize';
                    document.body.style.userSelect = 'none';
                    // The main chart is a REAL iframe (a separate browsing
                    // context) sitting right where the drawer's own left
                    // edge starts. Confirmed directly as a real bug: once
                    // the cursor's on-screen position crosses over it
                    // mid-drag, the BROWSER delivers mousemove straight to
                    // the iframe's own document instead of this one — this
                    // listener stops firing entirely until the cursor
                    // happens to re-enter the parent document, so a fast
                    // drag "escapes" and the resize just stops following
                    // the mouse. Disabling pointer-events on every iframe
                    // for the duration of the drag keeps the parent
                    // document catching every mousemove regardless of
                    // where on screen the cursor actually is; restored on
                    // mouseup below.
                    framesDisabled = Array.from(document.querySelectorAll('iframe'));
                    framesDisabled.forEach((f) => { f.style.pointerEvents = 'none'; });
                    e.preventDefault();
                });
                document.addEventListener('mousemove', (e) => {
                    if (!dragging || !panel) return;
                    // Panel is pinned to the screen's LEFT edge, so its
                    // width is just the cursor's distance from that edge
                    // (was window.innerWidth - e.clientX for the old
                    // right-pinned drawer, measuring from the right edge
                    // instead).
                    const newWidth = e.clientX;
                    const clamped = Math.max(320, Math.min(newWidth, window.innerWidth * 0.92));
                    panel.style.setProperty('width', clamped + 'px', 'important');
                });
                document.addEventListener('mouseup', () => {
                    if (handleEl) handleEl.classList.remove('ict-dragging');
                    dragging = false; panel = null; handleEl = null;
                    document.body.style.cursor = '';
                    document.body.style.userSelect = '';
                    framesDisabled.forEach((f) => { f.style.pointerEvents = ''; });
                    framesDisabled = [];
                });
            }
            </script>""",
            unsafe_allow_javascript=True,
        )
        with st.container(key="_menu_trigger_wrap"):
            with st.popover("☰", help="Layers, Strategy, Backtest, Settings & Charts"):
                st.markdown('<div class="ict-drawer-resize-handle"></div>', unsafe_allow_html=True)
                _render_layer_controls()

    # Read back what the fragment above just computed (or, on a rerun THIS
    # fragment itself didn't trigger, whatever it computed last) — this
    # outer-scope copy is what the one-time prefetch-warming pass and
    # _render_chart's OWN initial definition close over; _render_chart
    # additionally re-reads the same dict fresh at the top of its own body
    # (see below) so it's never stuck on a stale copy from before the
    # controls fragment's last rerun.
    _cc = st.session_state.get("_chart_controls", {})
    layers = _cc.get("layers", [])
    layer_tf = _cc.get("layer_tf", {})
    max_items_per_layer = _cc.get("max_items_per_layer", 5)
    selected_kill_zones = _cc.get("selected_kill_zones", [])
    show_mitigated = _cc.get("show_mitigated", False)
    show_volume = _cc.get("show_volume", False)
    show_log_scale = _cc.get("log_scale", False)
    show_rsi = _cc.get("show_rsi", False)
    show_macd = _cc.get("show_macd", False)
    show_bb = _cc.get("show_bb", False)
    show_atr = _cc.get("show_atr", False)
    show_indicators = _cc.get("show_ma", True)
    indicator_tf = _cc.get("ma_tf", "Chart TF")
    ma_fast_period = _cc.get("ma_fast_period", 20)
    ma_slow_period = _cc.get("ma_slow_period", 50)
    rsi_period = _cc.get("rsi_period", 14)
    macd_fast = _cc.get("macd_fast", 12)
    macd_slow = _cc.get("macd_slow", 26)
    macd_signal = _cc.get("macd_signal", 9)
    bb_period = _cc.get("bb_period", 20)
    bb_std = _cc.get("bb_std", 2.0)
    atr_period_ind = _cc.get("atr_period_ind", 14)
    displacement_ratio = _cc.get("displacement_ratio", 0.5)
    swing_atr_period = _cc.get("swing_atr_period", 14)
    swing_atr_mult = _cc.get("swing_atr_mult", 1.5)
    session_lookback_days = _cc.get("session_lookback_days", 60)
    eq_tolerance = _cc.get("eq_tolerance", 0.0015)
    ms_mode = _cc.get("ms_mode", "close")
    liquidity_reaction_window = _cc.get("liquidity_reaction_window", 5)
    price_projection_bars = _cc.get("price_projection_bars", PRICE_PROJECTION_BARS)
    show_volume_profile = _cc.get("show_volume_profile", False)
    vp_anchor = _cc.get("vp_anchor", "Full history")
    vp_anchor_date = _cc.get("vp_anchor_date")
    zone_opacity = _cc.get("zone_opacity", 0.25)
    data_source = _cc.get("data_source", "auto")
    overlay_tf = _cc.get("overlay_tf", "4h")
    _win_rate_exit_types = _cc.get("win_rate_exit_types", tuple(EXIT_EVENT_LABELS.keys()))
    _news_blackout = _cc.get("news_blackout")
    _confluence_keys = _cc.get("confluence_keys", [])
    _entry_rule = _cc.get("entry_rule", {"anchor": "current_price"})
    _exit_rule = _cc.get("exit_rule", {"anchor": "current_price"})
    _stop_rule = _cc.get("stop_rule", {"anchor": "current_price"})

    if not ticker:
        st.warning("Enter a ticker.")
        st.stop()

    tf_label = st.session_state[TF_KEY_BY_CHART["main"]]
    tf = TIMEFRAMES[tf_label]

    layers_state = {
        name: {
            "on": st.session_state[f"fvg_layer_{name}"],
            "tf": st.session_state[f"fvg_tf_{name}"],
        }
        for name in ICT_LAYERS
    }
    _save_key = (ticker, tf_label, layers_state)
    if st.session_state.get("_last_saved_state") != _save_key:
        _save_last_state(ticker, tf_label, layers_state)
        st.session_state["_last_saved_state"] = _save_key

    refresh_interval = CANDLE_SECONDS.get(tf_label)
    # Capped at 5s regardless of selection — previously this only sped up to
    # 1s while "main" was the selected panel and otherwise sat on its TF's
    # full 10-120s cadence, so selecting an HTF/LTF mini panel instead left
    # main showing a stale price/candle for up to two minutes next to
    # panels that (as of the same fix in _render_mini_chart) now tick every
    # 1s — the exact "charts out of sync" symptom this is closing. Capped at
    # 5s rather than forced to 1s like the mini panels: unlike them, main
    # re-runs full FVG/OB/swing/liquidity detection and a bulk shape rebuild
    # every tick (see the shape-loop perf history above), so unconditional
    # 1s here would pay that cost even while nobody's watching this panel.
    if refresh_interval is not None:
        refresh_interval = min(refresh_interval, 5) if selected_chart != "main" else 1

    @st.fragment(run_every=refresh_interval)
    def _render_chart():
        try:
            # Fresh every time THIS fragment itself reruns (its own click-
            # to-select, or the run_every tick above) — not the outer
            # script's closure-captured copy from whenever it last fully
            # ran. _render_layer_controls is a SEPARATE fragment; a change
            # made there updates session_state without rerunning this one,
            # so without this re-read, a Layers/Strategy/Settings change
            # wouldn't show up here until something else forced a full
            # rerun. Shadows the same-named outer-scope locals on purpose —
            # every reference below already uses these names unchanged.
            _cc = st.session_state.get("_chart_controls", {})
            layers = _cc.get("layers", [])
            # "Chart TF" resolved HERE, against this fragment's own current
            # tf_label (closed over from the outer scope, refreshed on every
            # full rerun the main TF selector triggers) — not resolved back
            # where the Layers tab's own selectbox is read, since that's a
            # SEPARATE fragment that only reruns on ITS OWN widget changes
            # and would otherwise freeze "Chart TF" at whatever the main TF
            # happened to be the last time a Layers-tab control was touched
            # (the exact cross-fragment staleness bug the R:R stat above hit).
            layer_tf = {_name: (tf_label if _tf == "Chart TF" else _tf)
                        for _name, _tf in _cc.get("layer_tf", {}).items()}
            max_items_per_layer = _cc.get("max_items_per_layer", 5)
            selected_kill_zones = _cc.get("selected_kill_zones", [])
            show_mitigated = _cc.get("show_mitigated", False)
            show_volume = _cc.get("show_volume", False)
            show_log_scale = _cc.get("log_scale", False)
            show_rsi = _cc.get("show_rsi", False)
            show_macd = _cc.get("show_macd", False)
            show_bb = _cc.get("show_bb", False)
            show_atr = _cc.get("show_atr", False)
            show_indicators = _cc.get("show_ma", True)
            indicator_tf = _cc.get("ma_tf", "Chart TF")
            ma_fast_period = _cc.get("ma_fast_period", 20)
            ma_slow_period = _cc.get("ma_slow_period", 50)
            rsi_period = _cc.get("rsi_period", 14)
            macd_fast = _cc.get("macd_fast", 12)
            macd_slow = _cc.get("macd_slow", 26)
            macd_signal = _cc.get("macd_signal", 9)
            bb_period = _cc.get("bb_period", 20)
            bb_std = _cc.get("bb_std", 2.0)
            atr_period_ind = _cc.get("atr_period_ind", 14)
            displacement_ratio = _cc.get("displacement_ratio", 0.5)
            swing_atr_period = _cc.get("swing_atr_period", 14)
            swing_atr_mult = _cc.get("swing_atr_mult", 1.5)
            session_lookback_days = _cc.get("session_lookback_days", 60)
            eq_tolerance = _cc.get("eq_tolerance", 0.0015)
            ms_mode = _cc.get("ms_mode", "close")
            liquidity_reaction_window = _cc.get("liquidity_reaction_window", 5)
            price_projection_bars = _cc.get("price_projection_bars", PRICE_PROJECTION_BARS)
            show_volume_profile = _cc.get("show_volume_profile", False)
            vp_anchor = _cc.get("vp_anchor", "Full history")
            vp_anchor_date = _cc.get("vp_anchor_date")
            zone_opacity = _cc.get("zone_opacity", 0.25)
            data_source = _cc.get("data_source", "auto")
            overlay_tf = _cc.get("overlay_tf", "4h")
            _win_rate_exit_types = _cc.get("win_rate_exit_types", tuple(EXIT_EVENT_LABELS.keys()))
            _news_blackout = _cc.get("news_blackout")
            _confluence_keys = _cc.get("confluence_keys", [])
            _entry_rule = _cc.get("entry_rule", {"anchor": "current_price"})
            _exit_rule = _cc.get("exit_rule", {"anchor": "current_price"})
            _stop_rule = _cc.get("stop_rule", {"anchor": "current_price"})

            forming_bar_ts = None  # set below iff the accumulator ran this render
            df = get_yf_ohlcv(ticker, period=tf["period"], interval=tf["fetch_interval"], provider=data_source)
            # Captured now, before resample/splice below touch df — those
            # don't reliably carry df.attrs through (pandas drops it on
            # concat/resample), and this is the dominant source for the
            # whole chart's content anyway, so it's not worth threading
            # provider info through every later transform just to relabel
            # the readout with whichever source served the last few rows.
            served_by = DATA_SOURCE_LABELS.get(df.attrs.get("provider"), "Yahoo Finance") if not df.empty else "—"
            if df.empty:
                # Short-circuit BEFORE the splice/resample below, not just
                # before the render — both assume a real DatetimeIndex
                # (resample_ohlc's own .resample(rule) call raises on an
                # empty df's default RangeIndex, confirmed directly: a
                # ticker every provider fails on doesn't cleanly fall
                # through to the "no data" message, it throws a confusing
                # pandas TypeError first). Nothing downstream of a truly
                # empty base fetch is meaningful anyway — nothing to splice
                # fresh rows onto, nothing to bucket into coarser candles.
                clicked = ict_chart(
                    [], f"{ticker}|{tf_label}|empty",
                    overlays={"rectangles": [], "price_lines": [], "markers": [], "ghost_candles": []},
                    options={"volume": False, "selected": st.session_state.get("selected_chart") == "main",
                             "error_message": f"No data for {ticker} at {tf_label}"},
                    ohlc={"ticker": ticker, "source": "—", "forming_bar_time": None},
                    height=1000,
                    display_tz=theme.get_display_tz(),
                )
                _select_chart("main", clicked)
                return
            if tf["resample"] is None and refresh_interval is not None:
                # Splice the cheap, fast-refreshing small-window fetch's
                # newest rows onto the slow-refreshing full-context one —
                # keeps the visible edge current every ~8s without paying
                # the expensive full-period fetch's cost that often. Native
                # intervals only (resample targets need the full raw window
                # to bucket correctly) and only where auto-refresh is even
                # on (daily+ never gets here, refresh_interval is None).
                df_latest = get_latest_bars(ticker, tf["fetch_interval"], provider=data_source)
                if not df_latest.empty:
                    df = pd.concat([df[df.index < df_latest.index[0]], df_latest])

                    # Yahoo's still-forming last bar comes back degenerate —
                    # Open == High == Low == Close, just the latest trade
                    # price repeated, not a real intrabar range. Confirmed
                    # directly: true on a fresh fetch too, not something the
                    # fast-poll introduced — it was always there, just far
                    # more visible now that the display refreshes every
                    # ~8-10s instead of once a minute (a flat line jumping
                    # to a new price every few seconds, instead of once).
                    # Each poll DOES capture a genuine price at that moment
                    # though, so accumulating the highest/lowest seen across
                    # polls approximates a real intrabar range from what we
                    # actually have — resets the moment the timestamp
                    # advances to a new bar. The captured "open" is exact
                    # only up to our own poll cadence (we might catch a bar
                    # a few seconds after it truly opened), an unavoidable
                    # approximation of a polling-based feed, not a push one.
                    last_ts = df.index[-1]
                    accum_key = f"_forming_bar_{ticker}_{tf_label}"
                    accum = st.session_state.get(accum_key)
                    last_open = float(df["Open"].iloc[-1])
                    last_high = float(df["High"].iloc[-1])
                    last_low = float(df["Low"].iloc[-1])
                    if accum is None or accum["ts"] != last_ts:
                        accum = {"ts": last_ts, "open": last_open, "high": last_high, "low": last_low}
                    else:
                        accum["high"] = max(accum["high"], last_high)
                        accum["low"] = min(accum["low"], last_low)
                    st.session_state[accum_key] = accum
                    df.loc[last_ts, "Open"] = accum["open"]
                    df.loc[last_ts, "High"] = accum["high"]
                    df.loc[last_ts, "Low"] = accum["low"]
                    forming_bar_ts = last_ts
            if tf["resample"]:
                df = resample_ohlc(df, tf["resample"])
            if df.empty:
                # A plain `return` here, not st.stop() — st.stop() used to
                # kill the entire script run from inside this fragment,
                # taking the mini charts and every control in the popovers
                # down with it over what's really just this one panel's
                # data. The chart component still renders (border, Fit
                # button, click-to-select all stay live), just with no
                # candles and this message where they'd be — "the UI broke"
                # and "this symbol has no data right now" should never look
                # like the same failure.
                clicked = ict_chart(
                    [], f"{ticker}|{tf_label}|empty",
                    overlays={"rectangles": [], "price_lines": [], "markers": [], "ghost_candles": []},
                    options={"volume": False, "selected": st.session_state.get("selected_chart") == "main",
                             "error_message": f"No data for {ticker} at {tf_label}"},
                    ohlc={"ticker": ticker, "source": "—", "forming_bar_time": None},
                    height=1000,
                    display_tz=theme.get_display_tz(),
                )
                _select_chart("main", clicked)
                return

            o_col, h_col, l_col, c_col = (("Open", "High", "Low", "Close") if "Close" in df
                                           else ("open", "high", "low", "close"))
            v_col = "Volume" if "Volume" in df else ("volume" if "volume" in df else None)

            # Kill zone no longer trims the candles down to just that window —
            # it highlights it instead (a shaded band per occurrence, below),
            # so the rest of the day's context stays visible. Axis is always
            # real NY wall-clock time, "fake UTC"-encoded so the chart shows
            # correct NY digits with zero timezone config.
            axis_secs = _ny_fake_utc_seconds_vec(df.index)
            # forming_bar_ts is always df.index[-1] when set (the accumulator
            # only ever touches the last row) — lets the frontend exclude this
            # bar's still-growing, poll-noisy high/low from autoscale (see
            # ict_chart's autoscaleInfoProvider) instead of rescaling the
            # whole visible price axis on every single poll tick.
            forming_bar_axis_sec = axis_secs[-1] if forming_bar_ts is not None else None

            # Still-live levels (unfilled FVG/OB, untouched BSL/SSL, ongoing
            # equilibrium) get capped by the detection logic at the last loaded
            # candle — dead-ending exactly on "now" reads as invalidated when
            # it's actually still active. Project those a few candles past the
            # last bar instead; already-invalidated levels (end != last bar)
            # keep stopping exactly where they were actually hit. Shared by
            # every reference frame below (see REFERENCE_TFS) — it's really
            # just "a little past now" on the main chart's OWN axis, which
            # stays correct regardless of which timeframe actually detected
            # the structure being extended.
            # Direct request: when Price Projection is on, still-open areas/
            # levels should reach as far as the projection itself does,
            # instead of stopping a token few candles past "now" — one
            # shared boundary (PRICE_PROJECTION_BARS) for both, so they can
            # never silently drift out of sync with each other.
            FUTURE_EXTEND_CANDLES = price_projection_bars if "Price Projection" in layers else 3
            bar_step = (axis_secs[-1] - axis_secs[-2]) if len(axis_secs) > 1 else 1
            future_edge = axis_secs[-1] + FUTURE_EXTEND_CANDLES * bar_step

            # .to_numpy() once up front rather than df[col].iloc[i] per cell
            # per row inside the loop — measured directly on a real 5m fetch
            # (16,930 rows, this timeframe's own native size before backfill):
            # the per-cell .iloc version cost ~1.9s EVERY render (this runs on
            # every 1s fragment tick while the main chart is selected, not
            # just on a timeframe switch), because pandas' scalar .iloc does
            # real per-call work (index alignment, block-manager lookup) that
            # numpy array indexing doesn't. The vectorized version below does
            # the identical output in ~0.015s — a ~125x difference that was
            # the dominant cause of 5m (this app's largest dataset) taking
            # ~30s to become responsive: renders were taking longer than the
            # fragment's own 1s refresh interval, so they piled up.
            _o, _h, _l, _c = (df[o_col].to_numpy(), df[h_col].to_numpy(),
                              df[l_col].to_numpy(), df[c_col].to_numpy())
            _v = df[v_col].to_numpy() if v_col is not None else None
            bars = [
                {
                    "time": axis_secs[i],
                    "open": float(_o[i]), "high": float(_h[i]),
                    "low": float(_l[i]), "close": float(_c[i]),
                    "volume": float(_v[i]) if _v is not None else 0.0,
                }
                for i in range(len(df))
            ]

            # Structure older than this doesn't get drawn at all — see
            # OVERLAY_LOOKBACK_DAYS. Detection itself still runs against the FULL
            # df (a zone formed 6 days ago needs the full forward context to know
            # whether a candle from 2 days ago filled it), only the RESULTING list
            # gets filtered down to recent origins before it's ever turned into an
            # overlay.
            overlay_cutoff = df.index[-1] - pd.Timedelta(days=OVERLAY_LOOKBACK_DAYS)
            recent_df = df[df.index >= overlay_cutoff]

            legend_items = []
            rectangles = []
            price_lines = []
            markers = []
            # Raw FVG/Order Block zones actually rendered this pass — not a
            # separate computation, the exact same post-filter/post-cap
            # lists the rectangles above are built from, so this table can
            # never show something the chart doesn't and vice versa. Legend
            # pills give a count; this gives the real numbers behind it —
            # "I need to see what happens under the hood," not just a tally.
            fvg_ob_rows = []
            # Same idea for Liquidity (BSL/SSL) — these only ever rendered
            # as unlabeled price_lines with a hover title, no way to see
            # the actual price/distance without reading raw chart data.
            # Confirmed directly trying to use this chart to plan a trade:
            # exact liquidity prices weren't answerable from the UI at all.
            liquidity_rows = []
            # Same idea again for Naked POCs — see the detector's own
            # docstring for what these are and why they persist across
            # sessions instead of resetting with each new profile.
            naked_poc_rows = []
            # One normalized entry per drawn geometry (rect/level/point),
            # built alongside the rows above in the same per-layer blocks —
            # what a click on the chart gets hit-tested against (Feature C,
            # "click a zone, get a confluence readout"). "type" is the
            # zone's own bullish/bearish bias where it has one, None where
            # it doesn't (Naked POC, Swing Points, Poor High/Low) — see
            # the click-handling block's own comment on why those skip
            # confluence scoring entirely rather than guess a direction.
            clickable_zones = []
            # Set inside the Naked POC layer block below when it runs —
            # stays None otherwise (feature off, or too little history for
            # even 2 sessions), so the detail expander further down can
            # check it safely either way.
            _poc_mig = None

            # Entry/SL/TP for the single top-ranked currently-open FVG/Order
            # Block/Liquidity Reaction on THIS chart's own ticker+timeframe —
            # drawn exactly the way every other ICT level on this chart
            # draws, a plain price_line, nothing else (see recommender.py
            # for the ranking: Edge Lab's own validated-edge status first,
            # a discretionary confluence tally second — that reasoning
            # stays internal to the pick, not spelled out on screen). Always
            # on, not gated by the layer toggles above — this isn't a
            # detection layer to switch on/off, it's the one concrete
            # "where's the stop, where's the target" answer for whatever
            # this chart is already showing.
            # One canonical "where is price right now" — the main chart's
            # own latest close. Used below for the Entry/SL/TP pick and
            # again further down for every layer's own above/below-price
            # filtering, regardless of which timeframe any given layer is
            # independently set to detect on — there's only one real
            # current price, not one per timeframe.
            current_price = float(df[c_col].iloc[-1])
            # A direct lookup, independent of the Backtest tab's own entry/
            # exit/stop rule below — a real live MA+FVG overlap should
            # show up here regardless of whatever rule happens to be
            # configured. The border highlight means "an EMA sits inside
            # this zone right now," not "this rule is pointed at it" —
            # see ma_fvg_starts's own docstring.
            _ma_fvg_starts = ma_fvg_starts(df)

            # Resolved unconditionally — the drawing loop below always
            # needs these now. (The Backtest tab's own R:R
            # preview resolves its own copy of these locally, in the SAME
            # fragment as the rule widgets themselves — see
            # _render_layer_controls — rather than reading a value
            # computed here: this fragment only ticks on its own schedule,
            # so a value written here lagged behind whatever rule was
            # actually selected until this fragment's next tick, which
            # could be seconds away or, on a non-auto-ticking timeframe,
            # never — confirmed directly as the cause of "R:R doesn't
            # update / looks wrong right after switching a rule.")
            # resolve_rule answers "what price does this rule mean RIGHT
            # NOW," nothing ranked or guessed. Any rule that can't resolve
            # (e.g. "3rd-nearest FVG" but only 2 exist) returns None rather
            # than falling back to a different, unrequested number.
            #
            # _active_scan_pick (sidebar "Scan watchlist"/"Scan timeframes")
            # takes priority over the Strategy tab's own rule when it
            # matches THIS ticker — a fixed snapshot from scan time, not
            # re-resolved live, since re-resolving "nearest FVG below
            # price" could land on a different zone entirely once price
            # has moved, silently showing something the scan never
            # actually picked.
            #
            # Unlocked (default): only applies when the timeframe ALSO
            # matches — switching timeframe falls through to the normal
            # rule below with nothing to clear, it just stops applying on
            # its own. Locked (the checkbox below, direct request: "cycle
            # through timeframes without the trade going away"): applies
            # regardless of the displayed timeframe — the pick's own real
            # price/timestamps are timeframe-independent, so it keeps
            # showing correctly no matter what tf_label is browsed to,
            # until a new pick replaces it (which resets the lock off) or
            # the ticker itself changes.
            _scan_pick = st.session_state.get("_active_scan_pick")
            _scan_pick_here = bool(_scan_pick) and _scan_pick["ticker"] == ticker
            if _scan_pick_here:
                st.checkbox(
                    f"🔒 Lock: {_scan_pick['label']} — keep showing across timeframes",
                    key="_scan_pick_locked",
                    help="On: this trade's exact entry/stop/target and confluence zones stay "
                         "pinned no matter which timeframe you switch to, replacing everything "
                         "else on the chart. Off (default): only shows while you're on the exact "
                         "timeframe it was found on.")
            _scan_pick_locked = _scan_pick_here and st.session_state.get("_scan_pick_locked", False)
            _scan_pick_active = _scan_pick_here and (_scan_pick_locked or _scan_pick["timeframe"] == tf_label)
            # Rebalance chain: ticker-scoped like _active_scan_pick above,
            # but NOT timeframe-scoped — it's a projection drawn in its
            # own forward time-slices past whatever's currently live, so
            # it renders the same regardless of which timeframe is
            # charted (same "real price is timeframe-independent"
            # reasoning as the lock checkbox).
            _tf_scan_chain = st.session_state.get("_tf_scan_chain")
            _chain_here = bool(_tf_scan_chain) and _tf_scan_chain[0]["setup"]["ticker"] == ticker
            _show_chain = _chain_here and st.session_state.get("_show_rebalance_chain", False)
            if _scan_pick_active:
                _entry_price, _stop_price, _exit_price = _scan_pick["entry"], _scan_pick["stop"], _scan_pick["target"]
            else:
                # Held, not re-resolved every tick: this fragment reruns
                # every 1-5s, and resolve_rule's own "nth-nearest zone"
                # selection is current_price-relative — as live price
                # nudges a few ticks, "nearest FVG below price" can flip
                # to a different zone entirely, making the rendered
                # trade box visibly reshuffle for no reason tied to the
                # setup itself. Direct request: hold the currently
                # detected trade fixed on the chart instead of letting
                # it move around as price updates. See
                # recommender.hold_resolved_trade's own docstring — it
                # only re-resolves once this exact trade is actually
                # invalidated (price reaches its own stop or target) or
                # the ticker/timeframe/rule itself changes.
                _rule_ctx_key = (f"{ticker}|{tf_label}|{json.dumps(_entry_rule, sort_keys=True)}|"
                                  f"{json.dumps(_exit_rule, sort_keys=True)}|{json.dumps(_stop_rule, sort_keys=True)}")
                _held_trade, _entry_price, _exit_price, _stop_price = hold_resolved_trade(
                    st.session_state.get("_held_chart_trade"), _rule_ctx_key,
                    _entry_rule, _exit_rule, _stop_rule, df, current_price)
                st.session_state["_held_chart_trade"] = _held_trade

            # Entry/SL/TP come from the Strategy tab's own rule by default —
            # the old Automatic mode (trusting best_trade_now's own
            # validation/confluence ranking for these same three lines,
            # unconditionally, on whatever timeframe happened to be
            # charted) and the Off toggle (hiding them) are both gone per
            # direct request. _active_scan_pick above is a narrower,
            # deliberate exception: an explicit "show me THIS specific
            # scan result" click, not best_trade_now silently driving the
            # chart on its own again.
            _entry_x = _ny_fake_utc_seconds(df.index[-1])
            # Direct request: "the current model for rendering active
            # trade is something I don't want to see... During the
            # backtest, the areas are rendered a specific way. I want
            # exactly that for the active trade" — solid flat fill +
            # visible border, the EXACT style the Backtest tab's own
            # "Show these trades on the chart" already draws for a
            # resolved trade (see this function's own backtest-trades
            # block further down) — not the old fading gradient band.
            # "the box should be printed from present to future a bit":
            # bounded to _trade_box_seconds, not open-ended out to
            # future_edge — _active_box_t1 is reused as the rebalance
            # chain's own starting point below (see _show_chain), so the
            # active trade's box and the chain's own boxes read as one
            # continuous, chained sequence rather than two unrelated
            # spans.
            _trade_box_seconds = _TF_BAR_SECONDS[tf_label] * 50
            _active_box_t1 = _entry_x + _trade_box_seconds
            # Direct request: "the current trade render... with the glow
            # entry/tp/sl, I want it to be gone... bring it closer to the
            # last bar printed." dashed=True (was False): a non-dashed
            # line ALWAYS stretches to the chart's own far right edge
            # regardless of t1 (see HLineRenderer's own comment) — the
            # glow/gradient treatment that came with it, AND the reason
            # these labels rendered far from the box they belong to
            # instead of right at its own edge next to the last real
            # candle. Dashed respects t0/t1 for real, same fix already
            # applied to the rebalance chain's own entry markers.
            for _price, _color, _title in [
                (_entry_price, theme.NEON_AMBER, "Entry"),
                (_stop_price, theme.NEON_MAGENTA, "SL"),
                (_exit_price, theme.NEON_GREEN, "TP"),
            ]:
                if _price is not None:
                    price_lines.append({"t0": _entry_x, "t1": _active_box_t1, "price": _price,
                                         "color": _hex_to_rgba(_color, 1.0), "title": _title,
                                         "line_width": 2, "dashed": True,
                                         # bool(...) - resolve_rule can return a raw
                                         # numpy float off a liquidity-anchored rule
                                         # (a swing point's own price), and comparing
                                         # that against another float yields
                                         # numpy.bool_, not a native bool. Confirmed
                                         # directly as a real bug: embedded straight
                                         # into this dict, it broke the component's
                                         # JSON serialization the moment a rule using
                                         # such a price got applied to the chart
                                         # ("Could not fetch <ticker>: ... Object of
                                         # type bool is not JSON serializable").
                                         "above": bool(_price >= (_entry_price if _entry_price is not None else _price))})
            if _entry_price is not None and _stop_price is not None:
                rectangles.append({"t0": _entry_x, "t1": _active_box_t1, "p0": _entry_price, "p1": _stop_price,
                                    "fill": _hex_to_rgba(theme.NEON_MAGENTA, 0.10),
                                    "border": _hex_to_rgba(theme.NEON_MAGENTA, 0.9)})
            if _entry_price is not None and _exit_price is not None:
                rectangles.append({"t0": _entry_x, "t1": _active_box_t1, "p0": _entry_price, "p1": _exit_price,
                                    "fill": _hex_to_rgba(theme.NEON_GREEN, 0.10),
                                    "border": _hex_to_rgba(theme.NEON_GREEN, 0.9),
                                    # Direct request: "for every trade, within the fill area,
                                    # should contain text with all confluences" — only available
                                    # when this box came from a scan pick (best_trade_now
                                    # actually scored it); a plain Strategy-tab rule has no
                                    # confluence data to show, so this is simply absent then.
                                    "body_lines": _confluence_body_lines(_scan_pick) if _scan_pick_active else None})

            # Snapshot of JUST the Entry/SL/TP lines + their risk/reward
            # bands, taken right here before any detection LAYER below
            # (FVG/OB/Liquidity/Swing/Premium-Discount/Naked POC/...) gets
            # a chance to append its own rectangles/price_lines onto the
            # same two lists. The isolated "lock trade" view (see
            # _scan_pick_overlays' own call site, near the ict_chart call
            # below) rebuilds `rectangles`/`price_lines` from THIS
            # snapshot instead of whatever those lists grow into by the
            # end of the script — otherwise every other layer's own
            # clutter would still leak into a view meant to show "those
            # exact zones, nothing else."
            _trade_only_rectangles = list(rectangles)
            _trade_only_price_lines = list(price_lines)

            # Every trade from the Backtest tab's own "Run backtest" run, each
            # as its own risk/reward box over its real entry-to-exit span —
            # opt-in via the "Show these trades on the chart" checkbox next
            # to that panel's results table, and only drawn when the stored
            # result still matches the CURRENT rule/ticker/timeframe (same
            # staleness key that panel already checks before trusting its
            # own cached result — a rule/ticker/timeframe change elsewhere
            # must not leave stale boxes from a different setup on screen).
            # backtest_custom_rule's own max_signals=100 already bounds
            # this to at most 200 rectangles, in line with this app's other
            # overlay volumes.
            if st.session_state.get("_bt_visualize_trades"):
                _bt_state = st.session_state.get("_backtest_result")
                _bt_key_now = f"{ticker}|{tf_label}|{_entry_rule}|{_exit_rule}|{_stop_rule}"
                _bt_matched = bool(_bt_state) and _bt_state.get("key") == _bt_key_now and bool(_bt_state.get("result"))
                _bt_drawn = 0
                _bt_off_axis = 0
                if _bt_matched:
                    _axis_min, _axis_max = (min(axis_secs), max(axis_secs)) if axis_secs else (None, None)
                    _bt_trades = _bt_state["result"]["trades"]
                    # Origin lines are a NEW addition on top of the existing
                    # 2-rectangles-per-trade budget (already relied on
                    # max_signals=100 capping total volume) — 3 lines/trade
                    # would triple that on its own (confirmed directly: 100
                    # trades produced 303 price_lines, on top of ~200
                    # rectangles and up to ~5000 backfill ghost-candle tiles
                    # already drawn every tick, and candles intermittently
                    # failed to render under that combined load). Capped
                    # independently of max_signals, at the most RECENT
                    # trades only (oldest ones are also the ones most likely
                    # already off-axis anyway) — rectangles for every trade
                    # are untouched, only the extra origin-line volume is
                    # bounded.
                    _origin_lines_from = max(0, len(_bt_trades) - 40)
                    for _ti, _t in enumerate(_bt_trades):
                        _t0, _t1 = _ny_fake_utc_seconds(_t["entry_time"]), _ny_fake_utc_seconds(_t["exit_time"])
                        # A rectangle whose t0/t1 falls before the chart's own
                        # earliest loaded bar has nothing to resolve its x
                        # coordinate against (lightweight-charts'
                        # timeToCoordinate returns null past the series'
                        # own range) and silently never draws — same root
                        # cause as the Tier-1 QA pass's backfill-anchor fix,
                        # just hitting a different overlay this time. Counted
                        # here rather than only guessed at, so the status
                        # line below can say which one actually happened.
                        if _axis_min is not None and (_t0 < _axis_min or _t1 < _axis_min):
                            _bt_off_axis += 1
                            continue
                        rectangles.append({
                            "t0": _t0, "t1": _t1, "p0": _t["entry"], "p1": _t["stop"],
                            "fill": _hex_to_rgba(theme.NEON_MAGENTA, 0.10),
                            "border": _hex_to_rgba(theme.NEON_MAGENTA, 0.9) if not _t["won"] else None,
                        })
                        rectangles.append({
                            "t0": _t0, "t1": _t1, "p0": _t["entry"], "p1": _t["target"],
                            "fill": _hex_to_rgba(theme.NEON_GREEN, 0.10),
                            "border": _hex_to_rgba(theme.NEON_GREEN, 0.9) if _t["won"] else None,
                        })
                        # Same "trace back to where this came from" treatment
                        # as Premium/Discount's swing-high/low lines — one
                        # per leg (entry/target/stop), each from ITS OWN
                        # resolved zone/level's formation time through to
                        # this trade's entry (the moment all three actually
                        # got used), at that leg's own price. Skipped per-leg
                        # whenever that rule was anchored to current_price
                        # (no zone to trace), its origin falls before the
                        # chart's own earliest loaded bar (nothing to
                        # resolve an x coordinate against — same reasoning
                        # as the off-axis trade skip above), or this trade
                        # falls outside the most-recent-40 origin-line budget.
                        if _ti < _origin_lines_from:
                            _bt_drawn += 1
                            continue
                        for _leg_key, _leg_price_key, _leg_color in (
                            ("entry_origin_time", "entry", theme.NEON_CYAN),
                            ("exit_origin_time", "target", theme.NEON_GREEN),
                            ("stop_origin_time", "stop", theme.NEON_MAGENTA),
                        ):
                            _origin_t = _t.get(_leg_key)
                            if _origin_t is None:
                                continue
                            _origin_secs = _ny_fake_utc_seconds(_origin_t)
                            if _axis_min is not None and _origin_secs < _axis_min:
                                continue
                            price_lines.append({
                                "t0": _origin_secs, "t1": _t0, "price": _t[_leg_price_key],
                                "color": _hex_to_rgba(_leg_color, 0.85),
                            })
                        _bt_drawn += 1
                st.session_state["_bt_viz_status"] = {
                    "checked": True, "matched": _bt_matched, "drawn": _bt_drawn, "off_axis": _bt_off_axis,
                    "total": len(_bt_state["result"]["trades"]) if _bt_matched else 0,
                }
            else:
                st.session_state["_bt_viz_status"] = {"checked": False}

            # Two EMAs, drawn as real lines on the main chart (not just an
            # inferred sidebar label) — see feedback_ui_obviousness: a
            # signal a human needs to notice should be visible at the thing
            # it's about, not just named in a list. Periods are now user-
            # adjustable (ma_fast_period/ma_slow_period, default 20/50 —
            # MA_FVG_PERIODS' own values) for READING the chart; the MA+FVG
            # strategy's own confluence/win-rate math still keys off the
            # fixed MA_FVG_PERIODS pair regardless of what's chosen here
            # (see the period controls' own help text).
            _chart_indicators = {}
            if show_indicators:
                # "Chart TF" (default) reuses the main df directly, same as
                # before this toggle existed. A specific choice instead
                # fetches THAT timeframe's own data and computes the EMAs
                # on it — same pattern layer_tf/overlay_tf already use for
                # "detect/draw on a different timeframe than the candles
                # themselves" — so viewing 15m candles with a steadier 1h
                # average laid over them is a plain timeframe swap, not a
                # new mechanism.
                if indicator_tf == "Chart TF" or indicator_tf == tf_label:
                    _ind_df = df
                else:
                    _ind_conf = TIMEFRAMES[indicator_tf]
                    _ind_df = get_yf_ohlcv(ticker, period=_ind_conf["period"], interval=_ind_conf["fetch_interval"],
                                            provider=data_source)
                    if _ind_conf["resample"]:
                        _ind_df = resample_ohlc(_ind_df, _ind_conf["resample"])
                if not _ind_df.empty:
                    _close_col = _ind_df["Close"] if "Close" in _ind_df else _ind_df["close"]
                    # Generic ma_fast/ma_slow keys (not ma20/ma50 literals) —
                    # the periods are now user-adjustable, so the frontend
                    # needs the chosen period alongside the line data to
                    # label it correctly rather than a hardcoded "EMA 20".
                    _chart_indicators["ma_fast"] = _series_to_points(ema(_close_col, ma_fast_period))
                    _chart_indicators["ma_slow"] = _series_to_points(ema(_close_col, ma_slow_period))
                    _chart_indicators["ma_fast_period"] = ma_fast_period
                    _chart_indicators["ma_slow_period"] = ma_slow_period

            # RSI/MACD/Bollinger Bands/ATR — plain technical indicators,
            # always read off the main chart's own df at its own timeframe
            # (no separate TF picker; see the Settings checkboxes for why).
            if (show_rsi or show_macd or show_bb or show_atr) and not df.empty:
                _ti_close = df["Close"] if "Close" in df else df["close"]
                if show_rsi:
                    _chart_indicators["rsi"] = _series_to_points(rsi(_ti_close, rsi_period))
                if show_macd:
                    _macd_line, _signal_line, _hist = macd(_ti_close, macd_fast, macd_slow, macd_signal)
                    _chart_indicators["macd"] = {
                        "macd": _series_to_points(_macd_line),
                        "signal": _series_to_points(_signal_line),
                        "histogram": _hist_to_points(_hist, theme.NEON_GREEN, theme.NEON_MAGENTA),
                    }
                if show_bb:
                    _bb_upper, _bb_basis, _bb_lower = bollinger_bands(_ti_close, bb_period, bb_std)
                    _chart_indicators["bb"] = {
                        "upper": _series_to_points(_bb_upper),
                        "basis": _series_to_points(_bb_basis),
                        "lower": _series_to_points(_bb_lower),
                    }
                if show_atr:
                    # Needs the full OHLC, not just Close — unlike RSI/MACD/
                    # BB above, True Range is a function of this bar's own
                    # high-low AND the gap from the prior close (see atr's
                    # own docstring), so it reads df directly rather than
                    # the already-sliced _ti_close series the others share.
                    _chart_indicators["atr"] = _series_to_points(atr(df, atr_period_ind))

            # Volume Profile — how much volume traded at each PRICE level
            # over this chart's own currently-loaded history (see
            # indicators.volume_profile's own docstring for the OHLC-based
            # approximation this uses, and why it honestly returns None for
            # Yahoo Forex, which never reports real volume). Rendered as a
            # dedicated overlay (bars anchored to the pane's own left edge,
            # via a real ict_chart primitive), not through the indicators=
            # pathway above — a price-bucketed histogram isn't a time
            # series, so it doesn't fit that shape.
            #
            # Anchored per the Settings tab's own picker — "Full history"
            # (default) keeps the original behavior (every bar currently
            # loaded for this timeframe); anything else trims to bars at or
            # after the chosen cutoff first, same idea as an Anchored Volume
            # Profile on other platforms. Requested directly: on a long
            # window (months of history) the busiest price can sit far from
            # where price is trading today, which reads as "wrong" even
            # when it's an honest reflection of the full period.
            volume_profile_buckets = []
            _vp_df = df
            # Set inside the block below when it actually runs — stays None
            # otherwise (feature off, or no usable volume), so the detail
            # expander further down can check it safely either way.
            _vp = None
            if show_volume_profile and vp_anchor != "Full history" and not df.empty:
                _vp_cutoff = None
                if vp_anchor == "Custom date" and vp_anchor_date is not None:
                    _vp_cutoff = pd.Timestamp(vp_anchor_date, tz=df.index.tz)
                elif vp_anchor == "Today":
                    _vp_cutoff = pd.Timestamp.now(tz=df.index.tz).normalize()
                elif vp_anchor == "This week":
                    _vp_now = pd.Timestamp.now(tz=df.index.tz)
                    _vp_cutoff = (_vp_now - pd.Timedelta(days=_vp_now.weekday())).normalize()
                elif vp_anchor == "This month":
                    _vp_cutoff = pd.Timestamp.now(tz=df.index.tz).normalize().replace(day=1)
                if _vp_cutoff is not None:
                    _vp_df = df[df.index >= _vp_cutoff]
            if show_volume_profile and not _vp_df.empty:
                _vp = volume_profile(_vp_df)
                if _vp is not None:
                    _hvn_idx = set(_vp["hvn_indices"])
                    _lvn_idx = set(_vp["lvn_indices"])
                    for _bi, _b in enumerate(_vp["buckets"]):
                        _is_poc = _b["price_low"] <= _vp["poc_price"] <= _b["price_high"]
                        # HVN = another real cluster besides POC (price likely
                        # stalls/consolidates here if revisited) — green, the
                        # same "stable/supportive" association this app's
                        # palette already uses elsewhere. LVN = a genuine
                        # thin spot (price likely moves FAST through if
                        # revisited, little resting interest to absorb it) —
                        # magenta, this app's existing "fast/volatile"
                        # association, kept at low opacity since the bar
                        # itself is already short by construction (a thin
                        # bucket) and doesn't need to also shout.
                        if _is_poc:
                            _vp_color = _hex_to_rgba(theme.NEON_CYAN, 0.55)
                        elif _bi in _hvn_idx:
                            _vp_color = _hex_to_rgba(theme.NEON_GREEN, 0.45)
                        elif _bi in _lvn_idx:
                            _vp_color = _hex_to_rgba(theme.NEON_MAGENTA, 0.30)
                        else:
                            _vp_color = _hex_to_rgba("#8e8e93", 0.30)
                        volume_profile_buckets.append({
                            "price_low": _b["price_low"], "price_high": _b["price_high"],
                            "volume_frac": _b["volume_frac"], "color": _vp_color,
                        })
                    _vp_t0 = _ny_fake_utc_seconds(_vp_df.index[0]) if vp_anchor != "Full history" else axis_secs[0]
                    price_lines.append({"t0": _vp_t0, "t1": future_edge, "price": _vp["poc_price"],
                                         "color": _hex_to_rgba(theme.NEON_CYAN, 0.9), "title": "POC", "above": True})
                    price_lines.append({"t0": _vp_t0, "t1": future_edge, "price": _vp["value_area_high"],
                                         "color": _hex_to_rgba(theme.NEON_AMBER, 0.7), "title": "VAH", "above": True})
                    price_lines.append({"t0": _vp_t0, "t1": future_edge, "price": _vp["value_area_low"],
                                         "color": _hex_to_rgba(theme.NEON_AMBER, 0.7), "title": "VAL", "above": False})
                    legend_items.append((f"Volume Profile ({_vp['shape_label']})",
                                          f"POC + {len(_hvn_idx)} HVN + {len(_lvn_idx)} LVN", theme.NEON_CYAN))

            # Every layer detects against whatever SINGLE timeframe its own
            # dropdown picked (see the Layers popover) — no more fixed 4h/
            # 15m pair, no more confluence filtering between an HTF and LTF
            # pass. When a layer's chosen timeframe is exactly the main
            # chart's own, this reuses the main chart's already-fetched
            # `df` directly rather than fetching it a second time.
            needed_tfs = {layer_tf[name] for name in layers}
            tf_frames = {}
            for req_tf in needed_tfs:
                if req_tf == tf_label:
                    rdf = df
                else:
                    req_conf = TIMEFRAMES[req_tf]
                    rdf = get_yf_ohlcv(ticker, period=req_conf["period"], interval=req_conf["fetch_interval"],
                                        provider=data_source)
                    # Checked before resample — same RangeIndex crash as
                    # the main chart's own fetch if this frame's fetch comes
                    # back empty while the main one didn't (e.g. a provider
                    # gap specific to this ticker+interval).
                    if rdf.empty:
                        continue
                    if req_conf["resample"]:
                        rdf = resample_ohlc(rdf, req_conf["resample"])
                if rdf.empty:
                    continue
                # Same reasoning as forming_bar_ts above: the last row of any
                # freshly-fetched intraday frame can still be actively
                # forming, not just this chart's own live splice — drop it
                # from detection either way.
                r_confirmed = rdf.iloc[:-1] if len(rdf) > 1 else rdf
                # Every detector below runs on r_confirmed, not rdf — so the
                # latest "end" any of them could ever report is
                # r_confirmed.index[-1], never rdf.index[-1] (one bar later).
                # Comparing against rdf's own last bar meant this NEVER
                # matched for a still-open/active zone or line, so nothing
                # ever actually extended to future_edge — every "active"
                # overlay silently stopped exactly where price had printed
                # so far instead of visibly continuing forward. Confirmed
                # directly: an open FVG's own "end" is set to n-1 (the last
                # scanned row, i.e. r_confirmed's own last row) in
                # detect_fvgs — comparing that against rdf's last row (one
                # bar ahead) was comparing two different rows by construction.
                _confirmed_last = r_confirmed.index[-1]
                tf_frames[req_tf] = {
                    "df": rdf, "confirmed": r_confirmed,
                    "t1_axis": (lambda end_ts, _last=_confirmed_last: future_edge if end_ts == _last
                                else _ny_fake_utc_seconds(end_ts)),
                }

            # A second timeframe's own candles, faded, on the SAME axis as
            # the main chart's — "overlaps perfectly" because each ghost
            # candle's box is drawn from its own real start/end timestamps
            # (see GhostCandlePrimitive in the frontend), not from bar
            # count, so a 4h ghost candle spans exactly the ~16 15m candles
            # (or however many) it actually covers. Independent of the
            # layer ref-frame toggles above on purpose — this is a display
            # choice, not part of structure detection, so it stays
            # available even with every layer switched off.
            ghost_candles = []
            if overlay_tf != "None" and overlay_tf != tf_label:
                ov_conf = TIMEFRAMES[overlay_tf]
                ovdf = get_yf_ohlcv(ticker, period=ov_conf["period"], interval=ov_conf["fetch_interval"], provider=data_source)
                # Same empty-before-resample ordering as everywhere else
                # this pattern shows up — resample_ohlc raises on an empty
                # frame's RangeIndex rather than just staying empty.
                if ov_conf["resample"] and not ovdf.empty:
                    ovdf = resample_ohlc(ovdf, ov_conf["resample"])
                # Capped at the most recent N bars, not the whole fetched
                # period — each ghost candle is its own drawn primitive (no
                # batched rendering the way a real series gets), so an
                # uncapped deep history here would mean redrawing hundreds
                # to thousands of individual shapes on every pan/zoom. 300
                # bars is generous context (50 days at 4h, ~1 year at 1D)
                # while staying cheap to repaint.
                ghost_candles.extend(_to_ghost_candles(ovdf.tail(300)))

            # Backfill history past where the main timeframe's own data
            # runs out (1m only ever has 7 days, 5m/15m/30m only 60, etc.
            # — see TIMEFRAMES) with the nearest genuinely-longer-history
            # timeframe's own real candles. Two problems, two different
            # fixes layered together:
            #
            # 1. lightweight-charts' timeScale flatly refuses to compute a
            #    coordinate for, or even scroll to, any time before the
            #    series' own first loaded bar — confirmed directly,
            #    timeToCoordinate returns null there and
            #    setVisibleLogicalRange silently clamps back to the real
            #    data's own span no matter how far negative a logical range
            #    is requested. A ghost-candle PRIMITIVE positioned there
            #    would just never draw. Fixed by prepending these bars into
            #    the chart's actual SERIES data instead, so the series'
            #    own scrollable range genuinely extends back that far.
            #
            # 2. But a real CandlestickSeries can't show a candle "wider"
            #    than any other — see the Ghost-candle primitive's own
            #    comment in the frontend: bar width comes from the chart's
            #    global barSpacing, not each bar's own duration, so a 1D
            #    backfill candle sitting in the same series as 1h candles
            #    renders exactly as narrow as they are — confirmed directly
            #    (identical pixel spacing measured across a 4h-to-1h
            #    boundary, timeToCoordinate returning evenly-spaced x's
            #    regardless of the real 4x time gap). Fixed by making
            #    these particular series bars fully transparent (real
            #    data, invisible rendering) and drawing the actual visible
            #    shape as ghost-candle primitives on top instead — which
            #    NOW resolve to real coordinates, since the invisible
            #    anchor bars extended the series' scrollable range for
            #    them to land on.
            #
            # `df` and everything derived from it (tf_frames, every
            # layer's own detection) stay untouched either way — only the
            # chart's own display gets these extra bars.
            # Deep-coverage backfill: the single "one tier up" candidate
            # only guarantees the MAIN timeframe's own history looks
            # continuous. A layer set to an even DEEPER timeframe than
            # that (e.g. main chart on 15m, FVG set to 1D) would still
            # have zones whose t0/t1 fall before this series' own earliest
            # bar — same null-coordinate problem as above, just triggered
            # by a layer's own choice instead of the main chart's. Confirmed
            # directly this was the actual mechanism behind FVG/OB zones
            # rendering inconsistently depending on the main chart's own
            # timeframe: whichever layer's chosen tf reached further back
            # than the series' own loaded range simply vanished, silently.
            #
            # tf_frames (built above) already holds every active layer's
            # own full dataframe — reused here instead of fetching
            # anything new: among the single-tier candidate and every
            # tf_frame's own df (each trimmed to before df.index[0]),
            # whichever reaches furthest back wins. When no active layer
            # needs more depth than the single-tier default already
            # covers, that candidate wins by construction — unchanged
            # behavior for the common case.
            _backfill_tf = _backfill_timeframe(tf_label)
            _bf_candidates = []  # (source timeframe label, df)
            if _backfill_tf is not None and not df.empty:
                _bf_conf = TIMEFRAMES[_backfill_tf]
                _tier_bfdf = get_yf_ohlcv(ticker, period=_bf_conf["period"], interval=_bf_conf["fetch_interval"],
                                           provider=data_source)
                if _bf_conf["resample"] and not _tier_bfdf.empty:
                    _tier_bfdf = resample_ohlc(_tier_bfdf, _bf_conf["resample"])
                if not _tier_bfdf.empty:
                    _bf_candidates.append((_backfill_tf, _tier_bfdf[_tier_bfdf.index < df.index[0]]))
            if not df.empty:
                for _tf_key, _frame in tf_frames.items():
                    _trimmed = _frame["df"][_frame["df"].index < df.index[0]]
                    if not _trimmed.empty:
                        _bf_candidates.append((_tf_key, _trimmed))
            _bf_candidates = [(k, c) for k, c in _bf_candidates if not c.empty]
            if _bf_candidates:
                _bf_src_tf, bfdf = min(_bf_candidates, key=lambda kc: kc[1].index[0])
                # 1500-bar cap — each ghost candle AND each invisible
                # anchor bar adds to what the chart has to track, so
                # uncapped depth would mean redrawing thousands of shapes
                # on every pan/zoom. Applied once, up front, so the anchor
                # bars and the ghost candles drawn over them always cover
                # the exact same span — previously these could silently
                # diverge (anchor bars used the full frame, ghost candles
                # capped separately), harmless before this change since
                # the single-tier candidate was always small, but not safe
                # to keep now that the source can be an arbitrary layer's
                # own deep timeframe.
                bfdf = bfdf.tail(1500)
                # The tiling loop below rebuilds up to 1500 anchor bars AND
                # up to _MAX_TOTAL_TILES (5000) ghost candles from scratch —
                # real dict/float-cast work, not free — and this whole
                # block runs unconditionally on EVERY fragment tick
                # (run_every=1-5s on an intraday timeframe), regardless of
                # whether bfdf itself actually changed. Confirmed directly
                # as a real, live contributor to a genuine multi-minute
                # stall (caught via lldb mid-hang, this time landing in
                # CPython's own float-to-string conversion — the cost of
                # re-serializing thousands of freshly-rebuilt dicts to the
                # component every tick) on top of the separately-fixed
                # per-timestamp tz-conversion loop: the backfill SOURCE
                # (bfdf) only changes when a new bar closes on whichever
                # coarser timeframe it's drawn from — hours apart, never
                # every tick — so recomputing this every single second was
                # pure waste stacking on top of everything else the same
                # tick already pays for. Cached the same way the mini HTF/
                # LTF panels' own overlays already are: keyed on exactly
                # what the loop below actually depends on, reused verbatim
                # whenever none of that has changed since the last tick.
                _bf_cache_key = (ticker, tf_label, _bf_src_tf,
                                  bfdf.index[0], bfdf.index[-1], len(bfdf))
                _bf_cached = st.session_state.get("_main_backfill_cache")
                if _bf_cached is not None and _bf_cached["key"] == _bf_cache_key:
                    backfill_bars, _bf_ghosts = _bf_cached["backfill_bars"], _bf_cached["bf_ghosts"]
                    bars = backfill_bars + bars
                    ghost_candles.extend(_bf_ghosts)
                    _bf_needs_compute = False
                else:
                    _bf_needs_compute = True

            if _bf_candidates and _bf_needs_compute:
                bf_o, bf_h, bf_l, bf_c = (("Open", "High", "Low", "Close") if "Close" in bfdf
                                           else ("open", "high", "low", "close"))
                _invisible = "rgba(0,0,0,0)"
                # Tile each real backfill candle across as many
                # native-timeframe-duration slots as it actually spans,
                # instead of one anchor bar per real candle regardless of
                # source granularity. lightweight-charts spaces every bar
                # by uniform INDEX, not by real elapsed time — so a single
                # coarser candle (e.g. one real month backfilling a weekly
                # chart) would occupy the exact same on-screen width as a
                # native bar, silently compressing ~4 weeks of real time
                # into the space of one and breaking the time axis's pace
                # right at that boundary (confirmed: this is what made
                # deep-zoomed history look "stretched"/misscaled). Floor
                # (not round) so the last slot always lands strictly
                # before the next real row's own timestamp, which
                # lightweight-charts requires for strictly-ascending
                # series data. Repeating the same OHLC across every slot
                # is an honest "we don't have finer data this far back"
                # staircase, not fabricated detail.
                _native_secs = _TF_BAR_SECONDS[tf_label]
                _bf_times = _ny_fake_utc_seconds_vec(bfdf.index)
                _bf_boundary = _ny_fake_utc_seconds(df.index[0])
                # One shared tiling granularity for the WHOLE backfilled
                # span, not native_secs scaled per row — a per-row count
                # scaled by a single global ratio was tried first and
                # confirmed directly to still break consistency: the ratio
                # scales down each row's SLOT COUNT, but every row still
                # spaced its own slots at the full native_secs apart, so a
                # row that floored down to 1 slot left a huge leftover gap
                # (tens of hours) sitting right next to a row that floored
                # to a fully-spaced few dozen 60-second slots — the exact
                # "different rates" symptom this exists to fix, just
                # relocated. Deriving ONE effective_secs from the total
                # span up front and using it for BOTH the slot count and
                # the slot spacing, on every row alike, is what actually
                # guarantees a uniform real-time-per-index ratio end to
                # end — never coarser than the main chart's own native
                # bar (no reason to out-resolve it), only ever widened
                # when the full-resolution total would exceed the budget.
                _bf_span = max(1, _bf_boundary - _bf_times[0]) if _bf_times else 1
                _effective_secs = max(_native_secs, -(-_bf_span // _MAX_TOTAL_TILES))  # ceil div
                # .to_numpy() once, not bfdf[col].iloc[i] per cell per row —
                # same fix, same reasoning, as the main bars list above.
                _bf_o_arr, _bf_h_arr, _bf_l_arr, _bf_c_arr = (
                    bfdf[bf_o].to_numpy(), bfdf[bf_h].to_numpy(),
                    bfdf[bf_l].to_numpy(), bfdf[bf_c].to_numpy())
                backfill_bars = []
                _bf_ghosts = []
                for i in range(len(bfdf)):
                    _row_t = _bf_times[i]
                    _next_t = _bf_times[i + 1] if i + 1 < len(_bf_times) else _bf_boundary
                    _n_slots = max(1, int((_next_t - _row_t) // _effective_secs))
                    _o = float(_bf_o_arr[i]); _h = float(_bf_h_arr[i])
                    _l = float(_bf_l_arr[i]); _c = float(_bf_c_arr[i])
                    _ghost_color = _hex_to_rgba(theme.NEON_CYAN if _c >= _o else theme.NEON_MAGENTA, 0.16)
                    for _s in range(_n_slots):
                        _t = _row_t + _s * _effective_secs
                        backfill_bars.append({
                            "time": _t, "open": _o, "high": _h, "low": _l, "close": _c, "volume": 0.0,
                            "color": _invisible, "borderColor": _invisible, "wickColor": _invisible,
                        })
                        _t_next = _row_t + (_s + 1) * _effective_secs if _s + 1 < _n_slots else _next_t
                        _bf_ghosts.append({"t0": _t, "t1": _t_next, "open": _o, "high": _h, "low": _l,
                                            "close": _c, "color": _ghost_color})
                bars = backfill_bars + bars
                ghost_candles.extend(_bf_ghosts)
                st.session_state["_main_backfill_cache"] = {
                    "key": _bf_cache_key, "backfill_bars": backfill_bars, "bf_ghosts": _bf_ghosts,
                }

            # Kill zones: a shaded vertical band per day each selected
            # session occurred on, rather than trimming the candles down to
            # just that window — the rest of the day stays visible, each
            # zone just gets called out, one color per session so several
            # selected at once stay distinguishable. Same 5-day recency
            # window as the other structure overlays, same reasoning:
            # otherwise a 60-day 30m chart draws dozens of bands per
            # session. Each band spans the full price range actually traded
            # in that window (with a little padding) rather than the whole
            # chart's range, so it reads as "this slice of time," not a
            # random stripe unrelated to what happened in it.
            if selected_kill_zones and tf_label in INTRADAY_TFS:
                idx_ny = recent_df.index.tz_convert("America/New_York")
                days = sorted(set(idx_ny.normalize()))
                for kz_name in selected_kill_zones:
                    kz_color = KILL_ZONE_COLORS[kz_name]
                    for kz_start, kz_end in KILL_ZONES[kz_name]:
                        for day in days:
                            if day.weekday() >= 5:
                                continue
                            band_start = pd.Timestamp.combine(day.date(), kz_start).tz_localize("America/New_York")
                            band_end = pd.Timestamp.combine(day.date(), kz_end).tz_localize("America/New_York")
                            in_band = (df.index >= band_start) & (df.index < band_end)
                            if not in_band.any():
                                continue
                            # lightweight-charts' timeToCoordinate returns null
                            # for a time that isn't an actual bar and can't be
                            # interpolated (confirmed directly: NY PM's 16:00:00
                            # end edge has no bar to land on — regular-hours data
                            # stops at 15:59:00 — so x2 came back null and the
                            # primitive's own null-guard silently skipped drawing
                            # the ENTIRE rectangle, label included, while NY AM's
                            # edges happened to line up with real bars and worked
                            # fine). Snapping to the first/last bar actually inside
                            # the window guarantees both edges resolve.
                            band_bars = df.index[in_band]
                            pad = (df.loc[in_band, h_col].max() - df.loc[in_band, l_col].min()) * 0.1
                            rectangles.append({
                                "t0": _ny_fake_utc_seconds(band_bars[0]), "t1": _ny_fake_utc_seconds(band_bars[-1]),
                                "p0": float(df.loc[in_band, l_col].min() - pad),
                                "p1": float(df.loc[in_band, h_col].max() + pad),
                                "fill": _hex_to_rgba(kz_color, 0.16), "border": _hex_to_rgba(kz_color, 0.6),
                                "label": KILL_ZONE_LABELS[kz_name],
                            })

            if "Price Projection" in layers:
                _proj = _compute_price_projection(df, c_col, n_bars=price_projection_bars)
                if _proj is not None:
                    _proj_bar_secs = _TF_BAR_SECONDS[tf_label]
                    _proj_n = len(_proj["center"])
                    for _pi in range(_proj_n):
                        _pt0 = _entry_x + _pi * _proj_bar_secs
                        _pt1 = _entry_x + (_pi + 1) * _proj_bar_secs
                        # Outer 2-sigma band first (wider, fainter), inner
                        # 1-sigma drawn right after so it layers visually
                        # on top as a "core" region within the wider one —
                        # same z-order-by-append-order every other layer
                        # here already relies on.
                        rectangles.append({"t0": _pt0, "t1": _pt1, "p0": _proj["lower2"][_pi],
                                            "p1": _proj["upper2"][_pi],
                                            "fill": _hex_to_rgba(theme.NEON_CYAN, 0.05), "border": None})
                        rectangles.append({"t0": _pt0, "t1": _pt1, "p0": _proj["lower1"][_pi],
                                            "p1": _proj["upper1"][_pi],
                                            "fill": _hex_to_rgba(theme.NEON_CYAN, 0.10), "border": None})
                    for _pi in range(_proj_n):
                        _pt0 = _entry_x + _pi * _proj_bar_secs
                        _pt1 = _entry_x + (_pi + 1) * _proj_bar_secs
                        # price_lines has no native slope (a straight
                        # horizontal segment per t0/t1, see ict_chart's own
                        # docstring) — chaining one short segment per future
                        # bar, each at that bar's own projected price,
                        # approximates the sloped trend line as a fine
                        # staircase. Label only on the LAST segment — one
                        # honest disclaimer, not the same text repeated
                        # n_bars times down the line.
                        price_lines.append({
                            "t0": _pt0, "t1": _pt1, "price": _proj["center"][_pi],
                            "color": _hex_to_rgba(theme.NEON_AMBER, 0.9), "line_width": 1, "dashed": True,
                            "title": "Trend extrapolation — NOT a validated prediction" if _pi == _proj_n - 1 else "",
                            "above": True,
                        })

            if "FVG" in layers:
                rf = tf_frames.get(layer_tf["FVG"])
                if rf is not None:
                    all_fvgs = detect_fvgs(rf["confirmed"], min_body_ratio=displacement_ratio)
                    fvgs = all_fvgs if show_mitigated else [g for g in all_fvgs if not g["filled"]]
                    # Two deliberately different selections merged together
                    # — direct request, for the whole detection system, not
                    # just replay: "one that tracks previous candles, and
                    # one that scans historical values." recent_zone_tracker
                    # (Engine A) adds every zone from the last 50 bars,
                    # uncapped; _nearest_by_price (Engine B, this layer's
                    # own pre-existing selection, unchanged) still finds the
                    # nearest max_items_per_layer above/below price from the
                    # WHOLE history. Both feed the exact same downstream
                    # win-rate/MA+FVG/indicator-confluence/render loop below
                    # — a zone from either engine is drawn identically.
                    fvgs = merge_zone_engines(
                        recent_zone_tracker(fvgs, rf["confirmed"], lookback_bars=50),
                        _nearest_by_price(fvgs, current_price, max_items_per_layer))
                    win_rates = _historical_win_rates(ticker, layer_tf["FVG"], "FVG", data_source,
                                                       exit_types=_win_rate_exit_types,
                                                       news_blackout=_news_blackout)
                    # A separate lookup, not per-zone — MA+FVG's own win
                    # rate (touch anchored to the EMA's level, not "touched
                    # anywhere in the gap") is a different, more specific
                    # stat than plain FVG's, shown INSTEAD of it for zones
                    # that are also MA+FVG matches.
                    ma_fvg_win_rates = _historical_win_rates(ticker, layer_tf["FVG"], "MA+FVG", data_source,
                                                              exit_types=_win_rate_exit_types,
                                                              news_blackout=_news_blackout)
                    # General indicator confluence (RSI/MACD/EMA, whichever
                    # the "Indicator confluence" control has selected) — a
                    # SEPARATE, additional check from the MA+FVG strategy
                    # tag above (that one is specifically EMA20/50; this is
                    # a plain visual/stat overlay on top, any indicator the
                    # user picked). Empty selection = no extra work at all.
                    _rf_close = rf["confirmed"]["Close"] if "Close" in rf["confirmed"] else rf["confirmed"]["close"]
                    fvg_confluence = zone_indicator_matches(fvgs, _rf_close, _confluence_keys) if _confluence_keys else {}
                    for g in fvgs:
                        color = theme.NEON_CYAN if g["type"] == "bullish" else theme.NEON_MAGENTA
                        wr = win_rates.get(g["type"])
                        # Label only renders on-chart for zones that already
                        # have a border (the frontend's own rule — see
                        # ict_chart/frontend/index.html's RectangleRenderer),
                        # which here means open ones. A filled gap's
                        # historical win rate isn't wrong to know, just not
                        # worth cluttering the chart for something no longer
                        # live — it's still in the detail table below either way.
                        label = _win_rate_label(wr)
                        # See feedback_ui_obviousness: an MA+FVG match needs
                        # to be visible AT the zone itself, not just named in
                        # the sidebar — a plain FVG and one the MA+FVG
                        # strategy also fired on must never look the same.
                        is_ma_fvg = g["start"] in _ma_fvg_starts
                        if is_ma_fvg:
                            ma_label = _win_rate_label(ma_fvg_win_rates.get(g["type"]))
                            label = f"{ma_label} · MA+FVG" if ma_label else "MA+FVG"
                        matched_inds = fvg_confluence.get(g["start"], [])
                        if matched_inds:
                            ind_tag = "+".join(INDICATOR_SPECS[k]["label"] for k in matched_inds)
                            label = f"{label} · {ind_tag}" if label else ind_tag
                        is_highlighted = is_ma_fvg or bool(matched_inds)
                        t0 = _ny_fake_utc_seconds(g["start"])
                        t1 = rf["t1_axis"](g["end"])
                        if g["filled"]:
                            # detect_fvgs' own "top"/"bottom" are the ACTIVE
                            # (eaten-into) edges — for a fully filled gap
                            # those clamp to each other, i.e. a ~zero-height
                            # box, effectively invisible. "raw_top"/
                            # "raw_bottom" are the gap's original, full-size
                            # boundaries — what show_mitigated is actually
                            # for: seeing where the gap WAS.
                            rectangles.append({
                                "t0": t0, "t1": t1, "p0": g["raw_bottom"], "p1": g["raw_top"],
                                "fill": _hex_to_rgba(color, zone_opacity),
                                "fill_to": _hex_to_rgba(color, 0.0), "border": None,
                                "label": None, "border_width": 1,
                            })
                        else:
                            # Still-open: the untested remainder at full
                            # strength. Consequent encroachment ("each wick
                            # that clips into it eats the zone away from one
                            # side" — see detect_fvgs' own comment) can have
                            # already chewed into part of a zone that's still
                            # technically open — rather than let that eaten
                            # slice just silently vanish (the box quietly
                            # shrinking with no trace of what used to be
                            # there), draw it as a second, faded rectangle
                            # alongside the live remainder: how much of this
                            # gap price has already used up, at a glance.
                            rectangles.append({
                                "t0": t0, "t1": t1, "p0": g["bottom"], "p1": g["top"],
                                "fill": _hex_to_rgba(color, zone_opacity),
                                "border": theme.CONFLUENCE_GOLD if is_highlighted else color,
                                "label": label,
                                "border_width": 3 if is_highlighted else 1,
                            })
                            if g["type"] == "bullish" and g["top"] < g["raw_top"]:
                                rectangles.append({
                                    "t0": t0, "t1": t1, "p0": g["top"], "p1": g["raw_top"],
                                    "fill": _hex_to_rgba(color, zone_opacity),
                                    "fill_to": _hex_to_rgba(color, 0.0), "border": None,
                                    "label": None, "border_width": 1,
                                })
                            elif g["type"] == "bearish" and g["bottom"] > g["raw_bottom"]:
                                rectangles.append({
                                    "t0": t0, "t1": t1, "p0": g["raw_bottom"], "p1": g["bottom"],
                                    "fill": _hex_to_rgba(color, zone_opacity),
                                    "fill_to": _hex_to_rgba(color, 0.0), "border": None,
                                    "label": None, "border_width": 1,
                                })
                        fvg_ob_rows.append({
                            "layer": "FVG", "tf": layer_tf["FVG"], "type": g["type"],
                            "top": g["top"], "bottom": g["bottom"],
                            "start": g["start"], "end": g["end"],
                            "status": "filled" if g["filled"] else "open",
                            "first_touch": g["first_touch"],
                            "hist_win_rate": _win_rate_label(wr, precision=1) or "insufficient data",
                        })
                        clickable_zones.append({
                            "kind": "rect", "layer": "FVG", "event_type": "fvg", "type": g["type"],
                            "top": g["raw_top"] if g["filled"] else g["top"],
                            "bottom": g["raw_bottom"] if g["filled"] else g["bottom"],
                            "start": g["start"], "end": g["end"], "df_ref": rf["confirmed"],
                        })
                    open_n = sum(1 for g in fvgs if not g["filled"])
                    legend_items.append((f"FVG ({layer_tf['FVG']})", f"{len(fvgs)} shown · {open_n} open", theme.NEON_CYAN))

            if "IFVG" in layers:
                rf = tf_frames.get(layer_tf["IFVG"])
                if rf is not None:
                    ifvgs = detect_ifvgs(rf["confirmed"], min_body_ratio=displacement_ratio)
                    # Same two-engine merge as every other zone layer — see
                    # the FVG block's own comment above. No "still open"
                    # eaten/encroachment shading here (see detect_ifvgs' own
                    # docstring: unlike an FVG, an IFVG's own top/bottom are
                    # fixed at the original gap's full formation size, not
                    # something that shrinks as price re-tests it), and no
                    # historical win-rate lookup yet — a new-enough concept
                    # that this project doesn't have its own win-rate
                    # methodology for it, unlike FVG/Order Block/Liquidity.
                    ifvgs = merge_zone_engines(
                        recent_zone_tracker(ifvgs, rf["confirmed"], lookback_bars=50),
                        _nearest_by_price(ifvgs, current_price, max_items_per_layer))
                    for z in ifvgs:
                        color = theme.NEON_CYAN if z["type"] == "bullish" else theme.NEON_MAGENTA
                        rectangles.append({
                            "t0": _ny_fake_utc_seconds(z["start"]), "t1": future_edge,
                            "p0": z["bottom"], "p1": z["top"],
                            "fill": _hex_to_rgba(color, zone_opacity), "border": color,
                            "label": "IFVG", "border_width": 2,
                        })
                        fvg_ob_rows.append({
                            "layer": "IFVG", "tf": layer_tf["IFVG"], "type": z["type"],
                            "top": z["top"], "bottom": z["bottom"],
                            "start": z["start"], "end": z["start"],
                            "status": "touched" if z["first_touch"] else "untouched",
                            "first_touch": z["first_touch"], "hist_win_rate": "not tracked yet",
                        })
                        clickable_zones.append({
                            "kind": "rect", "layer": "IFVG", "event_type": None, "type": z["type"],
                            "top": z["top"], "bottom": z["bottom"],
                            "start": z["start"], "end": rf["confirmed"].index[-1], "df_ref": rf["confirmed"],
                        })
                    legend_items.append((f"IFVG ({layer_tf['IFVG']})", f"{len(ifvgs)} shown", theme.NEON_CYAN))

            if "Order Blocks" in layers:
                rf = tf_frames.get(layer_tf["Order Blocks"])
                if rf is not None:
                    all_obs = detect_order_blocks(rf["confirmed"], min_body_ratio=displacement_ratio)
                    obs = all_obs if show_mitigated else [o for o in all_obs if not o["mitigated"]]
                    # Same two-engine merge as the FVG layer above — see
                    # its own comment.
                    obs = merge_zone_engines(
                        recent_zone_tracker(obs, rf["confirmed"], lookback_bars=50),
                        _nearest_by_price(obs, current_price, max_items_per_layer))
                    win_rates = _historical_win_rates(ticker, layer_tf["Order Blocks"], "Order Block", data_source,
                                                       exit_types=_win_rate_exit_types,
                                                       news_blackout=_news_blackout)
                    _rf_close = rf["confirmed"]["Close"] if "Close" in rf["confirmed"] else rf["confirmed"]["close"]
                    ob_confluence = zone_indicator_matches(obs, _rf_close, _confluence_keys) if _confluence_keys else {}
                    for ob in obs:
                        color = theme.NEON_GREEN if ob["type"] == "bullish" else theme.NEON_AMBER
                        opacity = zone_opacity * (0.3 if ob["mitigated"] else 1.0)
                        wr = win_rates.get(ob["type"])
                        label = _win_rate_label(wr)
                        matched_inds = ob_confluence.get(ob["start"], [])
                        if matched_inds:
                            ind_tag = "+".join(INDICATOR_SPECS[k]["label"] for k in matched_inds)
                            label = f"{label} · {ind_tag}" if label else ind_tag
                        ob_border = None if ob["mitigated"] else (theme.CONFLUENCE_GOLD if matched_inds else color)
                        rectangles.append({
                            "t0": _ny_fake_utc_seconds(ob["start"]), "t1": rf["t1_axis"](ob["end"]),
                            "p0": ob["bottom"], "p1": ob["top"],
                            "fill": _hex_to_rgba(color, opacity), "border": ob_border,
                            "label": label,
                            "border_width": 3 if matched_inds else 1,
                        })
                        fvg_ob_rows.append({
                            "layer": "Order Block", "tf": layer_tf["Order Blocks"], "type": ob["type"],
                            "top": ob["top"], "bottom": ob["bottom"],
                            "start": ob["start"], "end": ob["end"],
                            "status": "mitigated" if ob["mitigated"] else "unmitigated",
                            "first_touch": ob["first_touch"],
                            "hist_win_rate": _win_rate_label(wr, precision=1) or "insufficient data",
                        })
                        clickable_zones.append({
                            "kind": "rect", "layer": "Order Blocks", "event_type": "order_block",
                            "type": ob["type"], "top": ob["top"], "bottom": ob["bottom"],
                            "start": ob["start"], "end": ob["end"], "df_ref": rf["confirmed"],
                        })
                    unmit = sum(1 for o in obs if not o["mitigated"])
                    legend_items.append((f"Order Blocks ({layer_tf['Order Blocks']})", f"{len(obs)} shown · {unmit} unmit.", theme.NEON_GREEN))

            if "Breaker Block" in layers:
                rf = tf_frames.get(layer_tf["Breaker Block"])
                if rf is not None:
                    breakers = detect_breaker_blocks(rf["confirmed"], min_body_ratio=displacement_ratio)
                    # Same reasoning as the IFVG block above, applied to
                    # Order Blocks instead of FVGs — see detect_breaker_
                    # blocks' own docstring.
                    breakers = merge_zone_engines(
                        recent_zone_tracker(breakers, rf["confirmed"], lookback_bars=50),
                        _nearest_by_price(breakers, current_price, max_items_per_layer))
                    for z in breakers:
                        color = theme.NEON_GREEN if z["type"] == "bullish" else theme.NEON_AMBER
                        rectangles.append({
                            "t0": _ny_fake_utc_seconds(z["start"]), "t1": future_edge,
                            "p0": z["bottom"], "p1": z["top"],
                            "fill": _hex_to_rgba(color, zone_opacity), "border": color,
                            "label": "Breaker", "border_width": 2,
                        })
                        fvg_ob_rows.append({
                            "layer": "Breaker Block", "tf": layer_tf["Breaker Block"], "type": z["type"],
                            "top": z["top"], "bottom": z["bottom"],
                            "start": z["start"], "end": z["start"],
                            "status": "touched" if z["first_touch"] else "untouched",
                            "first_touch": z["first_touch"], "hist_win_rate": "not tracked yet",
                        })
                        clickable_zones.append({
                            "kind": "rect", "layer": "Breaker Block", "event_type": None, "type": z["type"],
                            "top": z["top"], "bottom": z["bottom"],
                            "start": z["start"], "end": rf["confirmed"].index[-1], "df_ref": rf["confirmed"],
                        })
                    legend_items.append((f"Breaker Block ({layer_tf['Breaker Block']})", f"{len(breakers)} shown", theme.NEON_GREEN))

            if "Swing Points" in layers:
                rf = tf_frames.get(layer_tf["Swing Points"])
                if rf is not None:
                    # rf["confirmed"], not rf["df"] — this used to run on the
                    # still-forming last bar too, which for a live-splicing
                    # timeframe changes on almost every tick (any new local
                    # high/low), defeating detect_swings' own @st.cache_data:
                    # a different last row means a different hash means a
                    # full rescan, every time. rf["confirmed"] only changes
                    # when a bar actually closes (confirmed empirically: two
                    # get_latest_bars fetches 9s apart returned byte-identical
                    # values for every already-closed row), so this now hits
                    # cache the same way every other detector call already
                    # does via rf["confirmed"] elsewhere in this function.
                    highs, lows = detect_swings(rf["confirmed"], atr_period=swing_atr_period, atr_mult=swing_atr_mult)
                    highs = highs[-max_items_per_layer:]
                    lows = lows[-max_items_per_layer:]
                    hi_color = _hex_to_rgba(theme.NEON_MAGENTA, 1.0)
                    lo_color = _hex_to_rgba(theme.NEON_GREEN, 1.0)
                    for p in highs:
                        markers.append({"time": _ny_fake_utc_seconds(p["time"]), "position": "aboveBar",
                                         "color": hi_color, "shape": "arrowDown"})
                        clickable_zones.append({"kind": "point", "layer": "Swing Points", "event_type": None,
                                                 "type": None, "price": p["price"], "time": p["time"],
                                                 "df_ref": rf["confirmed"]})
                    for p in lows:
                        markers.append({"time": _ny_fake_utc_seconds(p["time"]), "position": "belowBar",
                                         "color": lo_color, "shape": "arrowUp"})
                        clickable_zones.append({"kind": "point", "layer": "Swing Points", "event_type": None,
                                                 "type": None, "price": p["price"], "time": p["time"],
                                                 "df_ref": rf["confirmed"]})
                    legend_items.append((f"Swings ({layer_tf['Swing Points']})", f"{len(highs)}▲ {len(lows)}▼", theme.NEON_MAGENTA))

            if "Equal Highs/Lows" in layers:
                # Directional stop, not a wick touch: an EQH is a
                # resistance-type level, so only a candle actually CLOSING
                # above it counts as a break — a long upper wick poking
                # through without closing there hasn't invalidated the
                # level, it's arguably confirmed it (rejected right at
                # resistance). EQL mirrors this on closes below. Same
                # close-based logic detect_structure_breaks already uses
                # for BOS/CHoCH, just applied to this layer's own lines.
                #
                # Searched from the group's FIRST point, not its last:
                # detect_equal_levels groups purely by price proximity, with
                # no idea whether price already closed through that level
                # in between two "equal" touches — confirmed directly, a
                # 1W EQH pairing a 2021 high with an unrelated 2026 high
                # that just happened to land at a similar price kept
                # drawing straight through a real, decisive close-above in
                # March 2024 because the old search only looked for breaks
                # AFTER the group's LAST point (2026), skipping right over
                # the one in between. Searching from the first point
                # instead catches that. `exclude` keeps the group's own
                # member candles from counting as self-inflicted breaks —
                # those sit right at the shared price by construction and
                # can occasionally close a hair beyond the group's averaged
                # price, which would otherwise cut a freshly-formed
                # cluster's line the instant its own second point forms.
                rf = tf_frames.get(layer_tf["Equal Highs/Lows"])
                if rf is not None:
                    # rf["confirmed"] — see Swing Points' own comment above.
                    highs, lows = detect_swings(rf["confirmed"], atr_period=swing_atr_period, atr_mult=swing_atr_mult)
                    eq_count = 0
                    for group, color, label, direction in [
                        (detect_equal_levels(highs, tolerance=eq_tolerance), theme.NEON_MAGENTA, "EQH", "above"),
                        (detect_equal_levels(lows, tolerance=eq_tolerance), theme.NEON_GREEN, "EQL", "below"),
                    ]:
                        recent = sorted(group, key=lambda pts: max(p["time"] for p in pts))[-max_items_per_layer:]
                        for pts in recent:
                            times = sorted(p["time"] for p in pts)
                            price = sum(p["price"] for p in pts) / len(pts)
                            t1 = _line_stop_time(rf["confirmed"], times[0], price, direction=direction, exclude=set(times))
                            price_lines.append({"t0": _ny_fake_utc_seconds(times[0]), "t1": rf["t1_axis"](t1), "price": price,
                                                 "color": _hex_to_rgba(color, 1.0),
                                                 "title": f"{label} ({layer_tf['Equal Highs/Lows']})",
                                                 "above": direction == "above"})
                            clickable_zones.append({
                                "kind": "level", "layer": "Equal Highs/Lows", "event_type": "equal_highs_lows",
                                "type": "bearish" if direction == "above" else "bullish",
                                "price": price, "start": times[0], "end": t1, "df_ref": rf["confirmed"],
                            })
                            eq_count += 1
                    legend_items.append((f"Equal H/L ({layer_tf['Equal Highs/Lows']})", f"{eq_count} clusters", theme.NEON_MAGENTA))

            if "Market Structure" in layers:
                # BOS (Break of Structure, continuation) vs CHoCH (Change of
                # Character, the first close against the current trend bias
                # — the actually-actionable one) — see detect_structure_breaks'
                # own docstring for the bias-tracking logic. Colored the same
                # way research/live_scan.py's agent settings describe them:
                # CHoCH always amber regardless of direction (its own "pay
                # attention" identity), BOS colored by direction like every
                # other bullish/bearish layer on this chart.
                #
                # Drawn the same way Equal Highs/Lows draws its own levels —
                # a horizontal line from where the broken swing point
                # actually formed, stopping at the candle that broke it.
                # No _line_stop_time needed here the way EQH/EQL needs it:
                # b["end"] already IS the exact candle whose CLOSE broke the
                # level (the same candle detect_structure_breaks itself
                # fired the break on), so it's always a real, already-broken
                # stop point, never a "still live" one to extend into the
                # future. Replaces the old square/circle markers, which sat
                # on top of Swing Points' own arrows and made a confirmed
                # break unreadable on a busy chart — a level line needs no
                # separate shape vocabulary to stay legible.
                rf = tf_frames.get(layer_tf["Market Structure"])
                if rf is not None:
                    breaks = detect_structure_breaks(rf["confirmed"], mode=ms_mode,
                                                      atr_period=swing_atr_period, atr_mult=swing_atr_mult)
                    breaks = breaks[-max_items_per_layer:]
                    bos_n = choch_n = 0
                    for b in breaks:
                        is_choch = b["structure"] == "CHoCH"
                        color = theme.NEON_AMBER if is_choch else (theme.NEON_GREEN if b["type"] == "bullish" else theme.NEON_MAGENTA)
                        price_lines.append({"t0": _ny_fake_utc_seconds(b["start"]), "t1": _ny_fake_utc_seconds(b["end"]),
                                             "price": b["level"], "color": _hex_to_rgba(color, 1.0),
                                             "title": f"{b['structure']} ({layer_tf['Market Structure']})",
                                             "above": b["type"] == "bullish"})
                        clickable_zones.append({
                            "kind": "level", "layer": "Market Structure",
                            "event_type": "choch" if is_choch else "bos", "type": b["type"],
                            "price": b["level"], "start": b["start"], "end": b["end"], "df_ref": rf["confirmed"],
                        })
                        if is_choch:
                            choch_n += 1
                        else:
                            bos_n += 1
                    legend_items.append((f"Structure ({layer_tf['Market Structure']})", f"{bos_n} BOS · {choch_n} CHoCH", theme.NEON_AMBER))

            if "Premium/Discount" in layers:
                # The box is the current dealing range itself — the last
                # swing high and swing low price hasn't closed back through
                # yet (current_dealing_range reuses the exact pending-high/
                # pending-low tracking Market Structure's BOS/CHoCH uses),
                # not an arbitrary "last N days" window. It stays anchored
                # to those two swing points and extends into the future
                # until one side is actually broken, at which point the
                # NEXT rerun's swing/break state has already moved that
                # boundary — this box just always reflects wherever that
                # state currently sits.
                rf = tf_frames.get(layer_tf["Premium/Discount"])
                if rf is not None:
                    dr = current_dealing_range(rf["confirmed"], atr_period=swing_atr_period, atr_mult=swing_atr_mult)
                    if dr is not None:
                        t0 = _ny_fake_utc_seconds(dr["start"])
                        # Zone opacity here is dampened relative to the same
                        # slider's FVG/Order Block use — those are small
                        # event boxes; this one spans most of the chart's
                        # visible height, so full strength would wash
                        # everything else out.
                        fill_opacity = zone_opacity * 0.4
                        rectangles.append({"t0": t0, "t1": future_edge, "p0": dr["eq"], "p1": dr["top"],
                                            "fill": _hex_to_rgba(theme.NEON_MAGENTA, fill_opacity), "border": None})
                        rectangles.append({"t0": t0, "t1": future_edge, "p0": dr["bottom"], "p1": dr["eq"],
                                            "fill": _hex_to_rgba(theme.NEON_GREEN, fill_opacity), "border": None})
                        # Premium half (above EQ) = bearish bias, discount
                        # half (below EQ) = bullish — same "which side of
                        # fair value" reading this layer's own legend uses.
                        clickable_zones.append({
                            "kind": "rect", "layer": "Premium/Discount", "event_type": None, "type": "bearish",
                            "top": dr["top"], "bottom": dr["eq"], "start": dr["start"],
                            "end": rf["confirmed"].index[-1], "df_ref": rf["confirmed"],
                        })
                        clickable_zones.append({
                            "kind": "rect", "layer": "Premium/Discount", "event_type": None, "type": "bullish",
                            "top": dr["eq"], "bottom": dr["bottom"], "start": dr["start"],
                            "end": rf["confirmed"].index[-1], "df_ref": rf["confirmed"],
                        })
                        price_lines.append({"t0": t0, "t1": future_edge, "price": dr["eq"],
                                             "color": _hex_to_rgba(theme.NEON_AMBER, 1.0),
                                             "title": f"EQ 50% ({layer_tf['Premium/Discount']})", "above": True})
                        # The shaded rectangles above both start at t0 (the
                        # LATER of the two swing points' own confirmations),
                        # so the earlier-forming boundary's own actual swing
                        # candle can sit well to the left of the shading,
                        # with nothing marking exactly where it is. These
                        # trace from each boundary's own swing point through
                        # to the same right edge the shading already uses,
                        # same color convention as the two halves above (top
                        # = magenta, bottom = green) so a line and its own
                        # shaded half read as the same boundary.
                        price_lines.append({"t0": _ny_fake_utc_seconds(dr["top_time"]), "t1": future_edge,
                                             "price": dr["top"], "color": _hex_to_rgba(theme.NEON_MAGENTA, 1.0),
                                             "title": f"Swing high ({layer_tf['Premium/Discount']})", "above": True})
                        price_lines.append({"t0": _ny_fake_utc_seconds(dr["bottom_time"]), "t1": future_edge,
                                             "price": dr["bottom"], "color": _hex_to_rgba(theme.NEON_GREEN, 1.0),
                                             "title": f"Swing low ({layer_tf['Premium/Discount']})", "above": False})
                        legend_items.append((f"Premium/Discount ({layer_tf['Premium/Discount']})",
                                              f"range {dr['bottom']:.5g}–{dr['top']:.5g}", theme.NEON_AMBER))
                        # Explains the otherwise-confusing "price already
                        # wicked past this boundary, why hasn't the range
                        # moved" moment — a wick beyond a level without a
                        # CLOSE beyond it is a liquidity grab/false
                        # breakout, not a genuine break (see this
                        # function's own close-based invalidation), so the
                        # boundary correctly stays put; this just makes
                        # that visible instead of looking unexplained.
                        if dr["top_swept_at"] is not None:
                            markers.append({"time": _ny_fake_utc_seconds(dr["top_swept_at"]["time"]),
                                             "position": "aboveBar", "color": _hex_to_rgba(theme.NEON_AMBER, 1.0),
                                             "shape": "arrowDown", "text": "swept, not broken"})
                        if dr["bottom_swept_at"] is not None:
                            markers.append({"time": _ny_fake_utc_seconds(dr["bottom_swept_at"]["time"]),
                                             "position": "belowBar", "color": _hex_to_rgba(theme.NEON_AMBER, 1.0),
                                             "shape": "arrowUp", "text": "swept, not broken"})
                    else:
                        legend_items.append((f"Premium/Discount ({layer_tf['Premium/Discount']})",
                                              "not enough swing history yet", theme.NEON_AMBER))

            if "Liquidity" in layers:
                rf = tf_frames.get(layer_tf["Liquidity"])
                if rf is not None:
                    # detect_liquidity_levels already returns "the N nearest
                    # still-live levels on each side, closest to price
                    # first" by construction (see its own docstring) — it
                    # just used to ignore max_items_per_layer entirely,
                    # always defaulting to 2/2 regardless of what the
                    # Layers popover's own "Areas per side" field said.
                    # Wiring the same control in here is the fix, not an
                    # extra slice afterward.
                    above, below = detect_liquidity_levels(rf["confirmed"], n_above=max_items_per_layer,
                                                             n_below=max_items_per_layer,
                                                             atr_period=swing_atr_period, atr_mult=swing_atr_mult)
                    # A swept HIGH (BSL) points bearish, a swept LOW (SSL)
                    # points bullish — same direction convention
                    # detect_liquidity_sweeps' own `type` field already
                    # uses (see liquidity_event_win_rate), so BSL reads the
                    # "bearish" side of this lookup and SSL the "bullish".
                    liq_win_rates = _historical_win_rates(ticker, layer_tf["Liquidity"], "Liquidity", data_source,
                                                           exit_types=_win_rate_exit_types,
                                                           news_blackout=_news_blackout)
                    bsl_label = _win_rate_label(liq_win_rates.get("bearish"))
                    ssl_label = _win_rate_label(liq_win_rates.get("bullish"))
                    _rf_close = rf["confirmed"]["Close"] if "Close" in rf["confirmed"] else rf["confirmed"]["close"]
                    above_confluence = point_level_indicator_matches(above, "bearish", _rf_close, _confluence_keys) \
                        if _confluence_keys else {}
                    below_confluence = point_level_indicator_matches(below, "bullish", _rf_close, _confluence_keys) \
                        if _confluence_keys else {}
                    for lvl in above:
                        t1 = _line_stop_time(rf["confirmed"], lvl["time"], lvl["price"])
                        title = f"BSL ({layer_tf['Liquidity']})" + (f" · {bsl_label}" if bsl_label else "")
                        matched_inds = above_confluence.get(lvl["time"], [])
                        if matched_inds:
                            title += " · " + "+".join(INDICATOR_SPECS[k]["label"] for k in matched_inds)
                        bsl_color = theme.CONFLUENCE_GOLD if matched_inds else theme.NEON_MAGENTA
                        price_lines.append({"t0": _ny_fake_utc_seconds(lvl["time"]), "t1": rf["t1_axis"](t1),
                                             "price": lvl["price"], "color": _hex_to_rgba(bsl_color, 1.0),
                                             "title": title, "line_width": 2 if matched_inds else 1, "above": True})
                        liquidity_rows.append({"tf": layer_tf["Liquidity"], "kind": "BSL (buy-side)", "price": lvl["price"], "formed": lvl["time"]})
                        clickable_zones.append({
                            "kind": "level", "layer": "Liquidity", "event_type": None, "type": "bearish",
                            "price": lvl["price"], "start": lvl["time"], "end": t1, "df_ref": rf["confirmed"],
                        })
                    for lvl in below:
                        t1 = _line_stop_time(rf["confirmed"], lvl["time"], lvl["price"])
                        title = f"SSL ({layer_tf['Liquidity']})" + (f" · {ssl_label}" if ssl_label else "")
                        matched_inds = below_confluence.get(lvl["time"], [])
                        if matched_inds:
                            title += " · " + "+".join(INDICATOR_SPECS[k]["label"] for k in matched_inds)
                        ssl_color = theme.CONFLUENCE_GOLD if matched_inds else theme.NEON_AMBER
                        price_lines.append({"t0": _ny_fake_utc_seconds(lvl["time"]), "t1": rf["t1_axis"](t1),
                                             "price": lvl["price"], "color": _hex_to_rgba(ssl_color, 1.0),
                                             "title": title, "line_width": 2 if matched_inds else 1, "above": False})
                        liquidity_rows.append({"tf": layer_tf["Liquidity"], "kind": "SSL (sell-side)", "price": lvl["price"], "formed": lvl["time"]})
                        clickable_zones.append({
                            "kind": "level", "layer": "Liquidity", "event_type": None, "type": "bullish",
                            "price": lvl["price"], "start": lvl["time"], "end": t1, "df_ref": rf["confirmed"],
                        })

                    # rf["confirmed"], not rf["df"] — same caching fix as
                    # Swing Points above, plus this was ALSO the exact
                    # "compared against rdf's own last bar, never matches
                    # r_confirmed.index[-1]" bug the tf_frames construction
                    # comment already describes fixing elsewhere: rf["end"]
                    # (used by rf["t1_axis"] just above, for these same
                    # reaction rectangles' own right edge) could only ever
                    # equal rf["df"]'s last row, which never equals
                    # rf["confirmed"]'s own last row — so a still-active
                    # reaction never actually extended to future_edge, it
                    # silently stopped exactly where price had printed so
                    # far. Missed when that fix was applied to every other
                    # layer; this call site just hadn't been touched yet.
                    reactions = detect_liquidity_reactions(rf["confirmed"], max_candles_after=liquidity_reaction_window,
                                                            atr_period=swing_atr_period, atr_mult=swing_atr_mult)
                    reactions = reactions[-max_items_per_layer:]
                    for r in reactions:
                        color = theme.NEON_MAGENTA if r["type"] == "bearish" else theme.NEON_AMBER
                        rectangles.append({
                            "t0": _ny_fake_utc_seconds(r["start"]), "t1": rf["t1_axis"](r["end"]),
                            "p0": r["bottom"], "p1": r["top"],
                            "fill": _hex_to_rgba(color, 0.22), "border": _hex_to_rgba(color, 1.0),
                        })
                        clickable_zones.append({
                            "kind": "rect", "layer": "Liquidity", "event_type": "liquidity_reaction",
                            "type": r["type"], "top": r["top"], "bottom": r["bottom"],
                            "start": r["start"], "end": r["end"], "df_ref": rf["confirmed"],
                        })

                    legend_items.append((f"Liquidity ({layer_tf['Liquidity']})",
                                          f"{len(above)}BSL {len(below)}SSL · {len(reactions)} reaction(s)", theme.NEON_MAGENTA))

            if "Naked POC" in layers:
                rf = tf_frames.get(layer_tf["Naked POC"])
                # Both this and Poor High/Low below need a real INTRADAY
                # session to bucket — on a 1D+ layer timeframe, each bar
                # already covers a whole day, so "group by NY calendar
                # day" gives exactly one bar per group, which
                # _daily_volume_profiles' own MIN_BARS_PER_DAY guard
                # correctly refuses to treat as a real session profile.
                # Skipped here too (not just left to come back empty)
                # so the legend can say WHY plainly, instead of reading
                # "0 untouched" — confirmed directly as a real point of
                # confusion otherwise: this used to run anyway, silently
                # producing a technically-computed but meaningless result
                # instead of an honest "can't do this here."
                if rf is not None and CANDLE_SECONDS.get(layer_tf["Naked POC"]) is not None:
                    # Nearest to current price first, same "what's actually
                    # in play right now" ordering every other layer's own
                    # max_items_per_layer cap already uses — a naked POC
                    # from months back, far from price, is much less
                    # actionable than one price is sitting right next to.
                    nakeds = sorted(detect_naked_pocs(rf["confirmed"], max_days=session_lookback_days),
                                     key=lambda r: abs(r["price"] - current_price))[:max_items_per_layer]
                    for np_ in nakeds:
                        price_lines.append({
                            "t0": _ny_fake_utc_seconds(np_["time"]), "t1": future_edge, "price": np_["price"],
                            "color": _hex_to_rgba(theme.NEON_GREEN, 0.85),
                            "title": f"Naked POC ({np_['day']})", "dashed": True,
                            "above": np_["price"] >= current_price,
                        })
                        naked_poc_rows.append({"tf": layer_tf["Naked POC"], "day": np_["day"], "price": np_["price"]})
                        clickable_zones.append({
                            "kind": "level", "layer": "Naked POC", "event_type": None, "type": None,
                            "price": np_["price"], "start": np_["time"], "end": rf["confirmed"].index[-1],
                            "df_ref": rf["confirmed"],
                        })
                    legend_items.append((f"Naked POC ({layer_tf['Naked POC']})",
                                          f"{len(nakeds)} untouched", theme.NEON_GREEN))
                    # Same underlying per-session data (see poc_migration's
                    # own docstring) — is value migrating session to
                    # session, or basically flat? Surfaced in the Naked POC
                    # detail expander further down, not its own layer/
                    # checkbox — a derived read on data this layer is
                    # already fetching, not a new thing to opt into.
                    _poc_mig = poc_migration(rf["confirmed"])
                elif rf is not None:
                    legend_items.append((f"Naked POC ({layer_tf['Naked POC']})",
                                          "needs an intraday TF (≤4h)", theme.NEON_GREEN))

            if "Poor High/Low" in layers:
                rf = tf_frames.get(layer_tf["Poor High/Low"])
                if rf is not None and CANDLE_SECONDS.get(layer_tf["Poor High/Low"]) is not None:
                    # No "still live" concept here (see the detector's own
                    # docstring) — most-recent-first, same as Swing Points/
                    # Equal H-L/Structure above, not a nearest-to-price sort.
                    poor_hl = detect_poor_highs_lows(rf["confirmed"], max_days=session_lookback_days)[:max_items_per_layer]
                    for p in poor_hl:
                        markers.append({
                            "time": _ny_fake_utc_seconds(p["time"]),
                            "position": "aboveBar" if p["kind"] == "high" else "belowBar",
                            "color": _hex_to_rgba(theme.NEON_AMBER, 1.0),
                            "shape": "square",
                            "text": f"Poor {p['kind']} ({p['day']})",
                        })
                        clickable_zones.append({
                            "kind": "point", "layer": "Poor High/Low", "event_type": None, "type": None,
                            "price": p["price"], "time": p["time"], "df_ref": rf["confirmed"],
                        })
                    legend_items.append((f"Poor High/Low ({layer_tf['Poor High/Low']})",
                                          f"{len(poor_hl)} flagged", theme.NEON_AMBER))
                elif rf is not None:
                    legend_items.append((f"Poor High/Low ({layer_tf['Poor High/Low']})",
                                          "needs an intraday TF (≤4h)", theme.NEON_AMBER))

            # Changes exactly when a full chart rebuild is warranted (new
            # ticker/timeframe/session/mitigated-visibility/log-scale, or the
            # loaded WINDOW itself shifted — the oldest bar actually being
            # SENT moving means old candles rolled off or backfill depth
            # changed, either of which needs a refit).
            # Deliberately NOT keyed on axis_secs[-1] or len(df) anymore —
            # both changed every time a new candle simply opened, which
            # forced a full setData()+fitContent() (resetting the user's
            # pan/zoom) on a plain, ordinary tick that series.update() was
            # already fully capable of handling on its own (it patches the
            # last bar in place OR appends a new one, per its own docs —
            # see the frontend's isFullReload branch). Confirmed directly:
            # this was firing roughly once a minute on a 1m chart — exactly
            # matching "the screen resets on SOME updates," not most.
            #
            # Reads bars[0], not axis_secs[0]: axis_secs only reflects the
            # main df's own oldest bar, which never changes when a LAYER's
            # own timeframe (not the main chart's) is what widened the
            # backfill anchor bars prepended onto `bars` above — toggling
            # such a layer on/off would then compute the right deeper
            # backfill in Python but never actually trigger isFullReload on
            # the frontend, which would keep patching only the 2-bar
            # incremental tail and never receive the new array at all.
            # Isolated "lock trade" view — direct request: "when I click
            # one, I want to see the exact setup with those exact zones,
            # nothing else." Overrides everything every layer above just
            # built (FVG/OB/Liquidity/Swing/whatever's toggled on) with
            # JUST this one trade's own trigger zone + its specific
            # confluence-factor zones. Markers (Swing Points/Equal H-L
            # dots) aren't part of any trade's own geometry, so those stay
            # cleared too rather than left cluttering an otherwise-isolated
            # view. Entry/SL/TP price_lines (already built above, from the
            # same _scan_pick) are kept, not replaced — extended with the
            # confluence levels alongside them.
            # Rebalance chain view — direct request, sketched by hand:
            # each step gets its own risk/reward box (see
            # _rebalance_chain_overlays' own docstring) stacked forward
            # from future_edge, isolated the same way the lock-trade view
            # is (built from the SAME _trade_only_* snapshot, not
            # whatever every other layer above grew rectangles/
            # price_lines into) — composes with an active lock (both draw
            # at once) rather than one silently overriding the other.
            if _show_chain:
                _base_rects, _base_lines = list(_trade_only_rectangles), list(_trade_only_price_lines)
                if _scan_pick_active:
                    _pick_rects, _pick_conf_lines = _scan_pick_overlays(_scan_pick, future_edge, _ny_fake_utc_seconds)
                    _base_rects += _pick_rects
                    _base_lines += _pick_conf_lines
                # Chained from _active_box_t1 — exactly where the active
                # trade's OWN box (built above) ends — not future_edge,
                # so the whole sequence reads as one continuous path:
                # [active trade box][chain step 1][chain step 2]..., each
                # starting where the last one stopped. Same
                # _trade_box_seconds width as that box, for a consistent
                # rhythm across the whole chain. The chart's own default
                # zoom now reserves room for this (see options.
                # future_margin_frac on the ict_chart call below) instead
                # of the slice width itself trying to solve that.
                _chain_rects, _chain_lines = _rebalance_chain_overlays(
                    _tf_scan_chain, _active_box_t1, _trade_box_seconds)
                rectangles = _base_rects + _chain_rects
                price_lines = _base_lines + _chain_lines
                markers = []
            elif _scan_pick_active:
                _pick_rects, _pick_conf_lines = _scan_pick_overlays(_scan_pick, future_edge, _ny_fake_utc_seconds)
                rectangles = _trade_only_rectangles + _pick_rects
                price_lines = _trade_only_price_lines + _pick_conf_lines
                markers = []

            # Feature C's own confluence factors, drawn ADDITIVELY on top
            # of whatever's already showing (not an isolated replace like
            # the lock-trade view above) — a click is a "tell me more
            # about this," not "hide everything else." Reuses
            # _scan_pick_overlays unchanged by feeding it the same
            # scan_pick shape the lock-trade view itself uses.
            _clicked_result = st.session_state.get("_clicked_zone_result")
            if _clicked_result and _clicked_result.get("status") == "hit":
                _click_rects, _click_lines = _scan_pick_overlays(
                    _clicked_result["scan_pick_shape"], future_edge, _ny_fake_utc_seconds)
                rectangles = rectangles + _click_rects
                price_lines = price_lines + _click_lines

            # Clean-pattern streak study view — same additive convention
            # as the clicked-zone confluence factors just above: every
            # zone in the currently-selected streak drawn as its own
            # cyan-outlined box via _scan_pick_overlays, composed on top
            # of whatever else is showing rather than replacing it.
            _selected_streak = st.session_state.get("_selected_streak")
            if _selected_streak:
                _streak_shape = {
                    "direction": _selected_streak["bullish"] >= _selected_streak["bearish"] and "bullish" or "bearish",
                    "entry": None, "label": None, "source_zone": None,
                    "confluence_entry_details": [
                        {"label": f"{z['layer']} ({z['type']})",
                         "zones": [{"kind": "rect", "top": z["top"], "bottom": z["bottom"],
                                    "start": z["start"], "end": z["end"]}]}
                        for z in _selected_streak["zones"]
                    ],
                    "confluence_context_details": [],
                }
                _streak_rects, _streak_lines = _scan_pick_overlays(_streak_shape, future_edge, _ny_fake_utc_seconds)
                rectangles = rectangles + _streak_rects
                price_lines = price_lines + _streak_lines

            fingerprint = (f"{ticker}|{tf_label}|{'+'.join(sorted(selected_kill_zones))}|{show_mitigated}|"
                            f"{bars[0]['time'] if bars else 0}")

            # Every fragment tick used to re-serialize and re-send the ENTIRE
            # bars array — for a 7-day 1m chart, ~2700 rows over the wire
            # every 10s even though a steady-state tick only ever actually
            # changes the last one or two candles. Full array only goes out
            # on a genuine full reload (new ticker/timeframe/etc, matching
            # the frontend's own isFullReload split); the frontend merges a
            # smaller tail into its existing bar cache instead of replacing
            # it (see barsByTime in handleRender).
            _is_full_reload = st.session_state.get("_main_last_sent_fp") != fingerprint
            st.session_state["_main_last_sent_fp"] = fingerprint
            bars_payload = bars if _is_full_reload else bars[-2:]

            # Indicators got NONE of the incremental treatment bars/ghost
            # candles already have — _chart_indicators holds the WHOLE
            # loaded window's worth of MA/RSI/MACD/BB/ATR points, rebuilt
            # fresh from the current df every render, and it was being
            # sent (and the frontend was doing a full pane teardown +
            # rebuild for) IN FULL on every single tick, not just full
            # reloads — confirmed directly: on a live 1m chart with just
            # MA showing, this measured over 1MB of JSON, repeated on
            # close to every auto-refresh tick, dwarfing the bars payload
            # itself (which was already fixed to 2 rows). The reason a
            # plain content-hash skip (what ghost_candles/markers already
            # use) doesn't work here: unlike those, indicator SERIES
            # genuinely change every tick a new bar forms, so the hash
            # never matched and a full rebuild fired almost every time
            # anyway. Same fix as bars: full resend only when the
            # indicator SETTINGS themselves changed (or a genuine full
            # reload happened for any other reason) — otherwise send just
            # the last few points and let the frontend .update() them onto
            # its already-existing series instead of tearing everything
            # down.
            _ind_settings_fp = (
                show_indicators, ma_fast_period, ma_slow_period, indicator_tf,
                show_rsi, rsi_period, show_macd, macd_fast, macd_slow, macd_signal,
                show_bb, bb_period, bb_std, show_atr, atr_period_ind,
            )
            _indicators_full_reload = _is_full_reload or (
                st.session_state.get("_main_last_indicator_fp") != _ind_settings_fp)
            st.session_state["_main_last_indicator_fp"] = _ind_settings_fp
            if _indicators_full_reload:
                _chart_indicators["_full_reload"] = True
            else:
                _INDICATOR_TAIL = 3
                _trimmed_indicators = {}
                for _ik, _iv in _chart_indicators.items():
                    if isinstance(_iv, list):
                        _trimmed_indicators[_ik] = _iv[-_INDICATOR_TAIL:]
                    elif isinstance(_iv, dict):
                        _trimmed_indicators[_ik] = {
                            _ik2: (_iv2[-_INDICATOR_TAIL:] if isinstance(_iv2, list) else _iv2)
                            for _ik2, _iv2 in _iv.items()
                        }
                    else:
                        _trimmed_indicators[_ik] = _iv  # scalars (periods) — cheap, always included
                _trimmed_indicators["_full_reload"] = False
                _chart_indicators = _trimmed_indicators

            _main_overlays = {
                "rectangles": rectangles, "price_lines": price_lines,
                # lightweight-charts' series-markers plugin requires
                # markers pre-sorted ascending by time — Swing
                # Points appends bullish/bearish in two separate
                # loops, so the raw list isn't globally sorted.
                # Confirmed directly: passing it unsorted is why
                # bearish (arrowDown) markers were disappearing at
                # some zoom levels.
                "markers": sorted(markers, key=lambda m: m["time"]),
            }
            # ghost_candles/volume_profile only ever change on a genuine
            # full reload (ticker/timeframe/overlay-TF/backfill depth) —
            # never on a plain incremental tick — but were being rebuilt
            # and sent in full every single render regardless (measured at
            # ~250KB on a real 1m chart). The frontend's own
            # _overlayArrayChanged already skips the expensive
            # detach/reattach when the content matches what's already
            # showing, but it still has to receive and JSON-stringify the
            # whole array to find that out. Omitting the keys entirely on
            # an incremental tick (see applyOverlays' own matching change)
            # skips that comparison altogether instead of just skipping
            # the redraw — leaving whatever's already attached untouched,
            # which is exactly correct since this data hasn't changed.
            if _is_full_reload:
                _main_overlays["ghost_candles"] = ghost_candles
                _main_overlays["volume_profile"] = volume_profile_buckets
            clicked = ict_chart(
                bars_payload, fingerprint,
                overlays=_main_overlays,
                options={"log_scale": show_log_scale, "volume": show_volume,
                          "selected": st.session_state.get("selected_chart") == "main",
                          # A touch narrower than the mini panels' candles —
                          # lightweight-charts derives candle width straight
                          # from barSpacing with no separate thickness knob,
                          # see candleWidthFactor in the frontend.
                          "candle_width_factor": 0.85,
                          # Direct request: "default the chart to about 60%
                          # of the width to the right to make room for the
                          # trades" — the active-trade box and rebalance-
                          # chain boxes (see _trade_box_overlays) both start
                          # at the present and extend rightward; leaving
                          # this much of the fitted view empty on the right
                          # is what actually makes room for them without
                          # the user having to pan manually every time.
                          "future_margin_frac": 0.6},
                indicators=_chart_indicators,
                # symbol/interval are separate from ticker/source above (which
                # feed the on-chart readout text) — this pair is purely for
                # the frontend's own live-tick WebSocket subscription (see
                # connectLiveFeed in the frontend), which needs the exact TF
                # label regardless of what the readout happens to display.
                ohlc={"ticker": ticker, "source": served_by, "forming_bar_time": forming_bar_axis_sec,
                      "bar_seconds": bar_step if forming_bar_axis_sec is not None else None,
                      "symbol": ticker, "interval": tf_label,
                      # Read by the frontend's own resync check (see its
                      # comment on pythonSaysFullReload) — the ONLY reliable
                      # way it can tell "genuinely fresh data" apart from
                      # "fresh frontend, stale data," since both look
                      # identical from a reset frontend's own local state.
                      "is_full_reload": _is_full_reload},
                height=1000,
                display_tz=theme.get_display_tz(),
            )
            if clicked == _CHART_NEEDS_FULL_RELOAD:
                # This frontend instance's own cache was empty when it got
                # only the incremental tail above (bars_payload = bars[-2:])
                # — Python's own "already sent this fingerprint" bookkeeping
                # was stale relative to what's actually in the browser (see
                # _CHART_NEEDS_FULL_RELOAD's own comment). Clearing it here
                # makes THIS SAME fingerprint compute _is_full_reload=True on
                # the very next run, which is all the fix needs — nothing
                # about the fingerprint itself was wrong. A fragment-scoped
                # rerun (not _select_chart's full-app one) is enough: nothing
                # outside this fragment reads _main_last_sent_fp.
                #
                # Bounded per fingerprint: a genuine one-off desync (the
                # case above) only ever needs ONE retry to resolve.
                # Confirmed directly as a real, live bug otherwise: this
                # fragment's own run_every tick racing a user-triggered
                # rerun (e.g. clicking a scan result right as the fragment
                # was mid-tick) can make the frontend report the same
                # desync repeatedly, forever — st.rerun(scope="fragment")
                # firing in a tight loop, re-sending the FULL ~2,700-row
                # bars array every pass, which is exactly the multi-
                # second "chart loads extremely slow" stall reported.
                # Capping retries turns a recurring desync into "one
                # stale-looking frame, then give up and move on" instead
                # of an unbounded loop; a fresh fingerprint (a real
                # ticker/timeframe change) always gets its own full budget.
                _resync_state = st.session_state.get("_main_resync_tries")
                _resync_tries = _resync_state[1] + 1 if _resync_state and _resync_state[0] == fingerprint else 1
                st.session_state["_main_resync_tries"] = (fingerprint, _resync_tries)
                if _resync_tries <= 2:
                    st.session_state.pop("_main_last_sent_fp", None)
                    # Belt-and-suspenders alongside the OR in
                    # _indicators_full_reload's own computation (which
                    # already forces a full indicator resend whenever
                    # _is_full_reload does) — clearing this too means the
                    # corrective render can't possibly still think
                    # indicator SETTINGS are what's unchanged and skip a
                    # full resend on some future edge case this fingerprint
                    # alone doesn't cover.
                    st.session_state.pop("_main_last_indicator_fp", None)
                    st.rerun(scope="fragment")
            _select_chart("main", clicked)

            # Feature C — "click an area, scan for known patterns/levels
            # that are meaningful." Only a genuinely NEW click (a fresh
            # clickId) gets processed — repeat reruns that carry the same
            # already-handled click (e.g. an unrelated widget change
            # elsewhere on the page) shouldn't re-run the hit-test/
            # confluence-score work for nothing.
            if isinstance(clicked, dict) and clicked.get("clickId") is not None and \
                    st.session_state.get("_last_click_zone_test") != clicked.get("clickId"):
                st.session_state["_last_click_zone_test"] = clicked.get("clickId")
                _hit = _hit_test_zone(clicked, clickable_zones)
                if _hit is None:
                    st.session_state["_clicked_zone_result"] = {"status": "miss"}
                elif _hit.get("type") is None:
                    # Naked POC / Swing Points / Poor High/Low — no
                    # inherent direction, so no confluence/validation
                    # readout to fake one for; just the plain facts.
                    st.session_state["_clicked_zone_result"] = {
                        "status": "no_direction", "layer": _hit["layer"],
                        "price": _hit.get("price"),
                        "start": _hit.get("start") or _hit.get("time"),
                    }
                else:
                    _direction = _hit["type"]
                    _hit_price = (_hit["top"] + _hit["bottom"]) / 2 if _hit["kind"] == "rect" else _hit["price"]
                    _score, _factor_details = _confluence_score(_direction, _hit_price, _hit["df_ref"])
                    _event_type = _hit.get("event_type")
                    if _event_type and _event_type in EVENT_TYPE_TO_HYPOTHESIS:
                        _val_label, _val_tone = validation_badge(_event_type, ticker)
                    else:
                        _val_label, _val_tone = "no validation model defined for this layer yet", "info"
                    st.session_state["_clicked_zone_result"] = {
                        "status": "hit", "layer": _hit["layer"], "direction": _direction,
                        "price": _hit_price, "score": _score, "factor_details": _factor_details,
                        "validation_label": _val_label, "validation_tone": _val_tone,
                        "scan_pick_shape": {
                            "direction": _direction, "entry": _hit_price,
                            "label": f"Clicked {_hit['layer']}",
                            "source_zone": ({"start": _hit["start"], "end": _hit["end"],
                                              "top": _hit["top"], "bottom": _hit["bottom"]}
                                             if _hit["kind"] == "rect" else None),
                            "confluence_entry_details": _factor_details,
                            "confluence_context_details": [],
                        },
                    }

            # Direct request: "where is price more likely to head next" —
            # every currently-open level (FVG/IFVG/Order Block/Breaker
            # Block/resting liquidity) ranked by an actual historical
            # touch-PROBABILITY, not raw proximity. See recommender.py's
            # own module comment on rank_levels_by_visit_probability for
            # the full reasoning (why this is a different question from
            # any win-rate already on this page, and how the probability
            # itself is computed). Computed early since the tab list below
            # needs to know whether it has anything to show.
            _level_current_price = float(df[c_col].iloc[-1])
            _ranked_levels = rank_levels_by_visit_probability(df, _level_current_price, top_n=10)

            # Independent of naked_poc_rows on purpose — whether value is
            # migrating is a fact about the last few SESSIONS as a whole,
            # not about which of their POCs still happen to be untouched
            # right now (those are two different questions; a session's
            # POC can get touched a day later and drop off the list below
            # while the underlying migration trend it was part of is still
            # real). Shown plainly, above the fold, not tucked inside the
            # tabbed drawer below — this reads as a standalone "is a trend
            # actually building" verdict, not a footnote to the level list.
            if _poc_mig is not None:
                _mig_first, _mig_last = _poc_mig["days"][0], _poc_mig["days"][-1]
                if _poc_mig["direction"] == "flat":
                    st.caption(
                        f"POC migration ({_mig_first['day']} → {_mig_last['day']}, {len(_poc_mig['days'])} "
                        f"sessions): {_mig_first['price']:.5g} → {_mig_last['price']:.5g} "
                        f"({_poc_mig['pct_change']:+.1f}%) — flat/range-bound, no strong directional "
                        f"conviction session to session."
                    )
                else:
                    _mig_dir = "UP" if _poc_mig["direction"] == "up" else "DOWN"
                    st.caption(
                        f"POC migration ({_mig_first['day']} → {_mig_last['day']}, {len(_poc_mig['days'])} "
                        f"sessions): {_mig_first['price']:.5g} → {_mig_last['price']:.5g} "
                        f"({_poc_mig['pct_change']:+.1f}%) — value has been drifting **{_mig_dir}**, "
                        f"consistent with a developing trend rather than balance."
                    )

            # Every detector "detail" panel used to be its own always-visible
            # collapsed expander — up to 6 stacked header bars taking up
            # screen space before opening a single one. They now share ONE
            # expander with a tab per section inside it, so the collapsed
            # cost is exactly one header row no matter how many detectors
            # are active — a soft drawer instead of a wall of bars.
            _detail_tabs = []
            if legend_items:
                _detail_tabs.append(("Legend", "legend"))
            if fvg_ob_rows:
                _detail_tabs.append((f"🔍 FVG / OB ({len(fvg_ob_rows)})", "fvg_ob"))
            if liquidity_rows:
                _detail_tabs.append((f"🔍 Liquidity ({len(liquidity_rows)})", "liquidity"))
            if _ranked_levels:
                _detail_tabs.append((f"🎯 Where's price headed ({len(_ranked_levels)})", "levels"))
            if naked_poc_rows:
                _detail_tabs.append((f"🔍 Naked POC ({len(naked_poc_rows)})", "naked_poc"))
            if _vp is not None:
                _detail_tabs.append((f"🔍 Volume Profile — {_vp['shape_label']}", "volume_profile"))
            _detail_tabs.append(("🧪 Experiments", "experiments"))
            # Unconditionally appended, like Experiments above — this
            # needs to be discoverable BEFORE a first click, not appear
            # out of nowhere only after one happens.
            _detail_tabs.append(("🎯 Clicked zone", "clicked_zone"))
            # Also unconditionally appended — the streak SEARCH itself is
            # cheap (reuses the already-cached detect_fvgs/detect_order_
            # blocks this chart's own FVG/OB layers already call), so
            # there's no reason to gate the tab's existence on a layer
            # being toggled on first.
            _detail_tabs.append(("🧹 Clean patterns", "clean_patterns"))

            def _render_detail_body(_key):
                if _key == "legend":
                    fvg_legend(legend_items)
                    tiny("🟢/🔵 bullish · 🔴/🟠 bearish · scroll to zoom, drag to pan, "
                         "each layer detects on its own chosen timeframe (☰ to change) and shows only its most "
                         "recent items · Premium/Discount boxes the current swing high/low dealing range and "
                         "stays put until price closes through one side · open FVG/OB zones are labeled with "
                         "this ticker's own historical win rate for that direction (hover the detail table "
                         "below for what that number does and doesn't mean)")

                elif _key == "fvg_ob":
                    # The legend above is a tally; this is the actual
                    # data behind it — every FVG/Order Block zone
                    # currently drawn on the chart, not a
                    # re-detection, the exact same fvg_ob_rows list
                    # the rectangles were built from a few hundred
                    # lines up.
                    #
                    # Sorted open/unmitigated first, then by distance
                    # from the current close — a zone price has
                    # already fully traded through isn't "tradeable
                    # in the future" the way a still-live zone
                    # sitting just above/below current price is, and
                    # the ones dozens of percent away are academic
                    # compared to the ones price could reach this
                    # session. This is a relevance ranking for YOUR
                    # attention, not a claim any specific zone will
                    # hold.
                    #
                    # hist_win_rate closes part of that gap — a real
                    # number computed off this ticker's own history
                    # (_historical_win_rates above), not a guess —
                    # but it's still a plain raw-return win rate, NOT
                    # the permutation-test-plus-multiple-testing-
                    # correction rigor Research Lab and Edge Lab
                    # apply before calling anything an actual edge.
                    # Treat it as "how has this pattern's direction
                    # tended to resolve historically," not
                    # statistical proof.
                    current_price = float(df[c_col].iloc[-1])
                    _decimals = _price_decimals(current_price)
                    detail_df = pd.DataFrame(fvg_ob_rows)
                    detail_df["is_open"] = detail_df["status"].isin(["open", "unmitigated"])
                    detail_df["dist_pct"] = (((detail_df["top"] + detail_df["bottom"]) / 2 - current_price)
                                              / current_price * 100)
                    detail_df = detail_df.sort_values(["is_open", "dist_pct"], key=lambda s: s if s.name == "is_open" else s.abs(),
                                                        ascending=[False, True]).drop(columns="is_open").reset_index(drop=True)
                    st.dataframe(detail_df, hide_index=True, width="stretch", column_config={
                        "dist_pct": st.column_config.NumberColumn(
                            "dist_pct", format="%+.2f%%",
                            help=f"Distance from the current price ({current_price:.5g}) to this zone's midpoint — "
                                 "positive means the zone sits above price, negative means below. Sorted closest "
                                 "first within each status, since that's what price could realistically reach next."),
                        "layer": st.column_config.Column(help="FVG (3-candle imbalance) or Order Block (last "
                                                               "opposing candle before the displacement move)."),
                        "tf": st.column_config.Column(help="Reference timeframe this zone was detected on "
                                                            "(4h or 15m) — independent of the chart's own zoom."),
                        "type": st.column_config.Column(help="bullish = support read, drawn cyan/green. "
                                                               "bearish = resistance read, drawn magenta/amber."),
                        "top": st.column_config.NumberColumn(
                            format=f"%.{_decimals}f",
                            help="Zone's upper price boundary, as currently "
                                 "drawn — shrinks over time as price eats into "
                                 "the zone (consequent encroachment)."),
                        "bottom": st.column_config.NumberColumn(
                            format=f"%.{_decimals}f",
                            help="Zone's lower price boundary, same "
                                 "shrinking-over-time caveat as top."),
                        "start": st.column_config.Column(help="Bar where this zone formed."),
                        "end": st.column_config.Column(help="Bar where the zone fully filled/mitigated — or the "
                                                              "most recent bar, if it's still open."),
                        "status": st.column_config.Column(help="FVG: filled = price has fully traded back through "
                                                                 "it. Order Block: mitigated = same idea, price "
                                                                 "fully reclaimed it. open/unmitigated = still live."),
                        "first_touch": st.column_config.Column(help="First bar price wicked back into this zone at "
                                                                      "all — earlier than 'end', which only marks "
                                                                      "full mitigation. Empty if never touched yet."),
                        "hist_win_rate": st.column_config.Column(
                            "hist_win_rate", help="Share of this ticker's own past FVG/OB retracements in this same "
                                                   "direction that closed in the expected direction (raw return, no "
                                                   "cost adjustment) — computed "
                                                   "on the longest history this data source will serve, refreshed "
                                                   "hourly. A quick reference stat, not a validated edge: no "
                                                   "permutation test, no multiple-testing correction, no holdout "
                                                   "split. 'low data' means under 20 qualifying historical events "
                                                   "(shown anyway, just not trustworthy as a percentage); "
                                                   "'insufficient data' means zero. Treat it as a rough read, not "
                                                   "proof — no independent statistical validation is run on it."),
                    })

                elif _key == "liquidity":
                    # BSL/SSL only ever rendered as unlabeled dashed
                    # lines on the chart, with a hover title as the
                    # only way to see which is which — no exact
                    # price, no distance, no way to tell which ones
                    # are closest to actually mattering right now
                    # without reading raw chart data. Same fix as
                    # the FVG/OB table above, same "sort by what's
                    # actually relevant" logic.
                    current_price = float(df[c_col].iloc[-1])
                    _decimals = _price_decimals(current_price)
                    # A BSL level can legitimately show a NEGATIVE
                    # distance (sitting below the live price) —
                    # detection only ever checks "unswept as of the
                    # last CLOSED bar," but this table's distance is
                    # measured against the live, currently-forming
                    # candle. Confirmed directly: not a bug, just
                    # means price has already wicked through that
                    # level intrabar, on a candle that hasn't closed
                    # yet — once it does, that level will very
                    # likely flip to swept on the next rerun.
                    tiny("A BSL below price (or SSL above it) means the live candle has already wicked through "
                         "that level — detection only confirms 'swept' once the bar actually closes.")
                    liq_df = pd.DataFrame(liquidity_rows)
                    liq_df["dist_pct"] = (liq_df["price"] - current_price) / current_price * 100
                    liq_df = liq_df.sort_values("dist_pct", key=lambda s: s.abs()).reset_index(drop=True)
                    st.dataframe(liq_df, hide_index=True, width="stretch", column_config={
                        "tf": st.column_config.Column(help="Reference timeframe this level was detected on."),
                        "kind": st.column_config.Column(
                            help="BSL (buy-side liquidity) rests above a swing high — stops from shorts, a magnet "
                                 "for a bullish run. SSL (sell-side) rests below a swing low — stops from longs, "
                                 "a magnet for a bearish run. Both shown here are still LIVE — detect_liquidity_levels "
                                 "only ever returns levels no later candle has swept yet."),
                        "price": st.column_config.NumberColumn(
                            format=f"%.{_decimals}f", help="Exact price of the resting level."),
                        "formed": st.column_config.Column(help="The swing bar that created this level."),
                        "dist_pct": st.column_config.NumberColumn(
                            "dist_pct", format="%+.2f%%",
                            help=f"Distance from the current price ({current_price:.5g}). Sorted closest first — "
                                 "the nearest levels are what price could realistically reach next."),
                    })

                elif _key == "levels":
                    _decimals = _price_decimals(_level_current_price)
                    tiny("Each level's own historical touch rate, conditioned on how far it sat from price "
                         "when it formed — NOT a prediction, a plain fact about this ticker's own past: 'of "
                         "every level like this one, formed this far from price, what fraction eventually got "
                         "touched.' A level with too few historical examples to trust still shows, ranked "
                         "below every well-supported one, labeled 'not enough history' instead of a number.")
                    _lvl_rows = []
                    for r in _ranked_levels:
                        _lvl_rows.append({
                            "type": r["type"], "price": r["price"], "dist_pct": r["distance_pct"],
                            "touch_probability": f"{r['probability']*100:.0f}% (n={r['n']})" if r["sufficient"]
                                                  else (f"low data ({r['n']})" if r["n"] else "not enough history"),
                        })
                    st.dataframe(pd.DataFrame(_lvl_rows), hide_index=True, width="stretch", column_config={
                        "type": st.column_config.Column(help="FVG/IFVG/Order Block/Breaker Block are ranges; "
                                                              "Liquidity is a single resting swing high/low."),
                        "price": st.column_config.NumberColumn(
                            format=f"%.{_decimals}f",
                            help="This level's own price (range midpoint for "
                                 "FVG/IFVG/Order Block/Breaker Block)."),
                        "dist_pct": st.column_config.NumberColumn(
                            "dist_pct", format="%+.2f%%",
                            help=f"Distance from the current price ({_level_current_price:.5g})."),
                        "touch_probability": st.column_config.Column(
                            help="Historical fraction of this level's own type, formed this far from price at "
                                 "the time, that ever got touched at all — see the caption above."),
                    })

                elif _key == "naked_poc":
                    # Same "exact price/distance, not just a hover
                    # title" fix as Liquidity above, for the same
                    # reason.
                    current_price = float(df[c_col].iloc[-1])
                    _decimals = _price_decimals(current_price)
                    tiny("A daily session's own busiest traded price, still untouched by anything since that "
                         "session closed — these act as magnets. Drops off this list (and the chart) the instant "
                         "a later candle finally trades through it.")
                    np_df = pd.DataFrame(naked_poc_rows)
                    np_df["dist_pct"] = (np_df["price"] - current_price) / current_price * 100
                    np_df = np_df.sort_values("dist_pct", key=lambda s: s.abs()).reset_index(drop=True)
                    st.dataframe(np_df, hide_index=True, width="stretch", column_config={
                        "tf": st.column_config.Column(help="Reference timeframe the daily sessions were built from."),
                        "day": st.column_config.Column(help="The NY calendar day this POC formed on."),
                        "price": st.column_config.NumberColumn(
                            format=f"%.{_decimals}f", help="Exact price of the untouched POC."),
                        "dist_pct": st.column_config.NumberColumn(
                            "dist_pct", format="%+.2f%%",
                            help=f"Distance from the current price ({current_price:.5g}). Sorted closest first — "
                                 "the nearest ones are what price could realistically reach next."),
                    })

                elif _key == "volume_profile":
                    # Shape/HVN/LVN are already ON the chart
                    # (legend text, bucket coloring) — this tab is
                    # for the exact price levels those bucket
                    # colors don't otherwise show a number for,
                    # same "make the number reachable" reasoning
                    # as every table above.
                    _decimals = _price_decimals(_vp["poc_price"])
                    tiny(_vp["shape_description"])
                    st.caption(f"POC {_vp['poc_price']:.5g} · VAH {_vp['value_area_high']:.5g} · "
                               f"VAL {_vp['value_area_low']:.5g}")
                    _vp_nodes = (
                        [{"kind": "HVN", "price": (_vp["buckets"][i]["price_low"] + _vp["buckets"][i]["price_high"]) / 2,
                          "volume_frac": _vp["buckets"][i]["volume_frac"]} for i in _vp["hvn_indices"]]
                        + [{"kind": "LVN", "price": (_vp["buckets"][i]["price_low"] + _vp["buckets"][i]["price_high"]) / 2,
                            "volume_frac": _vp["buckets"][i]["volume_frac"]} for i in _vp["lvn_indices"]]
                    )
                    if _vp_nodes:
                        st.dataframe(pd.DataFrame(_vp_nodes), hide_index=True, width="stretch", column_config={
                            "kind": st.column_config.Column(
                                help="HVN = High Volume Node, a real second cluster besides POC — price likely "
                                     "stalls/consolidates here if revisited. LVN = Low Volume Node, a genuinely "
                                     "thin spot — price likely moves FAST through here if revisited, little "
                                     "resting interest to absorb it."),
                            "price": st.column_config.NumberColumn(
                                format=f"%.{_decimals}f", help="Midpoint price of this bucket."),
                            "volume_frac": st.column_config.ProgressColumn(
                                "volume", format="percent", min_value=0, max_value=1,
                                help="This bucket's own volume relative to the busiest bucket (POC)."),
                        })
                    else:
                        st.caption("No distinct HVN/LVN cleared the prominence filter on this profile.")

                elif _key == "experiments":
                    # The actual thesis, stated directly rather than
                    # inferred: trade sharp, fast reactions off known
                    # zones, but only when there's enough volatility to
                    # deliver the move — a consolidating range gives a
                    # reaction nothing to travel INTO. Settings below are
                    # tuned live against this chart's own currently-
                    # loaded history (switch timeframe/period for more);
                    # "Run deep backtest" is the one number here that's
                    # actually been permutation-tested — everything above
                    # it is fast feedback for tuning, not proof.
                    tiny("Trade sharp, fast reactions off known zones — but only when there's enough "
                         "volatility to actually deliver the move. Settings below run live against "
                         "this chart's own currently-loaded history. 'Run deep backtest' is the real "
                         "test (permutation test + holdout split); the numbers above it are just fast "
                         "feedback for tuning, not proof of anything on their own.")
                    _exp_c1, _exp_c2 = st.columns(2)
                    with _exp_c1:
                        _exp_detector = st.selectbox("Detector", list(experiments.DETECTOR_FNS.keys()),
                                                      key=f"exp_detector_{ticker}")
                        _exp_min_vol = st.slider(
                            "Min volatility percentile", 0.0, 1.0, 0.5, step=0.05,
                            key=f"exp_minvol_{ticker}",
                            help="0 = any regime, including consolidation. Higher means this only "
                                 "fires when the instrument is more volatile than its own recent norm "
                                 "right now — the whole point being to skip quiet, range-bound stretches.")
                        _exp_forward = st.number_input(
                            "Holding period (bars)", 1, 50, 5, key=f"exp_fwd_{ticker}",
                            help="How many bars forward the outcome is measured over — kept short on "
                                 "purpose ('in and out fast', not a multi-hour hold).")
                    with _exp_c2:
                        _exp_window = st.number_input(
                            "Reaction window (bars)", 1, 10, 2, key=f"exp_window_{ticker}",
                            help="How many bars after first touching the zone the reaction has to "
                                 "happen within to count as fast.")
                        _exp_mult = st.number_input(
                            "Reaction strength (× ATR)", 0.1, 5.0, 1.0, step=0.1, key=f"exp_mult_{ticker}",
                            help="How far price has to move away from the zone, in multiples of this "
                                 "instrument's own ATR, to count as a real reaction rather than a slow "
                                 "grind through it.")

                    _exp_events = experiments.build_experiment_events(
                        df, _exp_detector, _exp_min_vol, _exp_window, _exp_mult, _exp_forward)
                    _exp_stats = experiments.preview_stats(_exp_events)

                    if _exp_stats["n"] == 0:
                        st.caption("No qualifying setups with these settings on this chart's own loaded "
                                   "history — try loosening the volatility or reaction requirements, or "
                                   "switch to a longer timeframe/period for more history to search.")
                    else:
                        _exp_m1, _exp_m2, _exp_m3 = st.columns(3)
                        _exp_m1.metric("Qualifying setups", _exp_stats["n"])
                        _exp_m2.metric("Win rate (quick check)", f"{_exp_stats['win_rate']:.0%}")
                        _exp_m3.metric("Mean return (quick check)", f"{_exp_stats['mean_return']:+.3%}")

                        _exp_price_by_time = dict(zip(df.index, df[c_col].to_numpy()))
                        _exp_points = _exp_events.copy()
                        _exp_points["price"] = _exp_points["entry_time"].map(_exp_price_by_time)
                        _exp_chart_df = pd.DataFrame({"time": df.index, "close": df[c_col].to_numpy()})
                        _exp_base = alt.Chart(_exp_chart_df).mark_line(
                            color=theme.NEON_CYAN, opacity=0.6
                        ).encode(x="time:T", y=alt.Y("close:Q", scale=alt.Scale(zero=False), title=None))
                        _exp_marks = alt.Chart(_exp_points).mark_circle(size=90).encode(
                            x="entry_time:T", y="price:Q",
                            color=alt.Color("direction:N", scale=alt.Scale(
                                domain=["bullish", "bearish"],
                                range=[theme.NEON_GREEN, theme.NEON_MAGENTA]), legend=None),
                            tooltip=["entry_time:T", "direction:N", "raw_return:Q"])
                        st.altair_chart((_exp_base + _exp_marks).properties(height=280), width="stretch")

                    if st.button("🔬 Run deep backtest", key=f"exp_run_{ticker}"):
                        _exp_settings = {"detector": _exp_detector, "min_volatility_pctile": _exp_min_vol,
                                          "reaction_window": _exp_window, "reaction_mult": _exp_mult,
                                          "forward_bars": _exp_forward}
                        with st.spinner("Running permutation test..."):
                            _exp_trial = experiments.run_deep_backtest(ticker, tf_label, _exp_events, _exp_settings)
                        if _exp_trial["verdict"] == "INSUFFICIENT_DATA":
                            st.warning(f"Only {_exp_trial['n_events']} qualifying setups — need at least "
                                       "30 to run a real test. Loosen the settings or load more history.")
                        else:
                            _exp_same = _exp_trial["same_sign"]
                            _exp_passed = _exp_trial["holdout_verdict"] == "PASSED"
                            if _exp_passed and _exp_same:
                                st.success(
                                    f"Train p={_exp_trial['p_value_train']:.4f}, mean="
                                    f"{_exp_trial['mean_return_train']:+.3%} · Holdout mean="
                                    f"{_exp_trial['mean_return_holdout']:+.3%} — passed, same direction "
                                    "on both splits. Still just one trial; see the log below for the "
                                    "corrected view across everything ever tried.")
                            else:
                                st.warning(
                                    f"Train p={_exp_trial['p_value_train']:.4f}, mean="
                                    f"{_exp_trial['mean_return_train']:+.3%} · Holdout mean="
                                    f"{_exp_trial.get('mean_return_holdout', float('nan')):+.3%} "
                                    f"({_exp_trial['holdout_verdict']}"
                                    f"{', opposite sign from train' if not _exp_same else ''}) — "
                                    "didn't hold up out of sample.")

                    _exp_hist = experiments.load_experiment_trials()
                    if not _exp_hist.empty:
                        with st.expander(f"Every experiment ever run ({len(_exp_hist)}) — BH-corrected"):
                            _exp_scored = _exp_hist[_exp_hist["verdict"] == "SCORED"]
                            if _exp_scored.empty:
                                st.caption("Nothing scored yet — every run so far had too little data.")
                            else:
                                # Capped, not the full ~20k+ scored rows every
                                # time — this expander's own body still runs
                                # on every rerun regardless of whether it's
                                # open (Streamlit executes every tab/expander's
                                # content, it only hides the DOM for inactive
                                # ones), so an uncapped st.dataframe here was
                                # real, continuous serialization cost paid on
                                # every tick whether anyone was looking or
                                # not. Any survivor is kept regardless of
                                # recency (there's never more than a handful,
                                # and that's the one row worth never missing);
                                # otherwise just the most recent _EXP_TABLE_CAP.
                                _EXP_TABLE_CAP = 300
                                _exp_recent = _exp_scored.head(_EXP_TABLE_CAP)
                                _exp_old_survivors = _exp_scored[_exp_scored["survived"].fillna(False)
                                                                   & ~_exp_scored.index.isin(_exp_recent.index)]
                                _exp_display = (pd.concat([_exp_old_survivors, _exp_recent])
                                                 if not _exp_old_survivors.empty else _exp_recent)
                                _exp_cap_note = f"Showing {len(_exp_display)} of {len(_exp_scored)} scored trials"
                                if not _exp_old_survivors.empty:
                                    _exp_cap_note += f" — most recent {_EXP_TABLE_CAP} plus {len(_exp_old_survivors)} older survivor(s)"
                                else:
                                    _exp_cap_note += f" — most recent {_EXP_TABLE_CAP}"
                                st.caption(_exp_cap_note + ".")
                                st.dataframe(
                                    _exp_display[["logged_at", "ticker", "label", "n_events",
                                                  "mean_return_train", "p_value_train", "q_value_train",
                                                  "holdout_verdict", "survived"]],
                                    hide_index=True, width="stretch")

                elif _key == "clicked_zone":
                    _cz = st.session_state.get("_clicked_zone_result")
                    if not _cz:
                        st.caption("Click a drawn zone on the chart to see what's meaningful about it.")
                    elif _cz["status"] == "miss":
                        st.caption("That click didn't land on a known zone — try clicking closer to a "
                                   "drawn box or line.")
                    elif _cz["status"] == "no_direction":
                        st.markdown(f"**{_cz['layer']}** — price {_cz['price']:.5g}, formed {_cz['start']}")
                        st.caption("This layer has no inherent bullish/bearish bias, so there's no "
                                   "direction to score confluence or validation against.")
                    else:
                        _tone_fn = {"bullish": st.success, "warn": st.warning, "info": st.info}[_cz["validation_tone"]]
                        st.markdown(f"**{_cz['layer']}** — {_cz['direction']} · price {_cz['price']:.5g} · "
                                    f"{_cz['score']} confluence factor(s)")
                        _tone_fn(_cz["validation_label"])
                        if _cz["factor_details"]:
                            for _factor in _cz["factor_details"]:
                                st.markdown(f"- {_factor['label']}")
                        else:
                            st.caption("No other currently-active ICT reads agree with this direction "
                                       "right now.")

                elif _key == "clean_patterns":
                    st.caption("Runs of consecutive FVG/Order Block zones that ALL got genuinely "
                               "respected — a wick tapping one is fine (ICT's own 'liquidity grab, "
                               "still respected'), but a close through it or a full 100% wick "
                               "mitigation ends the run. Every streak below is still unbroken as of "
                               "the last loaded bar — 'end' means 'still holding', not 'finished'.")
                    _cp_col1, _cp_col2 = st.columns(2)
                    with _cp_col1:
                        _cp_min_streak = st.number_input("Minimum run length", min_value=2, max_value=20,
                                                          value=3, step=1, key="cp_min_streak")
                    with _cp_col2:
                        _cp_zone_types = st.multiselect("Zone types", ["FVG", "Order Block"],
                                                          default=["FVG", "Order Block"], key="cp_zone_types")
                    if not _cp_zone_types:
                        st.caption("Pick at least one zone type.")
                    else:
                        _cp_streaks = find_clean_respect_streaks(
                            df, zone_types=tuple(_cp_zone_types), min_streak=_cp_min_streak)
                        if not _cp_streaks:
                            st.caption(f"No runs of {_cp_min_streak}+ consecutive respected zones found "
                                       f"on this chart's own currently-loaded history.")
                        else:
                            st.caption(f"{len(_cp_streaks)} found, most recent first.")
                            for _cp_i, _cp_s in enumerate(reversed(_cp_streaks)):
                                _cp_cond = describe_streak_conditions(df, _cp_s)
                                _cp_dir = _cp_cond["dominant_direction"]
                                _cp_label = (f"#{len(_cp_streaks) - _cp_i} — {_cp_s['n_zones']} zones "
                                             f"({_cp_s['bullish']}▲ {_cp_s['bearish']}▼), {_cp_dir}, "
                                             f"{_cp_cond['n_bars']} bars, vol {_cp_cond['vol_ratio_vs_median']:.2f}x "
                                             f"median, net {_cp_cond['net_move_pct']:+.2f}%")
                                _cp_row1, _cp_row2 = st.columns([5, 1])
                                with _cp_row1:
                                    st.markdown(f"**{_cp_label}**")
                                    st.caption(f"{_cp_s['start']} → {_cp_s['end']} ({_cp_cond['duration']})")
                                with _cp_row2:
                                    if st.button("View", key=f"cp_view_{ticker}_{tf_label}_{_cp_i}"):
                                        st.session_state["_selected_streak"] = _cp_s
                                        st.rerun()

            if len(_detail_tabs) == 1:
                _only_label, _only_key = _detail_tabs[0]
                with st.expander(_only_label):
                    _render_detail_body(_only_key)
            elif _detail_tabs:
                with st.expander("📋 Detail"):
                    _tab_objs = st.tabs([label for label, _key in _detail_tabs])
                    for _tab_obj, (_label, _key) in zip(_tab_objs, _detail_tabs):
                        with _tab_obj:
                            _render_detail_body(_key)
        except Exception as e:
            st.error(f"Could not fetch {ticker}: {e}")

    # Warm the cache for every fetch this rerun is about to make — main
    # chart, its 4h/15m reference frames, and both mini charts — all at
    # once, before any of them run for real below. See _prefetch's own
    # docstring for why this is the fix for "switching charts is slow."
    # provider=data_source in every spec below has to exactly match what the
    # real downstream call further down actually passes — st.cache_data
    # keys off the full argument set, so a mismatch here would warm the
    # wrong cache entry and silently defeat the whole prefetch (back to the
    # "switching charts is slow" bug, just for non-Auto sources only).
    _prefetch_specs = [(get_yf_ohlcv, (ticker,),
                         {"period": tf["period"], "interval": tf["fetch_interval"], "provider": data_source})]
    if tf["resample"] is None and refresh_interval is not None:
        _prefetch_specs.append((get_latest_bars, (ticker, tf["fetch_interval"]), {"provider": data_source}))
    _needed_layer_tfs = {(tf_label if layer_tf[_name] == "Chart TF" else layer_tf[_name]) for _name in layers}
    for _req_tf in _needed_layer_tfs:
        if _req_tf == tf_label:
            continue  # main df already covers this, prefetched above
        _req_conf = TIMEFRAMES[_req_tf]
        _prefetch_specs.append((get_yf_ohlcv, (ticker,),
                                 {"period": _req_conf["period"], "interval": _req_conf["fetch_interval"],
                                  "provider": data_source}))
    # _historical_win_rates does its OWN separate, much-longer-history fetch
    # (see its own docstring on why: the chart's own TIMEFRAMES period is
    # too thin a sample for a win rate worth showing) — on a cold cache
    # that's a real multi-second blocking call apiece, confirmed directly:
    # ~9s for a 730-day 4h fetch the first time. Warmed here in the SAME
    # parallel batch as everything else instead of inline in _render_chart,
    # so by the time the FVG/OB block below actually calls this, it's a
    # cache hit, not a fresh fetch.
    for _wr_layer in ("FVG", "Order Blocks"):
        if _wr_layer in layers:
            _prefetch_specs.append((_historical_win_rates, (ticker, layer_tf[_wr_layer], _wr_layer, data_source),
                                     {"news_blackout": _news_blackout}))
    for _chart_id in ("htf", "ltf"):
        _mini_tf_key = st.session_state.get(TF_KEY_BY_CHART[_chart_id], DEFAULT_TF_BY_CHART[_chart_id])
        _mini_conf = TIMEFRAMES[_mini_tf_key]
        _prefetch_specs.append((get_yf_ohlcv, (ticker,),
                                 {"period": _mini_conf["period"], "interval": _mini_conf["fetch_interval"],
                                  "provider": data_source}))
    if overlay_tf != "None" and overlay_tf != tf_label:
        _ov_conf = TIMEFRAMES[overlay_tf]
        _prefetch_specs.append((get_yf_ohlcv, (ticker,),
                                 {"period": _ov_conf["period"],
                                  "interval": _ov_conf["fetch_interval"], "provider": data_source}))
    _backfill_tf = _backfill_timeframe(tf_label)
    if _backfill_tf is not None:
        _bf_conf = TIMEFRAMES[_backfill_tf]
        _prefetch_specs.append((get_yf_ohlcv, (ticker,),
                                 {"period": _bf_conf["period"],
                                  "interval": _bf_conf["fetch_interval"], "provider": data_source}))
    _prefetch(_prefetch_specs)

    # Also warm EVERY OTHER timeframe's own base candle history — not just
    # what THIS render needs — so whichever TF button gets clicked NEXT is
    # already a warm get_yf_ohlcv cache entry instead of a cold fetch.
    # Confirmed directly: a first-time (ticker, period, interval) combo
    # took ~3.9s end to end for the chart to update, a warm one ~2s — this
    # closes that gap for every TF, not just the ones already in play
    # above. Every backfill-source/overlay/layer timeframe this render
    # could ever reach for is one of these same TIMEFRAMES entries, so
    # this transitively warms those too — no separate backfill-chain loop
    # needed.
    #
    # Deliberately NOT folded into _prefetch_specs above (a first version
    # of this did exactly that, and it was wrong): _prefetch blocks THIS
    # render — the one the user is actually looking at right now — until
    # EVERY one of its specs finishes, active-TF included. Bundling 9 more
    # timeframes' worth of fetches into that same blocking wait meant a
    # slow background combo could delay the chart currently on screen for
    # no reason it would ever need that data. warm_in_background instead
    # fires these through a SEPARATE, small, persistent pool (see its own
    # docstring in data.py) that this script never waits on at all — this
    # render proceeds to _render_chart() the moment ITS OWN needs are met,
    # full stop, regardless of how the other 9 are doing.
    #
    # Sorted smallest-first (by estimated bar count) rather than in
    # TIMEFRAMES' own declared order — with only 3 background workers,
    # cheap timeframes (1D's ~700 bars) finish and free a slot almost
    # immediately, while the most expensive one (5m's ~17k bars) would
    # otherwise occupy a worker for the whole batch's duration if started
    # first, needlessly delaying everything queued behind it.
    _already_warm_tfs = {tf_label, _backfill_tf} | _needed_layer_tfs
    if overlay_tf != "None":
        _already_warm_tfs.add(overlay_tf)
    for _cid2 in CHART_IDS:
        _already_warm_tfs.add(st.session_state.get(TF_KEY_BY_CHART[_cid2], DEFAULT_TF_BY_CHART[_cid2]))
    _bg_warm_tfs = [k for k in TIMEFRAMES if k not in _already_warm_tfs]
    _bg_warm_tfs.sort(key=lambda k: _TF_PERIOD_DAYS[k] * 86400 / _TF_BAR_SECONDS[k])
    for _bg_tf_key in _bg_warm_tfs:
        _bg_tf_conf = TIMEFRAMES[_bg_tf_key]
        warm_in_background(get_yf_ohlcv, (ticker,),
                            {"period": _bg_tf_conf["period"],
                             "interval": _bg_tf_conf["fetch_interval"], "provider": data_source})

    def _render_replay_chart(rdf, r_ticker, r_tf_label, r_data_source, r_n_total):
        """The actual candles + zones + trade-pick for one replay step —
        split out from _render_bar_replay only so that function's own
        control-row logic isn't buried under this. rdf is a plain PREFIX
        SLICE of the full history (bars[:idx+1]) — every detector/
        best_trade_now call below sees ONLY that slice, so a zone or a
        trade pick can never reflect anything that hasn't "happened" yet
        at this point in the replay."""
        # Every tick genuinely IS new data (the array grows one bar every
        # step), so the fingerprint changes every tick by design — a full
        # rebuild is the honest, correct thing here, not something to dodge
        # via a stable fingerprint. What actually caused "adjusts to the
        # right side" (direct feedback) wasn't the rebuild itself, it was
        # the existing calendar-window RESTORE logic that runs after a
        # rebuild — built for the live chart's own occasional ticker/TF
        # switches, and fragile under a full reload firing every single
        # tick during Play. See "preserve_logical_range" on this chart's
        # own options below, and the frontend's own handling of it, for the
        # actual fix — a direct, synchronous save/restore around THIS
        # rebuild, with no debounce or calendar-time conversion involved.
        _idx = len(rdf) - 1

        o_col, h_col, l_col, c_col = (("Open", "High", "Low", "Close") if "Close" in rdf
                                       else ("open", "high", "low", "close"))
        v_col = "Volume" if "Volume" in rdf else ("volume" if "volume" in rdf else None)
        axis_secs = _ny_fake_utc_seconds_vec(rdf.index)
        _o, _h, _l, _c = (rdf[o_col].to_numpy(), rdf[h_col].to_numpy(),
                          rdf[l_col].to_numpy(), rdf[c_col].to_numpy())
        _v = rdf[v_col].to_numpy() if v_col is not None else None
        bars = [
            {"time": int(axis_secs[i]), "open": float(_o[i]), "high": float(_h[i]),
             "low": float(_l[i]), "close": float(_c[i]), "volume": float(_v[i]) if _v is not None else 0.0}
            for i in range(len(rdf))
        ]

        rectangles, price_lines = [], []
        last_x = int(axis_secs[-1])
        # Real (if invisible) trailing bars, extending the series' OWN
        # data past the last revealed candle — not just a wider requested
        # view. Confirmed directly: lightweight-charts silently clamps any
        # setVisibleLogicalRange request past the series' own last real
        # point, even by a single index, no matter how it got there — a
        # wider view has nowhere to go unless the series itself has real
        # points to occupy that space. Flat (repeats the last close) and
        # fully transparent (color/borderColor/wickColor all rgba(0,0,0,0))
        # — invisible, but real data the timeScale can position and scroll
        # into, the same anchor-bar technique this project's own backfill
        # code already uses for coarser-timeframe candles elsewhere. Direct
        # feedback: "I still can't scroll to see future timeline."
        #
        # 40, not a bigger number: fitContentWhenReady's own margin formula
        # (see its own comment) turns out self-referential once ANY padding
        # exists — its computed `to` always ends up past data.length-1 by
        # construction (it's lastIdx PLUS a positive margin), so the
        # frontend's own safety clamp always lands `to` exactly on the
        # padded array's own last index, regardless of future_margin_frac's
        # actual value. That makes THIS number, not that fraction, the
        # real lever on how much margin shows — and DEFAULT_VISIBLE_BARS
        # (120, frontend-side) is the total window width, so a bigger pad
        # count directly eats into how many REAL candles are visible by
        # default (confirmed directly: 90 padding bars left only 30 real
        # ones on screen, mostly blank space). 40 comfortably covers the
        # 30-bar trade box with a little room to spare while keeping most
        # of the default view real, meaningful price action.
        _last_close = float(_c[-1])
        for _p in range(1, 41):
            bars.append({
                "time": last_x + _p * _TF_BAR_SECONDS[r_tf_label],
                "open": _last_close, "high": _last_close, "low": _last_close, "close": _last_close, "volume": 0.0,
                "color": "rgba(0,0,0,0)", "borderColor": "rgba(0,0,0,0)", "wickColor": "rgba(0,0,0,0)",
            })

        # The "live areas" — currently-open FVG/order-block zones, using
        # the exact same detectors and color convention (bullish FVG=cyan,
        # bearish=magenta, bullish OB=green, bearish OB=amber) the live
        # chart's own FVG/Order Blocks layers already use — trimmed to the
        # essentials (no win-rate labels/indicator confluence) since this
        # is a focused replay view, not the full layer system.
        #
        # Two deliberately different selections, direct request ("one that
        # tracks previous candles, and one that scans historical values"):
        # recent_zone_tracker (Engine A) surfaces EVERY zone formed in the
        # last 50 bars, uncapped — "mark everything in recent past... with
        # surgical precision" — while historical_zone_scanner (Engine B)
        # finds the nearest zones above/below current price from the FULL
        # history regardless of age — "shows me where price might go even
        # if it exits the range." Distance-based selection ALONE (this
        # replay's own previous approach, and still the live chart's own
        # _nearest_by_price) was confirmed as a real bug for a fast-moving
        # replay: re-ranking "nearest overall" on every tick as price
        # wiggles made a genuinely still-open zone flicker in and out of
        # view for no reason tied to the zone itself. Age-based selection
        # (Engine A) doesn't have that failure mode; Engine B keeps the
        # "show me the actual nearest levels" behavior without being the
        # ONLY lens onto what's currently open.
        #
        # n_per_side used to be hardcoded to 2 here — confirmed directly
        # (offline replay of this exact ticker at bars 533-536) that every
        # single appearance/disappearance in a direct report of "areas
        # appear and disappear" was a REAL event (a zone getting mitigated
        # by actual price action, or a genuinely new one confirming on its
        # own displacement bar) — detect_order_blocks/detect_fvgs never
        # retroactively reclassify an already-confirmed candle as more
        # history streams in, and mitigation is monotonic (never reverses).
        # So there's no actual bug in what gets detected or when — but a
        # hardcoded 2-per-side cap on the distance-ranked sleeve means a
        # real zone getting bumped by a marginally closer newcomer reads as
        # constant churn, with almost no headroom to absorb it. Reusing the
        # user's own "Areas per side" setting (max_items_per_layer, same
        # control the live chart's own layers already respect) instead of
        # a hardcoded 2 gives real breathing room, still 100% accurate to
        # actual market events — this only changes how many currently-
        # active zones get shown at once, not which ones are real.
        _replay_max_items = st.session_state.get("_chart_controls", {}).get("max_items_per_layer", 5)
        current_price = float(_c[-1])
        _open_fvgs = [g for g in detect_fvgs(rdf) if not g["filled"]]
        _open_obs = [o for o in detect_order_blocks(rdf) if not o["mitigated"]]
        _fvgs = merge_zone_engines(
            recent_zone_tracker(_open_fvgs, rdf, lookback_bars=50),
            historical_zone_scanner(_open_fvgs, current_price, n_per_side=_replay_max_items))
        _obs = merge_zone_engines(
            recent_zone_tracker(_open_obs, rdf, lookback_bars=50),
            historical_zone_scanner(_open_obs, current_price, n_per_side=_replay_max_items))
        for g in _fvgs:
            color = theme.NEON_CYAN if g["type"] == "bullish" else theme.NEON_MAGENTA
            rectangles.append({"t0": _ny_fake_utc_seconds(g["start"]), "t1": last_x,
                                "p0": g["bottom"], "p1": g["top"],
                                "fill": _hex_to_rgba(color, 0.2), "border": _hex_to_rgba(color, 0.7),
                                "label": "FVG"})
        for ob in _obs:
            color = theme.NEON_GREEN if ob["type"] == "bullish" else theme.NEON_AMBER
            rectangles.append({"t0": _ny_fake_utc_seconds(ob["start"]), "t1": last_x,
                                "p0": ob["bottom"], "p1": ob["top"],
                                "fill": _hex_to_rgba(color, 0.2), "border": _hex_to_rgba(color, 0.7),
                                "label": "OB"})

        # "scans for live areas and displays potential trades as time
        # ticks" — the exact same engine behind the sidebar's Signals scan
        # and the live chart's own active-trade box (best_trade_now),
        # just handed this replay slice instead of the live df. Single-
        # timeframe confluence only (no HTF context_df) — a deliberate v1
        # scope cut, not an oversight: syncing a SECOND timeframe's own
        # frame to this same replay cursor is real extra work, left for a
        # follow-up if this turns out to matter in practice.
        picks = best_trade_now(rdf, r_ticker, TIMEFRAMES[r_tf_label]["fetch_interval"], r_data_source, top_n=1,
                                news_blackout=st.session_state.get("_chart_controls", {}).get("news_blackout"))
        trade_box_seconds = _TF_BAR_SECONDS[r_tf_label] * 30
        box_t1 = last_x + trade_box_seconds
        if picks:
            pick = picks[0]
            entry_price, stop_price, exit_price = pick["entry_price"], pick["sl_price"], pick["tp_price"]
            pattern = EVENT_TYPE_LABELS.get(pick["event_type"], pick["event_type"])
            for price, color, title in [(entry_price, theme.NEON_AMBER, "Entry"),
                                         (stop_price, theme.NEON_MAGENTA, "SL"),
                                         (exit_price, theme.NEON_GREEN, "TP")]:
                price_lines.append({"t0": last_x, "t1": box_t1, "price": price,
                                     "color": _hex_to_rgba(color, 1.0), "title": title,
                                     "line_width": 2, "dashed": True, "above": bool(price >= entry_price)})
            rectangles.append({"t0": last_x, "t1": box_t1, "p0": entry_price, "p1": stop_price,
                                "fill": _hex_to_rgba(theme.NEON_MAGENTA, 0.10),
                                "border": _hex_to_rgba(theme.NEON_MAGENTA, 0.9)})
            rectangles.append({"t0": last_x, "t1": box_t1, "p0": entry_price, "p1": exit_price,
                                "fill": _hex_to_rgba(theme.NEON_GREEN, 0.10),
                                "border": _hex_to_rgba(theme.NEON_GREEN, 0.9),
                                "body_lines": _confluence_body_lines(pick),
                                "label": f"{pattern} ({pick['direction']})"})
            caption = f"Potential trade: {pattern} ({pick['direction']}) · confluence {pick['confluence_score']}"
        else:
            caption = "No active setup at this point in history."
        st.caption(f"{caption} — {len(rdf)} of {r_n_total} bars revealed.")

        fingerprint = f"replay|{r_ticker}|{r_tf_label}|{_idx}"
        ict_chart(
            bars, fingerprint,
            overlays={"rectangles": rectangles, "price_lines": price_lines, "markers": [], "ghost_candles": []},
            # No always_fit (unlike the mini reference panels) — that
            # re-fits on every single reload, which for a chart whose
            # fingerprint changes every tick means every tick, exactly the
            # "rescales with every tick" behavior direct feedback pushed
            # back on. preserve_logical_range is the real fix for that: a
            # plain synchronous save-then-restore of the exact visible
            # logical range around this reload (see this option's own
            # frontend handling) — correct specifically because a replay
            # dataset only ever grows at the END (bar 0 never moves), so
            # the SAME logical indices still mean the same bars next tick,
            # with no need for the live chart's own calendar-time
            # conversion or debounced-settle machinery (built for an
            # arbitrary ticker/TF switch, and fragile once a reload fires
            # every tick during Play instead of on rare, deliberate
            # switches). future_margin_frac still sets the INITIAL window
            # (first render of a session, nothing to preserve yet) — needs
            # the padding bars above to actually be visible, since
            # lightweight-charts clamps any requested range past the
            # series' own last real point.
            options={"volume": False, "selected": False, "future_margin_frac": 0.4, "preserve_logical_range": True,
                     "log_scale": st.session_state.get("fvg_log_scale", False)},
            # No "symbol" key here on purpose — the frontend's connectLiveFeed
            # guard (`if (!tickerSymbol...) return`) treats a missing symbol
            # as "don't connect," which is exactly what a replay needs: this
            # component instance must never receive genuine live ticks,
            # crypto ticker or not.
            ohlc={"ticker": r_ticker, "source": "Replay", "forming_bar_time": None},
            height=650,
            key="ict_chart_replay",
            display_tz=theme.get_display_tz(),
        )

    def _render_bar_replay(r_ticker, r_tf_label, r_tf, r_data_source):
        """TradingView-style Bar Replay: rewind to a point in this ticker's
        own history, then step through it bar by bar (or auto-play) with
        _render_replay_chart above recomputing candles/zones/best-trade
        fresh from ONLY the bars revealed so far — the same read a live
        trader would have had at that exact moment. A separate, self-
        contained render path rather than a mode flag threaded through
        _render_chart: that function is a ~1500-line fragment with its own
        scattered fetch/splice/backfill/indicator machinery tuned for the
        live case (see its own module-level comments) — truncating every
        one of those call sites would risk that carefully-tuned live path
        to build something that only ever needs candles + two zone types +
        one trade pick. Reuses the same cached get_yf_ohlcv fetch and the
        exact same detectors/best_trade_now the live chart and sidebar
        scans already use, unmodified — they were always just "whatever df
        you hand them," a full history or a replay slice makes no
        difference to them.

        Everything lives in ONE fragment, including Play/Pause — a
        `run_every`-based design (redefining the decorator's own interval
        from an OUTER, non-fragment Play button, mirroring the main TF
        radio's own established split) was tried first and confirmed
        broken here: a full rerun that redefines this fragment with a new
        `run_every` does NOT reliably cancel whatever periodic tick the
        PREVIOUS instance already had scheduled, so Pause stopped updating
        the label without actually stopping the advance — the position
        kept climbing underneath a control that visibly claimed it was
        paused. Auto-play here instead self-drives via a plain
        `time.sleep()` + `st.rerun(scope="fragment")` loop at the tail of
        a single, always-`run_every=None` fragment: every click (Play,
        Pause, step, drag) is a fresh execution of this SAME fragment that
        checks "should I still be advancing" from scratch before ever
        looping again, so there is never more than one advance path alive
        at once. The tradeoff — a Pause click can take up to one sleep
        interval (0.25-2s) to land — is a small, honest cost next to a
        Pause button that silently doesn't pause."""
        base_df = get_yf_ohlcv(r_ticker, period=r_tf["period"], interval=r_tf["fetch_interval"], provider=r_data_source)
        if r_tf["resample"]:
            base_df = resample_ohlc(base_df, r_tf["resample"])
        if base_df.empty or len(base_df) < 25:
            st.warning(f"Not enough history for {r_ticker} at {r_tf_label} to replay.")
            return

        n = len(base_df)
        state_key = f"_replay_state_{r_ticker}_{r_tf_label}"
        idx_key = f"{state_key}_idx"
        if st.session_state.get(state_key, {}).get("n") != n:
            # Fresh ticker/timeframe, or the underlying history's own
            # length shifted under it (new data arrived) — reset to a sane
            # starting point rather than carry an index that might now
            # mean something else entirely.
            st.session_state[state_key] = {"playing": False, "speed": "1x", "n": n}
            st.session_state[idx_key] = min(n - 2, max(20, n // 3))

        @st.fragment
        def _replay_frame():
            # idx_key (the slider's own bound key) is the ONE canonical
            # store for the current position — every control that moves
            # it writes st.session_state[idx_key] directly, BEFORE the
            # slider widget below is instantiated, rather than handing it
            # a fresh `value=` each render. Confirmed directly as a real
            # bug otherwise: once a keyed widget has rendered once,
            # Streamlit uses ITS OWN persisted value on every later rerun
            # and silently ignores a new `value=` argument — a step/auto-
            # play update looked right for one render, then the slider's
            # own frozen state snapped it straight back on the next one.
            s = st.session_state[state_key]
            # A pending auto-play advance MUST land here, before the
            # slider below (key=idx_key) is instantiated this run — same
            # "can't modify a keyed widget's value after it's rendered"
            # rule applies WITHIN a single fragment pass too, not just
            # across full reruns. The tail of this function (see its own
            # sleep+rerun block) never touches idx_key directly; it just
            # flags the advance and reruns, so it's always applied here,
            # one step ahead of the widget that owns that key.
            if s.pop("_advance_pending", False):
                st.session_state[idx_key] = min(n - 1, st.session_state[idx_key] + 1)

            _play_col, _speed_col, _exit_col = st.columns([1.1, 1, 1.2])
            with _play_col:
                if st.button("⏸ Pause" if s["playing"] else "▶️ Play", key=f"{state_key}_play", width="stretch"):
                    s["playing"] = not s["playing"]
            with _speed_col:
                _speed_options = ["0.5x", "1x", "2x", "4x"]
                s["speed"] = st.selectbox("Speed", _speed_options, index=_speed_options.index(s["speed"]),
                                           key=f"{state_key}_speed", label_visibility="collapsed")
            with _exit_col:
                if st.button("✕ Exit replay", key=f"{state_key}_exit", width="stretch"):
                    # Can't write st.session_state["_replay_active"] directly
                    # here — that checkbox widget already rendered earlier
                    # in THIS run, and Streamlit raises on mutating a
                    # widget's own bound key post-instantiation. Deferred
                    # flag instead, applied at the very top of the next run
                    # before that checkbox exists yet (see its read site
                    # below).
                    st.session_state["_replay_exit_requested"] = True
                    st.session_state.pop(state_key, None)
                    st.session_state.pop(idx_key, None)
                    st.rerun()

            _slider_col, _back_col, _fwd_col = st.columns([6, 1, 1])
            with _back_col:
                if st.button("◀", key=f"{state_key}_back", help="Step back one bar", width="stretch"):
                    st.session_state[idx_key] = max(20, st.session_state[idx_key] - 1)
                    s["playing"] = False
            with _fwd_col:
                if st.button("▶", key=f"{state_key}_fwd", help="Step forward one bar", width="stretch"):
                    st.session_state[idx_key] = min(n - 1, st.session_state[idx_key] + 1)
                    s["playing"] = False
            with _slider_col:
                picked = st.slider(
                    "Replay position", 20, n - 1, key=idx_key, label_visibility="collapsed",
                    help="Drag to rewind — the chart, its FVG/order-block zones, and the best-trade pick "
                         "below all recompute using ONLY bars up to here, exactly as they'd have looked "
                         "live at that moment.")

            if picked >= n - 1:
                s["playing"] = False
            st.session_state[state_key] = s

            _render_replay_chart(base_df.iloc[: picked + 1], r_ticker, r_tf_label, r_data_source, n)

            if s["playing"] and picked < n - 1:
                time.sleep({"0.5x": 2.0, "1x": 1.0, "2x": 0.5, "4x": 0.25}[s["speed"]])
                # Re-check right before looping — a Pause/Exit click that
                # landed during the sleep above must win, not get overrun
                # by a stale decision made before it. Flags the advance
                # rather than applying it here directly — idx_key's own
                # widget (the slider above) already rendered THIS pass, so
                # writing it now would hit the exact same "modified after
                # instantiation" error; the next pass's own top handles it
                # instead (see this function's own opening lines).
                _latest = st.session_state.get(state_key)
                if _latest and _latest.get("playing"):
                    _latest["_advance_pending"] = True
                    st.session_state[state_key] = _latest
                    st.rerun(scope="fragment")

        _replay_frame()

    if st.session_state.pop("_replay_exit_requested", False):
        st.session_state["_replay_active"] = False
    st.session_state.setdefault("_replay_active", False)
    _replay_on = st.checkbox(
        "🎬 Bar Replay", key="_replay_active",
        help="Rewind this chart to a point in its own history, then step through it bar by bar (or "
             "auto-play) — candles, FVG/order-block zones, and the recommender's own \"best trade right "
             "now\" pick all unfold fresh from ONLY the bars revealed so far, exactly what a live trader "
             "would have seen at that moment. No lookahead into the future.")
    if _replay_on:
        _render_bar_replay(ticker, tf_label, tf, data_source)
    else:
        _render_chart()
