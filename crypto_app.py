"""
Crypto — the ICT terminal scoped to Bitcoin/crypto only. See app.py for
the separate Forex/Indices/Commodities counterpart ("Markets"); the two
share every underlying module (fvg.py, theme.py, data.py, recommender.py,
ict_chart) and differ only in which curated symbol list their own ticker
picker shows. Independent of market-dashboard — see README.md for why.
"""

import concurrent.futures
import json
import os
import threading
from datetime import time as dtime

import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

import backtest_ui
import theme
from data import get_crypto_universe, get_latest_bars, get_yf_ohlcv, is_ticker_alive, resample_ohlc, warm_in_background
from fvg import (
    current_dealing_range,
    detect_equal_levels,
    detect_fvgs,
    detect_liquidity_levels,
    detect_liquidity_reactions,
    detect_order_blocks,
    detect_structure_breaks,
    detect_swings,
)
from ict_chart import ict_chart
from indicators import bollinger_bands, ema, macd, rsi, volume_profile
from recommender import (
    EVENT_TYPE_LABELS,
    EXIT_EVENT_LABELS,
    INDICATOR_SPECS,
    MA_FVG_PERIODS,
    RULE_DETECTOR_LABELS,
    backtest_custom_rule,
    fvg_event_win_rate,
    fvg_run_up_stats,
    liquidity_event_win_rate,
    ma_fvg_event_win_rate,
    ma_fvg_starts,
    order_block_event_win_rate,
    point_level_indicator_matches,
    resolve_rule,
    rule_describe,
    scan_watchlist,
    zone_indicator_matches,
)
from research.data_loader import INTERVAL_MAX_PERIOD, load_history


st.set_page_config(page_title="Crypto", layout="wide", initial_sidebar_state="collapsed")
theme.inject(chart_layout=True)


def tiny(text):
    """Genuinely optional text — small and muted, never competing with real content."""
    st.markdown(f'<div class="tiny-note">{text}</div>', unsafe_allow_html=True)


def _hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


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
# Deliberately NOT the full permutation test research/backtest.py runs for
# Research Lab/Edge Lab — that's 5000 random-direction trials, built for a
# considered, one-at-a-time verdict, not something to re-run on every chart
# rerun for every zone. This is a plain raw-return win rate: enough for a
# quick on-chart reference number, not a claim of statistical significance.
# Long TTL (1h) because it's backward-looking history, not live price — it
# has no reason to change minute to minute the way the chart itself does.
_WIN_RATE_MIN_EVENTS = 20


@st.cache_data(ttl=3600, show_spinner=False)
def _historical_win_rates(ticker, tf_label, layer, provider, exit_types=tuple(EXIT_EVENT_LABELS.keys())):
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
    a real fill would have net after cost."""
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

    fn = {"FVG": fvg_event_win_rate, "Order Block": order_block_event_win_rate,
          "MA+FVG": ma_fvg_event_win_rate, "Liquidity": liquidity_event_win_rate}[layer]
    return fn(df, exit_types=exit_types, min_events=_WIN_RATE_MIN_EVENTS)


_EPOCH = pd.Timestamp("1970-01-01")


def _ny_fake_utc_seconds(ts):
    """lightweight-charts has no timezone setting — it always renders a
    UTCTimestamp's UTC wall-clock digits. The standard workaround: convert
    to the timezone you actually want shown (America/New_York, matching ICT
    kill-zone convention), strip the tz label, and encode THOSE digits as
    if they were UTC."""
    ny = ts.tz_convert("America/New_York").tz_localize(None)
    return int((ny - _EPOCH).total_seconds())


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
    ny_idx = idx.tz_convert("America/New_York").tz_localize(None)
    return ((ny_idx - _EPOCH) // pd.Timedelta(seconds=1)).tolist()


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
# Only used if the live Binance fetch below comes back empty (endpoint
# briefly down) — a search box with today's 2 majors beats one with none.
_FALLBACK_CRYPTO = {"BTC-USD": ("Bitcoin", "Crypto"), "ETH-USD": ("Ethereum", "Crypto")}


@st.cache_data(ttl=86400)
def _build_ticker_info():
    """Every live Binance USDT pair (no key needed) — this dashboard is
    crypto-only, see app.py for the Forex/Indices/Commodities counterpart.
    Falls back to _FALLBACK_CRYPTO if the live fetch comes back empty."""
    crypto = {symbol: (name, "Crypto") for symbol, name in get_crypto_universe()}
    return crypto if crypto else dict(_FALLBACK_CRYPTO)


TICKER_INFO = _build_ticker_info()
TICKER_NAMES = {t: info[0] for t, info in TICKER_INFO.items()}
TICKER_UNIVERSE = list(TICKER_INFO.keys())

# The sidebar watchlist scan's own universe — deliberately NOT the full
# TICKER_UNIVERSE (hundreds of live Binance pairs; scanning all of them on
# every click would be slow and mostly noise from thin/obscure pairs
# nobody's actually watching). A hand-picked set of the actually-liquid
# majors, same curated-list reasoning as app.py's CURATED_FOREX/INDICES.
SCAN_TICKERS = [t for t in ("BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD",
                             "DOGE-USD", "AVAX-USD", "LINK-USD", "LTC-USD") if t in TICKER_INFO]

# Display label <-> internal id for data.py's provider chain (see its own
# _all_providers). "Auto" is Yahoo-first-with-fallback, today's default
# behavior; picking one of the other four FORCES that single provider, with
# no fallback if it fails — a way to actually audit/compare a specific
# source rather than just hoping the right one served a given render.
DATA_SOURCES = {"Auto": "auto", "Yahoo Finance": "yahoo", "Binance": "binance",
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
_MAX_TOTAL_TILES = 5000


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


ICT_LAYERS = ["FVG", "Order Blocks", "Swing Points", "Equal Highs/Lows", "Market Structure", "Premium/Discount", "Liquidity"]
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
    # Read from session_state rather than a passed-in arg — this function is
    # a plain module-level def (not nested inside the settings popover's own
    # scope the way _render_chart is), same reason fvg_show_volume below is
    # read the same way.
    data_source = DATA_SOURCES.get(st.session_state.get("fvg_data_source", "Auto"), "auto")

    @st.fragment(run_every=refresh_interval)
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
                options={"log_scale": False, "volume": st.session_state.get("fvg_show_volume", False),
                          "selected": st.session_state.get("selected_chart") == chart_id,
                          # Plain candles at a glance — no O/H/L/C/source
                          # heading competing with the main chart's own
                          # for space in this smaller panel.
                          "hide_ohlc": True},
                ohlc={"ticker": "", "source": tf_key, "symbol": ticker, "interval": tf_key},
                height=380,
                key=f"ict_chart_mini_{chart_id}",
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
#
# Own file, not app.py's .last_state.json — confirmed directly this needs
# to be separate: sharing one file meant picking a ticker on Markets
# silently overwrote what Crypto would restore on its own next load (and
# vice versa), two independent dashboards fighting over one "last used"
# slot.
_LAST_STATE_FILE = os.path.join(os.path.dirname(__file__), ".last_state_crypto.json")


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

theme.render_project_nav("Crypto")

st.session_state.setdefault("selected_chart", "main")
selected_chart = st.session_state["selected_chart"]

# fvg_ticker is plain session_state now, not a selectbox's own value — the
# market-menu buttons below set it directly (see "Markets" expander), so
# there's no widget to read it FROM at this point in the script.
ticker = st.session_state.get("fvg_ticker", "BTC-USD").strip().upper()

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

# Sidebar: a multi-symbol scan across a curated set of major coins —
# collapsed by default (initial_sidebar_state="collapsed" above), so the
# chart stays the primary view. Same reasoning as app.py's own Markets
# counterpart: the "advanced, scan everything" functionality lives here
# instead of a separate page, one click from the chart it feeds.
with st.sidebar:
    st.subheader("🎯 Signals")
    st.caption("Scans the major coins for each one's own single best "
               "currently-active setup, ranked the same way as the "
               "on-chart pick (Edge Lab validation first, confluence "
               "second) — see which coin has the strongest read right "
               "now, not just whatever's already charted.")
    if st.button("🔍 Scan watchlist", key="signals_scan_btn", width="stretch"):
        _scan_conf = TIMEFRAMES["4h"]
        _scan_specs = [(get_yf_ohlcv, (t, _scan_conf["period"], _scan_conf["fetch_interval"], _prefetch_data_source), {})
                       for t in SCAN_TICKERS]
        _prefetch(_scan_specs)
        _scan_dfs = {}
        for _t in SCAN_TICKERS:
            try:
                _scan_dfs[_t] = get_yf_ohlcv(_t, period=_scan_conf["period"], interval=_scan_conf["fetch_interval"],
                                              provider=_prefetch_data_source)
            except Exception:
                _scan_dfs[_t] = None
        st.session_state["_signals_scan"] = scan_watchlist(_scan_dfs, _scan_conf["fetch_interval"], top_n=8)

    _scan_results = st.session_state.get("_signals_scan")
    if _scan_results is None:
        st.caption("Not scanned yet this session.")
    elif not _scan_results:
        st.caption("Scanned — nothing currently active anywhere on the watchlist right now.")
    else:
        _rank_dot = {3: "🟢", 2: "🟡", 1: "🟠", 0: "⚪"}
        for _s in _scan_results:
            _dot = _rank_dot.get(_s["validation_rank"], "⚪")
            _pattern = EVENT_TYPE_LABELS.get(_s["event_type"], _s["event_type"])
            _label = f"{_dot} {_s['ticker']} · {_pattern} ({_s['direction']}) · {_s['confluence_score']}/4"
            if st.button(_label, key=f"sig_jump_{_s['ticker']}_{_s['event_type']}_{_s['start']}", width="stretch"):
                st.session_state["fvg_ticker"] = _s["ticker"]
                st.rerun()

# A popover, not an expander — opening it floats the menu over the page
# instead of pushing the chart down (see app.py's Markets counterpart for
# the same swap and why: click-based, not hover, since hover-to-reveal
# would need Streamlit's own internal DOM/CSS rather than a stable public
# API). The button's own label always shows the current pick, so which
# market is loaded stays visible even collapsed. No category-button row
# here (unlike app.py's Markets counterpart) — this dashboard only ever
# has the one category, Crypto, so a button for it would just be a
# permanently-active no-op.
with st.popover(f"🔍 {_ticker_label(ticker)} · change market", width="stretch"):
    _cat_symbols = [t for t in TICKER_UNIVERSE if TICKER_INFO[t][1] == "Crypto"]
    _search_query = st.text_input(
        "Search", key="fvg_ticker_search", label_visibility="collapsed",
        placeholder=f"Search {len(_cat_symbols)} coins…",
    )
    if _search_query.strip():
        _q = _search_query.strip().upper()
        _matches = [t for t in _cat_symbols if _q in t.upper() or _q in TICKER_INFO[t][0].upper()]
    else:
        # Hundreds of live Binance pairs — showing an arbitrary
        # alphabetical slice isn't useful, and live-checking a batch
        # nobody asked for wastes the fetch. Wait for a real query.
        _matches = []
    _matches = sorted(_matches)[:40]

    if _matches:
        # Live-checked only for what's actually about to be shown.
        # Parallelized the same way chart-switching's own fetches are
        # (see _prefetch) since up to 40 of these can be uncached on a
        # first search.
        _prefetch([(is_ticker_alive, (t,), {}) for t in _matches])
        for t in _matches:
            _dot = "🟢" if is_ticker_alive(t) else "🔴"
            if st.button(f"{_dot}  {_ticker_label(t)}", key=f"fvg_pick_{t}", use_container_width=True):
                st.session_state["fvg_ticker"] = t
                st.rerun()
    elif _search_query.strip():
        st.caption("No matches.")
    else:
        st.caption(f"Type to search {len(_cat_symbols)} coins…")

# mtf_col used to carry the FULL Layers/Settings/Rules/Charts panel at a
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
# top-right corner (see the CSS + "_menu_trigger_wrap" container further
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
    # down (position: absolute; top:0; right:0) anchors to main_col's own
    # top-right corner, not the page's or some other column's. Tried
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
    # The 4 panels (Layers, Settings, Rules, Charts) used to be 3 separate
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
    # see app.py's own copy of this comment for the full rationale
    # (identical here, mirrored per this project's own convention of
    # duplicating page-level orchestration while sharing the underlying
    # modules): the rail existed only because collapsing it used to be the
    # sole way to give the chart its width back, which is no longer true
    # now that this panel is a popover/drawer floating OVER the chart.
    @st.fragment
    def _render_layer_controls():
        _prev_cc = st.session_state.get("_chart_controls")
        _tab_layers, _tab_settings, _tab_rules, _tab_charts = st.tabs([
            ":material/layers: Layers", ":material/tune: Settings",
            ":material/rule: Rules", ":material/candlestick_chart: Charts",
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
            # One timeframe dropdown per layer, not a fixed 4h/15m pair — each
            # layer detects against exactly the timeframe its own dropdown
            # says, independent of every other layer and of the main chart's
            # own candle timeframe. Defaults to whatever the main chart is
            # showing (see the seeding block above), so "follow the main
            # selector" is the out-of-the-box behavior; picking something else
            # here is a deliberate per-layer override, not a mistake to warn
            # about — there's no more "both toggles off" dead state to fall
            # into, a dropdown always has exactly one value.
            layers = []
            layer_tf = {}
            for name in ICT_LAYERS:
                name_col, tf_col = st.columns([3, 2])
                with name_col:
                    # No value= here — session_state is already seeded (see
                    # setdefault block above), from the persisted last-used
                    # state or LAYER_DEFAULTS, before this widget is created.
                    on = st.checkbox(name, key=f"fvg_layer_{name}")
                with tf_col:
                    chosen_tf = st.selectbox(f"{name} timeframe", ["Chart TF"] + list(TIMEFRAMES.keys()),
                                              key=f"fvg_tf_{name}", label_visibility="collapsed",
                                              help="'Chart TF' (default) follows whatever timeframe the "
                                                   "main chart itself is showing. Pick a specific one to "
                                                   "detect this layer on its own fixed timeframe instead, "
                                                   "independent of the main chart selector.")
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
                                           help="Show/hide the EMA 20 / EMA 50 lines on the main chart.")
            _ind_tf_options = ["Chart TF"] + list(TIMEFRAMES.keys())
            indicator_tf = st.selectbox(
                "MA timeframe", _ind_tf_options, index=0, key="fvg_indicator_tf",
                label_visibility="collapsed", disabled=not show_indicators,
                help="Which timeframe the EMA 20/50 lines are computed on. 'Chart TF' (default) "
                     "matches whatever timeframe the main chart itself is showing. Pick a higher "
                     "one (e.g. viewing 15m candles but computing the average on 1h closes) to "
                     "see a steadier, less noisy line laid over a more detailed chart.",
            )
            show_rsi = st.checkbox("Show RSI", value=False, key="fvg_show_rsi",
                                    help="Adds an RSI (14) pane below the main chart.")
            show_macd = st.checkbox("Show MACD", value=False, key="fvg_show_macd",
                                     help="Adds a MACD (12/26/9) pane below the main chart.")
            show_bb = st.checkbox("Show Bollinger Bands", value=False, key="fvg_show_bb",
                                   help="Adds Bollinger Bands (20-period, 2 std) over the candles "
                                        "on the main chart.")
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
        with _tab_settings, st.container(key="_panel_settings"):
            selected_kill_zones = st.multiselect("Kill zones", list(KILL_ZONES.keys()),
                                                  default=["NY AM (9:30–11:30 ET)"], key="fvg_kill_zones")
            show_mitigated = st.checkbox("Show mitigated/filled areas", value=False, key="fvg_show_mitigated")
            # Volume used to be permanently-on base chart furniture (no toggle) —
            # back to user-controlled, and defaulting off this time, on all three
            # panels (main + both mini charts) sharing this one setting rather
            # than three separate checkboxes for what's really one preference.
            show_volume = st.checkbox("Show volume", value=False, key="fvg_show_volume")
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
        with _tab_rules, st.container(key="_panel_rules"):
            # Sweep IS the rule builder now — a single rule is just a sweep
            # with every range narrowed to one value (same engine, same
            # results table), so there's exactly one way to build/test a
            # rule instead of a separate hand-built-widget tool plus a
            # sweep tool sitting next to each other. Pick any result row
            # below and "Use this rule on the chart" to drive Entry/Stop/
            # Target and the Backtest panel underneath — merged in per
            # direct request. backtest_ui.py owns the actual render logic,
            # shared with the standalone Backtest page (:8507) and
            # Markets' own Rules tab — same backtest_results.db underneath
            # regardless of which of the three called it. Footprint
            # included here (not in Markets) since it's crypto-only by
            # nature — Yahoo-only tickers (all Markets ever shows) can't
            # do this regardless.
            st.caption(
                "Build a rule as either 'current price' or the Nth-nearest FVG, Order Block, or Liquidity "
                "level above or below it. Test one exact combo (narrow every range below to a single value) "
                "or a whole grid of variations at once — then pick any result to drive the chart's Entry/"
                "Stop/Target and see its full trade history in the Backtest panel below."
            )
            _sweep_tab, _heatmap_tab, _footprint_tab = st.tabs(["Sweep", "Heatmap", "Footprint"])
            with _sweep_tab:
                backtest_ui.render_sweep_tab(TICKER_INFO, has_chart=True)
            with _heatmap_tab:
                backtest_ui.render_heatmap_tab()
            with _footprint_tab:
                backtest_ui.render_footprint_tab(TICKER_INFO)

            st.markdown("---")

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
                    st.caption("Select a result in Sweep above and click \"Use this rule on the chart\" — "
                               "nothing active yet.")
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
                                                               exit_types=_win_rate_exit_types)
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
                        # anywhere in Layers/Settings/Rules/Charts (one
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
            "show_ma": show_indicators, "ma_tf": indicator_tf, "show_volume_profile": show_volume_profile,
            "zone_opacity": zone_opacity, "data_source": data_source,
            "overlay_tf": overlay_tf, "win_rate_exit_types": _win_rate_exit_types,
            "confluence_keys": _confluence_keys,
            "entry_rule": _entry_rule, "exit_rule": _exit_rule, "stop_rule": _stop_rule,
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
        # before (the icon-row-driven Layers/Settings/Rules/Charts cycle
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
        # main_col's own top-right corner (a plain st.container(key=...)
        # doesn't reserve any layout space once its CSS position is
        # "absolute" — it's pulled clean out of the normal document flow,
        # floating over whatever would otherwise render there, confirmed
        # directly this leaves neither a horizontal gap beside it nor a
        # reserved column beneath it the way a dedicated st.columns()
        # split did before). main_col itself needs `position: relative`
        # so "top: 0; right: 0" resolves against ITS OWN box (the whole
        # chart column) rather than the page/viewport.
        #
        # st.popover's own default look is a small card anchored right
        # under its trigger button — per earlier direct request, restyled
        # into an actual right-edge drawer instead: full viewport height,
        # pinned to the right edge, sized to what its content actually
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
                right: 0 !important;
                z-index: 100 !important;
                width: auto !important;
            }
            div[data-testid="stPopoverBody"][aria-label="☰"] {
                position: fixed !important;
                inset: 0 0 0 auto !important;
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
                border-left: 1px solid rgba(255,255,255,0.15) !important;
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
            /* The Rules tab nests its OWN Sweep/Heatmap/Footprint
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
               RIGHT-pinned panel: native resize assumes a fixed top-left
               anchor with a free bottom-right corner the handle drags to a
               new position, but this panel is the opposite (right edge
               pinned, left edge free) — the handle itself sits glued to
               the screen's own right edge no matter how far you drag,
               since resizing never actually moves it, so the cursor and
               the handle immediately disconnect. This is a real element
               instead: a full-height strip at the panel's own LEFT edge
               (where the moving edge actually is), dragged via JS (see
               the st.html script below — plain st.markdown can't run
               <script>, confirmed directly; unsafe_allow_javascript=True
               is what actually executes it).
               ::after is the visible grip bar — a 4px accent line that
               brightens on hover. Sits ENTIRELY outside the panel's own
               padding box now (left:-14px, width:14px — was -5px/10px) —
               confirmed directly the old, narrower offset let the grip
               bar's own paint area overlap the first few pixels of real
               content (checkbox/label text right at the drawer's left
               edge); fully clearing the padding needs more than half the
               old hit area's width of clearance. */
            .ict-drawer-resize-handle {
                /* height:100vh instead of top:0;bottom:0 — confirmed
                   directly that bottom:0 resolves against the nearest
                   intermediate wrapper Streamlit puts around this
                   st.markdown call (itself height:auto, i.e. no definite
                   height), not the drawer's own fixed/100vh box, and
                   collapses to 0 rather than stretching. A fixed 100vh
                   sidesteps needing that containing-block chain to
                   resolve correctly at all. */
                position: absolute; left: -14px; top: 0; height: 100vh; width: 14px;
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
                    const newWidth = window.innerWidth - e.clientX;
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
            with st.popover("☰", help="Layers, Settings, Rules & Charts"):
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
    show_rsi = _cc.get("show_rsi", False)
    show_macd = _cc.get("show_macd", False)
    show_bb = _cc.get("show_bb", False)
    show_indicators = _cc.get("show_ma", True)
    indicator_tf = _cc.get("ma_tf", "Chart TF")
    show_volume_profile = _cc.get("show_volume_profile", False)
    zone_opacity = _cc.get("zone_opacity", 0.25)
    data_source = _cc.get("data_source", "auto")
    overlay_tf = _cc.get("overlay_tf", "4h")
    _win_rate_exit_types = _cc.get("win_rate_exit_types", tuple(EXIT_EVENT_LABELS.keys()))
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
            # so without this re-read, a Layers/Settings/Rules change
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
            show_rsi = _cc.get("show_rsi", False)
            show_macd = _cc.get("show_macd", False)
            show_bb = _cc.get("show_bb", False)
            show_indicators = _cc.get("show_ma", True)
            indicator_tf = _cc.get("ma_tf", "Chart TF")
            show_volume_profile = _cc.get("show_volume_profile", False)
            zone_opacity = _cc.get("zone_opacity", 0.25)
            data_source = _cc.get("data_source", "auto")
            overlay_tf = _cc.get("overlay_tf", "4h")
            _win_rate_exit_types = _cc.get("win_rate_exit_types", tuple(EXIT_EVENT_LABELS.keys()))
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
            FUTURE_EXTEND_CANDLES = 3
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
            # own latest close. Used below for the Entry/SL/TP pick (either
            # flavor) and again further down for every layer's own
            # above/below-price filtering, regardless of which timeframe
            # any given layer is independently set to detect on — there's
            # only one real current price, not one per timeframe.
            current_price = float(df[c_col].iloc[-1])
            # A direct lookup, independent of the Rules tab's own entry/
            # exit/stop rule below — a real live MA+FVG overlap should
            # show up here regardless of whatever rule happens to be
            # configured. The border highlight means "an EMA sits inside
            # this zone right now," not "this rule is pointed at it" —
            # see ma_fvg_starts's own docstring.
            _ma_fvg_starts = ma_fvg_starts(df)

            # Resolved unconditionally — the drawing loop below always
            # needs these now. (The Rules tab's own R:R
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
            _entry_price = resolve_rule(_entry_rule, df, current_price)
            _exit_price = resolve_rule(_exit_rule, df, current_price)
            _stop_price = resolve_rule(_stop_rule, df, current_price)

            # Entry/SL/TP always come from the Rules tab's own rule now —
            # the old Automatic mode (trusting best_trade_now's own
            # validation/confluence ranking for these same three lines) and
            # the Off toggle (hiding them) are both gone per direct request.
            _entry_x = _ny_fake_utc_seconds(df.index[-1])
            for _price, _color, _title in [
                (_entry_price, theme.NEON_AMBER, "Entry"),
                (_stop_price, theme.NEON_MAGENTA, "SL"),
                (_exit_price, theme.NEON_GREEN, "TP"),
            ]:
                if _price is not None:
                    price_lines.append({"t0": _entry_x, "t1": future_edge, "price": _price,
                                         "color": _hex_to_rgba(_color, 1.0), "title": _title,
                                         "line_width": 3, "dashed": False,
                                         "above": _price >= (_entry_price if _entry_price is not None else _price)})
            if _entry_price is not None and _stop_price is not None:
                # The risk itself, not just its two edges — a translucent
                # band between Entry and SL, fading in from the entry
                # candle to solid red at the live edge, same pairing and
                # same "builds toward now" language as the lines
                # themselves. See feedback_ui_obviousness.
                rectangles.append({"t0": _entry_x, "t1": future_edge, "p0": _entry_price, "p1": _stop_price,
                                    "fill": _hex_to_rgba(theme.NEON_MAGENTA, 0.02),
                                    "fill_to": _hex_to_rgba(theme.NEON_MAGENTA, 0.16),
                                    "extend_to_edge": True})
            if _entry_price is not None and _exit_price is not None:
                # The reward's own zone, mirroring the risk zone above —
                # without this TP's line alone reads as less
                # highlighted than SL's, even though both get the
                # identical glow/gradient treatment, since SL had a
                # whole tinted area behind it and TP didn't. See
                # feedback_ui_obviousness.
                rectangles.append({"t0": _entry_x, "t1": future_edge, "p0": _entry_price, "p1": _exit_price,
                                    "fill": _hex_to_rgba(theme.NEON_GREEN, 0.02),
                                    "fill_to": _hex_to_rgba(theme.NEON_GREEN, 0.16),
                                    "extend_to_edge": True})

            # Every trade from the Rules tab's own "Run backtest" run, each
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

            # The MA+FVG strategy's own two EMAs, drawn as real lines on
            # the main chart (not just an inferred sidebar label) — see
            # feedback_ui_obviousness: a signal a human needs to notice
            # should be visible at the thing it's about, not just named in
            # a list. Uses the SAME periods recommender.py checks for
            # overlap (MA_FVG_PERIODS), so the lines on screen are always
            # exactly what the strategy is actually reading.
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
                    for _period in MA_FVG_PERIODS:
                        _chart_indicators[f"ma{_period}"] = _series_to_points(ema(_close_col, _period))

            # RSI/MACD/Bollinger Bands — plain technical indicators, always
            # read off the main chart's own df at its own timeframe (no
            # separate TF picker; see the Settings checkboxes for why).
            if (show_rsi or show_macd or show_bb) and not df.empty:
                _ti_close = df["Close"] if "Close" in df else df["close"]
                if show_rsi:
                    _chart_indicators["rsi"] = _series_to_points(rsi(_ti_close))
                if show_macd:
                    _macd_line, _signal_line, _hist = macd(_ti_close)
                    _chart_indicators["macd"] = {
                        "macd": _series_to_points(_macd_line),
                        "signal": _series_to_points(_signal_line),
                        "histogram": _hist_to_points(_hist, theme.NEON_GREEN, theme.NEON_MAGENTA),
                    }
                if show_bb:
                    _bb_upper, _bb_basis, _bb_lower = bollinger_bands(_ti_close)
                    _chart_indicators["bb"] = {
                        "upper": _series_to_points(_bb_upper),
                        "basis": _series_to_points(_bb_basis),
                        "lower": _series_to_points(_bb_lower),
                    }

            # Volume Profile — how much volume traded at each PRICE level
            # over this chart's own currently-loaded history (see
            # indicators.volume_profile's own docstring for the OHLC-based
            # approximation this uses, and why it honestly returns None for
            # Yahoo Forex, which never reports real volume). Rendered as a
            # dedicated overlay (bars anchored to the pane's own left edge,
            # via a real ict_chart primitive), not through the indicators=
            # pathway above — a price-bucketed histogram isn't a time
            # series, so it doesn't fit that shape.
            volume_profile_buckets = []
            if show_volume_profile and not df.empty:
                _vp = volume_profile(df)
                if _vp is not None:
                    for _b in _vp["buckets"]:
                        _is_poc = _b["price_low"] <= _vp["poc_price"] <= _b["price_high"]
                        volume_profile_buckets.append({
                            "price_low": _b["price_low"], "price_high": _b["price_high"],
                            "volume_frac": _b["volume_frac"],
                            "color": _hex_to_rgba(theme.NEON_CYAN, 0.55) if _is_poc
                                     else _hex_to_rgba("#8e8e93", 0.30),
                        })
                    price_lines.append({"t0": axis_secs[0], "t1": future_edge, "price": _vp["poc_price"],
                                         "color": _hex_to_rgba(theme.NEON_CYAN, 0.9), "title": "POC", "above": True})
                    price_lines.append({"t0": axis_secs[0], "t1": future_edge, "price": _vp["value_area_high"],
                                         "color": _hex_to_rgba(theme.NEON_AMBER, 0.7), "title": "VAH", "above": True})
                    price_lines.append({"t0": axis_secs[0], "t1": future_edge, "price": _vp["value_area_low"],
                                         "color": _hex_to_rgba(theme.NEON_AMBER, 0.7), "title": "VAL", "above": False})

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

            if "FVG" in layers:
                rf = tf_frames.get(layer_tf["FVG"])
                if rf is not None:
                    all_fvgs = detect_fvgs(rf["confirmed"])
                    fvgs = all_fvgs if show_mitigated else [g for g in all_fvgs if not g["filled"]]
                    fvgs = _nearest_by_price(fvgs, current_price, max_items_per_layer)
                    win_rates = _historical_win_rates(ticker, layer_tf["FVG"], "FVG", data_source,
                                                       exit_types=_win_rate_exit_types)
                    # A separate lookup, not per-zone — MA+FVG's own win
                    # rate (touch anchored to the EMA's level, not "touched
                    # anywhere in the gap") is a different, more specific
                    # stat than plain FVG's, shown INSTEAD of it for zones
                    # that are also MA+FVG matches.
                    ma_fvg_win_rates = _historical_win_rates(ticker, layer_tf["FVG"], "MA+FVG", data_source,
                                                              exit_types=_win_rate_exit_types)
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
                                "fill": _hex_to_rgba(color, zone_opacity), "border": color,
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
                    open_n = sum(1 for g in fvgs if not g["filled"])
                    legend_items.append((f"FVG ({layer_tf['FVG']})", f"{len(fvgs)} shown · {open_n} open", theme.NEON_CYAN))

            if "Order Blocks" in layers:
                rf = tf_frames.get(layer_tf["Order Blocks"])
                if rf is not None:
                    all_obs = detect_order_blocks(rf["confirmed"])
                    obs = all_obs if show_mitigated else [o for o in all_obs if not o["mitigated"]]
                    obs = _nearest_by_price(obs, current_price, max_items_per_layer)
                    win_rates = _historical_win_rates(ticker, layer_tf["Order Blocks"], "Order Block", data_source,
                                                       exit_types=_win_rate_exit_types)
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
                        rectangles.append({
                            "t0": _ny_fake_utc_seconds(ob["start"]), "t1": rf["t1_axis"](ob["end"]),
                            "p0": ob["bottom"], "p1": ob["top"],
                            "fill": _hex_to_rgba(color, opacity), "border": None if ob["mitigated"] else color,
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
                    unmit = sum(1 for o in obs if not o["mitigated"])
                    legend_items.append((f"Order Blocks ({layer_tf['Order Blocks']})", f"{len(obs)} shown · {unmit} unmit.", theme.NEON_GREEN))

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
                    highs, lows = detect_swings(rf["confirmed"])
                    highs = highs[-max_items_per_layer:]
                    lows = lows[-max_items_per_layer:]
                    hi_color = _hex_to_rgba(theme.NEON_MAGENTA, 1.0)
                    lo_color = _hex_to_rgba(theme.NEON_GREEN, 1.0)
                    for p in highs:
                        markers.append({"time": _ny_fake_utc_seconds(p["time"]), "position": "aboveBar",
                                         "color": hi_color, "shape": "arrowDown"})
                    for p in lows:
                        markers.append({"time": _ny_fake_utc_seconds(p["time"]), "position": "belowBar",
                                         "color": lo_color, "shape": "arrowUp"})
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
                    highs, lows = detect_swings(rf["confirmed"])
                    eq_count = 0
                    for group, color, label, direction in [(detect_equal_levels(highs), theme.NEON_MAGENTA, "EQH", "above"),
                                                             (detect_equal_levels(lows), theme.NEON_GREEN, "EQL", "below")]:
                        recent = sorted(group, key=lambda pts: max(p["time"] for p in pts))[-max_items_per_layer:]
                        for pts in recent:
                            times = sorted(p["time"] for p in pts)
                            price = sum(p["price"] for p in pts) / len(pts)
                            t1 = _line_stop_time(rf["confirmed"], times[0], price, direction=direction, exclude=set(times))
                            price_lines.append({"t0": _ny_fake_utc_seconds(times[0]), "t1": rf["t1_axis"](t1), "price": price,
                                                 "color": _hex_to_rgba(color, 1.0),
                                                 "title": f"{label} ({layer_tf['Equal Highs/Lows']})",
                                                 "above": direction == "above"})
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
                    breaks = detect_structure_breaks(rf["confirmed"])
                    breaks = breaks[-max_items_per_layer:]
                    bos_n = choch_n = 0
                    for b in breaks:
                        is_choch = b["structure"] == "CHoCH"
                        color = theme.NEON_AMBER if is_choch else (theme.NEON_GREEN if b["type"] == "bullish" else theme.NEON_MAGENTA)
                        price_lines.append({"t0": _ny_fake_utc_seconds(b["start"]), "t1": _ny_fake_utc_seconds(b["end"]),
                                             "price": b["level"], "color": _hex_to_rgba(color, 1.0),
                                             "title": f"{b['structure']} ({layer_tf['Market Structure']})",
                                             "above": b["type"] == "bullish"})
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
                    dr = current_dealing_range(rf["confirmed"])
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
                                                             n_below=max_items_per_layer)
                    # A swept HIGH (BSL) points bearish, a swept LOW (SSL)
                    # points bullish — same direction convention
                    # detect_liquidity_sweeps' own `type` field already
                    # uses (see liquidity_event_win_rate), so BSL reads the
                    # "bearish" side of this lookup and SSL the "bullish".
                    liq_win_rates = _historical_win_rates(ticker, layer_tf["Liquidity"], "Liquidity", data_source,
                                                           exit_types=_win_rate_exit_types)
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
                        price_lines.append({"t0": _ny_fake_utc_seconds(lvl["time"]), "t1": rf["t1_axis"](t1),
                                             "price": lvl["price"], "color": _hex_to_rgba(theme.NEON_MAGENTA, 1.0),
                                             "title": title, "line_width": 2 if matched_inds else 1, "above": True})
                        liquidity_rows.append({"tf": layer_tf["Liquidity"], "kind": "BSL (buy-side)", "price": lvl["price"], "formed": lvl["time"]})
                    for lvl in below:
                        t1 = _line_stop_time(rf["confirmed"], lvl["time"], lvl["price"])
                        title = f"SSL ({layer_tf['Liquidity']})" + (f" · {ssl_label}" if ssl_label else "")
                        matched_inds = below_confluence.get(lvl["time"], [])
                        if matched_inds:
                            title += " · " + "+".join(INDICATOR_SPECS[k]["label"] for k in matched_inds)
                        price_lines.append({"t0": _ny_fake_utc_seconds(lvl["time"]), "t1": rf["t1_axis"](t1),
                                             "price": lvl["price"], "color": _hex_to_rgba(theme.NEON_AMBER, 1.0),
                                             "title": title, "line_width": 2 if matched_inds else 1, "above": False})
                        liquidity_rows.append({"tf": layer_tf["Liquidity"], "kind": "SSL (sell-side)", "price": lvl["price"], "formed": lvl["time"]})

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
                    reactions = detect_liquidity_reactions(rf["confirmed"])
                    reactions = reactions[-max_items_per_layer:]
                    for r in reactions:
                        color = theme.NEON_MAGENTA if r["type"] == "bearish" else theme.NEON_AMBER
                        rectangles.append({
                            "t0": _ny_fake_utc_seconds(r["start"]), "t1": rf["t1_axis"](r["end"]),
                            "p0": r["bottom"], "p1": r["top"],
                            "fill": _hex_to_rgba(color, 0.22), "border": _hex_to_rgba(color, 1.0),
                        })

                    legend_items.append((f"Liquidity ({layer_tf['Liquidity']})",
                                          f"{len(above)}BSL {len(below)}SSL · {len(reactions)} reaction(s)", theme.NEON_MAGENTA))

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

            clicked = ict_chart(
                bars_payload, fingerprint,
                overlays={"rectangles": rectangles, "price_lines": price_lines,
                          # lightweight-charts' series-markers plugin requires
                          # markers pre-sorted ascending by time — Swing
                          # Points appends bullish/bearish in two separate
                          # loops, so the raw list isn't globally sorted.
                          # Confirmed directly: passing it unsorted is why
                          # bearish (arrowDown) markers were disappearing at
                          # some zoom levels.
                          "markers": sorted(markers, key=lambda m: m["time"]),
                          "ghost_candles": ghost_candles, "volume_profile": volume_profile_buckets},
                options={"log_scale": False, "volume": show_volume,
                          "selected": st.session_state.get("selected_chart") == "main",
                          # A touch narrower than the mini panels' candles —
                          # lightweight-charts derives candle width straight
                          # from barSpacing with no separate thickness knob,
                          # see candleWidthFactor in the frontend.
                          "candle_width_factor": 0.85},
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
                st.session_state.pop("_main_last_sent_fp", None)
                st.rerun(scope="fragment")
            _select_chart("main", clicked)

            if legend_items:
                with st.expander("Legend"):
                    fvg_legend(legend_items)
                    tiny("🟢/🔵 bullish · 🔴/🟠 bearish · scroll to zoom, drag to pan, "
                         "each layer detects on its own chosen timeframe (☰ to change) and shows only its most "
                         "recent items · Premium/Discount boxes the current swing high/low dealing range and "
                         "stays put until price closes through one side · open FVG/OB zones are labeled with "
                         "this ticker's own historical win rate for that direction (hover the detail table "
                         "below for what that number does and doesn't mean)")

            # The legend above is a tally; this is the actual data behind it
            # — every FVG/Order Block zone currently drawn on the chart, not
            # a re-detection, the exact same fvg_ob_rows list the rectangles
            # were built from a few hundred lines up.
            #
            # Sorted open/unmitigated first, then by distance from the
            # current close — a zone price has already fully traded through
            # isn't "tradeable in the future" the way a still-live zone
            # sitting just above/below current price is, and the ones
            # dozens of percent away are academic compared to the ones
            # price could reach this session. This is a relevance ranking
            # for YOUR attention, not a claim any specific zone will hold.
            #
            # hist_win_rate closes part of that gap — a real number computed
            # off this ticker's own history (_historical_win_rates above),
            # not a guess — but it's still a plain raw-return win rate, NOT
            # the permutation-test-plus-multiple-testing-correction rigor
            # Research Lab and Edge Lab apply before calling anything an
            # actual edge. Treat it as "how has this pattern's direction
            # tended to resolve historically," not statistical proof.
            if fvg_ob_rows:
                current_price = float(df[c_col].iloc[-1])
                with st.expander(f"🔍 FVG / Order Block detail ({len(fvg_ob_rows)} zones, sorted by relevance)"):
                    detail_df = pd.DataFrame(fvg_ob_rows)
                    detail_df["is_open"] = detail_df["status"].isin(["open", "unmitigated"])
                    detail_df["dist_pct"] = (((detail_df["top"] + detail_df["bottom"]) / 2 - current_price)
                                              / current_price * 100)
                    detail_df = detail_df.sort_values(["is_open", "dist_pct"], key=lambda s: s if s.name == "is_open" else s.abs(),
                                                        ascending=[False, True]).drop(columns="is_open").reset_index(drop=True)
                    st.dataframe(detail_df, hide_index=True, use_container_width=True, column_config={
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
                        "top": st.column_config.NumberColumn(help="Zone's upper price boundary, as currently "
                                                                    "drawn — shrinks over time as price eats into "
                                                                    "the zone (consequent encroachment)."),
                        "bottom": st.column_config.NumberColumn(help="Zone's lower price boundary, same "
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

            # BSL/SSL only ever rendered as unlabeled dashed lines on the
            # chart, with a hover title as the only way to see which is
            # which — no exact price, no distance, no way to tell which
            # ones are closest to actually mattering right now without
            # reading raw chart data. Same fix as the FVG/OB table above,
            # same "sort by what's actually relevant" logic.
            if liquidity_rows:
                current_price = float(df[c_col].iloc[-1])
                with st.expander(f"🔍 Liquidity detail ({len(liquidity_rows)} levels, sorted by distance)"):
                    # A BSL level can legitimately show a NEGATIVE distance
                    # (sitting below the live price) — detection only ever
                    # checks "unswept as of the last CLOSED bar," but this
                    # table's distance is measured against the live,
                    # currently-forming candle. Confirmed directly: not a
                    # bug, just means price has already wicked through that
                    # level intrabar, on a candle that hasn't closed yet —
                    # once it does, that level will very likely flip to
                    # swept on the next rerun.
                    tiny("A BSL below price (or SSL above it) means the live candle has already wicked through "
                         "that level — detection only confirms 'swept' once the bar actually closes.")
                    liq_df = pd.DataFrame(liquidity_rows)
                    liq_df["dist_pct"] = (liq_df["price"] - current_price) / current_price * 100
                    liq_df = liq_df.sort_values("dist_pct", key=lambda s: s.abs()).reset_index(drop=True)
                    st.dataframe(liq_df, hide_index=True, use_container_width=True, column_config={
                        "tf": st.column_config.Column(help="Reference timeframe this level was detected on."),
                        "kind": st.column_config.Column(
                            help="BSL (buy-side liquidity) rests above a swing high — stops from shorts, a magnet "
                                 "for a bullish run. SSL (sell-side) rests below a swing low — stops from longs, "
                                 "a magnet for a bearish run. Both shown here are still LIVE — detect_liquidity_levels "
                                 "only ever returns levels no later candle has swept yet."),
                        "price": st.column_config.NumberColumn(help="Exact price of the resting level."),
                        "formed": st.column_config.Column(help="The swing bar that created this level."),
                        "dist_pct": st.column_config.NumberColumn(
                            "dist_pct", format="%+.2f%%",
                            help=f"Distance from the current price ({current_price:.5g}). Sorted closest first — "
                                 "the nearest levels are what price could realistically reach next."),
                    })
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
            _prefetch_specs.append((_historical_win_rates, (ticker, layer_tf[_wr_layer], _wr_layer, data_source), {}))
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

    _render_chart()
