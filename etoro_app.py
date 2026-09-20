"""
eToro — a SCAFFOLD for validating an ICT trading edge on eToro's demo
account. The actual edge/signal logic is still being worked out and
validated live against eToro's demo account in a SEPARATE Claude Desktop
session (not this dashboard, not this repo checkout) — this page exists
so the terminal shell (chart, ICT detector overlays, the on-chart "best
trade right now" panel, the "Validated edge" sidebar scanner, an eToro
account panel) is already standing and ready to wire up the moment that
edge logic lands here. ETORO_CURATED_UNIVERSE below and the best-trade-
now heuristics currently shown are PROVISIONAL placeholders carried over
from app.py/crypto_app.py's own confluence-only ranking, not the actual
validated edge — nothing on this page should be read as eToro-specific
trading advice yet.

Two independent data flows, easy to conflate, kept deliberately separate:

  1. Chart candles (OHLCV, for drawing candles/ICT structure) come from
     Yahoo Finance via data.py — the exact same provider chain app.py/
     crypto_app.py already use, entirely independent of eToro itself.
     This is what makes the chart work at all; it has nothing to do with
     the eToro account below.
  2. Live eToro account/trading data flows through Claude's OWN eToro MCP
     connection, used in conversation by a Claude Code/Desktop session
     that holds real API credentials this deployed Streamlit process does
     NOT have. That session snapshots the account to
     .cache/etoro_snapshot.json; this page only ever reads that file —
     see _load_etoro_snapshot's own docstring for why it must stay that
     way. This process makes NO live HTTP request of its own to any
     etoro.com host.

No order-placement/trade-execution UI exists here — there is no validated
edge yet to trigger one. See the comment near best_trade_now's call site
below for where that will eventually plug in.

Independent of market-dashboard and of app.py/crypto_app.py's own session
state — see README.md.
"""

import concurrent.futures
import json
import math
import os
import threading
from datetime import time as dtime

import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

import experiments
import theme
from data import get_latest_bars, get_yf_ohlcv, is_ticker_alive, resample_ohlc
from detectors import (
    current_dealing_range,
    detect_fvgs,
    detect_liquidity_levels,
    detect_order_blocks,
    detect_structure_breaks,
)
from ict_chart import ict_chart
from recommender import EVENT_TYPE_LABELS, best_trade_now

st.set_page_config(page_title="eToro", layout="wide", initial_sidebar_state="collapsed")
theme.inject(chart_layout=True)


def tiny(text):
    """Genuinely optional text — small and muted, same convention as
    app.py/crypto_app.py's own identical helper (theme.py's own
    .tiny-note CSS class, defined once, reused by every page)."""
    st.markdown(f'<div class="tiny-note">{text}</div>', unsafe_allow_html=True)


def _hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _price_decimals(price):
    """How many decimals a price needs to stay meaningful, based on its own
    magnitude — this page's universe spans a $1.08 forex pair, a $4,500
    index, and an $80,000 BTC print, so a fixed decimal count would either
    round the forex pair's real movement away or show a fraction-of-a-cent
    stock as "0.00". Same formula as app.py/crypto_app.py's own copy (kept
    identical, not imported — see this project's own README.md on why the
    three pages don't import each other)."""
    if price is None or price <= 0:
        return 2
    decimals = 5 - math.floor(math.log10(price))
    return max(2, min(10, decimals))


_EPOCH = pd.Timestamp("1970-01-01")


def _ny_fake_utc_seconds(ts):
    """lightweight-charts has no timezone setting of its own — encode the
    display timezone's own wall-clock digits as if they were UTC (the
    standard workaround; see ict_chart's own docstring). Same convention
    as app.py/crypto_app.py's identically-named helper."""
    disp = ts.tz_convert(theme.get_display_tz()).tz_localize(None)
    return int((disp - _EPOCH).total_seconds())


def _ny_fake_utc_seconds_vec(idx):
    """Vectorized form of _ny_fake_utc_seconds — see app.py/crypto_app.py's
    own copy for why this matters at real bar counts (a scalar per-
    timestamp loop pays pandas/pytz's own per-call DST-lookup cost on
    every bar; this does the identical output ~100x+ faster)."""
    disp_idx = idx.tz_convert(theme.get_display_tz()).tz_localize(None)
    return ((disp_idx - _EPOCH) // pd.Timedelta(seconds=1)).tolist()


def _nearest_by_price(zones, current_price, n, top_key="top", bottom_key="bottom"):
    """The `n` zones closest to current_price on each side (above AND
    below) — "active" as in immediately relevant to where price actually
    is, not just recently detected. Same as app.py/crypto_app.py's own
    identical helper; see there for the full reasoning."""
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


def _prefetch(specs):
    """Fires several get_yf_ohlcv/is_ticker_alive calls in parallel instead
    of one after another — same helper, same reasoning, as app.py/
    crypto_app.py's own _prefetch (see there): each real call site below
    just hits an already-warm st.cache_data entry afterward."""
    if len(specs) < 2:
        for fn, args, kwargs in specs:
            try:
                fn(*args, **kwargs)
            except Exception:
                pass
        return
    ctx = get_script_run_ctx()

    def _run(fn, args, kwargs):
        add_script_run_ctx(threading.current_thread(), ctx)
        return fn(*args, **kwargs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(specs)) as pool:
        futures = [pool.submit(_run, fn, args, kwargs) for fn, args, kwargs in specs]
        concurrent.futures.wait(futures)


# ---------------------------------------------------------------------------
# Instrument universe. eToro's own real catalog spans forex, indices, index
# futures/CFDs, commodities, stocks, AND crypto all on one platform — unlike
# app.py (Forex/Indices/Commodities only) or crypto_app.py (crypto only),
# this page needs all of them. Forex/Indices/Index Futures/Commodities below
# are the EXACT same symbol/name/category values as app.py's own
# CURATED_FOREX/CURATED_INDICES/CURATED_INDEX_FUTURES/CURATED_COMMODITIES
# (copied, not imported — see this module's own docstring on why the pages
# don't import each other) — those are already confirmed-live Yahoo tickers,
# no reason to re-derive or re-verify them here.
#
# PROVISIONAL / STARTER LIST — do not overthink this. Nobody yet knows which
# specific instruments the eToro-side validated edge will actually target
# (that work is happening in a separate session against the live eToro
# catalog, not here); this is a reasonable, broad starting universe to chart
# against while that's worked out, meant to be narrowed to whatever the edge
# actually trades — or extended, if it turns out to need something not
# listed here — once that's known. Not an attempt to mirror eToro's full
# instrument catalog.
ETORO_INDICES = {
    "^GSPC": ("S&P 500", "Indices"), "^DJI": ("Dow Jones Industrial Average", "Indices"),
    "^IXIC": ("Nasdaq Composite", "Indices"), "^RUT": ("Russell 2000", "Indices"),
    "^VIX": ("CBOE Volatility Index", "Indices"), "^FTSE": ("FTSE 100", "Indices"),
    "^N225": ("Nikkei 225", "Indices"), "^GDAXI": ("DAX", "Indices"),
}
ETORO_COMMODITIES = {
    "GC=F": ("Gold Futures", "Commodities"), "SI=F": ("Silver Futures", "Commodities"),
    "CL=F": ("Crude Oil Futures", "Commodities"),
}
ETORO_INDEX_FUTURES = {
    "NQ=F": ("Nasdaq-100 Futures", "Futures"), "ES=F": ("S&P 500 Futures", "Futures"),
    "YM=F": ("Dow Jones Futures", "Futures"), "RTY=F": ("Russell 2000 Futures", "Futures"),
}
ETORO_FOREX = {
    "EURUSD=X": ("Euro / US Dollar", "Forex"), "GBPUSD=X": ("British Pound / US Dollar", "Forex"),
    "USDJPY=X": ("US Dollar / Japanese Yen", "Forex"), "USDCHF=X": ("US Dollar / Swiss Franc", "Forex"),
    "AUDUSD=X": ("Australian Dollar / US Dollar", "Forex"), "USDCAD=X": ("US Dollar / Canadian Dollar", "Forex"),
    "NZDUSD=X": ("New Zealand Dollar / US Dollar", "Forex"), "EURJPY=X": ("Euro / Japanese Yen", "Forex"),
    "GBPJPY=X": ("British Pound / Japanese Yen", "Forex"), "EURGBP=X": ("Euro / British Pound", "Forex"),
}
# A modest starter set of large-cap stocks — standard Yahoo tickers, no
# suffix needed. Not curated with the same "confirmed live" rigor as the
# Forex/Indices/Futures/Commodities dicts above (those were hand-verified
# against Yahoo's own ticker conventions before being added to app.py); this
# is just "the obvious mega-caps," a placeholder set, same reasoning as the
# whole universe's own "provisional" comment above.
ETORO_STOCKS = {
    "AAPL": ("Apple", "Stocks"), "MSFT": ("Microsoft", "Stocks"), "GOOGL": ("Alphabet", "Stocks"),
    "AMZN": ("Amazon", "Stocks"), "NVDA": ("NVIDIA", "Stocks"), "TSLA": ("Tesla", "Stocks"),
    "META": ("Meta Platforms", "Stocks"),
}
# A handful of top-cap coins, Yahoo's own -USD convention (same one
# crypto_app.py's live Binance-sourced universe uses) — not that page's full
# live-fetched hundreds-of-pairs list, just the obvious majors, same
# "starter, not exhaustive" reasoning as ETORO_STOCKS above.
ETORO_CRYPTO = {
    "BTC-USD": ("Bitcoin", "Crypto"), "ETH-USD": ("Ethereum", "Crypto"),
    "SOL-USD": ("Solana", "Crypto"), "XRP-USD": ("XRP", "Crypto"),
}


@st.cache_data(ttl=86400)
def _build_ticker_info():
    info = dict(ETORO_INDICES)
    info.update(ETORO_COMMODITIES)
    info.update(ETORO_INDEX_FUTURES)
    info.update(ETORO_FOREX)
    info.update(ETORO_STOCKS)
    info.update(ETORO_CRYPTO)
    return info


TICKER_INFO = _build_ticker_info()
TICKER_NAMES = {t: info[0] for t, info in TICKER_INFO.items()}
TICKER_UNIVERSE = list(TICKER_INFO.keys())

# The ticker menu's top-level category buttons — order here is the order the
# buttons render in. Same {name: [tags]} shape as app.py's own
# TICKER_CATEGORIES (a list of tags per button, not just one, in case a
# button ever needs to span more than one category — none currently do).
TICKER_CATEGORIES = {
    "Forex": ["Forex"], "Indices": ["Indices"], "Futures": ["Futures"], "Commodities": ["Commodities"],
    "Stocks": ["Stocks"], "Crypto": ["Crypto"],
}

# Includes Binance (unlike app.py's own copy of this dict, which omits it —
# that page never charts a -USD ticker) since this universe includes crypto,
# same as crypto_app.py's own copy.
DATA_SOURCES = {"Auto": "auto", "Yahoo Finance": "yahoo", "Binance": "binance",
                 "Twelve Data": "twelvedata", "Tiingo": "tiingo"}
DATA_SOURCE_LABELS = {v: k for k, v in DATA_SOURCES.items()}


def _is_open(ticker):
    """24/7 for crypto; NYSE/Nasdaq hours (Mon-Fri 9:30-16:00 ET) for
    everything else — a simplification (no holiday calendar, and forex/
    futures genuinely trade near-24/5, not just NYSE hours), but enough to
    flag "don't bother, it's closed" at a glance. Identical simplification
    to app.py's own copy, same reasoning."""
    if ticker.endswith("-USD"):
        return True
    now = pd.Timestamp.now(tz="America/New_York")
    return now.weekday() < 5 and dtime(9, 30) <= now.time() < dtime(16, 0)


def _ticker_label(t):
    name = TICKER_NAMES.get(t)
    label = f"{t} — {name}" if name else t
    return label if _is_open(t) else f"{label}  ·  closed"


# Core interval bar — identical to app.py/crypto_app.py's own TIMEFRAMES
# (same fetch_interval/period/resample math; see either page's own comment
# on why 4h/1Y are resampled rather than native, and why 1h stays at 180d
# rather than sharing 4h's 730d fetch un-resampled).
TIMEFRAMES = {
    "1m":  {"fetch_interval": "1m",  "period": "7d",   "resample": None},
    "5m":  {"fetch_interval": "5m",  "period": "60d",  "resample": None},
    "15m": {"fetch_interval": "15m", "period": "60d",  "resample": None},
    "30m": {"fetch_interval": "30m", "period": "60d",  "resample": None},
    "1h":  {"fetch_interval": "60m", "period": "180d", "resample": None},
    "4h":  {"fetch_interval": "60m", "period": "730d", "resample": "4h"},
    "1D":  {"fetch_interval": "1d",  "period": "2y",   "resample": None},
    "1W":  {"fetch_interval": "1wk", "period": "10y",  "resample": None},
    "1M":  {"fetch_interval": "1mo", "period": "max",  "resample": None},
    "1Y":  {"fetch_interval": "3mo", "period": "max",  "resample": "1YE"},
}
# Real seconds per candle — used to size the Entry/SL/TP box's own forward
# projection. Same values as app.py/crypto_app.py's own _TF_BAR_SECONDS.
_TF_BAR_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400,
    "1D": 86400, "1W": 604800, "1M": 2629746, "1Y": 31556952,
}
# How often the chart re-fetches/redraws on its own, matched to each
# timeframe's own candle duration — same values/reasoning as app.py/
# crypto_app.py's own CANDLE_SECONDS. Daily+ gets no auto-refresh (the
# source data itself only turns over once a day).
CANDLE_SECONDS = {
    "1m": 10, "5m": 20, "15m": 30, "30m": 45, "1h": 60, "4h": 120,
    "1D": None, "1W": None, "1M": None, "1Y": None,
}
# How far back auto-detected structure stays visible — same reasoning as
# app.py/crypto_app.py's own OVERLAY_LOOKBACK_DAYS: detection still runs
# against the FULL fetched history, only the drawn overlays get windowed.
OVERLAY_LOOKBACK_DAYS = 5

# A deliberately smaller layer set than crypto_app.py's own Layers tab (no
# IFVG/Breaker Block/Equal Highs-Lows/Naked POC/Poor High-Low, no per-layer
# independent timeframe, no kill zones/indicators/volume profile) — this
# page is a scaffold for the eToro edge, not a re-implementation of every
# detector this project has. The dropped layers are untouched in
# detectors.py and available to wire back in once the actual edge needs
# them; nothing about this trimmed set removes them from the project.
ICT_LAYERS = ["FVG", "Order Blocks", "Market Structure", "Premium/Discount", "Liquidity"]
LAYER_DEFAULTS = {name: False for name in ICT_LAYERS}
# On by default even on a first load — same reasoning as crypto_app.py's
# identical default: it's the layer that answers "is this actually a good
# price to be looking at a trade at all," easy to forget even exists
# otherwise.
LAYER_DEFAULTS["Premium/Discount"] = True


def _build_layer_overlays(layers, confirmed_df, current_price, future_edge, max_items_per_side, zone_opacity):
    """Rectangles/price_lines for whichever of ICT_LAYERS are toggled on —
    same nearest-to-current-price selection app.py/crypto_app.py's own
    Layers system uses for FVG/Order Blocks/Liquidity (see _nearest_by_price),
    same single-current-range read for Premium/Discount and same "every
    break in the recency window" read for Market Structure as
    crypto_app.py's own mini HTF panel (_build_mini_overlays) uses for its
    "current state of delivery" line, just not capped to one.

    confirmed_df: the CONFIRMED slice (last, still-forming bar already
    dropped by the caller) — an in-progress candle shouldn't get to define
    a zone edge yet, same convention as every detector call elsewhere in
    this project.  Detection itself still runs on the full confirmed_df
    (a zone needs its whole forward context to know whether it's been
    filled); only the drawn RESULT gets windowed to OVERLAY_LOOKBACK_DAYS
    below, same "detect on everything, draw only what's recent" split
    app.py/crypto_app.py's own main chart uses."""
    rectangles, price_lines = [], []
    if len(confirmed_df) < 3:
        return rectangles, price_lines
    overlay_cutoff = confirmed_df.index[-1] - pd.Timedelta(days=OVERLAY_LOOKBACK_DAYS)

    if "FVG" in layers:
        open_fvgs = [g for g in detect_fvgs(confirmed_df) if not g["filled"] and g["start"] >= overlay_cutoff]
        for g in _nearest_by_price(open_fvgs, current_price, max_items_per_side):
            color = theme.NEON_CYAN if g["type"] == "bullish" else theme.NEON_MAGENTA
            rectangles.append({"t0": _ny_fake_utc_seconds(g["start"]), "t1": future_edge,
                                "p0": g["bottom"], "p1": g["top"],
                                "fill": _hex_to_rgba(color, zone_opacity), "border": color, "label": "FVG"})

    if "Order Blocks" in layers:
        open_obs = [o for o in detect_order_blocks(confirmed_df) if not o["mitigated"] and o["start"] >= overlay_cutoff]
        for ob in _nearest_by_price(open_obs, current_price, max_items_per_side):
            color = theme.NEON_GREEN if ob["type"] == "bullish" else theme.NEON_AMBER
            rectangles.append({"t0": _ny_fake_utc_seconds(ob["start"]), "t1": future_edge,
                                "p0": ob["bottom"], "p1": ob["top"],
                                "fill": _hex_to_rgba(color, zone_opacity), "border": color, "label": "OB"})

    if "Market Structure" in layers:
        for b in detect_structure_breaks(confirmed_df):
            if b["start"] < overlay_cutoff:
                continue
            is_choch = b["structure"] == "CHoCH"
            color = theme.NEON_AMBER if is_choch else (theme.NEON_GREEN if b["type"] == "bullish" else theme.NEON_MAGENTA)
            price_lines.append({"t0": _ny_fake_utc_seconds(b["start"]), "t1": future_edge,
                                 "price": b["level"], "color": _hex_to_rgba(color, 1.0),
                                 "title": f"{b['type'].capitalize()} {b['structure']}", "above": b["type"] == "bullish"})

    if "Premium/Discount" in layers:
        dr = current_dealing_range(confirmed_df)
        if dr is not None:
            t0 = _ny_fake_utc_seconds(dr["start"])
            rectangles.append({"t0": t0, "t1": future_edge, "p0": dr["eq"], "p1": dr["top"],
                                "fill": _hex_to_rgba(theme.NEON_MAGENTA, 0.14), "border": None})
            rectangles.append({"t0": t0, "t1": future_edge, "p0": dr["bottom"], "p1": dr["eq"],
                                "fill": _hex_to_rgba(theme.NEON_GREEN, 0.14), "border": None})
            price_lines.append({"t0": t0, "t1": future_edge, "price": dr["eq"],
                                 "color": _hex_to_rgba(theme.NEON_AMBER, 1.0), "title": "EQ 50%", "above": True})

    if "Liquidity" in layers:
        above, below = detect_liquidity_levels(confirmed_df, n_above=max_items_per_side, n_below=max_items_per_side)
        for lvl in above:
            price_lines.append({"t0": _ny_fake_utc_seconds(lvl["time"]), "t1": future_edge, "price": lvl["price"],
                                 "color": _hex_to_rgba(theme.NEON_MAGENTA, 1.0), "title": "BSL", "above": True})
        for lvl in below:
            price_lines.append({"t0": _ny_fake_utc_seconds(lvl["time"]), "t1": future_edge, "price": lvl["price"],
                                 "color": _hex_to_rgba(theme.NEON_AMBER, 1.0), "title": "SSL", "above": False})

    return rectangles, price_lines


# ---------------------------------------------------------------------------
# eToro account panel. Reads a SNAPSHOT another process already wrote —
# never makes a live call of its own. See this module's own top-of-file
# docstring for the full reasoning; this is the one enforcement point.
# ---------------------------------------------------------------------------
_ETORO_SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), ".cache", "etoro_snapshot.json")


def _load_etoro_snapshot():
    """Reads .cache/etoro_snapshot.json — written by a SEPARATE Claude Code
    session through its own eToro MCP connection (get-my-portfolio-summary/
    get-my-balances/get-my-positions-and-orders), not by this Streamlit
    process. This function must NEVER be changed to call out to eToro
    directly: this deployed process has no eToro API credentials of its
    own, and there is no live-HTTP fallback to fall back to — a stale or
    missing snapshot is refreshed by asking Claude to re-fetch and rewrite
    this file, not by wiring a request in here. Returns None on anything
    missing/corrupt — the caller renders an honest placeholder for that,
    not an error."""
    try:
        with open(_ETORO_SNAPSHOT_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, dict) and data.get("totals") else None
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError, TypeError):
        return None


def _snapshot_age_label(generated_at):
    try:
        age = pd.Timestamp.now(tz="UTC") - pd.Timestamp(generated_at)
        secs = age.total_seconds()
        if secs < 0:
            return "just now"
        if secs < 3600:
            return f"{int(secs // 60)} min ago"
        if secs < 86400:
            return f"{secs / 3600:.1f}h ago"
        return f"{secs / 86400:.1f}d ago"
    except Exception:
        return "unknown age"


def _stat_pills(items):
    """A strip of small color-coded pills for a handful of headline
    numbers — reuses theme.py's own .fvg-legend-strip/.fvg-legend-pill/
    .fvg-legend-dot/.fvg-legend-label/.fvg-legend-value CSS classes
    UNCHANGED, the exact same "small pill per stat" markup crypto_app.py's
    own fvg_legend() helper already uses for its FVG/OB legend counts,
    rather than inventing new markup for this panel's own totals row.
    items: list of (label, value_text, color) tuples."""
    pills = "".join(
        f'<div class="fvg-legend-pill" style="border-color:{color}66;">'
        f'<span class="fvg-legend-dot" style="background:{color};"></span>'
        f'<span class="fvg-legend-label">{label}</span>'
        f'<span class="fvg-legend-value" style="color:{color};">{value}</span>'
        f'</div>'
        for label, value, color in items
    )
    st.markdown(f'<div class="fvg-legend-strip">{pills}</div>', unsafe_allow_html=True)


def _render_etoro_account_panel():
    """Renders totals + open holdings/positions + pending orders from
    _load_etoro_snapshot() — see that function's own docstring for the
    "never call eToro directly" constraint this panel depends on. Reuses
    this project's own existing conventions for a stat readout (_stat_pills
    above, theme.py's own .fvg-legend-* CSS already used the same way by
    crypto_app.py's own fvg_legend()), a table (st.dataframe, the same
    convention crypto_app.py's own Backtest tab uses for its trades table),
    and a highlighted placeholder banner (theme.py's own .verdict-box CSS
    class, used the same way by institutional_app.py) rather than
    inventing new markup for any of them."""
    snapshot = _load_etoro_snapshot()
    if snapshot is None:
        st.markdown(
            f'<div class="verdict-box" style="border-color:{theme.NEON_AMBER};">'
            f'<span class="verdict-icon" style="color:{theme.NEON_AMBER};">●</span>'
            f'<div style="flex:1;">'
            f'<div class="verdict-headline" style="color:{theme.NEON_AMBER};">'
            f'No eToro snapshot yet — ask Claude to refresh it.</div>'
            f'<div class="verdict-detail">Expected at .cache/etoro_snapshot.json, written by a separate '
            f'Claude session through its own eToro MCP connection — this page never fetches it directly.'
            f'</div></div></div>',
            unsafe_allow_html=True,
        )
        return

    totals = snapshot["totals"]
    currency = snapshot.get("account_currency", "USD")
    st.caption(f"eToro {snapshot.get('account', 'demo')} account · {currency} · "
               f"snapshot {_snapshot_age_label(snapshot.get('generated_at'))}")

    _pnl_pct = (totals["unrealized_pnl"] / totals["total_value"] * 100) if totals.get("total_value") else None
    _pnl_value = (f"${totals['unrealized_pnl']:+,.2f} ({_pnl_pct:+.2f}%)" if _pnl_pct is not None
                  else f"${totals['unrealized_pnl']:+,.2f}")
    _pnl_color = theme.NEON_GREEN if totals["unrealized_pnl"] >= 0 else theme.NEON_MAGENTA
    _stat_pills([
        ("Total value", f"${totals['total_value']:,.2f}", theme.NEON_CYAN),
        ("Available cash", f"${totals['available_cash']:,.2f}", theme.NEON_CYAN),
        ("Unrealized P&L", _pnl_value, _pnl_color),
        ("Used margin", f"${totals['used_margin']:,.2f}", theme.NEON_AMBER),
    ])

    holdings = snapshot.get("holdings") or []
    if holdings:
        st.markdown("**Open holdings**")
        _rows = [{
            "Symbol": h["symbol"], "Name": h.get("name", h["symbol"]), "Units": h["units"],
            "Avg open": h["avg_open_rate"], "Current": h["current_rate"], "Value": h["value"],
            "P&L": h["pnl"], "P&L %": h["pnl_percent"], "Leverage": h.get("leverage", 1),
            "Positions": h.get("position_count", len(h.get("positions") or [])),
        } for h in holdings]
        st.dataframe(
            pd.DataFrame(_rows), width="stretch", hide_index=True,
            column_config={
                "Avg open": st.column_config.NumberColumn(format="%.5g"),
                "Current": st.column_config.NumberColumn(format="%.5g"),
                "Value": st.column_config.NumberColumn(format="$%.2f"),
                "P&L": st.column_config.NumberColumn(format="$%+.2f"),
                "P&L %": st.column_config.NumberColumn(format="%+.2f%%"),
                "Leverage": st.column_config.NumberColumn(format="%dx"),
            },
        )
    else:
        st.caption("No open holdings.")

    pending = snapshot.get("pending_orders") or []
    if pending:
        st.markdown("**Pending orders**")
        st.dataframe(pd.DataFrame(pending), width="stretch", hide_index=True)
    else:
        st.caption("No pending orders.")

    copied = snapshot.get("copied_traders") or []
    if copied:
        st.markdown("**Copied traders**")
        st.dataframe(pd.DataFrame(copied), width="stretch", hide_index=True)

    if snapshot.get("note"):
        tiny(snapshot["note"])


# ---------------------------------------------------------------------------
# Remembers the last ticker/timeframe/layers across page reloads and server
# restarts — own file, not app.py's/crypto_app.py's own .last_state*.json,
# same "each page needs its own slot" reasoning as crypto_app.py's own
# identical comment (sharing one file means picking a ticker on one page
# silently overwrites what another restores on its own next load).
# ---------------------------------------------------------------------------
_LAST_STATE_FILE = os.path.join(os.path.dirname(__file__), ".last_state_etoro.json")


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
st.session_state.setdefault("et_ticker", _last_state.get("ticker", "EURUSD=X"))
st.session_state.setdefault("et_active_category", "Forex")
st.session_state.setdefault("et_tf", _last_state.get("tf", "4h"))

# Lets an external link/bookmark drive the ticker via ?ticker=... — same
# convention as app.py/crypto_app.py's own identical block.
qp_ticker = st.query_params.get("ticker")
if qp_ticker and st.session_state.get("_last_qp_ticker") != qp_ticker:
    st.session_state["et_ticker"] = qp_ticker.strip().upper()
    st.session_state["_last_qp_ticker"] = qp_ticker

_last_layers = _last_state.get("layers", {})
for _name in ICT_LAYERS:
    st.session_state.setdefault(f"et_layer_{_name}", _last_layers.get(_name, LAYER_DEFAULTS[_name]))

theme.render_top_bar("eToro")

ticker = st.session_state.get("et_ticker", "EURUSD=X").strip().upper()

with st.expander("💰 eToro Account (demo)", expanded=True):
    _render_etoro_account_panel()

# ---------------------------------------------------------------------------
# Sidebar: the "Validated edge" scanner — the same statistically-validated
# tier app.py/crypto_app.py's own sidebar shows, reusing experiments.py's
# accumulated, BH-corrected trial log unchanged. Deliberately the ONLY
# sidebar scanner on this page (no "Scan watchlist"/"Scan timeframes"
# confluence-only scans, unlike crypto_app.py's own sidebar) — this page's
# whole purpose is the validated edge once it lands, not another surface
# for the discretionary-confluence scanners the other two pages already
# have.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("🔬 Validated edge")
    if st.button("🔬 Scan for validated setups", key="et_validated_scan_btn", width="stretch",
                 help="Checks every ticker/timeframe on this page's own universe that has ever cleared a "
                      "real permutation test (Experiments tab \"Run deep backtest\", BH-corrected across "
                      "everything ever tried) for a signal that's live right now — not just confluence, "
                      "actual statistical evidence. Usually empty; that's honest, not broken — nothing "
                      "eToro-specific has been validated through this mechanism yet."):
        _val_pairs = experiments.list_validated_pairs(TICKER_UNIVERSE)
        _val_specs = [(get_yf_ohlcv, (t, TIMEFRAMES[tf]["period"], TIMEFRAMES[tf]["fetch_interval"], "auto"), {})
                      for t, tf in _val_pairs]
        _prefetch(_val_specs)
        _val_hits = []
        for _val_ticker, _val_tf in _val_pairs:
            _val_conf = TIMEFRAMES[_val_tf]
            try:
                _val_df = get_yf_ohlcv(_val_ticker, period=_val_conf["period"], interval=_val_conf["fetch_interval"])
                if _val_conf["resample"] and not _val_df.empty:
                    _val_df = resample_ohlc(_val_df, _val_conf["resample"])
                _val_hit = experiments.find_live_validated_signal(_val_df, _val_ticker, _val_tf)
            except Exception:
                _val_hit = None
            if _val_hit:
                _val_hits.append(_val_hit)
        st.session_state["_et_validated_scan"] = _val_hits
        st.session_state["_et_validated_scan_checked"] = len(_val_pairs)

    _val_results = st.session_state.get("_et_validated_scan")
    if _val_results is None:
        st.caption("Not scanned yet this session.")
    elif not _val_results:
        _val_n_checked = st.session_state.get("_et_validated_scan_checked", 0)
        if _val_n_checked == 0:
            st.caption("Nothing on this universe has cleared a validated Experiments backtest yet — run "
                       "\"Run deep backtest\" in a chart's own 🧪 Experiments tab (Markets/Crypto) to build "
                       "one, or wait for the eToro-side edge validation to land here.")
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
            if st.button(_v_label, key=f"et_val_jump_{_v['ticker']}_{_v['tf_label']}_{_v['entry_time']}",
                         width="stretch", help=_v_help):
                st.session_state["et_ticker"] = _v["ticker"]
                st.session_state["et_tf"] = _v["tf_label"]
                st.session_state["_et_active_validated_pick"] = {
                    "ticker": _v["ticker"], "tf_label": _v["tf_label"], "direction": _v["direction"],
                    "entry": _v["entry_price"], "stop": _v["stop_price"], "target": _v["target_price"],
                    "label": _v["label"], "p_value_train": _v["p_value_train"], "n_events": _v["n_events"],
                    "mean_return_holdout": _v["mean_return_holdout"],
                }
                st.rerun()

# ---------------------------------------------------------------------------
# Ticker picker — a popover with a category-button row (six categories, this
# page's own universe spans all of them, unlike either single-category-ish
# source page) plus search, same click-based pattern as app.py's own
# "change market" popover (see there for why click, not hover).
# ---------------------------------------------------------------------------
with st.popover(f"🔍 {_ticker_label(ticker)} · change market", width="stretch"):
    _cat_cols = st.columns(len(TICKER_CATEGORIES))
    for _cat_col, _cat_name in zip(_cat_cols, TICKER_CATEGORIES.keys()):
        with _cat_col:
            _is_active_cat = st.session_state["et_active_category"] == _cat_name
            if st.button(_cat_name, key=f"et_cat_{_cat_name}", use_container_width=True,
                         type="primary" if _is_active_cat else "secondary"):
                st.session_state["et_active_category"] = _cat_name
                st.session_state["et_ticker_search"] = ""
                st.rerun()

    _cat_tags = TICKER_CATEGORIES[st.session_state["et_active_category"]]
    _cat_symbols = [t for t in TICKER_UNIVERSE if TICKER_INFO[t][1] in _cat_tags]
    _search_query = st.text_input(
        "Search", key="et_ticker_search", label_visibility="collapsed",
        placeholder=f"Search {st.session_state['et_active_category']} ({len(_cat_symbols)} available)…",
    )
    if _search_query.strip():
        _q = _search_query.strip().upper()
        _matches = [t for t in _cat_symbols if _q in t.upper() or _q in TICKER_INFO[t][0].upper()]
    else:
        # Every category here is a small, hand-curated/starter list — show
        # it all immediately, nothing gained from forcing a search first.
        _matches = list(_cat_symbols)
    _matches = sorted(_matches)[:40]

    if _matches:
        _prefetch([(is_ticker_alive, (t,), {}) for t in _matches])
        for t in _matches:
            _dot = "🟢" if is_ticker_alive(t) else "🔴"
            if st.button(f"{_dot}  {_ticker_label(t)}", key=f"et_pick_{t}", use_container_width=True):
                st.session_state["et_ticker"] = t
                st.rerun()
    elif _search_query.strip():
        st.caption("No matches.")

# main_col MUST stay a real st.columns()-produced column, not a plain
# st.container() — same load-bearing reason as app.py/crypto_app.py's own
# identical comment: theme.py's CHART_LAYOUT_CSS is built entirely on
# [data-testid="stColumn"] ancestor selectors, so a bare container silently
# loses the chart's own fill-height flex behavior.
main_col = st.columns(1)[0]

with main_col:
    # Bound directly to "et_tf" (already seeded above, before this widget is
    # created — same seed-before-widget-creation trick as the ticker) rather
    # than through a separate proxy key: app.py/crypto_app.py need that
    # extra indirection because ONE physical TF bar is shared by THREE
    # independently timeframe-able charts (main/HTF/LTF) and has to resync
    # depending on which one currently owns it. This page has a single
    # chart, so there's nothing to own-switch — binding straight to et_tf
    # means a sidebar "Validated edge" click that sets et_tf directly (see
    # below) is picked up on the very next rerun with no extra bookkeeping,
    # and no risk of a stale proxy value silently overwriting it back.
    st.radio("TF", list(TIMEFRAMES.keys()), key="et_tf", horizontal=True, label_visibility="collapsed")
    tf_label = st.session_state["et_tf"]
    tf = TIMEFRAMES[tf_label]
    refresh_interval = CANDLE_SECONDS.get(tf_label)

    # Layers/settings live in their own fragment (no run_every — reacts only
    # to its own widgets) so a checkbox toggle here doesn't force the whole
    # page (sidebar scan, account panel, ticker popover) to re-render — same
    # cross-fragment split, same reasoning, as app.py/crypto_app.py's own
    # _render_layer_controls/_render_chart pair: this fragment WRITES
    # session_state["_chart_controls"], the separately-auto-ticking chart
    # fragment below READS it back out at the top of its own body every
    # tick, so a change here reaches the chart on its own next tick without
    # this fragment needing to know anything about that schedule.
    @st.fragment
    def _render_controls():
        _prev_cc = st.session_state.get("_chart_controls")
        with st.popover("⚙️ Layers & settings", width="stretch"):
            st.markdown("**Detectors**")
            layers = [name for name in ICT_LAYERS if st.checkbox(name, key=f"et_layer_{name}")]
            st.markdown("---")
            max_items_per_side = st.number_input(
                "Areas per side", min_value=1, max_value=50, value=5, step=1, key="et_max_items",
                help="FVG/Order Blocks/Liquidity: shows the N zones closest to price on each side "
                     "(above + below) — the ones actually active right now.")
            zone_opacity = st.slider("Zone opacity", 0.05, 0.6, 0.25, 0.05, key="et_zone_opacity")
            data_source_label = st.selectbox("Data source", list(DATA_SOURCES.keys()), index=0,
                                              key="et_data_source")
            data_source = DATA_SOURCES[data_source_label]

        _new_cc = {"layers": layers, "max_items_per_side": max_items_per_side,
                   "zone_opacity": zone_opacity, "data_source": data_source}
        st.session_state["_chart_controls"] = _new_cc
        # Daily+ timeframes have no auto-tick of their own to catch a change
        # up on later, so force a full rerun there — same narrowly-scoped
        # fix (and same infinite-loop hazard avoided by checking _new_cc !=
        # _prev_cc first) as app.py/crypto_app.py's own identical comment.
        if _new_cc != _prev_cc and CANDLE_SECONDS.get(st.session_state.get("et_tf")) is None:
            st.rerun()

    _render_controls()

    @st.fragment(run_every=refresh_interval)
    def _render_chart():
        try:
            _cc = st.session_state.get("_chart_controls", {})
            layers = _cc.get("layers", [])
            max_items_per_side = _cc.get("max_items_per_side", 5)
            zone_opacity = _cc.get("zone_opacity", 0.25)
            data_source = _cc.get("data_source", "auto")

            df = get_yf_ohlcv(ticker, period=tf["period"], interval=tf["fetch_interval"], provider=data_source)
            served_by = DATA_SOURCE_LABELS.get(df.attrs.get("provider"), "Yahoo Finance") if not df.empty else "—"
            if df.empty:
                ict_chart(
                    [], f"{ticker}|{tf_label}|empty",
                    overlays={"rectangles": [], "price_lines": [], "markers": []},
                    options={"volume": False, "error_message": f"No data for {ticker} at {tf_label}"},
                    ohlc={"ticker": ticker, "source": "—"},
                    height=1000, key="ict_chart_etoro", display_tz=theme.get_display_tz(),
                )
                return

            forming_bar_ts = None
            if tf["resample"] is None and refresh_interval is not None:
                # Splice the cheap, fast-poll window onto the slow full
                # fetch, and accumulate the still-forming last bar's real
                # high/low across polls — same pattern, same reasoning, as
                # app.py/crypto_app.py's own identical block (Yahoo's own
                # still-forming bar comes back degenerate, O==H==L==C).
                df_latest = get_latest_bars(ticker, tf["fetch_interval"], provider=data_source)
                if not df_latest.empty:
                    df = pd.concat([df[df.index < df_latest.index[0]], df_latest])
                    last_ts = df.index[-1]
                    accum_key = f"_et_forming_bar_{ticker}_{tf_label}"
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
                ict_chart(
                    [], f"{ticker}|{tf_label}|empty",
                    overlays={"rectangles": [], "price_lines": [], "markers": []},
                    options={"volume": False, "error_message": f"No data for {ticker} at {tf_label}"},
                    ohlc={"ticker": ticker, "source": "—"},
                    height=1000, key="ict_chart_etoro", display_tz=theme.get_display_tz(),
                )
                return

            o_col, h_col, l_col, c_col = (("Open", "High", "Low", "Close") if "Close" in df
                                           else ("open", "high", "low", "close"))
            v_col = "Volume" if "Volume" in df else ("volume" if "volume" in df else None)
            axis_secs = _ny_fake_utc_seconds_vec(df.index)
            bar_step = (axis_secs[-1] - axis_secs[-2]) if len(axis_secs) > 1 else 1
            future_edge = axis_secs[-1] + 3 * bar_step
            forming_bar_axis_sec = axis_secs[-1] if forming_bar_ts is not None else None

            _o, _h, _l, _c = (df[o_col].to_numpy(), df[h_col].to_numpy(),
                              df[l_col].to_numpy(), df[c_col].to_numpy())
            _v = df[v_col].to_numpy() if v_col is not None else None
            bars = [{"time": axis_secs[i], "open": float(_o[i]), "high": float(_h[i]),
                     "low": float(_l[i]), "close": float(_c[i]), "volume": float(_v[i]) if _v is not None else 0.0}
                    for i in range(len(df))]
            current_price = float(df[c_col].iloc[-1])

            confirmed = df.iloc[:-1] if len(df) > 1 else df
            rectangles, price_lines = _build_layer_overlays(
                layers, confirmed, current_price, future_edge, max_items_per_side, zone_opacity)

            # ---- On-chart "best trade right now" -----------------------
            # A "Validated edge" sidebar pick (real permutation-test
            # evidence) takes priority over best_trade_now's own plain
            # confluence ranking when it matches this exact ticker+
            # timeframe — same "an explicit pick outranks the generic
            # read" precedence as app.py/crypto_app.py's own
            # _active_scan_pick. Not timeframe-locked the way that
            # mechanism optionally is here — switching away from the
            # picked timeframe just falls through to best_trade_now below,
            # nothing to unlock.
            #
            # NOTE for whoever wires up the real eToro edge: this is where
            # that lands. No order-placement/execution UI exists here on
            # purpose — there is no validated edge to trigger yet (see
            # this module's own top-of-file docstring). When it does, the
            # natural place for a "place this trade" action is right here,
            # next to whichever pick (validated-edge or best_trade_now)
            # is currently driving these three lines — and it would still
            # need to go through Claude's own eToro MCP connection in
            # conversation, never a direct call from this process (same
            # constraint _load_etoro_snapshot's docstring states for
            # reading account data).
            _vp = st.session_state.get("_et_active_validated_pick")
            _vp_here = bool(_vp) and _vp["ticker"] == ticker and _vp["tf_label"] == tf_label
            _entry_x = _ny_fake_utc_seconds(df.index[-1])
            _box_t1 = _entry_x + _TF_BAR_SECONDS[tf_label] * 30
            if _vp_here:
                entry_price, stop_price, exit_price = _vp["entry"], _vp["stop"], _vp["target"]
                pattern_label = f"Validated: {_vp['label']} ({_vp['direction']})"
                caption = (f"Validated edge: {_vp['label']} ({_vp['direction']}) · p={_vp['p_value_train']:.4f} · "
                           f"n={_vp['n_events']} · holdout {_vp['mean_return_holdout']:+.2%}. Target is a "
                           f"PROJECTION of the historical holdout mean move, not a real take-profit price.")
                body_lines = []
            else:
                picks = best_trade_now(df, ticker, tf["fetch_interval"], data_source, top_n=1)
                if picks:
                    pick = picks[0]
                    entry_price, stop_price, exit_price = pick["entry_price"], pick["sl_price"], pick["tp_price"]
                    pattern = EVENT_TYPE_LABELS.get(pick["event_type"], pick["event_type"])
                    pattern_label = f"{pattern} ({pick['direction']})"
                    _n_conf = len(pick.get("confluence_entry_details") or [])
                    body_lines = [f"{_n_conf} confluence factor(s)"]
                    _dec = _price_decimals(entry_price)
                    caption = (f"Potential trade: {pattern} ({pick['direction']}) · confluence "
                               f"{pick['confluence_score']} · Entry {entry_price:.{_dec}f} · "
                               f"Stop {stop_price:.{_dec}f} · Target {exit_price:.{_dec}f} — provisional "
                               f"confluence-only ranking, not the validated eToro edge yet.")
                else:
                    entry_price = stop_price = exit_price = None
                    pattern_label, body_lines = "", []
                    caption = "No active setup at this ticker/timeframe right now."

            if entry_price is not None:
                for _price, _color, _title in [(entry_price, theme.NEON_AMBER, "Entry"),
                                                (stop_price, theme.NEON_MAGENTA, "SL"),
                                                (exit_price, theme.NEON_GREEN, "TP")]:
                    price_lines.append({"t0": _entry_x, "t1": _box_t1, "price": _price,
                                         "color": _hex_to_rgba(_color, 1.0), "title": _title,
                                         "line_width": 2, "dashed": True, "above": bool(_price >= entry_price)})
                rectangles.append({"t0": _entry_x, "t1": _box_t1, "p0": entry_price, "p1": stop_price,
                                    "fill": _hex_to_rgba(theme.NEON_MAGENTA, 0.10),
                                    "border": _hex_to_rgba(theme.NEON_MAGENTA, 0.9)})
                rectangles.append({"t0": _entry_x, "t1": _box_t1, "p0": entry_price, "p1": exit_price,
                                    "fill": _hex_to_rgba(theme.NEON_GREEN, 0.10),
                                    "border": _hex_to_rgba(theme.NEON_GREEN, 0.9),
                                    "label": pattern_label, "body_lines": body_lines})

            fingerprint = f"{ticker}|{tf_label}|{len(df)}|{df.index[0]}"
            ict_chart(
                bars, fingerprint,
                overlays={"rectangles": rectangles, "price_lines": price_lines, "markers": []},
                options={"log_scale": False, "volume": False, "selected": True},
                ohlc={"ticker": ticker, "source": served_by, "interval": tf_label,
                      "forming_bar_time": forming_bar_axis_sec},
                height=1000, key="ict_chart_etoro", display_tz=theme.get_display_tz(),
            )
            st.caption(caption)
        except Exception as e:
            st.error(f"Chart error: {e}")

    _render_chart()

    _save_last_state(ticker, tf_label, {name: st.session_state.get(f"et_layer_{name}", False) for name in ICT_LAYERS})
