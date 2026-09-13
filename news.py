"""
Economic calendar — high-impact macro news, sourced from ForexFactory's own
public JSON calendar feed (nfs.faireconomy.media), the same feed its own
website widget uses. No API key, no HTML scraping.

Only the "thisweek" endpoint resolves on this mirror (confirmed directly:
lastweek/nextweek/thismonth/nextmonth all 404) — one continuous Sunday-
through-Saturday window that ForexFactory itself rolls forward onto the
new week starting Sunday. So "the calendar" here always means "the rest of
the current ForexFactory week," never a fixed N-day lookahead past that
boundary — good enough for "what's left this week" and for flagging
same-day news on the session pills, not a multi-week outlook. The feed
also never carries an "actual" value (checked directly — the key is simply
absent from every row, not just blank) — every note this module produces
is necessarily a BEFORE-the-release read, never a "here's what just
printed" one.

Presentation-agnostic like recommender.py — this module owns only the
calendar fetch/parse and the (deliberately modest) confluence-style notes;
news_app.py renders it, and theme.py's session pills consume
session_news_map() directly.
"""

import json
import os
from datetime import time as dtime

import pandas as pd
import requests
import streamlit as st

_FEED_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
_CACHE_TTL = 1800  # 30 min -- calendar entries barely change intra-day; theme.py's
                    # session-pill fragment reruns every 60s, so this cache is what
                    # keeps that from hammering the feed on every tick.

_CALENDAR_COLUMNS = ["title", "country", "time", "impact", "forecast", "previous"]

# Which of this project's own curated tickers (see app.py/crypto_app.py's
# CURATED_FOREX/CURATED_INDICES/CURATED_COMMODITIES, and crypto_app.py's own
# BTC/ETH) react to a given currency's news, plus which dashboard (port)
# each one lives on for the deep-link. Duplicated here rather than imported
# from app.py/crypto_app.py for the exact reason research/sessions.py's own
# KILL_ZONES duplication gives: those are runnable Streamlit scripts
# (st.set_page_config side effect on import), not safe import targets for
# another module — small, stable constants get mirrored instead.
MARKETS_PORT = 8501
CRYPTO_PORT = 8506

CURRENCY_ASSETS = {
    "USD": [
        ("EURUSD=X", "EUR/USD", MARKETS_PORT), ("GBPUSD=X", "GBP/USD", MARKETS_PORT),
        ("USDJPY=X", "USD/JPY", MARKETS_PORT), ("USDCHF=X", "USD/CHF", MARKETS_PORT),
        ("AUDUSD=X", "AUD/USD", MARKETS_PORT), ("USDCAD=X", "USD/CAD", MARKETS_PORT),
        ("NZDUSD=X", "NZD/USD", MARKETS_PORT),
        ("^GSPC", "S&P 500", MARKETS_PORT), ("^DJI", "Dow Jones", MARKETS_PORT),
        ("^IXIC", "Nasdaq", MARKETS_PORT), ("^VIX", "VIX", MARKETS_PORT),
        ("GC=F", "Gold", MARKETS_PORT), ("SI=F", "Silver", MARKETS_PORT), ("CL=F", "Crude Oil", MARKETS_PORT),
        ("BTC-USD", "Bitcoin", CRYPTO_PORT), ("ETH-USD", "Ethereum", CRYPTO_PORT),
    ],
    "EUR": [
        ("EURUSD=X", "EUR/USD", MARKETS_PORT), ("EURJPY=X", "EUR/JPY", MARKETS_PORT),
        ("EURGBP=X", "EUR/GBP", MARKETS_PORT), ("^GDAXI", "DAX", MARKETS_PORT),
    ],
    "GBP": [
        ("GBPUSD=X", "GBP/USD", MARKETS_PORT), ("GBPJPY=X", "GBP/JPY", MARKETS_PORT),
        ("EURGBP=X", "EUR/GBP", MARKETS_PORT), ("^FTSE", "FTSE 100", MARKETS_PORT),
    ],
    "JPY": [
        ("USDJPY=X", "USD/JPY", MARKETS_PORT), ("EURJPY=X", "EUR/JPY", MARKETS_PORT),
        ("GBPJPY=X", "GBP/JPY", MARKETS_PORT), ("^N225", "Nikkei 225", MARKETS_PORT),
    ],
    "CHF": [("USDCHF=X", "USD/CHF", MARKETS_PORT)],
    "AUD": [("AUDUSD=X", "AUD/USD", MARKETS_PORT)],
    "CAD": [("USDCAD=X", "USD/CAD", MARKETS_PORT), ("CL=F", "Crude Oil", MARKETS_PORT)],
    "NZD": [("NZDUSD=X", "NZD/USD", MARKETS_PORT)],
    # CNY has no direct instrument in this project's curated lists -- China
    # data mostly matters here as a read-through onto broad risk sentiment
    # and commodity demand, not a tradeable pair of its own.
    "CNY": [("GC=F", "Gold", MARKETS_PORT), ("CL=F", "Crude Oil", MARKETS_PORT), ("^N225", "Nikkei 225", MARKETS_PORT)],
}

# Which trading session (see theme.py's own TRADING_SESSIONS) a currency's
# news mostly lands in -- the currencies whose own central bank/data desk
# is actually awake and releasing during that window, not every currency
# that happens to move then. Feeds theme.py's session-pill "flash."
SESSION_CURRENCIES = {
    "Asia": {"JPY", "AUD", "NZD", "CNY"},
    "London": {"GBP", "EUR", "CHF"},
    "New York": {"USD", "CAD"},
}

# A small, deliberately conservative set of common indicator-name fragments
# where "beat vs miss" has a widely-taught, unambiguous direction for the
# currency -- NOT a full economic-calendar taxonomy, and not a prediction.
# Anything not matched here gets the generic volatility note instead of a
# guessed direction; see get_expectation_note()'s own docstring for why.
_HIGHER_IS_CURRENCY_POSITIVE = [
    "gdp", "retail sales", "pmi", "ism", "employment change", "non-farm",
    "nonfarm", "payrolls", "trade balance", "industrial production",
    "consumer confidence", "durable goods", "housing starts", "cpi", "ppi",
    "interest rate", "refinancing rate", "cash rate", "bank rate", "ocr",
]
_HIGHER_IS_CURRENCY_NEGATIVE = [
    "unemployment rate", "jobless claims", "claimant count",
]


@st.cache_data(ttl=_CACHE_TTL)
def _fetch_calendar_raw():
    """The network call, cached only on SUCCESS — st.cache_data never
    caches a raised exception, so a transient outage/rate-limit here
    doesn't get locked in for the full 30-minute TTL the way caching a
    fallback empty result would; the very next call (next 60s session-pill
    tick, or a manual page reload) just retries the fetch fresh instead."""
    resp = requests.get(_FEED_URL, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_calendar():
    """This ForexFactory week's calendar, parsed. Returns a DataFrame with
    columns title/country/time (tz-aware UTC)/impact/forecast/previous,
    sorted by time — or an empty DataFrame with the same columns if the
    feed is unreachable. A feed outage and "no events" deliberately look
    the same to every caller (an empty result, never an exception) — this
    is read by theme.py's session pills on every page, every 60s, and must
    never take the top bar down with it."""
    try:
        raw = _fetch_calendar_raw()
    except Exception:
        return pd.DataFrame(columns=_CALENDAR_COLUMNS)

    df = pd.DataFrame(raw)
    if df.empty:
        return pd.DataFrame(columns=_CALENDAR_COLUMNS)

    df["time"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"])
    for col in _CALENDAR_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[_CALENDAR_COLUMNS].sort_values("time").reset_index(drop=True)
    _record_calendar_history(df)
    return df


# ---------------------------------------------------------------------------
# News blackout — a second engine, for excluding bars around high-impact
# news from backtesting/win-rate stats rather than just flashing a session
# pill. This module's own top docstring is a hard constraint here: the only
# LIVE feed available is "the rest of the current ForexFactory week," never
# a real historical calendar. Three sources are combined instead of
# pretending one is enough:
#   1. The live feed above, when the requested range overlaps "this week"
#      (exact, comprehensive — every High-impact country/title ForexFactory
#      tracks, not just a curated subset).
#   2. A local accumulation cache (below): every High-impact row this app
#      has ever actually observed live gets appended here, so real
#      historical coverage grows for free the longer the app runs, with no
#      new network calls or dependencies. Empty until enough weeks pass —
#      an honest limitation, not silently glossed over.
#   3. US Non-Farm Payrolls' own fixed public schedule (first Friday of the
#      month, 8:30 ET) — computed, not fetched, so it's exact for ANY date,
#      past or future, unlike everything else here. The one event type
#      with a genuinely fixed rule; FOMC/CPI don't follow one precise
#      enough to hardcode without real risk of a silently wrong date
#      corrupting backtest stats, so they're deliberately left to sources 1
#      and 2 only rather than guessed at.
_HISTORY_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "news_history.json")
_HISTORY_KEY_COLS = ["title", "country", "time"]


def _load_history_cache():
    try:
        with open(_HISTORY_CACHE_PATH) as f:
            rows = json.load(f)
    except Exception:
        return pd.DataFrame(columns=_CALENDAR_COLUMNS)
    if not rows:
        return pd.DataFrame(columns=_CALENDAR_COLUMNS)
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"])
    return df[_CALENDAR_COLUMNS] if not df.empty else pd.DataFrame(columns=_CALENDAR_COLUMNS)


def _record_calendar_history(df):
    """Appends this fetch's own High-impact rows to the local accumulation
    cache, deduplicated by (title, country, time) so re-fetching the same
    still-current week every 30 minutes doesn't grow the file. Best-effort:
    a write failure (read-only filesystem, disk full) is swallowed rather
    than breaking the calendar fetch that triggered it — this cache is a
    bonus, not something any caller depends on existing."""
    high = df[df["impact"] == "High"]
    if high.empty:
        return
    try:
        existing = _load_history_cache()
        merged = pd.concat([existing, high], ignore_index=True)
        merged["time"] = pd.to_datetime(merged["time"], utc=True)
        merged = merged.drop_duplicates(subset=_HISTORY_KEY_COLS).sort_values("time")
        os.makedirs(os.path.dirname(_HISTORY_CACHE_PATH), exist_ok=True)
        out = merged.copy()
        out["time"] = out["time"].apply(lambda t: t.isoformat())
        with open(_HISTORY_CACHE_PATH, "w") as f:
            json.dump(out.to_dict(orient="records"), f)
    except Exception:
        pass


_NFP_TIME = dtime(8, 30)


def _nfp_release_times(start, end):
    """Every US Non-Farm Payrolls release (first Friday of the month, 8:30
    America/New_York — a fixed BLS-calendar rule with no real exceptions)
    falling in [start, end] (both tz-aware). The one event in this module
    computed rather than fetched or cached — see the section docstring
    above for why only this one gets that treatment."""
    start_ny = start.tz_convert("America/New_York")
    end_ny = end.tz_convert("America/New_York")
    out = []
    y, m = start_ny.year, start_ny.month
    while True:
        month_start = pd.Timestamp(year=y, month=m, day=1, tz="America/New_York")
        if month_start > end_ny:
            break
        friday_offset = (4 - month_start.weekday()) % 7  # Friday == weekday 4
        release = (month_start + pd.Timedelta(days=friday_offset)).replace(
            hour=_NFP_TIME.hour, minute=_NFP_TIME.minute, second=0, microsecond=0)
        if start_ny <= release <= end_ny:
            out.append(release.tz_convert("UTC"))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


# Reverse of CURRENCY_ASSETS — which currency codes' news actually moves a
# given ticker. Built once at import time so a currency added to
# CURRENCY_ASSETS is automatically covered here too, no separate table to
# keep in sync.
_TICKER_CURRENCIES = {}
for _currency, _assets in CURRENCY_ASSETS.items():
    for _ticker, _name, _port in _assets:
        _TICKER_CURRENCIES.setdefault(_ticker, set()).add(_currency)


def ticker_currencies(ticker):
    """Which currency codes' High-impact news should count as a blackout
    for `ticker` — empty set for one with no curated mapping (an honest
    empty result, same convention as impacted_assets)."""
    return _TICKER_CURRENCIES.get(ticker, set())


def high_impact_events_in_range(start, end, currencies=None):
    """Every known High-impact event between `start`/`end` (both tz-aware),
    merged from the live feed, the accumulation cache, and NFP's own fixed
    schedule (see this section's own docstring for what each does and
    doesn't cover), deduplicated by (title, country, time). Filtered to
    `currencies` when given. Columns match get_calendar()'s own shape."""
    live = get_calendar()
    if not live.empty:
        live = live[(live["time"] >= start) & (live["time"] <= end) & (live["impact"] == "High")]
    cached = _load_history_cache()
    if not cached.empty:
        cached = cached[(cached["time"] >= start) & (cached["time"] <= end)]
    nfp_times = _nfp_release_times(start, end)
    nfp = (pd.DataFrame({"title": "Non-Farm Payrolls", "country": "USD", "time": nfp_times,
                          "impact": "High", "forecast": "", "previous": ""})
           if nfp_times else pd.DataFrame(columns=_CALENDAR_COLUMNS))
    merged = pd.concat([live, cached, nfp], ignore_index=True)
    if merged.empty:
        return merged
    merged = merged.drop_duplicates(subset=_HISTORY_KEY_COLS).sort_values("time").reset_index(drop=True)
    if currencies:
        merged = merged[merged["country"].isin(currencies)]
    return merged


def blackout_mask(index, ticker, minutes_before=15, minutes_after=15):
    """Boolean numpy array aligned to `index` (a tz-aware DatetimeIndex,
    e.g. an OHLCV df's own .index), True on every bar whose own timestamp
    falls within `minutes_before`/`minutes_after` of ANY High-impact event
    for `ticker`'s own relevant currencies — the shared "cut red news
    windows" primitive every backtesting-data-trimming call site in this
    project uses. False everywhere when `ticker` has no curated currency
    mapping (an honest no-op, not a silent full exclusion) or `index` is
    empty."""
    n = len(index)
    if n == 0:
        return pd.array([], dtype=bool).to_numpy()
    currencies = ticker_currencies(ticker)
    if not currencies:
        return pd.array([False] * n, dtype=bool).to_numpy()
    before = pd.Timedelta(minutes=minutes_before)
    after = pd.Timedelta(minutes=minutes_after)
    events = high_impact_events_in_range(index.min() - before, index.max() + after, currencies=currencies)
    mask = pd.Series(False, index=index)
    for t in events["time"]:
        mask |= (index >= t - before) & (index <= t + after)
    return mask.to_numpy()


def filter_news_blackout(df, ticker, minutes_before=15, minutes_after=15):
    """`df` with every bar inside a news-blackout window removed — the
    direct "trim backtesting data" convenience wrapper around
    blackout_mask, for a caller that just wants a trimmed dataframe rather
    than the mask itself (most backtesting call sites want the mask, so
    they can gate on it without re-slicing df themselves)."""
    mask = blackout_mask(df.index, ticker, minutes_before, minutes_after)
    return df[~mask]


def high_impact_today(df=None, tz="America/New_York"):
    """Rows of `df` (get_calendar()'s own shape; fetched fresh if omitted)
    whose release time falls on TODAY in `tz` local time and whose impact
    is High. `tz` defaults to America/New_York, same as every other
    session-timing check in this project (TRADING_SESSIONS' own hours are
    ET) — not theme.get_display_tz(), which only ever governs what gets
    RENDERED, never what "today"/session-membership means internally."""
    if df is None:
        df = get_calendar()
    if df.empty:
        return df
    local_date = df["time"].dt.tz_convert(tz).dt.date
    today = pd.Timestamp.now(tz=tz).date()
    return df[(df["impact"] == "High") & (local_date == today)]


def session_news_map(df=None, tz="America/New_York"):
    """{session_name: [row, ...]} for each of TRADING_SESSIONS' three
    sessions, from today's high-impact events only, matched by
    SESSION_CURRENCIES — the direct input to theme.py's session-pill
    "flash." Always returns all three keys, empty list when nothing
    qualifies (including on a feed outage), so a caller never has to
    branch on the dict being incomplete."""
    todays = high_impact_today(df, tz=tz)
    out = {name: [] for name in SESSION_CURRENCIES}
    if todays.empty:
        return out
    for _, row in todays.iterrows():
        for session, currencies in SESSION_CURRENCIES.items():
            if row["country"] in currencies:
                out[session].append(row)
    return out


def impacted_assets(country):
    """[(ticker, display_name, dashboard_port), ...] for a calendar row's
    own country/currency code — empty list for one with no curated
    instrument mapped (e.g. "All", or a currency this project doesn't
    trade), an honest empty result rather than a guessed one."""
    return CURRENCY_ASSETS.get(country, [])


def get_expectation_note(title):
    """Plain-language "what this kind of print usually does" — a small,
    deliberately conservative playbook note, not a prediction of where
    price goes. Matches this project's own established honesty convention
    elsewhere (recommender.py's validation_rank: a heuristic/confluence
    read is always labeled as discretionary, never dressed up as proof).
    Since this feed never carries an "actual" value (see this module's own
    docstring), every note here is necessarily a before-the-fact
    conditional ("if it beats... if it misses..."), not a live call."""
    t = title.lower()
    if any(k in t for k in _HIGHER_IS_CURRENCY_POSITIVE):
        return ("If the actual beats forecast, that's typically read as currency-positive; a miss is typically "
                "currency-negative. Not guaranteed — the broader rate cycle and risk sentiment can override it.")
    if any(k in t for k in _HIGHER_IS_CURRENCY_NEGATIVE):
        return ("A worse-than-forecast reading (higher than expected) is typically currency-negative here; a "
                "better-than-forecast (lower) reading is typically currency-positive.")
    return ("No reliable beat/miss direction for this one — but high-impact prints still often trigger a fast "
            "liquidity sweep through a nearby high/low in the first few minutes before the real move settles. "
            "Expect a volatility spike and wide wicks right around the release.")
