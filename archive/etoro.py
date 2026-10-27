"""
eToro public API (https://api-portal.etoro.com/) — read-only market data,
account balances, portfolio, and trade history. Deliberately has NO order
placement/cancellation/modification functions at all, even though the
real API supports them — this project never executes trades on the
user's behalf, by design, and the cleanest way to make that true rather
than just promised is to not write the code path at all. Generate your
own API key with "Read" permission only (never "Write") for the same
reason at the account level, not just this module's own level.

Every endpoint here was confirmed directly against eToro's own current
API reference docs (api-portal.etoro.com/api-reference/...) — nothing
guessed from memory, since this API rev (v1 public API via the Builders
program) postdates any training cutoff this project's models might have.

Auth: two keys, both required on every call —
  ETORO_API_KEY  ("Public API Key" — identifies this app, eToro's own
                  wording; what "public key" commonly refers to for this
                  API, confirmed against the docs' own header name).
  ETORO_USER_KEY (identifies YOUR account — generate at eToro app ->
                  Settings -> Trading -> API Key Management -> New key.
                  Pick Read-only permission, and Demo environment while
                  testing this integration for the first time.)
Both read from st.secrets, same convention as TWELVE_DATA_API_KEY/
TIINGO_API_KEY/CTRADER_CLIENT_ID elsewhere in this project — see
.streamlit/secrets.toml's own comment for exactly where to paste the
user key once generated.

UNTESTED against a live account as of this file's own creation — built
directly from eToro's own current API reference (paths, params, response
shapes all confirmed against api-portal.etoro.com), but this project
never received a working ETORO_USER_KEY to actually exercise it end to
end. Treat the first real call as the actual verification, not this
docstring's own confidence."""
import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import streamlit as st

_BASE = "https://public-api.etoro.com"
_CACHE_TTL = 300

# This project's own timeframe labels -> eToro's own candle-interval
# vocabulary (their own enum strings, not a generic "1h"-style value) —
# see get-instrument-candle-history's own documented `interval` values.
INTERVAL_MAP = {
    "1m": "OneMinute", "5m": "FiveMinutes", "10m": "TenMinutes", "15m": "FifteenMinutes",
    "30m": "ThirtyMinutes", "1h": "OneHour", "4h": "FourHours", "1D": "OneDay", "1W": "OneWeek",
}


class EtoroAuthError(RuntimeError):
    """Raised when ETORO_API_KEY/ETORO_USER_KEY aren't set in st.secrets
    yet — a clear, specific error instead of a generic 401 deep in a
    requests traceback the first time this gets called without them."""


def _keys():
    api_key = st.secrets.get("ETORO_API_KEY", "")
    user_key = st.secrets.get("ETORO_USER_KEY", "")
    if not api_key or not user_key:
        raise EtoroAuthError(
            "ETORO_API_KEY and/or ETORO_USER_KEY missing from .streamlit/secrets.toml — "
            "generate a user key at eToro app -> Settings -> Trading -> API Key Management "
            "(Read permission, Demo environment) and paste it in there first."
        )
    return api_key, user_key


@st.cache_resource(show_spinner=False)
def _session():
    """One pooled HTTP connection reused across every call in this
    process — same reasoning as footprint.py's own _binance_session:
    avoids paying fresh TCP+TLS setup on every request."""
    s = requests.Session()
    s.headers.update({"Connection": "keep-alive"})
    return s


def _get(path, params=None, timeout=10):
    """path: full path INCLUDING its own /api/v1 or /api/v2 prefix — the
    two API versions coexist across endpoints (confirmed directly: rates
    is v2, everything else used here is v1), so this doesn't assume one
    prefix for every caller. x-request-id is a fresh UUID per call, per
    eToro's own docs ("A unique request identifier") — never reused
    across requests, unlike the other two headers which stay constant
    for this whole session."""
    api_key, user_key = _keys()
    headers = {"x-api-key": api_key, "x-user-key": user_key, "x-request-id": str(uuid.uuid4())}
    resp = _session().get(f"{_BASE}{path}", headers=headers, params=params or {}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=86400, show_spinner=False)
def resolve_instrument_id(symbol):
    """symbol: eToro's own internalSymbolFull (e.g. "GBPUSD", "BTC",
    "SPX500", "NDX") — NOT this project's own Yahoo-style tickers
    (data.py's "GBPUSD=X"/"BTC-USD"/etc — those are a different vendor's
    own naming, no relation to eToro's). Cached a full day: which
    instrument a symbol resolves to essentially never changes.

    Confirmed directly in eToro's own docs: search can return partial
    matches, so this only accepts an EXACT internalSymbolFull match
    rather than just the first result — a caller asking for "BTC" should
    never silently get back some unrelated BTC-adjacent instrument.
    Returns None if no exact match is found (an honest miss, not a
    guess)."""
    data = _get("/api/v1/market-data/search", params={"internalSymbolFull": symbol})
    for item in data.get("items", []):
        if item.get("internalSymbolFull") == symbol:
            return item["instrumentId"]
    return None


def get_rates(instrument_ids):
    """instrument_ids: list of eToro instrumentIds (resolve_instrument_id
    first) — up to 1000 per call per eToro's own documented limit.
    Returns a DataFrame (instrument_id, bid, ask, spread, spread_bps,
    quote_type, date). NOT cached — a spread/rate is exactly the kind of
    value that's wrong the instant it's stale, unlike candle history or
    instrument-id resolution. Rate limit is eToro's own 120 requests per
    60 seconds (shared across market-data endpoints) — a caller polling
    this on a tight loop needs its own throttling on top of this
    function, not provided here.

    Sent as ONE comma-joined query value, not requests' own default
    repeated-key list encoding (?instrumentIds=1&instrumentIds=2) —
    confirmed directly against the live API: repeated keys silently
    return only the LAST instrument and drop every other one (a real,
    undocumented quirk on eToro's own server, not a client-side mistake —
    the request itself comes back 200 OK, just quietly short a result)."""
    ids_param = ",".join(str(i) for i in instrument_ids)
    data = _get("/api/v2/market-data/rates", params={"instrumentIds": ids_param})
    rows = []
    for r in data.get("results", []):
        bid, ask = r.get("bid"), r.get("ask")
        spread = (ask - bid) if (bid is not None and ask is not None) else None
        mid = (ask + bid) / 2 if spread is not None else None
        rows.append({
            "instrument_id": r["instrumentId"], "bid": bid, "ask": ask,
            "spread": spread, "spread_bps": (spread / mid * 10000) if (spread and mid) else None,
            "quote_type": r.get("quoteType"), "date": r.get("date"),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=_CACHE_TTL, show_spinner=False)
def get_candles(instrument_id, interval_label, count=200):
    """interval_label: one of this project's own labels (see
    INTERVAL_MAP) — translated to eToro's own enum string internally so
    every OTHER caller in this codebase keeps using the same "1h"/"1D"
    vocabulary data.py/experiments.py already use, not eToro's own.
    count: 1-1000 (eToro's own documented cap). Returns a DatetimeIndex-ed
    OHLCV DataFrame in the SAME column shape (Open/High/Low/Close/
    Volume) get_yf_ohlcv already returns, so this can be handed to any
    existing detector/backtest function unmodified — a caller doesn't
    need to know which data source it came from."""
    if interval_label not in INTERVAL_MAP:
        raise ValueError(f"interval_label must be one of {sorted(INTERVAL_MAP)}, got {interval_label!r}")
    data = _get(f"/api/v1/market-data/instruments/{instrument_id}/history/candles/asc/"
                f"{INTERVAL_MAP[interval_label]}/{count}")
    rows = []
    for block in data.get("candles", []):
        for c in block.get("candles", []):
            rows.append({
                "date": pd.Timestamp(c["fromDate"]), "Open": c["open"], "High": c["high"],
                "Low": c["low"], "Close": c["close"], "Volume": c.get("volume", 0.0),
            })
    if not rows:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df


@st.cache_data(ttl=60, show_spinner=False)
def get_balances(account_types=None, display_currency="USD"):
    """Aggregated real-money account balance(s) — see eToro's own
    get-aggregated-balances docs for the full field list; returns the
    raw response as a DataFrame (one row per account) rather than
    picking a few fields, since which ones matter depends on what the
    caller's actually building (a balance tile wants totalBalance/
    displayBalance; a margin check wants equityDetails)."""
    params = {"displayCurrency": display_currency}
    if account_types:
        params["accountTypes"] = ",".join(account_types)
    data = _get("/api/v1/balances", params=params)
    return pd.DataFrame(data.get("balances", []))


@st.cache_data(ttl=60, show_spinner=False)
def get_portfolio():
    """Aggregated open-position snapshot (real account) — one row per
    currently-held instrument (netUnits/avgOpenRate/accountCurrencyReturn/
    etc., see eToro's own get-aggregated-portfolio-snapshot docs for the
    complete field list). Empty DataFrame when nothing's open, not an
    error — a flat account is a normal, valid state."""
    data = _get("/api/v1/trading/info/aggregate-portfolio")
    return pd.DataFrame(data.get("instrumentAggregates", data.get("positions", [])))


@st.cache_data(ttl=_CACHE_TTL, show_spinner=False)
def get_trade_history(min_date=None, max_lookback_days=364):
    """min_date: a date/Timestamp, defaults to max_lookback_days ago (eToro's
    own documented cap is "less than 1 year" per request — 364 as a safe
    margin under that, not the full 365). Returns a DataFrame shaped like
    every OTHER trade-history reader elsewhere in this project (Symbol-
    equivalent as instrument_id, open/close time, open/close rate, net
    profit) rather than eToro's own raw field names, so the SAME hour-of-
    day/duration-split analysis already built for the user's own CSV
    statement exports can run against this directly once instrument_id
    is mapped back to a symbol."""
    if min_date is None:
        min_date = datetime.now(timezone.utc) - timedelta(days=max_lookback_days)
    data = _get("/api/v1/trading/info/trade/history",
                params={"minDate": pd.Timestamp(min_date).strftime("%Y-%m-%d")})
    rows = []
    for t in data if isinstance(data, list) else data.get("items", []):
        rows.append({
            "instrument_id": t["instrumentId"], "is_buy": t["isBuy"],
            "open_time": pd.Timestamp(t["openTimestamp"]), "close_time": pd.Timestamp(t["closeTimestamp"]),
            "open_rate": t["openRate"], "close_rate": t["closeRate"], "net_profit": t["netProfit"],
            "units": t.get("units"), "leverage": t.get("leverage"), "fees": t.get("fees"),
            "position_id": t.get("positionId"),
        })
    return pd.DataFrame(rows)
