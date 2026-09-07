import concurrent.futures
import os
import time

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# Free-tier fallback providers, used only when Yahoo Finance itself fails or
# returns nothing (rate-limited, transient outage, or a ticker it doesn't
# recognize). Binance needs no signup at all. Twelve Data and Tiingo each
# need a free API key (twelvedata.com / tiingo.com) — put them in
# .streamlit/secrets.toml (Streamlit's own convention, auto-loaded on every
# run, gitignored by default) as:
#   TWELVE_DATA_API_KEY = "..."
#   TIINGO_API_KEY = "..."
# An env var (export TWELVE_DATA_API_KEY=...) works too and takes priority,
# handy for a one-off shell test. Left unset either way, that provider is
# silently skipped in the chain rather than erroring, so plain Yahoo-only
# behavior (today's default) is unaffected.
def _secret(key):
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""  # no secrets.toml at all — st.secrets raises rather than returning empty


TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY") or _secret("TWELVE_DATA_API_KEY")
TIINGO_API_KEY = os.environ.get("TIINGO_API_KEY") or _secret("TIINGO_API_KEY")


def _retry(fn, attempts=2, base_delay=0.5):
    """Transient hiccups (rate-limits, network blips) happen with any of
    these providers, not just Yahoo — retry once with a short backoff
    before giving up, instead of failing on the first blip.

    Deliberately NOT 3 attempts with a longer backoff (this function's own
    original numbers) — confirmed directly as a real "clogs the system"
    bug: _fetch_any's own fallback chain already tries up to 4 providers
    per ticker, and each one that's genuinely down (not just blipping,
    e.g. Yahoo rate-limiting an IP that's made a lot of requests this
    session) pays its own full timeout on EVERY attempt, not just the
    first — 3 attempts at yfinance's own 10s default timeout is up to 30+
    seconds sunk into a SINGLE provider that was never going to answer,
    BEFORE this chain even reaches the next one. A live sweep over just
    20 combos was directly observed taking 8+ minutes from exactly this:
    a handful of ticker/timeframe combos each eating 60+ seconds cycling
    through 2 failing providers. The fallback chain itself is what
    provides real resilience here (a still-alive provider is one hop
    away); grinding a single already-failing one harder isn't worth what
    it costs everything else running in the same process at the same
    time — the sweep isn't the only thing paying this, ANY chart
    interaction that needs a fresh (uncached) fetch pays it too."""
    last_err = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if i < attempts - 1:
                time.sleep(base_delay * (2 ** i))
    raise last_err


def _is_crypto(ticker):
    """Yahoo's own convention for crypto pairs — BTC-USD, ETH-USD, SOL-USD —
    is unambiguous against real equity tickers here: the one real collision
    risk, share-class tickers like BRK-B, doesn't end in -USD."""
    return ticker.upper().endswith("-USD")


def _period_to_days(period):
    if period == "max":
        return 3650
    num, unit = int(period[:-1]), period[-1]
    return num * 365 if unit == "y" else num


# ---------------------------------------------------------------------------
# Provider 1 (primary): Yahoo Finance, via yfinance. Unchanged behavior from
# before the fallback chain existed.
# ---------------------------------------------------------------------------
# max_workers=20, not a small fixed number — confirmed directly as a real,
# self-inflicted bottleneck at 4: app.py/crypto_app.py's own _prefetch()
# already runs with max_workers=len(specs) (unbounded, scales to however
# many combos a single render actually needs — main chart, its backfill
# tier, both mini HTF/LTF panels' own AND cross-timeframe fetches, an
# active rule's own R:R/backtest fetch, routinely 6-8+ at once). Every one
# of those calls, if it goes through Yahoo, funnels into THIS executor for
# its hard-timeout wrapper — a small fixed pool here would silently
# SERIALIZE calls that used to run fully in parallel before this wrapper
# existed, adding real wall-clock delay on exactly the cold-render case
# this whole timeout mechanism was built to speed up. 20 comfortably
# covers any realistic peak for this app without becoming the bottleneck
# itself; still bounded (not truly unlimited) so a pathological case can't
# spawn an unbounded number of threads.
_YF_TIMEOUT_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=20, thread_name_prefix="yf-hard-timeout")


def _call_with_hard_timeout(fn, timeout):
    """Enforces `timeout` regardless of what `fn` does internally — needed
    specifically for yfinance, confirmed directly (read straight from its
    own installed source, yfinance.data.YfData) to NOT reliably honor the
    timeout= kwarg Ticker.history() itself accepts: that value only reaches
    the actual PRICE request. The cookie/crumb authentication step every
    call needs first (YfData._get_cookie_and_crumb, and the _csrf/_basic
    strategies underneath it) is called with NO timeout argument at all in
    that path, defaulting to its OWN hardcoded 30s — completely
    independent of whatever was passed to history(). Caught this live: a
    real TCP connection to Yahoo's own cookie endpoint sat ESTABLISHED for
    60+ seconds straight (2 of our own _retry attempts x up to 30s each)
    while CPU stayed pegged, despite this project already passing
    timeout=5 to history() — that 5s was never in a position to matter.
    Running the call in its own thread and bounding THAT with
    concurrent.futures' own timeout (a hard wall-clock deadline on OUR
    side, independent of what yfinance's internals choose to do) is what
    actually caps worst-case latency; the abandoned thread may keep
    running yfinance's own slow request to completion in the background,
    but nothing on our side waits for it past `timeout` seconds. Raises
    concurrent.futures.TimeoutError on expiry, which _retry (or whichever
    caller) catches the same as any other transient failure — no special
    handling needed downstream."""
    future = _YF_TIMEOUT_EXECUTOR.submit(fn)
    return future.result(timeout=timeout)


def _fetch_yahoo(ticker, period, interval):
    # prepost=True so pre-market/after-hours candles are actually present —
    # needed for the RTH/ETH kill-zone highlights to mean anything (without
    # it, only the regular session is ever loaded, so an "RTH" band would
    # just span the entire visible day, nothing to distinguish it from). A
    # no-op for daily+ intervals and 24/7 tickers (crypto), so always on
    # rather than threading a conditional through every call site.
    # actions=False — Dividends/Stock Splits columns yfinance adds by
    # default are never read anywhere in this codebase; skip fetching them.
    # timeout=5 is passed to history() too (down from yfinance's own 10s
    # default) but is NOT the real bound anymore — see
    # _call_with_hard_timeout's own docstring for why that kwarg alone
    # doesn't reliably cap worst-case latency. The actual enforcement is
    # the outer 5s wrapped around the whole call, regardless of which of
    # yfinance's own internal steps (cookie/crumb auth vs. the price
    # request itself) turns out to be the slow one this time.
    def _do_fetch():
        return yf.Ticker(ticker).history(period=period, interval=interval, prepost=True, actions=False, timeout=5)
    df = _call_with_hard_timeout(_do_fetch, timeout=5)
    if not df.empty:
        df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# Provider 2: Binance public API — crypto only, no key, very high free rate
# limit (~1200 request-weight/min). First fallback for -USD tickers
# specifically because it needs no signup and is the fastest/most complete
# of the three for crypto.
# ---------------------------------------------------------------------------
_BINANCE_INTERVAL = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "60m": "1h", "1d": "1d", "1wk": "1w", "1mo": "1M",
}
# Rough candles/day per interval, used only to scope how many candles to ask
# for — a generous (calendar-day, not trading-hours) estimate is fine since
# it only sets an upper bound, never a hard cutoff on real data.
_BINANCE_BARS_PER_DAY = {
    "1m": 1440, "5m": 288, "15m": 96, "30m": 48, "1h": 24, "1d": 1, "1w": 1, "1M": 1,
}


def _fetch_binance(ticker, period, interval):
    bnc_interval = _BINANCE_INTERVAL.get(interval)
    if not _is_crypto(ticker) or bnc_interval is None:
        return pd.DataFrame()
    symbol = ticker.upper().replace("-USD", "") + "USDT"  # BTC-USD -> BTCUSDT
    # Scoped to what `period` actually needs instead of always asking for
    # Binance's max — request weight jumps from 1 to 5 above a 100-candle
    # limit, so a short lookback (e.g. get_latest_bars' 2-day poll) was
    # paying 5x the weight for a handful of candles it was going to use.
    limit = min(1000, max(2, int(_period_to_days(period) * _BINANCE_BARS_PER_DAY.get(bnc_interval, 24)) + 5))
    # timeout=5 — see _fetch_yahoo's own comment on why this shrank from
    # 10s; Binance is normally very fast, so this only ever bites when
    # it's genuinely struggling, in which case the fallback chain moving
    # on sooner is strictly better than waiting twice as long to find out.
    resp = requests.get(
        "https://api.binance.com/api/v3/klines",
        params={"symbol": symbol, "interval": bnc_interval, "limit": limit},
        timeout=5,
    )
    resp.raise_for_status()
    rows = resp.json()
    if not rows or isinstance(rows, dict):  # dict shape here means an error payload
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[
        "open_time", "Open", "High", "Low", "Close", "Volume",
        "close_time", "quote_vol", "trades", "taker_base", "taker_quote", "ignore",
    ])
    # utc=True — yfinance's own DatetimeIndex is always tz-aware (kill-zone
    # bands and the chart's own NY-wall-clock axis math both call
    # tz_convert() on it), so a tz-naive index from any fallback provider
    # would either crash there or, worse, silently plot at the wrong hour.
    # Binance's open_time is a genuine UTC ms epoch, so this is exact, not
    # an approximation.
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    # `limit` above still caps out at Binance's own max of 1000 candles for
    # a deep `period` — no pagination here, since this is a last-resort
    # fallback, not a like-for-like replacement for Yahoo's own deep history.
    return df.set_index("date")[["Open", "High", "Low", "Close", "Volume"]].astype(float)


# ---------------------------------------------------------------------------
# Provider 3: Twelve Data — stocks + crypto, free key (800 calls/day,
# 8/min). Preferred fallback for STOCK tickers since it covers them at all
# (Binance doesn't); tried before Tiingo for crypto too since its daily
# quota is roomier than Tiingo's hourly one.
# ---------------------------------------------------------------------------
_TWELVEDATA_INTERVAL = {
    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
    "60m": "1h", "1d": "1day", "1wk": "1week", "1mo": "1month",
}
# Same rough candles/day estimate as Binance's, used the same way — an
# upper bound on how many candles `period` could plausibly need.
_TWELVEDATA_BARS_PER_DAY = {
    "1min": 1440, "5min": 288, "15min": 96, "30min": 48,
    "1h": 24, "1day": 1, "1week": 1, "1month": 1,
}


def _fetch_twelvedata(ticker, period, interval):
    if not TWELVE_DATA_API_KEY:
        return pd.DataFrame()
    td_interval = _TWELVEDATA_INTERVAL.get(interval)
    if td_interval is None:
        return pd.DataFrame()
    symbol = ticker.replace("-USD", "/USD") if _is_crypto(ticker) else ticker
    # Scoped to what `period` actually needs instead of always requesting
    # Twelve Data's hard max (5000) — this provider only gets hit when
    # Yahoo's already down/rate-limited, exactly the moment a slower,
    # quota-hungrier request than necessary hurts most (800 calls/day,
    # 8/min free-tier limit).
    outputsize = min(5000, max(50, int(_period_to_days(period) * _TWELVEDATA_BARS_PER_DAY.get(td_interval, 24)) + 10))
    resp = requests.get(
        "https://api.twelvedata.com/time_series",
        # timezone=UTC is load-bearing, not cosmetic: confirmed directly
        # that Twelve Data's default is the *exchange's* local time for
        # intraday bars (e.g. NASDAQ -> America/New_York), not UTC — left
        # off, an intraday fetch would come back 4-5 wall-clock hours off
        # from Yahoo's and the other two providers' data.
        params={"symbol": symbol, "interval": td_interval, "outputsize": outputsize,
                "timezone": "UTC", "apikey": TWELVE_DATA_API_KEY},
        timeout=10,
    )
    resp.raise_for_status()
    values = resp.json().get("values")
    if not values:
        return pd.DataFrame()
    df = pd.DataFrame(values)
    # The API returns naive "YYYY-MM-DD HH:MM:SS" strings even with
    # timezone=UTC (no offset in the string itself) — tz_localize tags what
    # the request already guaranteed is UTC, same reasoning as Binance above.
    df["date"] = pd.to_datetime(df["datetime"]).dt.tz_localize("UTC")
    df = df.set_index("date")[["open", "high", "low", "close", "volume"]].astype(float)
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    return df.sort_index()  # Twelve Data returns newest-first


# ---------------------------------------------------------------------------
# Provider 4: Tiingo — stocks (EOD only on the free tier) + crypto, free key
# (1000 calls/day, 50/hour). Tried last: deepest history of the three but
# the tightest per-hour limit, and no reliable free intraday for stocks.
# ---------------------------------------------------------------------------
_TIINGO_CRYPTO_FREQ = {
    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "60m": "1hour", "1d": "1day",
}
_TIINGO_STOCK_RESAMPLE = {"1wk": "weekly", "1mo": "monthly"}  # "1d" needs no resampleFreq


def _fetch_tiingo(ticker, period, interval):
    if not TIINGO_API_KEY:
        return pd.DataFrame()
    start_date = (pd.Timestamp.utcnow() - pd.Timedelta(days=_period_to_days(period))).strftime("%Y-%m-%d")
    if _is_crypto(ticker):
        freq = _TIINGO_CRYPTO_FREQ.get(interval)
        if freq is None:
            return pd.DataFrame()
        symbol = ticker.upper().replace("-USD", "").lower() + "usd"  # BTC-USD -> btcusd
        resp = requests.get(
            "https://api.tiingo.com/tiingo/crypto/prices",
            params={"tickers": symbol, "startDate": start_date, "resampleFreq": freq, "token": TIINGO_API_KEY},
            timeout=10,
        )
        resp.raise_for_status()
        payload = resp.json()
        if not payload or not payload[0].get("priceData"):
            return pd.DataFrame()
        df = pd.DataFrame(payload[0]["priceData"])
    else:
        # Free-tier intraday needs a realtime add-on Tiingo doesn't grant for
        # free accounts — only daily+ is reliable here, so anything finer
        # skips this provider entirely rather than risking stale/empty data.
        if interval not in ("1d", "1wk", "1mo"):
            return pd.DataFrame()
        params = {"token": TIINGO_API_KEY, "startDate": start_date, "format": "json"}
        if interval in _TIINGO_STOCK_RESAMPLE:
            params["resampleFreq"] = _TIINGO_STOCK_RESAMPLE[interval]
        resp = requests.get(f"https://api.tiingo.com/tiingo/daily/{ticker.lower()}/prices",
                             params=params, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        if not payload:
            return pd.DataFrame()
        df = pd.DataFrame(payload)
    # Tiingo's own `date` field already carries an explicit UTC offset
    # ("...+00:00" for crypto, "...Z" for daily EOD) — pd.to_datetime picks
    # that up and returns a tz-aware (UTC) result on its own, unlike the
    # other two providers, so there's nothing to localize here.
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")[["open", "high", "low", "close", "volume"]].astype(float)
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    return df.sort_index()


def _provider_chain(ticker):
    """Yahoo always leads (it's the richest/fastest source when it's up) —
    Auto mode falls through the rest of this list only once it's failed or
    come back empty. Crypto tries the no-signup, no-limit Binance next;
    stocks skip straight to the two key-based providers since Binance only
    knows crypto pairs."""
    chain = [("yahoo", _fetch_yahoo)]
    if _is_crypto(ticker):
        chain += [("binance", _fetch_binance), ("twelvedata", _fetch_twelvedata), ("tiingo", _fetch_tiingo)]
    else:
        chain += [("twelvedata", _fetch_twelvedata), ("tiingo", _fetch_tiingo)]
    return chain


def _fetch_any(ticker, period, interval, provider="auto"):
    """provider="auto" is the fallback chain above. Any other value FORCES
    that single named provider — no fallback if it fails, since the whole
    point of picking one by hand (app.py's "Data source" setting) is to see
    exactly what that source gives you, not to quietly get Yahoo's data
    again under a different label."""
    chain = _provider_chain(ticker)
    if provider != "auto":
        chain = [(name, fn) for name, fn in chain if name == provider]
    for name, fetch_fn in chain:
        try:
            df = _retry(lambda fn=fetch_fn: fn(ticker, period, interval))
        except Exception:
            continue  # this provider's out (no key set, rate-limited, unknown symbol...) — try the next
        if not df.empty:
            df = df.copy()
            df.attrs["provider"] = name
            return df
    return pd.DataFrame()


# st.cache_data above is in-memory only — gone the moment the process
# restarts, which during a dev session (or just closing the laptop
# overnight) means re-paying the full historical fetch from scratch on
# every single restart. This second layer sits on disk, one small parquet
# file per (ticker, period, interval, provider) — same key shape as
# st.cache_data's own, deliberately: e.g. "60m" interval gets fetched with
# 3 different periods depending on caller (main chart's "1h" TF asks for
# 180d, its "4h" TF for 730d, ref_frames for 30d) — collapsing those into
# one cache entry would silently hand a 30-day request 730 days of data or
# vice versa. Costs a little cross-period reuse (a 730d fetch could in
# principle satisfy a 30d one) but that's a much smaller loss than a
# wrong-length dataset.
#
# A generous TTL (1h) is safe here specifically because it's ONLY used for
# get_yf_ohlcv's deep/base fetch, never get_latest_bars' fast poll — the
# live-splice logic in app.py already patches the last bar with fresh data
# from that separate 8s-ttl call regardless of how stale the base history
# underneath it is. Historical candles don't change after the fact, so an
# hour-old copy of everything except the current bar is exactly as correct
# as a fresh one, just free.
_DISK_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache", "ohlcv")
_DISK_CACHE_MAX_AGE = 3600


def _disk_cache_path(ticker, period, interval, provider):
    safe_ticker = ticker.upper().replace("/", "_")
    return os.path.join(_DISK_CACHE_DIR, f"{safe_ticker}_{period}_{interval}_{provider}.parquet")


def _disk_cache_read(path):
    try:
        if time.time() - os.path.getmtime(path) > _DISK_CACHE_MAX_AGE:
            return None
        df = pd.read_parquet(path)
    except Exception:
        return None  # missing, corrupt, unreadable — treat exactly like a cache miss
    provider_used = df.attrs.get("provider")
    if "_provider" in df.columns:
        provider_used = df["_provider"].iloc[0] if len(df) else provider_used
        df = df.drop(columns=["_provider"])
    df.attrs["provider"] = provider_used
    return df


def _disk_cache_write(path, df):
    try:
        os.makedirs(_DISK_CACHE_DIR, exist_ok=True)
        tagged = df.copy()
        tagged["_provider"] = df.attrs.get("provider", "")
        tagged.to_parquet(path)
    except Exception:
        pass  # a failed write just means the next restart pays full price again — not worth surfacing


def _disk_cache_read_stale(path):
    """Same file, same parsing as _disk_cache_read, but ignoring
    _DISK_CACHE_MAX_AGE entirely — used only as the BASE for an incremental
    top-up (see _incremental_topup below), never returned to a caller
    directly. A cache past its 1h TTL is still byte-correct for every bar
    except whatever's happened since it was written; throwing the whole
    thing away and re-fetching the full `period` from scratch (what
    happened before this existed) redoes 100% of the work to pick up the
    last few bars. Returns None on anything missing/corrupt, same as
    _disk_cache_read — this is a "maybe helps" path, never a hard
    dependency."""
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    if df.empty:
        return None
    provider_used = df.attrs.get("provider")
    if "_provider" in df.columns:
        provider_used = df["_provider"].iloc[0] if len(df) else provider_used
        df = df.drop(columns=["_provider"])
    df.attrs["provider"] = provider_used
    return df


def _incremental_topup(ticker, period, interval, provider, stale_df):
    """Extends `stale_df` (a disk cache past its 1h TTL) with whatever's
    new since its own last bar, instead of redoing the full `period` fetch
    from scratch. A provider's own history endpoint costs roughly the same
    wall-clock time per bar scanned server-side regardless of how much of
    it we actually needed — re-pulling 60 days to pick up the last hour's
    worth of 5m bars wastes almost all of that time re-fetching rows that
    hadn't changed. Returns None (never raises) on anything that makes a
    clean top-up unsafe; the caller falls through to the ordinary full
    fetch in that case, so this can only ever help, never introduce a new
    failure mode.

    "max" is deliberately excluded: it isn't a rolling window relative to
    now the way "7d"/"60d"/"2y" are, so there's no safe `cutoff` to trim
    a top-up back to "the same thing a fresh max fetch would give" —
    trimming to _period_to_days("max")'s own 3650-day placeholder could
    silently cut off genuinely older history a real max fetch would have
    included (BTC-USD, or a forex major with decades on file)."""
    if period == "max":
        return None
    try:
        last_ts = stale_df.index[-1]
        full_days = _period_to_days(period)
        # +2 days of overlap, not just "since last_ts" exactly — covers
        # weekend/holiday gaps (a stock/forex ticker's last bar could be a
        # Friday close) and any clock skew between here and the provider,
        # without materially changing the request's own size.
        gap_days = max(1, (pd.Timestamp.now(tz=last_ts.tz) - last_ts).days + 2)
        if gap_days >= full_days:
            return None  # stale copy doesn't save anything here — same size as a full fetch
        fresh = _fetch_any(ticker, f"{gap_days}d", interval, provider)
        if fresh.empty:
            return None
        if fresh.attrs.get("provider") != stale_df.attrs.get("provider"):
            # A different vendor answered this time (e.g. Yahoo was down
            # when `fresh` was fetched, still up when `stale_df` was) —
            # splicing two vendors' own close/volume conventions into one
            # series risks a visible discontinuity right at the seam.
            # Falling back to an ordinary full fetch (single vendor,
            # consistent throughout) is worth more than the time saved.
            return None
        # Yahoo's own short-window endpoint doesn't revise bars that
        # already closed (confirmed directly: two fetches 9s apart
        # returned byte-identical values for every already-closed row) —
        # so simply preferring `fresh` for any timestamp both sides share
        # is safe, not just "probably fine."
        merged = pd.concat([stale_df[stale_df.index < fresh.index[0]], fresh])
        merged.attrs["provider"] = fresh.attrs["provider"]
        # Trim back to the ORIGINALLY requested period — a caller asking
        # for period="60d" must keep getting ~60 days back, not however
        # much extra history `stale_df` happened to be carrying.
        cutoff = pd.Timestamp.now(tz=merged.index.tz) - pd.Timedelta(days=full_days)
        merged = merged[merged.index >= cutoff]
        return merged if not merged.empty else None
    except Exception:
        return None


@st.cache_data(ttl=60)
def get_yf_ohlcv(ticker, period="6mo", interval="1d", provider="auto"):
    # No st.spinner wrapper here — theme.py hides Streamlit's cache spinner
    # entirely already, so it bought nothing visible, and dropping it makes
    # this function pure Python + yfinance with zero Streamlit API calls,
    # which is what lets app.py's _prefetch() call it from a plain
    # ThreadPoolExecutor thread with no ScriptRunContext to worry about.
    path = _disk_cache_path(ticker, period, interval, provider)
    cached = _disk_cache_read(path)
    if cached is not None and not cached.empty:
        return cached
    # The fresh (within-TTL) disk cache above missed — either this exact
    # combo's never been fetched, or it has but the hour's up. Try
    # extending whatever's on disk regardless of age before paying for a
    # full re-fetch; see _incremental_topup's own docstring for why that's
    # both faster and lighter on the provider's rate limit. Any failure
    # here (missing file, corrupt, provider mismatch, "max" period, gap
    # too big to bother) falls straight through to the full fetch below —
    # this path only ever makes a cold cache faster, never a new way to fail.
    stale = _disk_cache_read_stale(path)
    if stale is not None:
        topped_up = _incremental_topup(ticker, period, interval, provider, stale)
        if topped_up is not None:
            _disk_cache_write(path, topped_up)
            return topped_up
    df = _fetch_any(ticker, period, interval, provider)
    if not df.empty:
        _disk_cache_write(path, df)
    return df


# Fetching the FULL period (e.g. 7d of 1m bars, for the 5-day ICT lookback)
# is the expensive part — confirmed by direct timing: 1.33s for 7d/1m vs
# 0.24s for a 2-day window. Most auto-refresh ticks only need to know
# whether the current/most recent bar has moved, not re-pull the whole
# window, so this covers that cheaply and often; the slow full fetch above
# can then run on a longer ttl (its historical rows barely change tick to
# tick) and app.py splices this one's freshest rows onto it.
@st.cache_data(ttl=8)
def get_latest_bars(ticker, interval, lookback="2d", provider="auto"):
    return _fetch_any(ticker, lookback, interval, provider)


@st.cache_data(ttl=60)
def resample_ohlc(df, rule):
    """Yahoo doesn't offer every candle size natively (no 4h/1y) — build them
    from the nearest native interval instead of silently substituting a
    different timeframe than what was actually asked for.

    Cached — confirmed directly as a real, previously-uncached cost:
    every caller re-resamples from scratch on every call regardless of
    whether the underlying data changed, and for a resample-needing
    timeframe (4h from 730 days of hourly data, ~17,500 rows) that's real
    work, not a rounding error. The main chart and both mini HTF/LTF
    panels all call this once per fragment tick — up to a few times a
    second combined — while its own input (a resample-needing
    timeframe's raw df skips the live-splice/forming-bar mutation
    entirely, see _render_chart/_render_mini_chart's own "if resample is
    None" gate) is only genuinely fresh once every 60s, bounded by
    get_yf_ohlcv's own cache TTL — matched here so this never serves data
    staler than the fetch it's built from, while skipping every
    redundant re-resample of the identical frame in between. Pure,
    deterministic, no side effects — safe to cache with zero behavior
    change for any caller."""
    cols = ("Open", "High", "Low", "Close", "Volume") if "Close" in df else ("open", "high", "low", "close", "volume")
    o, h, l, c, v = cols
    agg = {o: "first", h: "max", l: "min", c: "last"}
    if v in df.columns:
        agg[v] = "sum"
    out = df.resample(rule).agg(agg)
    return out.dropna(subset=[o, c])


# ---------------------------------------------------------------------------
# Ticker universe — real symbol lists for the ticker search box, not OHLCV
# candles. Cached a full day: exchange listings barely change hour to hour,
# no reason to pay a fresh request every rerun the way price data does.
# ---------------------------------------------------------------------------
@st.cache_data(ttl=86400)
def get_crypto_universe():
    """Every live USDT spot pair on Binance, symbol converted to Yahoo's
    own BTC-USD convention so it drops straight into the same ticker field
    everything else in the app already expects. No key needed."""
    try:
        resp = requests.get("https://api.binance.com/api/v3/exchangeInfo", timeout=15)
        resp.raise_for_status()
        symbols = resp.json().get("symbols", [])
    except Exception:
        return []
    return [
        (s["baseAsset"] + "-USD", s["baseAsset"])
        for s in symbols
        if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING" and s.get("isSpotTradingAllowed")
    ]


@st.cache_data(ttl=3600)
def is_ticker_alive(ticker):
    """Cheapest possible "does this actually chart" check for the ticker
    menu's red/green status dot — 5 days of daily bars through the exact
    same Yahoo-then-fallback chain everything else uses (so the dot means
    what it says: this is what would happen if you picked it), not a
    separate lighter-weight validity check that could disagree with the
    real fetch. Cached an hour since a symbol's tradability doesn't flip
    minute to minute, and the menu only ever calls this for whatever's
    currently visible after filtering (see app.py) — never the full list."""
    try:
        return not get_yf_ohlcv(ticker, period="5d", interval="1d").empty
    except Exception:
        return False
