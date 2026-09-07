"""
Historical OHLCV loader for the research pipeline — deliberately separate
from data.py's get_yf_ohlcv, not a duplicate of it. That one is tuned for the
LIVE dashboard: st.cache_data with a 60s TTL, meant to track a chart that's
being watched right now. Backtesting wants the opposite property — the same
request should return the exact same bars today and next month (closed
historical candles don't change), and this needs to run as a plain script/
cron job with no Streamlit runtime at all, not just "callable from a thread"
the way get_yf_ohlcv's own docstring notes it tolerates.

Reuses data.py's _fetch_any (provider fallback chain: Yahoo -> Binance ->
Twelve Data -> Tiingo) rather than reimplementing it — that logic has
nothing Streamlit-specific in it, it's just not cached the way this needs.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import _fetch_any  # noqa: E402

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")

# yfinance's own interval strings (what data.py's provider chain actually
# passes through) — "60m" not "1h", "1wk"/"1mo" not "1w"/"1M". Each maps to
# the longest period the source will actually serve at that granularity;
# asking for more just errors or silently truncates, so this is the real
# ceiling to interrogate up to, not a guess. Matches the same limits
# app.py's own TIMEFRAMES dict encodes for the live chart. Shared here
# (not just in research_app.py) so sweep.py and any other script use the
# exact same ceilings instead of a second, driftable copy.
INTERVAL_MAX_PERIOD = {
    "1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d", "60m": "730d", "1d": "max",
    # Weekly/monthly/quarterly were missing entirely, which silently fell
    # back to app.py's own default of "60d" for these intervals wherever a
    # caller did INTERVAL_MAX_PERIOD.get(interval, "60d") (see app.py's
    # _historical_win_rates) — 60 days of "1wk" bars is a dozen candles,
    # nowhere near enough to ever clear a minimum-event-count threshold, so
    # historical win rates silently never showed for any 1W/1M/1Y layer.
    "1wk": "max", "1mo": "max", "3mo": "max",
}
INTERVAL_LABELS = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "60m": "1h", "1d": "1d",
                    "1wk": "1W", "1mo": "1M", "3mo": "3M"}


def _cache_path(ticker, period, interval, provider):
    safe_ticker = ticker.replace("/", "_")
    return os.path.join(_CACHE_DIR, f"{safe_ticker}_{period}_{interval}_{provider}.parquet")


def load_history(ticker, period, interval, provider="auto", refresh=False):
    """Returns a DatetimeIndex-ed OHLCV DataFrame. Cached to disk indefinitely
    once fetched — the file itself is never deleted or TTL-expired, unlike
    the live dashboard's cache. But "kept forever" and "covers all of
    history" are NOT the same thing, and it's worth being explicit about
    which one this gives you:

    Sub-daily intervals have a hard ceiling on how far back the SOURCE will
    ever serve (Yahoo: ~7d at 1m, ~60d at 5m/15m/30m, ~730d at 60m — see
    research_app.py's INTERVAL_MAX_PERIOD). A single fetch can never return
    more than that, no matter what `period` asks for. Left as plain
    overwrite-on-refresh, calling this again after those bars have aged out
    of the source's own window would silently DROP the earlier days
    forever — the file wouldn't grow, it'd just slide.

    So a refresh MERGES the newly fetched window onto whatever's already on
    disk (newest values win on any overlap, since a bar Yahoo returns twice
    might have been revised) instead of replacing it — call this with
    refresh=True on some regular cadence (daily/weekly) and the file
    actually accumulates a longer history than any single API call could
    ever return, which is the only way to build a multi-month-plus 1m/5m/
    15m archive from a free source at all. A first-ever fetch (no existing
    file) has nothing to merge with, so it's just whatever the source gives."""
    path = _cache_path(ticker, period, interval, provider)
    if not refresh and os.path.exists(path):
        return pd.read_parquet(path)

    df = _fetch_any(ticker, period, interval, provider)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} ({period}/{interval}, provider={provider})")

    if os.path.exists(path):
        existing = pd.read_parquet(path)
        # concat then drop_duplicates(keep="last") — the fresh fetch's rows
        # sort after the existing ones, so an overlapping timestamp keeps
        # the JUST-fetched value rather than the stale cached one.
        df = pd.concat([existing, df]).sort_index()
        df = df[~df.index.duplicated(keep="last")]

    os.makedirs(_CACHE_DIR, exist_ok=True)
    df.to_parquet(path)
    return df
