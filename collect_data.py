"""
Scheduled data collector — the actual fix for "the accumulating logs have
gaps whenever nobody's looking at the dashboard." institutional.py's
scan_watchlist()/get_market_cap_snapshot() and news.py's get_calendar()
all record to their own local history logs as a side effect of being
called — but Streamlit only calls them when someone actually loads that
page. Nothing runs on a day nobody visits. This script is the same calls,
decoupled from any page view, meant to run on a schedule via
.github/workflows/collect.yml — GitHub Actions runs it whether or not
anyone's looked at anything, then commits whatever changed back to the
repo.

Also archives 1-minute bars for the four index futures added to the
Markets dashboard (NQ=F/ES=F/YM=F/RTY=F) — Yahoo's own 1m history for
these is short-lived (a handful of days), so without an accumulating
archive, anything older than that is gone the moment Yahoo ages it out.
Same accumulate-going-forward, honest-empty-until-it-isn't convention as
every other history log in this project (see institutional.py's own
docstring for why that's the only honest option when a feed has no way
to backfill) — this is that same idea already solved once for FX 1-minute
data by research/histdata_import.py, applied here to Yahoo instead.

Run directly: python3 collect_data.py
"""
import os

import pandas as pd

import institutional
import news
from data import get_yf_ohlcv

_INTRADAY_TICKERS = ["NQ=F", "ES=F", "YM=F", "RTY=F"]
_ARCHIVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "intraday_archive")


def _archive_intraday_bars(ticker):
    """Appends this run's freshly-fetched 1-minute bars to a persistent
    per-ticker archive, deduped by timestamp (keeping the freshest fetch
    of any bar the archive already had, in case Yahoo revised it) — the
    only way this project ever accumulates more than Yahoo's own few-day
    1m retention window. Returns how many genuinely new bars were added,
    or None on a fetch/write failure (best-effort, same convention as
    every other cache write in this project — a failed run just means
    this ticker's archive doesn't grow this cycle, not a crash)."""
    path = os.path.join(_ARCHIVE_DIR, f"{ticker.replace('=', '_')}_1m.parquet")
    try:
        fresh = get_yf_ohlcv(ticker, period="5d", interval="1m")
        if fresh.empty:
            return 0
        os.makedirs(_ARCHIVE_DIR, exist_ok=True)
        try:
            existing = pd.read_parquet(path)
        except Exception:
            existing = pd.DataFrame()
        combined = pd.concat([existing, fresh]) if not existing.empty else fresh.copy()
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        combined.to_parquet(path)
        return len(combined) - len(existing)
    except Exception:
        return None


def run():
    """Runs every collection step, independently — one feed being down
    doesn't skip the others. Returns a {step: status} report, printed by
    __main__ and useful in the Actions log to see exactly what happened
    on a given run without needing to dig through raw output."""
    results = {}

    try:
        bias = institutional.scan_watchlist()
        results["institutional_bias"] = f"{len(bias)} tickers scored"
    except Exception as e:
        results["institutional_bias"] = f"FAILED: {e}"

    try:
        snap = institutional.get_market_cap_snapshot()
        results["market_cap"] = "recorded" if snap is not None else "FAILED (feed unreachable)"
    except Exception as e:
        results["market_cap"] = f"FAILED: {e}"

    try:
        cal = news.get_calendar()
        results["news_calendar"] = f"{len(cal)} events this week"
    except Exception as e:
        results["news_calendar"] = f"FAILED: {e}"

    for ticker in _INTRADAY_TICKERS:
        added = _archive_intraday_bars(ticker)
        results[f"intraday_{ticker}"] = f"+{added} new bars" if added is not None else "FAILED"

    return results


if __name__ == "__main__":
    for name, status in run().items():
        print(f"{name}: {status}")
