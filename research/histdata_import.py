"""
Imports free FX historical data from histdata.com into the SAME disk cache
data_loader.load_history() reads — once imported, every other module
(events.py, backtest.py, live_scan.py, research_app.py) can use it exactly
like a Yahoo/Binance-fetched ticker, no special-casing anywhere else in the
pipeline. Real value here: Yahoo caps intraday FX history hard (7d at 1m,
730d at 60m — INTERVAL_MAX_PERIOD); histdata.com has clean 1-minute bars
back to the early 2000s for major pairs, for free.

Uses the `histdata` PyPI package (GET the pair/year page, scrape the hidden
form token, POST it to /get.php) rather than hand-rolling that dance —
confirmed directly against the live site before relying on it: the token
flow matched exactly what the package implements, and a real download
(EURUSD 2024, 1-minute, ~372k rows) came back clean.

NOT a live/refreshable source the way Yahoo/Binance are: histdata.com ships
whole-year archives for past years (or whole-month for the current year in
progress), not a "give me the latest N days" API. This is a one-shot/
periodic BACKFILL tool run by hand when you want deeper history for a pair
— live_scan.py's hourly schedule has no reason to call it and doesn't.

Timestamp handling: histdata.com's files are Eastern Standard Time WITHOUT
daylight-saving adjustment, year-round (a documented quirk of theirs — a
fixed UTC-5, not "whatever US/Eastern happens to be that day"). Converted
to real UTC here so it's directly comparable to the UTC-aware Yahoo/Binance
data everywhere else in this pipeline.
"""

import io
import os
import zipfile

import pandas as pd
from histdata.api import Platform, TimeFrame, download_hist_data

from research.data_loader import _CACHE_DIR, _cache_path

HISTDATA_PROVIDER = "histdata"
HISTDATA_PERIOD = "histdata-full"  # the `period` value this data lives under in load_history()'s cache
_DOWNLOAD_DIR = os.path.join(_CACHE_DIR, "_histdata_raw")

# histdata.com's own pair codes -> this pipeline's ticker convention
# (Yahoo's EURUSD=X style, so it lines up with anything already fetched
# under that name elsewhere in the project rather than introducing a
# second naming scheme).
KNOWN_PAIRS = [
    "eurusd", "gbpusd", "usdjpy", "usdchf", "audusd", "usdcad", "nzdusd",
    "eurgbp", "eurjpy", "gbpjpy", "xauusd",
]


def _parse_zip(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        csv_name = next(n for n in z.namelist() if n.endswith(".csv"))
        raw = z.read(csv_name).decode()
    df = pd.read_csv(
        io.StringIO(raw), sep=";", header=None,
        names=["datetime", "Open", "High", "Low", "Close", "Volume"],
    )
    naive = pd.to_datetime(df.pop("datetime"), format="%Y%m%d %H%M%S")
    # Etc/GMT+5 is POSIX-backwards (it means UTC-5, fixed, no DST) — exactly
    # histdata's own fixed-EST convention, not a real "US/Eastern" tz name
    # which would incorrectly shift with DST.
    df.index = naive.dt.tz_localize("Etc/GMT+5").dt.tz_convert("UTC")
    df.index.name = "date"
    return df[["Open", "High", "Low", "Close", "Volume"]]


def import_year(pair, year, ticker=None):
    """pair: histdata's own code, e.g. 'eurusd'. ticker: this pipeline's
    ticker string for the cache key — defaults to Yahoo's EURUSD=X
    convention. Downloads (or re-downloads if called again), parses, and
    MERGES into the interval="1m" cache the same way
    data_loader.load_history(..., refresh=True) does — importing multiple
    years accumulates rather than overwrites. Returns (ticker, total bars
    now cached for it)."""
    ticker = ticker or f"{pair.upper()}=X"
    os.makedirs(_DOWNLOAD_DIR, exist_ok=True)
    zip_path = download_hist_data(year=str(year), month=None, pair=pair,
                                   time_frame=TimeFrame.ONE_MINUTE, platform=Platform.GENERIC_ASCII,
                                   output_directory=_DOWNLOAD_DIR, verbose=False)
    df = _parse_zip(zip_path)

    path = _cache_path(ticker, HISTDATA_PERIOD, "1m", HISTDATA_PROVIDER)
    if os.path.exists(path):
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df]).sort_index()
        df = df[~df.index.duplicated(keep="last")]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path)
    return ticker, len(df)


def import_years(pair, start_year, end_year, ticker=None):
    """Convenience loop — import_year() for every year in [start_year, end_year]."""
    for year in range(start_year, end_year + 1):
        ticker, n = import_year(pair, year, ticker=ticker)
        print(f"  {pair.upper()} {year}: cache now has {n} bars total")
    return ticker


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Backfill FX history from histdata.com into the research cache.")
    p.add_argument("--pair", default="eurusd", help=f"One of: {', '.join(KNOWN_PAIRS)} (or any histdata.com pair code)")
    p.add_argument("--start-year", type=int, required=True)
    p.add_argument("--end-year", type=int, required=True)
    args = p.parse_args()

    print(f"Importing {args.pair.upper()} {args.start_year}-{args.end_year} from histdata.com...")
    ticker = import_years(args.pair, args.start_year, args.end_year)
    print(f"\nDone. Load it back with:\n"
          f'  research.data_loader.load_history("{ticker}", "{HISTDATA_PERIOD}", "1m", provider="{HISTDATA_PROVIDER}")')
