"""
Batch sweep across crypto tickers/intervals/sources — two separate outputs,
both cached to disk so research_app.py can display them without re-running
a ~15-30 min sweep on every page load:

1. Availability matrix: for every (ticker, interval, source) combo, how much
   history is actually there, or whether that combo fails outright. Answers
   "what's real" before any backtest params get chosen, same interrogate-
   first principle as the single-ticker flow in research_app.py, just
   across everything at once instead of one pick at a time.

2. Backtest sweep: the FVG-retracement hypothesis run on every ticker at
   the longer intervals (1h, 1d) — same forward_bars/min_body_ratio/cost as
   the BTC-USD runs already in the experiment log, so results are directly
   comparable rather than each ticker getting its own untracked parameter
   choice. Every run still goes through log_run — this sweep doesn't bypass
   the experiment log, it just automates firing a lot of entries into it at
   once, which raises the multiple-testing bar for trusting any single
   result even further (many more attempts logged now).

Run directly: python3 -m research.sweep
"""

import os

import pandas as pd

from research.backtest import run_event_study
from research.data_loader import INTERVAL_MAX_PERIOD, load_history
from research.events import extract_fvg_retracement_events
from research.experiment_log import log_run

# Top crypto pairs by liquidity/market cap that both Yahoo and Binance list
# under a consistent -USD / USDT naming convention — not an attempt to cover
# the whole market (get_crypto_universe() in data.py returns every live
# Binance USDT pair, hundreds of them; sweeping all of those would take
# hours for no real gain in what a "crypto overview" needs to show).
CRYPTO_TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD"]
SOURCES = ["yahoo", "binance"]
LONG_INTERVALS = ["60m", "1d"]  # "1h and up", per the actual request

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
AVAILABILITY_PATH = os.path.join(_CACHE_DIR, "availability_sweep.csv")
BACKTEST_SWEEP_PATH = os.path.join(_CACHE_DIR, "backtest_sweep.csv")


def run_availability_sweep(tickers=CRYPTO_TICKERS, intervals=None, sources=SOURCES):
    intervals = intervals or list(INTERVAL_MAX_PERIOD.keys())
    rows = []
    for ticker in tickers:
        for interval in intervals:
            for source in sources:
                try:
                    df = load_history(ticker, INTERVAL_MAX_PERIOD[interval], interval, provider=source)
                    rows.append({
                        "ticker": ticker, "interval": interval, "source": source,
                        "n_bars": len(df), "start": str(df.index[0]), "end": str(df.index[-1]),
                        "ok": True, "error": "",
                    })
                except Exception as e:
                    rows.append({
                        "ticker": ticker, "interval": interval, "source": source,
                        "n_bars": 0, "start": "", "end": "", "ok": False, "error": str(e)[:200],
                    })
                print(f"  {ticker:9s} {interval:4s} {source:8s} -> "
                      f"{'ok' if rows[-1]['ok'] else 'FAIL'} ({rows[-1]['n_bars']} bars)")

    df = pd.DataFrame(rows)
    os.makedirs(_CACHE_DIR, exist_ok=True)
    df.to_csv(AVAILABILITY_PATH, index=False)
    return df


def run_backtest_sweep(tickers=CRYPTO_TICKERS, intervals=LONG_INTERVALS,
                        forward_bars=50, min_body_ratio=0.5, cost_bps=10.0):
    rows = []
    for ticker in tickers:
        for interval in intervals:
            try:
                df = load_history(ticker, INTERVAL_MAX_PERIOD[interval], interval, provider="auto")
                events = extract_fvg_retracement_events(df, forward_bars=forward_bars, min_body_ratio=min_body_ratio)
                result = run_event_study(events, cost_bps=cost_bps)
                log_run(ticker, interval, INTERVAL_MAX_PERIOD[interval], "auto",
                        "FVG retracement -> continuation", forward_bars, cost_bps,
                        {"min_body_ratio": min_body_ratio, "sweep": True}, result)
                rows.append({
                    "ticker": ticker, "interval": interval, "n_events": result.get("n_events", 0),
                    "mean_return_after_cost": result.get("mean_return_after_cost"),
                    "win_rate_after_cost": result.get("win_rate_after_cost"),
                    "p_value": result.get("p_value_vs_random_direction"),
                    "verdict": result.get("verdict"),
                })
            except Exception as e:
                rows.append({"ticker": ticker, "interval": interval, "n_events": 0,
                              "mean_return_after_cost": None, "win_rate_after_cost": None,
                              "p_value": None, "verdict": f"ERROR: {e}"[:200]})
            print(f"  {ticker:9s} {interval:4s} -> {rows[-1]['verdict']}  "
                  f"(n={rows[-1]['n_events']}, p={rows[-1]['p_value']})")

    df = pd.DataFrame(rows)
    os.makedirs(_CACHE_DIR, exist_ok=True)
    df.to_csv(BACKTEST_SWEEP_PATH, index=False)
    return df


if __name__ == "__main__":
    print("=== Availability sweep ===")
    run_availability_sweep()
    print("\n=== Backtest sweep (1h, 1d) ===")
    run_backtest_sweep()
    print(f"\nSaved: {AVAILABILITY_PATH}\nSaved: {BACKTEST_SWEEP_PATH}")
