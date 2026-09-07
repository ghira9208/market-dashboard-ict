"""
Append-only record of every study ever run — the single most important thing
to persist in this pipeline, more than any additional price-data type. Event
studies are cheap to run and easy to re-run with slightly different
parameters until something clears p<0.05 by chance; without a permanent
record of every attempt, there's no way to later tell a real edge apart from
the 1-in-20 that a large enough search turns up on pure noise. Every run
gets logged here regardless of its verdict — NO_EDGE rows matter as much as
EDGE_FOUND ones, since "how many things did we try" is exactly what makes
the p-value on any one of them meaningful.
"""

import csv
import os
from datetime import datetime, timezone

_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "experiment_log.csv")

_FIELDS = [
    "logged_at", "ticker", "interval", "period", "provider", "hypothesis",
    "forward_bars", "cost_bps", "extra_params", "n_events",
    "mean_return_after_cost", "win_rate_after_cost", "p_value", "verdict",
]


def log_run(ticker, interval, period, provider, hypothesis, forward_bars, cost_bps, extra_params, result):
    """result: the dict returned by backtest.run_event_study. extra_params:
    the hypothesis-specific kwargs dict (e.g. {"min_body_ratio": 0.5})."""
    os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
    is_new = not os.path.exists(_LOG_PATH)
    with open(_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ticker": ticker, "interval": interval, "period": period, "provider": provider,
            "hypothesis": hypothesis, "forward_bars": forward_bars, "cost_bps": cost_bps,
            "extra_params": str(extra_params), "n_events": result.get("n_events"),
            "mean_return_after_cost": result.get("mean_return_after_cost"),
            "win_rate_after_cost": result.get("win_rate_after_cost"),
            "p_value": result.get("p_value_vs_random_direction"),
            "verdict": result.get("verdict"),
        })


def read_log():
    """Returns a pandas DataFrame of every logged run, oldest first. Empty
    DataFrame (with the right columns) if nothing's been logged yet."""
    import pandas as pd
    if not os.path.exists(_LOG_PATH):
        return pd.DataFrame(columns=_FIELDS)
    return pd.read_csv(_LOG_PATH)
