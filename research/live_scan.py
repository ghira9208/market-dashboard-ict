"""
The "continuously do something" agent — runs on a local schedule (launchd,
see research/com.marketdashboard.livescan.plist), not a cloud routine: this
project has no git repo and the whole point depends on local disk state
(the accumulating parquet caches from data_loader.py) persisting between
runs, which a cloud sandbox starting fresh every fire cannot give it.

Two jobs each run:
1. Refresh (accumulate) OHLCV caches for the watched pairs/intervals — the
   actual trigger the merge-on-refresh logic in data_loader.py needed and
   never had; nothing was calling refresh=True on any schedule before this.
2. Re-run the ICT detectors (fvg.py — same ones the live chart draws, same
   ones events.py already wraps for backtesting) on the freshest data, and
   append any FVG/order-block/liquidity-sweep NOT already in the running
   log. Deliberately just detection + logging, no signal, no trade, no
   alert-with-a-verdict — this session's own sweep found no validated edge
   yet, so there is nothing honest to act on. This is the raw material a
   future validated hypothesis would need, and a plain running record of
   "what formed and when" is itself useful before any of that exists.
"""

import os
import subprocess
from datetime import datetime, timezone

import pandas as pd

from research.agent_config import load_config
from research.data_loader import INTERVAL_MAX_PERIOD, load_history
from research.sequences import detect_judas_swing_setups
from fvg import (detect_fvgs, detect_order_blocks, detect_liquidity_reactions,
                  detect_equal_highs_lows, detect_structure_breaks)

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
LIVE_EVENTS_PATH = os.path.join(_CACHE_DIR, "live_events.csv")
_FIELDS = ["logged_at", "ticker", "interval", "event_type", "direction", "start", "end", "top", "bottom"]

# bos/choch both call detect_structure_breaks (st.cache_data-decorated, so
# the second call per df is a cache hit, not a recompute) and just filter
# to their own kind — registered as two separate toggles rather than one
# "market_structure" detector because BOS (routine continuation) and CHoCH
# (the actual reversal warning) are different enough signals that someone
# watching might reasonably want one without the other.
_DETECTORS = {
    "fvg": lambda df: [(g["start"], g["end"], g["type"], g["top"], g["bottom"]) for g in detect_fvgs(df)],
    "order_block": lambda df: [(o["start"], o["end"], o["type"], o["top"], o["bottom"]) for o in detect_order_blocks(df)],
    "liquidity_reaction": lambda df: [(r["start"], r["end"], r["type"], r["top"], r["bottom"]) for r in detect_liquidity_reactions(df)],
    "equal_highs_lows": lambda df: [(e["start"], e["end"], e["type"], e["top"], e["bottom"]) for e in detect_equal_highs_lows(df)],
    "bos": lambda df: [(b["start"], b["end"], b["type"], b["level"], b["level"])
                        for b in detect_structure_breaks(df) if b["structure"] == "BOS"],
    "choch": lambda df: [(b["start"], b["end"], b["type"], b["level"], b["level"])
                          for b in detect_structure_breaks(df) if b["structure"] == "CHoCH"],
    # The one detector that actually sends a notification, not just a log
    # row — see scan_for_new_structures below. Every other entry here is a
    # single, standalone signal; this is the completed 3-step Judas swing
    # playbook (sweep -> CHoCH -> retracement, research/sequences.py), which
    # is what "alert me when all conditions are met one after another"
    # meant — a single detector firing isn't that, a chain completing is.
    "judas_swing": lambda df: [(s["start"], s["end"], s["type"], s["top"], s["bottom"])
                                for s in detect_judas_swing_setups(df)],
}

_ALERT_EVENT_TYPES = {"judas_swing"}


def _notify(title, message):
    """macOS native notification banner — appropriate for a personal local
    agent (see live_scan.py's own module docstring on why this runs via
    launchd, not a cloud routine: everything about this agent is local-
    machine, this is too). Sanitizes quotes since ticker/message content
    gets interpolated into an AppleScript string literal."""
    safe_title = title.replace('"', "'")
    safe_message = message.replace('"', "'")
    try:
        subprocess.run(
            ["osascript", "-e", f'display notification "{safe_message}" with title "{safe_title}" sound name "Glass"'],
            capture_output=True, timeout=5,
        )
    except Exception as e:
        print(f"  (notification failed: {e})")


def _load_seen_keys():
    if not os.path.exists(LIVE_EVENTS_PATH):
        return set()
    df = pd.read_csv(LIVE_EVENTS_PATH)
    return set(zip(df["ticker"], df["interval"], df["event_type"], df["start"]))


def refresh_caches(tickers, intervals, provider="auto", on_progress=None):
    """on_progress(ticker, interval, n_bars, error) fires after EVERY combo,
    success or failure (n_bars=None on failure) — the launchd/CLI path
    leaves this None and just prints, same as before; research_app.py's
    live-run button passes a callback that updates on-page placeholders in
    real time instead of only being visible after the fact in a log file."""
    for ticker in tickers:
        for interval in intervals:
            try:
                df = load_history(ticker, INTERVAL_MAX_PERIOD[interval], interval, provider=provider, refresh=True)
                print(f"  refreshed {ticker} {interval}: {len(df)} bars, latest {df.index[-1]}")
                if on_progress:
                    on_progress(ticker, interval, len(df), None)
            except Exception as e:
                print(f"  FAILED {ticker} {interval}: {e}")
                if on_progress:
                    on_progress(ticker, interval, None, str(e))


def scan_for_new_structures(tickers, intervals, detectors, on_progress=None):
    """on_progress(ticker, interval, n_new_this_combo, n_new_total) fires
    after every ticker/interval combo is scanned — same reasoning as
    refresh_caches' callback."""
    seen = _load_seen_keys()
    new_rows = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    active_detectors = {name: fn for name, fn in _DETECTORS.items() if detectors.get(name)}

    for ticker in tickers:
        for interval in intervals:
            n_before = len(new_rows)
            try:
                df = load_history(ticker, INTERVAL_MAX_PERIOD[interval], interval)  # cached, no network
            except Exception:
                df = None
            if df is not None and not df.empty:
                for event_type, fn in active_detectors.items():
                    for start, end, direction, top, bottom in fn(df):
                        key = (ticker, interval, event_type, str(start))
                        if key not in seen:
                            new_rows.append({"logged_at": now, "ticker": ticker, "interval": interval,
                                              "event_type": event_type, "direction": direction, "start": str(start),
                                              "end": str(end), "top": top, "bottom": bottom})
                            seen.add(key)
            if on_progress:
                on_progress(ticker, interval, len(new_rows) - n_before, len(new_rows))

    if new_rows:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        is_new = not os.path.exists(LIVE_EVENTS_PATH)
        pd.DataFrame(new_rows).to_csv(LIVE_EVENTS_PATH, mode="a", header=is_new, index=False)

    alerts = [r for r in new_rows if r["event_type"] in _ALERT_EVENT_TYPES]
    for r in alerts:
        _notify(f"Judas Swing setup — {r['ticker']} {r['interval']}",
                f"{r['direction']} — sweep -> CHoCH -> retracement complete, zone {r['bottom']:.4g}-{r['top']:.4g}")

    print(f"  {len(new_rows)} new structure(s) logged ({len(alerts)} alerted).")
    return new_rows


if __name__ == "__main__":
    print(f"=== Live scan run: {datetime.now(timezone.utc).isoformat(timespec='seconds')} ===")
    cfg = load_config()
    if not cfg["enabled"]:
        print("Agent disabled in agent_config.json — skipping this run (soft kill switch, launchd job still scheduled).")
    else:
        print(f"Config: tickers={cfg['tickers']} intervals={cfg['intervals']} "
              f"detectors={[k for k, v in cfg['detectors'].items() if v]} provider={cfg['provider']}")
        print("Refreshing caches...")
        refresh_caches(cfg["tickers"], cfg["intervals"], provider=cfg["provider"])
        print("Scanning for new structures...")
        scan_for_new_structures(cfg["tickers"], cfg["intervals"], cfg["detectors"])
