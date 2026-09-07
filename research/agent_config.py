"""
Settings for the live_scan agent — a plain JSON file, not module constants,
because the agent runs as a fresh `python3 -m research.live_scan` process
launched by launchd every hour; there's no long-lived Python process for a
UI to hand settings to directly. The Streamlit control panel writes this
file, live_scan.py reads it at the start of every run — the file IS the
interface between them.

`enabled` is a soft kill switch independent of whether the launchd job
itself is loaded: flipping it off takes effect on the NEXT scheduled fire
with no launchctl call needed, while actually loading/unloading the job
(toggled separately, see research_app.py) controls whether it's scheduled
to fire at all. Two different "off"s on purpose — one instant and
reversible from the UI alone, one that stops the hourly wakeups entirely.
"""

import json
import os

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_config.json")

DEFAULT_CONFIG = {
    "enabled": True,
    "tickers": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD"],
    "intervals": ["15m", "60m", "1d"],
    "detectors": {"fvg": True, "order_block": True, "liquidity_reaction": True, "equal_highs_lows": True,
                  "bos": True, "choch": True, "judas_swing": True},
    "provider": "auto",
}


def load_config():
    if not os.path.exists(_CONFIG_PATH):
        return dict(DEFAULT_CONFIG)
    try:
        with open(_CONFIG_PATH) as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError):
        return dict(DEFAULT_CONFIG)
    # Merge over defaults rather than trusting the file alone — a config
    # saved before some future setting existed shouldn't crash live_scan.py
    # on a missing key, it should just get that setting's default.
    merged = dict(DEFAULT_CONFIG)
    merged.update(cfg)
    merged["detectors"] = {**DEFAULT_CONFIG["detectors"], **cfg.get("detectors", {})}
    return merged


def save_config(config):
    with open(_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
