"""
Turns live_scan.py's raw detection log (research/.cache/live_events.csv)
into concrete setups: entry, stop, target, live status, and a validation
badge — the layer live_scan.py's own docstring deliberately left out ("no
signal, no trade, no alert-with-a-verdict... nothing honest to act on"),
written before edge_lab existed to validate anything. edge_lab now DOES
clear some candidates on GBPUSD; this module is what makes that validation
visible next to the live crypto detections instead of leaving the two
projects unconnected — without overstating what it means (see
validation_badge below).

Entry/stop/target is NOT a new formula — it's the exact one already
implemented independently in research_app.py's render_example_event_chart
and the matching block in edge_lab_app.py, reused here rather than
reinvented: stop sits at the far edge of the detector's own zone
(invalidates the read if price re-enters past it), or 2xATR when the
detector carries no real zone (BOS/CHoCH log a flat top==bottom level);
target is a fixed 2R.

Presentation-agnostic on purpose — signals_app.py owns all Streamlit/Plotly
code, this module owns only the setup math and the validation lookup, so
both are usable without a running Streamlit process.
"""

import json
import os

import pandas as pd

from research.agent_config import load_config  # noqa: F401  (re-exported for signals_app.py convenience)
from research.data_loader import INTERVAL_MAX_PERIOD, load_history
from research.live_scan import LIVE_EVENTS_PATH

_EDGE_LAB_TRIALS_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "edge_lab", ".cache", "trials.jsonl")
)
# edge_lab/agent.py's own hardcoded TICKER constant — every trial in its log
# is implicitly this ticker; trials don't record it explicitly since it
# never varies, so this module has to know it separately to judge whether a
# validation result is even about the same market as the live setup.
EDGE_LAB_TICKER = "GBPUSD=X"

# research/events.HYPOTHESES keys, keyed by live_scan.py's own event_type
# names so a live_events.csv row can be looked up against edge_lab's log.
EVENT_TYPE_TO_HYPOTHESIS = {
    "fvg": "FVG retracement -> continuation",
    "order_block": "Order Block retracement -> continuation",
    "liquidity_reaction": "Liquidity Reaction retracement -> continuation",
    "equal_highs_lows": "Equal Highs/Lows -> sweep reversal",
    "bos": "BOS -> continuation",
    "choch": "CHoCH -> reversal",
    "judas_swing": "Judas Swing -> reversal",
}

EVENT_TYPE_LABELS = {
    "fvg": "FVG", "order_block": "Order Block", "liquidity_reaction": "Liquidity Reaction",
    "equal_highs_lows": "Equal Highs/Lows", "bos": "BOS", "choch": "CHoCH", "judas_swing": "Judas Swing",
}

# How far back (from now) a detected structure still counts as a live setup
# worth showing, scaled roughly to how long a fixed-2R target realistically
# needs to resolve at that timeframe before calling it stale rather than
# leaving every setup ever logged permanently "active".
LOOKBACK = {
    "1m": pd.Timedelta(hours=6), "5m": pd.Timedelta(days=2), "15m": pd.Timedelta(days=3),
    "30m": pd.Timedelta(days=5), "60m": pd.Timedelta(days=10), "1d": pd.Timedelta(days=90),
}


def _atr(df, lookback_bars=50):
    window = df.tail(lookback_bars)
    return float((window["High"] - window["Low"]).mean())


def compute_setup(row, df):
    """row: a plain dict for one live_events.csv record (ticker, interval,
    event_type, direction, start, end, top, bottom). df: that ticker/
    interval's cached OHLCV history (the same load_history(..., no
    refresh=True) call live_scan.py's own scan makes — no network here).

    Entry is taken at the close of the bar where the pattern's own `end`
    timestamp falls — the formation-confirming bar, since live_events.csv
    doesn't carry every detector's first_touch the way the backtest
    extractors in research/events.py do. That's a real simplification (a
    trader wouldn't always get filled exactly there) — flagged here, not
    hidden, since it changes what "entry" means slightly from the backtest
    examples this formula was copied from.

    Returns None if `end` falls outside df's cached range (a stale/aged-out
    cache) or the computed risk is zero."""
    end_time = pd.Timestamp(row["end"])
    if end_time.tzinfo is None:
        end_time = end_time.tz_localize("UTC")
    else:
        end_time = end_time.tz_convert("UTC")

    idx = df.index if df.index.tz is not None else df.index.tz_localize("UTC")
    pos = idx.searchsorted(end_time)
    if pos >= len(idx):
        pos = len(idx) - 1
    if pos < 0 or idx[pos] < end_time - pd.Timedelta(days=3):
        return None

    entry_time = df.index[pos]
    entry_price = float(df["Close"].iloc[pos])
    direction = row["direction"]
    top, bottom = row.get("top"), row.get("bottom")

    if pd.notna(top) and pd.notna(bottom) and top != bottom:
        sl_price = float(bottom) if direction == "bullish" else float(top)
    else:
        atr = _atr(df.iloc[:pos + 1])
        sl_price = entry_price - atr * 2 if direction == "bullish" else entry_price + atr * 2

    risk = abs(entry_price - sl_price)
    if risk <= 0:
        return None
    tp_price = entry_price + risk * 2 if direction == "bullish" else entry_price - risk * 2

    status, resolved_at, resolved_price = "active", None, None
    forward = df.iloc[pos + 1:]
    if not forward.empty:
        # Vectorized, not .iterrows() — with 5000+ current setups, some
        # forward-scanning hundreds of bars each on 5m/15m data, the
        # row-by-row Python loop this replaced was the actual reason a
        # single check_notable() run took minutes instead of seconds.
        highs, lows = forward["High"].to_numpy(), forward["Low"].to_numpy()
        if direction == "bullish":
            hit_tp_mask, hit_sl_mask = highs >= tp_price, lows <= sl_price
        else:
            hit_tp_mask, hit_sl_mask = lows <= tp_price, highs >= sl_price
        tp_idx = int(hit_tp_mask.argmax()) if hit_tp_mask.any() else None
        sl_idx = int(hit_sl_mask.argmax()) if hit_sl_mask.any() else None
        # An OHLC bar alone can't say which of TP/SL came first when a
        # single bar's range spans both — calling it against the trade is
        # the conservative read, so SL wins whenever it happens on the same
        # bar as TP or earlier, not just strictly earlier.
        if sl_idx is not None and (tp_idx is None or sl_idx <= tp_idx):
            status, resolved_at, resolved_price = "hit_sl", forward.index[sl_idx], sl_price
        elif tp_idx is not None:
            status, resolved_at, resolved_price = "hit_tp", forward.index[tp_idx], tp_price

    if status == "active":
        lookback = LOOKBACK.get(row["interval"], pd.Timedelta(days=3))
        if pd.Timestamp.now(tz="UTC") - entry_time > lookback:
            status = "expired"

    current_price = float(df["Close"].iloc[-1])
    reward = abs(tp_price - entry_price)
    return {
        **row,
        "entry_time": entry_time, "entry_price": entry_price,
        "sl_price": sl_price, "tp_price": tp_price, "risk": risk,
        "reward_risk": reward / risk,
        "distance_pct": (current_price - entry_price) / entry_price * 100,
        "status": status, "resolved_at": resolved_at, "resolved_price": resolved_price,
        "current_price": current_price,
    }


def load_live_setups():
    """Reads live_events.csv (huge and append-only — see live_scan.py's own
    docstring on why: it backfills the whole cached history the first time
    a ticker/interval/detector combo is scanned, not just brand-new bars),
    keeps only structures whose formation is still within their live-
    relevance LOOKBACK window, and computes a setup for each. Skips rows
    compute_setup can't resolve (e.g. a cache that's since aged past that
    timestamp) rather than raising."""
    if not os.path.exists(LIVE_EVENTS_PATH):
        return []
    events = pd.read_csv(LIVE_EVENTS_PATH)
    if events.empty:
        return []
    events["end_ts"] = pd.to_datetime(events["end"], utc=True, errors="coerce")
    events = events.dropna(subset=["end_ts"])
    now = pd.Timestamp.now(tz="UTC")
    # Vectorized, not .apply(axis=1) — live_events.csv is the full backfilled
    # history (100k+ rows), and a Python-level lambda per row there took over
    # two minutes; mapping the lookback per interval as its own column and
    # comparing as plain Series arithmetic is the same filter in well under a
    # second.
    events["_lookback"] = events["interval"].map(LOOKBACK).fillna(pd.Timedelta(days=3))
    recent = events[now - events["end_ts"] <= events["_lookback"]].sort_values("end_ts", ascending=False)

    setups = []
    df_cache = {}
    for _, row in recent.iterrows():
        key = (row["ticker"], row["interval"])
        if key not in df_cache:
            try:
                df_cache[key] = load_history(row["ticker"], INTERVAL_MAX_PERIOD[row["interval"]], row["interval"])
            except Exception:
                df_cache[key] = None
        df = df_cache[key]
        if df is None or df.empty:
            continue
        setup = compute_setup(row.drop(["end_ts", "_lookback"]).to_dict(), df)
        if setup is not None:
            setups.append(setup)
    return setups


def load_edge_lab_validation():
    """Mirrors edge_lab_app.py's leaderboard computation exactly — same BH
    correction over the same full cumulative trial log — so 'validated'
    means the identical thing here that it means there, then groups the
    result by hypothesis label. Returns {hypothesis: {"validated": bool,
    "best": trial_dict}}. Only hypotheses with at least one BH-significant
    trial appear at all; everything else is implicitly "no BH-significant
    trial exists yet", which validation_badge treats as unvalidated."""
    if not os.path.exists(_EDGE_LAB_TRIALS_PATH):
        return {}
    trials = []
    with open(_EDGE_LAB_TRIALS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                trials.append(json.loads(line))
    scored = [t for t in trials if t.get("verdict") == "SCORED"]
    if not scored:
        return {}

    from edge_lab.multiple_testing import benjamini_hochberg
    p_values = [t["p_value_train"] for t in scored]
    q_values, significant = benjamini_hochberg(p_values, alpha=0.05)
    for t, q, sig in zip(scored, q_values, significant):
        t["q_value_train"] = float(q)
        t["bh_significant"] = bool(sig)

    result = {}
    for t in scored:
        if not t.get("bh_significant"):
            continue
        hyp = t.get("hypothesis")
        passed = t.get("holdout_verdict") == "PASSED"
        current = result.get(hyp)
        if current is None or (passed and not current["validated"]):
            result[hyp] = {"validated": passed, "best": t}
    return result


def validation_badge(event_type, ticker):
    """(label, tone) for display next to one setup. tone is one of
    "bullish" (validated), "warn" (tested, not validated), "info" (edge_lab
    has never tested this market at all — see EDGE_LAB_TICKER)."""
    if ticker != EDGE_LAB_TICKER:
        return "not tested on this market — edge_lab has only run on GBPUSD so far", "info"
    hyp = EVENT_TYPE_TO_HYPOTHESIS.get(event_type)
    v = load_edge_lab_validation().get(hyp)
    if v is None:
        return "no BH-significant trial yet", "warn"
    if v["validated"]:
        best = v["best"]
        return (f"validated — q={best['q_value_train']:.4f} (BH-adjusted), "
                f"holdout p={best.get('p_value_holdout', float('nan')):.4f}"), "bullish"
    return "train-significant, not holdout-validated", "warn"
