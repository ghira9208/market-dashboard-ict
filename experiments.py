"""
The experiment builder — bridges tuning detector/regime settings live
against a chart and running those same settings through a real, Edge-Lab-
grade backtest. Built around a specific trading thesis, stated directly
rather than inferred from a request: trade sharp, fast reactions off
known zones (FVG/Order Block), but ONLY when the market has enough
volatility to actually deliver the move being looked for — a
consolidating range offers nothing to react FROM, so entering one is
just paying cost for no edge.

Three ingredients, each a checkable definition, not a vibe:

  - volatility regime: each bar's own ATR ranked against its trailing
    history (a rolling percentile, computed causally — no lookahead).
    High percentile = expansion, low = squeeze/consolidation.
  - reaction quality: how far price closed away from a zone's own
    ORIGINAL bounds (raw_top/raw_bottom — before consequent-encroachment
    eats into it) within a handful of bars of first touching it. Sharp
    and immediate scores high; a slow grind through the zone doesn't
    qualify at all.
  - fast resolution: forward_bars is deliberately short (a handful of
    bars) rather than the 10-20+ used elsewhere in this project — the
    actual thesis is "did this resolve FAST," not "did it eventually
    work out."

Reuses detectors.py's own zone detection (first_touch/raw_top/raw_bottom
already tracked there for exactly this kind of downstream use — nothing
new added to the detectors themselves) and research/evidence.py's
permutation-test engine. No new detection or statistics machinery, just
a new way of combining what already exists.

Every "deep backtest" run — pass or fail — is appended to
.cache/experiment_trials.jsonl, and load_experiment_trials() BH-corrects
across the WHOLE accumulated log, same "one p-value means nothing in
isolation" discipline as edge_lab/agent.py and institutional_backtest.py.
Tuning sliders and re-running until one trial looks good, then reporting
only that one, is exactly the p-hacking this correction step exists to
catch — it's applied here for the same reason.
"""
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

from detectors import detect_fvgs, detect_order_blocks
from edge_lab.multiple_testing import benjamini_hochberg
from research.evidence import run_event_study

_CACHE_TTL = 300
_TRIALS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "experiment_trials.jsonl")

DETECTOR_FNS = {"FVG": detect_fvgs, "Order Block": detect_order_blocks}


@st.cache_data(ttl=_CACHE_TTL)
def atr_series(df, period=14):
    high = (df["High"] if "High" in df else df["high"]).to_numpy()
    low = (df["Low"] if "Low" in df else df["low"]).to_numpy()
    close = (df["Close"] if "Close" in df else df["close"]).to_numpy()
    prev_close = np.empty_like(close)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return pd.Series(tr, index=df.index).rolling(period, min_periods=1).mean()


@st.cache_data(ttl=_CACHE_TTL)
def volatility_percentile(df, atr_period=14, lookback=100):
    """Each bar's own ATR, ranked against its trailing `lookback` bars —
    0.0 = the quietest this instrument has recently been, 1.0 = the most
    volatile. A rolling rank per bar, not one current-vs-history number,
    so every historical bar in a backtest gets a regime label computed
    only from data available AT that bar."""
    atr = atr_series(df, atr_period)
    return atr.rolling(lookback, min_periods=max(5, lookback // 2)).apply(
        lambda w: (w.iloc[-1] >= w).mean(), raw=False
    )


def _zone_reaction(df, zone, reaction_window, reaction_mult, atr, close_arr, pos_by_time):
    """Was the reaction off this zone SHARP — did price close away from
    its own ORIGINAL bounds by at least reaction_mult x ATR within
    reaction_window bars of first touching it? None means "nothing to
    judge" (never touched at all, or touched too close to the end of the
    loaded data to see reaction_window bars past it) — never a False
    disguised as a real negative result."""
    if zone.get("first_touch") is None:
        return None
    touch_pos = pos_by_time.get(zone["first_touch"])
    if touch_pos is None:
        return None
    end_pos = touch_pos + reaction_window
    if end_pos >= len(close_arr):
        return None
    atr_at_touch = atr.iloc[touch_pos]
    if pd.isna(atr_at_touch) or atr_at_touch <= 0:
        return None
    moved = close_arr[end_pos] - close_arr[touch_pos]
    direction = zone["type"]
    expected_sign = 1 if direction == "bullish" else -1
    qualifies = (moved * expected_sign) >= (reaction_mult * atr_at_touch)
    return {"qualifies": bool(qualifies), "direction": direction, "touch_pos": touch_pos}


@st.cache_data(ttl=_CACHE_TTL)
def build_experiment_events(df, detector_name, min_volatility_pctile, reaction_window,
                             reaction_mult, forward_bars, max_scan_bars=2000):
    """events_df (entry_time, raw_return, direction) — the exact shape
    research/evidence.run_event_study expects. Built by filtering EVERY
    historical zone this detector found (over whatever history `df`
    covers — switch the chart's own period/timeframe for more) down to
    the ones that both got a sharp, fast reaction AND fired while the
    volatility regime was at or above min_volatility_pctile at the
    moment of that reaction. raw_return is the ACTUAL forward return
    over forward_bars bars from the touch — never adjusted, never
    assumed."""
    detect_fn = DETECTOR_FNS[detector_name]
    zones = detect_fn(df, max_scan_bars=max_scan_bars, record_history=True)
    if not zones:
        return pd.DataFrame(columns=["entry_time", "raw_return", "direction"])

    vol_pctile = volatility_percentile(df)
    atr = atr_series(df)
    close = (df["Close"] if "Close" in df else df["close"])
    close_arr = close.to_numpy()
    n = len(df)
    pos_by_time = {t: i for i, t in enumerate(df.index)}

    rows = []
    for zone in zones:
        reaction = _zone_reaction(df, zone, reaction_window, reaction_mult, atr, close_arr, pos_by_time)
        if reaction is None or not reaction["qualifies"]:
            continue
        touch_pos = reaction["touch_pos"]
        vp = vol_pctile.iloc[touch_pos]
        if pd.isna(vp) or vp < min_volatility_pctile:
            continue
        exit_pos = touch_pos + forward_bars
        if exit_pos >= n:
            continue
        fwd_return = float((close_arr[exit_pos] - close_arr[touch_pos]) / close_arr[touch_pos])
        rows.append({"entry_time": df.index[touch_pos], "raw_return": fwd_return,
                     "direction": reaction["direction"]})
    return pd.DataFrame(rows)


def preview_stats(events_df, cost_bps=10.0):
    """Instant, no-permutation-test feedback for tuning sliders live — n /
    win-rate / mean-return only, cheap enough to recompute on every widget
    change. NOT a validated result on its own — see run_deep_backtest for
    that; this only exists to tell you if a setting is worth testing
    properly at all before spending a real permutation test on it."""
    if events_df.empty:
        return {"n": 0, "win_rate": None, "mean_return": None}
    cost = cost_bps / 10_000.0
    direction = np.where(events_df["direction"] == "bullish", 1, -1)
    signed = events_df["raw_return"].to_numpy() * direction - cost
    return {"n": len(events_df), "win_rate": float((signed > 0).mean()), "mean_return": float(signed.mean())}


def run_deep_backtest(ticker, tf_label, events_df, settings, cost_bps=10.0,
                       train_fraction=0.8, n_permutations=2000):
    """The real version: 80/20 time split, permutation test on each side
    (research/evidence.run_event_study, unmodified), same-sign check
    between train and holdout (the gap institutional_backtest.py found
    and fixed — a trial can clear significance on both splits while they
    disagree on direction; that's a near-constant classifier riding one
    regime in train and the opposite in holdout, not a real edge).
    Logged to the accumulating trials file regardless of outcome — see
    this module's own docstring on why. Returns the trial dict that was
    appended."""
    label = (f"{settings['detector']} · vol≥{settings['min_volatility_pctile']:.0%} · "
             f"react {settings['reaction_mult']}x/{settings['reaction_window']}b · "
             f"fwd {settings['forward_bars']}b")
    trial = {
        "logged_at": pd.Timestamp.now("UTC").isoformat(), "ticker": ticker, "tf_label": tf_label,
        "label": label, "settings": settings, "n_events": len(events_df),
    }
    if len(events_df) < 30:
        trial["verdict"] = "INSUFFICIENT_DATA"
        _append_trial(trial)
        return trial

    events_df = events_df.sort_values("entry_time").reset_index(drop=True)
    cutoff_pos = int(len(events_df) * train_fraction)
    cutoff = events_df["entry_time"].iloc[cutoff_pos]
    train = events_df[events_df["entry_time"] < cutoff]
    holdout = events_df[events_df["entry_time"] >= cutoff]
    train_result = run_event_study(train, cost_bps=cost_bps, direction_col="direction",
                                    n_permutations=n_permutations)
    holdout_result = run_event_study(holdout, cost_bps=cost_bps, direction_col="direction",
                                      n_permutations=n_permutations)

    n_train = train_result.get("n_events", 0)
    if n_train == 0:
        trial["verdict"] = "INSUFFICIENT_DATA"
        _append_trial(trial)
        return trial

    p_train = train_result.get("p_value_vs_random_direction")
    mean_train = train_result.get("mean_return_after_cost")
    mean_holdout = holdout_result.get("mean_return_after_cost")
    trial.update({
        "n_train": n_train, "n_holdout": holdout_result.get("n_events", 0),
        "mean_return_train": mean_train, "p_value_train": p_train,
        "mean_return_holdout": mean_holdout,
        "holdout_verdict": "PASSED" if holdout_result.get("verdict") == "EDGE_FOUND" else "FAILED",
        "same_sign": bool(mean_train is not None and mean_holdout is not None
                           and mean_train > 0 and mean_holdout > 0),
        "verdict": "SCORED",
    })
    _append_trial(trial)
    return trial


def _append_trial(trial):
    try:
        os.makedirs(os.path.dirname(_TRIALS_PATH), exist_ok=True)
        with open(_TRIALS_PATH, "a") as f:
            f.write(json.dumps(trial, default=str) + "\n")
    except Exception:
        pass


def load_experiment_trials():
    """Every trial ever run through run_deep_backtest, BH-corrected
    across the FULL log (not just today's session) — same convention as
    research/setups.py's load_edge_lab_validation. Empty DataFrame if
    nothing's been run yet."""
    if not os.path.exists(_TRIALS_PATH):
        return pd.DataFrame()
    trials = []
    with open(_TRIALS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                trials.append(json.loads(line))
    if not trials:
        return pd.DataFrame()

    df = pd.DataFrame(trials)
    scored = df[df["verdict"] == "SCORED"]
    if not scored.empty:
        q_values, significant = benjamini_hochberg(scored["p_value_train"].tolist(), alpha=0.05)
        df.loc[scored.index, "q_value_train"] = q_values
        df.loc[scored.index, "bh_significant"] = significant
        df["survived"] = False
        df.loc[scored.index, "survived"] = (
            df.loc[scored.index, "bh_significant"].fillna(False)
            & (df.loc[scored.index, "holdout_verdict"] == "PASSED")
            & df.loc[scored.index, "same_sign"].fillna(False)
        )
    return df.sort_values("logged_at", ascending=False).reset_index(drop=True)
