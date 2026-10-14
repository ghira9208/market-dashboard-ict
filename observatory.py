"""Strategy Observatory — on-demand, READ-ONLY analytics behind the
dashboard's Research > Strategy Observatory drill-down (2026-10-08).

Every function here re-simulates a combo's own REAL stored settings
against freshly-fetched data, the exact fetch-and-build pattern
validate_target_variants.py already uses — but nothing here ever calls
experiments._append_trial. That's deliberate, not an oversight: a
dashboard chart being viewed (or a parameter grid someone's dragging a
slider through) must never be able to write into the BH-corrected trial
log, shift a leaderboard count, or contaminate the research ledger with
throwaway exploratory noise. Where a real statistical test already
exists (experiments._score_split's permutation test), these functions
deliberately do NOT call it either — a parameter-sensitivity heatmap or a
regime matrix is a descriptive landscape, not a new pass/fail gate, so
it's scored with a plain mean/win-rate instead of paying for (and
implying the weight of) a 5000-draw permutation test per cell.

Everything here reads from experiments.py's existing _simulate_fixed_rr
output (mfe_r/mae_r/vol_regime/trend_regime, added alongside this module)
and the same 3 build_clean_*_fixed_rr_events builders every other research
surface in this project already uses — no new data pipeline, no new
simulation engine."""
import time

import pandas as pd

import experiments as ex
import data as research_data

BUILDER = {
    "clean_expansion_fixed_rr": ex.build_clean_expansion_fixed_rr_events,
    "clean_expansion_liquidity_fixed_rr": ex.build_clean_expansion_liquidity_fixed_rr_events,
    "clean_retracement_resumption_fixed_rr": ex.build_clean_retracement_resumption_fixed_rr_events,
}
# Mirrors validate_target_variants.py's own TF_FETCH exactly — kept as a
# separate small copy rather than importing that script module, since a
# script file is meant to be run, not imported as a library.
TF_FETCH = {
    "1D": dict(period="10y", interval="1d"),
    "1h": dict(period="730d", interval="1h"),
    "5m": dict(period="60d", interval="5m"),
}

_DETAIL_CACHE_TTL = 600  # 10 min — a real data fetch + simulation per combo
_detail_cache = {}
_GRID_CACHE_TTL = 1800  # 30 min — a full parameter grid is N^2 simulations
_grid_cache = {}


def _clean_settings(settings):
    # A trial's stored settings dict can carry a "_strategy_id" key
    # (run_deep_backtest_rr writes it in) — fatal if passed straight
    # through to a builder as **kwargs (same gotcha validate_target_
    # variants.py already documents and strips).
    return {k: v for k, v in settings.items() if not k.startswith("_")}


def effective_settings(settings, widen=1.0):
    """The settings the LIVE bot actually trades. A combo's stored settings carry its
    BASE rr_multiple; ticker_behavior's best_target_widen_mult (1.0, 1.5, 2.0, 3.0, 4.0 —
    validate_target_variants.py) multiplies the target distance on top, exactly as that
    script built its variants (rr_multiple = base_rr * widen). 57% of validated combos
    trade a widened target, so simulating the base rr shows a different trade than the
    one placed. Found 2026-10-13 comparing live signals against Observatory trades."""
    s = dict(settings)
    w = widen if widen and widen == widen else 1.0  # NaN/None -> unwidened
    if w != 1.0 and s.get("rr_multiple"):
        s["rr_multiple"] = round(s["rr_multiple"] * w, 6)
    return s


def _fetch_df(ticker, tf_label):
    if tf_label not in TF_FETCH:
        raise ValueError(f"no data-fetch recipe for tf_label {tf_label!r}")
    df = research_data.get_yf_ohlcv(ticker, **TF_FETCH[tf_label])
    if df.empty:
        raise ValueError(f"no data returned for {ticker} {tf_label}")
    return df


def _events_to_rows(events):
    rows = []
    for _, r in events.iterrows():
        et = r["entry_time"]
        rows.append({
            "entry_time": et.isoformat() if hasattr(et, "isoformat") else str(et),
            "r_real": float(r["r_real"]), "direction": r["direction"],
            "mfe_r": float(r["mfe_r"]), "mae_r": float(r["mae_r"]),
            "vol_regime": r["vol_regime"], "trend_regime": r["trend_regime"],
        })
    return rows


def trade_detail(ticker, tf_label, strategy, settings):
    """Every real simulated trade for this combo's own stored settings —
    entry_time/r_real/direction/mfe_r/mae_r/vol_regime/trend_regime,
    straight from _simulate_fixed_rr's own per-trade output. Backs the
    MFE/MAE scatter, the rolling-stability chart, and the regime matrix —
    one fetch+simulate per combo serves all three."""
    builder = BUILDER.get(strategy)
    if builder is None:
        raise ValueError(f"no fixed-R:R builder for strategy {strategy!r}")
    clean = _clean_settings(settings)
    key = (ticker, tf_label, strategy, tuple(sorted(clean.items())))
    cached = _detail_cache.get(key)
    if cached is not None and time.time() - cached[0] < _DETAIL_CACHE_TTL:
        return cached[1]
    df = _fetch_df(ticker, tf_label)
    events = builder(df, **clean)
    rows = _events_to_rows(events)
    _detail_cache[key] = (time.time(), rows)
    return rows


def regime_matrix(ticker, tf_label, strategy, settings):
    """Mean R and sample count per (vol_regime x trend_regime) cell — "is
    this combo's edge concentrated in one market condition, or does it
    hold up broadly." Descriptive only (see module docstring); unclassified
    trades (not enough trailing history at entry — see
    experiments._classify_regimes) are counted but never force-bucketed
    into a cell they don't honestly belong to."""
    rows = trade_detail(ticker, tf_label, strategy, settings)
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["r_real", "vol_regime", "trend_regime"])
    cells = []
    for vol in ("low", "normal", "high"):
        for trend in ("range", "trend"):
            sub = df[(df["vol_regime"] == vol) & (df["trend_regime"] == trend)]
            n = len(sub)
            cells.append({
                "vol_regime": vol, "trend_regime": trend, "n": n,
                "mean_r": float(sub["r_real"].mean()) if n else None,
            })
    n_unclassified = int(df["vol_regime"].isna().sum()) if len(df) else 0
    return {"cells": cells, "n_total": len(df), "n_unclassified": n_unclassified}


def rolling_stability(ticker, tf_label, strategy, settings, window_trades=30):
    """Rolling mean R + win rate over the trailing `window_trades` trades,
    in real chronological order — is the edge stable across its own real
    history, or concentrated in one stretch and fading since. n_in_window
    rides along on every point so an early, thin window never reads as a
    "stable" edge the sample size can't actually support."""
    rows = trade_detail(ticker, tf_label, strategy, settings)
    if not rows:
        return []
    df = pd.DataFrame(rows).sort_values("entry_time").reset_index(drop=True)
    out = []
    for i in range(len(df)):
        window = df.iloc[max(0, i - window_trades + 1): i + 1]
        out.append({
            "entry_time": df.iloc[i]["entry_time"],
            "rolling_mean_r": float(window["r_real"].mean()),
            "rolling_win_rate": float((window["r_real"] > 0).mean()),
            "n_in_window": int(len(window)),
        })
    return out


def parameter_grid(ticker, tf_label, strategy, settings, param_x, x_values, param_y, y_values):
    """2D parameter-sensitivity landscape: mean R per (param_x, param_y)
    cell, with the combo's own CURRENT production values marked —
    descriptive only (see module docstring for why this never touches the
    trial log or runs the permutation test). A single-cell spike next to
    the marked production point is the "suspiciously precise" case this
    chart exists to surface; a broad plateau around it is the reassuring
    one. Fetches the underlying OHLCV ONCE and re-simulates every grid
    cell against that same data — not one fetch per cell."""
    builder = BUILDER.get(strategy)
    if builder is None:
        raise ValueError(f"no fixed-R:R builder for strategy {strategy!r}")
    base = _clean_settings(settings)
    key = (ticker, tf_label, strategy, tuple(sorted(base.items())), param_x, tuple(x_values), param_y, tuple(y_values))
    cached = _grid_cache.get(key)
    if cached is not None and time.time() - cached[0] < _GRID_CACHE_TTL:
        return cached[1]

    df = _fetch_df(ticker, tf_label)
    cells = []
    for x in x_values:
        for y in y_values:
            variant = {**base, param_x: x, param_y: y}
            try:
                events = builder(df, **variant)
                n, mean_r = len(events), (float(events["r_real"].mean()) if len(events) else None)
            except Exception:
                n, mean_r = 0, None
            cells.append({param_x: x, param_y: y, "n": n, "mean_r": mean_r})

    result = {
        "cells": cells, "param_x": param_x, "param_y": param_y,
        "production_x": base.get(param_x), "production_y": base.get(param_y),
    }
    _grid_cache[key] = (time.time(), result)
    return result
