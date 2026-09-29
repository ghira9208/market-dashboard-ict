"""
A queryable summary of experiments.py's own accumulated trial log
(.cache/experiment_trials.jsonl — every deep-backtest ever run this
project's whole history, BH-corrected across the WHOLE log by
load_experiment_trials()) — one row per (ticker, timeframe, strategy
family) telling you, at a glance, "does anything validated exist here,
and what."

Genuinely additive: reads load_experiment_trials()'s own output,
writes nothing back to the trial log, changes no existing behavior.
rebuild_db() is the only way data gets in — call it after any new sweep
so the summary reflects what's actually been tried; nothing here re-runs
a backtest or fetches market data itself.

Why a real SQLite file (ticker_behavior.db) instead of just re-querying
load_experiment_trials() on demand: that function re-parses and re-BH-
corrects the ENTIRE trial log (88k+ rows and growing) on every call —
fine for one-off checks, wasteful for "let me look up three tickers back
to back." This is a cheap, indexed, disk-backed cache of the aggregation,
same reasoning as backtest_results.db already used elsewhere in this
project for the separate custom-rule sweep engine — a different dataset,
same "don't make the user wait on a re-scan every lookup" idea.
"""
import os
import re
import sqlite3

import pandas as pd

import experiments as ex

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ticker_behavior.db")

# The original detector-reaction strategy family (build_experiment_events)
# logs its OWN detector name as the whole label ("FVG", "Order Block", ...)
# with settings baked into the `settings` column instead of the label
# string, unlike every later strategy here whose label reads
# "STRATEGY NAME · param=value · ...". Splitting on " · " already isolates
# those correctly; the four detector names just happen to have no " · "
# in them at all, so they fall out of the same split with no special case.
_STRATEGY_RE = re.compile(r"\s*·\s*")

# The same tf_label spelling drift experiments.load_experiment_trials()
# itself already normalizes at its own root (TF_LABEL_NORMALIZE) — reused
# here rather than kept as a second, driftable copy. Since that
# normalization already happens inside load_experiment_trials(), every
# tf_label this module ever sees back from it is ALREADY canonical; this
# alias exists only for survivors_ui.py, which needs to normalize a
# tf_label it read from THIS module's own SQLite db the same way before
# using it as an app.py TIMEFRAMES[...] key.
def normalize_tf(tf_label):
    return ex.TF_LABEL_NORMALIZE.get(tf_label, tf_label)


_normalize_tf = normalize_tf


def _strategy_family(label):
    if not isinstance(label, str) or not label:
        return "unknown"
    return _STRATEGY_RE.split(label, maxsplit=1)[0].strip()


def rebuild_db():
    """Re-reads the WHOLE trial log and rewrites ticker_behavior.db from
    scratch (drop + recreate both tables) — the trial log is the only
    source of truth here, so a full rebuild is simpler and always
    correct, versus trying to incrementally patch a stale summary."""
    trials = ex.load_experiment_trials()
    if trials.empty:
        return
    trials = trials.copy()
    trials["tf_label"] = trials["tf_label"].map(_normalize_tf)
    trials["strategy"] = trials["label"].map(_strategy_family)

    # Found live (2026-09-29): re-validating a combo (revalidate_combos.py,
    # or any re-run of the exact same settings after a methodology change
    # like adding a real trading-cost assumption) APPENDS a new trial under
    # the identical (ticker, tf_label, label) — it never replaces the old
    # one. Without this dedup, best_row = idxmax(mean_return_train) below
    # picks across BOTH freely, and an older, more optimistic trial for the
    # exact same settings always wins over a newer, more honest one (a
    # frictionless pre-cost-model trial beats its own cost-aware re-run on
    # paper every time) — confirmed directly: ETH-USD 5m's own leaderboard
    # entry was still serving mean_return_holdout=1.879 from 2026-09-27,
    # not the 1.229 its own 2026-09-29 cost-aware re-validation produced.
    # Kelly sizing reads this number directly, so a stale duplicate here
    # means real position sizing bets against an edge larger than what's
    # actually been validated. Keeping only the newest trial per exact
    # settings (before ever comparing DIFFERENT settings against each
    # other) fixes this without touching load_experiment_trials()'s own
    # BH correction, which already ran across the full, undeduped log.
    trials = trials.sort_values("logged_at").drop_duplicates(
        subset=["ticker", "tf_label", "label"], keep="last")

    scored = trials[trials["verdict"] == "SCORED"].copy()

    rows = []
    for (ticker, tf_label, strategy), g in scored.groupby(["ticker", "tf_label", "strategy"]):
        survivors = g[g["survived"] == True]  # noqa: E712 — pandas bool column, not a real bool
        best = survivors if len(survivors) else g
        best_row = best.loc[best["mean_return_train"].idxmax()]
        rows.append({
            "ticker": ticker, "tf_label": tf_label, "strategy": strategy,
            "n_settings_tried": int(trials[(trials["ticker"] == ticker) & (trials["tf_label"] == tf_label)
                                            & (trials["strategy"] == strategy)].shape[0]),
            "n_scored": int(len(g)),
            "n_survived_bh": int(len(survivors)),
            "pct_same_sign": float(g["same_sign"].fillna(False).mean()),
            "pct_holdout_passed": float((g["holdout_verdict"] == "PASSED").mean()),
            "best_mean_return_train": float(best_row["mean_return_train"]),
            "best_mean_return_holdout": float(best_row["mean_return_holdout"])
            if pd.notna(best_row["mean_return_holdout"]) else None,
            "best_label": str(best_row["label"]),
            "validated": bool(len(survivors) > 0),
            "last_tested_at": str(g["logged_at"].max()),
        })
    strategy_ticker_tf = pd.DataFrame(rows)

    overview_rows = []
    for (ticker, tf_label), g in strategy_ticker_tf.groupby(["ticker", "tf_label"]):
        validated = g[g["validated"]]
        best = validated if len(validated) else g
        best_row = best.loc[best["best_mean_return_train"].idxmax()] if len(best) else None
        overview_rows.append({
            "ticker": ticker, "tf_label": tf_label,
            "n_strategies_tested": int(len(g)),
            "n_strategies_validated": int(len(validated)),
            "best_strategy": str(best_row["strategy"]) if best_row is not None else None,
            "best_return_train": float(best_row["best_mean_return_train"]) if best_row is not None else None,
            "best_return_holdout": (float(best_row["best_mean_return_holdout"])
                                     if best_row is not None and best_row["best_mean_return_holdout"] is not None
                                     else None),
        })
    ticker_overview = pd.DataFrame(overview_rows)

    con = sqlite3.connect(_DB_PATH)
    try:
        strategy_ticker_tf.to_sql("strategy_ticker_tf", con, if_exists="replace", index=False)
        ticker_overview.to_sql("ticker_overview", con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS idx_stt_ticker ON strategy_ticker_tf(ticker)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_ov_ticker ON ticker_overview(ticker)")
        con.commit()
    finally:
        con.close()


def ticker_report(ticker):
    """Everything ticker_behavior.db knows about one ticker: the overview
    (per timeframe, best validated strategy if any) and the full
    strategy-by-strategy breakdown underneath it. Two DataFrames."""
    con = sqlite3.connect(_DB_PATH)
    try:
        overview = pd.read_sql(
            "SELECT * FROM ticker_overview WHERE ticker = ? ORDER BY tf_label", con, params=(ticker,))
        detail = pd.read_sql(
            "SELECT * FROM strategy_ticker_tf WHERE ticker = ? ORDER BY tf_label, validated DESC, "
            "best_mean_return_train DESC", con, params=(ticker,))
    finally:
        con.close()
    return overview, detail


def validated_leaderboard():
    """Every (ticker, timeframe, strategy) that has ever survived full
    BH-correction — the "what actually works, anywhere" view, across the
    whole project's history."""
    con = sqlite3.connect(_DB_PATH)
    try:
        return pd.read_sql(
            "SELECT ticker, tf_label, strategy, n_scored, n_survived_bh, best_mean_return_train, "
            "best_mean_return_holdout, best_label, last_tested_at FROM strategy_ticker_tf "
            "WHERE validated = 1 ORDER BY best_mean_return_train DESC", con)
    finally:
        con.close()


if __name__ == "__main__":
    rebuild_db()
    ov = pd.read_sql("SELECT COUNT(*) AS n FROM ticker_overview", sqlite3.connect(_DB_PATH))
    print(f"ticker_behavior.db rebuilt at {_DB_PATH} — {ov['n'].iloc[0]} (ticker, timeframe) rows")
