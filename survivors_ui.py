"""
The "🏆 Survivors" tab — every strategy that has ever cleared a real
statistical bar (permutation-tested, held up on untouched holdout data,
BH-corrected across the WHOLE accumulated experiments.py trial log — see
ticker_behavior.py's own docstring) for this dashboard's own ticker
universe, browsable and one click from being drawn on the chart.

Called from app.py / crypto_app.py's own Strategy tab, each scoped to
whatever ticker universe that dashboard already covers — same
one-implementation-two-call-sites convention backtest_ui.py already
uses for the Sweep/Heatmap tabs, reading a different dataset
(ticker_behavior.db instead of backtest_results.db).
"""
import sqlite3

import pandas as pd
import streamlit as st

import experiments
import ticker_behavior
from data import get_yf_ohlcv, resample_ohlc

# Raw trial-log strategy names, in plain trader language for the card
# titles — anything not listed falls back to a title-cased version of the
# raw name (see _display_name), so a brand-new strategy family shows up
# readably with zero changes needed here.
_STRATEGY_DISPLAY_NAMES = {
    "clean_expansion_retracement": "Clean Expansion → Retracement",
    "FVG": "FVG Reaction",
    "Order Block": "Order Block Reaction",
    "IFVG": "Inverse FVG Reaction",
    "Breaker Block": "Breaker Block Reaction",
}


def _display_name(strategy):
    return _STRATEGY_DISPLAY_NAMES.get(strategy, strategy.replace("_", " ").title())


@st.cache_data(ttl=120)
def _load_leaderboard():
    try:
        con = sqlite3.connect(ticker_behavior._DB_PATH)
        try:
            return pd.read_sql(
                "SELECT * FROM strategy_ticker_tf WHERE validated = 1 "
                "ORDER BY best_mean_return_train DESC", con)
        finally:
            con.close()
    except Exception:
        return pd.DataFrame()


def _fmt_pct(x):
    return f"+{x:.2%}" if pd.notna(x) and x >= 0 else (f"{x:.2%}" if pd.notna(x) else "—")


def _apply_to_chart(row, tf_key_main, timeframes, data_source):
    tf_label = ticker_behavior.normalize_tf(row["tf_label"])
    conf = timeframes.get(tf_label)
    if conf is None:
        st.error(f"Unknown timeframe {tf_label!r} — can't fetch data for it.")
        return
    try:
        df = get_yf_ohlcv(row["ticker"], period=conf["period"], interval=conf["fetch_interval"],
                           provider=data_source)
        if conf["resample"] and not df.empty:
            df = resample_ohlc(df, conf["resample"])
    except Exception as e:
        st.error(f"Couldn't fetch {row['ticker']} / {tf_label}: {e}")
        return

    trials = experiments.load_experiment_trials()
    trials = trials.copy()
    trials["tf_label"] = trials["tf_label"].map(ticker_behavior.normalize_tf)
    match = trials[(trials["ticker"] == row["ticker"]) & (trials["tf_label"] == tf_label)
                   & (trials["label"] == row["best_label"]) & (trials.get("survived", False) == True)]
    if match.empty:
        st.error("This exact validated trial isn't in the log anymore — click "
                 ":material/refresh: Refresh below and try again.")
        return
    example = experiments.latest_validated_example(df, row["ticker"], tf_label, match.iloc[0])
    if example is None:
        st.warning("No instance of this pattern exists in the history currently loaded for "
                   f"{row['ticker']} / {tf_label} — nothing to show on the chart.")
        return

    st.session_state["fvg_ticker"] = row["ticker"]
    st.session_state[tf_key_main] = tf_label
    # Honesty on the chart itself, not just in a toast that's gone after
    # this rerun — a signal from 95 bars ago labeled identically to one
    # that's live right now reads as actionable when it isn't. Same
    # "touched X bar(s) ago" phrasing the sidebar's own Validated-edge
    # scan already uses, so the two surfaces describe staleness the
    # same way.
    _status = "live now" if example["is_live"] else f"{example['bars_since_touch']} bar(s) ago, not live"
    st.session_state["_active_scan_pick"] = {
        "ticker": example["ticker"], "timeframe": example["tf_label"], "direction": example["direction"],
        "entry": example["entry_price"], "stop": example["stop_price"], "target": example["target_price"],
        "label": f"✅ Validated: {_display_name(row['strategy'])} ({_status})",
        "context_timeframe": None,
        "source_zone": {"start": example["zone_start"], "end": None,
                         "top": example["zone_top"], "bottom": example["zone_bottom"]},
        "confluence_entry_details": [], "confluence_context_details": [],
        "validated_stats": {"p_value_train": example["p_value_train"], "n_events": example["n_events"],
                             "mean_return_holdout": example["mean_return_holdout"]},
    }
    st.session_state["_scan_pick_locked"] = False
    st.session_state.pop("_active_chart_rule", None)
    _when = "live right now" if example["is_live"] else f"{example['bars_since_touch']} bar(s) ago (most recent — not currently live)"
    st.toast(f"Applied — {row['ticker']} / {tf_label}, entered {_when}.")
    st.rerun()


def _render_card(row, tf_key_main, timeframes, data_source, key_prefix):
    with st.container(border=True):
        left, right = st.columns([3, 2])
        with left:
            st.markdown(f"**{row['ticker']}** · `{ticker_behavior.normalize_tf(row['tf_label'])}` "
                        f"— {_display_name(row['strategy'])}")
            _extra = row["n_survived_bh"] - 1
            _extra_note = f" (+{_extra} other validated setting{'s' if _extra != 1 else ''})" if _extra > 0 else ""
            st.caption(f":green[✅ Survived] a strict test on {row['n_scored']} settings tried{_extra_note} — "
                       f"profitable on data it never saw, checked against every other strategy this "
                       f"project has tried.")
        with right:
            m1, m2 = st.columns(2)
            m1.metric("Backtested return", _fmt_pct(row["best_mean_return_train"]))
            m2.metric("Holdout return", _fmt_pct(row["best_mean_return_holdout"]))
        if st.button("📍 Apply to chart", key=f"survivor_apply_{key_prefix}_{row['ticker']}_{row['tf_label']}_{row['strategy']}",
                     width="stretch",
                     help="Switches the chart to this exact ticker and timeframe and draws its most "
                          "recent signal (live right now if one's active, otherwise the last time it "
                          "fired) with entry/stop/target."):
            _apply_to_chart(row, tf_key_main, timeframes, data_source)


def render_survivors_tab(ticker, tf_key_main, timeframes, ticker_info, data_source):
    """ticker: the dashboard's currently-loaded ticker (surfaced first,
    front and center). tf_key_main: TF_KEY_BY_CHART["main"] — the
    session_state key that drives the main chart's own timeframe radio.
    timeframes: the caller's own TIMEFRAMES dict (period/fetch_interval/
    resample per label). ticker_info: the caller's own ticker universe
    ({ticker: (display_name, category)}) — restricts the leaderboard to
    tickers this dashboard actually covers, same scoping backtest_ui.py's
    render_sweep_tab already does with the same parameter."""
    st.caption(
        "Strategies that passed a strict test: profitable on the data they were built from, AND on "
        "completely separate data they'd never seen before, checked against every other idea this "
        "project has ever tried so a lucky fluke can't sneak through and look like an edge. Most "
        "tickers show nothing here — that's an honest result, not a broken page."
    )
    if st.button(":material/refresh: Refresh from trial log", key="survivors_refresh",
                 help="Re-scans the full experiment log for anything new that's cleared the bar since "
                      "this page last checked."):
        ticker_behavior.rebuild_db()
        _load_leaderboard.clear()
        st.rerun()

    leaderboard = _load_leaderboard()
    if leaderboard.empty:
        st.info("Nothing has ever survived correction yet — run some deep backtests in the "
                "Strategy/Experiments tabs, then come back and hit Refresh.")
        return

    universe = set(ticker_info.keys())
    board_here = leaderboard[leaderboard["ticker"].isin(universe)]

    st.markdown(f"#### On {ticker}")
    mine = board_here[board_here["ticker"] == ticker].sort_values("best_mean_return_train", ascending=False)
    if mine.empty:
        st.caption(f"Nothing validated yet for {ticker} on any timeframe — the normal case for most tickers.")
    else:
        for _, row in mine.iterrows():
            _render_card(row, tf_key_main, timeframes, data_source, key_prefix="mine")

    st.divider()
    with st.expander(f":material/public: Every validated strategy on this watchlist ({len(board_here)})",
                      expanded=mine.empty):
        if board_here.empty:
            st.caption("Nothing validated anywhere on this watchlist yet.")
        else:
            for _, row in board_here.sort_values("best_mean_return_train", ascending=False).iterrows():
                _render_card(row, tf_key_main, timeframes, data_source, key_prefix="all")
