"""
Shared render functions for the FVG / order block sweep engine: sweep
settings + results table, win-rate heatmap. Same backtest_custom_rule
engine the live Backtest tab's own single-rule backtest uses (see
recommender.py / backtest_engine.py). See footprint.py for the real
Binance-trade-data footprint chart — a separate feature that used to share
this file for purely historical reasons; it has no backtesting logic of
its own and nothing here depends on it.

Called from app.py / crypto_app.py's own Strategy tab (each scoped to
whatever ticker universe that dashboard already covers) — one
implementation, two call sites, both reading and writing the exact same
backtest_results.db through backtest_engine.py.
"""
import json
from datetime import time as dtime

import altair as alt
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

import backtest_engine as be
from recommender import RULE_DETECTOR_LABELS, ZONE_POINT_STEPS, zone_point_label


# ---------------------------------------------------------------- Sweep ----
def _render_progress(run_id):
    @st.fragment(run_every=1)
    def _inner():
        job = be.get_jobs().get(run_id)
        if not job:
            return
        if job["status"] == "running":
            pct = (job["done"] / job["total"]) if job["total"] else 0.0
            st.progress(pct, text=f"{job['done']:,} / {job['total']:,} combos — "
                                   f"now on {job['current']} — {job['elapsed']:.0f}s elapsed")
            if st.button("Cancel sweep", key=f"cancel_{run_id}"):
                be.cancel_sweep(run_id)
        elif job["status"] == "completed":
            st.success(f"Sweep done — {job['done']:,} combos in {job['elapsed']:.0f}s.")
        elif job["status"] == "cancelled":
            st.warning(f"Cancelled after {job['done']:,} / {job['total']:,} combos "
                       f"(results found so far are kept).")
        elif job["status"] == "error":
            st.error(f"Sweep failed: {job['error']}")
    _inner()


def render_sweep_tab(ticker_info):
    """ticker_info: {ticker: (display_name, category)} — the caller's own
    ticker universe (app.py's CURATED_*/crypto_app.py's crypto set).
    Picking a result row here is the ONLY way to build a rule at all
    (single rule = every range narrowed to one value, same engine, same
    table) — the old dedicated Direction/Entry/Exit/Stop widgets are gone,
    replaced by this table plus the row you click on it. Checking a row
    drives the caller's own chart's Entry/Stop/Target and Backtest tab
    immediately."""
    all_tickers = sorted(ticker_info.keys())
    _default_ticker = "BTC-USD" if "BTC-USD" in ticker_info else (all_tickers[0] if all_tickers else None)
    st.subheader("Settings")
    c1, c2 = st.columns(2)
    with c1:
        tickers = st.multiselect(
            "Tickers", all_tickers, default=[_default_ticker] if _default_ticker else [], key="bt_tickers",
            format_func=lambda t: f"{t} — {ticker_info[t][0]}",
            help="Which instruments to test the rule combos against.")
        timeframes = st.multiselect(
            "Timeframes", list(be.TF_CONFIG.keys()), default=["15m"], key="bt_timeframes")
        detectors = st.multiselect(
            "Detectors", list(RULE_DETECTOR_LABELS.keys()), default=["fvg"], key="bt_detectors",
            format_func=lambda d: RULE_DETECTOR_LABELS.get(d, d),
            help="Which zone types entry/exit/stop rules are allowed to use.")
        directions = st.multiselect(
            "Direction", ["bullish", "bearish"], default=["bullish", "bearish"], key="bt_directions",
            help="Bullish always enters below price and exits/stops above; bearish is the mirror — "
                 "same locked pairing as the live Strategy tab.")
    with c2:
        entry_rank = st.slider("Entry rank range", 1, 10, (1, 1), key="bt_entry_rank",
                                help="1st-nearest zone, 2nd-nearest, etc. — how far from price to look.")
        entry_zone_point = st.select_slider(
            "Entry zone point", ZONE_POINT_STEPS, value=0.5, format_func=zone_point_label,
            key="bt_entry_zone_point",
            help="Where in the zone entry triggers, in quarters — 0% (Near) is the edge closest to "
                 "current price (shallow, fills easily, less reward), 100% (Far) is the edge farthest "
                 "away (deep, may not fill, more reward). Same meaning for bullish and bearish rules — "
                 "the zone's own bottom/top flips between them, this doesn't.")
        exit_rank = st.slider("Exit rank range", 1, 10, (1, 3), key="bt_exit_rank")
        exit_zone_point = st.select_slider(
            "Target zone point", ZONE_POINT_STEPS, value=0.5, format_func=zone_point_label,
            key="bt_exit_zone_point", help="Same idea, for the target zone.")
        stop_rank = st.slider("Stop rank range", 1, 10, (1, 2), key="bt_stop_rank")
        stop_zone_point = st.select_slider(
            "Stop zone point", ZONE_POINT_STEPS, value=0.5, format_func=zone_point_label,
            key="bt_stop_zone_point", help="Same idea, for the stop zone.")
        bar_cap = st.number_input("Bar cap (how much history per ticker)", 100, 20000, 5000, step=100,
                                   key="bt_bar_cap")
        max_scan_bars = st.number_input(
            "Zone fill-tracking window (bars)", 100, 20000, 500, step=100, key="bt_max_scan_bars",
            help="How many bars ahead we keep watching a zone to see if price ever fills it. Too short and a "
                 "zone that takes longer than this to actually fill gets treated as 'still open' for the rest "
                 "of the whole test — even long after real price action would have closed it out, which can "
                 "make a rule look like it has more valid setups than it really does. Raising this catches "
                 "more real fills but makes the sweep slower, since every zone gets watched for longer.")

    use_session = st.checkbox("Restrict entries to a session window (NY time)", key="bt_use_session")
    if use_session:
        sc1, sc2 = st.columns(2)
        with sc1:
            sess_start = st.time_input("Session start", value=dtime(9, 30), key="bt_sess_start")
        with sc2:
            sess_end = st.time_input("Session end", value=dtime(11, 30), key="bt_sess_end")
    else:
        sess_start = sess_end = None

    use_news_blackout = st.checkbox(
        "Exclude high-impact news windows", key="bt_use_news_blackout",
        help="Skips opening any new trade on a bar sitting inside a red-folder news window for this "
             "ticker's own currencies (e.g. USD news for GBPUSD=X) — the same idea as the session "
             "window above, just gated on news timing instead of time of day. Coverage: this week's "
             "real ForexFactory calendar, everything this app has itself observed live over time "
             "(grows automatically, starts empty), and US Non-Farm Payrolls (first Friday of the "
             "month, computed exactly for any date). Other recurring events (FOMC, CPI) are only "
             "covered for weeks this app was actually running to see them live — see news.py.")
    if use_news_blackout:
        nbc1, nbc2 = st.columns(2)
        with nbc1:
            nb_before = st.selectbox("Minutes before", [5, 10, 15, 30], index=2, key="bt_nb_before")
        with nbc2:
            nb_after = st.selectbox("Minutes after", [5, 10, 15, 30], index=2, key="bt_nb_after")
    else:
        nb_before = nb_after = None

    with st.expander("Quality filters (optional)"):
        st.caption("Stricter filters run SLOWER, not faster — a setup that fails the filter still pays the "
                   "full detection cost and never opens a trade to skip ahead past.")
        qc1, qc2, qc3, qc4, qc5 = st.columns(5)
        with qc1:
            min_rr = st.number_input("Min R:R", 0.0, 50.0, 0.0, step=0.5, key="bt_min_rr")
        with qc2:
            min_stop_pct = st.number_input("Min stop distance (%)", 0.0, 10.0, 0.0, step=0.1, key="bt_min_stop_pct")
        with qc3:
            min_stop_abs = st.number_input("Min stop distance (price units)", 0.0, 100000.0, 0.0, step=1.0,
                                            key="bt_min_stop_abs")
        with qc4:
            max_trades_per_session = st.number_input("Max trades/session", 0, 20, 0, step=1,
                                                       key="bt_max_trades_session")
        with qc5:
            cost_pct_input = st.number_input(
                "Round-trip cost (%)", 0.0, 5.0, 0.0, step=0.01, key="bt_cost_pct",
                help="What a real fill actually costs you — spread crossed getting in AND out, plus any "
                     "commission — as a percent of the entry price. 0 (default) matches every result you've "
                     "seen so far: a trade keeps its full raw price move. Doesn't change whether a trade hits "
                     "stop or target, only what's left once it does — see the new 'net R' column below.")

    req = {
        "tickers": tickers, "timeframes": timeframes, "detectors": detectors, "directions": directions,
        "entry_rank": list(entry_rank), "exit_rank": list(exit_rank), "stop_rank": list(stop_rank),
        "entry_zone_point": entry_zone_point, "exit_zone_point": exit_zone_point,
        "stop_zone_point": stop_zone_point, "bar_cap": int(bar_cap), "max_scan_bars": int(max_scan_bars),
        "session": {"start": sess_start.strftime("%H:%M"), "end": sess_end.strftime("%H:%M")} if use_session else None,
        "min_rr": min_rr or None,
        "min_stop_pct": (min_stop_pct / 100) if min_stop_pct else None,
        "min_stop_abs": min_stop_abs or None,
        "max_trades_per_session": int(max_trades_per_session) or None,
        "cost_pct": (cost_pct_input / 100) if cost_pct_input else None,
        "news_blackout": {"minutes_before": nb_before, "minutes_after": nb_after} if use_news_blackout else None,
    }

    ready = bool(tickers and timeframes and detectors and directions)
    if ready:
        total_combos = len(be.build_combos(req)) * len(tickers) * len(timeframes)
        est = be.estimate_seconds(req, combos_count=total_combos) * be.calibration_multiplier()
        eta = f"{est:.0f}s" if est < 90 else f"{est / 60:.1f} min"
        st.caption(f"**{total_combos:,} combos** — estimated **{eta}** to run (calibrated against past runs).")
    else:
        st.caption("Pick at least one ticker, timeframe, detector, and direction.")

    run_clicked = st.button("Run sweep", type="primary", disabled=not ready)
    if run_clicked:
        ctx = get_script_run_ctx()
        run_id = be.start_sweep(req, script_ctx=ctx)
        st.session_state["bt_active_run"] = run_id
        st.rerun()

    active_run = st.session_state.get("bt_active_run")
    if active_run and active_run in be.get_jobs():
        _render_progress(active_run)

    st.divider()
    st.subheader("Results")
    runs = be.load_runs(500)
    all_rows = be.load_results(min_n_trades=0, limit=50000)
    if not all_rows:
        st.info("No sweeps run yet — set up the settings above and click Run sweep.")
    else:
        results_df = pd.DataFrame(all_rows)
        results_df["requested_at"] = pd.to_datetime(results_df["requested_at"], unit="s")

        fc1, fc2, fc3, fc4, fc5, fc6 = st.columns(6)
        with fc1:
            f_ticker = st.selectbox("Filter: ticker", ["All"] + sorted(results_df["ticker"].unique()),
                                     key="bt_f_ticker")
        with fc2:
            f_tf = st.selectbox("Filter: timeframe", ["All"] + sorted(results_df["timeframe"].unique()),
                                 key="bt_f_tf")
        with fc3:
            f_dir = st.selectbox("Filter: direction", ["All", "bullish", "bearish"], key="bt_f_dir")
        with fc4:
            # session_label is None (SQL NULL, shows up as NaN here) for any
            # row written before this filter existed — "not recorded," kept
            # distinct from "None" (a new row that genuinely ran with no
            # session window), so it gets its own honest bucket rather than
            # being silently folded into one of the other two.
            _sessions = sorted(s for s in results_df["session_label"].dropna().unique())
            _has_legacy = results_df["session_label"].isna().any()
            f_session = st.selectbox("Filter: session", ["All"] + _sessions +
                                      (["(not recorded)"] if _has_legacy else []), key="bt_f_session")
        with fc5:
            # Same "(not recorded)" honesty as session_label above — this
            # column didn't exist before news blackout filtering shipped.
            _news_blackouts = sorted(nb for nb in results_df["news_blackout_label"].dropna().unique())
            _has_legacy_nb = results_df["news_blackout_label"].isna().any()
            f_news_blackout = st.selectbox(
                "Filter: news blackout", ["All"] + _news_blackouts +
                (["(not recorded)"] if _has_legacy_nb else []), key="bt_f_news_blackout")
        with fc6:
            f_min_n = st.number_input("Min trades", 0, 1000, 1, key="bt_f_minn")

        view = results_df[results_df["n_trades"] >= f_min_n]
        if f_ticker != "All":
            view = view[view["ticker"] == f_ticker]
        if f_tf != "All":
            view = view[view["timeframe"] == f_tf]
        if f_dir != "All":
            view = view[view["direction"] == f_dir]
        if f_session == "(not recorded)":
            view = view[view["session_label"].isna()]
        elif f_session != "All":
            view = view[view["session_label"] == f_session]
        if f_news_blackout == "(not recorded)":
            view = view[view["news_blackout_label"].isna()]
        elif f_news_blackout != "All":
            view = view[view["news_blackout_label"] == f_news_blackout]

        def _filters_summary(row):
            # NaN is truthy in Python (bool(float('nan')) is True), so a
            # plain `if row[...]:` treated every legacy row's real SQL NULL
            # -> pandas-NaN value as a set filter, rendering "min R:R nan"
            # garbage instead of the correct "no filters recorded" -- pd.notna
            # is required here, not a truthiness check.
            parts = []
            if pd.notna(row["min_rr"]) and row["min_rr"]:
                parts.append(f"min R:R {row['min_rr']:g}")
            if pd.notna(row["min_stop_pct"]) and row["min_stop_pct"]:
                parts.append(f"min stop {row['min_stop_pct'] * 100:g}%")
            if pd.notna(row["min_stop_abs"]) and row["min_stop_abs"]:
                parts.append(f"min stop {row['min_stop_abs']:g}")
            if pd.notna(row["max_trades_per_session"]) and row["max_trades_per_session"]:
                parts.append(f"max {row['max_trades_per_session']:g}/session")
            if pd.notna(row.get("cost_pct")) and row["cost_pct"]:
                parts.append(f"cost {row['cost_pct'] * 100:g}%/trade")
            return " · ".join(parts) if parts else "—"

        view = view.copy()
        view["filters"] = view.apply(_filters_summary, axis=1)
        view["session_label"] = view["session_label"].fillna("(not recorded)")
        view["news_blackout_label"] = view["news_blackout_label"].fillna("(not recorded)")
        # Win rate and avg R:R alone can't tell you if a rule is actually
        # profitable -- a 30%-win-rate rule needs avg_rr > (1-wr)/wr just to
        # break even. Expectancy folds both into one number, in R: how many
        # R a trade following this exact rule earned on average, win or
        # lose. Positive = profitable on this sample; negative = a losing
        # rule no matter how good win_rate or avg_rr look in isolation.
        # Computed the same way regardless of cost_pct -- the RAW, before-
        # cost picture, so it's directly comparable across every row
        # whether or not that sweep had a cost set.
        view["expectancy_r"] = view["win_rate"] * view["avg_rr"] - (1 - view["win_rate"])
        # avg_net_r (from the engine itself, trade-by-trade) already has
        # any round-trip cost baked in -- identical to expectancy_r when
        # cost_pct was 0/unset, strictly more honest when it wasn't. Sits
        # next to expectancy_r so the gap between the two columns IS the
        # cost's real damage, at a glance, rather than a separate lookup.
        view["avg_net_r"] = view["avg_net_r"].where(pd.notna(view["avg_net_r"]), view["expectancy_r"])

        st.caption(f"{len(view):,} of {len(results_df):,} results, across {len(runs)} sweeps run so far. "
                   f"Click a row to select it — that's driving the chart's Entry/Stop/Target and the "
                   f"Backtest tab next door, immediately, so there's no separate 'build one rule by hand' "
                   f"tool anymore.")
        # A stable, self-contained frame: sorted once here, reset_index'd so
        # the "rows" positions the selection event reports back always mean
        # "position in THIS exact object" regardless of anything the user
        # does client-side (a column-header click-sort in the browser) —
        # indexed straight back into with .iloc below, no ambiguity about
        # which ordering a returned integer refers to.
        _display_df = (
            view[["ticker", "timeframe", "direction", "entry", "exit_rule", "stop",
                  "n_trades", "wins", "win_rate", "avg_rr", "expectancy_r", "avg_net_r", "bars_used",
                  "session_label", "news_blackout_label", "filters", "requested_at", "rule_json"]]
            .sort_values("avg_net_r", ascending=False, na_position="last")
            .reset_index(drop=True)
        )
        _sel_state = st.dataframe(
            _display_df.drop(columns=["rule_json"]),
            width="stretch", height=440, on_select="rerun", selection_mode="single-row", key="bt_results_table",
            column_config={
                "exit_rule": st.column_config.TextColumn("exit"),
                "win_rate": st.column_config.ProgressColumn("win rate", format="percent", min_value=0, max_value=1),
                "avg_rr": st.column_config.NumberColumn("avg R:R", format="%.2f"),
                "expectancy_r": st.column_config.NumberColumn(
                    "expectancy (R)", format="%+.2f",
                    help="Average result per trade, in R, combining win rate and avg R:R into one number. "
                         "Positive means this exact rule made money on this sample; negative means it lost, "
                         "no matter how good win rate or avg R:R look on their own. Before any round-trip "
                         "cost — see net R for the after-cost version."),
                "avg_net_r": st.column_config.NumberColumn(
                    "net R (after cost)", format="%+.2f",
                    help="Same as expectancy, but with the round-trip cost (set in Quality filters) actually "
                         "subtracted from every trade first. Equals expectancy exactly when no cost was set for "
                         "this sweep — the gap between the two columns is what costs actually took."),
                "session_label": st.column_config.TextColumn("session"),
                "news_blackout_label": st.column_config.TextColumn("news blackout"),
                "filters": st.column_config.TextColumn("quality filters"),
                "requested_at": st.column_config.DatetimeColumn("run at", format="MMM D, HH:mm"),
            },
        )

        _selected_rows = _sel_state.selection.rows if _sel_state and _sel_state.selection else []
        if _selected_rows:
            _row = _display_df.iloc[_selected_rows[0]]
            st.caption(f"Selected: **{_row['ticker']} · {_row['timeframe']} · {_row['direction']}** — "
                       f"Entry: {_row['entry']} · Exit: {_row['exit_rule']} · Stop: {_row['stop']}")
            if pd.isna(_row["rule_json"]):
                st.caption("This result predates rule tracking — re-run this exact combo through a fresh "
                           "sweep to make it usable on the chart.")
            else:
                # No button, no separate "now also check this other
                # checkbox" step — checking a row IS the "use this"
                # action, full stop. Per direct request: too many
                # clicks between "I picked a rule" and "I can see it,"
                # and each extra click was one more chance for a canvas-
                # rendered dataframe's own click-precision quirks (see
                # this session's own testing) to eat the interaction
                # silently. Guarded on an actual CHANGE so re-selecting
                # the same already-active row doesn't re-toast or
                # stomp back over a visualize-trades checkbox the user
                # deliberately turned off for THIS same rule.
                _rule = json.loads(_row["rule_json"])
                _new_active = {
                    "ticker": _row["ticker"], "timeframe": _row["timeframe"],
                    "direction": _rule["direction"], "entry_rule": _rule["entry_rule"],
                    "exit_rule": _rule["exit_rule"], "stop_rule": _rule["stop_rule"],
                }
                if st.session_state.get("_active_chart_rule") != _new_active:
                    st.session_state["_active_chart_rule"] = _new_active
                    # Mirrors the clear app.py/crypto_app.py's own "Scan
                    # timeframes" sidebar does in the other direction —
                    # without it, picking a sweep row for the SAME
                    # ticker/timeframe a scan result was clicked for
                    # earlier would silently do nothing (the chart's own
                    # read-site checks _active_scan_pick first), reading
                    # as "this button doesn't work."
                    st.session_state.pop("_active_scan_pick", None)
                    st.session_state["_bt_visualize_trades"] = True
                    st.toast(f"Now driving the chart's Entry/Stop/Target, the Backtest tab, and its "
                             f"trade boxes — switch to {_row['ticker']} / {_row['timeframe']} to see it.")
                    # Forcing an immediate st.rerun() HERE (mid-fragment,
                    # before the rest of THIS fragment run — including
                    # the "Show these trades on the chart" checkbox
                    # further down — ever gets to render once) was
                    # tried and confirmed directly to desync that
                    # checkbox's own frontend widget: session_state
                    # held True correctly (backend logging proved it),
                    # trades genuinely rendered on the chart, but the
                    # checkbox itself displayed unchecked until manually
                    # clicked — confusing since clicking it then would
                    # have flipped a WORKING True to False. _render_chart
                    # lives in its own SEPARATE auto-ticking fragment, so
                    # closing the gap without that race has to happen at
                    # the END of _render_layer_controls instead, once
                    # this run has already completed normally — see the
                    # _new_cc/_prev_cc check there.
                else:
                    st.caption("Already driving the chart's Entry/Stop/Target and Backtest tab.")


# -------------------------------------------------------------- Heatmap ----
def render_heatmap_tab():
    """Pools whatever's in the shared results DB — not scoped to any one
    dashboard's ticker universe, since a ticker swept from Markets and one
    swept from Crypto both live in the same table."""
    st.subheader("Win rate heatmap")
    dim = st.selectbox("Group by", ["Ticker x Timeframe", "Ticker x Direction", "Timeframe x Direction"],
                        key="bt_heat_dim")
    rows = be.load_results(min_n_trades=0, limit=50000)
    if not rows:
        st.info("No results yet — run a sweep in the Sweep tab first.")
    else:
        heat_df = pd.DataFrame(rows)
        dim_map = {
            "Ticker x Timeframe": ("ticker", "timeframe"),
            "Ticker x Direction": ("ticker", "direction"),
            "Timeframe x Direction": ("timeframe", "direction"),
        }
        gx, gy = dim_map[dim]
        grouped = heat_df.groupby([gx, gy]).agg(
            n_trades=("n_trades", "sum"), wins=("wins", "sum")).reset_index()
        grouped = grouped[grouped["n_trades"] > 0].copy()
        if grouped.empty:
            st.info("No completed trades yet for this grouping.")
        else:
            grouped["win_rate"] = grouped["wins"] / grouped["n_trades"]
            cell = alt.Chart(grouped).mark_rect().encode(
                x=alt.X(f"{gx}:N", title=gx.capitalize()),
                y=alt.Y(f"{gy}:N", title=gy.capitalize()),
                color=alt.Color("win_rate:Q", title="Win rate",
                                 scale=alt.Scale(scheme="redyellowgreen", domain=[0, 1])),
                tooltip=[gx, gy, alt.Tooltip("win_rate:Q", format=".1%", title="Win rate"),
                         alt.Tooltip("n_trades:Q", title="Trades pooled")],
            )
            label = alt.Chart(grouped).mark_text(baseline="middle").encode(
                x=f"{gx}:N", y=f"{gy}:N",
                text=alt.Text("win_rate:Q", format=".0%"),
                color=alt.condition("datum.win_rate > 0.5", alt.value("#111"), alt.value("#eee")),
            )
            st.altair_chart((cell + label).properties(height=max(240, 30 * grouped[gy].nunique())),
                             width="stretch")
            st.caption("Cell = pooled win rate across every result in that group. Darker green is better; "
                       "the number in each cell is the win rate, hover for how many trades it's built from.")


