"""
Shared render functions for the FVG / order block sweep engine: sweep
settings + results table, win-rate heatmap, and (for Binance-listed
crypto pairs) a real footprint chart built from actual trade-by-trade
data. Same backtest_custom_rule engine the live Rules tab's own single-
rule backtest uses (see recommender.py / backtest_engine.py).

Called from three places: backtest_app.py (the standalone Backtest page,
which pools results across every dashboard), and app.py / crypto_app.py's
own Rules tab (scoped to whatever ticker universe that dashboard already
covers) — one implementation, three call sites, all reading and writing
the exact same backtest_results.db through backtest_engine.py.
"""
import json
import math
from datetime import time as dtime

import altair as alt
import pandas as pd
import requests
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

import backtest_engine as be
from data import get_crypto_universe
from recommender import RULE_DETECTOR_LABELS, ZONE_POINT_STEPS, zone_point_label

# Mirrors app.py / crypto_app.py's own curated lists — duplicated rather
# than imported since each dashboard's own copy also drives its ticker-
# search UI (category buttons, etc.), not just the sweep universe here.
CURATED_INDICES = {
    "^GSPC": ("S&P 500", "Indices"), "^DJI": ("Dow Jones Industrial Average", "Indices"),
    "^IXIC": ("Nasdaq Composite", "Indices"), "^RUT": ("Russell 2000", "Indices"),
    "^VIX": ("CBOE Volatility Index", "Indices"), "^FTSE": ("FTSE 100", "Indices"),
    "^N225": ("Nikkei 225", "Indices"), "^GDAXI": ("DAX", "Indices"),
}
CURATED_COMMODITIES = {
    "GC=F": ("Gold Futures", "Commodities"), "SI=F": ("Silver Futures", "Commodities"),
    "CL=F": ("Crude Oil Futures", "Commodities"),
}
CURATED_FOREX = {
    "EURUSD=X": ("Euro / US Dollar", "Forex"), "GBPUSD=X": ("British Pound / US Dollar", "Forex"),
    "USDJPY=X": ("US Dollar / Japanese Yen", "Forex"), "USDCHF=X": ("US Dollar / Swiss Franc", "Forex"),
    "AUDUSD=X": ("Australian Dollar / US Dollar", "Forex"), "USDCAD=X": ("US Dollar / Canadian Dollar", "Forex"),
    "NZDUSD=X": ("New Zealand Dollar / US Dollar", "Forex"), "EURJPY=X": ("Euro / Japanese Yen", "Forex"),
    "GBPJPY=X": ("British Pound / Japanese Yen", "Forex"), "EURGBP=X": ("Euro / British Pound", "Forex"),
}
_FALLBACK_CRYPTO = {"BTC-USD": ("Bitcoin", "Crypto"), "ETH-USD": ("Ethereum", "Crypto")}


@st.cache_data(ttl=86400)
def get_ticker_universe():
    """Every ticker any of the three pages could offer for a sweep —
    indices/commodities/forex plus the full live crypto universe. A
    caller scoped to a narrower set (Markets: no crypto; Crypto: crypto
    only) builds its own restricted dict from its OWN existing ticker
    list instead of calling this — see render_sweep_tab's own docstring."""
    info = {}
    info.update(CURATED_INDICES)
    info.update(CURATED_COMMODITIES)
    info.update(CURATED_FOREX)
    info.update(_FALLBACK_CRYPTO)
    try:
        info.update({symbol: (name, "Crypto") for symbol, name in get_crypto_universe()})
    except Exception:
        pass
    return info


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


def render_sweep_tab(ticker_info, has_chart=False):
    """ticker_info: {ticker: (display_name, category)} — the caller's OWN
    universe (backtest_app.py's combined one, or a single dashboard's own
    narrower CURATED_*/crypto set), not necessarily get_ticker_universe().

    has_chart: whether the caller has a live chart to put a selected rule
    on (app.py/crypto_app.py: True; the standalone backtest_app.py page,
    with no chart at all: False, the default) — gates whether checking a
    result row here even makes sense to wire into a chart at all. Either
    way, picking a result row here is now the ONLY way to build a rule at all
    (single rule = every range narrowed to one value, same engine, same
    table) — the old dedicated Direction/Entry/Exit/Stop widgets are gone,
    replaced by this table plus the row you click on it."""
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
                 "same locked pairing as the live Rules tab.")
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

        fc1, fc2, fc3, fc4, fc5 = st.columns(5)
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

        _row_hint = ("driving the chart's Entry/Stop/Target and the Backtest panel below, immediately"
                     if has_chart else "the rule a single-combo sweep tests too")
        st.caption(f"{len(view):,} of {len(results_df):,} results, across {len(runs)} sweeps run so far. "
                   f"Click a row to select it — that's {_row_hint}, so there's no separate 'build one rule "
                   f"by hand' tool anymore.")
        # A stable, self-contained frame: sorted once here, reset_index'd so
        # the "rows" positions the selection event reports back always mean
        # "position in THIS exact object" regardless of anything the user
        # does client-side (a column-header click-sort in the browser) —
        # indexed straight back into with .iloc below, no ambiguity about
        # which ordering a returned integer refers to.
        _display_df = (
            view[["ticker", "timeframe", "direction", "entry", "exit_rule", "stop",
                  "n_trades", "wins", "win_rate", "avg_rr", "expectancy_r", "avg_net_r", "bars_used",
                  "session_label", "filters", "requested_at", "rule_json"]]
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
                "filters": st.column_config.TextColumn("quality filters"),
                "requested_at": st.column_config.DatetimeColumn("run at", format="MMM D, HH:mm"),
            },
        )

        _selected_rows = _sel_state.selection.rows if _sel_state and _sel_state.selection else []
        if _selected_rows:
            _row = _display_df.iloc[_selected_rows[0]]
            st.caption(f"Selected: **{_row['ticker']} · {_row['timeframe']} · {_row['direction']}** — "
                       f"Entry: {_row['entry']} · Exit: {_row['exit_rule']} · Stop: {_row['stop']}")
            if has_chart:
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
                        st.session_state["_bt_visualize_trades"] = True
                        st.toast(f"Now driving the chart's Entry/Stop/Target, the Backtest panel, and its "
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
                        st.caption("Already driving the chart's Entry/Stop/Target and Backtest panel below.")


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


# ------------------------------------------------------------ Footprint ----
def _binance_symbol(yahoo_ticker):
    return yahoo_ticker.replace("-USD", "").upper() + "USDT"


@st.cache_resource(show_spinner=False)
def _binance_session():
    """One pooled HTTP connection reused across every Binance call in this
    server process. Plain requests.get() opens a fresh TCP+TLS connection
    per call -- for a 150-call aggTrades pagination run that overhead alone
    measured out to well over a minute of pure connection setup, on top of
    the actual data transfer. A Session reuses the connection instead."""
    s = requests.Session()
    s.headers.update({"Connection": "keep-alive"})
    return s


@st.cache_data(ttl=20, show_spinner=False)
def _fetch_klines(symbol, interval, n_candles):
    resp = _binance_session().get("https://api.binance.com/api/v3/klines",
                                   params={"symbol": symbol, "interval": interval, "limit": n_candles}, timeout=10)
    resp.raise_for_status()
    return [{"open_time": r[0], "close_time": r[6], "open": float(r[1]), "high": float(r[2]),
              "low": float(r[3]), "close": float(r[4])} for r in resp.json()]


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_agg_trades(symbol, start_ms, end_ms, max_calls):
    """Paginates Binance's aggTrades via fromId, 1000 trades/call. A quiet
    pair (DOT: ~10 trades/candle-minute, measured directly) finishes in one
    call regardless of window size; a busy one (BTC: ~310 trades/candle-
    minute, measured directly — 150 1m-candles took 47 calls / 46.5k
    trades / 21.7s for real) can hit max_calls before reaching end_ms.
    Returns (trades, truncated) rather than silently handing back a
    partial window that LOOKS complete — the caller must tell the user."""
    session = _binance_session()
    all_trades = []
    resp = session.get("https://api.binance.com/api/v3/aggTrades",
                        params={"symbol": symbol, "startTime": start_ms, "limit": 1000}, timeout=10)
    resp.raise_for_status()
    batch = resp.json()
    all_trades.extend(batch)
    calls = 1
    truncated = False
    while batch and batch[-1]["T"] < end_ms:
        if calls >= max_calls:
            truncated = True
            break
        resp = session.get("https://api.binance.com/api/v3/aggTrades",
                            params={"symbol": symbol, "fromId": batch[-1]["a"] + 1, "limit": 1000}, timeout=10)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        all_trades.extend(batch)
        calls += 1
    return [t for t in all_trades if t["T"] <= end_ms], truncated


def _compute_value_area(totals_by_idx, poc_idx, va_pct=0.70):
    """Standard market-profile value-area algorithm: start at the point of
    control and expand outward one tick at a time, always adding whichever
    neighboring row (above or below the current range) has more volume,
    until the accumulated volume covers va_pct of the candle's total."""
    total = sum(totals_by_idx.values())
    if total <= 0:
        return poc_idx, poc_idx
    target = total * va_pct
    idxs = sorted(totals_by_idx.keys())
    pos = idxs.index(poc_idx)
    lo, hi = pos, pos
    acc = totals_by_idx[poc_idx]
    while acc < target and (lo > 0 or hi < len(idxs) - 1):
        vol_below = totals_by_idx[idxs[lo - 1]] if lo > 0 else -1
        vol_above = totals_by_idx[idxs[hi + 1]] if hi < len(idxs) - 1 else -1
        if vol_above >= vol_below:
            hi += 1
            acc += totals_by_idx[idxs[hi]]
        else:
            lo -= 1
            acc += totals_by_idx[idxs[lo]]
    return idxs[lo], idxs[hi]


def _detect_imbalance_stacks(levels_asc, ratio=3.0, min_vol=0.02):
    """Diagonal bid/ask imbalance, the standard order-flow convention: a
    price level's BUY volume is compared to the SELL volume one tick BELOW
    it (a buyer lifting the offer at this price vs. a seller who was
    passively offering one tick down), and vice versa for SELL imbalance
    one tick above. A level clearing `ratio` in either direction is
    "imbalanced"; three or more consecutive imbalanced levels on the same
    side is a "stack" — the classic footprint absorption/exhaustion
    signal. levels_asc: list of (tick_idx, buy, sell) sorted ascending."""
    n = len(levels_asc)
    buy_flag = [False] * n
    sell_flag = [False] * n
    for i, (_, buy, sell) in enumerate(levels_asc):
        if i > 0:
            sell_below = levels_asc[i - 1][2]
            if buy >= min_vol and buy > sell_below * ratio:
                buy_flag[i] = True
        if i < n - 1:
            buy_above = levels_asc[i + 1][1]
            if sell >= min_vol and sell > buy_above * ratio:
                sell_flag[i] = True

    def stacks(flags):
        out = [False] * n
        i = 0
        while i < n:
            if flags[i]:
                j = i
                while j < n and flags[j]:
                    j += 1
                if j - i >= 3:
                    for k in range(i, j):
                        out[k] = True
                i = j
            else:
                i += 1
        return out

    return stacks(buy_flag), stacks(sell_flag)


def _build_footprint_payload(symbol, interval, klines, trades, tick, imbalance_ratio):
    def tick_idx(p):
        return round(p / tick)

    trades_sorted = sorted(trades, key=lambda t: t["T"])
    ti = 0
    candles_out = []
    profile_totals = {}  # tick_idx -> {"buy","sell"}
    cum_delta = 0.0

    for k in klines:
        cell = {}
        while ti < len(trades_sorted) and trades_sorted[ti]["T"] <= k["close_time"]:
            t = trades_sorted[ti]
            if t["T"] >= k["open_time"]:
                idx = tick_idx(float(t["p"]))
                r = cell.setdefault(idx, {"buy": 0.0, "sell": 0.0})
                qty = float(t["q"])
                if t["m"]:
                    r["sell"] += qty
                else:
                    r["buy"] += qty
                pr = profile_totals.setdefault(idx, {"buy": 0.0, "sell": 0.0})
                if t["m"]:
                    pr["sell"] += qty
                else:
                    pr["buy"] += qty
            ti += 1

        base = {
            "time": pd.to_datetime(k["open_time"], unit="ms").strftime("%H:%M"),
            "open": k["open"], "high": k["high"], "low": k["low"], "close": k["close"],
        }
        if not cell:
            candles_out.append({**base, "levels": [], "poc_idx": None, "va_low_idx": None,
                                 "va_high_idx": None, "total_buy": 0.0, "total_sell": 0.0,
                                 "delta": 0.0, "cum_delta": round(cum_delta, 4)})
            continue

        totals = {idx: v["buy"] + v["sell"] for idx, v in cell.items()}
        poc_idx = max(totals.items(), key=lambda kv: kv[1])[0]
        va_low_idx, va_high_idx = _compute_value_area(totals, poc_idx)
        idxs_asc = sorted(cell.keys())
        levels_asc = [(i, cell[i]["buy"], cell[i]["sell"]) for i in idxs_asc]
        buy_stacks, sell_stacks = _detect_imbalance_stacks(levels_asc, ratio=imbalance_ratio)

        levels_out = []
        for pos, (idx, buy, sell) in enumerate(levels_asc):
            levels_out.append({
                "idx": idx, "price": round(idx * tick, 8), "buy": round(buy, 4), "sell": round(sell, 4),
                "buy_imb": buy_stacks[pos], "sell_imb": sell_stacks[pos],
            })
        total_buy = sum(v["buy"] for v in cell.values())
        total_sell = sum(v["sell"] for v in cell.values())
        delta = total_buy - total_sell
        cum_delta += delta
        candles_out.append({
            **base, "levels": levels_out, "poc_idx": poc_idx,
            "va_low_idx": va_low_idx, "va_high_idx": va_high_idx,
            "total_buy": round(total_buy, 4), "total_sell": round(total_sell, 4),
            "delta": round(delta, 4), "cum_delta": round(cum_delta, 4),
        })

    profile_out = sorted(
        [{"idx": idx, "price": round(idx * tick, 8), "buy": round(v["buy"], 4), "sell": round(v["sell"], 4)}
         for idx, v in profile_totals.items()],
        key=lambda r: r["idx"],
    )
    session_poc_idx = (max(profile_totals.items(), key=lambda kv: kv[1]["buy"] + kv[1]["sell"])[0]
                        if profile_totals else None)
    highs = [c["high"] for c in candles_out]
    lows = [c["low"] for c in candles_out]
    session = {
        "high": max(highs) if highs else None, "low": min(lows) if lows else None,
        "last": candles_out[-1]["close"] if candles_out else None,
        "total_buy": round(sum(c["total_buy"] for c in candles_out), 3),
        "total_sell": round(sum(c["total_sell"] for c in candles_out), 3),
        "total_delta": round(sum(c["delta"] for c in candles_out), 3),
        "poc_idx": session_poc_idx,
    }
    return {"symbol": symbol, "interval": interval, "tick": tick, "candles": candles_out,
            "profile": profile_out, "session": session}


_FOOTPRINT_JS = r"""
<div id="__WRAP_ID__" style="overflow-x:auto; overflow-y:hidden; border-radius:10px; border:1px solid rgba(255,255,255,0.08);">
  <canvas id="__CANVAS_ID__"></canvas>
</div>
<div id="__TOOLTIP_ID__" style="position:fixed; display:none; pointer-events:none; z-index:9999;
  background:#2c2c2e; color:#f5f5f7; border:1px solid rgba(255,255,255,0.12); border-radius:8px;
  padding:8px 10px; font:12px -apple-system,'SF Pro Display',sans-serif; box-shadow:0 8px 24px rgba(0,0,0,0.4);
  min-width:150px;"></div>
<script>
(function() {
  const payload = __PAYLOAD_JSON__;
  const candles = payload.candles;
  const profile = payload.profile;
  const session = payload.session;
  const tick = payload.tick;

  const C = {
    bg: '#1c1c1e', sep: 'rgba(255,255,255,0.09)', sepSoft: 'rgba(255,255,255,0.05)',
    text: '#f5f5f7', dim: '#98989d', buy: '#0A84FF', sell: '#FF453A',
    green: '#30D158', amber: '#FF9F0A', neutral: 'rgba(255,255,255,0.045)',
    va: 'rgba(255,255,255,0.045)',
  };
  const FONT_UI = '-apple-system, "SF Pro Display", "Segoe UI", sans-serif';
  const FONT_MONO = '"SF Mono", Menlo, Monaco, Consolas, monospace';

  function hexToRgb(hex) {
    const h = hex.replace('#', '');
    return [parseInt(h.slice(0,2),16), parseInt(h.slice(2,4),16), parseInt(h.slice(4,6),16)];
  }
  const BUY_RGB = hexToRgb(C.buy), SELL_RGB = hexToRgb(C.sell);
  function rgba(rgb, a) { return 'rgba(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ',' + a + ')'; }
  function fmt(n, d) {
    if (n === null || n === undefined || !isFinite(n)) return '—';
    return n.toLocaleString(undefined, {minimumFractionDigits: d, maximumFractionDigits: d});
  }
  function priceFmt(n) {
    const d = tick < 0.01 ? 4 : (tick < 0.1 ? 3 : (tick < 1 ? 2 : (tick < 10 ? 1 : 0)));
    return fmt(n, d);
  }
  // Volume/delta numbers, not price: a low-unit-price coin (DOT, DOGE...)
  // trades in raw-quantity amounts that run into the thousands even for
  // one price level, so these get k/M-compressed instead of full digits
  // -- shorter text means a narrower column, which is the whole point:
  // more candles fit on screen before you need to scroll.
  function volFmt(n) {
    if (n === null || n === undefined || !isFinite(n)) return '—';
    const sign = n < 0 ? '-' : '';
    const a = Math.abs(n);
    if (a >= 1000000) return sign + (a / 1000000).toFixed(a >= 10000000 ? 0 : 1) + 'M';
    if (a >= 1000) return sign + (a / 1000).toFixed(a >= 10000 ? 0 : 1) + 'k';
    if (a >= 1) return sign + a.toFixed(1);
    return sign + a.toFixed(a >= 0.1 ? 2 : 3);
  }

  if (!candles.length) return;

  // ---- global price axis (tick-index rows spanning the whole session) ----
  let hiIdx = -Infinity, loIdx = Infinity;
  candles.forEach(c => {
    if (c.high == null) return;
    hiIdx = Math.max(hiIdx, Math.round(c.high / tick));
    loIdx = Math.min(loIdx, Math.round(c.low / tick));
  });
  if (!isFinite(hiIdx)) { hiIdx = 1; loIdx = 0; }
  const rowIdxs = [];
  for (let i = hiIdx; i >= loIdx; i--) rowIdxs.push(i);
  const rowPos = new Map(rowIdxs.map((idx, i) => [idx, i]));

  // ---- layout ----
  const rowH = 20, candleLaneW = 11, numsW = 78, colW = candleLaneW + numsW;
  const marginLeft = 74, profileW = 130, headerH = 66, deltaH = 108, footerH = 30, pad = 14;
  const gridW = candles.length * colW;
  const gridH = rowIdxs.length * rowH;
  const totalW = marginLeft + gridW + profileW + pad * 2;
  const totalH = headerH + gridH + deltaH + footerH + pad;

  const wrap = document.getElementById('__WRAP_ID__');
  const canvas = document.getElementById('__CANVAS_ID__');
  const tooltip = document.getElementById('__TOOLTIP_ID__');
  const dpr = window.devicePixelRatio || 1;
  canvas.style.width = totalW + 'px';
  canvas.style.height = totalH + 'px';
  canvas.width = Math.round(totalW * dpr);
  canvas.height = Math.round(totalH * dpr);
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.textBaseline = 'alphabetic';

  ctx.fillStyle = C.bg;
  ctx.fillRect(0, 0, totalW, totalH);

  // ---- header: symbol + session stats ----
  ctx.fillStyle = C.text;
  ctx.font = '600 17px ' + FONT_UI;
  ctx.fillText(payload.symbol, pad, 27);
  ctx.font = '12px ' + FONT_UI;
  ctx.fillStyle = C.dim;
  ctx.fillText(payload.interval + ' candles · footprint', pad, 45);

  function stat(x, label, value, color) {
    ctx.font = '10px ' + FONT_UI;
    ctx.fillStyle = C.dim;
    ctx.textAlign = 'right';
    ctx.fillText(label, x, 22);
    ctx.font = '600 14px ' + FONT_MONO;
    ctx.fillStyle = color || C.text;
    ctx.fillText(value, x, 40);
    ctx.textAlign = 'left';
  }
  const deltaStr = (session.total_delta >= 0 ? '+' : '') + volFmt(session.total_delta);
  const statVals = [
    ['LAST', priceFmt(session.last), C.text],
    ['HIGH', priceFmt(session.high), C.buy],
    ['LOW', priceFmt(session.low), C.sell],
    ['VOLUME', volFmt(session.total_buy + session.total_sell), C.text],
    ['DELTA', deltaStr, session.total_delta >= 0 ? C.buy : C.sell],
  ];
  const gap = 108;
  let xr = totalW - pad;
  for (let i = statVals.length - 1; i >= 0; i--) {
    stat(xr, statVals[i][0], statVals[i][1], statVals[i][2]);
    xr -= gap;
  }

  ctx.strokeStyle = C.sep;
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, headerH - 0.5); ctx.lineTo(totalW, headerH - 0.5); ctx.stroke();

  // ---- price axis (left margin) ----
  ctx.font = '11px ' + FONT_MONO;
  ctx.fillStyle = C.dim;
  ctx.textAlign = 'right';
  rowIdxs.forEach((idx, i) => {
    if (i % 1 === 0) {
      const y = headerH + i * rowH + rowH / 2 + 4;
      ctx.fillText(priceFmt(idx * tick), marginLeft - 10, y);
    }
  });
  ctx.textAlign = 'left';

  // ---- candle grid ----
  candles.forEach((c, ci) => {
    const x0 = marginLeft + ci * colW;
    const laneX = x0, numX = x0 + candleLaneW;

    // value area tint (drawn first, under everything)
    if (c.va_low_idx != null) {
      const yTop = headerH + rowPos.get(c.va_high_idx) * rowH;
      const yBot = headerH + (rowPos.get(c.va_low_idx) + 1) * rowH;
      ctx.fillStyle = C.va;
      ctx.fillRect(numX, yTop, numsW, yBot - yTop);
    }

    // per-candle max cell volume, for cell alpha scaling
    let maxCell = 0;
    c.levels.forEach(l => { maxCell = Math.max(maxCell, l.buy, l.sell); });
    if (maxCell <= 0) maxCell = 1;

    // neutral fill for in-range-but-untraded rows
    const hiI = Math.round(c.high / tick), loI = Math.round(c.low / tick);
    for (let idx = loI; idx <= hiI; idx++) {
      const pos = rowPos.get(idx);
      if (pos === undefined) continue;
      ctx.fillStyle = C.neutral;
      ctx.fillRect(numX, headerH + pos * rowH, numsW, rowH - 1);
    }

    c.levels.forEach(l => {
      const pos = rowPos.get(l.idx);
      if (pos === undefined) return;
      const y = headerH + pos * rowH;
      const total = l.buy + l.sell;
      const dominant = l.buy >= l.sell;
      const alpha = 0.16 + 0.62 * Math.min(total / maxCell, 1);
      ctx.fillStyle = rgba(dominant ? BUY_RGB : SELL_RGB, alpha);
      ctx.fillRect(numX, y, numsW, rowH - 1);

      if (l.idx === c.poc_idx) {
        ctx.strokeStyle = C.amber;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(numX + 0.75, y + 0.75, numsW - 1.5, rowH - 2.5);
      }

      ctx.font = '10px ' + FONT_MONO;
      ctx.fillStyle = '#f5f5f7';
      ctx.textAlign = 'right';
      ctx.fillText(volFmt(l.buy), numX + numsW * 0.46, y + rowH - 6);
      ctx.fillStyle = C.dim;
      ctx.font = '9px ' + FONT_MONO;
      ctx.fillText('x', numX + numsW * 0.54, y + rowH - 6);
      ctx.textAlign = 'left';
      ctx.font = '10px ' + FONT_MONO;
      ctx.fillStyle = '#f5f5f7';
      ctx.fillText(volFmt(l.sell), numX + numsW * 0.58, y + rowH - 6);

      if (l.buy_imb) {
        ctx.fillStyle = C.buy;
        ctx.beginPath();
        ctx.moveTo(numX + numsW - 1, y + 3); ctx.lineTo(numX + numsW - 7, y + rowH/2); ctx.lineTo(numX + numsW - 1, y + rowH - 4);
        ctx.fill();
      }
      if (l.sell_imb) {
        ctx.fillStyle = C.sell;
        ctx.beginPath();
        ctx.moveTo(numX + 1, y + 3); ctx.lineTo(numX + 7, y + rowH/2); ctx.lineTo(numX + 1, y + rowH - 4);
        ctx.fill();
      }
    });

    // candle silhouette lane
    function yForPrice(p) {
      const idx = p / tick;
      // linear interpolation against the integer row grid
      const topIdx = rowIdxs[0];
      return headerH + (topIdx - idx) * rowH + rowH / 2;
    }
    const yHigh = yForPrice(c.high), yLow = yForPrice(c.low);
    const yOpen = yForPrice(c.open), yClose = yForPrice(c.close);
    const up = c.close >= c.open;
    ctx.strokeStyle = up ? C.buy : C.sell;
    ctx.fillStyle = up ? C.buy : C.sell;
    ctx.globalAlpha = 0.85;
    ctx.lineWidth = 1.5;
    const cx = laneX + candleLaneW / 2;
    ctx.beginPath(); ctx.moveTo(cx, yHigh); ctx.lineTo(cx, yLow); ctx.stroke();
    const bodyTop = Math.min(yOpen, yClose), bodyBot = Math.max(yOpen, yClose);
    ctx.fillRect(laneX + 2, bodyTop, candleLaneW - 4, Math.max(bodyBot - bodyTop, 1.5));
    ctx.globalAlpha = 1;

    // column separator + time label
    ctx.strokeStyle = C.sepSoft;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x0 + colW - 0.5, headerH); ctx.lineTo(x0 + colW - 0.5, headerH + gridH); ctx.stroke();
    ctx.font = '10px ' + FONT_MONO;
    ctx.fillStyle = C.dim;
    ctx.textAlign = 'center';
    ctx.fillText(c.time, x0 + colW / 2, headerH + gridH + 16);
    ctx.textAlign = 'left';
  });

  // outer grid border
  ctx.strokeStyle = C.sep;
  ctx.strokeRect(marginLeft + 0.5, headerH + 0.5, gridW - 1, gridH - 1);

  // ---- right margin: session volume profile ----
  const profX0 = marginLeft + gridW + 10;
  const profMaxW = profileW - 20;
  let maxProfTotal = 0;
  profile.forEach(r => { maxProfTotal = Math.max(maxProfTotal, r.buy + r.sell); });
  if (maxProfTotal <= 0) maxProfTotal = 1;
  const profByIdx = new Map(profile.map(r => [r.idx, r]));
  rowIdxs.forEach((idx, i) => {
    const r = profByIdx.get(idx);
    if (!r) return;
    const y = headerH + i * rowH;
    const buyW = (r.buy / maxProfTotal) * profMaxW;
    const sellW = (r.sell / maxProfTotal) * profMaxW;
    ctx.fillStyle = rgba(BUY_RGB, 0.55);
    ctx.fillRect(profX0, y + 2, buyW, rowH - 5);
    ctx.fillStyle = rgba(SELL_RGB, 0.55);
    ctx.fillRect(profX0 + buyW, y + 2, sellW, rowH - 5);
    if (idx === session.poc_idx) {
      ctx.fillStyle = C.amber;
      ctx.fillRect(profX0 - 4, y + 1, 3, rowH - 3);
    }
  });
  ctx.font = '10px ' + FONT_UI;
  ctx.fillStyle = C.dim;
  ctx.fillText('SESSION VOLUME PROFILE', profX0, headerH - 8);

  // ---- cumulative delta subplot ----
  const dY0 = headerH + gridH + 24;
  const dPlotH = deltaH - 30;
  let maxAbsCum = 0;
  candles.forEach(c => { maxAbsCum = Math.max(maxAbsCum, Math.abs(c.cum_delta)); });
  if (maxAbsCum <= 0) maxAbsCum = 1;
  const zeroY = dY0 + dPlotH / 2;

  ctx.strokeStyle = C.sepSoft;
  ctx.beginPath(); ctx.moveTo(marginLeft, dY0); ctx.lineTo(marginLeft, dY0 + dPlotH); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(marginLeft, zeroY); ctx.lineTo(marginLeft + gridW, zeroY); ctx.stroke();
  ctx.font = '10px ' + FONT_UI;
  ctx.fillStyle = C.dim;
  ctx.textAlign = 'right';
  ctx.fillText('CUM. DELTA', marginLeft - 10, dY0 + 10);
  ctx.textAlign = 'left';

  ctx.beginPath();
  candles.forEach((c, ci) => {
    const x = marginLeft + ci * colW + colW / 2;
    const y = zeroY - (c.cum_delta / maxAbsCum) * (dPlotH / 2 - 4);
    if (ci === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = session.total_delta >= 0 ? C.buy : C.sell;
  ctx.lineWidth = 2;
  ctx.stroke();
  candles.forEach((c, ci) => {
    const x = marginLeft + ci * colW + colW / 2;
    const y = zeroY - (c.cum_delta / maxAbsCum) * (dPlotH / 2 - 4);
    ctx.fillStyle = c.delta >= 0 ? C.buy : C.sell;
    ctx.beginPath(); ctx.arc(x, y, 2.5, 0, 7); ctx.fill();
  });

  // ---- footer legend ----
  const fY = headerH + gridH + deltaH + 18;
  ctx.font = '10.5px ' + FONT_UI;
  function legendDot(x, color, label) {
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(x, fY - 3, 4, 0, 7); ctx.fill();
    ctx.fillStyle = C.dim;
    ctx.fillText(label, x + 10, fY);
    return x + 10 + ctx.measureText(label).width + 22;
  }
  let lx = pad;
  lx = legendDot(lx, C.buy, 'buyers dominant');
  lx = legendDot(lx, C.sell, 'sellers dominant');
  lx = legendDot(lx, C.amber, 'point of control');
  ctx.fillStyle = C.dim;
  ctx.fillText('▶ ◀ stacked imbalance (' + payload.imbalance_ratio + ':1, 3+ levels) · shaded band = value area (70% of volume)', lx, fY);

  // ---- hover tooltip ----
  // Built via DOM APIs (createElement/textContent), never HTML-string
  // concatenation -- an open-angle-bracket tag pattern inside this
  // script's own JS text is enough to make Streamlit's sanitizer drop the
  // whole script block silently, confirmed directly.
  while (tooltip.firstChild) tooltip.removeChild(tooltip.firstChild);
  function ttLine(text, color, extraStyle) {
    const div = document.createElement('div');
    div.textContent = text;
    if (color) div.style.color = color;
    if (extraStyle) Object.assign(div.style, extraStyle);
    tooltip.appendChild(div);
    return div;
  }
  const ttTitle = ttLine('', null, {fontWeight: '600', marginBottom: '4px'});
  const ttBuy = ttLine('', C.buy);
  const ttSell = ttLine('', C.sell);
  const ttDelta = ttLine('', C.dim, {marginTop: '4px'});
  const ttPoc = ttLine('point of control', C.amber, {marginTop: '2px'});
  ttPoc.style.display = 'none';

  const levelByPos = new Map();
  candles.forEach((c, ci) => {
    c.levels.forEach(l => {
      const pos = rowPos.get(l.idx);
      if (pos !== undefined) levelByPos.set(ci + ':' + pos, {c, l});
    });
  });
  canvas.addEventListener('mousemove', (ev) => {
    const rect = canvas.getBoundingClientRect();
    const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
    if (mx < marginLeft || mx > marginLeft + gridW || my < headerH || my > headerH + gridH) {
      tooltip.style.display = 'none';
      return;
    }
    const ci = Math.floor((mx - marginLeft) / colW);
    const pos = Math.floor((my - headerH) / rowH);
    const hit = levelByPos.get(ci + ':' + pos);
    if (!hit) { tooltip.style.display = 'none'; return; }
    const { c, l } = hit;
    const total = l.buy + l.sell;
    const pctOfCandle = c.total_buy + c.total_sell > 0 ? (total / (c.total_buy + c.total_sell) * 100) : 0;
    const delta = l.buy - l.sell;
    ttTitle.textContent = c.time + '   ' + priceFmt(l.price);
    ttBuy.textContent = 'buy    ' + l.buy.toFixed(2);
    ttSell.textContent = 'sell   ' + l.sell.toFixed(2);
    ttDelta.textContent = 'delta ' + (delta >= 0 ? '+' : '') + delta.toFixed(2) + '  ·  ' + pctOfCandle.toFixed(0) + '% of candle';
    ttPoc.style.display = (l.idx === c.poc_idx) ? 'block' : 'none';
    tooltip.style.display = 'block';
    tooltip.style.left = (ev.clientX + 14) + 'px';
    tooltip.style.top = (ev.clientY + 14) + 'px';
  });
  canvas.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });
})();
</script>
"""


def _footprint_html(payload, imbalance_ratio):
    import json
    import uuid
    uid = uuid.uuid4().hex[:8]
    payload = {**payload, "imbalance_ratio": imbalance_ratio}
    html = (_FOOTPRINT_JS
            .replace("__WRAP_ID__", f"fp-wrap-{uid}")
            .replace("__CANVAS_ID__", f"fp-canvas-{uid}")
            .replace("__TOOLTIP_ID__", f"fp-tooltip-{uid}")
            .replace("__PAYLOAD_JSON__", json.dumps(payload)))
    return html


def render_footprint_tab(ticker_info):
    """Crypto only, always — a Markets-scoped caller (no crypto in its own
    ticker_info) gets an honest 'not available' rather than being called
    at all; app.py's Rules tab skips calling this entirely instead, since
    the reason (Yahoo-only tickers can't do this) isn't specific to what
    happens to be in ticker_info at the moment."""
    crypto_tickers = sorted(t for t, (_, cat) in ticker_info.items() if cat == "Crypto")
    st.subheader("Footprint chart")
    if not crypto_tickers:
        st.info("No crypto tickers available.")
    else:
        f1, f2, f3, f4, f5 = st.columns(5)
        with f1:
            _default_fp = "BTC-USD" if "BTC-USD" in crypto_tickers else crypto_tickers[0]
            fp_ticker = st.selectbox("Ticker", crypto_tickers, index=crypto_tickers.index(_default_fp),
                                      key="bt_fp_ticker", format_func=lambda t: f"{t} — {ticker_info[t][0]}")
        with f2:
            fp_interval = st.selectbox("Candle size", ["1m", "5m", "15m", "30m", "1h", "4h"],
                                        key="bt_fp_interval",
                                        help="Same intraday tiers as the main chart's timeframe picker. Stops at "
                                             "4h on purpose — a footprint chart reads individual trades inside "
                                             "each candle, and a daily-or-slower candle on a busy pair can hold "
                                             "hundreds of thousands of them, past what's practical to fetch.")
        with f3:
            fp_n = st.slider("Candles", 5, 150, 15, key="bt_fp_n",
                              help="Binance allows up to 1000 candles per call, but a busy pair (BTC, ETH) trades "
                                   "~300 times a minute — fetching their real trade history gets slow past a couple "
                                   "hundred candles. A quiet pair (DOT, and most smaller alts) trades far less "
                                   "often, so the same candle count loads almost instantly for those.")
        with f4:
            fp_tick = st.number_input("Price bucket size ($)", 0.001, 5000.0, 5.0, step=0.01, key="bt_fp_tick",
                                       help="How finely to group trades by price. Smaller = more rows, more detail. "
                                            "A coin under $10 (like DOT) usually wants 0.01 or smaller; BTC wants 5-50.")
        with f5:
            fp_ratio = st.number_input("Imbalance ratio", 1.5, 10.0, 3.0, step=0.5, key="bt_fp_ratio",
                                        help="How much bigger one side needs to be than the diagonal level on the "
                                             "other side to count as an imbalance. 3.0 is the common default.")

        load = st.button("Load footprint", key="bt_fp_load")
        if load:
            st.session_state["bt_fp_loaded"] = True

        if st.session_state.get("bt_fp_loaded"):
            symbol = _binance_symbol(fp_ticker)
            try:
                with st.spinner(f"Fetching real trades for {symbol} from Binance..."):
                    klines = _fetch_klines(symbol, fp_interval, fp_n)
                    if not klines:
                        st.warning(f"No data back from Binance for {symbol}.")
                        trades, payload = [], None
                    else:
                        # Scales with candle SIZE too, not just count -- a
                        # 4h candle holds ~240x the trades a 1m one does
                        # (same underlying trade rate, just a wider window),
                        # so the same candle count needs proportionally more
                        # pagination headroom the bigger the interval is.
                        interval_minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}
                        cost = fp_n * interval_minutes.get(fp_interval, 1)
                        max_calls = min(150, max(40, cost // 2))
                        trades, truncated = _fetch_agg_trades(
                            symbol, klines[0]["open_time"], klines[-1]["close_time"], max_calls)
                        payload = _build_footprint_payload(symbol, fp_interval, klines, trades, fp_tick, fp_ratio)
                if payload and any(c["levels"] for c in payload["candles"]):
                    if truncated:
                        st.warning(f"{symbol} trades fast enough that fetching all of it for {fp_n} × {fp_interval} "
                                   f"candles hit the safety limit ({max_calls} API calls) — the earliest candles "
                                   f"shown may be missing trades. Try fewer candles or a smaller candle size for a "
                                   f"complete picture.")
                    st.html(_footprint_html(payload, fp_ratio), unsafe_allow_javascript=True)
                    st.caption(f"{symbol} · {len(trades):,} real trades from Binance's public feed, bucketed into "
                               f"${fp_tick:g} price levels. Hover any cell for detail.")
                elif klines:
                    st.info("No trades came back for this window — try a larger candle size or fewer candles.")
            except requests.RequestException as e:
                st.error(f"Couldn't reach Binance: {e}")
        else:
            st.caption("Pick a ticker and click Load footprint — fetches real trades from Binance, takes a few seconds.")
