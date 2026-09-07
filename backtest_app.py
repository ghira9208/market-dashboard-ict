"""
Backtest — sweep the FVG / order block rule engine across every ticker
either dashboard covers, browse every result pooled together in one
sortable table, see win rate as a heatmap, and (for Binance-listed
crypto pairs) a real footprint chart built from actual trade-by-trade
data.

The render logic itself lives in backtest_ui.py, shared with app.py's and
crypto_app.py's own Rules tab (each scoped to just its own ticker
universe) — this page is the "everything pooled together" view, reading
and writing the exact same backtest_results.db via backtest_engine.py.
"""
import streamlit as st

import theme
from backtest_ui import get_ticker_universe, render_footprint_tab, render_heatmap_tab, render_sweep_tab

st.set_page_config(page_title="Backtest", layout="wide", initial_sidebar_state="collapsed")
theme.inject()
theme.render_project_nav("Backtest")

TICKER_INFO = get_ticker_universe()

st.title("Backtest")
st.caption("Sweep the FVG / order block rule engine, browse every result, and see win rate as a heatmap. "
           "Footprint charts (real buy/sell volume per price level) are available for crypto pairs only — "
           "Yahoo Finance never exposes individual trades, only Binance does. Pools results from every "
           "ticker swept here or from the Markets/Crypto dashboards' own Rules tab — same shared database.")

tab_sweep, tab_heatmap, tab_footprint = st.tabs(["Sweep", "Heatmap", "Footprint"])
with tab_sweep:
    render_sweep_tab(TICKER_INFO)
with tab_heatmap:
    render_heatmap_tab()
with tab_footprint:
    render_footprint_tab(TICKER_INFO)
