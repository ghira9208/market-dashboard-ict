"""
Home — the front door market-dashboard-ict never actually had. Three fully
separate Streamlit processes (ICT Terminal, Research Lab, Edge Lab) already
existed and cross-linked via theme.py's own small nav strip, but nothing
explained what the three were FOR as a set or showed where things actually
stand — someone landing here cold had no single page to start from.

Every number on this page is read fresh from the same files the other three
apps themselves read/write (research/experiment_log.py's CSV, edge_lab's
own trial log + BH correction, app.py's persisted .last_state.json) — never
hardcoded, so this page can't drift out of sync with what's actually true.
"""

import json
import os

import pandas as pd
import streamlit as st

import theme
from edge_lab.agent import load_all_trials
from edge_lab.multiple_testing import benjamini_hochberg
from research.experiment_log import read_log
from research.signals import load_live_setups

st.set_page_config(page_title="Home", layout="wide", initial_sidebar_state="collapsed")
theme.inject()

# Same override research_app.py/edge_lab_app.py both already carry: theme.py
# locks the page to a fixed, non-scrolling 100vh by default (correct for
# app.py's single-screen live chart), wrong for a normal top-to-bottom page
# like this one.
st.markdown("""
<style>
html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="stMainBlockContainer"] {
    height: auto !important;
    overflow: visible !important;
    display: block !important;
}
</style>
""", unsafe_allow_html=True)

theme.render_project_nav("Home")

TONE_COLOR = {"bullish": theme.NEON_GREEN, "bearish": theme.NEON_MAGENTA, "info": theme.NEON_CYAN, "warn": theme.NEON_AMBER}


def verdict_box(headline, detail, tone="info"):
    color = TONE_COLOR.get(tone, theme.NEON_CYAN)
    icon = {"bullish": "▲", "bearish": "▼", "info": "●", "warn": "▲"}.get(tone, "●")
    st.markdown(
        f'<div class="verdict-box" style="border-color:{color};">'
        f'<span class="verdict-icon" style="color:{color};">{icon}</span>'
        f'<div><div class="verdict-headline" style="color:{color};">{headline}</div>'
        f'<div class="verdict-detail">{detail}</div></div></div>',
        unsafe_allow_html=True,
    )


st.header("Home")
st.caption("Four tools built around one question: does discretionary ICT/SMC chart reading actually have a "
           "statistical edge? Watch it live, test one read of it by hand, let an agent search across hundreds "
           "of variations on its own, or see what the live scanner is currently detecting turned into a "
           "concrete entry/stop/target — nothing on any of the four ever counts as an edge until it survives "
           "correction for how many things were tried, not just one good-looking result.")

st.markdown("---")

_LAST_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".last_state.json")


def _ict_terminal_status():
    if not os.path.exists(_LAST_STATE_FILE):
        return "No session yet — open it to start watching a chart."
    with open(_LAST_STATE_FILE) as f:
        state = json.load(f)
    layers_on = [name for name, cfg in state.get("layers", {}).items() if cfg.get("on")]
    return (f"Currently watching **{state.get('ticker', '—')}** on **{state.get('tf', '—')}** "
            f"· {len(layers_on)} layer{'s' if len(layers_on) != 1 else ''} on"
            + (f" ({', '.join(layers_on)})" if layers_on else ""))


def _research_lab_status():
    log = read_log()
    if log.empty:
        return "No studies logged yet — run one from the Backtesting tab.", "info"
    n = len(log)
    n_edge = int((log["verdict"] == "EDGE_FOUND").sum())
    n_no_edge = n - n_edge
    pct = n_edge / n * 100
    detail = (f"**{n}** studies logged — **{n_edge}** EDGE_FOUND, **{n_no_edge}** NO_EDGE ({pct:.0f}% "
              f"EDGE_FOUND). At this many attempts, roughly 5% clearing raw p<0.05 is what pure chance alone "
              f"predicts — Research Lab's own experiment log exists specifically so that comparison is "
              f"possible instead of assumed.")
    return detail, ("warn" if pct <= 6 else "bullish")


def _edge_lab_status():
    trials = load_all_trials()
    scored = [t for t in trials if t.get("verdict") == "SCORED"]
    if not scored:
        return "No trials run yet — run a batch to start searching.", 0, 0
    p_values = [t["p_value_train"] for t in scored]
    q_values, significant = benjamini_hochberg(p_values, alpha=0.05)
    validated = 0
    for t, q, sig in zip(scored, q_values, significant):
        if sig and t.get("holdout_verdict") == "PASSED":
            validated += 1
    return (f"**{len(trials)}** trials run, **{len(scored)}** scored — **{validated}** validated edge"
            f"{'s' if validated != 1 else ''} (cleared Benjamini-Hochberg correction *and* the untouched "
            f"holdout split, not just one or the other)."), len(trials), validated


def _setups_status():
    try:
        setups = load_live_setups()
    except Exception as e:
        return f"Unavailable right now ({e}).", 0
    if not setups:
        return "No current setups within their live-relevance window — open it to run a scan.", 0
    n_active = sum(1 for s in setups if s["status"] == "active")
    return (f"**{len(setups)}** current setup{'s' if len(setups) != 1 else ''} from the live-scan log "
            f"(**{n_active}** still active) — entry/stop/target, not a signal; see the page's own validation "
            f"badge per setup."), len(setups)


c1, c2, c3, c4 = st.columns(4)

with c1:
    with st.container(border=True):
        st.subheader("ICT Terminal")
        st.caption("The live chart — every FVG, order block, sweep, and structure break auto-detected and "
                   "drawn as it happens, plus a historical win-rate badge on each open zone.")
        st.markdown(_ict_terminal_status())
        st.markdown('<a href="http://localhost:8501" target="_self">Open ICT Terminal →</a>', unsafe_allow_html=True)

with c2:
    with st.container(border=True):
        st.subheader("Research Lab")
        st.caption("Hand-pick one hypothesis, hook up real historical data, and get an honest permutation-test "
                   "verdict — every run logged, win or lose, so a p-value means something later.")
        research_detail, research_tone = _research_lab_status()
        st.markdown(research_detail)
        st.markdown('<a href="http://localhost:8503" target="_self">Open Research Lab →</a>', unsafe_allow_html=True)

with c3:
    with st.container(border=True):
        st.subheader("Edge Lab")
        st.caption("A GBPUSD-only agent that samples ICT triggers plus session/volatility/weekday filters on "
                   "its own, and only calls something an edge if it clears correction across every trial ever "
                   "run, against holdout data it's never touched.")
        edge_detail, edge_n, edge_validated = _edge_lab_status()
        st.markdown(edge_detail)
        st.markdown('<a href="http://localhost:8504" target="_self">Open Edge Lab →</a>', unsafe_allow_html=True)

with c4:
    with st.container(border=True):
        st.subheader("Setups")
        st.caption("What the live scanner is detecting right now, turned into a concrete entry/stop/target box "
                   "and a validation badge — labeled setups, not signals, on purpose.")
        setups_detail, _n_setups = _setups_status()
        st.markdown(setups_detail)
        st.markdown('<a href="http://localhost:8505" target="_self">Open Setups →</a>', unsafe_allow_html=True)

st.markdown("---")

log = read_log()
total_research = len(log) if not log.empty else 0
total_edge_lab = len(load_all_trials())
total_attempts = total_research + total_edge_lab
_, _, total_validated = _edge_lab_status() if total_edge_lab else (None, 0, 0)
total_research_edges = int((log["verdict"] == "EDGE_FOUND").sum()) if not log.empty else 0

if total_attempts == 0:
    verdict_box("NOTHING TESTED YET", "Neither Research Lab nor Edge Lab has logged a study yet — this is an "
                                       "honest empty state, not an error.", tone="info")
elif total_validated == 0:
    verdict_box("NO VALIDATED EDGE YET",
                f"Across {total_attempts:,} total attempts ({total_research} hand-picked Research Lab studies, "
                f"{total_edge_lab} Edge Lab search trials), {total_research_edges} cleared a raw, uncorrected "
                f"p<0.05 somewhere — but zero have survived proper multiple-testing correction against an "
                f"untouched holdout. That's the honest current state of this whole project, not a failure of "
                f"any one page.", tone="warn")
else:
    verdict_box(f"{total_validated} VALIDATED EDGE{'S' if total_validated != 1 else ''}",
                f"Out of {total_attempts:,} total attempts, {total_validated} survived Benjamini-Hochberg "
                f"correction AND held up on untouched holdout data — see Edge Lab's leaderboard for which.",
                tone="bullish")
