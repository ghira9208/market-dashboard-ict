"""
Setups — turns research/live_scan.py's raw detection log into concrete,
visualized setups: entry, stop, take-profit, and a details panel, for
whatever this project is currently watching (research/agent_config.json).

Deliberately named "setups", not "signals": research/signals.py's own
docstring (and live_scan.py's before it) is explicit that no pattern family
here has a statistically validated edge outside what edge_lab has actually
proven with permutation testing + Benjamini-Hochberg correction — and
edge_lab has only ever tested GBPUSD, a market this page doesn't even scan.
Calling these "signals" or "recommendations" would overstate that. Each
setup still shows a validation badge sourced straight from edge_lab's trial
log so the honest state is visible on the page, not just in a comment.

Entry/stop/target math and its chart treatment are NOT reinvented here —
they're research/signals.compute_setup, the same formula already
implemented independently in research_app.py and edge_lab_app.py for a
single backtest example, now applied to the live detection log.
"""

import os
import subprocess

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import theme
from research.agent_config import load_config, save_config
from research.data_loader import INTERVAL_LABELS, INTERVAL_MAX_PERIOD, load_history
from research.live_scan import LIVE_EVENTS_PATH, refresh_caches, scan_for_new_structures
from research.signals import EDGE_LAB_TICKER, EVENT_TYPE_LABELS, LOOKBACK, load_live_setups, validation_badge

st.set_page_config(page_title="Setups", layout="wide", initial_sidebar_state="collapsed")
theme.inject()

# Same override research_app.py/edge_lab_app.py/home_app.py all carry:
# theme.py locks the page to a fixed 100vh by default, correct for app.py's
# single-screen live chart, wrong for a normal top-to-bottom page.
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

theme.render_project_nav("Setups")

TONE_COLOR = {"bullish": theme.NEON_GREEN, "bearish": theme.NEON_MAGENTA, "info": theme.NEON_CYAN, "warn": theme.NEON_AMBER}
STATUS_TONE = {"active": "info", "hit_tp": "bullish", "hit_sl": "bearish", "expired": "warn"}
STATUS_LABELS = {"active": "Active", "hit_tp": "Hit TP", "hit_sl": "Hit SL", "expired": "Expired"}


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


def _hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


PLIST_LABEL = "com.marketdashboard.livescan"
PLIST_PATH = os.path.expanduser(f"~/Library/LaunchAgents/{PLIST_LABEL}.plist")


def _launchd_status():
    try:
        result = subprocess.run(["launchctl", "print", f"gui/{os.getuid()}/{PLIST_LABEL}"],
                                 capture_output=True, text=True, timeout=5)
    except Exception as e:
        return {"scheduled": False, "state": f"error: {e}"}
    if result.returncode != 0:
        return {"scheduled": False, "state": "not scheduled"}
    state = "running" if "state = running" in result.stdout else "idle (waiting for next hour)"
    return {"scheduled": True, "state": state}


@st.cache_data(ttl=120)
def _cached_setups():
    return load_live_setups()


st.header("🎯 Setups")
st.caption("Detected patterns from the live_scan agent, turned into a concrete entry/stop/target box — "
           "**not trade signals or recommendations**. No pattern family here has a validated statistical edge "
           "outside what Edge Lab has proven on GBPUSD (which this page doesn't even scan) — the validation "
           "badge on each setup says exactly where that stands, instead of implying more confidence than "
           "actually exists.")

cfg = load_config()
launchd = _launchd_status()
last_logged_at = None
n_logged_total = 0
if os.path.exists(LIVE_EVENTS_PATH):
    try:
        _tail = pd.read_csv(LIVE_EVENTS_PATH, usecols=["logged_at"])
        n_logged_total = len(_tail)
        last_logged_at = _tail["logged_at"].max()
    except Exception:
        pass

s1, s2, s3, s4 = st.columns(4)
s1.metric("Live-scan agent", "Enabled" if cfg["enabled"] else "Disabled")
s2.metric("Scheduled (launchd)", launchd["state"] if launchd["scheduled"] else "Off")
s3.metric("Last scan logged", last_logged_at.split("T")[0] + " " + last_logged_at.split("T")[1][:5] if last_logged_at else "—")
s4.metric("Structures ever logged", f"{n_logged_total:,}")

b1, b2, _ = st.columns([1, 1, 3])
run_clicked = b1.button("⚡ Scan now", type="primary", width="stretch",
                         help="Runs the same refresh + detect pass live_scan.py's hourly job runs, in this "
                              "process, then reloads the setups below.")
if b2.button("🔄 Reload setups", width="stretch"):
    _cached_setups.clear()
    st.rerun()

if run_clicked:
    with st.status("Running live scan...", expanded=True) as status_box:
        st.write(f"{len(cfg['tickers'])} tickers × {len(cfg['intervals'])} intervals")
        refresh_caches(cfg["tickers"], cfg["intervals"], provider=cfg["provider"])
        new_rows = scan_for_new_structures(cfg["tickers"], cfg["intervals"], cfg["detectors"])
        status_box.update(label=f"Done — {len(new_rows)} new structure(s) logged.", state="complete")
    _cached_setups.clear()
    st.rerun()

st.markdown("---")

setups = _cached_setups()

if not setups:
    verdict_box("NO CURRENT SETUPS", "Either the live-scan agent hasn't logged anything within the live-"
                                      "relevance window for its watched tickers/intervals yet, or nothing "
                                      "detected recently still resolves to a valid entry/stop. Click "
                                      "\"Scan now\" above, or check the Research Lab's Agent tab for the full "
                                      "detection log.", tone="warn")
    st.stop()

df = pd.DataFrame(setups)
df["pattern"] = df["event_type"].map(EVENT_TYPE_LABELS).fillna(df["event_type"])
badges = df.apply(lambda r: validation_badge(r["event_type"], r["ticker"]), axis=1)
df["validation_label"] = [b[0] for b in badges]
df["validation_tone"] = [b[1] for b in badges]

st.markdown("**Filters**")
f1, f2, f3, f4 = st.columns(4)
tickers_sel = f1.multiselect("Ticker", sorted(df["ticker"].unique()), default=sorted(df["ticker"].unique()))
intervals_sel = f2.multiselect("Interval", sorted(df["interval"].unique(), key=lambda i: list(LOOKBACK).index(i) if i in LOOKBACK else 99),
                                default=sorted(df["interval"].unique()), format_func=lambda i: INTERVAL_LABELS.get(i, i))
patterns_sel = f3.multiselect("Pattern", sorted(df["pattern"].unique()), default=sorted(df["pattern"].unique()))
status_sel = f4.multiselect("Status", list(STATUS_LABELS), default=["active"], format_func=lambda s: STATUS_LABELS[s])

filtered = df[df["ticker"].isin(tickers_sel) & df["interval"].isin(intervals_sel) &
              df["pattern"].isin(patterns_sel) & df["status"].isin(status_sel)]
filtered = filtered.sort_values("entry_time", ascending=False)

st.caption(f"{len(filtered)} of {len(df)} current setups shown (within each interval's live-relevance window — "
           "see research/signals.LOOKBACK).")

if filtered.empty:
    st.info("No setups match the current filters.")
    st.stop()

table = filtered.copy()
table["Entry"] = table["entry_time"].dt.strftime("%b %d, %H:%M")
table["R:R"] = table["reward_risk"].map(lambda x: f"{x:.1f}R")
table["Entry price"] = table["entry_price"].map(lambda x: f"{x:.5g}")
table["Stop"] = table["sl_price"].map(lambda x: f"{x:.5g}")
table["Target"] = table["tp_price"].map(lambda x: f"{x:.5g}")
table["Current"] = table["current_price"].map(lambda x: f"{x:.5g}")
table["Move"] = table["distance_pct"].map(lambda x: f"{x:+.2f}%")
table["Status"] = table["status"].map(STATUS_LABELS)
table["Validation"] = table["validation_label"]

display_cols = ["Entry", "ticker", "interval", "pattern", "direction", "Entry price", "Stop", "Target",
                 "R:R", "Current", "Move", "Status", "Validation"]
table_display = table[display_cols].rename(columns={"ticker": "Ticker", "interval": "Interval",
                                                       "pattern": "Pattern", "direction": "Direction"})


def _hl_status(row):
    tone = STATUS_TONE.get({v: k for k, v in STATUS_LABELS.items()}[row["Status"]], "info")
    color = _hex_to_rgba(TONE_COLOR[tone], 0.12)
    return [f"background-color: {color}"] * len(row)


st.dataframe(table_display.style.apply(_hl_status, axis=1), hide_index=True, width="stretch")

st.markdown("---")
st.subheader("Setup detail")

filtered = filtered.reset_index(drop=True)
labels = [f"{r.ticker} · {INTERVAL_LABELS.get(r.interval, r.interval)} · {EVENT_TYPE_LABELS.get(r.event_type, r.event_type)} "
          f"({r.direction}) · entry {r.entry_time.strftime('%b %d, %H:%M')}" for r in filtered.itertuples()]
picked = st.selectbox("Pick a setup to chart", options=list(range(len(filtered))), format_func=lambda i: labels[i])
setup = filtered.iloc[picked]

d1, d2, d3, d4, d5 = st.columns(5)
d1.metric("Entry", f"{setup['entry_price']:.5g}")
d2.metric("Stop", f"{setup['sl_price']:.5g}", delta=f"{(setup['sl_price'] - setup['entry_price']) / setup['entry_price'] * 100:+.2f}%")
d3.metric("Target", f"{setup['tp_price']:.5g}", delta=f"{(setup['tp_price'] - setup['entry_price']) / setup['entry_price'] * 100:+.2f}%")
d4.metric("Reward:risk", f"{setup['reward_risk']:.1f}R")
d5.metric("Status", STATUS_LABELS[setup["status"]])

tone = setup["validation_tone"]
verdict_box(f"Validation — {setup['pattern']}", setup["validation_label"], tone=tone)
if setup["ticker"] != EDGE_LAB_TICKER:
    st.caption("Edge Lab's search agent (edge_lab/agent.py) only ever runs against GBPUSD — a validated result "
               "there is evidence the PATTERN can work somewhere, not that it's been proven on this ticker.")


@st.cache_data(ttl=300)
def _chart_history(ticker, interval):
    return load_history(ticker, INTERVAL_MAX_PERIOD[interval], interval)


hist = _chart_history(setup["ticker"], setup["interval"])
pos_by_time = {t: i for i, t in enumerate(hist.index)}
entry_pos = pos_by_time.get(setup["entry_time"])

if entry_pos is None:
    st.caption("(chart unavailable — entry bar fell outside the currently cached window)")
else:
    window_bars = 60
    lo = max(0, entry_pos - window_bars)
    hi = min(len(hist), max(entry_pos + window_bars, (pos_by_time.get(setup["resolved_at"]) or entry_pos) + 10))
    window = hist.iloc[lo:hi]

    # Categorical x-axis, not a real datetime one, matching research_app.py's
    # render_example_event_chart — a real datetime axis renders actual
    # calendar gaps (weekends, closed markets) as literal blank space.
    x_labels = [t.strftime("%b %d, %H:%M") for t in window.index]
    label_by_time = dict(zip(window.index, x_labels))

    def _cat(t):
        return label_by_time.get(t) if t is not None and pd.notna(t) else None

    direction = setup["direction"]
    fig = go.Figure(data=[go.Candlestick(
        x=x_labels, open=window["Open"], high=window["High"], low=window["Low"], close=window["Close"],
        increasing_line_color=theme.NEON_GREEN, decreasing_line_color=theme.NEON_MAGENTA,
        increasing_fillcolor=theme.NEON_GREEN, decreasing_fillcolor=theme.NEON_MAGENTA, name="",
    )])

    entry_x = _cat(setup["entry_time"])
    end_x = _cat(setup["resolved_at"]) or x_labels[-1]
    if entry_x:
        entry_price = setup["entry_price"]
        for label, price, color in (("Take Profit", setup["tp_price"], theme.NEON_GREEN),
                                     ("Stop Loss", setup["sl_price"], theme.NEON_MAGENTA)):
            pct = (price - entry_price) / entry_price * 100
            fig.add_trace(go.Scatter(
                x=[entry_x, end_x, end_x, entry_x, entry_x],
                y=[entry_price, entry_price, price, price, entry_price],
                fill="toself", mode="lines", line=dict(width=0),
                fillcolor=_hex_to_rgba(color, 0.18), hoveron="fills",
                hoverinfo="text", text=f"{label}: {price:.5g} ({pct:+.3f}%)", showlegend=False, name="",
            ))
        fig.add_trace(go.Scatter(
            x=[entry_x], y=[entry_price], mode="markers",
            marker=dict(symbol="circle", size=9, color=theme.NEON_AMBER, line=dict(color="#000", width=1)),
            hoverinfo="text", text=f"Entry: {entry_price:.5g} @ {setup['entry_time']}", showlegend=False, name="",
        ))
        fig.add_vline(x=entry_x, line_color=theme.NEON_AMBER, line_dash="dot", line_width=1.5)
        if setup["resolved_at"] is not None and pd.notna(setup["resolved_at"]):
            resolved_x = _cat(setup["resolved_at"])
            if resolved_x:
                fig.add_vline(x=resolved_x, line_color=theme.NEON_CYAN, line_dash="dot", line_width=1.5)

    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=10, b=10),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(type="category", tickangle=0, nticks=8),
        xaxis_rangeslider_visible=False, showlegend=False, font=dict(color="#c9d6e3", size=11),
    )
    st.plotly_chart(fig, width="stretch")

st.markdown("---")
with st.expander("Live-scan agent settings"):
    st.caption("Same config research_app.py's Agent tab edits (research/agent_config.json) — changing it here "
               "affects the next scheduled run there too, and vice versa.")
    st.json(cfg)
    st.markdown('<a href="http://localhost:8503" target="_self">Open full agent controls in Research Lab →</a>',
                unsafe_allow_html=True)
