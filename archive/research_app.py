"""
Research Lab — customizable event-study backtesting UI, standalone from the
live ICT terminal (app.py) on its own port, same separation-of-concerns
reasoning as this project's own README gives for being standalone from
market-dashboard: a slow permutation-test run here should never be able to
affect the live chart, and vice versa.

Three tabs: a multi-asset Research overview (the crypto sweep + experiment
log), a Backtesting workbench (hook up one dataset once, then every
detector's own settings/results, all visible at once), and a control panel
for the live_scan agent (research/live_scan.py, scheduled via launchd — see
its own docstring). Every bit of state — agent status, detector results,
logs — renders HERE, on the page, not just reported in chat. A background
job or a finished computation nobody can see is functionally invisible no
matter how many times its status gets described in words.
"""

import os
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import theme
from data import resample_ohlc
from fvg import detect_structure_breaks
from research.agent_config import load_config, save_config
from research.backtest import run_event_study
from research.data_loader import INTERVAL_LABELS, INTERVAL_MAX_PERIOD, load_history
from research.events import HYPOTHESES
from research.experiment_log import log_run, read_log
from research.histdata_import import HISTDATA_PERIOD, HISTDATA_PROVIDER
from research.live_scan import LIVE_EVENTS_PATH, refresh_caches, scan_for_new_structures
from research.sessions import KILL_ZONES, filter_events_to_sessions
from research.spread import estimate_spread_bps
from research.sweep import AVAILABILITY_PATH, BACKTEST_SWEEP_PATH, CRYPTO_TICKERS, LONG_INTERVALS

# histdata.com only ever gives 1-minute bars — everything "1-minute and
# above" on that source is built by resampling the imported base, the same
# way the live chart already builds its own 4h timeframe from Yahoo's 60m
# (see app.py's TIMEFRAMES). None for "1m" means "use the base as-is."
FX_RESAMPLE_RULES = {"1m": None, "5m": "5min", "15m": "15min", "30m": "30min", "60m": "1h", "1d": "1D"}


def load_backtest_df(ticker, interval, provider):
    if provider == HISTDATA_PROVIDER:
        base = load_history(ticker, HISTDATA_PERIOD, "1m", provider=HISTDATA_PROVIDER)
        rule = FX_RESAMPLE_RULES.get(interval)
        return base if rule is None else resample_ohlc(base, rule)
    return load_history(ticker, INTERVAL_MAX_PERIOD[interval], interval, provider=provider)

st.set_page_config(page_title="Research Lab", layout="wide", initial_sidebar_state="collapsed")
theme.inject()

# theme.py locks html/body/stApp/stMain to a fixed 100vh with overflow:
# hidden — correct for app.py's single-screen, nothing-scrolls chart
# terminal (see its own comment: "Fullscreen, non-scrolling terminal"), but
# this page is a normal top-to-bottom form + results + growing experiment
# log, genuinely taller than one viewport. Both pages import the same
# theme.py, so this override lives here (scoped to this page only via
# source order — injected after theme.inject(), same selectors, later wins)
# rather than changing the shared file and risking app.py's layout.
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

TONE_COLOR = {"bullish": theme.NEON_GREEN, "bearish": theme.NEON_MAGENTA, "info": theme.NEON_CYAN, "warn": theme.NEON_AMBER}


def _hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# Kill zones checked for the "or exits the session" stop condition below —
# NY AM and London Open specifically (the two the user named), not the
# whole KILL_ZONES dict — an FVG retracement taken outside either doesn't
# have an obvious single "the session" to bound it against.
_MFE_SESSIONS = ["NY AM (9:30–11:30 ET)", "London Open (02:00–05:00 ET)"]


def compute_mfe(df, touch_pos, direction, max_lookahead=300):
    """Maximum Favorable Excursion: starting at the touch bar, how far did
    price run in the trade's own direction before EITHER a genuine reversal
    (the first CHoCH against this direction — reusing the same market-
    structure detector the chart's own Market Structure layer draws, not a
    hand-rolled "price turned" heuristic) OR the session containing the
    touch bar closed out (NY AM / London Open — whichever one the touch
    actually fell inside; bars outside both have no session bound, only
    the CHoCH one applies) — whichever comes first. Returns None if the
    touch bar has no room to look forward at all."""
    idx = df.index
    n = len(df)
    if touch_pos >= n - 1:
        return None
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    scan_end = min(n, touch_pos + 1 + max_lookahead)

    touch_ny = idx[touch_pos].tz_convert("America/New_York")
    session_name, session_end_t = None, None
    for name in _MFE_SESSIONS:
        for start, end in KILL_ZONES[name]:
            if start <= touch_ny.time() <= end:
                session_name, session_end_t = name, end
                break
        if session_name:
            break

    stop_pos = scan_end - 1
    stop_reason = "end of available data"
    if session_end_t is not None:
        for j in range(touch_pos + 1, scan_end):
            if idx[j].tz_convert("America/New_York").time() > session_end_t:
                stop_pos, stop_reason = j - 1, f"{session_name} session close"
                break

    opposite = "bearish" if direction == "bullish" else "bullish"
    pos_by_time = {idx[j]: j for j in range(touch_pos + 1, scan_end)}
    for b in detect_structure_breaks(df.iloc[touch_pos:scan_end]):
        if b["structure"] == "CHoCH" and b["type"] == opposite and b["end"] in pos_by_time:
            b_pos = pos_by_time[b["end"]]
            if b_pos < stop_pos:
                stop_pos, stop_reason = b_pos, "CHoCH reversal"

    stop_pos = max(stop_pos, touch_pos)
    if direction == "bullish":
        mfe_pos = touch_pos + int(np.argmax(high[touch_pos:stop_pos + 1]))
        mfe_price = float(high[mfe_pos])
    else:
        mfe_pos = touch_pos + int(np.argmin(low[touch_pos:stop_pos + 1]))
        mfe_price = float(low[mfe_pos])

    return {"mfe_pos": mfe_pos, "mfe_time": idx[mfe_pos], "mfe_price": mfe_price,
            "stop_pos": stop_pos, "stop_reason": stop_reason, "session": session_name}


def render_example_picker(events, key_prefix, max_examples=50):
    """Browse-through-examples control, shared by every detector's chart —
    a slider over the most recent `max_examples` events plus Prev/Next
    stepper buttons, all driving one shared session_state index so the
    three controls (and a page rerun) stay in sync. Returns the SELECTED
    ROW (a Series), or None if there are zero events to browse. Deliberately
    the most recent N, not a random/uniform sample across all history —
    the recent ones are what "how does this indicator work RIGHT NOW"
    actually means, and 50 recent + the raw-events table below covers the
    rest for anyone who wants deeper history."""
    if events.empty:
        return None
    pool = events.tail(max_examples).reset_index(drop=True)
    n = len(pool)
    idx_key = f"{key_prefix}_ex_idx"
    st.session_state.setdefault(idx_key, n - 1)  # default to the most recent
    st.session_state[idx_key] = min(st.session_state[idx_key], n - 1)  # clamp if a re-run shrank the pool

    c1, c2, c3 = st.columns([1, 5, 1])
    if c1.button("◀", key=f"{key_prefix}_ex_prev", disabled=st.session_state[idx_key] <= 0, use_container_width=True):
        st.session_state[idx_key] -= 1
    if c3.button("▶", key=f"{key_prefix}_ex_next", disabled=st.session_state[idx_key] >= n - 1, use_container_width=True):
        st.session_state[idx_key] += 1
    if n > 1:
        c2.slider(f"Example {st.session_state[idx_key] + 1} of {n} (most recent {n})", 0, n - 1,
                  key=idx_key, label_visibility="visible")
    else:
        c2.caption("Only 1 example available")

    return pool.iloc[st.session_state[idx_key]]


def render_example_event_chart(bt_df, ex, direction_col, window_bars=50, touch_col=None,
                                show_mfe=False, zone_start_col=None):
    """Small candlestick chart around ONE real detected instance (`ex`,
    picked by render_example_picker above — not always the most recent
    anymore) — so "how does this indicator work" has an actual answer
    on-screen instead of just numbers in a table.

    The zone rectangle spans from zone_start_col (the gap's own formation
    time, FVG's "gap_formed_at") to the touch point — NOT the whole visible
    chart. Confirmed directly this was worth fixing: drawing it edge-to-
    edge of the window made price action from BEFORE the gap even existed
    look like it was already "inside" the zone, which is exactly backwards
    — the touch-detection logic was always correct (first bar that re-
    enters the zone after formation), only the rectangle's own left edge
    was misleading.

    When the touch fell inside a recognized kill-zone session (NY AM /
    London Open — see compute_mfe), the chart window is the WHOLE session
    plus 1 hour of pre-session lead-in, not a fixed bar count — seeing the
    full session in context matters more here than a fixed window size.
    Falls back to a window_bars-wide bar-count window otherwise, still
    widened left to always include the zone's own formation bar."""
    pos_by_time = {t: i for i, t in enumerate(bt_df.index)}
    entry_pos = pos_by_time.get(ex["entry_time"])
    if entry_pos is None:
        st.caption("(example chart unavailable — entry bar fell outside the currently loaded window)")
        return

    touch_pos = pos_by_time.get(ex[touch_col]) if touch_col else None
    mfe = compute_mfe(bt_df, touch_pos, ex[direction_col]) if (show_mfe and touch_pos is not None) else None
    zone_start_time = ex.get(zone_start_col) if zone_start_col else None
    zone_start_pos = pos_by_time.get(zone_start_time) if pd.notna(zone_start_time) else None

    window = None
    if mfe is not None and mfe.get("session"):
        touch_ny = bt_df.index[touch_pos].tz_convert("America/New_York")
        s_start, s_end = next((s, e) for s, e in KILL_ZONES[mfe["session"]] if s <= touch_ny.time() <= e)
        day = touch_ny.normalize()
        win_start = (day + pd.Timedelta(hours=s_start.hour, minutes=s_start.minute) - pd.Timedelta(hours=1)).tz_convert(bt_df.index.tz)
        win_end = (day + pd.Timedelta(hours=s_end.hour, minutes=s_end.minute)).tz_convert(bt_df.index.tz)
        win_end = max(win_end, ex["exit_time"], mfe["mfe_time"])
        lo, hi = int(bt_df.index.searchsorted(win_start)), int(bt_df.index.searchsorted(win_end)) + 1
        if zone_start_pos is not None:
            lo = min(lo, zone_start_pos)
        window = bt_df.iloc[max(0, lo):min(len(bt_df), hi)]

    if window is None:
        center_pos = touch_pos if touch_pos is not None else entry_pos
        lo = max(0, center_pos - window_bars)
        hi = min(len(bt_df), center_pos + window_bars)
        if mfe is not None:
            hi = max(hi, min(len(bt_df), mfe["mfe_pos"] + 5))
        if zone_start_pos is not None:
            lo = min(lo, zone_start_pos)
        window = bt_df.iloc[lo:hi]

    direction = ex[direction_col]
    zone_color = theme.NEON_GREEN if direction == "bullish" else theme.NEON_MAGENTA

    # Categorical x-axis, not a real datetime one — a datetime axis renders
    # actual CALENDAR time, so any closed-market gap (a weekend for FX, a
    # holiday, overnight for stocks) shows up as literal blank space, with
    # every real candle squeezed into whatever's left. Confirmed directly:
    # a GBPUSD example spanning a Fri-close-to-Mon-open weekend rendered as
    # two tiny clusters of candles either side of a ~60-hour void. Category
    # mode treats each bar as the next discrete slot regardless of the real
    # time between it and its neighbor — the standard fix for candlestick
    # charts over non-continuous markets. Every x-reference below has to be
    # one of these exact label strings, not the underlying Timestamp, or
    # Plotly can't place it on a categorical axis at all.
    x_labels = [t.strftime("%b %d, %H:%M") for t in window.index]
    label_by_time = dict(zip(window.index, x_labels))

    def _cat(t):
        return label_by_time.get(t) if pd.notna(t) else None

    fig = go.Figure(data=[go.Candlestick(
        x=x_labels, open=window["Open"], high=window["High"], low=window["Low"], close=window["Close"],
        increasing_line_color=theme.NEON_GREEN, decreasing_line_color=theme.NEON_MAGENTA,
        increasing_fillcolor=theme.NEON_GREEN, decreasing_fillcolor=theme.NEON_MAGENTA,
        name="",
    )])
    has_zone = pd.notna(ex["zone_top"]) and pd.notna(ex["zone_bottom"]) and ex["zone_top"] != ex["zone_bottom"]
    if has_zone:
        zone_x0 = _cat(zone_start_time) or x_labels[0]
        # Right edge: the touch point when we have one (FVG), otherwise the
        # entry itself — NEVER the window's right edge. A detected zone
        # (order block, equal-highs/lows cluster, ...) is "behind" the
        # trade by construction — it's what triggered the entry — so
        # letting it visually run past entry/exit made it look like the
        # SAME shaded region as the Take Profit box below, when they're two
        # unrelated things drawn in the same color.
        zone_x1 = (_cat(ex[touch_col]) if touch_col else None) or _cat(ex["entry_time"]) or x_labels[-1]
        fig.add_shape(type="rect", xref="x", yref="y",
                      x0=zone_x0, x1=zone_x1, y0=ex["zone_bottom"], y1=ex["zone_top"],
                      fillcolor=_hex_to_rgba(zone_color, 0.15), line=dict(color=zone_color, width=1))
    if _cat(ex["entry_time"]):
        fig.add_vline(x=_cat(ex["entry_time"]), line_color=theme.NEON_AMBER, line_dash="dot", line_width=1.5)
    if _cat(ex["exit_time"]):
        fig.add_vline(x=_cat(ex["exit_time"]), line_color=theme.NEON_CYAN, line_dash="dot", line_width=1.5)

    # Classic entry/stop-loss/take-profit box — the risk a trader would
    # actually mark up before taking this, not just the forward-return
    # math. Stop sits beyond the zone that defines the setup (price
    # re-entering past it invalidates the whole read); BOS/CHoCH carry no
    # real zone (events.py duplicates their flat "level" into both
    # zone_top/zone_bottom), so those fall back to a stop sized off this
    # window's own average bar range instead of a zero-width one. Target
    # is a fixed 2R — the standard reward:risk these setups are framed
    # against, not a claim about where price actually went (raw_return/MFE
    # already show that). Drawn as filled Scatter traces, not add_shape —
    # shapes can't carry hover text in Plotly, traces can (hoveron="fills"
    # triggers over the whole filled area, not just its outline).
    sl_tp_caption = ""
    entry_x, exit_x = _cat(ex["entry_time"]), _cat(ex["exit_time"])
    if entry_x and exit_x:
        entry_price = float(ex["entry_price"])
        if pd.notna(ex["zone_top"]) and pd.notna(ex["zone_bottom"]) and ex["zone_top"] != ex["zone_bottom"]:
            sl_price = ex["zone_bottom"] if direction == "bullish" else ex["zone_top"]
        else:
            atr = float((window["High"] - window["Low"]).mean())
            sl_price = entry_price - atr * 2 if direction == "bullish" else entry_price + atr * 2
        risk = abs(entry_price - sl_price)
        if risk > 0:
            tp_price = entry_price + risk * 2 if direction == "bullish" else entry_price - risk * 2
            for label, price, color in (("Take Profit", tp_price, theme.NEON_GREEN), ("Stop Loss", sl_price, theme.NEON_MAGENTA)):
                pct = (price - entry_price) / entry_price * 100
                fig.add_trace(go.Scatter(
                    x=[entry_x, exit_x, exit_x, entry_x, entry_x],
                    y=[entry_price, entry_price, price, price, entry_price],
                    fill="toself", mode="lines", line=dict(width=0),
                    fillcolor=_hex_to_rgba(color, 0.18), hoveron="fills",
                    hoverinfo="text", text=f"{label}: {price:.5g} ({pct:+.3f}%)",
                    showlegend=False, name="",
                ))
            fig.add_trace(go.Scatter(
                x=[entry_x], y=[entry_price], mode="markers",
                marker=dict(symbol="circle", size=9, color=theme.NEON_AMBER, line=dict(color="#000", width=1)),
                hoverinfo="text", text=f"Entry: {entry_price:.5g} @ {ex['entry_time']}",
                showlegend=False, name="",
            ))
            sl_tp_caption = (f" · TP (green box) {tp_price:.5g} ({(tp_price - entry_price) / entry_price * 100:+.3f}%) "
                              f"/ SL (red box) {sl_price:.5g} ({(sl_price - entry_price) / entry_price * 100:+.3f}%) — fixed 2R target, hover boxes for detail")

    caption = (f"{direction.capitalize()} example — entry (amber dotted) {ex['entry_time']}, "
               f"exit (cyan dotted) {ex['exit_time']}, this trade's return: {ex['raw_return']*100:+.3f}%")
    caption += sl_tp_caption
    if has_zone:
        zone_shade = "green" if direction == "bullish" else "magenta"
        if pd.notna(zone_start_time):
            caption += f" · detector's zone shaded (pale {zone_shade}) from formation {zone_start_time} to touch"
        else:
            caption += (f" · detector's zone shaded (pale {zone_shade}) {ex['zone_bottom']:.5g}"
                        f"–{ex['zone_top']:.5g}, up to entry — this is the setup itself, not the TP/SL box below")
    if mfe is not None and mfe.get("session"):
        caption += f" · window: full {mfe['session']} session + 1h pre-session"

    if touch_pos is not None and _cat(ex[touch_col]):
        touch_price = float(bt_df["Close"].iloc[touch_pos])
        fig.add_trace(go.Scatter(x=[_cat(ex[touch_col])], y=[touch_price], mode="markers",
                                  marker=dict(symbol="diamond", size=10, color="#eaffff",
                                              line=dict(color=zone_color, width=1)),
                                  name="touch", showlegend=False))
        caption += f" · touch point (white diamond) {ex[touch_col]}"

        if mfe is not None and _cat(mfe["mfe_time"]):
            fig.add_trace(go.Scatter(x=[_cat(mfe["mfe_time"])], y=[mfe["mfe_price"]], mode="markers",
                                      marker=dict(symbol="star", size=12, color="#ffd700"), name="MFE", showlegend=False))
            fig.add_shape(type="line", x0=_cat(ex[touch_col]), x1=_cat(mfe["mfe_time"]),
                          y0=mfe["mfe_price"], y1=mfe["mfe_price"],
                          line=dict(color="#ffd700", width=1, dash="dash"))
            dist = mfe["mfe_price"] - touch_price
            dist_pct = dist / touch_price * 100
            caption += (f" · furthest level (gold star) before {mfe['stop_reason']}: {mfe['mfe_price']:.5g} "
                        f"({dist:+.5g}, {dist_pct:+.3f}% from touch)")

    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(type="category", tickangle=0, nticks=8),
        xaxis_rangeslider_visible=False, showlegend=False, font=dict(color="#c9d6e3", size=11),
    )
    st.caption(caption)
    st.plotly_chart(fig, use_container_width=True)


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


theme.render_project_nav("Research Lab")
st.header("\U0001f52c Research Lab")

tab_research, tab_backtest, tab_agent = st.tabs(["📊 Research", "📈 Backtesting", "🤖 Agent"])

# =============================================================================
# TAB 1: Research workbench (crypto overview sweep + single-ticker study)
# =============================================================================
with tab_research:
    st.caption("Event-study backtesting over the same ICT detectors the chart draws — an honest edge/no-edge verdict, not another indicator.")

    st.subheader("1. Crypto overview — all intervals, all sources")
    st.caption(f"Major pairs: {', '.join(CRYPTO_TICKERS)} · sources: yahoo, binance · "
               f"backtest sweep at {', '.join(INTERVAL_LABELS[i] for i in LONG_INTERVALS)} "
               "(FVG retracement, forward_bars=50, min_body_ratio=0.5, cost=10bps — same params as the "
               "BTC-USD runs already in the experiment log, so results are directly comparable).")

    if not os.path.exists(AVAILABILITY_PATH):
        st.info("No sweep run yet. From the project root: `python3 -m research.sweep` "
                "(takes a few minutes — fetches every ticker/interval/source combo).")
    else:
        avail_df = pd.read_csv(AVAILABILITY_PATH)
        mtime = datetime.fromtimestamp(os.path.getmtime(AVAILABILITY_PATH), tz=timezone.utc)
        st.markdown(f'<div class="tiny-note">Last run: {mtime.strftime("%Y-%m-%d %H:%M UTC")} · '
                    f're-run with <code>python3 -m research.sweep</code> to refresh</div>', unsafe_allow_html=True)

        with st.expander("Availability matrix (all intervals × both sources)", expanded=True):
            pivot = avail_df.pivot_table(index=["ticker", "interval"], columns="source",
                                          values="n_bars", aggfunc="first").reset_index()
            st.dataframe(pivot, hide_index=True, use_container_width=True)
            st.caption("n_bars per combo — 0 means that source has no data at all for that ticker/interval "
                       "(Binance is crypto-only and caps at 1000 candles per call regardless of period asked; "
                       "Yahoo has no such per-call cap but a hard per-interval history ceiling instead — "
                       "see INTERVAL_MAX_PERIOD).")
            failures = avail_df[~avail_df["ok"]]
            if not failures.empty:
                st.warning(f"{len(failures)} combo(s) failed outright:")
                st.dataframe(failures[["ticker", "interval", "source", "error"]], hide_index=True, use_container_width=True)
            with st.expander("Full detail (start/end per combo)"):
                st.dataframe(avail_df, hide_index=True, use_container_width=True)

        if not os.path.exists(BACKTEST_SWEEP_PATH):
            st.info("Availability sweep found, but no backtest sweep yet — same command covers both.")
        else:
            bt_df = pd.read_csv(BACKTEST_SWEEP_PATH)
            n_edge = (bt_df["verdict"] == "EDGE_FOUND").sum()
            with st.expander(f"Backtest sweep — 1h & 1d across {len(CRYPTO_TICKERS)} pairs "
                              f"({n_edge} of {len(bt_df)} cleared p<0.05)", expanded=True):
                st.caption(f"{n_edge}/{len(bt_df)} clearing raw p<0.05 is close to the ~{len(bt_df)*0.05:.1f} "
                           "you'd expect from pure chance alone at this many attempts — treat any single "
                           "EDGE_FOUND row here as a lead to validate out-of-sample (holdout period / "
                           "untouched asset), not a confirmed edge.")

                def _highlight_edge(row):
                    return ["background-color: rgba(57,255,20,0.12)" if row["verdict"] == "EDGE_FOUND" else ""] * len(row)

                st.dataframe(bt_df.style.apply(_highlight_edge, axis=1), hide_index=True, use_container_width=True)

    log_df = read_log()
    if not log_df.empty:
        st.subheader(f"2. Experiment log ({len(log_df)} runs)")
        st.caption("Every study ever run, regardless of verdict — the record that makes any single "
                   "p-value meaningful. See how many attempts it took before a verdict cleared p<0.05.")
        st.dataframe(log_df.iloc[::-1], hide_index=True, use_container_width=True)

# =============================================================================
# TAB 2: Backtesting — data hooked up ONCE, every detector visible at once
# =============================================================================
with tab_backtest:
    st.caption("Hook up historical data once — every detector's settings, what it found, and the result, "
               "all visible at once. No detector claims an edge until you run it; nothing here is silent.")

    d1, d2, d3 = st.columns([2, 1, 1])
    bt_ticker = d1.text_input("Ticker", value=st.session_state.get("bt_ticker_val", "GBPUSD=X"),
                               key="bt_ticker_val").strip().upper()
    bt_provider = d2.selectbox("Source", ["histdata", "auto", "yahoo", "binance"], key="bt_provider",
                                help="histdata = the free FX archive imported earlier (1-minute native, "
                                     "2000-2025 for GBPUSD) — everything above 1m is resampled from it on the fly.")
    _interval_choices = list(FX_RESAMPLE_RULES.keys()) if bt_provider == HISTDATA_PROVIDER else list(INTERVAL_MAX_PERIOD.keys())
    bt_interval = d3.selectbox("Interval", _interval_choices, index=min(2, len(_interval_choices) - 1),
                                format_func=lambda k: INTERVAL_LABELS.get(k, k), key="bt_interval")

    bt_sessions = st.multiselect(
        "Restrain to kill zones (optional)", list(KILL_ZONES.keys()), key="bt_sessions",
        help="Detection always runs on the full continuous history (FVG/order-block/swing patterns need "
             "consecutive bars, not a chopped-up one) — this filters the RESULTING events down to only "
             "those triggered inside the selected NY session window(s), same as picking a kill zone on the "
             "live chart. Leave empty to use every hour.")

    with st.spinner(f"Loading {bt_ticker} @ {INTERVAL_LABELS.get(bt_interval, bt_interval)} ({bt_provider})..."):
        try:
            bt_df_full = load_backtest_df(bt_ticker, bt_interval, bt_provider)
        except Exception as e:
            st.error(f"Could not load {bt_ticker} @ {bt_interval} from {bt_provider}: {e}")
            st.stop()

    # How much history to actually process — the single biggest speed lever
    # by far: processing time scales with bar count, and going from 25
    # years to 3 is roughly a 8x smaller dataset for roughly proportionally
    # less work. Defaults to a SMALL recent window rather than the full
    # history precisely so the page is fast out of the box; the full range
    # is still one drag away when you actually want the deep history test.
    _full_years = max(1, round((bt_df_full.index[-1] - bt_df_full.index[0]).days / 365.25))
    if _full_years > 1:
        bt_years_back = st.slider(
            f"Years to backtest (most recent — {_full_years} available)", 1, _full_years,
            min(_full_years, 3), key="bt_years_back",
            help="Limits processing to the most recent N years — smaller windows run dramatically faster "
                 "(e.g. all 7 detectors across 25 years of GBPUSD took ~90s; 3 years takes a few seconds). "
                 "The full imported history is always still there, just one drag away.")
        _cutoff = bt_df_full.index[-1] - pd.DateOffset(years=bt_years_back)
        bt_df = bt_df_full[bt_df_full.index >= _cutoff]
    else:
        bt_df = bt_df_full

    # Corwin-Schultz high-low estimate, from the loaded data itself — not a
    # blind guess. See research/spread.py's own docstring for the method.
    # Falls back to 10.0 (the old flat guess) only if the estimator can't
    # produce one at all (e.g. under 3 bars loaded).
    _est_spread = estimate_spread_bps(bt_df)
    bt_default_cost = round(_est_spread, 1) if _est_spread is not None else 10.0

    st.markdown(
        f'<div class="tiny-note">{bt_provider} · <b>{len(bt_df)} bars</b> · {bt_df.index[0]} → {bt_df.index[-1]}'
        f'{" · sessions: " + ", ".join(bt_sessions) if bt_sessions else " · all hours"}'
        f' · estimated spread: <b>{bt_default_cost:.2f} bps</b> (Corwin-Schultz, last 2000 bars — '
        f'used as each detector\'s default round-trip cost below, still freely adjustable)</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    run_all = st.button("⚡ Run all detectors", type="primary", use_container_width=True)

    if run_all:
        st.session_state["bt_results"] = {}
        names = list(HYPOTHESES.keys())
        bar = st.progress(0.0)
        status = st.empty()
        run_t0 = time.time()
        for i, name in enumerate(names):
            hyp = HYPOTHESES[name]
            status.markdown(f"`[{i + 1}/{len(names)}]` Running **{name}**... (elapsed so far: {time.time()-run_t0:.1f}s)")
            fwd = st.session_state.get(f"bt_fwd_{name}", 10)
            cost = st.session_state.get(f"bt_cost_{name}", bt_default_cost)
            extra = {k: st.session_state.get(f"bt_extra_{name}_{k}", default)
                     for k, label, lo, hi, default, step in hyp["extra_params"]}
            step_t0 = time.time()
            events = hyp["extract"](bt_df, forward_bars=fwd, **extra)
            events = filter_events_to_sessions(events, bt_sessions)
            if events.empty:
                result = {"n_events": 0, "verdict": "NO_EVENTS"}
            else:
                result = run_event_study(events, cost_bps=cost, direction_col=hyp["direction_col"])
                log_run(bt_ticker, bt_interval, bt_provider, bt_provider, name, fwd, cost,
                        {**extra, "sessions": bt_sessions}, result)
            step_elapsed = time.time() - step_t0
            st.session_state["bt_results"][name] = {"result": result, "events": events, "cost_bps": cost, "elapsed_s": step_elapsed}
            status.markdown(f"`[{i + 1}/{len(names)}]` **{name}** — {result.get('n_events', 0)} events in {step_elapsed:.2f}s")
            bar.progress((i + 1) / len(names))
        st.success(f"Done — {len(names)} detectors run against {len(bt_df)} bars in {time.time()-run_t0:.1f}s total.")

    # --- "In one look": every detector's last result, whether from just now
    # or a previous run this session --- persists via session_state so
    # switching tabs or re-running just ONE detector below doesn't blank it.
    if st.session_state.get("bt_results"):
        st.markdown("**Summary — every detector, one look**")
        rows = []
        for name in HYPOTHESES:
            entry = st.session_state["bt_results"].get(name, {})
            r = entry.get("result", {})
            n = r.get("n_events", 0)
            rows.append({
                "detector": name, "n_events": n,
                "mean_return": f"{r['mean_return_after_cost']*100:.3f}%" if n else "—",
                "win_rate": f"{r['win_rate_after_cost']*100:.1f}%" if n else "—",
                "p_value": f"{r['p_value_vs_random_direction']:.4f}" if n else "—",
                "verdict": r.get("verdict", "not run yet"),
                "time": f"{entry['elapsed_s']:.2f}s" if "elapsed_s" in entry else "—",
            })
        summary_df = pd.DataFrame(rows)

        def _hl_verdict(row):
            return ["background-color: rgba(57,255,20,0.12)" if row["verdict"] == "EDGE_FOUND" else ""] * len(row)

        st.dataframe(summary_df.style.apply(_hl_verdict, axis=1), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("**Per-detector settings — its own section, every input editable**")

    for name, hyp in HYPOTHESES.items():
        prior = st.session_state.get("bt_results", {}).get(name)
        header = name if not prior else f"{name}  —  {prior['result'].get('n_events', 0)} events, {prior['result'].get('verdict', '—')}"
        with st.expander(header):
            if "Liquidity Reaction" in name:
                st.caption("⚠ This detector only ever reports the MOST RECENT reaction per direction (a "
                           "live-display design, not a historical scanner — see fvg.py's own docstring) — "
                           "expect very few events here regardless of how much history is loaded.")
            with st.form(f"bt_form_{name}"):
                c1, c2 = st.columns(2)
                c1.slider("Forward bars (horizon)", 1, 50, 10, key=f"bt_fwd_{name}")
                c2.slider("Round-trip cost (bps)", 0.0, 50.0, bt_default_cost, 0.5, key=f"bt_cost_{name}",
                          help="Subtracted from every trade before any stat is computed — spread + slippage "
                               "+ fees, bundled into one number. Defaulted from a Corwin-Schultz estimate off "
                               "this data's own recent High/Low ranges (research/spread.py), not a blind "
                               "guess — still freely adjustable.")
                if hyp["extra_params"]:
                    ecols = st.columns(len(hyp["extra_params"]))
                    for col, (k, label, lo, hi, default, step) in zip(ecols, hyp["extra_params"]):
                        col.slider(label, lo, hi, default, step, key=f"bt_extra_{name}_{k}")
                rerun_clicked = st.form_submit_button(f"Run {name}", type="primary")

            if rerun_clicked:
                fwd = st.session_state[f"bt_fwd_{name}"]
                cost = st.session_state[f"bt_cost_{name}"]
                extra = {k: st.session_state[f"bt_extra_{name}_{k}"] for k, *_ in hyp["extra_params"]}
                _t0 = time.time()
                with st.spinner("Extracting events..."):
                    events = hyp["extract"](bt_df, forward_bars=fwd, **extra)
                    events = filter_events_to_sessions(events, bt_sessions)
                if events.empty:
                    result = {"n_events": 0, "verdict": "NO_EVENTS"}
                else:
                    with st.spinner(f"Running permutation test over {len(events)} events..."):
                        result = run_event_study(events, cost_bps=cost, direction_col=hyp["direction_col"])
                    log_run(bt_ticker, bt_interval, bt_provider, bt_provider, name, fwd, cost,
                            {**extra, "sessions": bt_sessions}, result)
                _elapsed = time.time() - _t0
                st.session_state.setdefault("bt_results", {})[name] = {"result": result, "events": events, "cost_bps": cost, "elapsed_s": _elapsed}
                prior = st.session_state["bt_results"][name]
                st.toast(f"{name}: {result.get('n_events', 0)} events in {_elapsed:.2f}s")
                st.rerun()

            if prior:
                r, events, cost = prior["result"], prior["events"], prior["cost_bps"]
                if "elapsed_s" in prior:
                    st.caption(f"⏱ Ran in {prior['elapsed_s']:.2f}s")
                if r.get("n_events", 0) == 0:
                    verdict_box("NO EVENTS", "Zero qualifying events for this data/settings/session combo — "
                                              "try a different interval, wider settings, or no session restriction.", tone="warn")
                else:
                    tone = "bullish" if r["verdict"] == "EDGE_FOUND" else "warn"
                    detail = (f"n={r['n_events']} · mean return after cost {r['mean_return_after_cost']*100:.3f}% · "
                              f"win rate {r['win_rate_after_cost']*100:.1f}% · "
                              f"p={r['p_value_vs_random_direction']:.4f} vs. random direction")
                    verdict_box(r["verdict"].replace("_", " "), detail, tone=tone)

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Events", r["n_events"])
                    m2.metric("Mean return (after cost)", f"{r['mean_return_after_cost']*100:.3f}%")
                    m3.metric("Win rate", f"{r['win_rate_after_cost']*100:.1f}%")
                    m4.metric("p-value", f"{r['p_value_vs_random_direction']:.4f}")

                    if r.get("by_type"):
                        by_type_df = pd.DataFrame([
                            {"type": t, "n": s["n"], "win_rate": f"{s['win_rate']*100:.1f}%",
                             "mean_return": f"{s['mean_return_after_cost']*100:.3f}%"}
                            for t, s in r["by_type"].items()
                        ])
                        st.dataframe(by_type_df, hide_index=True, use_container_width=True)

                    signed = events["raw_return"] * np.where(events[hyp["direction_col"]] == "bullish", 1, -1) - cost / 10_000.0
                    st.bar_chart(signed.to_frame("signed_return_after_cost"))

                    st.markdown("**Examples — browse up to 50 real detected instances**")
                    ex = render_example_picker(events, key_prefix=f"bt_{name}")
                    if ex is not None:
                        if "FVG" in name:
                            render_example_event_chart(bt_df, ex, hyp["direction_col"],
                                                        touch_col="touched_at", show_mfe=True,
                                                        zone_start_col="gap_formed_at")
                        else:
                            render_example_event_chart(bt_df, ex, hyp["direction_col"])

                    with st.expander("Raw events"):
                        st.dataframe(events, hide_index=True, use_container_width=True)

# =============================================================================
# TAB 2: Agent control panel — live_scan.py, scheduled via launchd
# =============================================================================
PLIST_LABEL = "com.marketdashboard.livescan"
PLIST_PATH = os.path.expanduser(f"~/Library/LaunchAgents/{PLIST_LABEL}.plist")


def _launchd_status():
    try:
        result = subprocess.run(["launchctl", "print", f"gui/{os.getuid()}/{PLIST_LABEL}"],
                                 capture_output=True, text=True, timeout=5)
    except Exception as e:
        return {"scheduled": False, "state": f"error: {e}", "last_exit": None}
    if result.returncode != 0:
        return {"scheduled": False, "state": "not scheduled", "last_exit": None}
    out = result.stdout
    state = "running" if "state = running" in out else "idle (waiting for next hour)"
    last_exit = next((ln.split("=")[-1].strip() for ln in out.splitlines() if "last exit code" in ln), None)
    return {"scheduled": True, "state": state, "last_exit": last_exit}


def _tail_file(path, n=30):
    if not os.path.exists(path):
        return "(no log yet)"
    with open(path) as f:
        lines = f.readlines()
    return "".join(lines[-n:]) or "(empty)"


with tab_agent:
    st.caption("live_scan.py (research/live_scan.py) — refreshes crypto OHLCV caches and logs newly-formed "
               "FVG/order-block/liquidity-sweep structures. No signals, no trades — just a running record. "
               "Runs on your machine via launchd, not in the cloud (this project has no git repo, and the "
               "whole point depends on local disk state persisting between runs).")

    status = _launchd_status()
    s1, s2, s3 = st.columns(3)
    s1.metric("Scheduled (launchd)", "ON" if status["scheduled"] else "OFF")
    s2.metric("Current state", status["state"])
    s3.metric("Last exit code", status["last_exit"] or "—")

    st.markdown("**Controls**")
    b1, b2, b3 = st.columns(3)
    if status["scheduled"]:
        if b1.button("⏹ Stop schedule (launchctl unload)", use_container_width=True):
            subprocess.run(["launchctl", "unload", PLIST_PATH], capture_output=True, text=True)
            st.rerun()
    else:
        if b1.button("▶ Start schedule (launchctl load)", type="primary", use_container_width=True):
            subprocess.run(["launchctl", "load", PLIST_PATH], capture_output=True, text=True)
            st.rerun()
    run_live_clicked = b2.button("⚡ Run now (live)", type="primary", use_container_width=True)
    if b3.button("🔄 Refresh status", use_container_width=True):
        st.rerun()

    if run_live_clicked:
        # Runs the SAME refresh_caches/scan_for_new_structures the launchd
        # job calls, but in THIS process instead of launchctl-starting a
        # separate one — the separate-process path (still what the hourly
        # schedule uses) has no way to show progress here as it happens; the
        # page could only re-read the finished log file after the fact.
        # Real numbers updating live means actually doing the work here.
        cfg_now = load_config()
        total_combos = max(len(cfg_now["tickers"]) * len(cfg_now["intervals"]), 1)

        st.markdown("**Live run in progress**")
        st.caption(f"{len(cfg_now['tickers'])} tickers × {len(cfg_now['intervals'])} intervals = {total_combos} combos")

        refresh_bar = st.progress(0.0)
        refresh_status = st.empty()
        refresh_counts = {"done": 0, "bars": 0, "failed": 0}

        def _refresh_progress(ticker, interval, n_bars, error):
            refresh_counts["done"] += 1
            if error:
                refresh_counts["failed"] += 1
                refresh_status.markdown(f"`[{refresh_counts['done']}/{total_combos}]` ⚠ **{ticker} {INTERVAL_LABELS[interval]}** failed: {error}")
            else:
                refresh_counts["bars"] += n_bars
                refresh_status.markdown(f"`[{refresh_counts['done']}/{total_combos}]` refreshed **{ticker} {INTERVAL_LABELS[interval]}** — "
                                         f"{n_bars} bars ({refresh_counts['bars']} total bars refreshed so far, "
                                         f"{refresh_counts['failed']} failed)")
            refresh_bar.progress(refresh_counts["done"] / total_combos)

        refresh_caches(cfg_now["tickers"], cfg_now["intervals"], provider=cfg_now["provider"], on_progress=_refresh_progress)

        scan_bar = st.progress(0.0)
        scan_status = st.empty()
        scan_counts = {"done": 0}

        def _scan_progress(ticker, interval, n_new_here, n_new_total):
            scan_counts["done"] += 1
            scan_status.markdown(f"`[{scan_counts['done']}/{total_combos}]` scanned **{ticker} {INTERVAL_LABELS[interval]}** — "
                                  f"{n_new_here} new here, **{n_new_total} new structure(s) total so far**")
            scan_bar.progress(scan_counts["done"] / total_combos)

        new_rows = scan_for_new_structures(cfg_now["tickers"], cfg_now["intervals"], cfg_now["detectors"], on_progress=_scan_progress)

        st.success(f"Run complete — {refresh_counts['bars']} bars refreshed across {total_combos} combos "
                   f"({refresh_counts['failed']} failed), {len(new_rows)} new structure(s) logged.")
        if new_rows:
            st.dataframe(pd.DataFrame(new_rows), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("**Settings**")
    cfg = load_config()
    with st.form("agent_settings"):
        enabled = st.checkbox("Enabled (soft switch — off skips all work on the next scheduled fire, "
                               "without touching the launchd schedule itself)", value=cfg["enabled"])
        tickers = st.multiselect("Tickers to watch", CRYPTO_TICKERS, default=cfg["tickers"])
        intervals = st.multiselect("Intervals to watch", list(INTERVAL_MAX_PERIOD.keys()),
                                    default=cfg["intervals"], format_func=lambda k: INTERVAL_LABELS[k])
        dc1, dc2, dc3 = st.columns(3)
        det_fvg = dc1.checkbox("FVG detector", value=cfg["detectors"]["fvg"], help=(
            "Fair Value Gap — a 3-candle imbalance: a strong displacement candle leaves a price "
            "void the two candles around it never traded through. Read as a zone price may return "
            "to 'fill' before continuing in the original direction."))
        det_ob = dc2.checkbox("Order block detector", value=cfg["detectors"]["order_block"], help=(
            "The last opposing candle right before a strong displacement move breaks clean through "
            "it — ICT's proxy for 'a large position was likely built here before the move.' Watched "
            "as potential support/resistance on a retest."))
        det_liq = dc3.checkbox("Liquidity reaction detector", value=cfg["detectors"]["liquidity_reaction"], help=(
            "Price sweeps a recent swing high/low (taking out resting stops) and then reacts from a "
            "nearby order block shortly after — the classic 'stop hunt, then reverse' pattern."))
        dc4, dc5, dc6 = st.columns(3)
        det_eq = dc4.checkbox("Equal highs/lows detector", value=cfg["detectors"].get("equal_highs_lows", True), help=(
            "Clusters of 2+ swing highs (or lows) sitting within 0.15% of each other — read as a pool "
            "of resting liquidity (stops bunched at a shared level) and a likely magnet for a future "
            "sweep."))
        det_bos = dc5.checkbox("BOS detector", value=cfg["detectors"].get("bos", True), help=(
            "Break of Structure — price closes past the most recent swing point in the direction that "
            "matches the CURRENT trend bias. A routine continuation confirmation, not a warning sign."))
        det_choch = dc6.checkbox("CHoCH detector", value=cfg["detectors"].get("choch", True), help=(
            "Change of Character — price closes past the most recent swing point AGAINST the current "
            "trend bias: the first real sign the trend may be reversing."))
        dc7, _dc8, _dc9 = st.columns(3)
        det_judas = dc7.checkbox("🔔 Judas swing detector", value=cfg["detectors"].get("judas_swing", True), help=(
            "The full chained playbook, not a single signal: liquidity sweep -> CHoCH confirms the "
            "reversal -> price retraces into a fresh FVG/order block in the new direction. Fires only "
            "once all three complete in order — and this is the ONLY detector that sends a real macOS "
            "notification, not just a log row. No statistical edge has been validated for this "
            "sequence yet (or any other signal tested so far) — this recognizes when the classic "
            "playbook completed, it isn't a claim that following it wins."))
        provider = st.selectbox("Data source", ["auto", "yahoo", "binance"],
                                 index=["auto", "yahoo", "binance"].index(cfg["provider"]))
        save_clicked = st.form_submit_button("Save settings", type="primary")

    if save_clicked:
        save_config({
            "enabled": enabled, "tickers": tickers, "intervals": intervals,
            "detectors": {"fvg": det_fvg, "order_block": det_ob, "liquidity_reaction": det_liq,
                          "equal_highs_lows": det_eq, "bos": det_bos, "choch": det_choch,
                          "judas_swing": det_judas},
            "provider": provider,
        })
        st.success("Saved — takes effect on the next run (scheduled, or click Run now above).")
        st.rerun()

    st.markdown("---")
    st.markdown("**Logs**")
    _research_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research", ".cache")
    l1, l2 = st.columns(2)
    with l1:
        st.caption("stdout (research/.cache/live_scan.log)")
        st.code(_tail_file(os.path.join(_research_cache_dir, "live_scan.log")), language=None)
    with l2:
        st.caption("stderr (research/.cache/live_scan.err.log) — Streamlit's own boilerplate warnings are expected here")
        st.code(_tail_file(os.path.join(_research_cache_dir, "live_scan.err.log")), language=None)

    st.markdown("---")
    st.markdown("**Recently detected structures**")
    if os.path.exists(LIVE_EVENTS_PATH):
        events_df = pd.read_csv(LIVE_EVENTS_PATH)
        st.caption(f"{len(events_df)} total logged · showing the 50 most recent")
        st.dataframe(events_df.tail(50).iloc[::-1], hide_index=True, use_container_width=True)
    else:
        st.info("No events logged yet — run the agent at least once.")
