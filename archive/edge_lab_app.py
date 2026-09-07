"""
Edge Lab — a GBP-only agent that searches for a trading edge on its own,
instead of testing a fixed menu of hypotheses a human already picked (that's
what Research Lab's Backtesting tab does). Standalone page, own port, same
separation-of-concerns reasoning as Research Lab being standalone from the
live ICT terminal: a long search batch here should never affect either of
the other two pages.

The search itself (edge_lab/agent.py) samples a candidate — one of the same
7 ICT triggers Research Lab already knows, plus new filters (session,
timeframe, volatility regime, day-of-week) — scores it with the exact same
permutation-test machinery (research/backtest.run_event_study), and logs
every trial whether or not it was significant. What makes this safe to run
unattended and call "search" rather than "p-hack": a train/holdout split
fixed once at first run, and Benjamini-Hochberg correction recomputed over
the FULL cumulative trial log every time results are shown — see
edge_lab/agent.py's own docstring for why both of those are load-bearing,
not decoration.
"""

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import theme
from edge_lab import agent
from edge_lab.multiple_testing import benjamini_hochberg
from research.events import HYPOTHESES

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


# Shown as a hover tooltip on each dataframe column's header (st.column_config
# below) — every table on this page reuses whichever of these its own
# columns happen to include, so a term explained once (e.g. q_value_train,
# shared by the leaderboard and the raw log) never drifts into two
# different explanations.
COLUMN_HELP = {
    "family": "The (trigger, session, volatility regime, weekday) combination this row groups trials by.",
    "family_key": "Internal id for the family — same grouping as 'family', just as a single string.",
    "candidate": "The exact sampled trigger + filters + timeframe this row scored.",
    "n_trials": "How many trials have been run for this exact family so far.",
    "win_rate": "Share of this family's trials that cleared raw train significance (p<0.05, before multiple-"
                "testing correction) with a positive return — a fast signal used only to steer sampling, not a "
                "final verdict.",
    "ucb1_score": "How attractive this family looks to the agent right now: win rate plus a bonus that shrinks "
                  "the more the family has already been tried. A never-tried family scores effectively infinite, "
                  "so the agent always explores every family once before it starts exploiting what's worked.",
    "n_train": "Number of qualifying events found in the training period (everything before the train/holdout "
               "split date).",
    "n_holdout": "Number of qualifying events found in the untouched holdout period (everything from the split "
                 "date onward) — kept out of the search entirely.",
    "mean_return_train": "Average return per trade after estimated round-trip cost, on the training period.",
    "mean_return_holdout": "Average return per trade after cost, on the holdout period only.",
    "win_rate_train": "Share of training-period trades that were profitable after cost.",
    "win_rate_holdout": "Share of holdout-period trades that were profitable after cost.",
    "p_value_train": "Raw permutation-test p-value on train: how often a random direction label would beat this "
                      "candidate's actual result by pure chance. Lower is stronger, but not yet adjusted for how "
                      "many other candidates were tried.",
    "p_value_holdout": "Same permutation test, run only on the untouched holdout period — the real out-of-sample "
                        "check.",
    "q_value_train": "Benjamini-Hochberg-adjusted p-value, recomputed across every trial ever logged — accounts "
                      "for how many candidates were tried, unlike the raw p-value. Must be < 0.05 to count as "
                      "train-significant.",
    "train_significant": "True if this trial's RAW p-value cleared 0.05 with a positive return — a quick, "
                          "unadjusted flag. q_value_train is the corrected version that actually decides the "
                          "leaderboard.",
    "bh_significant": "True if this trial survives Benjamini-Hochberg correction across the full trial log — the "
                       "real train-side bar.",
    "holdout_verdict": "PASSED means this candidate's holdout-only results also point the same direction (p<0.10, "
                        "a looser bar since holdout is a smaller slice by design). FAILED or NOT_ENOUGH_DATA means "
                        "it didn't — this is the actual out-of-sample check, more important than the train-side "
                        "stats.",
    "cost_bps": "Estimated round-trip cost (spread + slippage), in basis points, subtracted from every trade "
                "before any stat here is computed — from the Corwin-Schultz estimator on this candidate's own "
                "data, not a flat guess.",
    "verdict": "SCORED means there were enough training events to run the permutation test. INSUFFICIENT_DATA "
               "means too few events for this exact candidate/session/filter combo to test at all.",
    "label": "Human-readable summary of this trial's exact configuration.",
    "hypothesis": "Which of the 7 base ICT triggers this trial used.",
    "extra_params": "The trigger's own tunable parameters for this trial (e.g. FVG's minimum displacement body "
                     "ratio), sampled at random within its usual range.",
    "interval": "Candlestick timeframe this trial ran on.",
    "session": "Kill-zone session this trial restricted entries to, if any.",
    "vol_regime": "Volatility-regime filter: only 'high' or 'low' realized-volatility entries counted, relative "
                  "to this instrument's own rolling median, if set.",
    "weekday": "Restricts entries to a single weekday only (0=Monday..4=Friday), if set.",
    "forward_bars": "How many bars forward this trial measured the outcome over.",
}


def _column_config(columns):
    return {c: st.column_config.Column(help=COLUMN_HELP[c]) for c in columns if c in COLUMN_HELP}


def _hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def render_candidate_chart(bt_df, ex, direction_col, window_bars=50):
    """Ported from research_app.py's render_example_event_chart — same
    categorical x-axis (a real datetime axis renders FX weekend closures as
    a blank void, see that function's own comment) and same hoverable
    entry/SL/TP box logic, trimmed of the FVG-only MFE/session-window
    machinery this generic candidate table doesn't carry. NOT imported —
    research_app.py is a runnable Streamlit script with its own
    st.set_page_config(), the same reason research/sessions.py gives for
    duplicating app.py's KILL_ZONES instead of importing them."""
    pos_by_time = {t: i for i, t in enumerate(bt_df.index)}
    entry_pos = pos_by_time.get(ex["entry_time"])
    if entry_pos is None:
        st.caption("(example chart unavailable — entry bar fell outside the currently loaded window)")
        return
    lo = max(0, entry_pos - window_bars)
    hi = min(len(bt_df), entry_pos + window_bars)
    window = bt_df.iloc[lo:hi]

    direction = ex[direction_col]
    zone_color = theme.NEON_GREEN if direction == "bullish" else theme.NEON_MAGENTA

    x_labels = [t.strftime("%b %d, %H:%M") for t in window.index]
    label_by_time = dict(zip(window.index, x_labels))

    def _cat(t):
        return label_by_time.get(t) if pd.notna(t) else None

    fig = go.Figure(data=[go.Candlestick(
        x=x_labels, open=window["Open"], high=window["High"], low=window["Low"], close=window["Close"],
        increasing_line_color=theme.NEON_GREEN, decreasing_line_color=theme.NEON_MAGENTA,
        increasing_fillcolor=theme.NEON_GREEN, decreasing_fillcolor=theme.NEON_MAGENTA, name="",
    )])

    has_zone = pd.notna(ex["zone_top"]) and pd.notna(ex["zone_bottom"]) and ex["zone_top"] != ex["zone_bottom"]
    if has_zone:
        zone_x1 = _cat(ex["entry_time"]) or x_labels[-1]
        fig.add_shape(type="rect", xref="x", yref="y", x0=x_labels[0], x1=zone_x1,
                      y0=ex["zone_bottom"], y1=ex["zone_top"],
                      fillcolor=_hex_to_rgba(zone_color, 0.15), line=dict(color=zone_color, width=1))

    entry_x, exit_x = _cat(ex["entry_time"]), _cat(ex["exit_time"])
    if entry_x:
        fig.add_vline(x=entry_x, line_color=theme.NEON_AMBER, line_dash="dot", line_width=1.5)
    if exit_x:
        fig.add_vline(x=exit_x, line_color=theme.NEON_CYAN, line_dash="dot", line_width=1.5)

    caption = (f"{direction.capitalize()} example — entry (amber dotted) {ex['entry_time']}, "
               f"exit (cyan dotted) {ex['exit_time']}, this trade's return: {ex['raw_return']*100:+.3f}%")

    if entry_x and exit_x:
        entry_price = float(ex["entry_price"])
        if has_zone:
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
                    hoverinfo="text", text=f"{label}: {price:.5g} ({pct:+.3f}%)", showlegend=False, name="",
                ))
            fig.add_trace(go.Scatter(
                x=[entry_x], y=[entry_price], mode="markers",
                marker=dict(symbol="circle", size=9, color=theme.NEON_AMBER, line=dict(color="#000", width=1)),
                hoverinfo="text", text=f"Entry: {entry_price:.5g} @ {ex['entry_time']}", showlegend=False, name="",
            ))
            caption += (f" · TP (green box) {tp_price:.5g} ({(tp_price - entry_price) / entry_price * 100:+.3f}%) "
                        f"/ SL (red box) {sl_price:.5g} ({(sl_price - entry_price) / entry_price * 100:+.3f}%) — fixed 2R target, hover boxes for detail")

    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(type="category", tickangle=0, nticks=8),
        xaxis_rangeslider_visible=False, showlegend=False, font=dict(color="#c9d6e3", size=11),
    )
    st.caption(caption)
    st.plotly_chart(fig, use_container_width=True)


st.set_page_config(page_title="Edge Lab", layout="wide", initial_sidebar_state="collapsed")
theme.inject()

# theme.py locks html/body/stApp/stMain to a fixed 100vh with overflow:
# hidden — correct for app.py's single-screen, nothing-scrolls chart
# terminal, wrong for this page (a normal top-to-bottom form + leaderboard +
# growing trial log, genuinely taller than one viewport). Same override
# research_app.py already carries for the same reason — copied here rather
# than changed in the shared file, so app.py's layout stays untouched.
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

theme.render_project_nav("Edge Lab")
st.header("\U0001f9ec Edge Lab — GBPUSD edge search")
st.caption("An agent that samples ICT triggers + session/volatility/weekday filters on its own, scores each one "
           "with the same permutation test Research Lab uses, and gets better at choosing WHAT to try next as its "
           "trial log grows — across runs, not just within one. Nothing here is hidden: every trial, win or lose, "
           "is logged and shown below.")

trials = agent.load_all_trials()
scored = [t for t in trials if t.get("verdict") == "SCORED"]

with st.spinner("Loading GBPUSD history..."):
    base_1m = agent.load_base_1m()
cutoff = agent.get_split_cutoff(base_1m)

c1, c2, c3 = st.columns(3)
c1.metric("Total trials logged", len(trials),
          help="Every candidate this agent has ever scored, across every 'Run search batch' click ever made.")
c2.metric("Scored (enough data)", len(scored),
          help="Trials that found enough training-period events (20+) to actually run the permutation test on. "
               "The rest hit INSUFFICIENT_DATA — too narrow a filter combination to test at all.")
c3.metric("Train / holdout split", cutoff.strftime("%Y-%m-%d"),
          help="Everything before this date is what the search agent is allowed to look at. Everything from this "
               "date onward is holdout — set once, the very first time this agent ever ran, and never touched by "
               "the search since.")
st.caption(f"GBPUSD 1m history: {len(base_1m):,} bars, {base_1m.index[0]} → {base_1m.index[-1]}. "
           f"Holdout is everything from {cutoff.strftime('%Y-%m-%d')} onward — fixed the first time this agent "
           "ever ran and never recomputed, so it stays untouched by the search no matter how much more history "
           "gets imported later.")

st.markdown("---")
n_trials = st.slider("Trials to run this batch", 5, 100, 15,
                      help="Each trial extracts events for one sampled candidate and runs a permutation test on "
                           "both the train and holdout slices. ~1-3s per trial depending on how many events it finds.")
run_clicked = st.button("⚡ Run search batch", type="primary", use_container_width=True)

if run_clicked:
    bar = st.progress(0.0)
    status = st.empty()
    t0 = time.time()

    def _on_progress(i, n, record):
        verdict = record.get("holdout_verdict", record.get("verdict", "?"))
        status.markdown(f"`[{i}/{n}]` **{record['label']}** — {record.get('n_train', 0)} train events, "
                         f"{verdict} (elapsed {time.time() - t0:.1f}s)")
        bar.progress(i / n)

    new_records, trials = agent.run_search_batch(n_trials=n_trials, on_progress=_on_progress)
    scored = [t for t in trials if t.get("verdict") == "SCORED"]
    # Both halves of the gate, on the RAW (not yet BH-corrected) train flag
    # — holdout_verdict=="PASSED" alone isn't news; plenty of candidates
    # that were never even train-significant will still randomly "pass" a
    # weaker holdout bar. The leaderboard below re-checks this properly
    # with BH correction over the full log; this toast is just a same-
    # session preview, worded to match what the leaderboard will actually
    # show rather than over-claiming.
    n_passed = sum(1 for r in new_records if r.get("train_significant") and r.get("holdout_verdict") == "PASSED")
    st.success(f"Done — {n_trials} trials in {time.time() - t0:.1f}s. "
               f"{n_passed} cleared both train significance and holdout this batch (see leaderboard below for "
               "the multiple-testing-corrected verdict).")

st.markdown("---")
st.subheader("Family standings — why the agent tries what it tries next")
st.caption("UCB1 over (trigger, session, volatility regime, weekday) — families never yet tried always sort first "
           "(that's correct UCB1 behavior, not a bug: it tries everything once before exploiting). win_rate is the "
           "share of a family's own trials that cleared raw train significance with a positive return — a fast, "
           "unadjusted signal used only to steer sampling, not the final verdict below.")
standings = agent.family_standings(trials)
if standings.empty:
    st.caption("No trials yet — run a batch above.")
else:
    st.dataframe(standings.head(20), hide_index=True, use_container_width=True,
                 column_config=_column_config(standings.columns))

st.markdown("---")
st.subheader("Leaderboard — train-significant candidates, holdout-checked")

if not scored:
    st.info("No scored trials yet. Run a batch above to get started.")
else:
    p_values = [t["p_value_train"] for t in scored]
    q_values, significant = benjamini_hochberg(p_values, alpha=0.05)
    for t, q, sig in zip(scored, q_values, significant):
        t["q_value_train"] = float(q)
        t["bh_significant"] = bool(sig)

    leaders = [t for t in scored if t["bh_significant"]]
    leaders.sort(key=lambda t: t["q_value_train"])

    st.caption(f"{len(scored)} scored trials total, {sum(1 for t in scored if t['train_significant'])} cleared raw "
               f"p<0.05 on train, {len(leaders)} survive Benjamini-Hochberg correction across all {len(scored)} "
               "trials ever run — this is why the count usually shrinks from the raw-significant one.")

    if not leaders:
        verdict_box("NO VALIDATED EDGE YET", "Nothing has survived multiple-testing correction so far. That's an "
                                              "honest, expected outcome early on — run more trials, or check back "
                                              "once the family standings above show real exploitation instead of "
                                              "still working through untried families.", tone="warn")
    else:
        rows = []
        for t in leaders:
            rows.append({
                "candidate": t["label"], "n_train": t["n_train"],
                "mean_return_train": f"{t['mean_return_train']*100:+.3f}%",
                "win_rate_train": f"{t['win_rate_train']*100:.1f}%",
                "q_value_train": f"{t['q_value_train']:.4f}",
                "n_holdout": t.get("n_holdout", 0),
                "holdout_verdict": t.get("holdout_verdict", "—"),
            })
        lb_df = pd.DataFrame(rows)

        def _hl_holdout(row):
            color = "rgba(57,255,20,0.12)" if row["holdout_verdict"] == "PASSED" else (
                "rgba(255,46,154,0.10)" if row["holdout_verdict"] == "FAILED" else "")
            return [f"background-color: {color}"] * len(row)

        st.dataframe(lb_df.style.apply(_hl_holdout, axis=1), hide_index=True, use_container_width=True,
                     column_config=_column_config(lb_df.columns))

        holdout_passed = [t for t in leaders if t.get("holdout_verdict") == "PASSED"]
        if not holdout_passed:
            verdict_box("TRAIN-SIGNIFICANT, NOT HOLDOUT-VALIDATED",
                        f"{len(leaders)} candidate(s) survive BH correction on train but none held up on the "
                        "untouched holdout period yet — this is exactly the gate doing its job, not a failure. "
                        "A candidate this agent calls a real edge has to clear both.", tone="warn")
        else:
            best = holdout_passed[0]
            detail = (f"n_train={best['n_train']} · mean return after cost {best['mean_return_train']*100:+.3f}% "
                       f"· win rate {best['win_rate_train']*100:.1f}% · q={best['q_value_train']:.4f} (BH-adjusted) "
                       f"· holdout: n={best.get('n_holdout', 0)}, mean return "
                       f"{best.get('mean_return_holdout', 0)*100:+.3f}%, p={best.get('p_value_holdout', float('nan')):.4f}")
            verdict_box(f"BEST VALIDATED: {best['label']}", detail, tone="bullish")

            candidate = agent.candidate_from_trial(best)
            bt_df = agent.resample_for_interval(base_1m, candidate.interval)
            events = agent.extract_candidate_events(candidate, bt_df)
            if not events.empty:
                spec = HYPOTHESES[candidate.hypothesis]
                st.markdown("**Most recent real instance:**")
                render_candidate_chart(bt_df, events.iloc[-1], spec["direction_col"])

st.markdown("---")
with st.expander(f"Raw trial log — every trial ever run ({len(trials)})"):
    if trials:
        log_df = pd.DataFrame(trials).iloc[::-1]
        # "family" (a list) and "extra_params" (a dict) per row break
        # pyarrow's column-type inference for st.dataframe outright
        # (confirmed directly: ArrowTypeError, not just an ugly render) —
        # stringify them, nothing else in this table needs to stay
        # structured since it's a read-only log view.
        for col in ("family", "extra_params"):
            if col in log_df.columns:
                log_df[col] = log_df[col].astype(str)
        st.dataframe(log_df, hide_index=True, use_container_width=True,
                     column_config=_column_config(log_df.columns))
    else:
        st.caption("Nothing logged yet.")
