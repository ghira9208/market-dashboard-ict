"""
Institutional — "where big money is, direction is more probable," turned
into a system that cuts noise instead of dumping raw derivatives metrics.
See institutional.py for the actual signal math and the full reasoning
behind why three independently-computed signals are summed rather than
any one trusted alone before this page calls anything a real lean (kept
presentation-agnostic, same convention as news.py/recommender.py — this
page only renders it).
"""
import pandas as pd
import streamlit as st

import institutional as inst
import theme

st.set_page_config(page_title="Institutional", layout="wide", initial_sidebar_state="collapsed")
theme.inject()
theme.render_top_bar("Institutional")

_BIAS_COLOR = {
    3: theme.NEON_GREEN, 2: theme.NEON_GREEN, 1: theme.NEON_GREEN,
    0: theme.NEON_AMBER,
    -1: theme.NEON_MAGENTA, -2: theme.NEON_MAGENTA, -3: theme.NEON_MAGENTA,
}


def _verdict(color, headline, detail=""):
    st.markdown(
        f'<div class="verdict-box" style="border-color:{color};">'
        f'<span class="verdict-icon" style="color:{color};">●</span>'
        f'<div style="flex:1;">'
        f'<div class="verdict-headline" style="color:{color};">{headline}</div>'
        + (f'<div class="verdict-detail">{detail}</div>' if detail else "")
        + '</div></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- Macro ----
st.subheader("Total crypto market cap")
snapshot = inst.get_market_cap_snapshot()
if snapshot is None:
    st.warning("Couldn't reach the market-cap feed right now — try refreshing in a bit.")
else:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total market cap", f"${snapshot['total_market_cap_usd'] / 1e12:.3f}T",
               f"{snapshot['change_24h_pct']:+.2f}% (24h)")
    m2.metric("BTC dominance", f"{snapshot['btc_dominance']:.1f}%")
    m3.metric("ETH dominance", f"{snapshot['eth_dominance']:.1f}%")
    m4.metric("Alts (ex-BTC/ETH)", f"{100 - snapshot['btc_dominance'] - snapshot['eth_dominance']:.1f}%")

    history = inst.get_market_cap_history()
    if len(history) >= 3:
        chart_df = history.set_index("date")[["total_market_cap_usd"]] / 1e12
        chart_df.columns = ["Total market cap ($T)"]
        st.line_chart(chart_df, height=180)
        dom_df = history.set_index("date")[["btc_dominance", "eth_dominance"]]
        dom_df.columns = ["BTC dominance %", "ETH dominance %"]
        st.line_chart(dom_df, height=140)
    else:
        st.caption(
            f"Building up real history from here — {len(history)} day(s) recorded so far. CoinGecko's free "
            "tier has no historical total-market-cap series to backfill from, so this chart only ever covers "
            "days this page has actually been open to record a snapshot; check back as it accumulates."
        )

st.divider()

# ------------------------------------------------------------ Watchlist ----
st.subheader("Big money bias")
with st.popover("ℹ️", width="content"):
    st.markdown(
        "Three independently-computed reads on which way size is leaning are summed into one score — no "
        "single one is trusted alone, since each is well-known to give false reads constantly by itself:\n\n"
        "- **Positioning (Open Interest + Price)** — does open interest (fresh contracts, real capital at "
        "risk) rising or falling line up with the direction price actually moved.\n"
        "- **Smart Money Divergence** — does Binance's own top-trader tier (their largest accounts by size) "
        "sit meaningfully more bullish or bearish than the retail-heavy crowd, not just directionally long "
        "(the crowd is almost always net-long anyway — what matters is leaning MORE than that baseline).\n"
        "- **Order Flow (CVD vs. Price)** — does net taker buy/sell volume (real order flow) actually back "
        "the direction price moved, or is the move happening on thin/opposite flow (a rally on net selling "
        "is usually short-covering, not fresh demand).\n\n"
        "Multiple agreeing is what makes a **Strong** lean; just one firing is a **Mild** lean; disagreement "
        "or no signal is **No clear read** — an honest 'nothing to trust here,' never a forced pick. Funding "
        "rate is shown separately as a crowding flag, not folded into the score — it measures how expensive "
        "current leverage is, not which way size is leaning.\n\n"
        "Not included: on-chain whale tracking and spot-ETF flows (no free API without scraping), and CME "
        "futures/COT (free but weekly-lag) — real candidates for later."
    )

results = inst.scan_watchlist()
rows = []
for r in results:
    fr = r["funding"]["funding_rate"]
    rows.append({
        "Ticker": r["ticker"], "Score": r["score"], "Read": r["label"],
        "Funding": f"{fr * 100:+.4f}%" if fr is not None else "—",
    })
table_df = pd.DataFrame(rows)
st.dataframe(
    table_df, width="stretch", hide_index=True,
    column_config={
        "Score": st.column_config.NumberColumn("Score", format="%+d", help="-3..+3 — see the ℹ️ above for how this is built."),
    },
)

_bias_history = inst.get_bias_history()
_bias_days = _bias_history["date"].nunique() if not _bias_history.empty else 0
st.caption(
    f"Recording every score above to disk daily — {_bias_days} day(s) accumulated so far. Binance's own OI/ratio "
    "data disappears after 30 days regardless; this log is the only way to ever have more than that to check "
    "whether these scores actually predict anything, rather than just looking directionally reasonable today."
)

st.divider()

# ---------------------------------------------------------- Drill-down ----
st.subheader("Detail")
ticker = st.selectbox("Ticker", [r["ticker"] for r in results], key="inst_detail_ticker")
picked = next(r for r in results if r["ticker"] == ticker)

color = _BIAS_COLOR.get(picked["score"], theme.NEON_AMBER)
_verdict(color, f"{picked['ticker']} — {picked['label']} (score {picked['score']:+d})")

for sig in picked["signals"]:
    if not sig["sufficient"]:
        st.caption(f"○ {sig['name']}: not enough data yet")
        continue
    arrow = "▲" if sig["direction"] > 0 else ("▼" if sig["direction"] < 0 else "—")
    sig_color = theme.NEON_GREEN if sig["direction"] > 0 else (theme.NEON_MAGENTA if sig["direction"] < 0 else theme.NEON_AMBER)
    st.markdown(
        f'<div style="display:flex; gap:8px; align-items:baseline; margin:4px 0;">'
        f'<span style="color:{sig_color}; font-weight:700;">{arrow}</span>'
        f'<span>{sig["name"]}: <b>{sig["detail"]}</b></span></div>',
        unsafe_allow_html=True,
    )

funding = picked["funding"]
if funding["funding_rate"] is not None:
    st.caption(f"Funding rate: {funding['funding_rate'] * 100:+.4f}% · {funding['note']}")

oi_df = inst.get_open_interest_history(ticker)
top_df = inst.get_top_trader_ratio(ticker)
global_df = inst.get_global_ratio(ticker)
taker_df = inst.get_taker_volume_history(ticker, interval="1d", limit=30)

c1, c2, c3 = st.columns(3)
with c1:
    st.caption("Open interest (contracts)")
    if not oi_df.empty:
        st.area_chart(oi_df.set_index("time")[["oi"]], height=220)
    else:
        st.caption("No data.")
with c2:
    st.caption("Top traders vs. crowd — long/short ratio")
    if not top_df.empty and not global_df.empty:
        merged = pd.merge(
            top_df.rename(columns={"long_short_ratio": "Top traders"}),
            global_df.rename(columns={"long_short_ratio": "Crowd (all accounts)"}),
            on="time", how="outer",
        ).set_index("time")
        st.line_chart(merged, height=220)
    else:
        st.caption("No data.")
with c3:
    st.caption("CVD — cumulative taker buy/sell delta")
    if not taker_df.empty:
        cvd_df = taker_df.set_index("time")[["delta"]].cumsum()
        cvd_df.columns = ["CVD"]
        st.area_chart(cvd_df, height=220)
    else:
        st.caption("No data.")

st.caption(
    "Binance USDT-margined futures data — 30 days of history max (Binance's own limit on these endpoints, "
    "not something this page can extend). Six-pair watchlist, same list research/sweep.py already uses for "
    "its own crypto batch sweep."
)
