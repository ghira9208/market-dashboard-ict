"""
News — a ForexFactory-sourced economic calendar, scoped to high-impact
releases: which of this project's own Markets/Crypto instruments tend to
react, plus a plain-language "what usually happens" playbook note. See
news.py for the data/heuristic layer this page only renders (kept
presentation-agnostic, same convention as recommender.py). That module's
own session_news_map() is also what drives the flashing red badge on the
top bar's ASIA/LON/NY session pills (see theme.py's _render_session_pills)
— this page is where that flash points to.
"""

import pandas as pd
import streamlit as st

import theme
from news import SESSION_CURRENCIES, get_calendar, get_expectation_note, impacted_assets

st.set_page_config(page_title="News", layout="wide", initial_sidebar_state="collapsed")
theme.inject()
theme.render_top_bar("News")

df = get_calendar()

if df.empty:
    st.warning("Couldn't reach the calendar feed right now — try refreshing in a bit.")
    st.stop()

display_tz = theme.get_display_tz()
df = df.assign(local_time=df["time"].dt.tz_convert(display_tz))
df = df.assign(local_date=df["local_time"].dt.date)
now_local = pd.Timestamp.now(tz=display_tz)

# ---- The one thing worth seeing on open, before anything else: is there
# real (High-impact) news today, and what's next -- regardless of whatever
# the impact filter below is set to. This replaces a page-title-plus-
# paragraph opener that was the same wall of text on every single visit;
# direct request: "spoon serve me the critical bits," not a header I don't
# need reminding of every time.
high = df[df["impact"] == "High"]
today_high = high[high["local_date"] == now_local.date()]
upcoming_high = high[high["time"] >= now_local]


def _countdown(target_time):
    delta = target_time - now_local
    days, rem = divmod(int(delta.total_seconds()), 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    return f"{days}d {hours}h" if days else f"{hours}h {minutes}m"


today_remaining = today_high[today_high["time"] >= now_local]
if not today_high.empty and not today_remaining.empty:
    _next = today_remaining.iloc[0]
    banner = f"{len(today_high)} high-impact event(s) today — next: {_next['title']} ({_next['country']}) in {_countdown(_next['time'])}"
    banner_color = theme.NEON_MAGENTA
elif not today_high.empty:
    banner = f"{len(today_high)} high-impact event(s) today — all already released."
    banner_color = theme.NEON_AMBER
elif not upcoming_high.empty:
    _next = upcoming_high.iloc[0]
    banner = f"No high-impact news today. Next this week: {_next['title']} ({_next['country']}) in {_countdown(_next['time'])}"
    banner_color = theme.NEON_AMBER
else:
    banner = "No high-impact news left on the calendar this week."
    banner_color = theme.NEON_GREEN

st.markdown(
    f'<div class="verdict-box" style="border-color:{banner_color};">'
    f'<span class="verdict-icon" style="color:{banner_color};">●</span>'
    f'<div class="verdict-headline" style="color:{banner_color};">{banner}</div></div>',
    unsafe_allow_html=True,
)

holidays = df[df["impact"] == "Holiday"]
if not holidays.empty:
    names = ", ".join(f'{r["title"]} ({r["country"]})' for _, r in holidays.iterrows())
    st.caption(f"Market holiday(s) this week — thinner liquidity likely: {names}")

with st.container(horizontal=True, vertical_alignment="center"):
    impact_choice = st.segmented_control(
        "Impact filter", options=["High", "High + Medium", "All"],
        default="High", required=True, key="news_impact_filter", label_visibility="collapsed",
    )
    with st.popover("ℹ️", width="content"):
        st.markdown(
            "High-impact releases from ForexFactory's own public calendar feed, covering the rest of this "
            "trading week (the feed rolls onto the new week every Sunday) — not a multi-week outlook. Each "
            "one lists which of this dashboard's own instruments tend to react, plus a plain-language note "
            "on what a beat or miss usually means. These are typical-playbook reads, not predictions — "
            "nothing here is a trade signal."
        )

if impact_choice == "High":
    shown = df[df["impact"] == "High"]
elif impact_choice == "High + Medium":
    shown = df[df["impact"].isin(["High", "Medium"])]
else:
    shown = df[df["impact"] != "Holiday"]

upcoming = shown[shown["time"] >= now_local]
past = shown[shown["time"] < now_local]

# Color-codes each currency by which trading session it mostly wakes up in
# (see news.SESSION_CURRENCIES) -- ties this page's badges back to the same
# ASIA/LON/NY color language the top bar's own session pills already use.
_CURRENCY_COLOR = {}
for _session, _currencies in SESSION_CURRENCIES.items():
    for _c in _currencies:
        _CURRENCY_COLOR[_c] = theme.SESSION_COLORS[_session]

IMPACT_COLOR = {"High": theme.NEON_MAGENTA, "Medium": theme.NEON_AMBER, "Low": theme.NEON_CYAN}


def _render_event(row):
    color = IMPACT_COLOR.get(row["impact"], theme.NEON_CYAN)
    country_color = _CURRENCY_COLOR.get(row["country"], theme.NEON_CYAN)
    time_str = row["local_time"].strftime("%a %H:%M")
    forecast = row["forecast"] or "—"
    previous = row["previous"] or "—"
    assets = impacted_assets(row["country"])
    if assets:
        asset_html = "".join(
            f'<a class="news-asset-pill" href="http://localhost:{port}/?ticker={ticker}" target="_blank">{name}</a>'
            for ticker, name, port in assets
        )
    else:
        asset_html = '<span class="tiny-note" style="margin:0;">No instrument in this dashboard\'s own curated list.</span>'
    note = get_expectation_note(row["title"])
    st.markdown(
        f'<div class="verdict-box" style="border-color:{color};">'
        f'<span class="verdict-icon" style="color:{color};">●</span>'
        f'<div style="flex:1;">'
        f'<div class="verdict-headline" style="color:{color};">{time_str}&nbsp;&nbsp;'
        f'<span class="news-country-badge" style="background:rgba(255,255,255,0.08); color:{country_color};">{row["country"]}</span>'
        f'{row["title"]}</div>'
        f'<div class="verdict-detail">Forecast {forecast} &nbsp;·&nbsp; Previous {previous}</div>'
        f'<div class="verdict-detail">{note}</div>'
        f'<div class="news-asset-row">{asset_html}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def _render_group(events):
    for day, day_events in events.groupby("local_date", sort=True):
        if day == now_local.date():
            label = f"Today — {day.strftime('%a %b %d')}"
        elif day == (now_local + pd.Timedelta(days=1)).date():
            label = f"Tomorrow — {day.strftime('%a %b %d')}"
        else:
            label = day.strftime("%A %b %d")
        st.markdown(f'<div class="news-day-header">{label}</div>', unsafe_allow_html=True)
        for _, row in day_events.sort_values("time").iterrows():
            _render_event(row)


if upcoming.empty:
    st.markdown('<div class="tiny-note">No upcoming events match this filter for the rest of the week.</div>',
                unsafe_allow_html=True)
else:
    _render_group(upcoming)

if not past.empty:
    with st.expander(f"Earlier this week ({len(past)})"):
        _render_group(past)
