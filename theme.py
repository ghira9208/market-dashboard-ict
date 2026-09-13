import html
from datetime import time as dtime

import pandas as pd
import streamlit as st

import news

NEON_CYAN = "#0A84FF"
NEON_MAGENTA = "#FF453A"
NEON_GREEN = "#30D158"
NEON_AMBER = "#FF9F0A"
# A zone/level's own bullish-or-bearish color (cyan/magenta for FVG, green/
# amber for Order Blocks) already carries that meaning — recommender.py's
# zone_indicator_matches/point_level_indicator_matches confluence flag needs
# its OWN color, not a thicker version of the same one, so "this zone also
# has indicator confluence" reads as its own distinct signal rather than a
# subtler variant of "bullish" or "bearish." A rich metallic gold (not the
# brighter systemYellow already used for EMA 20/London — reusing that would
# make a confluence highlight look like an indicator line) was picked
# specifically to read as "premium," per direct request.
CONFLUENCE_GOLD = "#D4AF37"

CSS = """
<style>
[data-testid="stMainBlockContainer"] {
    padding-top: 1.2rem !important;
    padding-bottom: 1rem !important;
}

:root {
    --neon-cyan: #0A84FF;
    --neon-magenta: #FF453A;
    --neon-green: #30D158;
    --neon-amber: #FF9F0A;
    --bg-deep: #1c1c1e;
    --bg-panel: #2c2c2e;
    --bg-panel-2: #3a3a3c;
    --text-main: #f5f5f7;
    --text-dim: #98989d;
    --separator: rgba(255,255,255,0.1);
    --font-system: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", "Segoe UI", sans-serif;
}

/* Flat, near-solid app background — macOS's own dark-mode surfaces don't
   carry a visible grid/glow pattern, they're just a clean, quiet color. */
[data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: var(--bg-deep);
}

/* Kiosk mode: this is a single-purpose chart terminal, not a page anyone
   navigates away from — the Deploy/menu header has no use here. Used to be
   `display: none` outright, which also fixed stAppToolbar re-enabling
   pointer-events:auto on itself and eating clicks meant for the ticker box.
   BUT display:none on the header also unmounts stExpandSidebarButton (the
   only way back in once the sidebar's own collapse arrow is clicked) —
   confirmed directly: collapsing the Signals sidebar then left no control
   anywhere on screen to reopen it. Fix: keep the header mounted but
   invisible and non-interactive (pointer-events:none, transparent), punch
   a single hole back open for the expand button alone, and hide the
   Deploy/menu chrome by testid instead of nuking the whole header. */
[data-testid="stHeader"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    pointer-events: none !important;
}
[data-testid="stToolbar"] { pointer-events: none !important; }
[data-testid="stToolbarActions"] { display: none !important; }
[data-testid="stExpandSidebarButton"] {
    pointer-events: auto !important;
    color: var(--neon-cyan) !important;
    /* Now anchored via position:fixed rather than the old relative-offset
       nudge — the Signals sidebar moved to the RIGHT edge of the screen
       (see [data-testid="stSidebar"] below), but this button lives in
       stHeader, which still lays it out assuming a left-side sidebar (its
       own default left:19-ish position, confirmed earlier via
       getBoundingClientRect). A relative offset only shifts it from
       wherever that native layout puts it — no relative nudge gets it
       all the way to the opposite edge, so this switches to fixed
       positioning entirely, anchored to the exact same 0-40px band
       render_top_bar's own row and "☰" (now on the left) share, mirrored
       to the right edge instead of the left. */
    position: fixed !important;
    top: 6px !important;
    left: auto !important;
    right: 18px !important;
}

/* Native top-right "Running..." rerun indicator — fires on every fragment
   tick (several charts auto-refresh every 1s), so left on it never stops
   flickering. The custom stSpinner boxes already cover real loading state
   for external fetches, so this one is pure noise — hide it. */
[data-testid="stStatusWidget"] { display: none !important; }

/* Streamlit's OWN automatic cache-miss spinner used to need its own rule
   here, separate from the app's manual st.spinner(...) calls — this one is
   injected for free by every @st.cache_data-decorated function on a cache
   miss, with its text always the raw Python call (e.g.
   "Running detect_fvgs(...)."), and at the page's real column width it had
   no white-space handling and wrapped character-by-character into an
   unreadable column of single letters ("what's that text while loading" —
   traced via a MutationObserver, since it's gone again by the time you'd
   inspect it by hand). Now covered by the broader stSpinner rule further
   down, which hides every spinner uniformly — kept as its own comment here
   since the wrapping-bug diagnosis is worth keeping on record. */

/* Streamlit fades an element container to ~33% opacity while its fragment
   is re-running — a normal, sensible loading affordance for something that
   normally reruns rarely. Confirmed directly (sampled getComputedStyle
   every 200ms): the chart containers cycle smoothly from opacity 1 down to
   0.33 and back on EVERY auto-refresh tick, which for a 1m chart is every
   ~8-10s — a live chart visibly "breathing"/fading every few seconds reads
   as it constantly losing connection, exactly backwards from the "live"
   feel the fast-refresh work was for. Same category of native busy-chrome
   already hidden above (stStatusWidget) for the same reason. */
[class*="st-key-ict_chart"] { opacity: 1 !important; }

/* Sidebar was retired in favor of the top nav bar for a while (hidden
   entirely) — back in use for the Markets/Crypto watchlist scan, collapsed
   by default so the chart stays the primary view. The styling below
   predates that retirement and was left in place the whole time, which is
   why reactivating it needed nothing more than removing the old blanket
   `display: none`. */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--bg-panel) 0%, var(--bg-deep) 100%);
    border-right: none;
    border-left: 1px solid var(--separator);
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    /* stAppViewContainer lays out stSidebar + stMain as plain flex-row
       siblings in DOM order (sidebar first) — order:2 is the entire flip
       to the right edge, per direct request (Signals panel moved right,
       ☰ Settings moved left so the two don't share a corner). */
    order: 2;
}
/* Streamlit's own collapse animation slides the sidebar off with a
   hardcoded translateX(-300px) (confirmed via getComputedStyle on a
   collapsed sidebar) — built assuming a LEFT-edge sidebar, where sliding
   further left hides it. Now on the right, sliding further left would
   instead slide it OVER the chart. translateX(100%) mirrors the intent
   (off toward the sidebar's own free edge) using a percentage of the
   sidebar's own width rather than a hardcoded px value, so it still
   fully hides the sidebar even after a manual drag-resize changes its
   width from the 300px default. */
[data-testid="stSidebar"][aria-expanded="false"] {
    transform: translateX(100%) !important;
}

/* Streamlit's own collapse-sidebar arrow (the "«" icon in stSidebarHeader)
   ships hidden by default on any desktop-width viewport -- its own baked-in
   CSS only flips it to visibility:visible under a max-width:576px media
   query, so once the sidebar is open there's normally no visible way to
   close it again on desktop short of a hard refresh. Forced visible here
   per direct request. stSidebarHeader also lives INSIDE stSidebarContent
   (the part that scrolls once results fill the sidebar), so without
   pinning it, the button scrolls away with everything else — sticky keeps
   it docked at the top of that scroll area instead, with a solid
   background so scrolled content doesn't show through underneath it. */
[data-testid="stSidebarCollapseButton"] {
    visibility: visible !important;
    /* Native position (inside stSidebarHeader, vertically centered in its
       own ~60px-tall header band) lands this a good 10px lower than the
       expand chevron's own fixed top:6px/right:18px (see
       stExpandSidebarButton above) — confirmed directly via
       getBoundingClientRect on both (collapse: top 16-44; expand: top
       6-34), a visible jump between the two toggle states now that both
       sit in the same top-right corner (per direct report). stSidebarHeader
       has no transform, so position:fixed here still escapes cleanly to
       true viewport coordinates — pinning it to the EXACT same spot as
       the expand chevron makes the icon toggle in place instead of
       jumping. right:18px keeps working regardless of the sidebar's own
       (user-resizable) width since the sidebar always spans to the
       viewport's own right edge. */
    position: fixed !important;
    top: 6px !important;
    right: 18px !important;
    left: auto !important;
    z-index: 101 !important;
}
/* position:fixed above escapes the sidebar's own translateX collapse
   animation entirely (confirmed directly: getBoundingClientRect showed it
   sitting at the exact same on-screen spot whether aria-expanded was true
   or false) — before that override existed, this button was a plain
   normal-flow child that physically slid off-screen WITH the sidebar when
   collapsed, which is what actually made it disappear; visibility:visible
   !important above only ever undid Streamlit's own default-hidden state,
   it was never what hid this on its own. Escaping the transform for
   alignment (this button and stExpandSidebarButton now share one exact
   spot instead of jumping between toggle states) broke that free ride —
   both buttons ended up stacked on screen at once, rendering as a single
   garbled icon (direct report, screenshot: overlapping double-chevrons).
   Explicit state check restores "only show when there's something to
   collapse" now that position no longer does it for free. */
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarCollapseButton"] {
    display: none !important;
}
[data-testid="stSidebarHeader"] {
    position: sticky;
    top: 0;
    z-index: 2;
    background: var(--bg-panel);
}

[data-testid="stSidebar"] [data-testid="stButton"] button {
    background: transparent !important;
    border: none !important;
    padding: 4px 0 18px 0 !important;
    box-shadow: none !important;
    text-align: left !important;
    border-radius: 0 !important;
    border-bottom: 1px solid var(--separator) !important;
    margin-bottom: 10px;
}
[data-testid="stSidebar"] [data-testid="stButton"] button p {
    font-family: var(--font-system) !important;
    font-size: 1.2rem !important;
    font-weight: 600;
    letter-spacing: 0;
    text-transform: none !important;
    color: var(--neon-cyan) !important;
    white-space: normal !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] button:hover p {
    color: #ffffff !important;
}

[data-testid="stSidebar"] [data-testid="stRadioGroup"] label {
    font-family: var(--font-system);
    letter-spacing: 0.3px;
    text-transform: uppercase;
    font-weight: 600;
    color: var(--text-dim);
}

[data-testid="stSidebar"] [data-testid="stRadioGroup"] label:has(input:checked) {
    color: var(--neon-cyan) !important;
}

/* Segmented pill controls (timeframe pickers, Top 10/100 toggles, etc.) in the main area */
[data-testid="stMain"] [data-testid="stRadioGroup"] {
    gap: 4px;
    background: rgba(255,255,255,0.05);
    border: 1px solid var(--separator);
    border-radius: 20px;
    padding: 3px;
    width: fit-content;
}
[data-testid="stMain"] [data-testid="stRadioGroup"] label {
    font-family: var(--font-system);
    font-variant-numeric: tabular-nums;
    font-size: 0.78rem;
    padding: 2px 12px;
    border-radius: 16px;
    color: var(--text-dim);
}
[data-testid="stMain"] [data-testid="stRadioGroup"] label:has(input:checked) {
    background: rgba(10,132,255,0.15);
    color: var(--neon-cyan) !important;
}
[data-testid="stMain"] [data-testid="stRadioGroup"] label:has(input:checked) [data-testid="stMarkdownContainer"] p {
    font-weight: 700;
}
[data-testid="stMain"] [data-testid="stRadioGroup"] label [data-testid="stMarkdownContainer"] p {
    font-family: var(--font-system) !important;
    font-variant-numeric: tabular-nums;
}
/* The old selector (div[role="radiogroup"] > label > div:first-child) was
   written against an older Streamlit DOM and silently stopped matching —
   confirmed directly: current markup nests each option as
   [data-testid="stRadioOption"] > ... > div (dot wrapper) + stMarkdownContainer
   (text), several levels deep, with churny auto-generated class names at
   every level. Targeting by structure (the sibling immediately before the
   text container) instead of by class name survives that churn. */
[data-testid="stMain"] [data-testid="stRadioOption"] div:has(> [data-testid="stMarkdownContainer"]) > div:first-child {
    display: none;
}

/* Headings: sentence case, semibold, plain solid color — no uppercase,
   no letter-spacing, no glow. Apple's own headings (Settings, Stocks,
   System Information) are just weight and size doing the work. */
[data-testid="stHeading"] h1, [data-testid="stHeading"] h2, [data-testid="stHeading"] h3 {
    font-family: var(--font-system) !important;
    font-weight: 600;
    color: var(--text-main) !important;
}

[data-testid="stHeading"] h3 {
    font-size: 1.05rem !important;
    color: var(--neon-cyan) !important;
    border-left: 3px solid var(--neon-cyan);
    padding-left: 10px;
}

/* Loading state used to be made deliberately loud (a pulsing glow box) on
   the theory that Streamlit's default spinner is too easy to miss. Reversed
   on request: the page should stay perfectly still while loading — no
   visible spinner, no pulse, nothing moving — same "hide native busy
   chrome" treatment already applied to stStatusWidget and stCacheSpinner
   above, just extended to this app's own intentional st.spinner() calls
   too. The chart itself is still the loading feedback: it simply doesn't
   update until the fetch resolves. */
[data-testid="stSpinner"] {
    display: none !important;
}

[data-testid="stCaptionContainer"] {
    color: var(--text-dim) !important;
    font-family: var(--font-system);
}

[data-testid="stMarkdownContainer"] p {
    font-family: var(--font-system);
    color: var(--text-main);
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, var(--bg-panel), var(--bg-panel-2));
    border: 1px solid var(--separator);
    border-radius: 14px;
    padding: 8px 10px 6px 10px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
}

/* Popovers (Layers ☰, Settings ⚙️) and the market-menu expander default to
   Streamlit's own generous padding (23px / 16px respectively, confirmed
   directly) — fine for one card, heavy once a popover holds a dense list
   of rows the way these two do. */
[data-testid="stPopoverBody"] {
    padding: 10px 12px !important;
}
[data-testid="stExpanderDetails"] {
    padding: 6px 4px 10px !important;
}
[data-testid="stExpander"] {
    border-color: var(--separator) !important;
}

[data-testid="stMetricValue"] {
    font-family: var(--font-system) !important;
    font-variant-numeric: tabular-nums;
    color: var(--neon-cyan) !important;
}

[data-testid="stMetricLabel"] {
    font-family: var(--font-system) !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-size: 0.78rem !important;
    color: var(--text-dim) !important;
}

/* Minimalist top nav bar — overrides the bulkier card-button style, scoped to
   the row that follows our .topnav-marker so World-page nav cards keep theirs. */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .topnav-marker) {
    border-bottom: 1px solid var(--separator);
    padding-bottom: 8px !important;
    margin-bottom: 14px !important;
}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .topnav-marker) [data-testid="stButton"] button {
    padding: 4px 6px !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .topnav-marker) [data-testid="stButton"] button p {
    font-family: var(--font-system) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.3px;
    text-transform: uppercase;
    font-weight: 600;
    color: var(--text-dim) !important;
    white-space: nowrap;
}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .topnav-marker) [data-testid="stButton"] button:hover p {
    color: var(--neon-cyan) !important;
}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .topnav-marker) [data-testid="stButton"]:first-child button p {
    font-family: var(--font-system) !important;
    font-size: 0.85rem !important;
    font-weight: 700;
    letter-spacing: 0;
    color: var(--neon-cyan) !important;
}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .topnav-marker) [data-testid="stBaseButton-primary"] {
    background: rgba(10,132,255,0.15) !important;
    border-radius: 8px !important;
}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .topnav-marker) [data-testid="stBaseButton-primary"] p {
    color: var(--neon-cyan) !important;
}
/* Section cards — what makes each area visually distinct.
   Streamlit's bordered container is a plain [data-testid="stVerticalBlock"] with a
   default border; target only the one whose direct child holds our chip, since these
   blocks nest (an unscoped :has(.sec-chip) would also match ancestor wrappers). */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .sec-chip) {
    background: var(--bg-panel) !important;
    border: 1px solid var(--separator) !important;
    border-radius: 14px !important;
    padding: 14px 18px !important;
    margin-bottom: 18px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
}
.sec-chip {
    display: inline-block;
    font-family: var(--font-system);
    font-size: 0.68rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    color: var(--neon-cyan);
    opacity: 0.9;
    background: rgba(10,132,255,0.1);
    border: 1px solid rgba(10,132,255,0.3);
    border-radius: 20px;
    padding: 3px 12px;
    margin: 6px 0 12px 0;
}

.tiny-note {
    font-family: var(--font-system);
    font-size: 0.72rem;
    color: var(--text-dim);
    letter-spacing: 0;
    line-height: 1.4;
    margin: 2px 0 8px 0;
}

.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: var(--font-system);
    font-variant-numeric: tabular-nums;
    font-size: 0.72rem;
    color: var(--text-dim);
    margin-bottom: 6px;
}
.live-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--neon-magenta);
    box-shadow: 0 0 6px var(--neon-magenta);
    animation: livepulse 1.4s ease-in-out infinite;
}
@keyframes livepulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.35; transform: scale(0.7); }
}

.market-status-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 2px 0 16px 0;
}
.market-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: var(--font-system);
    font-variant-numeric: tabular-nums;
    font-size: 0.7rem;
    padding: 4px 10px 4px 8px;
    border-radius: 14px;
    background: rgba(255,255,255,0.05);
    border: 1px solid var(--separator);
}
.market-pill-name {
    color: var(--text-dim);
    letter-spacing: 0;
}
.market-pill-status {
    font-weight: 700;
    letter-spacing: 0;
}
.market-pill-status-open { color: var(--neon-green); }
.market-pill-status-closed { color: var(--text-dim); }
.market-dot-open, .market-dot-closed {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}
.market-dot-open {
    background: var(--neon-green);
    box-shadow: 0 0 6px var(--neon-green);
    animation: livepulse 1.4s ease-in-out infinite;
}
.market-dot-closed {
    background: #48484a;
}

.fvg-legend-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 8px 0 4px 0;
}
.fvg-legend-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: var(--font-system);
    font-variant-numeric: tabular-nums;
    font-size: 0.7rem;
    padding: 4px 10px;
    border-radius: 14px;
    background: rgba(255,255,255,0.05);
    border: 1px solid var(--separator);
}
.fvg-legend-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}
.fvg-legend-label {
    color: var(--text-dim);
    letter-spacing: 0;
}
.fvg-legend-value {
    font-weight: 700;
}

.news-day-header {
    font-family: var(--font-system);
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    color: var(--text-dim);
    margin: 18px 0 8px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--separator);
}
.news-country-badge {
    display: inline-block;
    font-family: var(--font-system);
    font-weight: 700;
    font-size: 0.68rem;
    letter-spacing: 0.4px;
    padding: 2px 7px;
    border-radius: 5px;
    margin-right: 6px;
}
.news-asset-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 6px;
}
.news-asset-pill {
    display: inline-block;
    font-family: var(--font-system);
    font-size: 0.68rem;
    color: var(--text-main);
    text-decoration: none;
    background: rgba(255,255,255,0.05);
    border: 1px solid var(--separator);
    border-radius: 12px;
    padding: 2px 9px;
    transition: border-color 0.15s ease, color 0.15s ease;
}
.news-asset-pill:hover {
    border-color: var(--neon-cyan);
    color: var(--neon-cyan);
}

.verdict-box {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    background: var(--bg-panel);
    border-left: 3px solid;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 2px 0 12px 0;
}
.verdict-icon { font-size: 1.15rem; line-height: 1.3; }
.verdict-headline {
    font-family: var(--font-system);
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0;
}
.verdict-detail {
    font-family: var(--font-system);
    color: var(--text-dim);
    font-size: 0.8rem;
    margin-top: 2px;
    line-height: 1.35;
}

[data-testid="stAlertContainer"] {
    border-radius: 8px !important;
    border: 1px solid var(--separator) !important;
    background: var(--bg-panel) !important;
    backdrop-filter: blur(6px);
}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {
    border-color: rgba(48,209,88,0.45) !important;
}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {
    border-color: rgba(255,159,10,0.45) !important;
}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {
    border-color: rgba(10,132,255,0.4) !important;
}
[data-testid="stAlertContainer"] p {
    font-family: var(--font-system) !important;
    color: var(--text-main) !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--separator) !important;
    border-radius: 8px;
    overflow: hidden;
}

hr { border-color: var(--separator) !important; }

/* padding used to be 16px 6px — fine for a handful of big nav cards, but
   this rule is unscoped (every st.button on the page, not just those),
   and the ticker-menu results list (dozens of rows shown as buttons) made
   that read as thick, spaced-out pills instead of a compact list. Cut to
   something a dense list can actually use; hover/active state still reads
   fine at this size. */
[data-testid="stButton"] button {
    width: 100%;
    background: linear-gradient(145deg, var(--bg-panel), var(--bg-panel-2)) !important;
    border: 1px solid var(--separator) !important;
    border-radius: 8px !important;
    padding: 5px 10px !important;
    color: var(--neon-cyan) !important;
    transition: all 0.15s ease;
    white-space: nowrap;
    overflow: hidden;
}
[data-testid="stButton"] button:hover {
    border-color: var(--neon-cyan) !important;
    color: #ffffff !important;
}
[data-testid="stButton"] button p {
    font-family: var(--font-system) !important;
    font-weight: 600;
    font-size: 0.75rem !important;
    letter-spacing: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* st.form_submit_button renders under a DIFFERENT testid (stFormSubmitButton)
   than plain st.button (stButton) — the block above never matched it, so a
   type="primary" submit button fell through to Streamlit's own default
   primary styling: solid neon-cyan fill with cyan text on top of it,
   confirmed directly as an invisible label (cyan-on-cyan) on the Research
   Lab's "Run study" button. Same look as the plain-button style above,
   applied under the actual testid this widget uses. */
[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(145deg, var(--bg-panel), var(--bg-panel-2)) !important;
    border: 1px solid rgba(10,132,255,0.4) !important;
    border-radius: 8px !important;
    padding: 6px 18px !important;
    color: var(--neon-cyan) !important;
    transition: all 0.15s ease;
}
[data-testid="stFormSubmitButton"] button:hover {
    border-color: var(--neon-cyan) !important;
    color: #ffffff !important;
}
[data-testid="stFormSubmitButton"] button p {
    font-family: var(--font-system) !important;
    font-weight: 600;
    font-size: 0.75rem !important;
    letter-spacing: 0;
}

[data-testid="stTextInput"] input, [data-testid="stSelectbox"] div {
    font-family: var(--font-system) !important;
    font-variant-numeric: tabular-nums;
}

/* Streamlit's default multiselect tag: solid neon-cyan fill + white text —
   confirmed directly (rgb(0,255,242) bg, rgb(255,255,255) text), both
   near-max luminance, so the text all but disappears into the fill. Same
   dark-translucent-fill + colored-border pill language already used
   elsewhere on this page (.market-pill, .fvg-legend-pill) reads clearly
   instead. */
[data-testid="stMultiSelectTagsContainer"] [data-tag] {
    background: rgba(10,132,255,0.15) !important;
    border: 1px solid rgba(10,132,255,0.4) !important;
    border-radius: 14px !important;
}
[data-testid="stMultiSelectTagsContainer"] [data-tag] span,
[data-testid="stMultiSelectTagsContainer"] [data-tag] button {
    color: var(--neon-cyan) !important;
}

/* Per-layer 4h/15m reference-frame checkboxes in the Layers (☰) popover —
   these sit right next to each ICT layer's own on/off checkbox, so a
   second checkbox SQUARE there reads as three near-identical widgets in a
   row. Wanted instead: just the "4h"/"15m" text itself, dim when off,
   highlighted when on — matching the TF bar's own pill language elsewhere
   on this page (label carries the state, no separate control glyph).
   Confirmed directly against the real DOM: Streamlit stamps
   data-selected="true" on the checkbox's own <label> when checked — a
   more direct hook than chasing the hidden <input>'s :checked state — and
   the checkmark glyph is the label's other child div, everything except
   the [data-testid="stWidgetLabel"] one, so hiding "the other div" doesn't
   depend on child ordering. Scoped to just these two checkboxes via the
   same invisible-marker-plus-:has() pattern as .sec-chip above — a bare
   [data-testid="stCheckbox"] selector would also restyle every OTHER
   checkbox on the page (the layer toggles themselves, "Show mitigated",
   "Show volume"). */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .ref-tf-toggle) [data-testid="stCheckbox"] {
    display: flex;
    justify-content: center;
}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .ref-tf-toggle) [data-testid="stCheckbox"] label {
    cursor: pointer;
    gap: 0 !important;
}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .ref-tf-toggle)
    [data-testid="stCheckbox"] label > div:not([data-testid="stWidgetLabel"]) {
    display: none;
}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .ref-tf-toggle)
    [data-testid="stCheckbox"] [data-testid="stWidgetLabel"] p {
    font-family: var(--font-system);
    font-variant-numeric: tabular-nums;
    font-size: 0.72rem;
    color: var(--text-dim);
    font-weight: 400;
    letter-spacing: 0;
    white-space: nowrap;
    transition: color 0.15s ease;
}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .ref-tf-toggle)
    [data-testid="stCheckbox"] label[data-selected="true"] [data-testid="stWidgetLabel"] p {
    color: var(--neon-cyan) !important;
    font-weight: 700;
}

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 5px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.25); }

/* Cross-links between this project's separate Streamlit processes (ICT
   Terminal / Research Lab each run on their own port — see each app's own
   README/docstring on why they're standalone processes, not multipage
   panels of one app) — a plain <a href> to the other port, not st.tabs or
   Streamlit's own multipage nav, since those only work within a single
   running server. Deliberately a single thin line: app.py's kiosk-mode
   layout budgets every pixel of vertical space to the chart itself (see
   its own "Fullscreen, non-scrolling terminal" comment), so this can't
   afford to be a full header bar the way research_app.py could support. */
.project-nav {
    display: flex; gap: 10px; align-items: center;
    font-family: var(--font-system); font-size: 0.68rem; letter-spacing: 0.3px;
    text-transform: uppercase; line-height: 1;
    /* No left padding needed anymore — it used to clear
       stExpandSidebarButton (the sidebar collapse chevron), which sat
       position:absolute at top:16/left:19/width:28 directly on top of
       this nav's default flush-left text (confirmed via a user
       screenshot: the chevron rendered right on top of "MARKETS"). That
       chevron now lives on the RIGHT edge instead (see theme.py's own
       stSidebar/stExpandSidebarButton rules — Signals sidebar moved
       right, ☰ Settings moved left, per direct request), and the ☰
       trigger that took its place on the left is already cleared by
       .st-key-_top_bar's own left inset below, not by padding here. No
       bottom padding (there was 6px here) now that render_top_bar
       centers this nav inside its own fixed-height row via flex
       align-items — that padding pushed the text's own visual center
       down within its flex slot, one of several small per-element
       offsets a user screenshot caught by marking every piece that
       wasn't sitting on the same line as the rest. */
    padding: 0; flex-shrink: 0;
}
.project-nav-current { color: var(--neon-cyan); font-weight: 600; }
.project-nav-sep { color: var(--text-dim); opacity: 0.5; }
.project-nav-link { color: var(--text-dim); text-decoration: none; transition: color 0.15s ease; }
.project-nav-link:hover { color: var(--neon-cyan); }

/* The HTF/LTF mini charts (app.py/crypto_app.py's _render_mini_chart,
   shown in the side panel's "Charts" tab) never get their real height
   applied through Streamlit's own component auto-resize (the
   streamlit:setFrameHeight postMessage the ict_chart component sends).
   Confirmed directly: that message correctly reaches Streamlit's parent
   frame every second (both mini charts tick on their own independent
   schedule regardless of which side-panel tab is active) but the iframe
   element's own height never updates, staying stuck at Streamlit's
   ~150px fallback — these two component instances are born hidden (the
   side panel's default tab is Layers, not Charts), and Streamlit's
   frontend appears to skip registering an iframe for auto-resize if it
   isn't visible at mount time, with no retroactive registration once it
   becomes visible. A direct height override sidesteps that broken
   registration entirely. 380 matches the height= these two components
   are called with in Python — keep the two in sync if that ever changes. */
.st-key-ict_chart_mini_htf iframe,
.st-key-ict_chart_mini_ltf iframe {
    height: 380px !important;
}

/* render_top_bar's own unified row — project nav, session pills, and the
   timezone selector all in the one slim strip the nav used to have to
   itself, with padding-right clearing space for the "☰" popover trigger
   (.st-key-_menu_trigger_wrap in app.py/crypto_app.py — a SEPARATE
   position:absolute element, defined deep in the chart-rendering code
   and pinned to this exact same top-right corner; not moved here, just
   left room for). flex:0 0 auto on the nav/timezone ends keeps them
   their own natural width; flex:1 on the sessions block in the middle
   is what actually absorbs whatever space is left, so the row reflows
   cleanly at any viewport width instead of the three pieces fighting
   over fixed widths.

   position:fixed at top:0, not normal document flow — confirmed directly
   (a user screenshot marking every misaligned piece with an X) that
   Streamlit's own block-container top padding put this row's natural
   flow position ~37px lower than "☰"'s own band (0-40px, confirmed via
   getComputedStyle on .st-key-_menu_trigger_wrap) — two things that are
   supposed to read as one bar were sitting on different lines.
   height:40px + align-items:center matches "☰"'s own band exactly, so
   both now share it instead of each guessing at a top offset
   independently. Streamlit's own native sidebar-expand chevron sat in
   its own different band (stHeader's own flex-centering put it at
   16-44, not 0-40) — that one's Streamlit's own chrome, not something
   render_top_bar renders, so it's nudged into line separately via its
   own position:fixed override on [data-testid="stExpandSidebarButton"]
   itself (this file's own Kiosk-mode section, above).
   render_top_bar's own Python side adds a plain spacer div right after
   this row — position:fixed takes it out of normal flow entirely, so
   without one the ticker-search box below would render UNDERNEATH it
   instead of below it.

   left/right insets: 58px on the left clears the ☰ menu trigger (now on
   the left, .st-key-_menu_trigger_wrap in app.py/crypto_app.py — a
   SEPARATE position:absolute element in main_col, pinned to this same
   top-left corner, not moved here, just left room for); a smaller 50px
   on the right clears the Signals sidebar's own collapse chevron (now
   on the right, [data-testid="stExpandSidebarButton"] above — narrower
   than the ☰ trigger, hence the smaller reservation). Both sides were
   flipped together, per direct request (Signals panel → right,
   ☰ Settings → left) — was left:0/right:58px when the menu trigger and
   chevron sat on the opposite edges from where they are now. */
.st-key-_top_bar {
    position: fixed !important;
    top: 0 !important;
    left: 58px !important;
    right: 50px !important;
    /* Streamlit sets an explicit width (100%) on its own stHorizontalBlock
       containers — left, width, AND right all having definite values at
       once over-constrains a position:fixed box's geometry, and per the
       CSS box-model spec the browser silently drops 'right' to resolve
       it (confirmed directly: right stayed exactly viewport-width past
       its intended edge). width:auto here is what actually lets left+
       right jointly determine this box's width again. */
    width: auto !important;
    height: 40px !important;
    z-index: 100 !important;
    display: flex !important;
    align-items: center !important;
    background: var(--bg-deep) !important;
}
.st-key-_top_bar_nav {
    flex: 0 1 auto !important;
    min-width: 0 !important;
    overflow: hidden !important;
}
.st-key-_top_bar_sessions {
    flex: 1 1 auto !important;
    display: flex !important;
    justify-content: center !important;
    min-width: 0 !important;
    overflow: hidden !important;
}
.st-key-_top_bar_tz {
    flex: 0 0 auto !important;
}
/* Streamlit's own stMarkdownContainer carries a -16px margin-bottom by
   default (a global reset elsewhere in this file doesn't touch it) —
   confirmed directly via getBoundingClientRect on every ancestor: it's
   exactly why the nav and session pills (both rendered via st.markdown)
   sat 8px lower than the timezone selectbox (a native widget, never
   passes through stMarkdownContainer, so never picks up this margin) —
   all three are centered by the same flex align-items:center on
   .st-key-_top_bar, but a negative margin on only two of the three
   children shifts where THEIR content lands within that centered slot.
   Zeroing it here, scoped to just these two markdown children, is what
   actually finishes the alignment a user screenshot caught with X marks
   on every piece that didn't sit on the same line as the rest. */
.st-key-_top_bar_nav [data-testid="stMarkdownContainer"],
.st-key-_top_bar_sessions [data-testid="stMarkdownContainer"] {
    margin-bottom: 0 !important;
}

/* _render_session_pills' own compact live indicator — one pill per
   session (see TRADING_SESSIONS), each a color-coded dot (lit + pulsing
   when that session is open, dim gray when closed — same livepulse
   animation the rest of this file already uses for "live") plus its
   short code. Replaced a full multi-row Gantt-style timeline (hour
   ruler, per-occurrence bands, a NOW line) that didn't fit in the one
   slim row everything else here lives in — this keeps the one thing
   that actually mattered (which session(s) are open right now) at a
   glance, in the space actually available. Refreshed every 60s (its own
   @st.fragment) — a session boundary only needs minute precision. */
.session-pill-row {
    display: flex;
    align-items: center;
    gap: 14px;
    height: 40px;
}
.session-pill {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: 5px;
}
/* High-impact news today for this session's own currencies (see news.py's
   session_news_map) -- a small red badge pulsing at the pill's corner,
   deliberately separate from session-pill-dot's own open/closed color so
   "there's news today" never gets mistaken for "this session is live." */
.session-pill[data-news="true"]::after {
    content: "";
    position: absolute;
    top: -3px;
    right: -4px;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--neon-magenta);
    box-shadow: 0 0 6px var(--neon-magenta);
    animation: livepulse 1.2s ease-in-out infinite;
}
.session-pill-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}
.session-pill[data-active="true"] .session-pill-dot {
    animation: livepulse 1.4s ease-in-out infinite;
}
.session-pill-code {
    font-family: var(--font-system);
    font-variant-numeric: tabular-nums;
    font-weight: 700;
    font-size: 0.72rem;
    letter-spacing: 0.3px;
}
.session-pill[data-active="true"] .session-pill-code {
    color: var(--text-main);
}
.session-pill[data-active="false"] .session-pill-code {
    color: var(--text-dim);
}
/* Below ~900px the nav+sessions+tz row no longer fits at full size — with
   nothing here, the sessions block's centered pill-row overflowed its own
   flex box (min-width:0/overflow:hidden above stops that spillover from
   visually landing on top of the nav/tz text either side of it, but on its
   own that just hard-clips instead). This shrinks everything enough that a
   normal narrow browser window (confirmed directly at 510px, a real width
   this bar had never been tested against) still shows every piece legibly
   instead of relying on the clip. */
@media (max-width: 900px) {
    .project-nav {
        font-size: 0.6rem;
        gap: 6px;
        padding-left: 36px;
    }
    .session-pill-row {
        gap: 8px;
    }
    .session-pill-code {
        font-size: 0.64rem;
    }
}
@media (max-width: 560px) {
    .session-pill-code {
        display: none;
    }
    .session-pill-row {
        gap: 10px;
    }
}
</style>
"""

# Fullscreen terminal, fixed to exactly the viewport at the outermost
# level (html/body/stApp/stMain/stMainBlockContainer all pinned to
# 100vh) — but NOT non-scrolling overall: the one direct child of
# stMainBlockContainer that actually holds page content scrolls
# internally (overflow-y:auto) once that content — chart + an opened
# detail drawer — is taller than the viewport, rather than the browser
# ever growing a page-level scrollbar or the chart's own size changing to
# make room. Built for the chart pages (Markets/Crypto) — the chart
# component itself gets a real fixed height (see the
# stElementContainer:has(iframe) rule below) instead of the static
# height= attribute passed from Python; its own internal html/body/#chart
# already resolve height:100% down to the canvas, so it just needs that
# outer box to actually receive a real pixel height. A page with no chart
# iframe (Backtest) has no reason to ever fill the whole viewport this
# way — applying this there silently ate the page's scrollbar the moment
# content (e.g. the session-window fields) pushed past one screen's
# height, confirmed directly by reproducing it in a shrunk viewport and
# finding stMain/stAppViewContainer/stMainBlockContainer all pinned to
# overflow:hidden with real content still below the fold and genuinely
# unreachable (confirmed again, the same way, while building the
# internal-scroll behavior above: scrollHeight/clientHeight measured
# directly in devtools, not assumed). inject()'s own chart_layout flag is
# the fix — off by default, Markets/Crypto opt in.
CHART_LAYOUT_CSS = """
<style>
html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {
    height: 100vh !important;
    overflow: hidden !important;
}
[data-testid="stMain"] {
    height: 100vh !important;
    overflow: hidden !important;
}
[data-testid="stMainBlockContainer"] {
    height: 100vh !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
    padding: 0.3rem 0.4rem !important;
}
/* Streamlit's default ~1rem gap between stacked elements/columns reads as
   dead space once the layout is this dense (ticker row to chart, chart to
   legend, mini chart to mini chart, column to column) — shrink it
   everywhere rather than fighting it element by element. */
[data-testid="stHorizontalBlock"] {
    gap: 0.3rem !important;
    flex: 1 1 auto;
    min-height: 0;
}
/* overflow-y:auto, not hidden — confirmed directly (devtools measurement)
   this is the ONE box in the whole ancestor chain that's actually
   clamped to the real viewport height (~977px) with genuine overflow
   (content taller than that once a detail drawer opens) — every level
   below it (stLayoutWrapper, stColumn, its own stVerticalBlock) just
   sizes itself to match its own content, never triggering a scrollbar
   of its own. With overflow:hidden here, that overflow had nowhere to
   go: it was silently eaten with no scrollbar anywhere, not clipped-but-
   reachable — confirmed via scrollHeight (1371px) vs clientHeight (977px)
   on this exact element, canScroll true, yet zero pixels of it were
   actually reachable. This is the fix for "anything that occupies more
   space should make a scrollable page." overflow-x stays hidden — this
   box should never scroll sideways. */
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {
    flex: 1 1 auto;
    min-height: 0;
    display: flex;
    flex-direction: column;
    overflow-x: hidden;
    overflow-y: auto;
}
/* Only the row that actually holds a chart iframe should be sized
   distinctly from the top ticker+TF row — the top row also wraps a
   stHorizontalBlock (it's `st.columns([1,3])` too), and matching on that
   alone made it grow right along with the charts row, splitting the
   page's height between them instead of the ticker row just hugging its
   own ~50px of content. Confirmed directly: before this fix the
   ticker/TF row measured 312px tall — nearly all of it empty space above
   the actual charts.
   Deliberately a fixed vh HEIGHT with flex-grow/shrink both OFF, not
   flex:1 1 auto filling whatever's left — that older approach made the
   chart's own size depend on its SIBLINGS (opening a detail expander
   below it shrank the chart to make room, confirmed as the actual
   complaint: "I don't want the chart to resize... anything that
   occupies more space should make a scrollable page"). A real viewport
   resize still legitimately resizes this (vh is relative to the window,
   so shrinking the browser shrinks it too — "responsive up to a point"),
   but sibling content growing underneath it (a footer opening) no longer
   can — the surrounding column's own overflow-y:auto (below) scrolls
   instead. min-height is a hard floor so an extreme window resize can't
   crush it below a readable candle chart. */
/* This wrapper isn't the chart itself — :has(iframe) matches on ANY
   descendant, and since @st.fragment groups its whole output (chart +
   the detail expander below it, see the fragment-boundary rule further
   down) into one shared stLayoutWrapper, this is that combined box. It
   must stay naturally sized (flex:0 0 auto, no explicit height) so it
   can grow when the detail expander opens; the actual fixed-height
   chart lives one level deeper, on the stElementContainer:has(iframe)
   rule below, which wraps ONLY the chart's own single component call.
   Pinning THIS wrapper's height too (an earlier mistake) clipped the
   expander's content without giving it anywhere to scroll, so it spilled
   out and visually overlapped whatever sibling came after it. */
[data-testid="stMainBlockContainer"] [data-testid="stLayoutWrapper"]:has(iframe) {
    flex: 0 0 auto;
}
/* Tried max-height here instead of height, so a column shorter than the
   available space (chart + a short/closed detail drawer) wouldn't leave
   dead black space below it. Reverted: without a definite height,
   overflow-y:auto stopped establishing a real scroll boundary at all —
   content past this box just got silently eaten by
   stMainBlockContainer's own overflow:hidden instead of scrolling
   (unreachable, no scrollbar, no error). Reliable scrolling when the
   detail drawer opens matters more than the empty space when it's
   closed, so this stays height:100%. */
[data-testid="stHorizontalBlock"] [data-testid="stColumn"] {
    height: 100%;
    overflow-y: auto;
}
[data-testid="stColumn"] [data-testid="stVerticalBlock"] {
    height: 100%;
    display: flex;
    flex-direction: column;
    gap: 0.2rem !important;
}
/* Default every nested layout wrapper inside a column (e.g. a sub-row of
   st.columns() for a ticker+TF bar sitting above a chart) to its own
   natural content height — the recurring version of the bug fixed above,
   one nesting level up: ANY stLayoutWrapper the browser's default flex
   rules would otherwise stretch to fill leftover space (its content-sized
   siblings, growing to match) ends up as dead space above/around whatever
   it actually contains. Only the one wrapper that holds a chart iframe
   (the rule below, which — because :has() adds specificity — wins over
   this one regardless of source order) gets a real fixed size instead. */
[data-testid="stColumn"] [data-testid="stLayoutWrapper"] {
    flex: 0 0 auto;
}
/* @st.fragment wraps its own output in one more stLayoutWrapper, nested
   inside the column's already-fixed outer one — same "this is the
   fragment's combined chart+detail-expander box, not the chart alone"
   caveat as the stMainBlockContainer rule above, just one level deeper
   now that the chart render call lives inside a fragment. Stays
   naturally sized for the same reason: the expander inside it needs
   room to actually grow into when opened, rather than being clipped by
   a height set here. */
[data-testid="stColumn"] [data-testid="stLayoutWrapper"]:has(iframe) {
    flex: 0 0 auto;
}
[data-testid="stColumn"] [data-testid="stElementContainer"] {
    flex: 0 0 auto;
}
/* THIS is the actual chart — stElementContainer wraps exactly one
   element (one Python call), so :has(iframe) here can only match the
   ict_chart() component's own container, never the detail expander next
   to it (a separate call, its own separate container). Fixed height so
   opening/closing that sibling expander can never change the chart's own
   size — see this file's own CHART_LAYOUT_CSS docstring for the full
   reasoning ("I don't want the chart to resize... anything that occupies
   more space should make a scrollable page"). */
[data-testid="stColumn"] [data-testid="stElementContainer"]:has(iframe) {
    flex: 0 0 auto;
    height: 65vh;
    min-height: 320px;
    display: flex;
}
[data-testid="stColumn"] [data-testid="stElementContainer"]:has(iframe) > div {
    flex: 1 1 auto;
    min-height: 0;
    height: 100%;
    width: 100%;
}
[data-testid="stCustomComponentV1"] {
    height: 100% !important;
    width: 100% !important;
    display: block !important;
}
</style>
"""


def inject(chart_layout=False):
    st.markdown(CSS, unsafe_allow_html=True)
    if chart_layout:
        st.markdown(CHART_LAYOUT_CSS, unsafe_allow_html=True)


# Every running project page + its port — both apps call render_project_nav
# with their own name so the list (and which entry reads as "current") is
# the same everywhere without hand-syncing it in two files. Add a new
# project here and it shows up in the nav on every page automatically.
PROJECTS = [
    ("Markets", "http://localhost:8501"),
    ("Crypto", "http://localhost:8506"),
    ("News", "http://localhost:8507"),
    ("Institutional", "http://localhost:8508"),
]


def render_project_nav(current):
    parts = []
    for name, url in PROJECTS:
        if parts:
            parts.append('<span class="project-nav-sep">·</span>')
        if name == current:
            parts.append(f'<span class="project-nav-current">{name}</span>')
        else:
            parts.append(f'<a class="project-nav-link" href="{url}" target="_self">{name}</a>')
    st.markdown(f'<div class="project-nav">{"".join(parts)}</div>', unsafe_allow_html=True)


# A curated set of financial-center + common personal timezones — not the
# full IANA database, which would make the selectbox below unusably long.
# IANA names (not fixed UTC offsets) so DST is handled automatically the
# same way America/New_York already is everywhere else in this project.
# Each entry's own short code (e.g. "NY") is what actually shows in the
# compact "UTC-5 NY" selector — the parenthesized long form in the key is
# only ever read as the option's searchable label.
TIMEZONE_OPTIONS = {
    "New York (ET)": ("America/New_York", "NY"),
    "Chicago (CT)": ("America/Chicago", "CHI"),
    "Los Angeles (PT)": ("America/Los_Angeles", "LA"),
    "London (GMT/BST)": ("Europe/London", "LON"),
    "Frankfurt (CET/CEST)": ("Europe/Berlin", "FRA"),
    "Dubai (GST)": ("Asia/Dubai", "DXB"),
    "Mumbai (IST)": ("Asia/Kolkata", "MUM"),
    "Singapore (SGT)": ("Asia/Singapore", "SIN"),
    "Hong Kong (HKT)": ("Asia/Hong_Kong", "HKG"),
    "Tokyo (JST)": ("Asia/Tokyo", "TOK"),
    "Sydney (AEST/AEDT)": ("Australia/Sydney", "SYD"),
    "UTC": ("UTC", "UTC"),
}
DEFAULT_DISPLAY_TZ_LABEL = "New York (ET)"
_DISPLAY_TZ_KEY = "display_tz_label"


def _utc_offset_str(iana):
    offset = pd.Timestamp.now(tz=iana).utcoffset()
    total_min = int(offset.total_seconds() // 60)
    sign = "+" if total_min >= 0 else "-"
    h, m = divmod(abs(total_min), 60)
    return f"UTC{sign}{h}" + (f":{m:02d}" if m else "")


def get_display_tz():
    """The IANA zone every chart's own displayed time (candle axis, OHLC
    corner, live ticks — everything that goes through app.py/
    crypto_app.py's own _ny_fake_utc_seconds, and the matching JS-side
    conversion ict_chart's frontend applies to live WebSocket ticks) is
    currently rendered in — whatever render_top_bar()'s own timezone
    selector last set, or DEFAULT_DISPLAY_TZ_LABEL's zone before it's
    ever been touched.
    Deliberately NOT what kill-zone/session detection logic runs on —
    those stay America/New_York internally regardless (real exchange
    hours don't change because of what you'd prefer to look at), only
    the RENDERED digits follow this."""
    label = st.session_state.get(_DISPLAY_TZ_KEY, DEFAULT_DISPLAY_TZ_LABEL)
    return TIMEZONE_OPTIONS.get(label, TIMEZONE_OPTIONS[DEFAULT_DISPLAY_TZ_LABEL])[0]


# The market's own full session windows (Tokyo/London/New York), Eastern
# wall-clock time — America/New_York resolves EST/EDT automatically
# wherever it's used below, same convention as every other session-timing
# check in this project. Deliberately separate from app.py/crypto_app.py's
# own KILL_ZONES: those are narrow ICT-specific high-probability windows
# WITHIN a session (e.g. "NY AM" is only the first 2 hours of New York's
# 9-hour session) meant for chart shading, not "is the market open" —
# different question, different (wider) hours, kept as its own constant
# rather than reusing/renaming those.
TRADING_SESSIONS = [
    ("Asia", dtime(19, 0), dtime(4, 0), "ASIA"),  # Tokyo — wraps past midnight ET
    ("London", dtime(3, 0), dtime(12, 0), "LON"),
    ("New York", dtime(8, 0), dtime(17, 0), "NY"),
]

SESSION_COLORS = {
    "Asia": NEON_CYAN,
    "London": "#FFD60A",  # systemYellow — already this project's own EMA 20 color
    "New York": NEON_GREEN,
}


def _session_is_active(now_t, start, end):
    if start <= end:
        return start <= now_t < end
    return now_t >= start or now_t < end  # wraps midnight (Asia)


def _market_closed_for_weekend(now):
    """Global forex/CFD market closes Friday 17:00 ET, reopens Sunday
    17:00 ET. _session_is_active only ever looks at time-of-day, so
    without this, Saturday (or Sunday before reopen) would still show
    whichever session's clock-hours happen to overlap right now as
    falsely "open" — e.g. London's 03:00-12:00 window lighting up on a
    Saturday morning even though nothing is trading."""
    wd, t = now.weekday(), now.time()  # Monday=0 ... Sunday=6
    if wd == 5:  # Saturday — closed all day
        return True
    if wd == 4 and t >= dtime(17, 0):  # Friday after close
        return True
    if wd == 6 and t < dtime(17, 0):  # Sunday before reopen
        return True
    return False


def render_top_bar(current):
    """The whole top strip in one row — project nav, live Asia/London/New
    York session pills, and the display-timezone selector — with the "☰"
    popover trigger (defined separately, deep in app.py/crypto_app.py's
    own chart-rendering code) pinned on top of the same right-hand corner
    this row's own padding-right leaves clear for it. Replaced three
    separately-stacked pieces (render_project_nav's own row, a full
    Gantt-style multi-row session timeline, a second-row timezone
    selector) per direct request to fit everything into the one slim bar
    the nav and "☰" already lived in — the detailed timeline (hour ruler,
    per-occurrence bands, a NOW line) traded away real information for
    space it didn't have; a compact pill per session (color-coded dot +
    code, lit and pulsing when that session is open) keeps the one thing
    that actually mattered at a glance. position:fixed (see this row's
    own CSS) takes it out of normal document flow, so the plain spacer
    div below is what keeps the ticker-search box underneath from
    rendering hidden behind it — 40px matches the row's own CSS height
    exactly, kept in sync there, not recomputed here."""
    with st.container(key="_top_bar", horizontal=True, vertical_alignment="center", gap="small"):
        with st.container(key="_top_bar_nav"):
            render_project_nav(current)
        with st.container(key="_top_bar_sessions"):
            _render_session_pills()
        with st.container(key="_top_bar_tz"):
            _render_tz_select()
    st.markdown('<div style="height:40px;"></div>', unsafe_allow_html=True)


def _render_session_pills():
    @st.fragment(run_every=60)
    def _inner():
        now = pd.Timestamp.now(tz="America/New_York")
        now_t = now.time()
        market_closed = _market_closed_for_weekend(now)
        try:
            news_by_session = news.session_news_map()
        except Exception:
            # A feed outage/timeout must never take the whole top bar down
            # with it -- pills just render with no news flash that tick.
            news_by_session = {}
        pills = []
        for name, start, end, code in TRADING_SESSIONS:
            active = (not market_closed) and _session_is_active(now_t, start, end)
            color = SESSION_COLORS[name]
            hours = f'{start.strftime("%H:%M")}–{end.strftime("%H:%M")} ET'
            status = "closed for the weekend" if market_closed else ("open now" if active else "closed")
            todays_news = news_by_session.get(name, [])
            title = f"{name} session — {status} ({hours}). Sessions can overlap."
            if todays_news:
                items = "; ".join(
                    f'{row["title"]} ({row["country"]}, {row["time"].tz_convert("America/New_York").strftime("%H:%M")} ET)'
                    for row in todays_news[:4]
                )
                title += f" High-impact news today: {items}."
            dot_style = f"background:{color}; box-shadow:0 0 6px {color};" if active else "background:#48484a;"
            pills.append(
                f'<span class="session-pill" data-active="{"true" if active else "false"}" '
                f'data-news="{"true" if todays_news else "false"}" title="{html.escape(title)}">'
                f'<span class="session-pill-dot" style="{dot_style}"></span>'
                f'<span class="session-pill-code">{code}</span>'
                f'</span>'
            )
        st.markdown(f'<div class="session-pill-row">{"".join(pills)}</div>', unsafe_allow_html=True)

    _inner()


def _render_tz_select():
    """A plain st.selectbox, not a live-updating clock: the compact
    "UTC-5 NY"-style label only changes twice a year (DST rollover), so
    unlike _render_session_pills this needs no fragment. Reading the
    result is get_display_tz() — every _ny_fake_utc_seconds call site in
    app.py/crypto_app.py, and every ict_chart(display_tz=...) call, goes
    through that, not this function directly."""
    labels = list(TIMEZONE_OPTIONS.keys())
    current = st.session_state.get(_DISPLAY_TZ_KEY, DEFAULT_DISPLAY_TZ_LABEL)
    st.selectbox(
        "Display timezone", labels, index=labels.index(current) if current in labels else 0,
        key=_DISPLAY_TZ_KEY, label_visibility="collapsed", width=120,
        format_func=lambda label: f"{_utc_offset_str(TIMEZONE_OPTIONS[label][0])} {TIMEZONE_OPTIONS[label][1]}",
        help="Every chart's displayed time follows this — candle axis, OHLC corner, live ticks. "
             "Session hours (the pills on the left) stay in their own real-world hours regardless.",
    )
