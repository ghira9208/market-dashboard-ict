import streamlit as st

NEON_CYAN = "#0A84FF"
NEON_MAGENTA = "#FF453A"
NEON_GREEN = "#30D158"
NEON_AMBER = "#FF9F0A"

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
    border-right: 1px solid var(--separator);
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
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
    /* Left padding clears stExpandSidebarButton — it lives in stHeader,
       position:absolute at top:16/left:19/width:28 (see theme.py's own
       header-fix comment above), which sits directly on top of this nav's
       default flush-left text whenever the sidebar is collapsed. Confirmed
       directly via a user screenshot: the chevron rendered right on top of
       "MARKETS". Always reserved (not just when collapsed) — simpler than
       detecting collapse state in CSS, and the extra gap is invisible when
       the button isn't there. */
    padding: 0 0 6px 44px; flex-shrink: 0;
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
</style>
"""

# Fullscreen, non-scrolling terminal: fixed to exactly the viewport, no
# page-level scroll ("nothing outside of it"). Built for the chart pages
# (Markets/Crypto) — the chart component fills whatever's left after the
# ticker row via flex; its own internal html/body/#chart already resolve
# height:100% down to the canvas, so it just needs its outer iframe box to
# actually receive a real pixel height instead of the static height=
# attribute passed from Python. A page with no chart iframe (Backtest) has
# no reason to ever fill the whole viewport and clip its own overflow —
# applying this there silently ate the page's scrollbar the moment content
# (e.g. the session-window fields) pushed past one screen's height,
# confirmed directly by reproducing it in a shrunk viewport and finding
# stMain/stAppViewContainer/stMainBlockContainer all pinned to
# overflow:hidden with real content still below the fold. inject()'s own
# chart_layout flag is the fix — off by default, Markets/Crypto opt in.
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
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {
    flex: 1 1 auto;
    min-height: 0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}
/* Only the row that actually holds a chart iframe should flex-grow to fill
   leftover height — the top ticker+TF row also wraps a stHorizontalBlock
   (it's `st.columns([1,3])` too), and matching on that alone made it grow
   right along with the charts row, splitting the page's height between them
   instead of the ticker row just hugging its own ~50px of content. Confirmed
   directly: before this fix the ticker/TF row measured 312px tall — nearly
   all of it empty space above the actual charts. */
[data-testid="stMainBlockContainer"] [data-testid="stLayoutWrapper"]:has(iframe) {
    flex: 1 1 auto;
    min-height: 0;
}
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
   this one regardless of source order) should ever flex-grow. */
[data-testid="stColumn"] [data-testid="stLayoutWrapper"] {
    flex: 0 0 auto;
}
/* @st.fragment wraps its own output in one more stLayoutWrapper, nested
   inside the column's already-fixed outer one — same "wrapper doesn't
   inherit height, so is left the default column-of-content height, and
   the flex-fill chain right in front of it before it can reach the
   iframe" break as before, just one level deeper now that the chart
   render call lives inside a fragment. */
[data-testid="stColumn"] [data-testid="stLayoutWrapper"]:has(iframe) {
    flex: 1 1 auto;
    min-height: 0;
}
[data-testid="stColumn"] [data-testid="stElementContainer"] {
    flex: 0 0 auto;
}
[data-testid="stColumn"] [data-testid="stElementContainer"]:has(iframe) {
    flex: 1 1 auto;
    min-height: 0;
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
    ("Backtest", "http://localhost:8507"),
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
