"""
Real bidirectional Streamlit component wrapping lightweight-charts (canvas-
based, TradingView's own open-source renderer) — replaces the earlier
Plotly + hand-rolled pan/zoom-JS approach.

The key property this buys over `st.components.v1.html()`: `declare_component`
serves the frontend from a stable `src=` URL, so the iframe navigates ONCE on
first mount. Every later call to `ict_chart(...)` — including once-a-second
fragment reruns — delivers new args via postMessage into that SAME running
page, not a fresh `srcdoc=` reload. That's what makes `series.update()`
(patch just the last bar) possible instead of tearing down and rebuilding the
whole chart on every tick.
"""

import os

import streamlit.components.v1 as components

_component_func = components.declare_component(
    "ict_chart",
    path=os.path.join(os.path.dirname(__file__), "frontend"),
)


def ict_chart(bars, fingerprint, overlays=None, options=None, tick_labels=None, indicators=None,
              ohlc=None, height=700, key="ict_chart"):
    """
    bars: list of {"time": <seconds, int>, "open", "high", "low", "close", "volume"} dicts, ascending,
          unique times. `time` should be pre-encoded as "fake UTC" — the wall-clock digits of whatever
          timezone you want displayed, packed into a UTCTimestamp as if already UTC (lightweight-charts
          has no timezone setting of its own; this is the standard workaround). When a kill-zone-style
          session filter is active and gap-free display is wanted, `time` should instead be sequential
          bar indices (0,1,2,...) — pair that with `tick_labels` so the axis still shows real times.
    fingerprint: any string that changes exactly when a FULL rebuild is needed (ticker, timeframe,
                 kill zone, log-scale toggle, or bar count/first-bar all factor in) — unchanged means
                 "same dataset, only the last bar may have moved," which the frontend patches
                 incrementally via series.update() instead of a full setData() rebuild.
    overlays: {
        "rectangles": [{"t0", "t1": <same units as bar time>, "p0", "p1": price, "fill": css color, "border": css color|None,
            "label": str|None — drawn top-left of the rect in the border color; only shows when border is set,
            "border_width": int|None (default 1) — stroke width in CSS px,
            "fill_to": css color|None — when set, fills as a horizontal gradient from "fill" (left) to this
                color (right) instead of a flat wash; used for FVG mitigation slivers (fading out the eaten-
                into portion) rather than a uniform tint}],
        "price_lines": [{"t0", "t1": <same units as bar time>, "price": float, "color": css color, "title": str}] —
            drawn as a dashed segment from t0 to t1 (NOT a full-width native price line — there's no such
            thing as a time-bounded native price line, so this is a hand-drawn primitive instead), with
            the title as an in-pane label at its right end.
        "markers": [{"time": <same units as bar time>, "position": "aboveBar"|"belowBar", "color": css color, "shape": "arrowUp"|"arrowDown"}],
        "volume_profile": [{"price_low", "price_high": float, "volume_frac": float in (0, 1] (the busiest
            bucket is exactly 1.0), "color": css color}] — a price-bucketed volume histogram (see
            indicators.volume_profile), drawn as horizontal bars anchored to the pane's own LEFT edge
            (fixed pixel position, NOT tied to any bar's time) — each bar's own length is volume_frac
            times a fixed fraction of the pane's current width, so bars stay proportional to each other
            at any zoom level. Order doesn't matter; color is fully pre-computed in Python (e.g. the
            Point-of-Control bucket highlighted, the rest a uniform dim tone).
    }
    options: {"log_scale": bool, "volume": bool, "selected": bool} — "selected" draws a thin
        inset border around the chart, for a multi-chart page to show which one a shared
        control (e.g. a timeframe picker) currently applies to.

    Returns a value that changes (a fresh timestamp) each time the user clicks the chart
    (a plain click, not a pan-drag) — compare it against the last-seen value to detect a
    "this chart was picked" event; None/unchanged means no new click since the last read.
    tick_labels: {<bar time>: "display label"} — only needed for the sequential-index gap-free case above;
                 omit (or leave bar times as real fake-UTC seconds) and the axis formats them directly.
    indicators: {
        "ma20": [{"time", "value"}] | None — overlaid on the main price pane.
        "ma50": [{"time", "value"}] | None — same shape, own distinct color, same pane.
        "bb": {"upper": [...], "basis": [...], "lower": [...]} | None — same shape as ma20/ma50, main pane.
        "rsi": [{"time", "value"}] | None — its own pane below, with 30/70 reference lines.
        "macd": {"macd": [...], "signal": [...], "histogram": [...]} | None — its own pane below;
                histogram points may include "color" for pos/neg bars.
      Each list entry's "time" uses the same units as bar time. Any key omitted/None/empty is
      simply not drawn — indicator panes for RSI/MACD only exist when their data is present.
    }
    key: MUST stay a constant across reruns for the SAME logical chart (don't derive it from ticker/data) —
         Streamlit auto-keys unkeyed/content-varying component calls, and a key that changes when the
         data changes forces a full iframe remount, silently defeating incremental updates entirely.
         Confirmed directly: an unstable key discarded the whole persistent-JS-state premise this
         component exists for. Pass a different literal key only when you actually want a second,
         independent chart instance on the same page.
    """
    return _component_func(
        bars=bars,
        fingerprint=fingerprint,
        overlays=overlays or {},
        options=options or {},
        tick_labels=tick_labels or {},
        indicators=indicators or {},
        ohlc=ohlc or {},
        height=height,
        key=key,
        default=None,
    )
