"""
ICT kill-zone session windows (NY time) — duplicated from app.py's own
KILL_ZONES rather than imported from it, deliberately: app.py is a runnable
Streamlit script (module-level st.set_page_config() etc.), not a safe
library for another Streamlit process to import — doing so would re-execute
its entire page. Small, stable, ICT-standard constants; mirror any change
here if app.py's own definition ever changes.

filter_events_to_sessions() restrains a backtest to specific session
windows ("only count Judas swing setups that triggered during NY AM," not
scanning all 24 hours indiscriminately) — but it filters the EXTRACTED
EVENTS by their trigger time, not the raw OHLCV bars before detection. That
distinction is load-bearing: app.py's own chart used to trim candles down
to just the kill-zone window and moved away from it on purpose (see its own
comment: "Kill zone no longer trims the candles down to just that window").
Fair Value Gaps, order blocks, and swing points all depend on temporally
CONSECUTIVE bars — trimming out the hours between sessions first would
create artificial gaps that corrupt those patterns before detection even
runs. Detection always sees the full continuous history; only the resulting
events get filtered down to the sessions you actually care about.
"""

from datetime import time as dtime

import pandas as pd

KILL_ZONES = {
    "NY AM (9:30–11:30 ET)": [(dtime(9, 30), dtime(11, 30))],
    "NY PM (13:30–16:00 ET)": [(dtime(13, 30), dtime(16, 0))],
    "London Open (02:00–05:00 ET)": [(dtime(2, 0), dtime(5, 0))],
    "Asian Session (19:00–21:00 ET)": [(dtime(19, 0), dtime(21, 0))],
    "RTH (9:30–16:00 ET)": [(dtime(9, 30), dtime(16, 0))],
    "ETH (pre + after hours)": [(dtime(4, 0), dtime(9, 30)), (dtime(16, 0), dtime(20, 0))],
}


def filter_events_to_sessions(events_df, session_names, time_col="entry_time"):
    """Keeps only rows whose time_col, converted to NY local time, falls
    inside any window of any selected session. Empty/None session_names
    returns events_df unchanged — no restriction, the default."""
    if not session_names or events_df.empty:
        return events_df
    times = pd.to_datetime(events_df[time_col])
    if times.dt.tz is None:
        times = times.dt.tz_localize("UTC")
    ny_time = times.dt.tz_convert("America/New_York").dt.time

    mask = pd.Series(False, index=events_df.index)
    for name in session_names:
        for start, end in KILL_ZONES[name]:
            mask = mask | ny_time.apply(lambda t, s=start, e=end: s <= t <= e)
    return events_df[mask]
