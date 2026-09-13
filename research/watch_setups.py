"""
The "watch already-computed setups for a notable change" script — runs
every 5 minutes via a Claude Code scheduled task (not launchd: this one's
short-lived and cheap enough that hand-rolling a plist felt like overkill
for something manageable from chat), separate from live_scan.py's own
hourly pattern-DETECTION job on purpose: this does no network refresh and
no re-detection, just re-evaluates setups already computable from whatever
OHLCV is already cached, so running it 12x as often as live_scan costs
nothing extra in API calls.

"Notable" was scoped down from an earlier draft that also considered
reward:risk — dropped once it became clear every setup carries an
identical fixed 2R by construction (research/setups.compute_setup), so it
can never distinguish one setup from another. What's left is genuinely
discriminating and fully deterministic, so this stays a plain script
(matching live_scan.py's own judas_swing-only alert, which already covers
one of these three cases) rather than an LLM judgment call re-run every 5
minutes for no reason:
  1. A setup's status just flipped to hit_tp or hit_sl (new — nothing
     watches already-active setups for resolution today).
  2. A judas_swing setup just went active (already live_scan.py's own bar
     for "worth a notification" — kept here too so a Judas swing whose
     live_events.csv row landed between two live_scan runs, or that
     resolves later, still surfaces even though live_scan itself only
     notifies once, at detection).
  3. A setup's pattern is edge_lab-validated for its market (rare today —
     edge_lab has only ever tested GBPUSD, and live_scan doesn't scan it —
     see research/setups.validation_badge).

State (research/.cache/watch_state.json) remembers each setup's
last-notified status, keyed the same way live_events.csv itself dedupes
(ticker, interval, event_type, start) — so a resolved setup notifies once
on the transition, not every 5 minutes for as long as it stays resolved.
"""

import json
import os
import subprocess

from research.setups import load_live_setups, validation_badge

_STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "watch_state.json")


def _notify(title, message):
    safe_title = title.replace('"', "'")
    safe_message = message.replace('"', "'")
    try:
        subprocess.run(
            ["osascript", "-e", f'display notification "{safe_message}" with title "{safe_title}" sound name "Glass"'],
            capture_output=True, timeout=5,
        )
    except Exception as e:
        print(f"  (notification failed: {e})")


def _load_state():
    if not os.path.exists(_STATE_PATH):
        return {}
    try:
        with open(_STATE_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_state(state):
    os.makedirs(os.path.dirname(_STATE_PATH), exist_ok=True)
    with open(_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def _key(s):
    return f"{s['ticker']}|{s['interval']}|{s['event_type']}|{s['start']}"


_MAX_INDIVIDUAL_NOTIFICATIONS = 3


def check_notable():
    # A missing state file means this is the very first run ever — not
    # "everything currently active/resolved just happened". Without this
    # distinction, a cold start reads every already-resolved setup as a
    # fresh resolution and every judas_swing as freshly formed: confirmed
    # directly, an early test run with no state file fired 3291 individual
    # osascript notifications (and took 14 real minutes doing it, since
    # each osascript subprocess call costs ~0.25s — see _notify). First run
    # only seeds the baseline; genuine transitions only count from the
    # second run onward.
    is_first_run = not os.path.exists(_STATE_PATH)

    setups = load_live_setups()
    state = _load_state()
    new_state = {}
    notable = []

    for s in setups:
        key = _key(s)
        prev_status = state.get(key)
        new_state[key] = s["status"]
        if is_first_run:
            continue

        resolved_now = s["status"] in ("hit_tp", "hit_sl") and prev_status == "active"
        fresh_judas = s["event_type"] == "judas_swing" and prev_status is None
        fresh_validated = prev_status is None and validation_badge(s["event_type"], s["ticker"])[1] == "bullish"

        if resolved_now:
            notable.append((s, f"{'Hit target' if s['status'] == 'hit_tp' else 'Hit stop'} — "
                                f"{s['ticker']} {s['interval']} {s['event_type']} ({s['direction']}), "
                                f"entry {s['entry_price']:.5g}, resolved {s['resolved_price']:.5g}"))
        elif fresh_judas:
            notable.append((s, f"Judas swing setup — {s['ticker']} {s['interval']} ({s['direction']}), "
                                f"entry {s['entry_price']:.5g}, stop {s['sl_price']:.5g}, target {s['tp_price']:.5g}"))
        elif fresh_validated:
            notable.append((s, f"Validated-pattern setup — {s['ticker']} {s['interval']} {s['event_type']} "
                                f"({s['direction']}), entry {s['entry_price']:.5g}"))

    # Capped, not one osascript call per item — a burst (e.g. after this
    # script was down for a while, or a volatile session resolves several
    # setups at once) firing dozens of individual banner notifications back
    # to back is its own kind of spam, and each subprocess call has real
    # (~0.25s) overhead on top of that.
    if 0 < len(notable) <= _MAX_INDIVIDUAL_NOTIFICATIONS:
        for s, message in notable:
            _notify(f"Setup — {s['ticker']} {INTERVAL_LABEL(s['interval'])}", message)
    elif len(notable) > _MAX_INDIVIDUAL_NOTIFICATIONS:
        _notify(f"{len(notable)} setups worth a look", "Open the Setups page — several changed at once.")

    _save_state(new_state)
    print(f"Checked {len(setups)} setup(s), {len(notable)} notable{' (first run — baseline only, no notifications)' if is_first_run else ''}.")
    for s, message in notable:
        print(f"  - {message}")
    return notable


def INTERVAL_LABEL(interval):
    return {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "60m": "1h", "1d": "1d"}.get(interval, interval)


if __name__ == "__main__":
    check_notable()
