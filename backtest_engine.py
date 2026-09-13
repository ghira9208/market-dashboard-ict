"""Shared FVG/OB rule-sweep engine for the Backtest page.

Runs backtest_custom_rule (recommender.py) across a grid of detector/rank/
direction combos per ticker/timeframe, in a background thread so the
Streamlit page stays responsive, and persists every completed result to a
local SQLite file so results survive app restarts and accumulate across
runs. This replaces the earlier workflow where sweeps were run by hand
through standalone scripts and results lived only in a separate published
Artifact's own database — same math (backtest_custom_rule, unchanged), now
running natively in the app instead of relayed through Claude.
"""
import json
import sqlite3
import threading
import time
import uuid
from datetime import time as dtime
from pathlib import Path

from data import get_yf_ohlcv, resample_ohlc
from news import blackout_mask
from recommender import backtest_custom_rule, rule_describe

DB_PATH = Path(__file__).parent / "backtest_results.db"

TF_CONFIG = {
    "1m":  {"fetch_interval": "1m",  "period": "7d",   "resample": None},
    "5m":  {"fetch_interval": "5m",  "period": "60d",  "resample": None},
    "15m": {"fetch_interval": "15m", "period": "60d",  "resample": None},
    "30m": {"fetch_interval": "30m", "period": "60d",  "resample": None},
    "1h":  {"fetch_interval": "60m", "period": "180d", "resample": None},
    "4h":  {"fetch_interval": "60m", "period": "730d", "resample": "4h"},
    "1D":  {"fetch_interval": "1d",  "period": "2y",   "resample": None},
    "1W":  {"fetch_interval": "1wk", "period": "10y",  "resample": None},
    "1M":  {"fetch_interval": "1mo", "period": "max",  "resample": None},
    "1Y":  {"fetch_interval": "3mo", "period": "max",  "resample": "1YE"},
}

# One shared dict for the whole server process (all sessions see the same
# jobs) — plain module global rather than st.session_state since a
# background thread has no Streamlit session of its own to write into.
_JOBS = {}
_JOBS_LOCK = threading.Lock()


# Columns added after the table already had real accumulated results in
# it — migrated in with ALTER TABLE below rather than a fresh CREATE, so
# existing rows (and the sweeps that produced them) aren't lost. A
# pre-migration row reads back with these as SQL NULL — genuinely
# "not recorded," not "recorded as off/none" — the two are kept distinct
# rather than collapsed, since only NEW rows can honestly claim the
# session was off vs. just never captured.
_RESULTS_EXTRA_COLUMNS = {
    "session_label": "TEXT", "min_rr": "REAL", "min_stop_pct": "REAL",
    "min_stop_abs": "REAL", "max_trades_per_session": "INTEGER",
    "entry_zone_point": "REAL", "exit_zone_point": "REAL", "stop_zone_point": "REAL",
    "max_scan_bars": "INTEGER", "cost_pct": "REAL", "avg_net_r": "REAL", "rule_json": "TEXT",
    "news_blackout_label": "TEXT",
}


def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT, requested_at REAL,
            ticker TEXT, timeframe TEXT, direction TEXT,
            entry TEXT, exit_rule TEXT, stop TEXT,
            n_trades INTEGER, wins INTEGER, win_rate REAL,
            avg_rr REAL, bars_used INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY, requested_at REAL, config TEXT,
            status TEXT, total_combos INTEGER, done_combos INTEGER,
            elapsed_seconds REAL
        )
    """)
    existing = {row[1] for row in conn.execute("PRAGMA table_info(results)")}
    for col, col_type in _RESULTS_EXTRA_COLUMNS.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE results ADD COLUMN {col} {col_type}")
    return conn


def _session_label(req):
    s = req.get("session")
    return f"{s['start']}-{s['end']}" if s else "None"


def _news_blackout_label(req):
    nb = req.get("news_blackout")
    return f"±{nb['minutes_before']}/{nb['minutes_after']}min" if nb else "None"


def build_combos(req):
    """Entry's side is locked to direction, same as exit/stop — bullish
    enters below price, bearish enters above — matching the fix already
    applied to the live Rules tab (a free-choice entry side produced a
    large fraction of guaranteed-0-trade combos, confirmed directly in
    real sweep output).

    Entry/exit(target)/stop each get their own zone_point (0.0-1.0,
    bottom to top of the zone) instead of one shared value — e.g. entry
    at 75% of the zone, stop at 25% — matching the live Rules tab's own
    per-field control. Each is a single fixed choice for the whole sweep,
    not swept across combos (same as when there was only one shared
    value): with 5 quarter-steps each, sweeping all three would multiply
    combo count by 125x, which nothing about this request asked for."""
    entry_zone_point = req.get("entry_zone_point", req.get("zone_point", 0.5))
    exit_zone_point = req.get("exit_zone_point", req.get("zone_point", 0.5))
    stop_zone_point = req.get("stop_zone_point", req.get("zone_point", 0.5))
    combos = []
    for direction in req["directions"]:
        entry_side = "below" if direction == "bullish" else "above"
        exit_side = "above" if direction == "bullish" else "below"
        stop_side = "below" if direction == "bullish" else "above"
        for entry_det in req["detectors"]:
            for entry_n in range(req["entry_rank"][0], req["entry_rank"][1] + 1):
                entry_rule = {"anchor": "next_event", "detector": entry_det, "side": entry_side,
                              "n": entry_n, "zone_point": entry_zone_point}
                for exit_det in req["detectors"]:
                    for exit_n in range(req["exit_rank"][0], req["exit_rank"][1] + 1):
                        exit_rule = {"anchor": "next_event", "detector": exit_det, "side": exit_side,
                                     "n": exit_n, "zone_point": exit_zone_point}
                        for stop_det in req["detectors"]:
                            for stop_n in range(req["stop_rank"][0], req["stop_rank"][1] + 1):
                                stop_rule = {"anchor": "next_event", "detector": stop_det, "side": stop_side,
                                             "n": stop_n, "zone_point": stop_zone_point}
                                combos.append((direction, entry_rule, exit_rule, stop_rule))
    return combos


def estimate_seconds(req, combos_count=None):
    """Mirrors the published Artifact dashboard's own calibrated formula
    (seconds-per-combo scales with bar count ~bars^1.279; a session window
    speeds things up since bars outside it skip detection entirely; quality
    filters slow things down since a rejected setup still pays full
    detection cost). Kept in sync by hand — same constants, no shared
    import between the artifact's JS and this Python module.

    Doesn't yet factor in a non-default max_scan_bars (raising it makes
    every per-zone fill-scan more expensive, on top of everything already
    modeled here) — calibration_multiplier() still self-corrects the
    overall estimate against real elapsed time, just noisier once sweeps
    at different max_scan_bars settings are mixed into its own history."""
    if combos_count is None:
        combos_count = len(build_combos(req)) * len(req["tickers"]) * len(req["timeframes"])
    bars = req["bar_cap"]
    per_combo = 0.000683 * (bars ** 1.279)
    speedup = 1.0
    if req.get("session"):
        sh, sm = (int(x) for x in req["session"]["start"].split(":"))
        eh, em = (int(x) for x in req["session"]["end"].split(":"))
        hours = ((eh * 60 + em) - (sh * 60 + sm)) / 60.0
        if hours > 0:
            speedup = min(20.0 / hours, 15.0)
    slowdown = 1.5 if (req.get("min_rr") or req.get("min_stop_pct") or req.get("min_stop_abs")) else 1.0
    return combos_count * per_combo * slowdown / speedup


def _parse_session(req):
    s = req.get("session")
    if not s:
        return None
    sh, sm = (int(x) for x in s["start"].split(":"))
    eh, em = (int(x) for x in s["end"].split(":"))
    return (dtime(sh, sm), dtime(eh, em))


def get_jobs():
    return _JOBS


def start_sweep(req, script_ctx=None):
    """Kicks off a sweep in a background thread and returns its run_id
    immediately. script_ctx (from streamlit.runtime.scriptrunner.
    get_script_run_ctx(), called on the main thread before this) is
    attached to the worker thread so get_yf_ohlcv's own st.cache_data
    bookkeeping doesn't warn about a missing context — same pattern
    app.py's _prefetch already uses for its own background fetches."""
    run_id = uuid.uuid4().hex[:12]
    combos = build_combos(req)
    total = len(combos) * len(req["tickers"]) * len(req["timeframes"])
    job = {
        "run_id": run_id, "status": "running", "total": total, "done": 0,
        "started": time.time(), "elapsed": 0.0, "error": None,
        "current": "", "cancel": False, "req": req,
    }
    with _JOBS_LOCK:
        _JOBS[run_id] = job

    def _worker():
        if script_ctx is not None:
            from streamlit.runtime.scriptrunner import add_script_run_ctx
            add_script_run_ctx(threading.current_thread(), script_ctx)
        conn = _connect()
        conn.execute(
            "INSERT INTO runs (run_id, requested_at, config, status, total_combos, done_combos, elapsed_seconds) "
            "VALUES (?,?,?,?,?,?,?)",
            (run_id, job["started"], json.dumps(req), "running", total, 0, 0.0),
        )
        conn.commit()
        session = _parse_session(req)
        try:
            for ticker in req["tickers"]:
                if job["cancel"]:
                    break
                for tf_label in req["timeframes"]:
                    if job["cancel"]:
                        break
                    job["current"] = f"{ticker} / {tf_label}"
                    conf = TF_CONFIG[tf_label]
                    df = get_yf_ohlcv(ticker, period=conf["period"], interval=conf["fetch_interval"])
                    if conf["resample"] and not df.empty:
                        df = resample_ohlc(df, conf["resample"])
                    df = df.tail(req["bar_cap"])
                    nb = req.get("news_blackout")
                    # Computed per ticker/timeframe (not once for the whole
                    # sweep like `session`) — unlike a time-of-day window,
                    # which currency's news counts depends on the ticker
                    # itself, and the event times have to be resolved
                    # against THIS df's own actual bar range.
                    nb_mask = blackout_mask(df.index, ticker, nb["minutes_before"], nb["minutes_after"]) if nb else None
                    for direction, entry_rule, exit_rule, stop_rule in combos:
                        if job["cancel"]:
                            break
                        r = backtest_custom_rule(
                            df, entry_rule, exit_rule, stop_rule, session=session,
                            min_rr=req.get("min_rr"), min_stop_pct=req.get("min_stop_pct"),
                            min_stop_abs=req.get("min_stop_abs"),
                            max_trades_per_session=req.get("max_trades_per_session"),
                            max_scan_bars=req.get("max_scan_bars", 500),
                            cost_pct=req.get("cost_pct"),
                            news_blackout_mask=nb_mask,
                        )
                        avg_rr = None
                        if r["trades"]:
                            rrs = [abs(t["target"] - t["entry"]) / abs(t["entry"] - t["stop"])
                                   for t in r["trades"] if t["entry"] != t["stop"]]
                            avg_rr = (sum(rrs) / len(rrs)) if rrs else None
                        # The exact structured rule, not just its plain-
                        # English description — lets a caller (the Rules
                        # tab's "use this rule on the chart" flow) rebuild
                        # this precise combo from a results-table row and
                        # re-simulate it against whatever ticker/timeframe
                        # is actually on screen, instead of only being able
                        # to read what it was in words.
                        rule_json = json.dumps({
                            "direction": direction, "entry_rule": entry_rule,
                            "exit_rule": exit_rule, "stop_rule": stop_rule,
                        })
                        conn.execute(
                            "INSERT INTO results (run_id, requested_at, ticker, timeframe, direction, entry, "
                            "exit_rule, stop, n_trades, wins, win_rate, avg_rr, bars_used, session_label, "
                            "min_rr, min_stop_pct, min_stop_abs, max_trades_per_session, entry_zone_point, "
                            "exit_zone_point, stop_zone_point, max_scan_bars, cost_pct, avg_net_r, rule_json, "
                            "news_blackout_label) "
                            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (run_id, job["started"], ticker, tf_label, direction,
                             rule_describe(entry_rule), rule_describe(exit_rule), rule_describe(stop_rule),
                             r["n"], r["wins"], r["win_rate"], avg_rr, len(df), _session_label(req),
                             req.get("min_rr"), req.get("min_stop_pct"), req.get("min_stop_abs"),
                             req.get("max_trades_per_session"), entry_rule.get("zone_point"),
                             exit_rule.get("zone_point"), stop_rule.get("zone_point"),
                             req.get("max_scan_bars", 500), req.get("cost_pct"), r.get("avg_net_r"), rule_json,
                             _news_blackout_label(req)),
                        )
                        job["done"] += 1
                        job["elapsed"] = time.time() - job["started"]
                        if job["done"] % 25 == 0:
                            conn.commit()
                            conn.execute("UPDATE runs SET done_combos=?, elapsed_seconds=? WHERE run_id=?",
                                         (job["done"], job["elapsed"], run_id))
            conn.commit()
            job["status"] = "cancelled" if job["cancel"] else "completed"
            conn.execute("UPDATE runs SET status=?, done_combos=?, elapsed_seconds=? WHERE run_id=?",
                         (job["status"], job["done"], job["elapsed"], run_id))
            conn.commit()
        except Exception as e:
            job["status"] = "error"
            job["error"] = str(e)
            conn.execute("UPDATE runs SET status=? WHERE run_id=?", ("error", run_id))
            conn.commit()
        finally:
            conn.close()

    threading.Thread(target=_worker, daemon=True).start()
    return run_id


def cancel_sweep(run_id):
    job = _JOBS.get(run_id)
    if job:
        job["cancel"] = True


def load_results(ticker=None, timeframe=None, direction=None, session_label=None,
                  news_blackout_label=None, min_n_trades=1, limit=20000):
    conn = _connect()
    conn.row_factory = sqlite3.Row
    q = "SELECT * FROM results WHERE n_trades >= ?"
    params = [min_n_trades]
    if ticker:
        q += " AND ticker = ?"; params.append(ticker)
    if timeframe:
        q += " AND timeframe = ?"; params.append(timeframe)
    if direction:
        q += " AND direction = ?"; params.append(direction)
    if session_label:
        q += " AND session_label = ?"; params.append(session_label)
    if news_blackout_label:
        q += " AND news_blackout_label = ?"; params.append(news_blackout_label)
    q += " ORDER BY requested_at DESC LIMIT ?"
    params.append(limit)
    rows = [dict(r) for r in conn.execute(q, params)]
    conn.close()
    return rows


def load_runs(limit=50):
    conn = _connect()
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM runs ORDER BY requested_at DESC LIMIT ?", (limit,))]
    conn.close()
    return rows


def calibration_multiplier():
    """Average of (actual elapsed / formula-predicted) across every
    completed run so far, replaying estimate_seconds against each run's
    own stored config — same self-correcting idea as the Artifact
    dashboard's live estimate, kept independent since there's no shared
    state between the two."""
    runs = [r for r in load_runs(200) if r["status"] == "completed" and r["elapsed_seconds"]]
    if not runs:
        return 1.0
    ratios = []
    for r in runs:
        req = json.loads(r["config"])
        predicted = estimate_seconds(req, combos_count=r["total_combos"])
        if predicted > 0:
            ratios.append(r["elapsed_seconds"] / predicted)
    return (sum(ratios) / len(ratios)) if ratios else 1.0
