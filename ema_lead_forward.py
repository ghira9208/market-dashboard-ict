"""Forward test of one lead from studies/streak_polish_study.py: on the validated clean_* combos, entries made while
the EMAs are NOT yet lined up with the trade earn more R than entries made after they line up.

Found on history (4,329 trades, 78 combos): not-aligned +1.71R vs aligned +1.09R, same sign in all three time
slices. It was a post-hoc flip of the rule that was actually tested, so the only fair evidence left is data
that did not exist when the idea was chosen. This collects it automatically. It never trades and changes
nothing live.

Frozen in advance (.cache/ema_lead_forward_spec.json; refuses to run if the file is edited):
  * the feature: at the confirm bar, EMA20 vs EMA50 ordered in the trade's direction = "aligned"; the lead
    says NOT aligned does better. Closes up to and including the confirm bar only;
  * the trades: every trade the frozen list of validated combos (ticker, timeframe, strategy, settings,
    target widening) would have taken, simulated by the same code as the backtest, with entry strictly
    AFTER the freeze moment and counted only once its whole holding window has data (a rule on time, not on
    the outcome, so it cannot favour winners);
  * the verdict, once, when at least 400 forward trades exist and each group has at least 100: CONFIRMED only
    if not-aligned beats aligned on mean R, the within-combo permutation p < 0.01, the 95% interval of the
    difference (resampling whole calendar days) lies above zero, and the difference is still positive at
    6 bps cost. Before that, numbers are interim and labelled so.

Daily via launchd. Idempotent. Append-only ledger: .cache/ema_lead_forward_ledger.jsonl.

    python3 ema_lead_forward.py
"""
import hashlib
import json
import os
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CACHE = os.path.join(HERE, ".cache")
SPEC_PATH = os.path.join(CACHE, "ema_lead_forward_spec.json")
LEDGER_PATH = os.path.join(CACHE, "ema_lead_forward_ledger.jsonl")
STATUS_PATH = os.path.join(CACHE, "ema_lead_forward_status.json")
MIN_TRADES, MIN_GROUP, ALPHA = 400, 100, 0.01
COST_BPS, STRESS_BPS = 2.0, 6.0


def _spec_hash(spec):
    core = {k: v for k, v in spec.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()[:16]


def create_spec(combos, now, path=SPEC_PATH):
    spec = {"created": now.isoformat(), "ema_fast": 20, "ema_slow": 50, "cost_bps": COST_BPS,
            "stress_bps": STRESS_BPS, "min_trades": MIN_TRADES, "min_group": MIN_GROUP, "alpha": ALPHA,
            "lead": "entries with EMA20/EMA50 NOT ordered in the trade direction earn more R", "combos": combos}
    spec["sha256"] = _spec_hash(spec)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(spec, f, indent=2, default=str)
    return spec


def load_spec(path=SPEC_PATH):
    with open(path) as f:
        spec = json.load(f)
    if spec.get("sha256") != _spec_hash(spec):
        raise RuntimeError("ema_lead_forward_spec.json was edited after it was frozen — refusing to run. "
                           "Restore it, or start a new, clearly separate test.")
    return spec


def _read_ledger(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def frozen_combos():
    """The validated combos right now, each with the exact settings its backtest used."""
    import experiments as ex
    import observatory as obs
    import ticker_behavior as tb
    from overlay_tools import FIND
    lb, trials, out = tb.validated_leaderboard(), ex.load_experiment_trials(), []
    for _, c in lb.iterrows():
        m = trials[(trials.ticker == c.ticker) & (trials.tf_label == c.tf_label) & (trials.label == c.best_label)]
        if m.empty or c.strategy not in FIND:
            continue
        s = obs.effective_settings({k: v for k, v in m.iloc[0]["settings"].items() if not k.startswith("_")},
                                   c.get("best_target_widen_mult"))
        out.append({"ticker": c.ticker, "tf": c.tf_label, "strategy": c.strategy,
                    "settings": {k: (v.item() if hasattr(v, "item") else v) for k, v in s.items()}})
    return sorted(out, key=lambda c: (c["ticker"], c["tf"], c["strategy"]))


def combo_trades(df, combo, created, cost_bps):
    """Trades one combo would have taken, entered after `created` and fully resolved, with the EMA feature.
    Returns a list of dicts (entry UTC iso, r at cost_bps, aligned)."""
    import experiments as ex
    from overlay_tools import FIND, _utc, ema_state
    s = combo["settings"]
    touches = FIND[combo["strategy"]](df, min_streak=s["min_streak"], retr_window_bars=s["retr_window_bars"])
    ev = ex._simulate_fixed_rr(df, touches, s["rr_multiple"], s["atr_mult_stop"], s["max_bars"], cost_bps)
    if ev.empty:
        return []
    _, fast, slow, _ = ema_state(df)
    pos_of = {t: i for i, t in enumerate(df.index)}
    by_entry = {df.index[t["confirm_pos"] + 1]: t for t in touches if t["confirm_pos"] + 1 < len(df)}
    n, rows = len(df), []
    for _, e in ev.iterrows():
        t, entry_pos = by_entry.get(e["entry_time"]), pos_of.get(e["entry_time"])
        if t is None or entry_pos is None or _utc(e["entry_time"]) <= created:
            continue
        if entry_pos + s["max_bars"] > n - 1:  # holding window not fully observed yet (decided by time only)
            continue
        cp = t["confirm_pos"]
        sign = 1 if t["direction"] == "bullish" else -1
        rows.append({"entry": _utc(e["entry_time"]).isoformat(), "r": float(e["r_real"]),
                     "aligned": bool(sign * (fast[cp] - slow[cp]) > 0)})
    return rows


def advance(spec, fetch, ledger_path=LEDGER_PATH):
    """Record new forward trades. fetch(ticker, tf) -> OHLC DataFrame. Returns the number appended."""
    created = pd.Timestamp(spec["created"])
    created = created.tz_localize("UTC") if created.tz is None else created.tz_convert("UTC")
    seen = {(r["combo"], r["entry"]) for r in _read_ledger(ledger_path)}
    new = []
    for c in spec["combos"]:
        cid = f'{c["ticker"]}|{c["tf"]}|{c["strategy"]}'
        try:
            df = fetch(c["ticker"], c["tf"])
            base = combo_trades(df, c, created, spec["cost_bps"])
            hard = {r["entry"]: r["r"] for r in combo_trades(df, c, created, spec["stress_bps"])}
        except Exception as exc:
            print("skip", cid, repr(exc)[:80])
            continue
        for r in base:
            if (cid, r["entry"]) not in seen:
                new.append({"combo": cid, "strategy": c["strategy"], "tf": c["tf"], "entry": r["entry"], "r": r["r"],
                            "r_stress": hard.get(r["entry"], r["r"]), "aligned": r["aligned"]})
    if new:
        os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
        with open(ledger_path, "a") as fh:
            for r in new:
                fh.write(json.dumps(r) + "\n")
    return len(new)


def status(spec, ledger_path=LEDGER_PATH, rng=None):
    from overlay_tools import day_cluster_ci, strat_diff_p
    rng = rng or np.random.default_rng(5)
    rows = _read_ledger(ledger_path)
    out = {"spec_sha256": spec["sha256"], "frozen_at": spec["created"], "forward_trades": len(rows)}
    if not rows:
        out["verdict"] = "collecting — no forward trades yet"
        return out
    df = pd.DataFrame(rows)
    df["entry"] = pd.to_datetime(df["entry"], utc=True)
    df["naligned"] = ~df["aligned"].astype(bool)
    n_na, n_al = int(df["naligned"].sum()), int((~df["naligned"]).sum())
    out.update({"not_aligned_n": n_na, "aligned_n": n_al})
    if n_na and n_al:
        out["mean_r_not_aligned"] = round(float(df[df.naligned]["r"].mean()), 3)
        out["mean_r_aligned"] = round(float(df[~df.naligned]["r"].mean()), 3)
        out["diff_not_aligned_minus_aligned"] = round(out["mean_r_not_aligned"] - out["mean_r_aligned"], 3)
        out["diff_at_stress_cost"] = round(float(df[df.naligned]["r_stress"].mean() - df[~df.naligned]["r_stress"].mean()), 3)
    if len(df) < spec["min_trades"] or min(n_na, n_al) < spec["min_group"]:
        out["verdict"] = (f"collecting — interim only, judged at {spec['min_trades']} trades with "
                          f"{spec['min_group']}+ in each group ({len(df)} total, {n_na} not aligned, {n_al} aligned)")
        return out
    d, p = strat_diff_p(df, "naligned", rng, b=5000)
    lo, hi = day_cluster_ci(df, "naligned", rng, b=3000)
    out.update({"p_within_combo": round(p, 4), "day_cluster_CI95": [lo, hi]})
    ok = d > 0 and p < spec["alpha"] and lo > 0 and out["diff_at_stress_cost"] > 0
    out["verdict"] = "CONFIRMED on out-of-sample data" if ok else "NOT confirmed — drop the idea"
    return out


def _notify_once(st):
    verdict = st.get("verdict", "")
    if not (verdict.startswith("CONFIRMED") or verdict.startswith("NOT confirmed")):
        return
    try:
        sys.path.insert(0, "/Users/apple/dev/edge-pipeline")
        from notify import notify
        notify("EMA-not-aligned forward test", f"{verdict}. Not aligned {st.get('mean_r_not_aligned')}R vs aligned "
               f"{st.get('mean_r_aligned')}R over {st['forward_trades']} forward trades.", severity="info",
               dedup_key=("ema_lead_forward", verdict[:9]))
    except Exception as exc:
        print("notify failed:", repr(exc)[:80])


def main():
    import observatory as obs
    now = datetime.now(timezone.utc)
    if not os.path.exists(SPEC_PATH):
        spec = create_spec(frozen_combos(), now)
        print(f"froze spec: {len(spec['combos'])} combos at {spec['created']} (sha {spec['sha256']})")
    spec = load_spec()
    n_new = advance(spec, obs._fetch_df)
    st = status(spec)
    with open(STATUS_PATH, "w") as f:
        json.dump({**st, "updated": now.isoformat(), "new_trades": n_new}, f, indent=2)
    print(json.dumps({**st, "new_trades": n_new}, indent=2))
    _notify_once(st)


if __name__ == "__main__":
    main()
