"""Paper forward test of 180-day weekly time-series momentum — the only idea in
the research round that was positive on all three time slices (+0.040 / +0.003
/ +0.051 R per week) without being statistically distinguishable from luck.
Testing more variants on the same history would be fishing; the only fair
evidence left is data that did not exist when the idea was chosen. This
collects it, automatically, and never trades.

Frozen in advance (written once to .cache/tsmom_forward_spec.json; the script
refuses to run if the file is edited):
  * the rule: sign of the past 180-day return, decided at each Friday close,
    held the next week, sized by 60-day volatility (floor 0.5%, results capped
    at +/-5 volatility units) — exactly tsmom_backtest.weekly_arrays, the same
    code the backtest ran;
  * the universe: the assets that had enough history in the backtest;
  * the first decision: the first Friday AFTER this file is created, so no
    return it is scored on had happened when it was created;
  * the verdict: judged once, at 104 realized weeks. It counts as confirmed only
    if the book's mean weekly R is positive, beats a circular-shift null at
    p < 0.05, and is still positive at 8 bps costs. Before that, interim numbers
    are shown and labelled as such — no peeking-based conclusions.

Weekly via launchd (Saturday morning). Idempotent: rerunning records nothing
twice. Append-only ledger: .cache/tsmom_forward_ledger.jsonl.

    python3 tsmom_forward.py            # record new decisions, realize finished weeks, print status
"""
import hashlib
import json
import os
import sys
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tsmom_backtest as tm  # noqa: E402

CACHE = os.path.join(HERE, ".cache")
SPEC_PATH = os.path.join(CACHE, "tsmom_forward_spec.json")
LEDGER_PATH = os.path.join(CACHE, "tsmom_forward_ledger.jsonl")
STATUS_PATH = os.path.join(CACHE, "tsmom_forward_status.json")
JUDGE_AT_WEEKS = 104
STRESS_BPS = 8.0
LOOKBACK = 180


def _spec_hash(spec):
    core = {k: v for k, v in spec.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:16]


def first_friday_after(d):
    """The first Friday strictly after date d."""
    days = (4 - d.weekday()) % 7
    return d + pd.Timedelta(days=days or 7)


def create_spec(universe, today, path=SPEC_PATH):
    spec = {
        "created": today.isoformat(), "lookback_days": LOOKBACK, "vol_window": tm.VOL_WINDOW,
        "vol_floor": tm.VOL_FLOOR, "u_clip": tm.U_CLIP, "cost_bps": tm.COST_BPS,
        "first_decision_friday": first_friday_after(pd.Timestamp(today)).date().isoformat(),
        "judge_at_weeks": JUDGE_AT_WEEKS, "stress_bps": STRESS_BPS, "universe": sorted(universe),
    }
    spec["sha256"] = _spec_hash(spec)
    with open(path, "w") as f:
        json.dump(spec, f, indent=2)
    return spec


def load_spec(path=SPEC_PATH):
    with open(path) as f:
        spec = json.load(f)
    if spec.get("sha256") != _spec_hash(spec):
        raise RuntimeError("tsmom_forward_spec.json was edited after it was frozen — refusing to run. "
                           "Restore it, or start a new, clearly separate test.")
    return spec


def _read_ledger(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _bucket_complete(frame, i, today):
    """Is week i's data final? Yes if a later week already exists, or its Friday has passed."""
    if i < len(frame) - 1:
        return True
    friday = frame["week"].iloc[i].end_time.date()
    # newest bucket: its Friday must be over (a same-day bar can still be forming) and the data must
    # actually reach the end of the week, not be a stale fetch
    return today > friday and frame.index[i].date() >= (friday - pd.Timedelta(days=2))


def advance(spec, closes, today, ledger_path=LEDGER_PATH):
    """Record new decisions and realize finished weeks. closes: {asset: daily close Series}.
    Returns (n_new_decisions, n_new_realized)."""
    rows = _read_ledger(ledger_path)
    decided = {(r["week"], r["asset"]) for r in rows if r["type"] == "decision"}
    realized = {(r["week"], r["asset"]) for r in rows if r["type"] == "realized"}
    first = spec["first_decision_friday"]
    new, n_dec, n_real = [], 0, 0
    for asset in spec["universe"]:
        if asset not in closes:
            continue
        f = tm.weekly_signal_frame(closes[asset], spec["lookback_days"], spec["vol_window"])
        if len(f) < 3:
            continue
        pos = f["pos"].to_numpy()
        for i in range(len(f)):
            wk = str(f["week"].iloc[i])
            friday = f["week"].iloc[i].end_time.date().isoformat()
            if friday < first or not _bucket_complete(f, i, today) or pos[i] == 0 or not np.isfinite(f["vol"].iloc[i]):
                continue
            if (wk, asset) not in decided:
                new.append({"type": "decision", "week": wk, "friday": friday, "asset": asset, "pos": float(pos[i]),
                            "vol": float(max(f["vol"].iloc[i], spec["vol_floor"])), "close": float(f["wc"].iloc[i])})
                n_dec += 1
            if (wk, asset) not in realized and i + 1 < len(f) and _bucket_complete(f, i + 1, today) \
                    and np.isfinite(f["next_ret"].iloc[i]):
                vol = float(max(f["vol"].iloc[i], spec["vol_floor"]))
                u = float(np.clip(f["next_ret"].iloc[i] / vol, -spec["u_clip"], spec["u_clip"]))
                prev = pos[i - 1] if i > 0 and np.isfinite(pos[i - 1]) and pos[i - 1] != 0 else pos[i]  # backtest's turnover convention
                turn = abs(pos[i] - prev)
                new.append({"type": "realized", "week": wk, "friday": friday, "asset": asset, "pos": float(pos[i]),
                            "u": u, "turnover": float(turn), "vol": vol, "next_ret": float(f["next_ret"].iloc[i])})
                n_real += 1
    if new:
        os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
        with open(ledger_path, "a") as fh:
            for r in new:
                fh.write(json.dumps(r) + "\n")
    return n_dec, n_real


def book_series(ledger_path, cost_bps):
    """Weekly equal-risk book: mean over assets of pos*u - cost, indexed by decision Friday."""
    rows = [r for r in _read_ledger(ledger_path) if r["type"] == "realized"]
    if not rows:
        return pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df = pd.DataFrame(rows)
    df["cost"] = (cost_bps / 1e4 / 2) * df["turnover"] / df["vol"]
    df["r"] = df["pos"] * df["u"] - df["cost"]
    book = df.groupby("friday")["r"].mean().sort_index()
    P = df.pivot(index="friday", columns="asset", values="pos").sort_index().fillna(0.0)
    U = df.pivot(index="friday", columns="asset", values="u").sort_index().fillna(0.0)
    C = df.pivot(index="friday", columns="asset", values="cost").sort_index().fillna(0.0)
    return book, P, U, C


def shift_p_forward(P, U, C, rng, n_shifts=5000, min_shift=13):
    n = len(P)
    if n < 2 * min_shift + 2:
        return None
    Pa, Ua, Ca = P.to_numpy().T, U.to_numpy().T, C.to_numpy().T  # assets x weeks
    n_act = np.maximum((Ua != 0).sum(axis=0), 1)
    obs = float(((Pa * Ua - Ca).sum(axis=0) / n_act).mean())
    sims = []
    for k in rng.integers(min_shift, n - min_shift, size=n_shifts):
        sims.append(((np.roll(Pa, k, axis=1) * Ua - np.roll(Ca, k, axis=1)).sum(axis=0) / n_act).mean())
    return float((1 + (np.array(sims) >= obs).sum()) / (1 + n_shifts))


def status(spec, ledger_path=LEDGER_PATH, rng=None):
    rng = rng or np.random.default_rng(3)
    book, P, U, C = book_series(ledger_path, spec["cost_bps"])
    hard, _, _, _ = book_series(ledger_path, spec["stress_bps"])
    n = len(book)
    out = {"spec_sha256": spec["sha256"], "realized_weeks": n, "judge_at_weeks": spec["judge_at_weeks"],
           "first_decision_friday": spec["first_decision_friday"],
           "decisions_recorded": sum(1 for r in _read_ledger(ledger_path) if r["type"] == "decision")}
    if n:
        out.update({"mean_r_per_week": round(float(book.mean()), 4),
                    "info_ratio_annualised": round(float(book.mean() / book.std(ddof=1) * np.sqrt(52)), 2) if n > 1 else None,
                    "pct_weeks_positive": round(float((book > 0).mean() * 100), 1),
                    "mean_r_per_week_at_stress_cost": round(float(hard.mean()), 4)})
    if n < spec["judge_at_weeks"]:
        out["verdict"] = f"collecting — interim only, judged at {spec['judge_at_weeks']} realized weeks ({n} so far)"
    else:
        p = shift_p_forward(P, U, C, rng)
        out["p_value_circular_shift"] = None if p is None else round(p, 4)
        ok = (out["mean_r_per_week"] > 0 and p is not None and p < 0.05 and out["mean_r_per_week_at_stress_cost"] > 0)
        out["verdict"] = "CONFIRMED on out-of-sample data" if ok else "NOT confirmed — retire the idea"
    return out


def main():
    import data as research_data
    from universe import universe
    today = datetime.now(timezone.utc).date()
    if not os.path.exists(SPEC_PATH):
        eligible = []
        for t in universe():
            try:
                df = research_data.get_yf_ohlcv(t, period="10y", interval="1d")
            except Exception:
                continue
            if df is not None and not df.empty and len(df) >= 500 and len(tm.weekly_arrays(df["Close"], LOOKBACK)) >= tm.MIN_WEEKS:
                eligible.append(t)
        spec = create_spec(eligible, today)
        print(f"froze spec: {len(eligible)} assets, first decision Friday {spec['first_decision_friday']} (sha {spec['sha256']})")
    spec = load_spec()
    closes = {}
    for t in spec["universe"]:
        try:
            df = research_data.get_yf_ohlcv(t, period="10y", interval="1d")
            if df is not None and not df.empty:
                closes[t] = df["Close"]
        except Exception:
            pass
    n_dec, n_real = advance(spec, closes, today)
    st = status(spec)
    with open(STATUS_PATH, "w") as f:
        json.dump({**st, "updated": datetime.now(timezone.utc).isoformat(), "new_decisions": n_dec, "new_realized": n_real}, f, indent=2)
    print(json.dumps({**st, "new_decisions": n_dec, "new_realized": n_real}, indent=2))


if __name__ == "__main__":
    main()
