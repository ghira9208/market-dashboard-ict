"""Does "wait for commitment" improve the clean_* strategies?

Idea (user, 2026-10-21): enter only once the EMAs have settled in the trade's direction and the
trend has lasted slightly longer than the minimum the detectors search for.

Rules are FIXED IN ADVANCE (nothing is tuned on this data, so every slice is out-of-sample):
  L1  streak length >= min_streak + 1      (all validated combos use min_streak = 3)
  L2  streak length >= min_streak + 2
  E1  EMA20 vs EMA50 ordered in the trade's direction at the confirm bar
  E2  E1 and price within 1 ATR of EMA20 at the confirm bar   ("normalised": not stretched away)
  W3  delay entry 3 bars, only if E1 still holds then and the zone's far edge has not been closed through
Everything is read from bars up to and including the confirm bar (W3: up to the delayed bar), so no
lookahead. The first four are OVERLAYS (they only decide which existing trades to skip, like
overlay_tests.py); W3 is a different entry and is re-simulated.

Judged on pooled validated-combo trades, 60/20/20 time slices per combo, permutation p-value from
shuffling the feature label WITHIN each combo (a better combo cannot fake an effect), a calendar-day
cluster bootstrap on the pooled difference, and a 6 bps cost stress. Bonferroni over the 5 rules.

    python3 streak_polish_study.py
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd

import experiments as ex
import observatory as obs
import ticker_behavior as tb
from overlay_tests import split_labels

B = 5000
RULES = ["L1", "L2", "E1", "E2", "W3"]
DELAY = 3
COVER = [0, 0]  # trades with no streak length found / found


def _utc(x):
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def streak_lengths(df, strategy, min_streak, retr_window_bars):
    """confirm_pos -> length of the qualifying streak that produced the touch (mirrors the pairing the
    production detectors use; asserted against their own touches by the caller)."""
    merged, qualifying = ex._scan_clean_zone_streaks(df, min_streak, retr_window_bars)
    out = {}
    if strategy == "clean_retracement_resumption_fixed_rr":
        for a, b in zip(qualifying[:-1], qualifying[1:]):
            if a["direction"] != b["direction"]:
                out[ex._confirmed_zone_pos(b["zones"][-1])] = len(b["zones"])
    else:
        for s in qualifying:
            last_pos, d = s["last_pos"], s["direction"]
            z = next((z for z in merged if last_pos < z["_pos"] <= last_pos + retr_window_bars and z["type"] != d), None)
            if z is not None:
                out[ex._confirmed_zone_pos(z)] = len(s["zones"])
    return out


def ema_state(df):
    close = df["Close"].to_numpy(dtype=float)
    f = pd.Series(close).ewm(span=20, adjust=False).mean().to_numpy()
    s = pd.Series(close).ewm(span=50, adjust=False).mean().to_numpy()
    atr = ex.atr_series(df).to_numpy()
    return close, f, s, atr


FIND = {
    "clean_expansion_fixed_rr": ex._clean_expansion_touches,
    "clean_expansion_liquidity_fixed_rr": ex._clean_expansion_liquidity_touches,
    "clean_retracement_resumption_fixed_rr": ex._clean_retracement_resumption_touches,
}


def combo_rows(c, s, df, cost_bps):
    """One row per baseline trade with its R, features, and (where W3 applies) the delayed-entry R."""
    kw = dict(min_streak=s["min_streak"], retr_window_bars=s["retr_window_bars"])
    touches = FIND[c.strategy](df, **kw)
    lens = streak_lengths(df, c.strategy, **kw)
    ev = ex._simulate_fixed_rr(df, touches, s["rr_multiple"], s["atr_mult_stop"], s["max_bars"], cost_bps)
    if ev.empty:
        return []
    close, f, sl, atr = ema_state(df)
    pos_of = {t: i for i, t in enumerate(df.index)}
    by_entry = {df.index[t["confirm_pos"] + 1]: t for t in touches if t["confirm_pos"] + 1 < len(df)}
    # W3: delayed touches that still qualify, simulated on their own
    delayed = []
    for t in touches:
        p = t["confirm_pos"] + DELAY
        if p >= len(df) - 1:
            continue
        sign = 1 if t["direction"] == "bullish" else -1
        edge = t["zone_bottom"] if sign > 0 else t["zone_top"]
        broke = (close[t["confirm_pos"] + 1:p + 1] < edge).any() if sign > 0 else (close[t["confirm_pos"] + 1:p + 1] > edge).any()
        if sign * (f[p] - sl[p]) > 0 and not broke:
            delayed.append({**t, "confirm_pos": p})
    ev_w = ex._simulate_fixed_rr(df, delayed, s["rr_multiple"], s["atr_mult_stop"], s["max_bars"], cost_bps) if delayed else None
    rows = []
    for _, e in ev.sort_values("entry_time").iterrows():
        t = by_entry.get(e["entry_time"])
        if t is None:
            continue
        cp, sign = t["confirm_pos"], (1 if t["direction"] == "bullish" else -1)
        n = lens.get(cp)
        COVER[0 if n is None else 1] += 1
        e1 = bool(sign * (f[cp] - sl[cp]) > 0)
        a = atr[cp]
        e2 = bool(e1 and a > 0 and abs(close[cp] - f[cp]) <= a)
        rows.append({"combo": f"{c.ticker}|{c.tf_label}|{c.strategy}", "strategy": c.strategy, "tf": c.tf_label,
                     "ticker": c.ticker, "entry": _utc(e["entry_time"]), "r": float(e["r_real"]),
                     "L1": bool(n is not None and n >= s["min_streak"] + 1),
                     "L2": bool(n is not None and n >= s["min_streak"] + 2), "E1": e1, "E2": e2})
    # attach slice labels (per combo, by time)
    rows.sort(key=lambda r: r["entry"])
    for r, lab in zip(rows, split_labels(len(rows))):
        r["slice"] = lab
    wrows = []
    if ev_w is not None and not ev_w.empty:
        ws = sorted(ev_w["entry_time"])
        cut = (ws[int(len(ws) * 0.6)] if len(ws) > 5 else None, ws[int(len(ws) * 0.8)] if len(ws) > 5 else None)
        for _, e in ev_w.iterrows():
            et = _utc(e["entry_time"])
            lab = "research" if cut[0] is None or et < _utc(cut[0]) else ("validation" if et < _utc(cut[1]) else "locked")
            wrows.append({"combo": f"{c.ticker}|{c.tf_label}|{c.strategy}", "strategy": c.strategy,
                          "entry": et, "r": float(e["r_real"]), "slice": lab})
    return rows, wrows


def load(cost_bps):
    lb, trials = tb.validated_leaderboard(), ex.load_experiment_trials()
    base, wait = [], []
    for _, c in lb.iterrows():
        m = trials[(trials.ticker == c.ticker) & (trials.tf_label == c.tf_label) & (trials.label == c.best_label)]
        if m.empty or c.strategy not in FIND:
            continue
        s = obs.effective_settings({k: v for k, v in m.iloc[0]["settings"].items() if not k.startswith("_")},
                                   c.get("best_target_widen_mult"))
        try:
            df = obs._fetch_df(c.ticker, c.tf_label)
            r = combo_rows(c, s, df, cost_bps)
        except Exception as exc:  # one bad ticker must not sink the study
            print("skip", c.ticker, c.tf_label, c.strategy, repr(exc)[:80])
            continue
        if r:
            base += r[0]
            wait += r[1]
    return pd.DataFrame(base), pd.DataFrame(wait)


def strat_diff_p(df, col, rng, b=2000):
    """mean R (feature true) - mean R (feature false); p from shuffling the label within each combo."""
    groups = [(g["r"].to_numpy(), g[col].to_numpy().astype(bool)) for _, g in df.groupby("combo")]
    groups = [(r, m) for r, m in groups if 0 < m.sum() < len(m)]
    if not groups:
        return float("nan"), float("nan")

    def diff(masks):
        t = np.concatenate([r[m] for (r, _), m in zip(groups, masks)])
        f = np.concatenate([r[~m] for (r, _), m in zip(groups, masks)])
        return t.mean() - f.mean()

    obs_d = diff([m for _, m in groups])
    sims = np.array([diff([rng.permutation(m) for _, m in groups]) for _ in range(b)])
    return float(obs_d), float((1 + (sims >= obs_d).sum()) / (1 + b))


def day_cluster_ci(df, col, rng, b=2000):
    """95% CI of the pooled true-minus-false difference, resampling whole calendar days."""
    d = df.assign(day=df["entry"].dt.date)
    days = d["day"].unique()
    g = {k: v for k, v in d.groupby("day")}
    out = []
    for _ in range(b):
        s = pd.concat([g[k] for k in rng.choice(days, len(days))])
        t, f = s[s[col]]["r"], s[~s[col]]["r"]
        if len(t) and len(f):
            out.append(t.mean() - f.mean())
    return [round(float(np.percentile(out, 2.5)), 3), round(float(np.percentile(out, 97.5)), 3)]


def judge_overlay(df, col, rng):
    res = {"share_true": round(float(df[col].mean()), 3)}
    for sl in ("research", "validation", "locked", "all"):
        s = df if sl == "all" else df[df["slice"] == sl]
        if len(s) < 30 or s[col].nunique() < 2:
            res[sl] = {"n": int(len(s))}
            continue
        d, p = strat_diff_p(s, col, rng)
        res[sl] = {"n": int(len(s)), "mean_true": round(float(s[s[col]]["r"].mean()), 3),
                   "mean_false": round(float(s[~s[col]]["r"].mean()), 3), "diff": round(d, 3),
                   "p": round(p, 4), "total_R_if_only_true": round(float(s[s[col]]["r"].sum()), 1),
                   "total_R_all": round(float(s["r"].sum()), 1)}
    res["day_cluster_CI_all"] = day_cluster_ci(df, col, rng)
    return res


def judge_wait(base, wait, rng, b=2000):
    out = {}
    for sl in ("research", "validation", "locked", "all"):
        bs = base if sl == "all" else base[base["slice"] == sl]
        ws = wait if sl == "all" else wait[wait["slice"] == sl]
        if len(bs) < 30 or len(ws) < 30:
            out[sl] = {"n_base": int(len(bs)), "n_wait": int(len(ws))}
            continue
        d = ws["r"].mean() - bs["r"].mean()
        # day-cluster bootstrap of the difference of per-trade means
        days = sorted(set(bs["entry"].dt.date) | set(ws["entry"].dt.date))
        gb, gw = dict(list(bs.groupby(bs["entry"].dt.date))), dict(list(ws.groupby(ws["entry"].dt.date)))
        sims = []
        for _ in range(b):
            pick = rng.choice(len(days), len(days))
            sb = pd.concat([gb[days[i]] for i in pick if days[i] in gb])
            sw = pd.concat([gw[days[i]] for i in pick if days[i] in gw])
            sims.append(sw["r"].mean() - sb["r"].mean())
        out[sl] = {"n_base": int(len(bs)), "n_wait": int(len(ws)), "mean_base": round(float(bs["r"].mean()), 3),
                   "mean_wait": round(float(ws["r"].mean()), 3), "diff": round(float(d), 3),
                   "CI95": [round(float(np.percentile(sims, 2.5)), 3), round(float(np.percentile(sims, 97.5)), 3)],
                   "total_R_base": round(float(bs["r"].sum()), 1), "total_R_wait": round(float(ws["r"].sum()), 1)}
    return out


def main():
    rng = np.random.default_rng(11)
    base, wait = load(2.0)
    print("streak length found for", COVER[1], "of", sum(COVER), "trades")
    print(f"{len(base)} baseline trades, {base['combo'].nunique()} combos; W3 trades {len(wait)}; "
          f"slices {base['slice'].value_counts().to_dict()}")
    print("streak-length mix:", base.groupby("strategy")[["L1", "L2", "E1", "E2"]].mean().round(2).to_dict("index"))
    results = {"overlays": {}, "wait": {}, "by_strategy": {}, "cost_stress_6bps": {}}
    for col in ("L1", "L2", "E1", "E2"):
        results["overlays"][col] = judge_overlay(base, col, rng)
        for strat in base["strategy"].unique():
            sub = base[base["strategy"] == strat]
            if sub[col].nunique() == 2:
                d, p = strat_diff_p(sub, col, rng, b=1000)
                results["by_strategy"].setdefault(col, {})[strat] = {
                    "n": int(len(sub)), "share_true": round(float(sub[col].mean()), 3),
                    "mean_true": round(float(sub[sub[col]]["r"].mean()), 3),
                    "mean_false": round(float(sub[~sub[col]]["r"].mean()), 3), "diff": round(d, 3), "p": round(p, 4)}
    results["wait"]["W3"] = judge_wait(base, wait, rng)
    base6, wait6 = load(6.0)
    for col in ("L1", "L2", "E1", "E2"):
        s = base6
        results["cost_stress_6bps"][col] = {
            "mean_true": round(float(s[s[col]]["r"].mean()), 3), "mean_false": round(float(s[~s[col]]["r"].mean()), 3),
            "mean_all": round(float(s["r"].mean()), 3)}
    results["cost_stress_6bps"]["W3"] = {"mean_wait": round(float(wait6["r"].mean()), 3),
                                          "mean_base": round(float(base6["r"].mean()), 3)}
    print(json.dumps(results, indent=1, default=str))
    with open(os.path.join(HERE, ".cache", "streak_polish.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
