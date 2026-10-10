"""Time-series momentum, weekly: at each Friday close go long an asset whose
past `lookback`-day return is positive and short one whose is negative; hold
the next week; size by the asset's own recent volatility so every asset
risks the same.

Why this isn't run through the fixed stop/target engine: positions are held
for weeks with no stop, and the unit of evidence is a WEEK of holding, not a
discrete trade.

Why the significance test differs from the entry-family sweeps: those flip a
coin per trade (real direction vs the opposite). Here a position is
persistent — the same long can last for months — so coin-flipping each week
independently would treat strongly serially-correlated positions as
independent evidence and overstate significance. The null used instead is a
CIRCULAR SHIFT: slide the whole position series along the return series by a
random lag (at least half a year). That keeps every property of both series
— persistence of positions, volatility clustering — and destroys only the
alignment between "what I held" and "what then happened". p-values use the
+1 correction (never exactly 0).

Everything else follows the shared pipeline: chronological 60/20/20
RESEARCH/VALIDATION/LOCKED split, holdout pass + same-sign rule,
Benjamini-Hochberg across the ledger. Trials go to .cache/tsmom_trials.jsonl,
never the main ledger. A PORTFOLIO trial per lookback (equal-risk across all
assets, one shared shift) is the headline — individual assets are noisy and
the published evidence is for the diversified book.

    python3 tsmom_backtest.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

LEDGER = os.path.join(HERE, ".cache", "tsmom_trials.jsonl")
LOOKBACKS = (30, 60, 90, 180)  # trading days
VOL_WINDOW = 60
COST_BPS = 2.0  # same flat round-trip cost every backtest here uses; stress-tested separately
MIN_WEEKS = 260
N_SHIFTS = 5000
MIN_SHIFT = 26
DETECTOR_VERSION = 1
# Pre-specified data-quality safeguards (added after a first run was poisoned by
# stale early-crypto prices: ARB-USD is flat 36% of its history, its volatility
# estimate hit zero and one week came out at 6,790 expected-vol units). Both are
# standard for vol-targeted books and are not tuned on results:
#   VOL_FLOOR  a 0.5% weekly-vol floor, so no asset is ever levered 1/vol with a
#              near-zero estimate (pegged FX and stale feeds);
#   U_CLIP     a weekly result is capped at +/-5 expected-vol units, so one
#              genuine fat-tail week (the DOGE pump) cannot be the whole average.
VOL_FLOOR = 0.005
U_CLIP = 5.0


def weekly_signal_frame(close, lookback, vol_window=VOL_WINDOW):
    """One row per week (indexed by that week's last trading day): the decision
    made at its close — pos (+1/-1/0) and the volatility used to size it — plus
    wc (the week's closing price) and next_ret (the FOLLOWING week's return,
    NaN for the newest week, whose future has not happened yet). `week` is the
    W-FRI period each row belongs to. Shared by the backtest (weekly_arrays) and
    the live forward test (tsmom_forward.py) so both run the identical signal."""
    d = close.dropna().astype(float).copy()
    d = d[d > 0]
    if getattr(d.index, "tz", None) is not None:
        d.index = d.index.tz_localize(None)
    ret = d.pct_change()
    sig = np.sign(d / d.shift(lookback) - 1)
    vol = ret.rolling(vol_window).std() * np.sqrt(5)
    week = d.index.to_period("W-FRI")
    last = pd.Series(np.arange(len(d))).groupby(week).last().to_numpy()
    wc, s_, v = d.iloc[last], sig.iloc[last], vol.iloc[last]
    out = pd.DataFrame({"pos": s_.to_numpy(), "vol": v.to_numpy(), "wc": wc.to_numpy(),
                        "next_ret": wc.pct_change().shift(-1).to_numpy()}, index=wc.index)
    out["week"] = week[last]
    return out


def weekly_arrays(close, lookback, cost_bps=COST_BPS, vol_window=VOL_WINDOW):
    """-> DataFrame indexed by the week-end the position was DECIDED at, with
    pos (+1/-1, held the following week), u (the following week's return in
    units of expected weekly volatility), cost (in the same units) and r =
    pos*u - cost. Nothing at row t uses information after week t's close."""
    out = weekly_signal_frame(close, lookback, vol_window)[["pos", "vol", "next_ret"]]
    out = out.dropna()
    out = out[(out["pos"] != 0) & np.isfinite(out["vol"]) & np.isfinite(out["next_ret"])]
    out["vol"] = out["vol"].clip(lower=VOL_FLOOR)
    out["u"] = (out["next_ret"] / out["vol"]).clip(-U_CLIP, U_CLIP)
    turnover = out["pos"].diff().abs().fillna(0.0)
    out["cost"] = (cost_bps / 1e4 / 2) * turnover / out["vol"]
    out["r"] = out["pos"] * out["u"] - out["cost"]
    return out[["pos", "u", "cost", "r"]]


def slice_bounds(n):
    return {"research": (0, int(n * 0.6)), "validation": (int(n * 0.6), int(n * 0.8)), "locked": (int(n * 0.8), n)}


def shift_p(pos, u, cost, lo, hi, rng, n_shifts=N_SHIFTS):
    """P(a random circular shift of the positions earns at least the real mean
    over [lo, hi))."""
    n = len(pos)
    obs = float((pos[lo:hi] * u[lo:hi] - cost[lo:hi]).mean())
    ks = rng.integers(MIN_SHIFT, n - MIN_SHIFT, size=n_shifts)
    sims = np.empty(n_shifts)
    for i, k in enumerate(ks):
        sims[i] = (np.roll(pos, k)[lo:hi] * u[lo:hi] - np.roll(cost, k)[lo:hi]).mean()
    return obs, float((1 + (sims >= obs).sum()) / (1 + n_shifts))


def score_series(pos, u, cost, rng):
    n = len(pos)
    b = slice_bounds(n)
    res = {}
    for name, (lo, hi) in b.items():
        res[name] = shift_p(pos, u, cost, lo, hi, rng)
        res[name + "_n"] = hi - lo
        r = pos[lo:hi] * u[lo:hi] - cost[lo:hi]
        res[name + "_std"] = float(r.std(ddof=1)) if len(r) > 1 else None
    return res


def trial_dict(ticker, lookback, res, n_events):
    import experiments as ex
    import strategy_identity
    import data as research_data
    settings = {"lookback_days": lookback, "rebalance": "weekly", "vol_window": VOL_WINDOW}
    family = "tsmom_weekly"
    label = f"{family}_fixed_rr · lookback={lookback}d · weekly · volscaled"
    sid = strategy_identity.compute_strategy_id(
        family=family, settings=settings, target_widen_mult=1.0,
        data_pipeline_version=research_data.DATA_PIPELINE_VERSION, detector_logic_version=DETECTOR_VERSION,
        cost_bps=COST_BPS)
    (m_r, p_r), (m_v, p_v), (m_l, p_l) = res["research"], res["validation"], res["locked"]
    return {
        "logged_at": pd.Timestamp.now("UTC").isoformat(), "ticker": ticker, "tf_label": "1D", "label": label,
        "settings": settings, "n_events": n_events, "strategy_id": sid, "scale_out_enabled": False,
        "target_widen_mult": 1.0, "split_mode": "research_validation_locked_circular_shift",
        "research_fraction": 0.6, "validation_fraction": 0.2, "verdict": "SCORED",
        "n_train": res["research_n"], "n_holdout": res["validation_n"],
        "mean_return_train": m_r, "p_value_train": p_r, "std_return_train": res["research_std"],
        "mean_return_holdout": m_v, "p_value_holdout": p_v, "std_return_holdout": res["validation_std"],
        "holdout_verdict": "PASSED" if (p_v < 0.05 and m_v > 0) else "FAILED",
        "same_sign": bool(m_r > 0 and m_v > 0),
        "locked_n": res["locked_n"], "locked_mean_return": m_l, "locked_p_value": p_l,
        "locked_std_return": res["locked_std"],
    }


def run(tickers, fetch, rng, append):
    """fetch(ticker) -> daily close Series or None. Returns the trials written."""
    frames = {lb: {} for lb in LOOKBACKS}
    trials = []
    for t in tickers:
        close = fetch(t)
        if close is None or len(close) < 500:
            continue
        for lb in LOOKBACKS:
            w = weekly_arrays(close, lb)
            if len(w) < MIN_WEEKS:
                continue
            frames[lb][t] = w
            res = score_series(w["pos"].to_numpy(), w["u"].to_numpy(), w["cost"].to_numpy(), rng)
            tr = trial_dict(t, lb, res, len(w))
            append(tr)
            trials.append(tr)
    for lb in LOOKBACKS:
        if len(frames[lb]) < 5:
            continue
        idx = sorted(set().union(*[f.index for f in frames[lb].values()]))
        P = np.array([f["pos"].reindex(idx).to_numpy() for f in frames[lb].values()])
        U = np.array([f["u"].reindex(idx).to_numpy() for f in frames[lb].values()])
        C = np.array([f["cost"].reindex(idx).fillna(0.0).to_numpy() for f in frames[lb].values()])
        has = ~np.isnan(P) & ~np.isnan(U)
        P, U = np.where(has, P, 0.0), np.where(has, U, 0.0)

        def port(P_, C_):
            n_act = np.maximum(has.sum(axis=0), 1)
            return (P_ * U - C_).sum(axis=0) / n_act

        n = len(idx)
        res = {}
        for name, (lo, hi) in slice_bounds(n).items():
            obs = float(port(P, C)[lo:hi].mean())
            ks = rng.integers(MIN_SHIFT, n - MIN_SHIFT, size=N_SHIFTS // 5)
            sims = np.array([port(np.roll(P, k, axis=1), np.roll(C, k, axis=1))[lo:hi].mean() for k in ks])
            res[name] = (obs, float((1 + (sims >= obs).sum()) / (1 + len(sims))))
            res[name + "_n"] = hi - lo
            pr = port(P, C)[lo:hi]
            res[name + "_std"] = float(pr.std(ddof=1)) if len(pr) > 1 else None
        tr = trial_dict("PORTFOLIO", lb, res, n)
        tr["n_assets"] = len(frames[lb])
        append(tr)
        trials.append(tr)
    return trials


def main():
    import experiments as ex
    import data as research_data
    from run_ema_sweep import universe
    tickers = universe()
    ex._TRIALS_PATH = LEDGER
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    if os.path.exists(LEDGER):
        os.remove(LEDGER)  # deterministic full re-run; the ledger is this script's alone

    def fetch(t):
        try:
            df = research_data.get_yf_ohlcv(t, period="10y", interval="1d")
            return None if df is None or df.empty else df["Close"]
        except Exception:
            return None

    trials = run(tickers, fetch, np.random.default_rng(11), ex._append_trial)
    print(f"{len(trials)} trials -> {LEDGER}")


if __name__ == "__main__":
    main()
