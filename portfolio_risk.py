"""Risk statistics and Monte Carlo for sequences of trade results in R.

Everything here works on plain arrays of R-multiples (a trade's profit divided
by what it risked), so a number like "10R drawdown" has one fixed meaning: ten
times whatever you risk per trade. At 0.5% risk per trade, 10R is a 5% drawdown.

  max_drawdown_r        worst peak-to-trough fall of the cumulative-R curve
  profit_factor         gross wins / gross losses (1.0 = break-even)
  expected_max_dd       bootstrap distribution of the worst drawdown over the
                        next `horizon` trades — comparable across strategies
                        with different histories, unlike a historical maximum
  simulate_equity       Monte Carlo of a COMPOUNDING account risking a fixed
                        fraction per R, by resampling single trades (iid) or
                        whole calendar days (block), which keeps same-day
                        clustering of correlated trades together

The iid bootstrap treats every trade as independent. When assets move together
the real losses arrive in bunches; the day-block bootstrap keeps each day's
trades together and so shows the drawdown that independence hides.
"""
import numpy as np
import pandas as pd


def max_drawdown_r(r):
    cum = np.cumsum(np.asarray(r, dtype=float))
    if len(cum) == 0:
        return 0.0
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    return float((peak - cum).max())


def profit_factor(r):
    r = np.asarray(r, dtype=float)
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / losses)


def longest_losing_streak(r):
    best = cur = 0
    for v in np.asarray(r, dtype=float):
        cur = cur + 1 if v <= 0 else 0
        best = max(best, cur)
    return best


def _dd_of_paths(paths):
    """Max drawdown (R) of each row of a (n_sims, n_trades) matrix of trade results."""
    cum = np.cumsum(paths, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.zeros((cum.shape[0], 1)), cum], axis=1), axis=1)[:, 1:]
    return (peak - cum).max(axis=1)


def expected_max_dd(r, horizon=100, n_sims=4000, rng=None):
    """-> {median, p95} max drawdown in R over `horizon` trades drawn at random
    (with replacement) from this strategy's own results, plus P(down after horizon)."""
    rng = rng or np.random.default_rng(0)
    r = np.asarray(r, dtype=float)
    draws = r[rng.integers(0, len(r), size=(n_sims, horizon))]
    dd = _dd_of_paths(draws)
    return {"dd_median": float(np.median(dd)), "dd_p95": float(np.percentile(dd, 95)),
            "p_down": float((draws.sum(axis=1) < 0).mean())}


def combo_risk_table(trades, horizon=100, rng=None):
    """One row per combo from a trades frame with columns combo, r (chronological within combo)."""
    rng = rng or np.random.default_rng(0)
    rows = []
    for combo, g in trades.groupby("combo"):
        r = g["r"].to_numpy()
        if len(r) < 10:
            continue
        e = expected_max_dd(r, horizon, rng=rng)
        rows.append({"combo": combo, "n": len(r), "mean_r": r.mean(), "win_pct": (r > 0).mean() * 100,
                     "profit_factor": profit_factor(r), "hist_max_dd_r": max_drawdown_r(r),
                     "dd_median_r": e["dd_median"], "dd_p95_r": e["dd_p95"], "p_down": e["p_down"],
                     "edge_over_dd": (r.mean() * horizon) / max(e["dd_p95"], 1e-9),
                     "loss_streak": longest_losing_streak(r)})
    return pd.DataFrame(rows)


def simulate_equity(r, risk_frac, n_trades, n_sims=2000, rng=None, day_ids=None):
    """Compounding equity paths, starting at 1.0, each trade changing equity by
    (1 + risk_frac * R). If `day_ids` (one label per trade) is given, resamples
    whole days of trades until n_trades are collected (block bootstrap) instead
    of single trades (iid). Returns an (n_sims, n_trades + 1) matrix."""
    rng = rng or np.random.default_rng(0)
    r = np.asarray(r, dtype=float)
    if day_ids is None:
        draws = r[rng.integers(0, len(r), size=(n_sims, n_trades))]
    else:
        codes, uniques = pd.factorize(pd.Series(day_ids))
        blocks = [r[codes == k] for k in range(len(uniques))]
        draws = np.empty((n_sims, n_trades))
        for s in range(n_sims):
            out, have = [], 0
            while have < n_trades:
                b = blocks[rng.integers(0, len(blocks))]
                out.append(b)
                have += len(b)
            draws[s] = np.concatenate(out)[:n_trades]
    growth = np.maximum(1.0 + risk_frac * draws, 0.0)  # a loss can wipe equity to zero, never below
    return np.concatenate([np.ones((n_sims, 1)), np.cumprod(growth, axis=1)], axis=1)


def equity_summary(paths):
    final = paths[:, -1]
    peak = np.maximum.accumulate(paths, axis=1)
    dd = (1 - paths / peak).max(axis=1)
    return {"median_final": float(np.median(final)), "p5_final": float(np.percentile(final, 5)),
            "p95_final": float(np.percentile(final, 95)), "p_loss": float((final < 1).mean()),
            "dd_median": float(np.median(dd)), "dd_p95": float(np.percentile(dd, 95)),
            "p_dd_over_20": float((dd > 0.20).mean()), "p_dd_over_30": float((dd > 0.30).mean())}
