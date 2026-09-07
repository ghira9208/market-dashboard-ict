"""
Event-study statistical testing — deliberately NOT a portfolio/PnL simulator.
A single-signal event study (forward return after each event vs. a null
distribution) is the simplest thing that can honestly answer "does this
event carry directional information at all" — a full backtester (position
sizing, overlapping trades, portfolio equity curve) is a real next step, but
only worth building once something survives this cheaper, harder-to-fool-
yourself test. See the pipeline recap: no edge here means iterate on the
signal definition, not skip ahead to backtesting something with a stronger
tool that's just as likely to rediscover the same non-edge with more noise.

The test is a permutation test, not a plain "is the mean above zero" check:
null hypothesis is "the event's predicted direction (bullish FVG -> long,
bearish FVG -> short) carries no real information" — i.e. a coin flip would
do just as well. That specifically controls for the asset's own drift during
the backtest window: if BTC simply trended up across the sample, ANY
direction label (real or random) picks up some of that drift, so comparing
against "mean of random ±1 labels on these same events" isolates whatever
the FVG-direction label adds ON TOP of that background drift, rather than
mistaking background drift itself for edge.
"""

import numpy as np


def run_event_study(events_df, cost_bps=10.0, n_permutations=5000, seed=0, direction_col="fvg_type"):
    """events_df: output of events.extract_fvg_retracement_events.
    cost_bps: assumed round-trip cost (spread + slippage + fees) in basis
    points, subtracted from every trade before any stat is computed — a
    pessimistic-by-design assumption since free OHLCV has no real fill/
    order-book data to model this from directly (see this module's own
    docstring on backtest realism)."""
    if events_df.empty:
        return {"n_events": 0, "verdict": "NO_DATA", "note": "No qualifying events found."}

    rng = np.random.default_rng(seed)
    cost = cost_bps / 10_000.0

    raw = events_df["raw_return"].to_numpy()
    direction = np.where(events_df[direction_col] == "bullish", 1, -1)
    signed_after_cost = raw * direction - cost

    observed_mean = float(signed_after_cost.mean())

    # Null distribution: same raw returns, direction label randomized —
    # answers "if the FVG-type label were meaningless, how often would a
    # random labeling alone produce a mean this extreme."
    n = len(raw)
    null_means = np.empty(n_permutations)
    for i in range(n_permutations):
        rand_dir = rng.choice([-1, 1], size=n)
        null_means[i] = (raw * rand_dir - cost).mean()

    p_value = float((np.abs(null_means) >= abs(observed_mean)).mean())

    by_type = {}
    for t in ("bullish", "bearish"):
        mask = events_df[direction_col] == t
        if mask.any():
            sub = signed_after_cost[mask.to_numpy()]
            by_type[t] = {
                "n": int(mask.sum()),
                "win_rate": float((sub > 0).mean()),
                "mean_return_after_cost": float(sub.mean()),
            }

    verdict = "EDGE_FOUND" if (p_value < 0.05 and observed_mean > 0) else "NO_EDGE"

    return {
        "n_events": n,
        "cost_bps": cost_bps,
        "mean_return_after_cost": observed_mean,
        "win_rate_after_cost": float((signed_after_cost > 0).mean()),
        "p_value_vs_random_direction": p_value,
        "by_type": by_type,
        "verdict": verdict,
    }
