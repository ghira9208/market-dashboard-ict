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

EDGE_FOUND also requires two things a bare p-value can't guarantee, both
added together (2026-10-06, external review of this pipeline's BH/FDR
methodology prompted checking for them directly):

1. A minimum sample size (MIN_EVENTS_FOR_EDGE). The permutation test above
   is honest about significance but says nothing about whether a result is
   PRACTICALLY trustworthy — a 3-event lucky streak and a 500-event real
   edge can clear p < 0.05 identically. This is the real equivalent of the
   review's "statistical significance != trading significance" point:
   congratulations, you've statistically proven that dust exists.
2. A bootstrap confidence floor (bootstrap_ci_lower_5pct): resamples WITH
   replacement from this event population's own real signed-after-cost
   returns, n_bootstrap times, and takes the 5th percentile of the
   resampled means. Requiring that lower bound to still be positive is the
   real Monte Carlo robustness check the same review asked for — same
   bootstrap technique etoro-dashboard/risk_metrics.py's own
   monte_carlo_drawdown_percentiles already uses for live drawdown risk,
   applied here to the DISCOVERY gate instead: an edge whose point estimate
   is positive but whose own plausible resamples routinely dip negative is
   exactly the fragile, one-lucky-trade kind of "discovery" BH alone can't
   catch, since permutation reshuffles labels on the SAME fixed outcomes
   rather than asking how stable the mean itself is to resampling.
"""

import numpy as np
import pandas as pd

MIN_EVENTS_FOR_EDGE = 30  # the standard rule-of-thumb floor for a sample mean to not be dominated by one or two lucky/unlucky events; roughly where the CLT starts being a reasonable approximation


def _bootstrap_mean_ci_lower(signed_returns, n_bootstrap=2000, pct=5, seed=0):
    """The Monte Carlo piece: resamples `signed_returns` (already direction-
    signed and cost-adjusted, real observed values — nothing simulated)
    WITH replacement, n_bootstrap times, and returns the pct-th percentile
    of the resampled means. A low/negative value means this population's
    own apparent edge is fragile to which exact events happened to be in
    the sample — not yet trustworthy even if the point estimate and the
    permutation p-value both look fine."""
    rng = np.random.default_rng(seed)
    n = len(signed_returns)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(signed_returns, size=n, replace=True)
        boot_means[i] = sample.mean()
    return float(np.percentile(boot_means, pct))


def run_event_study(events_df, cost_bps=10.0, n_permutations=5000, n_bootstrap=2000, seed=0,
                     direction_col="fvg_type", ticker=None, news_blackout=None):
    """events_df: output of events.extract_fvg_retracement_events (or any
    of the other extract_*_events functions — this only reads
    entry_time/raw_return/direction_col, the shape they all share).
    cost_bps: assumed round-trip cost (spread + slippage + fees) in basis
    points, subtracted from every trade before any stat is computed — a
    pessimistic-by-design assumption since free OHLCV has no real fill/
    order-book data to model this from directly (see this module's own
    docstring on backtest realism).

    ticker/news_blackout: both None (default, unchanged behavior), or
    ticker plus a (minutes_before, minutes_after) pair — drops any event
    whose own entry_time falls inside a high-impact news window for
    ticker's own relevant currencies (see news.blackout_mask), before any
    stat is computed. Same "don't count a trade that would have opened
    right into a release" reasoning as market-dashboard-ict's live
    dashboards (recommender.py's pattern_win_rate/backtest_custom_rule) —
    this is the same engine, just applied to this module's own events_df
    shape instead of an OHLCV df's row index. Both must be given together;
    ticker alone or news_blackout alone is a no-op (nothing to look up a
    ticker's currencies against without the other).

    n_bootstrap: resamples for the bootstrap confidence floor (see this
    module's own docstring) — only spent when n_events already clears
    MIN_EVENTS_FOR_EDGE, since bootstrapping a tiny sample just measures
    its own tininess."""
    if events_df.empty:
        return {"n_events": 0, "verdict": "NO_DATA", "note": "No qualifying events found."}

    if ticker and news_blackout:
        import news
        mask = news.blackout_mask(pd.DatetimeIndex(events_df["entry_time"]), ticker,
                                   news_blackout[0], news_blackout[1])
        events_df = events_df[~mask]
        if events_df.empty:
            return {"n_events": 0, "verdict": "NO_DATA",
                    "note": "No qualifying events left after excluding high-impact news windows."}

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

    # Bootstrap CI only when there's enough data for resampling it to mean
    # anything — below MIN_EVENTS_FOR_EDGE the verdict already fails on the
    # sample-size gate regardless, so skip the 2000-resample cost.
    bootstrap_ci_lower = (_bootstrap_mean_ci_lower(signed_after_cost, n_bootstrap=n_bootstrap, seed=seed)
                           if n >= MIN_EVENTS_FOR_EDGE else None)

    verdict = "EDGE_FOUND" if (
        p_value < 0.05 and observed_mean > 0
        and n >= MIN_EVENTS_FOR_EDGE
        and bootstrap_ci_lower is not None and bootstrap_ci_lower > 0
    ) else "NO_EDGE"

    return {
        "n_events": n,
        "cost_bps": cost_bps,
        "mean_return_after_cost": observed_mean,
        "win_rate_after_cost": float((signed_after_cost > 0).mean()),
        "p_value_vs_random_direction": p_value,
        "min_events_required": MIN_EVENTS_FOR_EDGE,
        "bootstrap_ci_lower_5pct": bootstrap_ci_lower,
        "by_type": by_type,
        "verdict": verdict,
    }
