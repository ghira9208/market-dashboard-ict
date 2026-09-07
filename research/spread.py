"""
Corwin-Schultz (2012) high-low spread estimator — replaces the flat, guessed
round-trip cost (a slider defaulting to an arbitrary 10bps) with a number
actually computed from the data already in hand. No new data source needed:
it works from each bar's own High/Low alone, on the insight that a bar's
range contains both real price volatility AND the bid-ask bounce (a trade
at the ask prints a local high, a trade at the bid prints a local low, even
with zero underlying price movement) — volatility scales with sqrt(time),
the spread component doesn't, so comparing a 2-bar-window's range against
the sum of two 1-bar ranges isolates an estimate of the spread alone.

Standard in market-microstructure literature specifically because it needs
only OHLC — no tick/bid-ask feed required, which is this project's actual
situation for crypto/stocks (Yahoo/Binance klines never carry bid/ask) even
though FX tick data with REAL historical bid/ask exists on histdata.com and
was deliberately skipped for size (see research/histdata_import.py). If
that's ever imported for a specific pair, its true tick spread would be a
strictly better number than this estimate — this is the general-purpose
fallback that works everywhere else.

Reference: Corwin & Schultz, "A Simple Way to Estimate Bid-Ask Spreads from
Daily High and Low Prices," Journal of Finance, 2012.
"""

import numpy as np

_K1 = 3 - 2 * np.sqrt(2)  # the paper's own constant, appears twice in the formula


def estimate_spread_bps(df, window=2000):
    """Returns the estimated round-trip spread, in basis points, averaged
    over the last `window` bars (recent conditions matter more than a
    25-year-old average — None uses the whole dataset). Per-bar-pair
    estimates that come out negative or undefined (a known artifact of the
    formula on thin/choppy data, flagged in the original paper too) are
    dropped rather than allowed to drag the average down or NaN it out."""
    high = (df["High"] if "High" in df else df["high"]).to_numpy()
    low = (df["Low"] if "Low" in df else df["low"]).to_numpy()
    if window:
        high, low = high[-window:], low[-window:]
    if len(high) < 3:
        return None

    h1, l1 = high[:-1], low[:-1]
    h2, l2 = high[1:], low[1:]
    hh = np.maximum(h1, h2)
    ll = np.minimum(l1, l2)

    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.log(h1 / l1) ** 2 + np.log(h2 / l2) ** 2
        gamma = np.log(hh / ll) ** 2
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / _K1 - np.sqrt(gamma / _K1)
        spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))

    spread = spread[np.isfinite(spread)]
    spread = spread[spread > 0]  # negative estimates are a known formula artifact, not a real "negative spread"
    if len(spread) == 0:
        return None
    return float(np.mean(spread)) * 10_000  # fraction -> basis points
