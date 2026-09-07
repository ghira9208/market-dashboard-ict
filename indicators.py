"""Standard technical indicators — plain pandas, no ICT-specific logic (see
fvg.py for that). Kept separate since these are generic/well-known formulas
anyone would recognize, not this project's own detection logic."""

import numpy as np


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    """Wilder's RSI via the standard EWM approximation (alpha=1/period) —
    the same variant virtually every retail charting tool (including
    TradingView) uses by default."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(series, fast=12, slow=26, signal=9):
    """Returns (macd_line, signal_line, histogram)."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def bollinger_bands(series, period=20, num_std=2):
    """Returns (upper, basis, lower)."""
    basis = series.rolling(period).mean()
    std = series.rolling(period).std()
    return basis + num_std * std, basis, basis - num_std * std


def volume_profile(df, n_buckets=24, value_area_pct=0.70):
    """Approximates a volume profile (how much volume traded at each PRICE
    level, not each time bar) from plain OHLCV — a real tick-level profile
    needs individual trades, which this project only has for Binance's own
    real trade stream (see the Footprint chart); this is the standard
    OHLC-based approximation almost every retail platform falls back to
    otherwise. Each bar's own volume is split across every price bucket its
    own [low, high] range touches, proportional to how much of that bar's
    range falls in each bucket — a flat bar (high == low, e.g. a stale or
    illiquid print) puts its whole volume in the single bucket containing
    that price instead of dividing by a zero-width range.

    Returns None — an honest empty state, not a profile built from
    fabricated numbers — when there's no usable volume at all (confirmed
    directly: Yahoo Forex tickers report Volume as a constant 0, so this
    is a real, not hypothetical, case) or when every bar has the exact
    same price (a zero-width overall range, nothing to bucket).

    Otherwise returns {"buckets": [{"price_low", "price_high", "volume",
    "volume_frac"}, ...] (ascending by price, volume_frac = this bucket's
    volume / the busiest bucket's volume, so the busiest reads exactly
    1.0), "poc_price": the busiest bucket's own midpoint, "value_area_high"
    /"value_area_low": the narrowest price band containing value_area_pct
    of the total distributed volume, built by repeatedly extending
    whichever side (above the current band or below it) holds the next
    largest neighboring bucket — the standard "expand from POC" value-area
    algorithm."""
    h_col, l_col = ("High", "Low") if "High" in df else ("high", "low")
    v_col = "Volume" if "Volume" in df else ("volume" if "volume" in df else None)
    if v_col is None:
        return None
    high = df[h_col].to_numpy(dtype=float)
    low = df[l_col].to_numpy(dtype=float)
    volume = df[v_col].to_numpy(dtype=float)
    if not np.isfinite(volume).any() or float(np.nansum(volume)) <= 0:
        return None

    price_lo = float(np.nanmin(low))
    price_hi = float(np.nanmax(high))
    if not (price_hi > price_lo):
        return None
    bucket_width = (price_hi - price_lo) / n_buckets
    bucket_edges = price_lo + bucket_width * np.arange(n_buckets + 1)
    bucket_lows, bucket_highs = bucket_edges[:-1], bucket_edges[1:]

    bucket_volumes = np.zeros(n_buckets)
    bar_range = high - low
    for i in range(len(df)):
        v = volume[i]
        if not np.isfinite(v) or v <= 0:
            continue
        rng = bar_range[i]
        if rng <= 0:
            idx = min(int((low[i] - price_lo) / bucket_width), n_buckets - 1)
            bucket_volumes[idx] += v
            continue
        overlap = np.clip(np.minimum(bucket_highs, high[i]) - np.maximum(bucket_lows, low[i]), 0, None)
        bucket_volumes += v * (overlap / rng)

    max_vol = float(bucket_volumes.max())
    if max_vol <= 0:
        return None
    poc_idx = int(np.argmax(bucket_volumes))
    poc_price = float((bucket_lows[poc_idx] + bucket_highs[poc_idx]) / 2)

    lo_idx = hi_idx = poc_idx
    covered = bucket_volumes[poc_idx]
    target = float(bucket_volumes.sum()) * value_area_pct
    while covered < target and (lo_idx > 0 or hi_idx < n_buckets - 1):
        below_vol = bucket_volumes[lo_idx - 1] if lo_idx > 0 else -1.0
        above_vol = bucket_volumes[hi_idx + 1] if hi_idx < n_buckets - 1 else -1.0
        if above_vol >= below_vol:
            hi_idx += 1
            covered += bucket_volumes[hi_idx]
        else:
            lo_idx -= 1
            covered += bucket_volumes[lo_idx]

    buckets = [
        {"price_low": float(bucket_lows[i]), "price_high": float(bucket_highs[i]),
         "volume": float(bucket_volumes[i]), "volume_frac": float(bucket_volumes[i] / max_vol)}
        for i in range(n_buckets)
    ]
    return {"buckets": buckets, "poc_price": poc_price,
            "value_area_high": float(bucket_highs[hi_idx]), "value_area_low": float(bucket_lows[lo_idx])}
