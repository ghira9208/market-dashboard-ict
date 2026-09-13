"""Standard technical indicators — plain pandas, no ICT-specific logic (see
detectors.py for that). Kept separate since these are generic/well-known
formulas anyone would recognize, not this project's own detection logic."""

import numpy as np
import pandas as pd


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


def atr(df, period=14):
    """Average True Range — Wilder's own smoothed measure of how much a
    price typically moves per bar, in the instrument's raw price units
    (not a percentage, not bounded). Needs the full OHLC, not just Close,
    unlike ema/rsi/macd/bollinger_bands above — True Range for a bar is
    the LARGEST of: that bar's own high-low, the gap up from the prior
    close to this bar's high, or the gap down to this bar's low —
    capturing a gap/overnight move a plain high-low range would miss
    entirely. Smoothed the same Wilder's EWM way rsi() above already
    does (alpha=1/period, not a plain rolling mean) — the standard,
    industry-default variant, so this reads the same as ATR(14) on any
    other charting platform."""
    h_col, l_col, c_col = ("High", "Low", "Close") if "High" in df else ("high", "low", "close")
    high, low, close = df[h_col], df[l_col], df[c_col]
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


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
    algorithm. "hvn_indices"/"lvn_indices": up to 3 indices each into
    `buckets` — High/Low Volume Nodes, the profile's other local peaks/
    valleys beyond just POC (see their own comment below for the exact
    rule), busiest-first for hvn / thinnest-first for lvn. Either can be
    empty on a profile too flat/noisy to clear the prominence filter.
    "shape"/"shape_label"/"shape_description": classic Market Profile
    shape typing (see its own comment below) — shape is a stable machine
    key ("normal"/"p_shape"/"b_shape"/"double_distribution"), shape_label
    the display name, shape_description one plain-language sentence on
    what it means for how this session traded."""
    h_col, l_col = ("High", "Low") if "High" in df else ("high", "low")
    v_col = "Volume" if "Volume" in df else ("volume" if "volume" in df else None)
    if v_col is None:
        return None
    high = df[h_col].to_numpy(dtype=float)
    low = df[l_col].to_numpy(dtype=float)
    volume = df[v_col].to_numpy(dtype=float)
    if not np.isfinite(volume).any() or float(np.nansum(volume)) <= 0:
        return None

    # A single bad print — a provider-side data glitch, not a real trade —
    # can otherwise swamp the entire profile. Confirmed directly on live
    # data: two DOT-USD hourly bars (180d/60m fetch) reported ~18 BILLION
    # volume apiece against a ~1.4 MILLION per-bar median across that same
    # fetch — roughly 13,000x — accounting for 87% of the dataset's total
    # reported volume between just those two bars. Every bucket's weight
    # came down to essentially two ticks, collapsing POC/VAH/VAL onto
    # whatever price those two happened to print at, regardless of where
    # real trading actually concentrated (the on-chart symptom: VAH/POC/
    # VAL clustered within a few cents of each other, floating nowhere
    # near the visible candles). Capping each bar's OWN contribution at
    # the dataset's own 99th percentile — not a fixed constant, so this
    # adapts to whatever "normal" looks like for this specific ticker/
    # interval/volume-unit combination — neutralizes an extreme outlier's
    # influence while leaving every genuinely-high-but-real bar's relative
    # weight essentially untouched (a bar sitting at the 90th percentile
    # is unaffected either way).
    _nonzero_volume = volume[np.isfinite(volume) & (volume > 0)]
    if len(_nonzero_volume) > 0:
        volume = np.minimum(volume, np.percentile(_nonzero_volume, 99))

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

    # High/Low Volume Nodes — a real profile usually has more than one
    # meaningful cluster, not just the single busiest bucket (POC): a
    # High Volume Node is a local peak where trading concentrated (price
    # tends to stall/consolidate there if revisited), a Low Volume Node a
    # local valley where little did (price tends to move THROUGH fast if
    # revisited, since there's little resting interest to absorb it) —
    # the standard "fast market" read professional order-flow desks pull
    # off a profile beyond just POC/VAH/VAL. "Local peak/valley" = higher/
    # lower than BOTH neighbors (an edge bucket compares to its one
    # neighbor only). Filtered by a minimum prominence relative to the
    # busiest bucket, not just "any local wiggle" — confirmed directly:
    # without a threshold, a real 24-bucket profile flagged 8-10 "peaks,"
    # most just one bucket taller than its immediate neighbor by a few
    # percent, nothing a trader would actually call a distinct node.
    # poc_idx is excluded from HVN candidates — it's already the global
    # max (trivially also a local one) and gets its own separate marker.
    def _is_local_peak(i):
        left_ok = i == 0 or bucket_volumes[i] > bucket_volumes[i - 1]
        right_ok = i == n_buckets - 1 or bucket_volumes[i] > bucket_volumes[i + 1]
        return left_ok and right_ok

    def _is_local_valley(i):
        left_ok = i == 0 or bucket_volumes[i] < bucket_volumes[i - 1]
        right_ok = i == n_buckets - 1 or bucket_volumes[i] < bucket_volumes[i + 1]
        return left_ok and right_ok

    hvn_indices = sorted(
        (i for i in range(n_buckets) if i != poc_idx and _is_local_peak(i) and bucket_volumes[i] >= 0.35 * max_vol),
        key=lambda i: bucket_volumes[i], reverse=True,
    )[:3]
    lvn_indices = sorted(
        (i for i in range(n_buckets) if _is_local_valley(i) and bucket_volumes[i] <= 0.15 * max_vol),
        key=lambda i: bucket_volumes[i],
    )[:3]

    # Classic Market Profile shape typing (Steidlmayer's own terminology,
    # still the standard professional vocabulary for "what kind of day/
    # session was this"), read straight off the same bucket distribution:
    #
    # Double Distribution — a real SECOND cluster far enough from POC to
    # be its own separate node (reusing the HVN filter above, not a fresh
    # threshold) rather than just a wide-but-single hump. Reads as "the
    # market spent real time in two distinct zones" — typically a trend
    # day that broke cleanly out of one balance area into another, or a
    # session with two separate news-driven regimes. Checked first: a
    # profile can look lopsided toward one third by the upper/lower-share
    # test below even while its real story is "two peaks," so this takes
    # priority over that read.
    #
    # Otherwise, where the bulk of volume actually sits divides into
    # three plain-language reads: P-shape (heavy in the upper third —
    # short-covering/bullish acceptance, price rallied and volume built
    # up AT the highs, not on the way there), b-shape (mirror, heavy in
    # the lower third — bearish distribution, a top forming/acceptance at
    # lower prices), or Normal/Balanced (volume centered, the standard
    # bell-curve rotational read — no strong directional conviction
    # either way). 0.45 (vs. an even one-third split's own 0.333) is
    # "meaningfully more than its fair share," not "technically greater."
    if any(abs(i - poc_idx) >= 0.35 * n_buckets for i in hvn_indices):
        shape, shape_label = "double_distribution", "Double Distribution"
        shape_description = ("Two separate value areas — the market spent real time in two distinct zones, "
                              "often seen on a trend day that broke out of one balance into another.")
    else:
        third = max(1, n_buckets // 3)
        total_vol = float(bucket_volumes.sum())
        upper_share = float(bucket_volumes[-third:].sum()) / total_vol
        lower_share = float(bucket_volumes[:third].sum()) / total_vol
        if upper_share >= 0.45:
            shape, shape_label = "p_shape", "P-shape"
            shape_description = ("Bullish acceptance — heavy trading concentrated near the top of the range, "
                                  "consistent with a rally that's being accepted at these higher prices, not "
                                  "just wicking through them.")
        elif lower_share >= 0.45:
            shape, shape_label = "b_shape", "b-shape"
            shape_description = ("Bearish distribution — heavy trading concentrated near the bottom of the "
                                  "range, consistent with a decline being accepted at these lower prices, not "
                                  "just wicking through them.")
        else:
            shape, shape_label = "normal", "Normal/Balanced"
            shape_description = ("Rotational — volume centered around one area with no strong directional "
                                  "conviction either way, the standard 'balanced auction' read.")

    return {"buckets": buckets, "poc_price": poc_price,
            "value_area_high": float(bucket_highs[hi_idx]), "value_area_low": float(bucket_lows[lo_idx]),
            "hvn_indices": hvn_indices, "lvn_indices": lvn_indices,
            "shape": shape, "shape_label": shape_label, "shape_description": shape_description}
