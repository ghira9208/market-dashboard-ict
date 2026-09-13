"""
"Where big money is, direction is more probable" — turned into a system
that actually cuts noise instead of just dumping raw derivatives metrics on
a page. Presentation-agnostic, same convention as news.py/recommender.py:
this module owns the data fetch + signal math, institutional_app.py only
renders it.

The core design decision: no single metric here is trusted alone. Open
interest, long/short positioning, order flow, and funding rate are all
individually well-known to be noisy — each one gives false reads
constantly on its own. So this module never reports "open interest went
up, must be bullish." Instead, THREE independently-computed proxies for
"which way size is leaning" are summed into institutional_bias()'s score,
each contributing -1/0/+1 — a real lean only shows up as a "Strong" read
when multiple of them agree, never off one signal alone:

  1. oi_price_regime — does rising/falling open interest (fresh positions
     opening/closing) line up with the direction price actually moved.
  2. smart_money_divergence — do Binance's own "top trader" accounts
     (their own size-based tier, the closest free proxy to "big players"
     this API exposes) sit net-long or net-short RELATIVE to the retail-
     heavy global account ratio — not just directionally bullish/bearish
     in isolation, since retail is usually net-long anyway; what matters
     is whether size is leaning MORE bullish or MORE bearish than the
     crowd.
  3. cvd_price_divergence — does net taker buy/sell volume (CVD, real
     order flow) actually back the direction price moved, or is the move
     happening on thin/opposite flow (a rally on net selling is usually
     short-covering, not fresh demand; a selloff on net buying is usually
     absorption, not fresh supply).

When multiple agree, that agreement IS the noise-reduction — independently
noisy signals rarely align by chance. When they don't, the honest answer
is "no clear read," not a forced pick either direction.

Funding rate is reported separately as a crowding/context flag, not
folded into the score — it measures how expensive/stretched current
leveraged positioning is, not which way size is actually leaning, so
averaging it in with the other two would blur two different questions
into one number.

Explicitly NOT attempted here, and why:
  - On-chain whale-wallet tracking: needs Glassnode/Nansen/a paid feed,
    nothing free covers this.
  - Spot ETF flows: the single most literal "institutional footprint"
    data that exists, but no clean free API — every aggregator (Farside,
    SoSoValue) is scraped from daily press releases, not a real feed.
  - CME futures open interest / CFTC COT reports: genuinely free and
    genuinely institutional (COT explicitly separates "Leveraged Funds"
    from "Dealer" positions), but weekly-lag data — a real v2 candidate,
    cut from v1 to ship the live Binance-based core first.

CVD (cumulative volume delta) WAS cut for the same "needs a full trade-
tape fetch" reason, and is no longer — footprint.py's own per-session CVD
does read raw aggTrades, but Binance's ordinary kline response already
carries each candle's own taker-buy-base-asset-volume field for free
(taker sell = candle volume - taker buy; delta = buy - sell), so a
daily-resolution CVD costs exactly one more small kline fetch, the same
cost profile as every other signal in this module. get_taker_volume_history
below is that fetch; cvd_price_divergence turns it into the third signal.

Binance's own Futures Data endpoints (openInterestHist,
topLongShortPositionRatio, globalLongShortAccountRatio) are hard-limited
to the last 30 days regardless of requested limit — confirmed directly,
not documented clearly by Binance itself. Total market cap has the
opposite problem: CoinGecko's free /global endpoint is a live snapshot
with no historical series at all, so real history for that one only
builds up from points this app has actually been running to observe live
— same accumulate-going-forward pattern news.py's own calendar history
cache uses, empty until enough days pass, honestly labeled as such rather
than faked.
"""
import json
import os

import numpy as np
import pandas as pd
import requests
import streamlit as st

from data import get_yf_ohlcv

_FAPI_BASE = "https://fapi.binance.com"

# Started as the same six pairs research/sweep.py uses for its own crypto
# batch sweep, extended with three more on request (DOT/IOTA/FIL) — the
# two lists are independent now, this one is this page's own watchlist,
# not obligated to match research/sweep.py's. Every entry confirmed
# directly (not assumed) to resolve on both feeds this module needs:
# Binance USDT-margined futures (OI/ratio/taker-volume) and Yahoo (spot
# daily price for oi_price_regime/cvd_price_divergence's own price leg).
# Not an attempt to cover the whole market; a focused watchlist for a
# focused tool.
CRYPTO_TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD",
                   "DOT-USD", "IOTA-USD", "FIL-USD"]

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
_MCAP_HISTORY_PATH = os.path.join(_CACHE_DIR, "market_cap_history.json")
_BIAS_HISTORY_PATH = os.path.join(_CACHE_DIR, "institutional_bias_history.json")


def _binance_symbol(yahoo_ticker):
    return yahoo_ticker.replace("-USD", "").upper() + "USDT"


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_futures_json(path, params_tuple):
    """params_tuple, not a dict: st.cache_data hashes its arguments to key
    the cache, and a plain dict isn't hashable — a tuple of (key, value)
    pairs is, and turns back into a dict with one line below."""
    params = dict(params_tuple)
    resp = requests.get(f"{_FAPI_BASE}{path}", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_open_interest_history(ticker, period="1d", limit=30):
    """Daily sum of open contracts across Binance USDT-margined futures for
    this ticker — rising means NEW positions opening (real conviction,
    someone put fresh capital at risk), falling means positions closing
    (either profit-taking, stopping out, or liquidation — this alone can't
    tell which). Returns an empty DataFrame on any fetch failure, same
    honest-empty convention as news.py's get_calendar()."""
    symbol = _binance_symbol(ticker)
    try:
        raw = _fetch_futures_json("/futures/data/openInterestHist",
                                   (("symbol", symbol), ("period", period), ("limit", limit)))
    except Exception:
        return pd.DataFrame(columns=["time", "oi", "oi_value_usd"])
    if not raw:
        return pd.DataFrame(columns=["time", "oi", "oi_value_usd"])
    df = pd.DataFrame(raw)
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["oi"] = df["sumOpenInterest"].astype(float)
    df["oi_value_usd"] = df["sumOpenInterestValue"].astype(float)
    return df[["time", "oi", "oi_value_usd"]].sort_values("time").reset_index(drop=True)


def get_top_trader_ratio(ticker, period="1d", limit=30):
    """Binance's own "top trader" tier (their largest accounts by size,
    the closest free proxy to "big players" this API exposes) — the
    fraction of them net long vs net short. This is a POSITION-weighted
    ratio (topLongShortPositionRatio), not account-count — one whale
    holding a huge position counts as its actual size, not as "one
    account," which is what actually matters for "where is size leaning."
    """
    symbol = _binance_symbol(ticker)
    try:
        raw = _fetch_futures_json("/futures/data/topLongShortPositionRatio",
                                   (("symbol", symbol), ("period", period), ("limit", limit)))
    except Exception:
        return pd.DataFrame(columns=["time", "long_short_ratio"])
    if not raw:
        return pd.DataFrame(columns=["time", "long_short_ratio"])
    df = pd.DataFrame(raw)
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["long_short_ratio"] = df["longShortRatio"].astype(float)
    return df[["time", "long_short_ratio"]].sort_values("time").reset_index(drop=True)


def get_global_ratio(ticker, period="1d", limit=30):
    """Same shape as get_top_trader_ratio, but Binance's ALL-accounts
    (account-count-weighted, retail-heavy) ratio — the crowd this module
    compares "top traders" against, not a signal read alone."""
    symbol = _binance_symbol(ticker)
    try:
        raw = _fetch_futures_json("/futures/data/globalLongShortAccountRatio",
                                   (("symbol", symbol), ("period", period), ("limit", limit)))
    except Exception:
        return pd.DataFrame(columns=["time", "long_short_ratio"])
    if not raw:
        return pd.DataFrame(columns=["time", "long_short_ratio"])
    df = pd.DataFrame(raw)
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["long_short_ratio"] = df["longShortRatio"].astype(float)
    return df[["time", "long_short_ratio"]].sort_values("time").reset_index(drop=True)


def get_funding_rate_history(ticker, limit=30):
    """Perpetual futures funding rate — positive means longs pay shorts
    (longs are the crowded, leverage-heavy side paying for the privilege),
    negative the reverse. Reported as context, not folded into
    institutional_bias's score — see this module's own docstring on why."""
    symbol = _binance_symbol(ticker)
    try:
        raw = _fetch_futures_json("/fapi/v1/fundingRate", (("symbol", symbol), ("limit", limit)))
    except Exception:
        return pd.DataFrame(columns=["time", "funding_rate"])
    if not raw:
        return pd.DataFrame(columns=["time", "funding_rate"])
    df = pd.DataFrame(raw)
    df["time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    return df[["time", "funding_rate"]].sort_values("time").reset_index(drop=True)


def _daily_price_series(ticker, days=45):
    """Yahoo daily closes, trimmed to just enough history to cover
    Binance's own 30-day window plus room for the regime lookback — reuses
    data.py's already-cached get_yf_ohlcv, no new fetch machinery."""
    df = get_yf_ohlcv(ticker, period="3mo", interval="1d")
    if df.empty:
        return pd.DataFrame(columns=["date", "close"])
    close_col = "Close" if "Close" in df else "close"
    out = df.tail(days).copy()
    out["date"] = out.index.tz_convert("UTC").date
    out["close"] = out[close_col]
    return out[["date", "close"]].reset_index(drop=True)


def _classify_oi_regime(oi_chg_pct, price_chg_pct):
    """Pure decision rule behind oi_price_regime, split out so
    institutional_backtest.py can run this EXACT classification against
    historical windows instead of a hand-copied second version that could
    quietly drift from what the live page actually computes."""
    oi_up, price_up = oi_chg_pct > 0, price_chg_pct > 0
    if oi_up and price_up:
        return "Long Buildup", 1
    elif oi_up and not price_up:
        return "Short Buildup", -1
    elif not oi_up and not price_up:
        return "Long Liquidation", -1
    else:
        return "Short Covering", 1


def oi_price_regime(ticker, lookback=7):
    """Classifies the last `lookback` days into one of four institutional-
    positioning regimes, ICT/futures-desk standard reading of OI-vs-price:

      Long Buildup    (OI up,   price up)   — fresh long conviction, real
                                               new capital backing the move.
      Short Buildup   (OI up,   price down) — fresh short conviction.
      Long Liquidation(OI down, price down) — longs forced/choosing out;
                                               confirms weakness, but isn't
                                               fresh conviction of its own.
      Short Covering  (OI down, price up)   — shorts closing, not new
                                               buying — the weaker of the
                                               two "price up" readings.

    direction: +1 (Long Buildup or Short Covering), -1 (Short Buildup or
    Long Liquidation), 0 if either series doesn't have enough history to
    judge. persistence: fraction of individual day-over-day OI changes
    within the lookback window that had the same sign as the overall
    lookback-start-to-end change — a regime that held steadily for 7
    straight days reads as more real than one where OI whipsawed and just
    happened to net the same direction."""
    oi_df = get_open_interest_history(ticker)
    price_df = _daily_price_series(ticker)
    if len(oi_df) < lookback + 1 or len(price_df) < lookback + 1:
        return {"regime": "Insufficient data", "direction": 0, "persistence": None,
                "oi_change_pct": None, "price_change_pct": None}

    oi_df = oi_df.assign(date=oi_df["time"].dt.date)
    merged = pd.merge(oi_df[["date", "oi"]], price_df, on="date", how="inner").tail(lookback + 1)
    if len(merged) < 2:
        return {"regime": "Insufficient data", "direction": 0, "persistence": None,
                "oi_change_pct": None, "price_change_pct": None}

    oi_chg_pct = (merged["oi"].iloc[-1] - merged["oi"].iloc[0]) / merged["oi"].iloc[0] * 100
    price_chg_pct = (merged["close"].iloc[-1] - merged["close"].iloc[0]) / merged["close"].iloc[0] * 100
    regime, direction = _classify_oi_regime(float(oi_chg_pct), float(price_chg_pct))

    daily_oi_diff = merged["oi"].diff().dropna()
    overall_sign = 1 if oi_chg_pct > 0 else -1
    persistence = float((np.sign(daily_oi_diff) == overall_sign).mean()) if len(daily_oi_diff) else None

    return {"regime": regime, "direction": direction, "persistence": persistence,
            "oi_change_pct": float(oi_chg_pct), "price_change_pct": float(price_chg_pct)}


# Binance's own retail base rate: across this ticker list, the global
# (all-accounts) ratio sits net-long almost permanently — this project's
# own live check found global ratios of 1.5-1.65 (60-62% long) against top
# traders at 2.1-2.2 (68-69% long) on an ordinary day, not an extreme one.
# A raw top-vs-global difference of "top traders are also long" would
# trivially fire "bullish" on every single day, telling you nothing — what
# actually matters is whether size is leaning MORE bullish or bearish than
# that base rate, hence comparing in log-space (symmetric around zero
# regardless of which side of 1.0 the ratios sit on) with a real threshold
# to clear before calling it a divergence instead of noise.
_DIVERGENCE_THRESHOLD = 0.15  # log-ratio units; ~16% relative gap between top and global ratios


def _classify_smart_money(top_ratio, global_ratio):
    """Pure decision rule behind smart_money_divergence — see
    _classify_oi_regime's own docstring for why this is split out."""
    if top_ratio <= 0 or global_ratio <= 0:
        return 0, None
    divergence = float(np.log(top_ratio) - np.log(global_ratio))
    if divergence > _DIVERGENCE_THRESHOLD:
        direction = 1
    elif divergence < -_DIVERGENCE_THRESHOLD:
        direction = -1
    else:
        direction = 0
    return direction, divergence


def smart_money_divergence(ticker):
    """Compares Binance's own top-trader (position-weighted) long/short
    ratio against the all-accounts (retail-heavy) ratio, in log-space so
    "top traders 2x long, global 1x long" and "top traders 1x long, global
    0.5x long" — the same RELATIVE lean — score the same. direction: +1
    when top traders lean meaningfully more bullish than the crowd, -1
    meaningfully more bearish, 0 when they're not meaningfully different
    (the common case — most days aren't a real divergence) or data is
    missing."""
    top_df = get_top_trader_ratio(ticker)
    global_df = get_global_ratio(ticker)
    if top_df.empty or global_df.empty:
        return {"direction": 0, "divergence": None, "top_ratio": None, "global_ratio": None}

    top_ratio = float(top_df["long_short_ratio"].iloc[-1])
    global_ratio = float(global_df["long_short_ratio"].iloc[-1])
    if top_ratio <= 0 or global_ratio <= 0:
        return {"direction": 0, "divergence": None, "top_ratio": top_ratio, "global_ratio": global_ratio}

    direction, divergence = _classify_smart_money(top_ratio, global_ratio)
    return {"direction": direction, "divergence": divergence, "top_ratio": top_ratio, "global_ratio": global_ratio}


def get_taker_volume_history(ticker, interval="1d", limit=30):
    """Per-candle taker buy volume, close price, and total volume, straight
    from Binance's own kline response — no extra trade-tape fetch needed
    (see this module's own docstring on why that used to be the blocker).
    taker_sell = volume - taker_buy; delta = taker_buy - taker_sell for
    that one candle. Returns an empty DataFrame on any fetch failure, same
    honest-empty convention as every other fetcher here."""
    symbol = _binance_symbol(ticker)
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
    try:
        raw = _fetch_futures_json("/fapi/v1/klines",
                                   (("symbol", symbol), ("interval", interval), ("limit", limit)))
    except Exception:
        return pd.DataFrame(columns=["time", "close", "volume", "delta"])
    if not raw:
        return pd.DataFrame(columns=["time", "close", "volume", "delta"])
    df = pd.DataFrame(raw, columns=cols)
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    taker_buy = df["taker_buy_volume"].astype(float)
    df["delta"] = 2 * taker_buy - df["volume"]  # taker_buy - (volume - taker_buy)
    return df[["time", "close", "volume", "delta"]]


def _classify_cvd(cvd_net, price_chg_pct):
    """Pure decision rule behind cvd_price_divergence — see
    _classify_oi_regime's own docstring for why this is split out."""
    price_up, cvd_up = price_chg_pct > 0, cvd_net > 0
    if price_up and cvd_up:
        return "Confirmed Uptrend", 1
    elif price_up and not cvd_up:
        return "Unconfirmed Rally", -1
    elif not price_up and not cvd_up:
        return "Confirmed Downtrend", -1
    else:
        return "Unconfirmed Selloff", 1


def cvd_price_divergence(ticker, lookback=7):
    """Classifies the last `lookback` daily candles by whether NET taker
    buy/sell pressure (CVD — the running sum of each candle's own delta)
    actually backs the direction price moved, ICT/order-flow standard
    reading of "is this move real or thin":

      Confirmed Uptrend   (price up,   CVD net positive) — the rally is
                            backed by real net buying, not just drift.
      Unconfirmed Rally   (price up,   CVD net negative) — price rose
                            while net selling actually dominated (short
                            covering / thin book) — the weaker read.
      Confirmed Downtrend (price down, CVD net negative) — the selloff is
                            backed by real net selling.
      Unconfirmed Selloff (price down, CVD net positive) — price fell
                            while net buying actually dominated (stop hunt
                            / absorption) — the weaker read.

    direction: +1 for the two "real buying pressure" regimes (Confirmed
    Uptrend, Unconfirmed Selloff), -1 for the two "real selling pressure"
    regimes, 0 if there isn't enough history yet."""
    df = get_taker_volume_history(ticker, interval="1d", limit=lookback + 1)
    if len(df) < lookback + 1:
        return {"regime": "Insufficient data", "direction": 0, "cvd_net": None, "price_change_pct": None}

    window = df.tail(lookback + 1)
    cvd_net = float(window["delta"].iloc[1:].sum())
    price_chg_pct = float((window["close"].iloc[-1] - window["close"].iloc[0]) / window["close"].iloc[0] * 100)
    regime, direction = _classify_cvd(cvd_net, price_chg_pct)

    return {"regime": regime, "direction": direction, "cvd_net": cvd_net, "price_change_pct": price_chg_pct}


# A funding rate this far from zero (per 8h interval, Binance's standard
# cadence) marks positioning as genuinely stretched rather than the small
# positive drift that sits there most of the time — 0.01%/8h annualizes to
# ~11%, a real cost to hold the crowded side, not background noise.
_FUNDING_STRETCH_THRESHOLD = 0.0001


def funding_context(ticker):
    """The crowding flag, kept separate from institutional_bias's score —
    see this module's own docstring for why. Returns the latest funding
    rate plus a plain-language read of whether it's stretched enough to
    matter."""
    df = get_funding_rate_history(ticker, limit=1)
    if df.empty:
        return {"funding_rate": None, "note": None}
    rate = float(df["funding_rate"].iloc[-1])
    if rate > _FUNDING_STRETCH_THRESHOLD:
        note = "Longs paying to hold — leveraged long positioning is crowded/expensive right now."
    elif rate < -_FUNDING_STRETCH_THRESHOLD:
        note = "Shorts paying to hold — leveraged short positioning is crowded/expensive right now."
    else:
        note = "Unremarkable — no real crowding either way."
    return {"funding_rate": rate, "note": note}


_BIAS_LABELS = {
    3: "Strong accumulation lean", 2: "Strong accumulation lean", 1: "Mild accumulation lean",
    0: "No clear read",
    -1: "Mild distribution lean", -2: "Strong distribution lean", -3: "Strong distribution lean",
}


def institutional_bias(ticker):
    """The one number this whole module exists to produce: sums
    oi_price_regime, smart_money_divergence, and cvd_price_divergence into
    a -3..+3 score — see this module's own docstring for the full
    reasoning on why three independent reads beat trusting any one alone.
    Score 0 covers BOTH "no signal fired" and "signals disagreed,"
    deliberately not distinguished in the label (both mean "nothing to
    trust here"), but the full per-signal breakdown is always returned
    too, so a caller can show exactly why, never just a bare number."""
    regime = oi_price_regime(ticker)
    divergence = smart_money_divergence(ticker)
    cvd = cvd_price_divergence(ticker)
    funding = funding_context(ticker)

    signals = [
        {"name": "Positioning (Open Interest + Price)", "direction": regime["direction"],
         "detail": regime["regime"],
         "sufficient": regime["direction"] != 0 or regime["regime"] != "Insufficient data"},
        {"name": "Smart Money Divergence (top traders vs. crowd)", "direction": divergence["direction"],
         "detail": (f"top {divergence['top_ratio']:.2f} vs crowd {divergence['global_ratio']:.2f}"
                    if divergence["top_ratio"] is not None else "no data"),
         "sufficient": divergence["top_ratio"] is not None},
        {"name": "Order Flow (CVD vs. Price)", "direction": cvd["direction"],
         "detail": cvd["regime"],
         "sufficient": cvd["direction"] != 0 or cvd["regime"] != "Insufficient data"},
    ]
    usable = [s for s in signals if s["sufficient"]]
    score = sum(s["direction"] for s in usable)
    label = _BIAS_LABELS.get(score, "No clear read")

    # "raw" carries every sub-function's own full return dict, not just the
    # flattened signals list above — needed by _record_bias_snapshot below
    # to log enough detail that a future backtest can test each sub-signal
    # on its own, not just the combined score (deciding in advance which
    # ones "mattered" and only logging those would bake today's guess about
    # significance into data meant to test that guess later).
    return {"ticker": ticker, "score": score, "label": label, "signals": signals, "funding": funding,
            "raw": {"oi_price_regime": regime, "smart_money_divergence": divergence, "cvd_price_divergence": cvd}}


_BIAS_HISTORY_COLUMNS = [
    "date", "ticker", "score", "label", "price",
    "oi_direction", "oi_regime", "oi_change_pct", "oi_persistence",
    "smd_direction", "smd_divergence", "smd_top_ratio", "smd_global_ratio",
    "cvd_direction", "cvd_regime", "cvd_net", "cvd_price_change_pct",
    "funding_rate",
]


def _load_bias_history():
    """Every daily institutional_bias() snapshot this app has ever
    recorded, one row per (date, ticker) — empty (or short) until enough
    days pass running this page, same honest-limitation convention as
    get_market_cap_history(). This log is the ONLY way this module can
    ever accumulate more than ~30 days of institutional-signal history:
    Binance's own OI/ratio endpoints are hard-capped at 30 days regardless
    of what's requested (confirmed directly, see this module's own
    docstring), so a real backtest of whether this score means anything
    has to be built from days this page was actually running to record
    one — there's no way to backfill it after the fact."""
    try:
        with open(_BIAS_HISTORY_PATH) as f:
            rows = json.load(f)
    except Exception:
        return pd.DataFrame(columns=_BIAS_HISTORY_COLUMNS)
    if not rows:
        return pd.DataFrame(columns=_BIAS_HISTORY_COLUMNS)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _record_bias_snapshot(results):
    """Appends today's institutional_bias() result for every ticker in
    `results` to the local history log — overwrites each ticker's own
    prior entry for TODAY if this runs again the same day (the page
    reloading shouldn't multiply how many "days" a future backtest thinks
    it has), same accumulate-going-forward, dedupe-by-day pattern as
    _record_mcap_snapshot. Logs each sub-signal's own raw numbers, not
    just the combined score, so a future backtest can test them
    individually too. Best-effort: a write failure is swallowed, same
    convention as every other cache write in this module."""
    try:
        existing = _load_bias_history()
        today = pd.Timestamp.utcnow().date()
        tickers_today = {r["ticker"] for r in results}
        if not existing.empty:
            existing = existing[~((existing["date"] == today) & (existing["ticker"].isin(tickers_today)))]

        new_rows = []
        for r in results:
            oi = r["raw"]["oi_price_regime"]
            smd = r["raw"]["smart_money_divergence"]
            cvd = r["raw"]["cvd_price_divergence"]
            price_df = _daily_price_series(r["ticker"], days=1)
            price = float(price_df["close"].iloc[-1]) if not price_df.empty else None
            new_rows.append({
                "date": str(today), "ticker": r["ticker"], "score": r["score"], "label": r["label"],
                "price": price,
                "oi_direction": oi["direction"], "oi_regime": oi["regime"],
                "oi_change_pct": oi["oi_change_pct"], "oi_persistence": oi["persistence"],
                "smd_direction": smd["direction"], "smd_divergence": smd["divergence"],
                "smd_top_ratio": smd["top_ratio"], "smd_global_ratio": smd["global_ratio"],
                "cvd_direction": cvd["direction"], "cvd_regime": cvd["regime"],
                "cvd_net": cvd["cvd_net"], "cvd_price_change_pct": cvd["price_change_pct"],
                "funding_rate": r["funding"]["funding_rate"],
            })

        merged = (pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
                  if not existing.empty else pd.DataFrame(new_rows))
        merged = merged.sort_values(["date", "ticker"]).reset_index(drop=True)
        os.makedirs(_CACHE_DIR, exist_ok=True)
        out = merged.copy()
        out["date"] = out["date"].astype(str)
        with open(_BIAS_HISTORY_PATH, "w") as f:
            json.dump(out.to_dict(orient="records"), f)
    except Exception:
        pass


def get_bias_history():
    """Every daily institutional_bias snapshot recorded so far, across
    every ticker — for a future backtest, or just showing "N days
    recorded" honestly in the meantime. See _load_bias_history's own
    docstring for why this log exists at all."""
    return _load_bias_history()


def scan_watchlist(tickers=CRYPTO_TICKERS):
    """institutional_bias for every ticker in the list, sorted by
    |score| descending — "which ticker has the clearest big-money read
    right now" at a glance, same ranking-table idea as recommender.py's
    scan_watchlist. Recording today's snapshot to the local history log
    is a side effect of every call, not a separate step a caller needs to
    remember — same convention as get_market_cap_snapshot()."""
    results = [institutional_bias(t) for t in tickers]
    _record_bias_snapshot(results)
    return sorted(results, key=lambda r: abs(r["score"]), reverse=True)


# ---------------------------------------------------------------------------
# Total market cap — macro backdrop, not itself a directional signal.
@st.cache_data(ttl=300, show_spinner=False)
def _fetch_global_snapshot():
    resp = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
    resp.raise_for_status()
    return resp.json()["data"]


def _load_mcap_history():
    try:
        with open(_MCAP_HISTORY_PATH) as f:
            rows = json.load(f)
    except Exception:
        return pd.DataFrame(columns=["date", "total_market_cap_usd", "btc_dominance", "eth_dominance"])
    if not rows:
        return pd.DataFrame(columns=["date", "total_market_cap_usd", "btc_dominance", "eth_dominance"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _record_mcap_snapshot(snapshot):
    """Appends today's snapshot to the local history cache, once per day
    (overwriting today's own prior entry if this runs again the same day)
    — same accumulate-going-forward pattern as news.py's calendar history,
    for the same reason: CoinGecko's free tier has no historical total-
    market-cap series to backfill from, so real history here only ever
    covers days this app was actually running to observe it. Best-effort:
    a write failure is swallowed, same convention as news.py's own cache."""
    try:
        existing = _load_mcap_history()
        today = pd.Timestamp.utcnow().date()
        existing = existing[existing["date"] != today]
        new_row = pd.DataFrame([{**snapshot, "date": today}])
        merged = pd.concat([existing, new_row], ignore_index=True).sort_values("date")
        os.makedirs(_CACHE_DIR, exist_ok=True)
        out = merged.copy()
        out["date"] = out["date"].astype(str)
        with open(_MCAP_HISTORY_PATH, "w") as f:
            json.dump(out.to_dict(orient="records"), f)
    except Exception:
        pass


def get_market_cap_snapshot():
    """Current total crypto market cap + BTC/ETH dominance + 24h change,
    or None on a feed outage — same honest-empty-state convention as
    news.py's get_calendar(). Recording to the local history cache is a
    side effect of every successful call, not a separate step a caller
    needs to remember."""
    try:
        data = _fetch_global_snapshot()
    except Exception:
        return None
    snapshot = {
        "total_market_cap_usd": float(data["total_market_cap"]["usd"]),
        "btc_dominance": float(data["market_cap_percentage"].get("btc", 0)),
        "eth_dominance": float(data["market_cap_percentage"].get("eth", 0)),
        "change_24h_pct": float(data.get("market_cap_change_percentage_24h_usd", 0)),
    }
    _record_mcap_snapshot(snapshot)
    return snapshot


def get_market_cap_history():
    """Every daily snapshot this app has ever recorded — empty (or short)
    until enough days pass running this page. Honest limitation, not
    glossed over; see this module's own docstring."""
    return _load_mcap_history()
