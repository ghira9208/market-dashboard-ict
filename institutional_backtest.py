"""
Backtests whether institutional.py's own signals actually predict forward
returns — not just "look directionally reasonable today." This is the
follow-through on that question, not a new statistical framework: it
reuses research/evidence.py's run_event_study (the exact permutation-test
engine every ICT hypothesis in this project is held to) and
edge_lab/multiple_testing's Benjamini-Hochberg correction, applied to a
SEPARATE trial universe from edge_lab's own GBPUSD log — mixing crypto-
institutional trials into that log would silently change what "how many
things were tried" means for every FX trial already logged there (see
edge_lab/agent.py's own docstring on this exact point; the same reasoning
applies here in reverse).

Two genuinely different trial classes, because the underlying signals have
genuinely different amounts of real history available — reported
separately, never blended, so a reader can't mistake one for the other:

  1. "cvd" trials — cvd_price_divergence tested against Binance's ordinary
     kline history, which goes back YEARS (no artificial cap). Real
     train/holdout split (80/20 by time), same as edge_lab/agent.py.
     Adequately powered; a genuine finding here would mean something.

  2. "combined_score" trials — the full 3-signal institutional_bias score
     (OI regime + smart-money divergence + CVD divergence together)
     tested against the ONLY historical window that exists for it right
     now: Binance's /futures/data/* endpoints are hard-capped at 30 days
     regardless of what's requested (confirmed directly — see
     institutional.py's own docstring), leaving ~15-20 usable days per
     ticker after lookback/forward trimming. Too small to hold out a
     slice and still have anything left, so these run train-only, no
     holdout claim, and every printed line says LOW-POWER. This class
     will only get more powerful by waiting: institutional.py's own
     accumulating daily log (.cache/institutional_bias_history.json,
     started this session) is the actual fix, not anything this script
     can do today. Once that log has enough independent days, this
     script should switch to reading it (see _combined_score_events —
     it already tries the log first) instead of re-deriving the same
     capped 30-day snapshot every run forever.

Every trial — both classes, every parameter tried — goes into ONE
Benjamini-Hochberg correction, because that's the honest answer to "how
many things did we try." No result here is used to justify running the
grid again with different parameters; that would be exactly the p-hacking
this project's own standards (and this script's own correction step)
exist to catch. Whatever survives is reported. If nothing does, that's
the honest result and it's reported exactly as plainly.

Run directly: python3 -m institutional_backtest
"""
import numpy as np
import pandas as pd

import institutional as inst
from edge_lab.multiple_testing import benjamini_hochberg
from research.evidence import run_event_study

# Binance USDT-M perpetual taker fee is ~4-5bps per side (~8-10bps round
# trip) on the majors this watchlist covers; a flat, conservative
# assumption rather than per-ticker estimated (unlike research/spread.py's
# FX estimator, which needs high/low columns this module's own kline fetch
# doesn't carry) — deliberately on the pessimistic side, not tuned to help
# any particular trial clear significance.
_CRYPTO_COST_BPS = 15.0
_TRAIN_FRACTION = 0.8
_N_PERMUTATIONS = 2000
_CVD_LOOKBACK = 7  # matches institutional.cvd_price_divergence's own default
_CVD_FORWARD_BARS_OPTIONS = [3, 5, 10]
_COMBINED_LOOKBACK = 7  # matches institutional.oi_price_regime's own default
_COMBINED_FORWARD_BARS_OPTIONS = [3, 5]
_MIN_LOG_EVENTS_PER_TICKER = 15  # below this, the accumulating log isn't ready yet — fall back


def _cvd_events_for_ticker(ticker, lookback=_CVD_LOOKBACK, forward_bars=5):
    """events_df (entry_time, raw_return, direction) for every qualifying
    bar in this ticker's FULL available daily kline history — same shape
    research/events.py's own extract_*_events functions produce, so
    run_event_study scores it unmodified. direction comes from
    institutional._classify_cvd — the IDENTICAL rule the live page uses,
    not a hand-copied second version. raw_return is this ticker's own
    actual forward return over the next forward_bars days, computed
    straight from the same kline closes."""
    df = inst.get_taker_volume_history(ticker, interval="1d", limit=1500)
    n = len(df)
    if n < lookback + forward_bars + 20:
        return pd.DataFrame(columns=["entry_time", "raw_return", "direction"])

    close = df["close"].to_numpy()
    delta = df["delta"].to_numpy()
    times = df["time"].to_numpy()

    rows = []
    for i in range(lookback, n - forward_bars):
        cvd_net = float(delta[i - lookback + 1: i + 1].sum())
        price_chg_pct = float((close[i] - close[i - lookback]) / close[i - lookback] * 100)
        _, direction = inst._classify_cvd(cvd_net, price_chg_pct)
        fwd_return = float((close[i + forward_bars] - close[i]) / close[i])
        rows.append({"entry_time": times[i], "raw_return": fwd_return,
                     "direction": "bullish" if direction > 0 else "bearish"})
    return pd.DataFrame(rows)


def _combined_score_events_from_log(ticker, forward_bars):
    """Prefers institutional.py's own accumulating daily log
    (.cache/institutional_bias_history.json) over re-deriving the same
    capped 30-day Binance snapshot every run — this is what lets the
    combined-score trial class actually gain power over calendar time
    instead of being stuck at ~20 days forever. Returns empty until the
    log has enough independent days for this ticker; see the module
    docstring for why that's expected today, not a bug."""
    hist = inst.get_bias_history()
    if hist.empty:
        return pd.DataFrame(columns=["entry_time", "raw_return", "direction"])
    sub = hist[(hist["ticker"] == ticker) & (hist["score"] != 0) & hist["price"].notna()]
    sub = sub.sort_values("date").reset_index(drop=True)
    if len(sub) < _MIN_LOG_EVENTS_PER_TICKER + forward_bars:
        return pd.DataFrame(columns=["entry_time", "raw_return", "direction"])

    price = sub["price"].to_numpy()
    dates = pd.to_datetime(sub["date"]).to_numpy()
    score = sub["score"].to_numpy()
    rows = []
    for i in range(len(sub) - forward_bars):
        fwd_return = float((price[i + forward_bars] - price[i]) / price[i])
        rows.append({"entry_time": dates[i], "raw_return": fwd_return,
                     "direction": "bullish" if score[i] > 0 else "bearish"})
    return pd.DataFrame(rows)


def _combined_score_events_live(ticker, lookback=_COMBINED_LOOKBACK, forward_bars=3):
    """Fallback when the accumulating log isn't ready yet (today): the
    full 3-signal combined score, recomputed at each day within the ONLY
    historical window Binance's OI/ratio endpoints actually serve right
    now (~25-30 days), using the IDENTICAL pure classifiers the live page
    calls — never a hand-copied second version. LOW-POWER BY
    CONSTRUCTION — see the module docstring."""
    oi_df = inst.get_open_interest_history(ticker, limit=30)
    top_df = inst.get_top_trader_ratio(ticker, limit=30)
    global_df = inst.get_global_ratio(ticker, limit=30)
    cvd_df = inst.get_taker_volume_history(ticker, interval="1d", limit=30)
    if oi_df.empty or top_df.empty or global_df.empty or cvd_df.empty:
        return pd.DataFrame(columns=["entry_time", "raw_return", "direction"])

    oi_df = oi_df.assign(date=oi_df["time"].dt.date)
    top_df = top_df.assign(date=top_df["time"].dt.date)[["date", "long_short_ratio"]].rename(
        columns={"long_short_ratio": "top_ratio"})
    global_df = global_df.assign(date=global_df["time"].dt.date)[["date", "long_short_ratio"]].rename(
        columns={"long_short_ratio": "global_ratio"})
    cvd_df = cvd_df.assign(date=cvd_df["time"].dt.date)[["date", "close", "delta"]]

    merged = (oi_df[["date", "oi"]].merge(top_df, on="date", how="inner")
              .merge(global_df, on="date", how="inner").merge(cvd_df, on="date", how="inner")
              .sort_values("date").reset_index(drop=True))
    n = len(merged)
    if n < lookback + forward_bars + 5:
        return pd.DataFrame(columns=["entry_time", "raw_return", "direction"])

    oi = merged["oi"].to_numpy()
    close = merged["close"].to_numpy()
    delta = merged["delta"].to_numpy()
    top_ratio = merged["top_ratio"].to_numpy()
    global_ratio = merged["global_ratio"].to_numpy()
    dates = merged["date"].to_numpy()

    rows = []
    for i in range(lookback, n - forward_bars):
        oi_chg_pct = float((oi[i] - oi[i - lookback]) / oi[i - lookback] * 100)
        price_chg_pct = float((close[i] - close[i - lookback]) / close[i - lookback] * 100)
        _, oi_dir = inst._classify_oi_regime(oi_chg_pct, price_chg_pct)
        smd_dir, _ = inst._classify_smart_money(float(top_ratio[i]), float(global_ratio[i]))
        cvd_net = float(delta[i - lookback + 1: i + 1].sum())
        _, cvd_dir = inst._classify_cvd(cvd_net, price_chg_pct)
        score = oi_dir + smd_dir + cvd_dir
        if score == 0:
            continue
        fwd_return = float((close[i + forward_bars] - close[i]) / close[i])
        rows.append({"entry_time": pd.Timestamp(dates[i]), "raw_return": fwd_return,
                     "direction": "bullish" if score > 0 else "bearish"})
    return pd.DataFrame(rows)


def _direction_balance(events_df):
    """Fraction of events on the MINORITY side of bullish/bearish — low
    means the classifier is barely conditioning on anything day-to-day
    (near-constant one direction), which lets a multi-year regime (not the
    signal itself) drive the whole result; see _run_cvd_trial's own
    same_sign comment for a real example this caught."""
    counts = events_df["direction"].value_counts()
    if len(counts) < 2:
        return 0.0, counts.to_dict()
    return float(counts.min() / counts.sum()), counts.to_dict()


def _run_cvd_trial(ticker, forward_bars):
    events_df = _cvd_events_for_ticker(ticker, forward_bars=forward_bars)
    label = f"CVD · {ticker} · {forward_bars}d fwd"
    if len(events_df) < 40:
        return {"trial_class": "cvd", "label": label, "ticker": ticker, "forward_bars": forward_bars,
                "n_events": len(events_df), "verdict": "INSUFFICIENT_DATA"}
    balance, direction_counts = _direction_balance(events_df)

    cutoff_pos = int(len(events_df) * _TRAIN_FRACTION)
    cutoff = events_df["entry_time"].iloc[cutoff_pos]
    train = events_df[events_df["entry_time"] < cutoff]
    holdout = events_df[events_df["entry_time"] >= cutoff]
    train_result = run_event_study(train, cost_bps=_CRYPTO_COST_BPS, direction_col="direction",
                                    n_permutations=_N_PERMUTATIONS)
    holdout_result = run_event_study(holdout, cost_bps=_CRYPTO_COST_BPS, direction_col="direction",
                                      n_permutations=_N_PERMUTATIONS)

    n_train = train_result.get("n_events", 0)
    if n_train == 0:
        return {"trial_class": "cvd", "label": label, "ticker": ticker, "forward_bars": forward_bars,
                "n_events": len(events_df), "verdict": "INSUFFICIENT_DATA"}

    p_train = train_result.get("p_value_vs_random_direction")
    mean_train = train_result.get("mean_return_after_cost")
    mean_holdout = holdout_result.get("mean_return_after_cost")
    # bh_significant (below, in run_backtest) only ever checks p_value_train's
    # MAGNITUDE — a strongly-significant NEGATIVE train mean (the signal
    # reliably backwards, not absent) clears that bar exactly as easily as a
    # positive one. edge_lab/agent.py's own holdout_verdict has the same gap
    # in isolation (EDGE_FOUND only requires holdout's OWN mean > 0, nothing
    # ties it to train's sign) — normally harmless since a real signal keeps
    # its sign, but caught directly here: DOGE-USD's CVD classifier calls
    # "bearish" ~98% of the time (it's barely conditioning on daily CVD at
    # all), so train (a mostly-up multi-year window) and holdout (a mostly-
    # down one) land on OPPOSITE signs by pure regime luck, and both
    # independently cleared their own threshold. "same_sign" below is the
    # extra check that catches exactly this — required for "survived" in
    # run_backtest, not just p<0.05 on both sides.
    same_sign = bool(mean_train is not None and mean_holdout is not None
                      and mean_train > 0 and mean_holdout > 0)
    row = {
        "trial_class": "cvd", "label": label, "ticker": ticker, "forward_bars": forward_bars,
        "n_events": len(events_df), "n_train": n_train, "n_holdout": holdout_result.get("n_events", 0),
        "mean_return_train": mean_train,
        "p_value_train": p_train,
        "train_significant": bool(p_train is not None and p_train < 0.05),
        "holdout_verdict": "PASSED" if holdout_result.get("verdict") == "EDGE_FOUND" else "FAILED",
        "mean_return_holdout": mean_holdout,
        "same_sign": same_sign,
        "direction_balance": balance, "direction_counts": direction_counts,
        "verdict": "SCORED",
    }
    return row


def _run_combined_trial(ticker, forward_bars):
    label = f"Combined score · {ticker} · {forward_bars}d fwd"
    events_df = _combined_score_events_from_log(ticker, forward_bars)
    source = "log"
    if len(events_df) < 20:
        events_df = _combined_score_events_live(ticker, forward_bars=forward_bars)
        source = "live_30day_snapshot"

    if len(events_df) < 8:
        return {"trial_class": "combined_score", "label": label, "ticker": ticker,
                "forward_bars": forward_bars, "source": source, "n_events": len(events_df),
                "verdict": "INSUFFICIENT_DATA"}

    balance, direction_counts = _direction_balance(events_df)
    # Too little data to hold out a slice and have anything left — train-only,
    # no holdout claim, flagged LOW-POWER regardless of what the p-value says.
    result = run_event_study(events_df, cost_bps=_CRYPTO_COST_BPS, direction_col="direction",
                              n_permutations=_N_PERMUTATIONS)
    p_train = result.get("p_value_vs_random_direction")
    return {
        "trial_class": "combined_score", "label": label, "ticker": ticker, "forward_bars": forward_bars,
        "source": source, "n_events": len(events_df),
        "mean_return_train": result.get("mean_return_after_cost"),
        "p_value_train": p_train,
        "train_significant": bool(p_train is not None and p_train < 0.05),
        "holdout_verdict": "NOT_APPLICABLE (LOW-POWER, no holdout split)",
        "direction_balance": balance, "direction_counts": direction_counts,
        "verdict": "SCORED",
        "low_power": True,
    }


def run_backtest(tickers=inst.CRYPTO_TICKERS):
    """Runs every trial described in this module's own docstring, applies
    ONE Benjamini-Hochberg correction across all of them together, and
    returns (trials, summary) — trials is every trial attempted (pass or
    fail, same "log everything, decide nothing in advance" convention as
    edge_lab/agent.py), summary is the printable verdict."""
    trials = []
    for ticker in tickers:
        for fb in _CVD_FORWARD_BARS_OPTIONS:
            trials.append(_run_cvd_trial(ticker, fb))
        for fb in _COMBINED_FORWARD_BARS_OPTIONS:
            trials.append(_run_combined_trial(ticker, fb))

    scored = [t for t in trials if t["verdict"] == "SCORED"]
    if scored:
        p_values = [t["p_value_train"] for t in scored]
        q_values, significant = benjamini_hochberg(p_values, alpha=0.05)
        for t, q, sig in zip(scored, q_values, significant):
            t["q_value_train"] = float(q)
            t["bh_significant"] = bool(sig)

    survived = [t for t in scored if t.get("bh_significant") and t["holdout_verdict"] == "PASSED"
                and t.get("same_sign")]
    low_power_hits = [t for t in scored if t.get("bh_significant") and t.get("low_power")]

    summary = {
        "n_trials": len(trials),
        "n_scored": len(scored),
        "n_insufficient_data": len(trials) - len(scored),
        "n_bh_significant": sum(1 for t in scored if t.get("bh_significant")),
        "n_survived_with_holdout": len(survived),
        "n_low_power_hits": len(low_power_hits),
        "survived": survived,
        "low_power_hits": low_power_hits,
    }
    return trials, summary


if __name__ == "__main__":
    print("Running institutional-signal backtest — CVD (full history, train/holdout) "
          f"+ combined score (log if ready, else live 30-day snapshot) across {len(inst.CRYPTO_TICKERS)} tickers...")
    trials, summary = run_backtest()

    print(f"\n{summary['n_trials']} trials attempted, {summary['n_scored']} scored, "
          f"{summary['n_insufficient_data']} had insufficient data.")
    print(f"{summary['n_bh_significant']} cleared Benjamini-Hochberg correction on their TRAIN split.")
    print(f"Of those: {summary['n_survived_with_holdout']} also passed a real holdout split (the CVD class only — "
          f"this is the only number here that would mean 'a real, out-of-sample-checked edge').")
    print(f"{summary['n_low_power_hits']} combined-score hits cleared correction but have NO holdout check "
          "(too little data to split) — reported, not trusted; see the module docstring.")

    if summary["survived"]:
        print("\nSurvived (BH-significant, holdout-passed, AND train/holdout agree on sign):")
        for t in summary["survived"]:
            print(f"  {t['label']}: p={t['p_value_train']:.4f} q={t['q_value_train']:.4f} "
                  f"train_mean={t['mean_return_train']:+.4%} holdout_mean={t['mean_return_holdout']:+.4%} "
                  f"(n_train={t['n_train']}, n_holdout={t['n_holdout']}) "
                  f"direction_split={t['direction_counts']}")
    else:
        print("\nNothing survived BH correction with a real holdout check that also agrees with train on "
              "sign. That's the honest result of this run, not a reason to change parameters and try again — "
              "see the module docstring on why doing that would be p-hacking, not more rigor.")

    if summary["low_power_hits"]:
        print("\nLOW-POWER combined-score hits (train-only, no holdout, small n — informational, not proof):")
        for t in summary["low_power_hits"]:
            print(f"  {t['label']} [{t['source']}]: p={t['p_value_train']:.4f} q={t['q_value_train']:.4f} "
                  f"mean={t['mean_return_train']:+.4%} (n={t['n_events']}) direction_split={t['direction_counts']} "
                  f"minority_share={t['direction_balance']:.0%}")
