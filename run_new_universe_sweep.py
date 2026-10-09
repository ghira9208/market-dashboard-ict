"""New-universe sweep: the 73 real, cTrader-tradeable instruments found
this session that this project's research has never touched — confirmed
directly (not guessed) in two steps: (1) pulled this account's real
/symbols list off the cTrader bridge and classified it into FX crosses,
metals, energy, and major international indices (filtering out ~420
individual stock names that showed up in the same list); (2) tested every
candidate against this project's own research_data.get_yf_ohlcv — 73 of
74 came back with real, usable history (only NZDSEK=X failed). None of
these 73 have ever been run through any _fixed_rr strategy before.

Same exact 27-combo settings grid and 3 canonical timeframes (1D/1h/5m)
run_ticker_sweep.py already established for this platform's own research
— reused here unchanged, not reinvented — against all THREE _fixed_rr
strategies (run_ticker_sweep.py only needed two of them; clean_expansion_
fixed_rr is included here since none of these 73 tickers have it either).

Skips any (strategy, ticker, tf_label, label) triple already in the trial
log — safe to re-run. Appends to .cache/experiment_trials.jsonl, then
rebuilds ticker_behavior.db so anything that survives BH-correction and
holdout shows up on the real leaderboard automatically.

Run standalone: python3 run_new_universe_sweep.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import experiments as ex
import ticker_behavior
import data as research_data

TICKERS = [
    # Metals / energy — real cTrader symbols confirmed, yfinance data confirmed
    "PA=F", "PL=F", "BZ=F",
    # Major international indices
    "^FTSE", "^GDAXI", "^FCHI", "FTSEMIB.MI", "^IBEX", "^SSMI",
    "^AEX", "^AXJO", "^HSI", "^STOXX50E",
    # FX crosses this broker carries beyond the 6 already traded
    "AUDCAD=X", "AUDCHF=X", "AUDJPY=X", "AUDNZD=X", "AUDSGD=X", "AUDZAR=X",
    "CADCHF=X", "CADJPY=X", "CADMXN=X", "CHFJPY=X", "CHFSGD=X",
    "EURAUD=X", "EURCAD=X", "EURCHF=X", "EURCNH=X", "EURCZK=X", "EURDKK=X",
    "EURGBP=X", "EURHKD=X", "EURHUF=X", "EURJPY=X", "EURMXN=X", "EURNOK=X",
    "EURNZD=X", "EURPLN=X", "EURSEK=X", "EURSGD=X", "EURTRY=X", "EURZAR=X",
    "GBPAUD=X", "GBPCAD=X", "GBPCHF=X", "GBPDKK=X", "GBPJPY=X", "GBPNOK=X",
    "GBPNZD=X", "GBPSEK=X", "GBPSGD=X", "GBPTRY=X", "GBPZAR=X",
    "NOKJPY=X", "NOKSEK=X", "NZDCAD=X", "NZDCHF=X", "NZDJPY=X", "NZDSGD=X",
    "NZDUSD=X", "SEKJPY=X", "SGDJPY=X", "USDCNH=X", "USDCZK=X", "USDDKK=X",
    "USDHUF=X", "USDMXN=X", "USDNOK=X", "USDPLN=X", "USDSEK=X", "USDSGD=X",
    "USDTRY=X", "USDZAR=X",
]
assert len(TICKERS) == 73

TF_FETCH = {
    "1D": dict(period="10y", interval="1d"),
    "1h": dict(period="730d", interval="1h"),
    "5m": dict(period="60d", interval="5m"),
}

STRATEGIES = {
    "clean_expansion_fixed_rr": {
        "builder": ex.build_clean_expansion_fixed_rr_events,
        "fixed": {"min_streak": 3, "retr_window_bars": 40},
    },
    "clean_expansion_liquidity_fixed_rr": {
        "builder": ex.build_clean_expansion_liquidity_fixed_rr_events,
        "fixed": {"min_streak": 3, "retr_window_bars": 40, "sweep_window_bars": 10},
    },
    "clean_retracement_resumption_fixed_rr": {
        "builder": ex.build_clean_retracement_resumption_fixed_rr_events,
        "fixed": {"min_streak": 3, "retr_window_bars": 40},
    },
}

# Identical grid to run_ticker_sweep.py's own — not reinvented, reused.
_ATR_RR_MAXBARS = (
    [(atr, rr, 40) for atr in (0.75, 1.0, 1.5) for rr in (0.3, 0.5, 0.75, 1.0, 1.5)]
    + [(atr, rr, 60) for atr in (1.5, 2.0, 2.5) for rr in (1.5, 2.0, 2.5, 3.0)]
)
assert len(_ATR_RR_MAXBARS) == 27


def _label(strategy, atr, rr, win, maxbars):
    base = f"{strategy} · rr={rr} · atr={atr} · entry(win={win}b)"
    return base if maxbars == 40 else f"{base} · maxbars={maxbars}"


def _already_tried():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "experiment_trials.jsonl")
    seen = set()
    if not os.path.exists(path):
        return seen
    with open(path) as f:
        for line in f:
            try:
                t = json.loads(line)
            except Exception:
                continue
            label = t.get("label") or ""
            strat = next((s for s in STRATEGIES if label.startswith(s)), None)
            if strat is None:
                continue
            tf = t.get("tf_label") or ""
            tf = "1D" if tf.lower() == "1d" else tf
            seen.add((strat, t.get("ticker"), tf, label))
    return seen


def main():
    seen = _already_tried()
    print(f"{len(seen)} (strategy, ticker, tf, settings) combos already in the trial log — skipping those.")

    df_cache = {}
    n_run, n_skipped, n_scored, n_passed, n_fetch_failed = 0, 0, 0, 0, 0
    t0 = time.time()

    for strategy, spec in STRATEGIES.items():
        builder, fixed = spec["builder"], spec["fixed"]
        for ticker in TICKERS:
            for tf_label, fetch_kwargs in TF_FETCH.items():
                cache_key = (ticker, tf_label)
                if cache_key not in df_cache:
                    try:
                        df_cache[cache_key] = research_data.get_yf_ohlcv(ticker, **fetch_kwargs)
                    except Exception as e:
                        print(f"  fetch failed {ticker} {tf_label}: {e}")
                        df_cache[cache_key] = None
                        n_fetch_failed += 1
                df = df_cache[cache_key]
                if df is None or df.empty:
                    continue

                for atr, rr, maxbars in _ATR_RR_MAXBARS:
                    settings = {**fixed, "atr_mult_stop": atr, "rr_multiple": rr, "max_bars": maxbars}
                    label = _label(strategy, atr, rr, fixed["retr_window_bars"], maxbars)
                    if (strategy, ticker, tf_label, label) in seen:
                        n_skipped += 1
                        continue
                    try:
                        events = builder(df, **settings)
                    except Exception as e:
                        print(f"  builder failed {strategy} {ticker} {tf_label} {settings}: {e}")
                        continue
                    trial = ex.run_deep_backtest_rr(ticker, tf_label, events, settings, label)
                    n_run += 1
                    if trial.get("verdict") == "SCORED":
                        n_scored += 1
                        if trial.get("holdout_verdict") == "PASSED" and trial.get("same_sign"):
                            n_passed += 1
                            print(f"  PASSED  {ticker:10s} {tf_label:4s} {label}")

                elapsed = time.time() - t0
                print(f"[{elapsed:6.0f}s] {strategy:38s} {ticker:10s} {tf_label:4s} "
                      f"done (run={n_run} skipped={n_skipped} scored={n_scored} passed={n_passed})")

    print(f"\nDone in {time.time()-t0:.0f}s. New trials run: {n_run}, scored: {n_scored}, "
          f"passed (candidate-validated): {n_passed}. Fetch failures: {n_fetch_failed}.")
    print("Rebuilding ticker_behavior.db ...")
    ticker_behavior.rebuild_db()
    print("Done.")


if __name__ == "__main__":
    main()
