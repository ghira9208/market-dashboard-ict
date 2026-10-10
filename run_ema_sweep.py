"""Discovery sweep for the four EMA families (see ema_strategies.py).

Every setting x ticker x timeframe goes through the SAME statistical machinery
as every other strategy here — run_deep_backtest_rr_with_lock: 60/20/20
RESEARCH / VALIDATION / LOCKED time split, 5000-draw permutation test against
a direction-flipped null, holdout pass + same-sign rule, then Benjamini-
Hochberg across the whole log.

Deliberately written to its OWN ledger (.cache/ema_sweep_trials.jsonl), not
the main experiment_trials.jsonl: ~2,000 exploratory trials added to the main
log would raise its BH denominator and could quietly demote strategies that
are already live, for reasons unrelated to them. Within this ledger the
correction is still honest — its m is exactly the number of tests THIS sweep
ran. Nothing here is promoted automatically; a survivor is a candidate that
still has to be re-validated through the main pipeline.

Resumable: re-running skips (ticker, timeframe, label) rows already logged.

    python3 run_ema_sweep.py [--workers 3] [--tickers EURUSD=X,GC=F] [--tfs 1h,1D]
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SWEEP_LEDGER = os.path.join(HERE, ".cache", "ema_sweep_trials.jsonl")
TF_FETCH = {"1D": dict(period="10y", interval="1d"), "1h": dict(period="730d", interval="1h")}


def _configure():
    import experiments as ex
    import ema_strategies as es
    ex._TRIALS_PATH = SWEEP_LEDGER
    ex.DETECTOR_LOGIC_VERSIONS.update(es.DETECTOR_VERSIONS)
    return ex, es


def label_for(family, settings):
    parts = " ".join(f"{k}={'/'.join(map(str, v)) if isinstance(v, (tuple, list)) else v}"
                     for k, v in settings.items() if k != "rr_multiple")
    return f"{family}_fixed_rr · {parts} · rr={settings['rr_multiple']} · atr=1.5 · maxbars=60"


def _done_keys():
    done = set()
    if os.path.exists(SWEEP_LEDGER):
        with open(SWEEP_LEDGER) as f:
            for line in f:
                try:
                    t = json.loads(line)
                    done.add((t["ticker"], t["tf_label"], t["label"]))
                except (ValueError, KeyError):
                    pass
    return done


def sweep_ticker(args):
    ticker, tfs = args
    ex, es = _configure()
    import data as research_data
    done = _done_keys()
    n = 0
    for tf in tfs:
        grid = [(fam, s, label_for(fam, s)) for fam, s in es.settings_grid()]
        todo = [g for g in grid if (ticker, tf, g[2]) not in done]
        if not todo:
            continue
        try:
            df = research_data.get_yf_ohlcv(ticker, **TF_FETCH[tf])
        except Exception as e:
            print(f"  {ticker} {tf}: fetch failed ({str(e)[:60]})", flush=True)
            continue
        if df is None or df.empty or len(df) < 500:
            print(f"  {ticker} {tf}: not enough data ({0 if df is None else len(df)} bars)", flush=True)
            continue
        for fam, settings, label in todo:
            try:
                events = es.BUILDERS[f"{fam}_fixed_rr"](df, **settings)
                ex.run_deep_backtest_rr_with_lock(ticker, tf, events, settings, label)
                n += 1
            except Exception as e:
                print(f"  {ticker} {tf} {label[:40]}: {str(e)[:70]}", flush=True)
    return ticker, n


def universe():
    """Every ticker the main research ledger has ever scored — the real
    researched universe, read BEFORE this process redirects its ledger."""
    import experiments as ex
    t = ex.load_experiment_trials()
    return sorted(t[t["verdict"] == "SCORED"]["ticker"].unique())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--tickers", default="")
    ap.add_argument("--tfs", default="1h,1D")
    args = ap.parse_args()
    tickers = [t for t in args.tickers.split(",") if t] or universe()
    tfs = [t for t in args.tfs.split(",") if t]
    print(f"{len(tickers)} tickers x {len(tfs)} timeframes x {len(__import__('ema_strategies').settings_grid())} settings "
          f"= up to {len(tickers) * len(tfs) * len(__import__('ema_strategies').settings_grid())} trials -> {SWEEP_LEDGER}", flush=True)
    t0, total = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for ticker, n in pool.map(sweep_ticker, [(t, tfs) for t in tickers]):
            total += n
            print(f"[{time.time() - t0:6.0f}s] {ticker:12s} +{n:3d} trials (running total {total})", flush=True)
    print(f"done: {total} new trials in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
