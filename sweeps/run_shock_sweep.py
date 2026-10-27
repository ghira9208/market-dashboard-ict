"""Discovery sweep for the news-shock family (shock_strategies.py).

Same machinery and isolation as run_pattern_sweep.py: its OWN ledger
(.cache/shock_sweep_trials.jsonl), so these exploratory trials never touch the main ledger's
Benjamini-Hochberg denominator, and nothing here is promoted automatically.

    python3 sweeps/run_shock_sweep.py [--workers 3] [--tickers GBPUSD=X] [--tfs 5m,1h]
    python3 sweeps/ema_sweep_report.py shock_strategies shock_sweep_trials.jsonl
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

LEDGER = os.path.join(HERE, ".cache", "shock_sweep_trials.jsonl")
TF_FETCH = {"1h": dict(period="730d", interval="1h"), "5m": dict(period="60d", interval="5m")}


def _configure():
    import experiments as ex
    import shock_strategies as ss
    ex._TRIALS_PATH = LEDGER
    ex.DETECTOR_LOGIC_VERSIONS.update(ss.DETECTOR_VERSIONS)
    return ex, ss


def label_for(family, settings):
    parts = " ".join(f"{k}={v}" for k, v in settings.items() if k != "rr_multiple")
    return f"{family}_fixed_rr · {parts} · rr={settings['rr_multiple']} · atr=1.5 · maxbars=60"


def _done_keys():
    done = set()
    if os.path.exists(LEDGER):
        with open(LEDGER) as f:
            for line in f:
                try:
                    t = json.loads(line)
                    done.add((t["ticker"], t["tf_label"], t["label"]))
                except (ValueError, KeyError):
                    pass
    return done


def sweep_ticker(args):
    ticker, tfs = args
    ex, ss = _configure()
    import data as research_data
    done, n = _done_keys(), 0
    for tf in tfs:
        todo = [(fam, s, label_for(fam, s)) for fam, s in ss.settings_grid(tf)
                if tf in ss.TIMEFRAMES[fam] and (ticker, tf, label_for(fam, s)) not in done]
        if not todo:
            continue
        try:
            df = research_data.get_yf_ohlcv(ticker, **TF_FETCH[tf])
        except Exception as e:
            print(f"  {ticker} {tf}: fetch failed ({str(e)[:60]})", flush=True)
            continue
        if df is None or df.empty or len(df) < 500:
            continue
        for fam, settings, label in todo:
            try:
                events = ss.BUILDERS[f"{fam}_fixed_rr"](df, **settings)
                ex.run_deep_backtest_rr_with_lock(ticker, tf, events, settings, label)
                n += 1
            except Exception as e:
                print(f"  {ticker} {tf} {label[:40]}: {str(e)[:70]}", flush=True)
    return ticker, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--tickers", default="")
    ap.add_argument("--tfs", default="5m,1h")
    args = ap.parse_args()
    from universe import universe
    tickers = [t for t in args.tickers.split(",") if t] or universe()
    tfs = [t for t in args.tfs.split(",") if t]
    print(f"{len(tickers)} tickers x {tfs} -> {LEDGER}", flush=True)
    t0, total = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for ticker, n in pool.map(sweep_ticker, [(t, tfs) for t in tickers]):
            total += n
            print(f"[{time.time() - t0:6.0f}s] {ticker:12s} +{n:3d} trials (running total {total})", flush=True)
    print(f"done: {total} new trials in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
