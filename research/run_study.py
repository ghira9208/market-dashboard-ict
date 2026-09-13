"""
CLI entry point for the research pipeline's first end-to-end slice:

    python3 -m research.run_study --ticker BTC-USD --interval 15m --period 60d

Wires data_loader -> events -> backtest and prints an honest verdict. Run
from the project root (or as -m research.run_study from anywhere, since
data_loader.py puts the project root on sys.path itself).
"""

import argparse

from research.evidence import run_event_study
from research.data_loader import load_history
from research.events import extract_fvg_retracement_events
from research.experiment_log import log_run


def main():
    p = argparse.ArgumentParser(description="FVG-retracement event study.")
    p.add_argument("--ticker", default="BTC-USD")
    p.add_argument("--interval", default="15m")
    p.add_argument("--period", default="60d")
    p.add_argument("--provider", default="auto")
    p.add_argument("--forward-bars", type=int, default=10, help="Bars ahead to measure return over.")
    p.add_argument("--cost-bps", type=float, default=10.0, help="Assumed round-trip cost, in basis points.")
    p.add_argument("--refresh", action="store_true", help="Force a fresh data pull instead of using the disk cache.")
    p.add_argument("--news-blackout-before", type=int, default=None,
                    help="Exclude events whose entry falls within this many minutes BEFORE a high-impact news "
                         "release for this ticker's own currencies. Requires --news-blackout-after too.")
    p.add_argument("--news-blackout-after", type=int, default=None,
                    help="Same as --news-blackout-before, minutes AFTER the release.")
    args = p.parse_args()
    news_blackout = ((args.news_blackout_before, args.news_blackout_after)
                      if args.news_blackout_before is not None and args.news_blackout_after is not None else None)

    print(f"Loading {args.ticker} {args.interval} ({args.period}, provider={args.provider})...")
    df = load_history(args.ticker, args.period, args.interval, provider=args.provider, refresh=args.refresh)
    print(f"  {len(df)} bars, {df.index[0]} -> {df.index[-1]}")

    events = extract_fvg_retracement_events(df, forward_bars=args.forward_bars)
    print(f"Found {len(events)} FVG-retracement events (forward_bars={args.forward_bars}).")

    result = run_event_study(events, cost_bps=args.cost_bps, ticker=args.ticker, news_blackout=news_blackout)
    log_run(args.ticker, args.interval, args.period, args.provider, "FVG retracement -> continuation",
            args.forward_bars, args.cost_bps, {}, result)

    print("\n--- Event study result ---")
    print(f"n_events:                 {result.get('n_events')}")
    if result.get("n_events"):
        print(f"cost assumed (round trip): {result['cost_bps']:.1f} bps")
        print(f"mean return after cost:    {result['mean_return_after_cost']*100:.3f}%")
        print(f"win rate after cost:       {result['win_rate_after_cost']*100:.1f}%")
        print(f"p-value vs random direction: {result['p_value_vs_random_direction']:.4f}")
        for t, stats in result.get("by_type", {}).items():
            print(f"  {t:8s} n={stats['n']:4d}  win_rate={stats['win_rate']*100:.1f}%  "
                  f"mean_ret={stats['mean_return_after_cost']*100:.3f}%")
    print(f"\nVERDICT: {result['verdict']}")
    if result.get("verdict") == "NO_EDGE":
        print("(p >= 0.05, or mean return after cost <= 0 — the direction label isn't "
              "adding anything a coin flip wouldn't, at this cost assumption.)")


if __name__ == "__main__":
    main()
