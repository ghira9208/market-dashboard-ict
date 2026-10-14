"""One-off check: does _simulate_fixed_rr's flat cost_bps=2.0 (experiments.py:3290)
match what fee_engine.py says the bot actually pays per instrument?

_simulate_fixed_rr's own docstring justifies 2.0bps as "~0.9bps real cTrader
commission on FX majors (fee_engine.py) + ~1-2bps assumed spread" but nothing
ever calls fee_engine.py to check that holds for every instrument, or that
0.9bps is still the right anchor for non-FX instruments (futures-CFDs, crypto).
fee_engine.py only models commission (no spread, no swap — see its own
docstring), so this script can only validate the commission slice of the
2.0bps assumption: if real commission alone already exceeds 2.0bps, the
backtest is understated regardless of real spread; if it's comfortably
under, that's inconclusive since spread isn't measured here.

Read-only: does not import or modify experiments.py, bot_engine.py, or any
sizing/execution code.
"""
import sys

sys.path.insert(0, "/Users/apple/dev/edge-pipeline")

import brokers  # noqa: E402
import fee_engine  # noqa: E402
from brokers.base import BrokerAuthError  # noqa: E402
from brokers.ctrader import TICKER_SYMBOL  # noqa: E402
from dashboard_data import DEFAULT_TICKERS  # noqa: E402

BACKTEST_COST_BPS = 2.0  # mirrors experiments.py's _simulate_fixed_rr default (duplicated
# intentionally — this is a one-off check, not a pipeline dependency)

REPRESENTATIVE_NOTIONALS_USD = [500, 1500, 5000]  # brackets this account's real scale
# (STARTING_BALANCE=1000, half-Kelly risk capped ~4%/trade — portfolio_backtest.py)

FLAG_THRESHOLD_PCT = 50.0

# Doc-verified real values pinned in tests/test_fee_engine.py — used only when the
# live cTrader bridge (localhost:8700) isn't reachable. Every other instrument is
# reported as NO LIVE DATA rather than guessed, per fee_engine.py's own convention.
_FALLBACK_FEE_INFO = {
    "GBPUSD": {
        "commissionType": "USD_PER_MILLION_USD",
        "preciseTradingCommissionRate": 4500000000,
        "_source": "tests/test_fee_engine.py (doc-verified, not live)",
    },
    "BTCUSD": {
        "commissionType": "USD_PER_MILLION_USD",
        "preciseTradingCommissionRate": 0,
        "_source": "tests/test_fee_engine.py (doc-verified, not live)",
    },
    "ETHUSD": {
        "commissionType": "USD_PER_MILLION_USD",
        "preciseTradingCommissionRate": 0,
        "_source": "tests/test_fee_engine.py (doc-verified, not live)",
    },
}


def _get_live_broker():
    try:
        broker = brokers.build_broker("dev_user", "ctrader")
    except BrokerAuthError:
        return None
    return broker


def _fee_info_for(broker, symbol):
    """Returns (fee_info, source_label) or (None, reason)."""
    if broker is not None:
        try:
            info = broker.symbol_fee_info(symbol)
        except BrokerAuthError:
            info = None
        except Exception:
            info = None
        if info is not None:
            return info, "live"
    fallback = _FALLBACK_FEE_INFO.get(symbol)
    if fallback is not None:
        return fallback, "fallback"
    return None, "NO LIVE DATA"


def _relative_diff_pct(real_bps, assumed_bps):
    if assumed_bps == 0:
        return None
    return (real_bps - assumed_bps) / assumed_bps * 100.0


def main():
    broker = _get_live_broker()
    bridge_status = "reachable" if broker is not None else "NOT reachable (falling back to doc-verified values)"
    print(f"cTrader bridge: {bridge_status}\n")

    header = (
        f"{'ticker':<10} {'symbol':<14} {'source':<9} {'commission_bps':>14} "
        + " ".join(f"bps@${n:<6}" for n in REPRESENTATIVE_NOTIONALS_USD)
        + f" {'vs_2.0bps_%diff':>16}  flag"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for ticker in sorted(DEFAULT_TICKERS):
        symbol = TICKER_SYMBOL.get(ticker)
        if symbol is None:
            print(f"{ticker:<10} {'(none)':<14} {'—':<9} no broker symbol mapping for this ticker")
            continue

        fee_info, source = _fee_info_for(broker, symbol)
        if fee_info is None:
            print(f"{ticker:<10} {symbol:<14} {'—':<9} NO LIVE DATA — not fabricated")
            continue

        comm_pct = fee_engine.commission_pct_roundtrip(fee_info)
        if comm_pct is None:
            print(f"{ticker:<10} {symbol:<14} {source:<9} unsupported commissionType "
                  f"({fee_info.get('commissionType')!r}) — fee_engine returns None, not a guess")
            continue
        comm_bps = comm_pct * 100.0

        dollar_bps_by_notional = []
        for notional in REPRESENTATIVE_NOTIONALS_USD:
            dollars = fee_engine.commission_dollars_roundtrip(fee_info, notional)
            dollar_bps_by_notional.append(None if dollars is None else dollars / notional * 10_000.0)

        # Worst-case (largest) bps across representative sizes is what actually
        # exercises the minimum-commission floor at this account's smaller sizes.
        worst_bps = max([b for b in dollar_bps_by_notional if b is not None] + [comm_bps])
        rel_diff = _relative_diff_pct(worst_bps, BACKTEST_COST_BPS)
        flagged = rel_diff is not None and abs(rel_diff) > FLAG_THRESHOLD_PCT

        rows.append({
            "ticker": ticker, "symbol": symbol, "source": source,
            "comm_bps": comm_bps, "dollar_bps_by_notional": dollar_bps_by_notional,
            "worst_bps": worst_bps, "rel_diff": rel_diff, "flagged": flagged,
        })

        dollar_cols = " ".join(
            f"{'n/a':>9}" if b is None else f"{b:>9.2f}" for b in dollar_bps_by_notional
        )
        flag_str = "<<< DIVERGES" if flagged else ""
        rel_str = "n/a" if rel_diff is None else f"{rel_diff:+.0f}%"
        print(f"{ticker:<10} {symbol:<14} {source:<9} {comm_bps:>14.3f} {dollar_cols} "
              f"{rel_str:>16}  {flag_str}")

    print()
    understated = [r for r in rows if r["flagged"] and r["worst_bps"] > BACKTEST_COST_BPS]
    overstated = [r for r in rows if r["flagged"] and r["worst_bps"] <= BACKTEST_COST_BPS]

    if understated:
        print(f"CRITICAL — real commission ALONE exceeds the {BACKTEST_COST_BPS}bps assumption "
              "(backtest understates real cost regardless of unmeasured spread):")
        for r in understated:
            print(f"  - {r['ticker']} ({r['symbol']}): real commission ~{r['worst_bps']:.2f}bps "
                  f"vs. assumed {BACKTEST_COST_BPS}bps ({r['rel_diff']:+.0f}%)")
    else:
        print("No instrument's real commission alone exceeded the flat 2.0bps assumption.")

    if overstated:
        print(f"\nBelow assumption by >{FLAG_THRESHOLD_PCT:.0f}% (commission-only — inconclusive on "
              "its own, since real spread isn't measured here and could still close the gap):")
        for r in overstated:
            print(f"  - {r['ticker']} ({r['symbol']}): real commission ~{r['worst_bps']:.2f}bps "
                  f"vs. assumed {BACKTEST_COST_BPS}bps ({r['rel_diff']:+.0f}%)")

    print("\nCaveat: fee_engine.py models COMMISSION only (no spread, no swap). A real-commission "
          "bps comfortably under 2.0bps does NOT confirm the 2.0bps total is correct — only that "
          "commission alone doesn't blow through it. A real-commission bps already OVER 2.0bps is "
          "the only unambiguous signal here: the backtest's cost assumption is understated for "
          "that instrument regardless of real spread.")


if __name__ == "__main__":
    main()
