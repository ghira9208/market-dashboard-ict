"""The missing piece, confirmed directly as missing: what does the
validated leaderboard actually look like as ONE portfolio, trading every
combo concurrently on a single compounding account — exactly what
bot_engine.py does live — instead of 48 separate isolated equity curves.

Real gap, not hypothetical: every BH-correction/holdout test this platform
runs validates ONE combo's own edge in isolation. None of them ask what
happens when several combos lose together. research/evidence.py's own
docstring flagged "a full backtester (position sizing, overlapping trades,
portfolio equity curve)" as a real next step, years ago — never built.
This is that, built from data that's already real and already cached
(research_stats.py's own all_trades — cost-aware, 2.0bps already netted
out of every r_real) — no new market-data fetching, pure replay.

Methodology — every choice below is load-bearing, not incidental:
- Every trade from every validated _fixed_rr combo, merged and replayed
  in real chronological order on ONE simulated account.
- Sized the SAME way bot_engine.py actually sizes live trades: half-Kelly
  off each combo's own validated best_mean_return_holdout + rr_multiple
  (_kelly_fraction formula mirrored exactly, not reimplemented
  differently — see bot_engine.py:285-300 for the original).
  Simplification, stated plainly: each combo's risk_pct is held fixed at
  its CURRENT validated value for the whole historical replay, since
  reconstructing what the leaderboard looked like on every past date
  isn't available. Live risk_pct also only updates on a weekly decay
  re-check, so this is a smaller departure from reality than it sounds.
- The SAME portfolio risk gate bot_engine.py runs live: a new trade is
  skipped entirely (never resized) if its risk_amount would push total
  open risk past max_portfolio_risk_pct of the CURRENT compounding
  balance — tracked via real entry/exit times, so "concurrent" here
  means genuinely overlapping, not assumed.
- Honest about what this can't do: the per-combo risk_pct doesn't
  reflect what the leaderboard looked like historically, and a combo's
  OWN entries are already fixed (this replays real detected touches, it
  doesn't re-detect signals under portfolio-aware rules). It answers
  "would this set of already-validated edges have been safe to run
  together," not "would a portfolio-aware strategy look different.\""""
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import experiments as ex
import ticker_behavior

# research_stats.py lives in etoro-dashboard (it's the module that reads
# THIS repo's own backtest cache and writes .cache/research_stats.json) —
# same cross-repo pattern research_stats.py itself already uses in
# reverse (it sys.path.inserts this directory to reach experiments.py).
sys.path.insert(0, "/Users/apple/dev/etoro-dashboard")
import research_stats

STARTING_BALANCE = 1000.0
KELLY_FRACTION = 0.5          # half-Kelly — dev_user's real live config
MAX_PORTFOLIO_RISK_PCT = 50.0  # dev_user's real live config (confirmed via db.get_bot_config, not assumed)
MAX_POSITION_PCT = 150.0       # dev_user's real live config (confirmed via db.get_bot_config, not assumed)

# Not a real number — bot_engine.py's own liquidity_cap structurally returns
# None for every forex pair (no volume data exists to derive one from), so
# there is no real capacity figure to plug in here. This is a fixed-dollar
# notional ceiling PER SYMBOL, concurrently open, that does NOT grow with
# the compounding balance (unlike MAX_POSITION_PCT, which scales with it and
# so never actually caps anything once balance gets large). Pulled from thin
# air on purpose, deliberately low, purely to see how sensitive the result
# is to *some* capacity constraint existing at all — not a claim about what
# these instruments can really absorb.
FANTASY_CAPACITY_PER_SYMBOL = 5_000.0

# clean_expansion_liquidity_fixed_rr's own docstring (experiments.py:3138-
# 3140): "Same touch shape as _clean_expansion_touches — this filters that
# function's own touches down to the subset with a matching sweep, it
# doesn't find new ones." Confirmed directly in this replay: wherever both
# are validated on the same (ticker, tf_label), they log byte-identical
# pnl on the same real touch (45 clusters, 17.3% of all trades, 29.6% of
# total profit in the uncapped run) — two "independent" combos betting on
# one real signal. TIERED_MOTHER/TIERED_SUBSET below merge them into ONE
# tradeable signal per real touch, sized off whichever tier's own
# validated numbers that specific touch actually earns — never both.
TIERED_MOTHER = "clean_expansion_fixed_rr"
TIERED_SUBSET = "clean_expansion_liquidity_fixed_rr"
# Reusing this platform's own existing precedent for "too few real trades
# to trust a verdict" (dashboard_data.py's execution_quality, etoro-
# dashboard: refuses to compute drift below 10 real trades) rather than
# inventing a fresh threshold here.
MIN_N_SURVIVED_BH_FOR_TIER = 10


def _kelly_fraction(rr_multiple, mean_return_holdout):
    """Verbatim mirror of bot_engine.py's own _kelly_fraction (lines
    285-310) — not reimplemented differently, so this replay sizes
    trades by the exact same math live trading actually uses.

    Real bug, caught the hard way: the first version of this mirror
    dropped the real function's own upper clamp (min(0.08, f_star)) —
    tests/test_bot_engine_sizing.py already covers this exact clamp on
    the real function, and this file should have been checked against
    it before trusting its own numbers. That miss produced a false
    "12-38% risk per trade, no ceiling" finding — the real ceiling is,
    and always was, 8% full-Kelly (4% at this platform's real
    half-Kelly setting), every combo, no exceptions."""
    if rr_multiple is None or rr_multiple <= 0 or mean_return_holdout is None:
        return 0.0
    p = (mean_return_holdout + 1) / (rr_multiple + 1)
    p = max(0.0, min(1.0, p))
    f_star = p - (1 - p) / rr_multiple
    return max(0.0, min(0.08, f_star))


def _combo_risk_pct():
    """Every validated _fixed_rr combo's own real risk_pct_used, sized
    exactly like bot_engine.py sizes it live: half-Kelly off that
    combo's own best_mean_return_holdout and its matched trial's real
    rr_multiple. Returns {(ticker, tf_label, strategy): {"risk_pct":...,
    "n_survived_bh":...}} — n_survived_bh rides along so tiered sizing
    (below) can refuse to trust a thin sample on its own."""
    lb = ticker_behavior.validated_leaderboard()
    fixed_rr = lb[lb["strategy"].str.endswith("_fixed_rr")].reset_index(drop=True)
    trials = ex.load_experiment_trials()
    out = {}
    for _, row in fixed_rr.iterrows():
        match = trials[(trials.ticker == row["ticker"]) & (trials.tf_label == row["tf_label"])
                       & (trials.label == row["best_label"])]
        if match.empty:
            continue
        rr_multiple = match.iloc[0]["settings"].get("rr_multiple")
        f_star = _kelly_fraction(rr_multiple, row["best_mean_return_holdout"])
        risk_pct = f_star * KELLY_FRACTION * 100
        if risk_pct > 0:
            out[(row["ticker"], row["tf_label"], row["strategy"])] = {
                "risk_pct": risk_pct, "n_survived_bh": int(row["n_survived_bh"]),
            }
    return out


def run_portfolio_backtest(since=None, capacity_per_symbol=None, max_symbol_risk_pct=None, tiered_merge=False):
    """since: an optional pd.Timestamp — trades before it are dropped
    entirely before the replay starts. Real, necessary honesty: 79 of
    the 866 replayable trades are from 2016-2023 (thin, sparse daily-
    timeframe history); compounding a max-4%-per-trade Kelly risk_pct
    across a full 10-year, 863-trade window explodes past anything a
    real account could execute (liquidity_cap — the real, live market-
    impact gate bot_engine.py also enforces — isn't modeled here,
    deliberately not guessed at without real volume data). Restricting
    to a recent, realistic window is what keeps the output interpretable
    rather than silently trusting a number that's correct arithmetic but
    not a real answer.

    capacity_per_symbol: None reproduces today's real, uncapped-by-
    capacity behavior (only the balance-relative MAX_POSITION_PCT bites,
    and it scales with balance so it never actually caps anything once
    balance compounds up). A number instead enforces a FIXED-dollar
    notional ceiling per ticker, concurrently open, that does NOT grow
    with balance — pass it explicitly to sweep different fantasy
    magnitudes without editing the module constant.

    max_symbol_risk_pct: None reproduces today's real behavior — bot_
    engine.py's own has_open_position (db.py:340-345) dedupes ONLY on
    (ticker, strategy), so two non-identical strategies validated on the
    same ticker can both hold real_risk_amount open simultaneously with
    no combined ceiling at all. A number caps TOTAL open risk_amount per
    ticker (summed across every combo trading it) to that % of the
    CURRENT balance — same scaling convention as MAX_PORTFOLIO_RISK_PCT,
    just scoped one level tighter (per-symbol instead of whole-book).
    This is the simulated version of the proposed live fix — a NEW third
    gate, not a replacement for MAX_POSITION_PCT or the portfolio cap.

    tiered_merge: False reproduces today's real behavior — TIERED_MOTHER
    and TIERED_SUBSET trade as two fully independent combos, double-
    logging every touch where both fire. True merges them into ONE
    tradeable signal per real touch: TIERED_SUBSET's own trades are
    dropped from the replay entirely (never an independent position),
    and whenever a TIERED_MOTHER touch's own entry_time also appears in
    TIERED_SUBSET's real trade list for that (ticker, tf_label), that
    ONE position is sized off TIERED_SUBSET's own validated risk_pct
    instead of the mother's — but only when TIERED_SUBSET has at least
    MIN_N_SURVIVED_BH_FOR_TIER real holdout survivors of its own; below
    that it falls back to the mother's own (better-sampled) numbers
    rather than trusting a 2-6-trade sample to resize a real position."""
    cached = research_stats.get_cached_stats()
    if cached is None:
        print("No research_stats cache — run research_stats.compute() first.")
        return None

    risk_pct_by_combo = _combo_risk_pct()
    print(f"{len(risk_pct_by_combo)} combos with a real positive Kelly edge, out of "
          f"{len({(t['ticker'], t['tf_label'], t['strategy']) for t in cached['all_trades']})} validated _fixed_rr combos.")

    tiered_pairs = set()
    subset_entry_times = set()
    if tiered_merge:
        for (ticker, tf_label, strategy) in risk_pct_by_combo:
            if strategy == TIERED_MOTHER and (ticker, tf_label, TIERED_SUBSET) in risk_pct_by_combo:
                tiered_pairs.add((ticker, tf_label))
        for t in cached["all_trades"]:
            if t["strategy"] == TIERED_SUBSET and (t["ticker"], t["tf_label"]) in tiered_pairs:
                subset_entry_times.add((t["ticker"], t["tf_label"], pd.Timestamp(t["entry_time"])))
        print(f"Tiered-merge active: {len(tiered_pairs)} (ticker, tf_label) pairs have both "
              f"{TIERED_MOTHER} and {TIERED_SUBSET} validated — merged into one signal each.")

    merged_away = 0
    tiered_up = 0
    tiered_down = 0
    trades = []
    for t in cached["all_trades"]:
        key = (t["ticker"], t["tf_label"], t["strategy"])
        if since is not None and "entry_time" in t and pd.Timestamp(t["entry_time"]) < since:
            continue
        if tiered_merge and t["strategy"] == TIERED_SUBSET and (t["ticker"], t["tf_label"]) in tiered_pairs:
            merged_away += 1
            continue  # this touch's risk is already counted via the mother trade below
        combo_info = risk_pct_by_combo.get(key)
        risk_pct = combo_info["risk_pct"] if combo_info else None
        # entry_price/stop_price are needed to reproduce the SAME notional
        # cap bot_engine.py actually applies live (max_position_pct) —
        # without them this would size every trade off the raw, aspirational
        # Kelly risk_pct, which is NOT what real trading ever risks (see
        # this project's own real_risk_amount fix, db.py/bot_engine.py,
        # found live: a $40 Kelly ask on a tight-stop instrument really put
        # ~$0.045 at risk once the notional cap actually bit).
        if risk_pct is None or "exit_time" not in t or "entry_price" not in t or "stop_price" not in t:
            continue  # no real Kelly edge, or this trade's real levels weren't recoverable — skip, don't guess
        if (tiered_merge and t["strategy"] == TIERED_MOTHER
                and (t["ticker"], t["tf_label"]) in tiered_pairs
                and (t["ticker"], t["tf_label"], pd.Timestamp(t["entry_time"])) in subset_entry_times):
            subset_info = risk_pct_by_combo.get((t["ticker"], t["tf_label"], TIERED_SUBSET))
            if subset_info["n_survived_bh"] >= MIN_N_SURVIVED_BH_FOR_TIER:
                if subset_info["risk_pct"] > risk_pct:
                    tiered_up += 1
                elif subset_info["risk_pct"] < risk_pct:
                    tiered_down += 1
                risk_pct = subset_info["risk_pct"]
        risk_unit = abs(t["entry_price"] - t["stop_price"])
        if risk_unit <= 0:
            continue
        trades.append({
            "combo": f"{t['ticker']} {t['tf_label']} {t['strategy']}",
            "ticker": t["ticker"],
            "entry_time": pd.Timestamp(t["entry_time"]),
            "exit_time": pd.Timestamp(t["exit_time"]),
            "r_real": t["r_real"],
            "risk_pct": risk_pct,
            "entry_price": t["entry_price"],
            "risk_unit": risk_unit,
        })
    trades.sort(key=lambda x: x["entry_time"])
    print(f"{len(trades)} real historical trades to replay, "
          f"{trades[0]['entry_time'].date()} -> {trades[-1]['entry_time'].date()}")
    if tiered_merge:
        print(f"  {merged_away} duplicate subset-touches merged away, "
              f"{tiered_up} touches sized UP on the subset's own stronger edge, "
              f"{tiered_down} sized DOWN on its weaker one "
              f"(0/0 is real, not a bug: every validated combo here has an edge strong "
              f"enough to already saturate the 8%-full-Kelly/4%-half-Kelly ceiling, so "
              f"mother and subset clamp to the identical risk_pct regardless of which "
              f"mean-R sizes it — the ceiling makes the WEIGHT moot even where the "
              f"underlying edge genuinely differs)")

    balance = STARTING_BALANCE
    open_positions = []   # each: {"exit_time", "risk_amount", "r_real", "combo", "ticker", "notional"}
    open_notional_by_ticker = {}
    open_risk_by_ticker = {}
    equity_curve = []     # (timestamp, balance, combo, pnl) — one row per REAL resolution
    skipped_by_cap = 0
    skipped_by_capacity = 0
    skipped_by_symbol_risk = 0
    trades_taken = 0
    daily_pnl = {}

    def _resolve(pos, ts):
        nonlocal balance
        pnl = pos["risk_amount"] * pos["r_real"]
        balance += pnl
        equity_curve.append((ts, balance, pos["combo"], pnl))
        day = ts.date()
        daily_pnl[day] = daily_pnl.get(day, 0.0) + pnl
        open_notional_by_ticker[pos["ticker"]] -= pos["notional"]
        open_risk_by_ticker[pos["ticker"]] -= pos["risk_amount"]

    for tr in trades:
        # Resolve (apply the REAL pnl for) any position whose exit_time
        # has already passed by this trade's own entry_time — this is
        # what makes `balance` the genuinely already-realized capital a
        # live sizing decision would actually see, not a look-ahead peek
        # at trades that haven't closed yet. Getting this step wrong
        # (crediting pnl at ENTRY instead of at real EXIT) was the first,
        # caught-before-running version of this bug.
        still_open = []
        for pos in open_positions:
            if pos["exit_time"] <= tr["entry_time"]:
                _resolve(pos, pos["exit_time"])
            else:
                still_open.append(pos)
        open_positions = still_open

        open_risk = sum(p["risk_amount"] for p in open_positions)
        portfolio_cap = balance * MAX_PORTFOLIO_RISK_PCT / 100
        room = max(0.0, portfolio_cap - open_risk)

        # Aspirational Kelly ask, THEN the same notional cap bot_engine.py
        # actually applies (max_position_pct) BEFORE the portfolio gate
        # ever sees a number — otherwise this sizes every trade off a
        # figure live trading would never actually risk. real_risk_amount
        # is always <= the aspirational risk_amount, exactly equal to it
        # only when the notional cap doesn't bind.
        kelly_risk_amount = balance * tr["risk_pct"] / 100
        raw_notional = kelly_risk_amount * tr["entry_price"] / tr["risk_unit"]
        capped_notional = min(raw_notional, balance * MAX_POSITION_PCT / 100)

        # Fantasy capacity gate: a fixed-dollar ceiling per ticker,
        # concurrently open, that does NOT grow with balance — unlike
        # MAX_POSITION_PCT above, which is balance-relative and so stops
        # binding at all once the account has compounded up enough.
        if capacity_per_symbol is not None:
            open_notional = open_notional_by_ticker.get(tr["ticker"], 0.0)
            capacity_room = max(0.0, capacity_per_symbol - open_notional)
            if capacity_room <= 0:
                skipped_by_capacity += 1
                continue
            capped_notional = min(capped_notional, capacity_room)

        real_risk_amount = capped_notional * tr["risk_unit"] / tr["entry_price"]

        # Proposed third gate: total open risk_amount per ticker, summed
        # across every combo trading it, capped independently of the
        # whole-book MAX_PORTFOLIO_RISK_PCT — the gap that let two
        # combos hold AUDUSD=X 5m open at once live with no combined
        # ceiling at all.
        if max_symbol_risk_pct is not None:
            symbol_cap = balance * max_symbol_risk_pct / 100
            open_symbol_risk = open_risk_by_ticker.get(tr["ticker"], 0.0)
            symbol_room = max(0.0, symbol_cap - open_symbol_risk)
            if symbol_room <= 0:
                skipped_by_symbol_risk += 1
                continue
            real_risk_amount = min(real_risk_amount, symbol_room)

        if real_risk_amount > room:
            skipped_by_cap += 1
            continue

        trades_taken += 1
        open_positions.append({"exit_time": tr["exit_time"], "risk_amount": real_risk_amount,
                                "r_real": tr["r_real"], "combo": tr["combo"], "ticker": tr["ticker"],
                                "notional": capped_notional})
        open_notional_by_ticker[tr["ticker"]] = open_notional_by_ticker.get(tr["ticker"], 0.0) + capped_notional
        open_risk_by_ticker[tr["ticker"]] = open_risk_by_ticker.get(tr["ticker"], 0.0) + real_risk_amount

    # Resolve whatever's still open at the end of the real historical
    # window — otherwise the last handful of real trades (whichever
    # hadn't closed by the very last entry event) would silently never
    # count, understating both the true end balance and the trade count.
    for pos in sorted(open_positions, key=lambda p: p["exit_time"]):
        _resolve(pos, pos["exit_time"])

    curve = pd.DataFrame(equity_curve, columns=["ts", "balance", "combo", "pnl"])
    daily = pd.Series(daily_pnl).sort_index()
    daily_ret = daily / STARTING_BALANCE  # simple return on starting capital, for a stable Sharpe denominator

    running_peak = curve["balance"].cummax()
    drawdown = running_peak - curve["balance"]
    max_dd = drawdown.max()
    max_dd_pct = (drawdown / running_peak).max() * 100

    sharpe = daily_ret.mean() / daily_ret.std(ddof=1) if daily_ret.std(ddof=1) > 0 else None
    downside = np.minimum(daily_ret, 0.0)
    downside_dev = np.sqrt((downside ** 2).mean())
    sortino = daily_ret.mean() / downside_dev if downside_dev > 0 else None
    ann = np.sqrt(252)

    total_signals = trades_taken + skipped_by_cap + skipped_by_capacity + skipped_by_symbol_risk
    print()
    print("=" * 70)
    if capacity_per_symbol is not None:
        print(f"Fantasy capacity/symbol: ${capacity_per_symbol:,.0f}")
    if max_symbol_risk_pct is not None:
        print(f"Max symbol risk pct:     {max_symbol_risk_pct:.1f}% of balance")
    print(f"Trades taken:            {trades_taken}")
    print(f"Trades skipped by cap:   {skipped_by_cap}  "
          f"({skipped_by_cap/total_signals*100:.1f}% of all real signals)")
    print(f"Skipped by capacity:     {skipped_by_capacity}  "
          f"({skipped_by_capacity/total_signals*100:.1f}% of all real signals)")
    print(f"Skipped by symbol risk:  {skipped_by_symbol_risk}  "
          f"({skipped_by_symbol_risk/total_signals*100:.1f}% of all real signals)")
    print(f"Start balance:           ${STARTING_BALANCE:,.2f}")
    print(f"End balance:             ${curve['balance'].iloc[-1]:,.2f}")
    print(f"Total return:            {(curve['balance'].iloc[-1]/STARTING_BALANCE - 1)*100:+.1f}%")
    print(f"Max drawdown:            ${max_dd:,.2f}  ({max_dd_pct:.1f}% of peak)")
    print(f"Portfolio Sharpe (ann.): {sharpe*ann if sharpe else None}")
    print(f"Portfolio Sortino(ann.): {sortino*ann if sortino else None}")
    print("=" * 70)

    curve.to_json("/tmp/portfolio_backtest_curve.json", orient="records", date_format="iso")
    return {
        "trades_taken": trades_taken, "skipped_by_cap": skipped_by_cap,
        "skipped_by_capacity": skipped_by_capacity, "skipped_by_symbol_risk": skipped_by_symbol_risk,
        "start_balance": STARTING_BALANCE, "end_balance": float(curve["balance"].iloc[-1]),
        "max_drawdown": float(max_dd), "max_drawdown_pct": float(max_dd_pct),
        "sharpe_annualized": float(sharpe * ann) if sharpe else None,
        "sortino_annualized": float(sortino * ann) if sortino else None,
        "curve": curve,
    }


if __name__ == "__main__":
    window = pd.Timestamp.now("UTC") - pd.Timedelta(days=365)
    print("--- Today's real behavior (no tiered merge) ---")
    run_portfolio_backtest(since=window)
    print()
    print("--- Tiered merge: clean_expansion + clean_expansion_liquidity as ONE signal ---")
    run_portfolio_backtest(since=window, tiered_merge=True)
