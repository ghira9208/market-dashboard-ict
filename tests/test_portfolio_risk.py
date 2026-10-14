"""portfolio_risk.py — drawdown/profit-factor math with hand-checkable answers,
and the one property the whole block-bootstrap exists for: keeping same-day
trades together must show MORE drawdown than treating them as independent."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import portfolio_risk as pr


def test_max_drawdown_is_peak_to_trough_in_r():
    assert pr.max_drawdown_r([1, 1, -1, -1, -1, 2]) == pytest.approx(3.0)   # peak +2, trough -1
    assert pr.max_drawdown_r([1, 2, 3]) == 0.0
    assert pr.max_drawdown_r([-1, -1]) == pytest.approx(2.0)                # drawdown measured from the start too
    assert pr.max_drawdown_r([]) == 0.0


def test_profit_factor():
    assert pr.profit_factor([2, 2, -1, -1]) == pytest.approx(2.0)
    assert pr.profit_factor([1, 1]) == float("inf")
    assert pr.profit_factor([-1, -2]) == 0.0
    assert np.isnan(pr.profit_factor([0, 0]))


def test_longest_losing_streak_counts_breakeven_as_a_non_win():
    assert pr.longest_losing_streak([1, -1, -1, 0, 1, -1]) == 3
    assert pr.longest_losing_streak([1, 1]) == 0


def test_expected_drawdown_is_zero_for_a_strategy_that_only_wins():
    e = pr.expected_max_dd([1.0] * 50, horizon=40, n_sims=200)
    assert e["dd_median"] == 0.0 and e["p_down"] == 0.0


def test_expected_drawdown_grows_with_the_chance_of_losing():
    rng = np.random.default_rng(1)
    safe = pr.expected_max_dd([1] * 80 + [-1] * 20, horizon=100, rng=rng)
    risky = pr.expected_max_dd([1] * 55 + [-1] * 45, horizon=100, rng=rng)
    assert risky["dd_p95"] > safe["dd_p95"]
    assert risky["p_down"] > safe["p_down"]


def test_combo_table_skips_tiny_samples_and_reports_each_combo():
    t = pd.DataFrame({"combo": ["A"] * 40 + ["B"] * 5, "r": [1.0, -1.0] * 20 + [1.0] * 5})
    table = pr.combo_risk_table(t)
    assert list(table["combo"]) == ["A"] and table.iloc[0]["n"] == 40


def test_equity_starts_at_one_and_compounds():
    p = pr.simulate_equity([1.0], risk_frac=0.1, n_trades=3, n_sims=2)
    assert p.shape == (2, 4) and np.allclose(p[:, 0], 1.0)
    assert np.allclose(p[:, -1], 1.1 ** 3)


def test_a_total_loss_can_zero_the_account_but_never_make_it_negative():
    p = pr.simulate_equity([-5.0], risk_frac=0.5, n_trades=4, n_sims=3)
    assert (p >= 0).all() and np.allclose(p[:, -1], 0.0)


def test_block_bootstrap_keeps_a_days_trades_together():
    # day 1 has two big losses together, other days are single small wins: a block draw of day 1 always
    # delivers BOTH losses back to back; iid never has to.
    r = np.array([-3.0, -3.0] + [0.5] * 10)
    days = np.array([0, 0] + list(range(1, 11)))
    paths = pr.simulate_equity(r, 0.01, 12, n_sims=500, rng=np.random.default_rng(0), day_ids=days)
    steps = np.diff(np.log(paths), axis=1)
    big = (steps < np.log(1 - 0.03 + 1e-9) + 1e-9)  # a -3R step
    # in every path, -3R steps come in adjacent pairs (the two losses of the same day)
    for row in big[:50]:
        idx = np.flatnonzero(row)
        if len(idx) >= 2:
            assert (np.diff(idx) == 1).any()


def test_clustered_losses_produce_deeper_drawdowns_than_independent_ones():
    rng = np.random.default_rng(5)
    # 300 days, each with 4 trades; 10% of days are "bad": all four lose together, else all win a little
    r, days = [], []
    for d in range(300):
        bad = rng.random() < 0.1
        for _ in range(4):
            r.append(-1.0 if bad else 0.35)
            days.append(d)
    iid = pr.equity_summary(pr.simulate_equity(r, 0.02, 400, n_sims=1500, rng=np.random.default_rng(1)))
    blk = pr.equity_summary(pr.simulate_equity(r, 0.02, 400, n_sims=1500, rng=np.random.default_rng(1), day_ids=days))
    assert blk["dd_p95"] > iid["dd_p95"] * 1.1


def test_equity_summary_fields():
    s = pr.equity_summary(pr.simulate_equity([1.0, -1.0], 0.01, 50, n_sims=100, rng=np.random.default_rng(0)))
    assert {"median_final", "p5_final", "p95_final", "p_loss", "dd_median", "dd_p95", "p_dd_over_20"} <= set(s)
    assert 0 <= s["p_loss"] <= 1
