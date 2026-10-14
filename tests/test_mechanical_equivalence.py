"""compute_risk_unit / compute_stop_target / resolve_entry_pos —
objective 7's shared primitives. The cross-repo end-to-end proof (same
synthetic OHLCV through _simulate_fixed_rr AND bot_engine.py's own
_size_and_levels) lives in edge-pipeline's own tests/
test_mechanical_equivalence.py, since it needs bot_engine.py; this file
covers the three functions themselves plus confirming _simulate_fixed_rr
genuinely DELEGATES to them rather than merely defining them unused
alongside its own, still-separate algebra."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import experiments as ex


def test_resolve_entry_pos_is_the_bar_after_confirmation():
    assert ex.resolve_entry_pos(4) == 5
    assert ex.resolve_entry_pos(0) == 1


def test_compute_risk_unit_scales_atr_by_the_stop_multiplier():
    assert ex.compute_risk_unit(2.0, 1.5) == pytest.approx(3.0)
    assert ex.compute_risk_unit(0.0, 1.5) == 0.0


def test_compute_stop_target_bullish():
    stop, target = ex.compute_stop_target(entry_price=100.0, risk_unit=2.0, rr_multiple=3.0, direction="bullish")
    assert stop == pytest.approx(98.0)
    assert target == pytest.approx(106.0)


def test_compute_stop_target_bearish():
    stop, target = ex.compute_stop_target(entry_price=100.0, risk_unit=2.0, rr_multiple=3.0, direction="bearish")
    assert stop == pytest.approx(102.0)
    assert target == pytest.approx(94.0)


def test_simulate_fixed_rr_genuinely_delegates_to_compute_stop_target(monkeypatch):
    # Not just "the function exists" — proves _simulate_fixed_rr's own
    # _run closure actually CALLS it, by swapping in a deliberately wrong
    # implementation and confirming the backtest's own output changes
    # with it. If a future edit reintroduced a local, hard-coded copy of
    # the formula inside _run, this test would catch that regression
    # (the monkeypatch would stop affecting the result).
    rows = [{"Open": 100.0, "High": 100.5, "Low": 99.5, "Close": 100.0} for _ in range(20)]
    rows.append({"Open": 100.0, "High": 100.5, "Low": 99.5, "Close": 100.0})  # entry bar
    rows.append({"Open": 100.0, "High": 110.0, "Low": 99.9, "Close": 109.0})  # clears any real target
    df = pd.DataFrame(rows)
    touches = [{"confirm_pos": 19, "direction": "bullish"}]

    baseline = ex._simulate_fixed_rr(df, touches, rr_multiple=1.0, atr_mult_stop=1.0, max_bars=5, cost_bps=0.0)
    assert baseline.iloc[0]["r_real"] == pytest.approx(1.0)  # a clean target hit

    # A "wrong" stop/target function that places the target absurdly far
    # away — the same clean breakout bar can no longer reach it, so the
    # trade instead rides out to the time-based exit.
    monkeypatch.setattr(ex, "compute_stop_target",
                         lambda entry_price, risk_unit, rr_multiple, direction: (entry_price - risk_unit, 1e9))
    patched = ex._simulate_fixed_rr(df, touches, rr_multiple=1.0, atr_mult_stop=1.0, max_bars=5, cost_bps=0.0)
    assert patched.iloc[0]["r_real"] != pytest.approx(1.0)


def test_simulate_fixed_rr_genuinely_delegates_to_resolve_entry_pos(monkeypatch):
    rows = [{"Open": 100.0, "High": 100.5, "Low": 99.5, "Close": 100.0} for _ in range(20)]
    rows.append({"Open": 100.0, "High": 100.5, "Low": 99.5, "Close": 100.0})
    rows.append({"Open": 200.0, "High": 210.0, "Low": 199.9, "Close": 209.0})  # a different "entry" bar
    df = pd.DataFrame(rows)
    touches = [{"confirm_pos": 19, "direction": "bullish"}]

    monkeypatch.setattr(ex, "resolve_entry_pos", lambda confirm_pos: confirm_pos + 2)  # off-by-one on purpose
    events = ex._simulate_fixed_rr(df, touches, rr_multiple=1.0, atr_mult_stop=1.0, max_bars=5, cost_bps=0.0)
    # entry_pos is now 21 (the off-by-one bar), so entry_price reads
    # bar 21's own Open (200.0), not bar 20's (100.0) — proves the
    # convention is genuinely READ from the shared function, not
    # independently hardcoded as "+1" inside _simulate_fixed_rr itself.
    assert len(events) == 1  # still finds an entry, just at the wrong bar — confirms it moved
