"""ema_lead_forward.py — a forward test is only worth something if it is frozen, counts only trades that
happened after the freeze, never counts a trade whose holding window is not fully observed, never records
anything twice, and only declares success on the pre-registered bar."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ema_lead_forward as fw
import streak_polish_study as sps

SETTINGS = {"min_streak": 3, "retr_window_bars": 40, "rr_multiple": 1.0, "atr_mult_stop": 1.0, "max_bars": 40}
COMBO = {"ticker": "TST", "tf": "1h", "strategy": "clean_expansion_fixed_rr", "settings": SETTINGS}


def _df(n=600, seed=1):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"Open": close, "High": close + 0.4, "Low": close - 0.4, "Close": close,
                         "Volume": 1.0}, index=idx)


@pytest.fixture
def fake_finder(monkeypatch):
    def finder(df, min_streak, retr_window_bars):
        return [{"confirm_pos": p, "direction": "bullish", "zone_top": 1.0, "zone_bottom": 0.5, "touch_pos": p}
                for p in (100, 200, 590)]
    monkeypatch.setitem(sps.FIND, "clean_expansion_fixed_rr", finder)


def test_edited_spec_is_refused(tmp_path):
    path = str(tmp_path / "spec.json")
    spec = fw.create_spec([COMBO], pd.Timestamp("2026-02-01", tz="UTC"), path=path)
    assert fw.load_spec(path)["sha256"] == spec["sha256"]
    tampered = json.loads(Path(path).read_text())
    tampered["alpha"] = 0.5
    Path(path).write_text(json.dumps(tampered))
    with pytest.raises(RuntimeError):
        fw.load_spec(path)


def test_only_trades_after_the_freeze_and_fully_resolved_count(fake_finder):
    df = _df()
    before_all = pd.Timestamp("2025-01-01", tz="UTC")
    rows = fw.combo_trades(df, COMBO, before_all, 2.0)
    entries = {r["entry"] for r in rows}
    assert df.index[101].isoformat() in entries and df.index[201].isoformat() in entries
    assert df.index[591].isoformat() not in entries  # 591 + 40 > last bar: window not fully observed
    late = fw.combo_trades(df, COMBO, df.index[150], 2.0)
    assert {r["entry"] for r in late} == {df.index[201].isoformat()}  # the pos-101 trade predates the freeze


def test_advance_is_idempotent(fake_finder, tmp_path):
    spec = {"created": "2025-01-01T00:00:00+00:00", "cost_bps": 2.0, "stress_bps": 6.0, "combos": [COMBO]}
    ledger = str(tmp_path / "l.jsonl")
    first = fw.advance(spec, lambda t, tf: _df(), ledger)
    assert first >= 2
    assert fw.advance(spec, lambda t, tf: _df(), ledger) == 0
    rows = [json.loads(x) for x in Path(ledger).read_text().splitlines()]
    assert all(r["r_stress"] <= r["r"] + 1e-9 for r in rows)  # higher cost never improves a trade


def _ledger(tmp_path, n, edge):
    rng = np.random.default_rng(3)
    rows = []
    for i in range(n):
        aligned = bool(i % 4)  # 25% not aligned
        r = rng.normal(1.0 + (0 if aligned else edge), 1.5)
        rows.append({"combo": f"C{i % 6}|1h|clean_expansion_fixed_rr", "strategy": "clean_expansion_fixed_rr",
                     "tf": "1h", "entry": (pd.Timestamp("2026-03-01", tz="UTC") + pd.Timedelta(hours=i * 3)).isoformat(),
                     "r": r, "r_stress": r - 0.3, "aligned": aligned})
    p = tmp_path / "l.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return str(p)


SPEC = {"sha256": "x", "created": "2026-03-01T00:00:00+00:00", "min_trades": 400, "min_group": 100, "alpha": 0.01}


def test_status_reports_collecting_below_the_bar(tmp_path):
    st = fw.status(SPEC, _ledger(tmp_path, 120, 2.0))
    assert st["verdict"].startswith("collecting")


def test_status_confirms_a_real_effect(tmp_path):
    st = fw.status(SPEC, _ledger(tmp_path, 900, 2.0))
    assert st["verdict"].startswith("CONFIRMED"), st


def test_status_rejects_no_effect(tmp_path):
    st = fw.status(SPEC, _ledger(tmp_path, 900, 0.0))
    assert st["verdict"].startswith("NOT confirmed"), st
