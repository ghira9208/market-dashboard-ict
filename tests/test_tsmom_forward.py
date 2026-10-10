"""tsmom_forward.py — the forward test is only worth anything if it is frozen,
idempotent, never uses a not-yet-final week, and computes EXACTLY what the
backtest computed (otherwise "it worked forward" would mean nothing)."""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tsmom_forward as fw
import tsmom_backtest as tm


def _series(n, seed, start="2012-01-02"):
    rng = np.random.default_rng(seed)
    drift = np.repeat(rng.choice([-0.0012, 0.0012], size=n // 90 + 1), 90)[:n]
    return pd.Series(100 * np.cumprod(1 + drift + rng.normal(0, 0.008, n)), index=pd.bdate_range(start, periods=n))


@pytest.fixture
def env(tmp_path):
    closes = {"A": _series(3000, 1), "B": _series(3000, 2)}
    spec_path, ledger = str(tmp_path / "spec.json"), str(tmp_path / "ledger.jsonl")
    return closes, spec_path, ledger


def _spec(env, first="2020-01-03", **kw):
    closes, spec_path, _ = env
    spec = fw.create_spec(list(closes), date(2019, 12, 30), path=spec_path)
    spec["first_decision_friday"] = first
    spec.update(kw)
    spec["sha256"] = fw._spec_hash(spec)
    return spec


def test_first_friday_after_is_strictly_after():
    assert fw.first_friday_after(pd.Timestamp("2026-10-09")).date() == date(2026, 10, 16)   # a Friday -> next Friday
    assert fw.first_friday_after(pd.Timestamp("2026-10-08")).date() == date(2026, 10, 9)
    assert fw.first_friday_after(pd.Timestamp("2026-10-10")).date() == date(2026, 10, 16)


def test_the_spec_is_frozen_and_tampering_is_refused(tmp_path):
    path = str(tmp_path / "spec.json")
    fw.create_spec(["A"], date(2026, 10, 9), path=path)
    assert fw.load_spec(path)["lookback_days"] == 180
    spec = json.load(open(path)); spec["lookback_days"] = 90
    json.dump(spec, open(path, "w"))
    with pytest.raises(RuntimeError, match="edited"):
        fw.load_spec(path)


def test_nothing_before_the_first_decision_friday_is_ever_recorded(env):
    closes, _, ledger = env
    spec = _spec(env, first="2023-06-02")
    fw.advance(spec, closes, date(2025, 1, 1), ledger)
    rows = [json.loads(l) for l in open(ledger)]
    assert rows and min(r["friday"] for r in rows) >= "2023-06-02"


def test_rerunning_records_nothing_twice(env):
    closes, _, ledger = env
    spec = _spec(env)
    fw.advance(spec, closes, date(2025, 1, 1), ledger)
    n1 = sum(1 for _ in open(ledger))
    assert fw.advance(spec, closes, date(2025, 1, 1), ledger) == (0, 0)
    assert sum(1 for _ in open(ledger)) == n1


def test_a_week_is_not_scored_until_the_following_week_is_complete(env):
    closes, _, ledger = env
    spec = _spec(env)
    # as of a Saturday: the Friday that just closed is DECIDED but cannot be realized yet
    sat = date(2021, 3, 6)
    sub = {a: s[s.index <= pd.Timestamp("2021-03-05")] for a, s in closes.items()}
    fw.advance(spec, sub, sat, ledger)
    rows = [json.loads(l) for l in open(ledger)]
    last_decision = max(r["friday"] for r in rows if r["type"] == "decision")
    last_realized = max(r["friday"] for r in rows if r["type"] == "realized")
    assert last_decision == "2021-03-05" and last_realized < last_decision


def test_a_friday_still_in_progress_is_not_decided(env):
    closes, _, ledger = env
    spec = _spec(env)
    friday = date(2021, 3, 5)
    sub = {a: s[s.index <= pd.Timestamp(friday)] for a, s in closes.items()}
    fw.advance(spec, sub, friday, ledger)  # "today" IS that Friday: its bar may still be forming
    rows = [json.loads(l) for l in open(ledger)]
    assert "2021-03-05" not in {r["friday"] for r in rows if r["type"] == "decision"}


def test_forward_results_equal_the_backtests_exactly(env):
    # the key invariant: same signal code, same sizing, same costs -> identical numbers
    closes, _, ledger = env
    spec = _spec(env, first="2018-01-05")
    fw.advance(spec, closes, date(2024, 6, 1), ledger)
    realized = pd.DataFrame([json.loads(l) for l in open(ledger) if '"realized"' in l])
    for asset, close in closes.items():
        bt = tm.weekly_arrays(close, 180)
        bt.index = [str(p.end_time.date()) for p in bt.index.to_period("W-FRI")]
        got = realized[realized["asset"] == asset].set_index("friday")
        common = [d for d in got.index if d in bt.index]
        assert len(common) > 100
        for d in common[:150]:
            r_fw = got.loc[d, "pos"] * got.loc[d, "u"] - (tm.COST_BPS / 1e4 / 2) * got.loc[d, "turnover"] / got.loc[d, "vol"]
            assert r_fw == pytest.approx(bt.loc[d, "r"], abs=1e-9)


def test_status_is_interim_until_the_judgement_horizon_then_decides(env):
    closes, _, ledger = env
    spec = _spec(env, first="2019-01-04", judge_at_weeks=104)
    upto = lambda d: {a: x[x.index <= pd.Timestamp(d)] for a, x in closes.items()}  # a real run never sees past today
    fw.advance(spec, upto("2019-11-29"), date(2019, 12, 1), ledger)
    early = fw.status(spec, ledger)
    assert early["realized_weeks"] < 104 and early["verdict"].startswith("collecting")
    fw.advance(spec, upto("2024-11-29"), date(2024, 12, 1), ledger)
    late = fw.status(spec, ledger)
    assert late["realized_weeks"] >= 104
    assert late["verdict"] in ("CONFIRMED on out-of-sample data", "NOT confirmed — retire the idea")
    assert 0 < late["p_value_circular_shift"] <= 1


def test_status_with_an_empty_ledger_is_honest(env):
    spec = _spec(env)
    st = fw.status(spec, env[2])
    assert st["realized_weeks"] == 0 and st["verdict"].startswith("collecting")
