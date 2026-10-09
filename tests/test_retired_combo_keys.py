"""retired_combo_keys() — the live-usable key set bot_engine.py's own
run_cycle() now excludes from trading, closing a real gap found while
wiring objective 8's LIVE confirmation: a combo manually RETIRED
(invalidated_combos) after already earning validated=1 had nothing
stopping it from continuing to trade — DECAYED already had its own
live-side exclusion (_decayed_combo_keys/_exclude_decayed), RETIRED
never did."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import experiments as ex
import ticker_behavior
from ticker_behavior import retired_combo_keys
from strategy_identity import compute_strategy_id


@pytest.fixture
def isolated_log(tmp_path, monkeypatch):
    trials_path = tmp_path / "trials.jsonl"
    db_path = tmp_path / "ticker_behavior.db"
    monkeypatch.setattr(ex, "_TRIALS_PATH", str(trials_path))
    monkeypatch.setattr(ticker_behavior, "_DB_PATH", str(db_path))
    return trials_path


def _settings():
    return {"min_streak": 3, "retr_window_bars": 40, "atr_mult_stop": 1.5, "rr_multiple": 3.0, "max_bars": 60}


def _write_trial(path, **fields):
    base = {
        "logged_at": "2026-01-01T00:00:00+00:00", "ticker": "EURUSD=X", "tf_label": "5m",
        "label": "clean_expansion_fixed_rr · rr=3.0 · atr=1.5 · maxbars=60",
        "settings": _settings(),
        "n_events": 100, "verdict": "SCORED", "n_train": 80, "n_holdout": 20,
        "mean_return_train": 0.5, "p_value_train": 0.01, "mean_return_holdout": 0.4,
        "holdout_verdict": "PASSED", "same_sign": True,
    }
    base.update(fields)
    with open(path, "a") as f:
        f.write(json.dumps(base) + "\n")


def test_empty_when_nothing_has_ever_been_retired(isolated_log):
    _write_trial(isolated_log)
    ticker_behavior.rebuild_db()
    assert retired_combo_keys() == set()


def test_includes_a_manually_retired_combo_even_though_it_also_validated(isolated_log):
    # The exact real gap this closes: a genuine, stored identity exists
    # (validated=1 would be True) AND it was manually retired. RETIRED
    # must still be excluded from the live key set regardless.
    sid = compute_strategy_id(family="clean_expansion_fixed_rr", settings=_settings(), target_widen_mult=1.0,
                               data_pipeline_version=ex.research_data.DATA_PIPELINE_VERSION,
                               detector_logic_version=ex.DETECTOR_LOGIC_VERSIONS.get("clean_expansion", 0),
                               cost_bps=ex.DEFAULT_COST_BPS)
    _write_trial(isolated_log, strategy_id=sid, target_widen_mult=1.0,
                 invalidated_reason="manually pulled for review")
    ticker_behavior.rebuild_db()
    assert ("EURUSD=X", "5m", "clean_expansion_fixed_rr") in retired_combo_keys()


def test_does_not_include_a_combo_that_was_never_retired(isolated_log):
    _write_trial(isolated_log, ticker="EURUSD=X")
    _write_trial(isolated_log, ticker="GBPUSD=X", invalidated_reason="pulled")
    ticker_behavior.rebuild_db()
    keys = retired_combo_keys()
    assert ("GBPUSD=X", "5m", "clean_expansion_fixed_rr") in keys
    assert ("EURUSD=X", "5m", "clean_expansion_fixed_rr") not in keys
