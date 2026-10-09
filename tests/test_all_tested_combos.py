"""all_tested_combos() — the deliberately UNFILTERED sibling of
validated_leaderboard(). Real bug this exists to fix, confirmed live
2026-10-08: bot_engine.py's own _close_stale_positions used to resolve a
combo's max_bars from validated_leaderboard() alone — fine until the
item-10 fail-closed identity gate demoted a combo to validated=0 AFTER a
position had already opened under its earlier validated=1 state.
validated_leaderboard() then had nothing left to look up, and that
position's own time-stop silently stopped being enforced (NZDJPY=X 1h
ran 5 bars past its own 60-bar limit with zero enforcement). Risk
management on an already-open position must never depend on whether its
combo is STILL eligible for a new entry."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import experiments as ex
import ticker_behavior
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
    import json
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


def test_includes_a_combo_that_validated_leaderboard_excludes(isolated_log):
    # A legacy/unidentified-only survivor: real holdout-passing evidence
    # (would show HOLDOUT_VALIDATED), but validated=0 (no real stored
    # strategy_id) — validated_leaderboard() correctly excludes it; this
    # function must NOT.
    _write_trial(isolated_log)  # no strategy_id at all
    ticker_behavior.rebuild_db()

    assert ticker_behavior.validated_leaderboard().empty
    all_combos = ticker_behavior.all_tested_combos()
    match = all_combos[(all_combos["ticker"] == "EURUSD=X") & (all_combos["tf_label"] == "5m")
                        & (all_combos["strategy"] == "clean_expansion_fixed_rr")]
    assert len(match) == 1
    assert match.iloc[0]["validated"] == 0
    assert match.iloc[0]["best_label"] == "clean_expansion_fixed_rr · rr=3.0 · atr=1.5 · maxbars=60"


def test_also_includes_genuinely_validated_combos(isolated_log):
    sid = compute_strategy_id(family="clean_expansion_fixed_rr", settings=_settings(), target_widen_mult=1.0,
                               data_pipeline_version=ex.research_data.DATA_PIPELINE_VERSION,
                               detector_logic_version=ex.DETECTOR_LOGIC_VERSIONS.get("clean_expansion", 0),
                               cost_bps=ex.DEFAULT_COST_BPS)
    _write_trial(isolated_log, strategy_id=sid, target_widen_mult=1.0)
    ticker_behavior.rebuild_db()

    all_combos = ticker_behavior.all_tested_combos()
    assert len(all_combos) == 1
    assert all_combos.iloc[0]["validated"] == 1
    assert len(ticker_behavior.validated_leaderboard()) == 1


def test_a_combo_with_pure_noise_still_appears_with_its_own_best_label(isolated_log):
    # Even a combo with no real edge at all (EXPERIMENTAL lifecycle)
    # still appears here with a real best_label — _close_stale_positions
    # needs that lookup key for ANY already-open position, regardless of
    # how weak the combo's own evidence ultimately turned out to be.
    _write_trial(isolated_log, p_value_train=0.9, mean_return_holdout=-0.3,
                 holdout_verdict="FAILED", same_sign=False)
    ticker_behavior.rebuild_db()

    all_combos = ticker_behavior.all_tested_combos()
    assert len(all_combos) == 1
    assert all_combos.iloc[0]["lifecycle_state"] == "EXPERIMENTAL"
    assert all_combos.iloc[0]["best_label"]
