"""run_deep_backtest_rr_with_lock — objective 3's third, LOCKED dataset.
Protects the final holdout from adaptive research by never letting a third,
genuinely-untouched slice of time feed into any selection decision, while
reusing run_deep_backtest_rr's own statistical test (_score_split) so every
existing consumer (survived/bh_significant, the leaderboard, live
enforcement) reads a locked-split trial exactly like any other.

validate_target_variants.py is this function's one real caller — a
genuinely new research surface (target-distance widening), started clean on
the 3-way split rather than retrofitting run_deep_backtest_rr's own existing
80/20 call sites."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import experiments as ex

_SETTINGS = {"min_streak": 3, "retr_window_bars": 40, "atr_mult_stop": 1.5, "rr_multiple": 3.0, "max_bars": 60}


def _events(n, r_real, r_flip):
    """n synthetic events, byte-identical r_real/r_flip per row — lets a
    test fix exactly what each time-slice's outcome is, instead of relying
    on noisy real data to happen to land one way or another."""
    return pd.DataFrame({
        "entry_time": pd.date_range("2020-01-01", periods=n, freq="h"),
        "r_real": [r_real] * n, "r_flip": [r_flip] * n, "direction": ["bullish"] * n,
    })


def _concat(*parts):
    return pd.concat(parts, ignore_index=True).assign(
        entry_time=lambda d: pd.date_range("2020-01-01", periods=len(d), freq="h"))


@pytest.fixture(autouse=True)
def _isolated_trials(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "_TRIALS_PATH", str(tmp_path / "trials.jsonl"))
    yield


def test_rejects_fractions_that_leave_no_locked_remainder():
    events = _events(40, 1.0, -1.0)
    with pytest.raises(AssertionError):
        ex.run_deep_backtest_rr_with_lock("EURUSD=X", "5m", events, _SETTINGS, "clean_expansion_fixed_rr",
                                           research_fraction=0.8, validation_fraction=0.2)


def test_insufficient_data_below_30_events_never_computes_a_split():
    events = _events(10, 1.0, -1.0)
    trial = ex.run_deep_backtest_rr_with_lock("EURUSD=X", "5m", events, _SETTINGS, "clean_expansion_fixed_rr")
    assert trial["verdict"] == "INSUFFICIENT_DATA"
    assert "locked_mean_return" not in trial


def test_split_sizes_match_the_configured_fractions():
    # 50 events, defaults (0.6 / 0.2 / 0.2) -> exactly 30 / 10 / 10.
    research = _events(30, 1.0, -1.0)
    validation = _events(10, 1.0, -1.0)
    locked = _events(10, -5.0, 5.0)
    events = _concat(research, validation, locked)

    trial = ex.run_deep_backtest_rr_with_lock("EURUSD=X", "5m", events, _SETTINGS, "clean_expansion_fixed_rr")

    assert trial["n_train"] == 30  # research
    assert trial["n_holdout"] == 10  # validation
    assert trial["locked_n"] == 10
    assert trial["n_train"] + trial["n_holdout"] + trial["locked_n"] == len(events)


def test_a_badly_performing_locked_slice_never_flips_the_verdict():
    # The core guarantee: RESEARCH and VALIDATION are both strongly
    # positive and significant; LOCKED is strongly negative. If locked
    # data leaked into the research/validation scoring (the exact mistake
    # objective 3 exists to prevent), holdout_verdict/same_sign would flip
    # to FAILED/False. It must not.
    research = _events(30, 1.0, -1.0)
    validation = _events(10, 1.0, -1.0)
    locked = _events(10, -5.0, 5.0)
    events = _concat(research, validation, locked)

    trial = ex.run_deep_backtest_rr_with_lock("EURUSD=X", "5m", events, _SETTINGS, "clean_expansion_fixed_rr")

    assert trial["verdict"] == "SCORED"
    assert trial["holdout_verdict"] == "PASSED"
    assert trial["same_sign"] is True
    assert trial["mean_return_train"] == pytest.approx(1.0)
    assert trial["mean_return_holdout"] == pytest.approx(1.0)
    # The bad slice is recorded faithfully for monitoring...
    assert trial["locked_mean_return"] == pytest.approx(-5.0)
    # ...but never near either decision field above.


def test_locked_fields_are_absent_from_the_plain_function():
    # run_deep_backtest_rr (the existing 80/20 function, untouched by this
    # phase) must never gain locked_* fields — only the new function emits
    # them. Confirms the two stayed genuinely separate, per the plan's own
    # "does NOT touch run_deep_backtest_rr" requirement.
    research = _events(30, 1.0, -1.0)
    validation = _events(10, 1.0, -1.0)
    locked = _events(10, -5.0, 5.0)
    events = _concat(research, validation, locked)
    plain_trial = ex.run_deep_backtest_rr("EURUSD=X", "5m", events, _SETTINGS, "clean_expansion_fixed_rr")
    assert "locked_mean_return" not in plain_trial
    assert "locked_n" not in plain_trial


def test_schema_parity_with_the_plain_function_for_existing_consumers():
    # survived/bh_significant (ticker_behavior.py / load_experiment_trials)
    # only ever read these field names — a locked-split trial must carry
    # every one of them under the identical name, or it silently stops
    # being eligible for promotion despite genuinely passing.
    events = _concat(_events(30, 1.0, -1.0), _events(10, 1.0, -1.0), _events(10, -5.0, 5.0))
    locked_trial = ex.run_deep_backtest_rr_with_lock("EURUSD=X", "5m", events, _SETTINGS, "clean_expansion_fixed_rr")
    required = {"strategy_id", "mean_return_train", "p_value_train", "mean_return_holdout",
                "p_value_holdout", "holdout_verdict", "same_sign", "verdict", "n_train", "n_holdout"}
    assert required.issubset(locked_trial.keys())


def test_strategy_id_is_identical_to_the_plain_functions_for_the_same_settings():
    # split_mode/research_fraction are deliberately NOT part of identity —
    # a combo re-run through the locked function must compete for (and
    # supersede) the SAME dedup slot a plain-function trial already
    # occupies, not open a second, permanently-parallel identity.
    events = _concat(_events(30, 1.0, -1.0), _events(10, 1.0, -1.0), _events(10, -5.0, 5.0))
    plain = ex.run_deep_backtest_rr("EURUSD=X", "5m", events, _SETTINGS, "clean_expansion_fixed_rr", target_widen_mult=2.0)
    locked = ex.run_deep_backtest_rr_with_lock("EURUSD=X", "5m", events, _SETTINGS, "clean_expansion_fixed_rr",
                                                target_widen_mult=2.0)
    assert plain["strategy_id"] == locked["strategy_id"]


def test_split_mode_is_recorded_for_the_research_ledger():
    events = _concat(_events(30, 1.0, -1.0), _events(10, 1.0, -1.0), _events(10, -5.0, 5.0))
    trial = ex.run_deep_backtest_rr_with_lock("EURUSD=X", "5m", events, _SETTINGS, "clean_expansion_fixed_rr")
    assert trial["split_mode"] == "research_validation_locked"
    assert trial["research_fraction"] == 0.6
    assert trial["validation_fraction"] == 0.2


def test_run_deep_backtest_rr_unaffected_by_the_score_split_extraction():
    # Regression pin for the _score_split refactor: run_deep_backtest_rr's
    # own 80/20 behavior must be byte-identical to what it always computed,
    # not subtly changed by factoring its inner scoring out to a shared
    # function also used by run_deep_backtest_rr_with_lock.
    events = _events(40, 1.0, -1.0)
    trial = ex.run_deep_backtest_rr("EURUSD=X", "5m", events, _SETTINGS, "clean_expansion_fixed_rr", seed=0)
    assert trial["n_train"] == 32  # 40 * 0.8
    assert trial["n_holdout"] == 8
    assert trial["mean_return_train"] == pytest.approx(1.0)
    assert trial["mean_return_holdout"] == pytest.approx(1.0)
    assert trial["p_value_train"] == pytest.approx(0.0, abs=1e-9)
    assert trial["holdout_verdict"] == "PASSED"
    assert trial["same_sign"] is True
