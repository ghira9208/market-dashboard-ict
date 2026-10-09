"""derive_lifecycle_state / rebuild_db()'s own lifecycle_state column —
objective 8's lifecycle ladder (EXPERIMENTAL -> TRAIN_SIGNIFICANT ->
HOLDOUT_VALIDATED -> PROSPECTIVE -> LIVE, DECAYED/RETIRED as terminal
overrides), derived from signals this pipeline already computes elsewhere
rather than a new, separately-persisted state machine.

The one rule under every test here: only an explicit True promotes a
rung. False or None/unknown both refuse the promotion — this is how
"ambiguous lifecycle state defaults to EXPERIMENTAL and blocks live
entry" is actually satisfied, not asserted."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import experiments as ex
import ticker_behavior
from ticker_behavior import derive_lifecycle_state
from strategy_identity import compute_strategy_id


# --------------------------------------------------- pure derivation unit tests

def test_no_signal_at_all_defaults_to_experimental():
    assert derive_lifecycle_state() == "EXPERIMENTAL"


def test_bh_significant_alone_is_train_significant():
    assert derive_lifecycle_state(any_bh_significant=True) == "TRAIN_SIGNIFICANT"


def test_survived_is_holdout_validated_even_without_a_real_identity():
    # Legacy/unversioned evidence — real holdout-passing evidence, just
    # never run through the identity-tagged pipeline. Must show as
    # HOLDOUT_VALIDATED, not collapse to EXPERIMENTAL just because
    # has_identified_survivor is unknown/False.
    assert derive_lifecycle_state(any_bh_significant=True, any_survived=True,
                                   has_identified_survivor=False) == "HOLDOUT_VALIDATED"
    assert derive_lifecycle_state(any_bh_significant=True, any_survived=True) == "HOLDOUT_VALIDATED"


def test_identified_survivor_without_locked_or_live_confirmation_stays_holdout_validated():
    # Validated (real stored identity) but neither the new locked-slice
    # check nor live-code-match confirmation has run yet — must NOT
    # jump to PROSPECTIVE or LIVE on a guess.
    state = derive_lifecycle_state(any_survived=True, has_identified_survivor=True,
                                    has_positive_locked_result=None, strategy_id_matches_live=None)
    assert state == "HOLDOUT_VALIDATED"


def test_a_confirmed_non_positive_locked_result_also_never_promotes():
    # False, not just None, must still refuse the promotion — confirms
    # the rule is "only True promotes," not "only False/known-bad blocks."
    state = derive_lifecycle_state(any_survived=True, has_identified_survivor=True,
                                    has_positive_locked_result=False)
    assert state == "HOLDOUT_VALIDATED"


def test_identified_survivor_with_a_positive_locked_result_is_prospective():
    state = derive_lifecycle_state(any_survived=True, has_identified_survivor=True,
                                    has_positive_locked_result=True)
    assert state == "PROSPECTIVE"


def test_prospective_requires_a_real_identity_not_just_a_positive_locked_number():
    # has_positive_locked_result=True alone, without has_identified_
    # survivor, must not promote past HOLDOUT_VALIDATED — the locked
    # slice means nothing if the combo's own identity isn't real.
    state = derive_lifecycle_state(any_survived=True, has_identified_survivor=False,
                                    has_positive_locked_result=True)
    assert state == "HOLDOUT_VALIDATED"


def test_live_requires_both_identity_and_a_confirmed_live_code_match():
    state = derive_lifecycle_state(any_survived=True, has_identified_survivor=True,
                                    strategy_id_matches_live=True)
    assert state == "LIVE"


def test_a_known_false_live_match_falls_back_not_to_prospective_but_holdout_validated():
    # A real, CONFIRMED mismatch (not unknown) — same semantics as phase
    # (b)'s own live enforcement: code changed since this validated, so
    # it's not LIVE; it's also not "prospective, basically live" — it's
    # exactly as validated as it ever was, nothing more.
    state = derive_lifecycle_state(any_survived=True, has_identified_survivor=True,
                                    strategy_id_matches_live=False)
    assert state == "HOLDOUT_VALIDATED"


def test_decayed_overrides_an_otherwise_live_eligible_combo():
    state = derive_lifecycle_state(any_survived=True, has_identified_survivor=True,
                                    strategy_id_matches_live=True, is_decayed=True)
    assert state == "DECAYED"


def test_retired_overrides_everything_including_decayed():
    state = derive_lifecycle_state(any_survived=True, has_identified_survivor=True,
                                    strategy_id_matches_live=True, is_decayed=True, is_retired=True)
    assert state == "RETIRED"


def test_lifecycle_states_constant_lists_every_rung_derive_can_reach():
    import inspect
    src = inspect.getsource(derive_lifecycle_state)
    for state in ticker_behavior.LIFECYCLE_STATES:
        assert state in src, f"{state} listed in LIFECYCLE_STATES but never returned by derive_lifecycle_state"


# ------------------------------------------- integration via rebuild_db()

@pytest.fixture
def isolated_log(tmp_path, monkeypatch):
    trials_path = tmp_path / "trials.jsonl"
    db_path = tmp_path / "ticker_behavior.db"
    decay_path = tmp_path / "decay_flags.json"
    monkeypatch.setattr(ex, "_TRIALS_PATH", str(trials_path))
    monkeypatch.setattr(ticker_behavior, "_DB_PATH", str(db_path))
    monkeypatch.setattr(ticker_behavior, "_DECAY_CACHE_PATH", str(decay_path))
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


def _lifecycle_row(ticker="EURUSD=X", tf_label="5m", strategy="clean_expansion_fixed_rr"):
    rows = pd.read_sql(f"SELECT * FROM strategy_ticker_tf WHERE ticker='{ticker}' AND tf_label='{tf_label}' "
                        f"AND strategy='{strategy}'", __import__("sqlite3").connect(ticker_behavior._DB_PATH))
    assert len(rows) == 1
    return rows.iloc[0]


def test_a_clean_bh_significant_only_trial_is_train_significant(isolated_log):
    # Significant on train (p=0.01, single candidate) but fails holdout —
    # the real "looked promising, didn't survive" case.
    _write_trial(isolated_log, p_value_train=0.01, mean_return_holdout=-0.2,
                 holdout_verdict="FAILED", same_sign=False)
    ticker_behavior.rebuild_db()
    assert _lifecycle_row()["lifecycle_state"] == "TRAIN_SIGNIFICANT"


def test_a_real_identified_survivor_with_no_locked_data_is_holdout_validated(isolated_log):
    sid = compute_strategy_id(family="clean_expansion_fixed_rr", settings=_settings(), target_widen_mult=1.0,
                               data_pipeline_version=ex.research_data.DATA_PIPELINE_VERSION,
                               detector_logic_version=ex.DETECTOR_LOGIC_VERSIONS.get("clean_expansion", 0),
                               cost_bps=ex.DEFAULT_COST_BPS)
    _write_trial(isolated_log, strategy_id=sid, target_widen_mult=1.0)
    ticker_behavior.rebuild_db()
    row = _lifecycle_row()
    assert row["validated"] == 1
    assert row["lifecycle_state"] == "HOLDOUT_VALIDATED"


def test_a_legacy_untagged_survivor_is_holdout_validated_but_not_validated_for_trading(isolated_log):
    _write_trial(isolated_log)  # no strategy_id at all — pre-tagging evidence
    ticker_behavior.rebuild_db()
    row = _lifecycle_row()
    assert row["validated"] == 0  # not live-tradeable...
    assert row["lifecycle_state"] == "HOLDOUT_VALIDATED"  # ...but the evidence itself is real


def test_a_positive_locked_result_on_an_identified_survivor_is_prospective(isolated_log):
    sid = compute_strategy_id(family="clean_expansion_fixed_rr", settings=_settings(), target_widen_mult=1.0,
                               data_pipeline_version=ex.research_data.DATA_PIPELINE_VERSION,
                               detector_logic_version=ex.DETECTOR_LOGIC_VERSIONS.get("clean_expansion", 0),
                               cost_bps=ex.DEFAULT_COST_BPS)
    _write_trial(isolated_log, strategy_id=sid, target_widen_mult=1.0, locked_mean_return=0.3)
    ticker_behavior.rebuild_db()
    assert _lifecycle_row()["lifecycle_state"] == "PROSPECTIVE"


def test_a_non_positive_locked_result_stays_holdout_validated(isolated_log):
    sid = compute_strategy_id(family="clean_expansion_fixed_rr", settings=_settings(), target_widen_mult=1.0,
                               data_pipeline_version=ex.research_data.DATA_PIPELINE_VERSION,
                               detector_logic_version=ex.DETECTOR_LOGIC_VERSIONS.get("clean_expansion", 0),
                               cost_bps=ex.DEFAULT_COST_BPS)
    _write_trial(isolated_log, strategy_id=sid, target_widen_mult=1.0, locked_mean_return=-0.1)
    ticker_behavior.rebuild_db()
    assert _lifecycle_row()["lifecycle_state"] == "HOLDOUT_VALIDATED"


def test_a_decayed_combo_shows_decayed_even_though_it_would_otherwise_validate(isolated_log):
    sid = compute_strategy_id(family="clean_expansion_fixed_rr", settings=_settings(), target_widen_mult=1.0,
                               data_pipeline_version=ex.research_data.DATA_PIPELINE_VERSION,
                               detector_logic_version=ex.DETECTOR_LOGIC_VERSIONS.get("clean_expansion", 0),
                               cost_bps=ex.DEFAULT_COST_BPS)
    _write_trial(isolated_log, strategy_id=sid, target_widen_mult=1.0)
    with open(ticker_behavior._DECAY_CACHE_PATH, "w") as f:
        json.dump({"checked_at": 0, "decayed": [
            {"ticker": "EURUSD=X", "tf_label": "5m", "strategy": "clean_expansion_fixed_rr"}]}, f)
    ticker_behavior.rebuild_db()
    row = _lifecycle_row()
    assert row["validated"] == 1  # the dedicated promotion gate is unaffected...
    assert row["lifecycle_state"] == "DECAYED"  # ...but its CURRENT lifecycle state reflects reality


def test_a_retired_combo_shows_retired_even_with_a_real_survivor(isolated_log):
    sid = compute_strategy_id(family="clean_expansion_fixed_rr", settings=_settings(), target_widen_mult=1.0,
                               data_pipeline_version=ex.research_data.DATA_PIPELINE_VERSION,
                               detector_logic_version=ex.DETECTOR_LOGIC_VERSIONS.get("clean_expansion", 0),
                               cost_bps=ex.DEFAULT_COST_BPS)
    _write_trial(isolated_log, strategy_id=sid, target_widen_mult=1.0,
                 invalidated_reason="manually pulled for review")
    ticker_behavior.rebuild_db()
    assert _lifecycle_row()["lifecycle_state"] == "RETIRED"


def test_a_big_enough_losing_locked_sample_auto_retires(isolated_log):
    # The actual automation the user asked for 2026-10-08 ("shouldn't
    # there be an automation for such a thing? let's rely no work on
    # manual") — a real, big-enough-to-trust locked-slice loss on an
    # otherwise-validated combo must retire it with ZERO manual action,
    # and the reason must make clear this fired automatically.
    sid = compute_strategy_id(family="clean_expansion_fixed_rr", settings=_settings(), target_widen_mult=1.0,
                               data_pipeline_version=ex.research_data.DATA_PIPELINE_VERSION,
                               detector_logic_version=ex.DETECTOR_LOGIC_VERSIONS.get("clean_expansion", 0),
                               cost_bps=ex.DEFAULT_COST_BPS)
    _write_trial(isolated_log, strategy_id=sid, target_widen_mult=1.0,
                  locked_mean_return=-0.25, locked_n=15)
    ticker_behavior.rebuild_db()

    row = _lifecycle_row()
    assert row["lifecycle_state"] == "RETIRED"
    assert ("EURUSD=X", "5m", "clean_expansion_fixed_rr") in ticker_behavior.retired_combo_keys()
    invalidated = ticker_behavior.recently_invalidated()
    reason = invalidated[invalidated["ticker"] == "EURUSD=X"]["reason"].iloc[0]
    assert "AUTO-RETIRED" in reason
    assert "15 events" in reason


def test_a_thin_losing_locked_sample_does_not_auto_retire(isolated_log):
    sid = compute_strategy_id(family="clean_expansion_fixed_rr", settings=_settings(), target_widen_mult=1.0,
                               data_pipeline_version=ex.research_data.DATA_PIPELINE_VERSION,
                               detector_logic_version=ex.DETECTOR_LOGIC_VERSIONS.get("clean_expansion", 0),
                               cost_bps=ex.DEFAULT_COST_BPS)
    _write_trial(isolated_log, strategy_id=sid, target_widen_mult=1.0,
                  locked_mean_return=-0.25, locked_n=5)  # below _AUTO_RETIRE_MIN_LOCKED_N
    ticker_behavior.rebuild_db()

    row = _lifecycle_row()
    assert row["lifecycle_state"] != "RETIRED"
    assert ("EURUSD=X", "5m", "clean_expansion_fixed_rr") not in ticker_behavior.retired_combo_keys()


def test_a_manual_invalidation_reason_is_never_overwritten_by_auto_retirement(isolated_log):
    sid = compute_strategy_id(family="clean_expansion_fixed_rr", settings=_settings(), target_widen_mult=1.0,
                               data_pipeline_version=ex.research_data.DATA_PIPELINE_VERSION,
                               detector_logic_version=ex.DETECTOR_LOGIC_VERSIONS.get("clean_expansion", 0),
                               cost_bps=ex.DEFAULT_COST_BPS)
    _write_trial(isolated_log, strategy_id=sid, target_widen_mult=1.0, locked_mean_return=-0.25, locked_n=15,
                  invalidated_reason="manually pulled for an unrelated reason")
    ticker_behavior.rebuild_db()

    invalidated = ticker_behavior.recently_invalidated()
    reason = invalidated[invalidated["ticker"] == "EURUSD=X"]["reason"].iloc[0]
    assert reason == "manually pulled for an unrelated reason"  # the human's own reason wins, verbatim
    assert "AUTO-RETIRED" not in reason


def test_auto_retirement_reverses_itself_once_a_fresher_locked_result_turns_positive(isolated_log):
    # Reversible by construction, no manual un-retire step — re-evaluated
    # fresh from whatever the CURRENT trial log says, every single call.
    sid = compute_strategy_id(family="clean_expansion_fixed_rr", settings=_settings(), target_widen_mult=1.0,
                               data_pipeline_version=ex.research_data.DATA_PIPELINE_VERSION,
                               detector_logic_version=ex.DETECTOR_LOGIC_VERSIONS.get("clean_expansion", 0),
                               cost_bps=ex.DEFAULT_COST_BPS)
    _write_trial(isolated_log, logged_at="2026-01-01T00:00:00+00:00", strategy_id=sid, target_widen_mult=1.0,
                  locked_mean_return=-0.25, locked_n=15)
    ticker_behavior.rebuild_db()
    assert _lifecycle_row()["lifecycle_state"] == "RETIRED"

    # A fresher re-sweep of the SAME real identity, now with a genuinely
    # positive locked result.
    _write_trial(isolated_log, logged_at="2026-01-02T00:00:00+00:00", strategy_id=sid, target_widen_mult=1.0,
                  locked_mean_return=0.30, locked_n=15)
    ticker_behavior.rebuild_db()

    row = _lifecycle_row()
    assert row["lifecycle_state"] != "RETIRED"
    assert ("EURUSD=X", "5m", "clean_expansion_fixed_rr") not in ticker_behavior.retired_combo_keys()


def test_an_insufficient_trial_with_no_real_signal_is_experimental(isolated_log):
    _write_trial(isolated_log, p_value_train=0.9, mean_return_holdout=-0.3,
                 holdout_verdict="FAILED", same_sign=False)
    ticker_behavior.rebuild_db()
    assert _lifecycle_row()["lifecycle_state"] == "EXPERIMENTAL"
