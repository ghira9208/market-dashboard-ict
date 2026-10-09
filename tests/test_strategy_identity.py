"""compute_strategy_id — the function every later research-hardening phase
in this pass depends on: a trial, a leaderboard row, and a live decision can
only cross-check "same strategy" if this produces a stable, sensitive
fingerprint. These tests pin exactly what it's sensitive to and, just as
importantly, what it deliberately ignores (sizing, scale-out) — see the
module's own docstring for why those exclusions are deliberate, not gaps."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy_identity import compute_strategy_id, git_commit

_BASE_SETTINGS = {"min_streak": 3, "retr_window_bars": 40, "atr_mult_stop": 1.5, "rr_multiple": 3.0, "max_bars": 60}


def _id(**overrides):
    kwargs = dict(family="clean_expansion_fixed_rr", settings=_BASE_SETTINGS, target_widen_mult=1.0,
                   data_pipeline_version=1, detector_logic_version=1, cost_bps=2.0)
    kwargs.update(overrides)
    return compute_strategy_id(**kwargs)


def test_same_inputs_produce_the_same_id():
    assert _id() == _id()


def test_a_different_family_changes_the_id():
    assert _id(family="clean_retracement_resumption_fixed_rr") != _id()


def test_a_different_atr_mult_stop_changes_the_id():
    other = {**_BASE_SETTINGS, "atr_mult_stop": 2.5}
    assert _id(settings=other) != _id()


def test_a_different_rr_multiple_changes_the_id():
    other = {**_BASE_SETTINGS, "rr_multiple": 1.0}
    assert _id(settings=other) != _id()


def test_a_different_max_bars_changes_the_id():
    other = {**_BASE_SETTINGS, "max_bars": 40}
    assert _id(settings=other) != _id()


def test_a_different_target_widen_mult_changes_the_id():
    # The whole mechanism validate_target_variants.py depends on: a
    # widen=3.0 trial must be a genuinely different identity from widen=1.0
    # under an otherwise-identical settings dict, so each can independently
    # survive (or not) the BH/holdout gate on its own merits.
    assert _id(target_widen_mult=3.0) != _id(target_widen_mult=1.0)


def test_a_different_data_pipeline_version_changes_the_id():
    assert _id(data_pipeline_version=2) != _id(data_pipeline_version=1)


def test_a_different_detector_logic_version_changes_the_id():
    assert _id(detector_logic_version=2) != _id(detector_logic_version=1)


def test_a_different_cost_bps_changes_the_id():
    assert _id(cost_bps=5.0) != _id(cost_bps=2.0)


def test_settings_key_order_does_not_change_the_id():
    reordered = {"max_bars": 60, "rr_multiple": 3.0, "atr_mult_stop": 1.5, "retr_window_bars": 40, "min_streak": 3}
    assert _id(settings=reordered) == _id()


def test_an_underscore_prefixed_settings_key_is_ignored():
    # settings["_strategy_id"] gets written back onto the settings dict
    # AFTER computing the id (see run_deep_backtest_rr) — if a caller ever
    # accidentally re-fed that tagged dict back into compute_strategy_id,
    # the id must not change out from under itself by hashing its own
    # previous output.
    tagged = {**_BASE_SETTINGS, "_strategy_id": "whatever-was-here-before"}
    assert _id(settings=tagged) == _id()


def test_an_extra_key_left_in_settings_still_changes_the_id():
    # compute_strategy_id has no sizing_mode/kelly_fraction parameter of its
    # own — sizing is excluded by never being passed in (see the module
    # docstring), not by being filtered out of whatever settings dict a
    # caller hands it. If a caller mistakenly left a sizing-ish key inside
    # settings, it WOULD still change the id (settings is hashed whole) —
    # documenting that callers must pass only the real entry/stop/target/
    # time-exit settings dict, never raw account config.
    with_extra = {**_BASE_SETTINGS, "kelly_fraction": 0.9}
    assert _id(settings=with_extra) != _id()


def test_git_commit_returns_none_for_a_non_git_directory(tmp_path):
    assert git_commit(str(tmp_path)) is None


def test_git_commit_is_cached_per_directory(tmp_path, monkeypatch):
    import strategy_identity
    monkeypatch.setitem(strategy_identity._git_commit_cache, str(tmp_path), "cached-value")
    assert git_commit(str(tmp_path)) == "cached-value"


def test_git_commit_on_this_real_repo_returns_a_short_hex_hash():
    commit = git_commit(str(Path(__file__).resolve().parent.parent))
    assert commit is not None
    assert len(commit) == 12
    int(commit, 16)  # raises ValueError if not valid hex
