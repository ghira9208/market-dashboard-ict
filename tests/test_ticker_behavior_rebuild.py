"""rebuild_db()'s strategy_id-aware dedup AND fail-closed promotion gate.

Dedup (2026-10-07): pins the exact bug caught before shipping — an earlier
version compared each trial's own STORED strategy_id, which only exists on
freshly-run trials. An old trial (logged before strategy_id tagging
existed) and a fresh re-run of the IDENTICAL settings would then look like
two DIFFERENT identities (missing vs. present), and because the old one is
often more optimistic, it kept winning the best-row selection — the exact
silent-stale-data failure mode this whole mechanism exists to prevent, just
inverted. Fixed by recomputing a BEST-EFFORT _effective_strategy_id from
each row's own settings dict + today's version constants, used ONLY for
collapsing duplicate dedup entries.

Fail-closed promotion (2026-10-07, same day, stricter requirement added
after review): that same best-effort _effective_strategy_id must NEVER be
used to decide `validated` (bot_engine.py's live-tradeable gate) — doing so
would silently promote old, never-identity-tagged evidence to "this is what
live code validated," which it demonstrably isn't (no stored identity means
no proof it was ever scored under THIS methodology). `validated` requires a
real, STORED strategy_id on at least one surviving trial — a combo whose
only survivors are legacy/unversioned stays validated=False (excluded from
validated_leaderboard(), never live-tradeable) until a real re-validation
produces a properly-identified survivor."""
import json
import sqlite3
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


def test_an_untagged_old_trial_and_a_fresh_retest_of_the_same_settings_collapse_to_one_row(isolated_log):
    # The real bug: old trial has NO strategy_id field at all (predates
    # tagging); new trial has one (written by run_deep_backtest_rr). Same
    # exact settings, same methodology — must dedup to ONE row, keeping the
    # newer (here: the more honest, lower) number, with a real strategy_id.
    # The fresh trial's strategy_id must be the REAL hash run_deep_backtest_rr
    # would actually compute for these settings under TODAY's version
    # constants — same as the old trial's own retroactive fallback would
    # compute — not an arbitrary placeholder. That equality is the whole
    # point: it's what makes "same settings, same current methodology"
    # correctly collapse to one, regardless of which one happens to be
    # tagged.
    real_id = compute_strategy_id(family="clean_expansion_fixed_rr", settings=_settings(), target_widen_mult=1.0,
                                   data_pipeline_version=1, detector_logic_version=1, cost_bps=2.0)
    _write_trial(isolated_log, logged_at="2026-01-01T00:00:00+00:00", mean_return_train=0.9,
                  mean_return_holdout=0.85)  # old, optimistic, untagged
    _write_trial(isolated_log, logged_at="2026-10-07T00:00:00+00:00", mean_return_train=0.4,
                  mean_return_holdout=0.35, strategy_id=real_id)  # fresh, honest, correctly tagged

    ticker_behavior.rebuild_db()
    lb = ticker_behavior.validated_leaderboard()

    assert len(lb) == 1
    row = lb.iloc[0]
    assert row["best_strategy_id"] is not None
    # The newer trial wins (keep="last" on logged_at) — confirms the fix,
    # not just "a row exists": the stale, more optimistic number must NOT
    # be what live sizing reads.
    assert row["best_mean_return_train"] == pytest.approx(0.4)


def test_two_old_untagged_trials_of_different_settings_both_survive_dedup_independently(isolated_log):
    # Dedup must not over-collapse: genuinely different settings (even both
    # missing strategy_id) are different trials, not duplicates of each
    # other — both land in the full strategy_ticker_tf table as distinct
    # dedup survivors. (validated_leaderboard() itself correctly shows
    # NEITHER — see the fail-closed tests below: legacy/untagged evidence,
    # however strong, is never live-eligible on its own.)
    _write_trial(isolated_log, label="clean_expansion_fixed_rr · rr=3.0 · atr=1.5 · maxbars=60",
                  settings={"min_streak": 3, "retr_window_bars": 40, "atr_mult_stop": 1.5,
                            "rr_multiple": 3.0, "max_bars": 60})
    _write_trial(isolated_log, label="clean_expansion_fixed_rr · rr=2.0 · atr=1.0 · maxbars=40",
                  settings={"min_streak": 3, "retr_window_bars": 40, "atr_mult_stop": 1.0,
                            "rr_multiple": 2.0, "max_bars": 40})

    ticker_behavior.rebuild_db()
    con = sqlite3.connect(ticker_behavior._DB_PATH)
    try:
        n_scored = con.execute("SELECT n_scored FROM strategy_ticker_tf").fetchone()[0]
    finally:
        con.close()

    # One (ticker, tf_label, strategy) GROUP — best_row picks the stronger
    # of the two by mean_return_train — but n_scored=2 proves both were
    # real, independently-dedup-surviving trials underneath, not silently
    # merged into one.
    assert n_scored == 2
    assert ticker_behavior.validated_leaderboard().empty  # neither is identified — see below


def test_a_genuine_methodology_change_competes_instead_of_overwriting(isolated_log):
    # Same (ticker, tf_label, label, settings) exactly — but each trial
    # carries its OWN real strategy_id, as if logged under a genuinely
    # different DETECTOR_LOGIC_VERSION at the time (what run_deep_backtest_rr
    # actually does — it stamps whatever was true AT WRITE TIME, not
    # something rebuild_db() could ever reconstruct from settings alone).
    # The two trials must NOT collapse into one via dedup, so the newer
    # methodology's own result competes on its own merits rather than
    # silently inheriting or overwriting the old one's.
    id_v1 = compute_strategy_id(family="clean_expansion_fixed_rr", settings=_settings(), target_widen_mult=1.0,
                                 data_pipeline_version=1, detector_logic_version=1, cost_bps=2.0)
    id_v2 = compute_strategy_id(family="clean_expansion_fixed_rr", settings=_settings(), target_widen_mult=1.0,
                                 data_pipeline_version=1, detector_logic_version=2, cost_bps=2.0)
    assert id_v1 != id_v2  # sanity: the two scenarios really are different identities

    _write_trial(isolated_log, logged_at="2026-01-01T00:00:00+00:00", mean_return_train=0.3, strategy_id=id_v1)
    _write_trial(isolated_log, logged_at="2026-01-02T00:00:00+00:00", mean_return_train=0.6, strategy_id=id_v2)

    ticker_behavior.rebuild_db()
    lb = ticker_behavior.validated_leaderboard()

    # Still one leaderboard row (same ticker/tf_label/strategy triple) —
    # but n_scored reflects BOTH surviving as distinct trials underneath,
    # not one silently replacing the other.
    assert len(lb) == 1
    assert lb.iloc[0]["n_scored"] == 2
    assert lb.iloc[0]["best_strategy_id"] == id_v2  # the higher-mean_return_train one wins, honestly


def test_the_same_real_strategy_id_under_two_different_label_spellings_still_dedups(isolated_log):
    # REAL BUG, caught live 2026-10-08: validate_target_variants.py's own
    # double-widen-suffix regression wrote two trials with the IDENTICAL
    # real stored strategy_id (same settings, same target_widen_mult,
    # same version constants — genuinely the same experiment) but
    # cosmetically different label strings ("...widen=4.0" vs
    # "...widen=2.0 · widen=4.0"). With "label" previously part of the
    # dedup subset, this cosmetic difference alone was enough to defeat
    # dedup — both rows survived as if they were competing, independent
    # trials, and the older, cosmetically-broken one could still win
    # best_row on a tied or marginally higher mean_return_train. A real
    # stored strategy_id IS the identity; the label string must never be
    # a second, competing source of truth for what counts as "the same
    # trial run twice."
    same_id = compute_strategy_id(family="clean_expansion_fixed_rr", settings=_settings(), target_widen_mult=4.0,
                                   data_pipeline_version=1, detector_logic_version=1, cost_bps=2.0)
    _write_trial(isolated_log, logged_at="2026-01-01T00:00:00+00:00",
                  label="clean_expansion_fixed_rr · rr=3.0 · atr=1.5 · maxbars=60 · widen=2.0 · widen=4.0",
                  mean_return_train=0.5, strategy_id=same_id, target_widen_mult=4.0)
    _write_trial(isolated_log, logged_at="2026-01-02T00:00:00+00:00",
                  label="clean_expansion_fixed_rr · rr=3.0 · atr=1.5 · maxbars=60 · widen=4.0",
                  mean_return_train=0.5, strategy_id=same_id, target_widen_mult=4.0)

    ticker_behavior.rebuild_db()
    lb = ticker_behavior.validated_leaderboard()

    assert len(lb) == 1
    assert lb.iloc[0]["n_scored"] == 1  # deduped to ONE real trial, not two competing rows
    assert lb.iloc[0]["best_label"] == "clean_expansion_fixed_rr · rr=3.0 · atr=1.5 · maxbars=60 · widen=4.0"


def test_an_untagged_old_trial_and_a_differently_tagged_new_trial_both_survive(isolated_log):
    # The old trial has no recorded strategy_id (falls back to a retroactive
    # best-effort guess under TODAY's constants); the new trial has a real,
    # DIFFERENT stored one (e.g. logged under an older DETECTOR_LOGIC_VERSION
    # than what's active today). These must not be silently treated as the
    # same identity just because one is a fallback guess.
    other_id = compute_strategy_id(family="clean_expansion_fixed_rr", settings=_settings(), target_widen_mult=1.0,
                                    data_pipeline_version=1, detector_logic_version=99, cost_bps=2.0)
    _write_trial(isolated_log, logged_at="2026-01-01T00:00:00+00:00", mean_return_train=0.3)  # untagged
    _write_trial(isolated_log, logged_at="2026-01-02T00:00:00+00:00", mean_return_train=0.6, strategy_id=other_id)

    ticker_behavior.rebuild_db()
    lb = ticker_behavior.validated_leaderboard()

    assert len(lb) == 1
    assert lb.iloc[0]["n_scored"] == 2


# --------------------------------------------------- fail-closed promotion --

def test_a_combo_with_only_legacy_untagged_survivors_is_never_validated(isolated_log):
    # The core fail-closed requirement: even a strong, genuinely-surviving
    # trial (BH-significant, holdout PASSED, same-sign) must NOT make a
    # combo live-tradeable if it has no real recorded strategy_id. Strong
    # evidence with no proof of WHAT was actually run isn't live-eligible
    # evidence.
    _write_trial(isolated_log, mean_return_train=0.9, mean_return_holdout=0.85)  # untagged, no strategy_id at all

    ticker_behavior.rebuild_db()

    assert ticker_behavior.validated_leaderboard().empty
    con = sqlite3.connect(ticker_behavior._DB_PATH)
    try:
        row = con.execute(
            "SELECT validated, best_strategy_id, n_survived_bh, n_survived_identified FROM strategy_ticker_tf"
        ).fetchone()
    finally:
        con.close()
    validated, best_strategy_id, n_survived_bh, n_survived_identified = row
    assert validated == 0  # SQLite stores bool as 0/1
    assert best_strategy_id is None
    assert n_survived_bh == 1       # the real BH survivor count, for dashboard visibility
    assert n_survived_identified == 0  # but zero of them are live-eligible


def test_a_combo_gains_validation_the_moment_a_real_identified_survivor_lands(isolated_log):
    # The flip side, proving this isn't just "permanently stuck off" — a
    # genuine re-validation (same settings, now run through the real
    # strategy_id-tagged pipeline) makes it live-eligible immediately,
    # exactly the "becomes eligible again on real re-validation" promise.
    real_id = compute_strategy_id(family="clean_expansion_fixed_rr", settings=_settings(), target_widen_mult=1.0,
                                   data_pipeline_version=1, detector_logic_version=1, cost_bps=2.0)
    _write_trial(isolated_log, logged_at="2026-01-01T00:00:00+00:00", mean_return_train=0.3)  # legacy, untagged
    ticker_behavior.rebuild_db()
    assert ticker_behavior.validated_leaderboard().empty  # not yet — only legacy evidence exists

    _write_trial(isolated_log, logged_at="2026-01-02T00:00:00+00:00", mean_return_train=0.3, strategy_id=real_id)
    ticker_behavior.rebuild_db()

    lb = ticker_behavior.validated_leaderboard()
    assert len(lb) == 1
    assert lb.iloc[0]["best_strategy_id"] == real_id


def test_a_mix_of_legacy_and_identified_survivors_only_the_identified_one_can_win(isolated_log):
    # Even when a legacy survivor scores HIGHER than the identified one,
    # the identified one must still be what "best_*" reflects for a
    # validated combo — never silently let unversioned evidence look like
    # it's what got validated, even as a display artifact.
    real_id = compute_strategy_id(family="clean_expansion_fixed_rr", settings=_settings(), target_widen_mult=1.0,
                                   data_pipeline_version=1, detector_logic_version=1, cost_bps=2.0)
    _write_trial(isolated_log, logged_at="2026-01-01T00:00:00+00:00", mean_return_train=0.95)  # legacy, HIGHER score
    _write_trial(isolated_log, logged_at="2026-01-02T00:00:00+00:00", mean_return_train=0.30,
                  strategy_id=real_id)  # identified, lower score

    ticker_behavior.rebuild_db()
    lb = ticker_behavior.validated_leaderboard()

    assert len(lb) == 1
    assert lb.iloc[0]["best_strategy_id"] == real_id
    assert lb.iloc[0]["best_mean_return_train"] == pytest.approx(0.30)  # NOT 0.95 — the legacy row never wins display either


def test_rebuild_db_never_mutates_the_real_db_file_when_a_write_fails(isolated_log, monkeypatch):
    # Migration safety: a failure partway through must leave the EXISTING
    # db exactly as it was, never a half-updated file. Simulated here by
    # making the ticker_overview write explode after strategy_ticker_tf's
    # own write would have already succeeded on the old, no-longer-atomic
    # implementation.
    _write_trial(isolated_log, mean_return_train=0.3,
                  strategy_id=compute_strategy_id(family="clean_expansion_fixed_rr", settings=_settings(),
                                                   target_widen_mult=1.0, data_pipeline_version=1,
                                                   detector_logic_version=1, cost_bps=2.0))
    ticker_behavior.rebuild_db()  # establishes a real, working db first
    assert len(ticker_behavior.validated_leaderboard()) == 1

    import pandas as pd
    original_to_sql = pd.DataFrame.to_sql

    def _boom(self, name, *a, **k):
        if name == "ticker_overview":
            raise RuntimeError("simulated failure mid-rebuild")
        return original_to_sql(self, name, *a, **k)

    monkeypatch.setattr(pd.DataFrame, "to_sql", _boom)
    try:
        with pytest.raises(RuntimeError, match="simulated failure mid-rebuild"):
            ticker_behavior.rebuild_db()
    finally:
        # Restore directly, NOT monkeypatch.undo() — that would also
        # revert the isolated_log fixture's OWN db-path patches (same
        # monkeypatch instance), silently falling through to the REAL
        # production ticker_behavior.db for the assertion below instead
        # of this test's isolated tmp_path one. Caught exactly this way:
        # the real db still had the pre-best_target_widen_mult schema,
        # and the assertion below failed with "no such column" against
        # production data, not this test's own fixture.
        pd.DataFrame.to_sql = original_to_sql

    # The REAL (isolated) db must be untouched — still the working,
    # pre-failure state.
    assert len(ticker_behavior.validated_leaderboard()) == 1
    import os
    assert not os.path.exists(f"{ticker_behavior._DB_PATH}.tmp")  # no leftover temp file either
