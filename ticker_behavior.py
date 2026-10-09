"""
A queryable summary of experiments.py's own accumulated trial log
(.cache/experiment_trials.jsonl — every deep-backtest ever run this
project's whole history, BH-corrected across the WHOLE log by
load_experiment_trials()) — one row per (ticker, timeframe, strategy
family) telling you, at a glance, "does anything validated exist here,
and what."

Genuinely additive: reads load_experiment_trials()'s own output,
writes nothing back to the trial log, changes no existing behavior.
rebuild_db() is the only way data gets in — call it after any new sweep
so the summary reflects what's actually been tried; nothing here re-runs
a backtest or fetches market data itself.

Why a real SQLite file (ticker_behavior.db) instead of just re-querying
load_experiment_trials() on demand: that function re-parses and re-BH-
corrects the ENTIRE trial log (88k+ rows and growing) on every call —
fine for one-off checks, wasteful for "let me look up three tickers back
to back." This is a cheap, indexed, disk-backed cache of the aggregation,
same reasoning as backtest_results.db already used elsewhere in this
project for the separate custom-rule sweep engine — a different dataset,
same "don't make the user wait on a re-scan every lookup" idea.
"""
import json
import os
import re
import sqlite3

import pandas as pd

import data as research_data
import experiments as ex
import strategy_identity

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ticker_behavior.db")
_DECAY_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "decay_flags.json")

LIFECYCLE_STATES = ("EXPERIMENTAL", "TRAIN_SIGNIFICANT", "HOLDOUT_VALIDATED", "PROSPECTIVE",
                     "LIVE", "DECAYED", "RETIRED")

_AUTO_RETIRE_MIN_LOCKED_N = 10
"""The minimum locked-slice sample size rebuild_db() requires before it
will AUTOMATICALLY retire a combo on a bad locked result (see rebuild_db's
own auto-retirement block). Added 2026-10-08, at the user's own explicit
request ("shouldn't there be an automation for such a thing? let's rely
no work on manual") — this is a deliberate reversal of run_deep_backtest_
rr_with_lock's own earlier-written stance ("never automatic, never
triggered by a locked score looking good or bad"); that was this
project's own conservative DEFAULT, not a constraint the user can't
override, and they've now explicitly chosen automation over a standing
manual review step. The floor itself stays conservative regardless: below
~10 events a locked-slice mean can flip sign from pure noise (the same
"manufacture no edge from noise" reasoning _kelly_fraction's own small-
sample floor already applies elsewhere) — retiring a real, validated
combo is a far higher-stakes, harder-to-reverse action than sizing a
position smaller, so this never fires on a thin sample, even an
automatic one. Reversible by construction, every single rebuild: this
isn't a persisted flag, it's recomputed fresh from the CURRENT trial log
each call — a combo whose locked evidence turns positive on a later
re-sweep simply stops qualifying, no manual un-retire step needed either."""
"""Objective 8's lifecycle ladder — a pure DERIVATION over signals that
mostly already exist elsewhere (BH correction, the holdout gate,
validate_target_variants.py's locked slice, revalidate_combos.py's decay
check, the manual invalidated_combos reason log), not a new, separately-
persisted state machine. See derive_lifecycle_state's own docstring for
how each rung is reached."""


def _decayed_combo_keys():
    """{(ticker, tf_label, strategy)} currently failing revalidate_
    combos.py's own weekly re-validation against fresh data — same file,
    same format, same honesty convention (empty set, not an error, if
    the check has never run) as bot_engine.py's own _decayed_combo_keys.
    Read directly here (same repo, relative path) rather than imported
    from bot_engine.py, which lives in a different repo entirely and
    would be a far heavier, real cross-project dependency for one JSON
    file both sides already read independently."""
    if not os.path.exists(_DECAY_CACHE_PATH):
        return set()
    try:
        with open(_DECAY_CACHE_PATH) as f:
            data = json.load(f)
    except Exception:
        return set()
    return {(d["ticker"], d["tf_label"], d["strategy"]) for d in data.get("decayed", [])}


def derive_lifecycle_state(*, any_bh_significant=None, any_survived=None, has_identified_survivor=None,
                            is_decayed=False, is_retired=False, has_positive_locked_result=None,
                            strategy_id_matches_live=None):
    """EXPERIMENTAL -> TRAIN_SIGNIFICANT -> HOLDOUT_VALIDATED -> PROSPECTIVE
    -> LIVE, with DECAYED/RETIRED as terminal overrides — derived from
    signals this pipeline already computes elsewhere, never a second,
    independently-maintained state machine:

      any_bh_significant        g["bh_significant"].any() — passed BH correction at least once
      any_survived               g["survived"].any() — BH + raw holdout(p<0.05, positive) + same_sign,
                                  TODAY's existing "survived" flag, unchanged
      has_identified_survivor    this combo's own `validated` flag (a REAL, stored strategy_id —
                                  never the dedup step's retroactive guess; see rebuild_db's own
                                  fail-closed-on-identity comment)
      has_positive_locked_result the best identified trial's own locked_mean_return > 0
                                  (run_deep_backtest_rr_with_lock's genuinely untouched third slice)
      strategy_id_matches_live   does the combo's recorded identity match CURRENT live code —
                                  this module has no way to know that itself (it would need
                                  bot_engine.py's own live constants); always None from here,
                                  left for a live-side caller (bot_engine.py, which already
                                  computes this via _strategy_id_mismatch) to supply for real

    RULE, applied at every rung: only an explicit True promotes. False
    OR None/unknown both refuse that promotion — this is deliberately
    how "ambiguous defaults to EXPERIMENTAL and blocks live entry" is
    satisfied: nothing here is ever promoted on a guess, so a combo with
    no confirmed signal at all falls all the way to EXPERIMENTAL, and a
    combo missing ONLY the live-identity confirmation can rise no higher
    than PROSPECTIVE, never LIVE.

    DECAYED/RETIRED are checked first and win outright regardless of any
    other signal — a combo that once validated but no longer holds up
    (DECAYED, revalidate_combos.py) or was manually pulled (RETIRED,
    this module's own invalidated_combos table) is not "still
    HOLDOUT_VALIDATED really" for display purposes; its current,
    operative state is the override."""
    if is_retired:
        return "RETIRED"
    if is_decayed:
        return "DECAYED"
    if has_identified_survivor is True and strategy_id_matches_live is True:
        return "LIVE"
    if has_identified_survivor is True and has_positive_locked_result is True:
        return "PROSPECTIVE"
    if any_survived is True:
        return "HOLDOUT_VALIDATED"
    if any_bh_significant is True:
        return "TRAIN_SIGNIFICANT"
    return "EXPERIMENTAL"

# The original detector-reaction strategy family (build_experiment_events)
# logs its OWN detector name as the whole label ("FVG", "Order Block", ...)
# with settings baked into the `settings` column instead of the label
# string, unlike every later strategy here whose label reads
# "STRATEGY NAME · param=value · ...". Splitting on " · " already isolates
# those correctly; the four detector names just happen to have no " · "
# in them at all, so they fall out of the same split with no special case.
_STRATEGY_RE = re.compile(r"\s*·\s*")

# The same tf_label spelling drift experiments.load_experiment_trials()
# itself already normalizes at its own root (TF_LABEL_NORMALIZE) — reused
# here rather than kept as a second, driftable copy. Since that
# normalization already happens inside load_experiment_trials(), every
# tf_label this module ever sees back from it is ALREADY canonical; this
# alias exists only for survivors_ui.py, which needs to normalize a
# tf_label it read from THIS module's own SQLite db the same way before
# using it as an app.py TIMEFRAMES[...] key.
def normalize_tf(tf_label):
    return ex.TF_LABEL_NORMALIZE.get(tf_label, tf_label)


_normalize_tf = normalize_tf


def _strategy_family(label):
    if not isinstance(label, str) or not label:
        return "unknown"
    return _STRATEGY_RE.split(label, maxsplit=1)[0].strip()


def rebuild_db():
    """Re-reads the WHOLE trial log and rewrites ticker_behavior.db from
    scratch (drop + recreate both tables) — the trial log is the only
    source of truth here, so a full rebuild is simpler and always
    correct, versus trying to incrementally patch a stale summary."""
    trials = ex.load_experiment_trials()
    if trials.empty:
        return
    trials = trials.copy()
    trials["tf_label"] = trials["tf_label"].map(_normalize_tf)
    trials["strategy"] = trials["label"].map(_strategy_family)

    # Found live (2026-09-29): re-validating a combo (revalidate_combos.py,
    # or any re-run of the exact same settings after a methodology change
    # like adding a real trading-cost assumption) APPENDS a new trial under
    # the identical (ticker, tf_label, label) — it never replaces the old
    # one. Without this dedup, best_row = idxmax(mean_return_train) below
    # picks across BOTH freely, and an older, more optimistic trial for the
    # exact same settings always wins over a newer, more honest one (a
    # frictionless pre-cost-model trial beats its own cost-aware re-run on
    # paper every time) — confirmed directly: ETH-USD 5m's own leaderboard
    # entry was still serving mean_return_holdout=1.879 from 2026-09-27,
    # not the 1.229 its own 2026-09-29 cost-aware re-validation produced.
    # Kelly sizing reads this number directly, so a stale duplicate here
    # means real position sizing bets against an edge larger than what's
    # actually been validated. Keeping only the newest trial per exact
    # settings (before ever comparing DIFFERENT settings against each
    # other) fixes this without touching load_experiment_trials()'s own
    # BH correction, which already ran across the full, undeduped log.
    #
    # strategy_id: prefer each row's own STORED value (written by
    # run_deep_backtest_rr using whatever DETECTOR_LOGIC_VERSIONS/
    # DATA_PIPELINE_VERSION/DEFAULT_COST_BPS were ACTUALLY in effect at that
    # exact write time — the only honest record of that, since none of
    # those constants are themselves stored anywhere else) — and only fall
    # back to recomputing from the row's own settings dict, under TODAY's
    # version constants, when nothing was stored at all (every trial
    # logged before this field existed).
    #
    # CONFIRMED AS A REAL BUG caught before shipping, in an earlier version
    # of this that used ONLY the stored value with no fallback: an old
    # trial (missing strategy_id — pandas NaN) and a fresh re-run of the
    # IDENTICAL settings (has one) compared as two DIFFERENT identities
    # purely because one happened to be tagged and the other didn't — and
    # since the old, untagged trial is usually the more optimistic one, it
    # kept winning best-row selection. The silent-overwrite failure mode
    # this whole mechanism exists to prevent, just inverted. A SECOND
    # version that recomputed for EVERY row unconditionally (ignoring the
    # stored value even when present) fixed that but broke the opposite,
    # rarer case: two trials logged under a REAL detector-version bump
    # between them, with byte-identical settings, retroactively recompute
    # to the SAME id today (today's version constants are uniform, and the
    # old settings dict never recorded what version actually ran) — so a
    # genuine methodology change would wrongly collapse instead of
    # competing. Preferring the stored value fixes both: new trials keep
    # their true at-write-time identity; only genuinely unrecorded history
    # falls back to a best-effort reconstruction.
    def _effective_id(r):
        stored = r.get("strategy_id") if hasattr(r, "get") else r["strategy_id"] if "strategy_id" in r else None
        if pd.notna(stored):
            return stored
        if not isinstance(r["settings"], dict):
            return None
        family = r["strategy"]
        detector_family = family[: -len("_fixed_rr")] if family.endswith("_fixed_rr") else family
        return strategy_identity.compute_strategy_id(
            family=family, settings=r["settings"], target_widen_mult=1.0,
            data_pipeline_version=research_data.DATA_PIPELINE_VERSION,
            detector_logic_version=ex.DETECTOR_LOGIC_VERSIONS.get(detector_family, 0),
            cost_bps=ex.DEFAULT_COST_BPS)

    if "strategy_id" not in trials.columns:
        trials["strategy_id"] = None
    trials["_effective_strategy_id"] = trials.apply(_effective_id, axis=1)
    # "label" deliberately NOT in this subset (removed 2026-10-08) — REAL
    # BUG caught live re-running validate_target_variants.py after the
    # phase-(d) switch to run_deep_backtest_rr_with_lock: two trials for
    # the IDENTICAL real identity (same stored strategy_id, same settings,
    # same target_widen_mult — confirmed directly: byte-identical
    # mean_return_train to 6 decimals) survived as separate, competing
    # rows purely because one had a clean label and the other had
    # validate_target_variants.py's own cosmetic double-widen-suffix bug
    # ("...widen=2.0 · widen=4.0") baked into its label string. With
    # "label" in the dedup key, that cosmetic difference was enough to
    # defeat dedup entirely — the stacked-label trial kept tying/beating
    # the clean one on mean_return_train and winning best_row, which is
    # exactly how the suffix bug could never fully self-correct even
    # after the _base_label() fix stopped it from compounding further.
    # _effective_strategy_id already IS the real identity (settings,
    # target_widen_mult, and version constants, byte-for-byte) — the
    # label string is a cosmetic, human-readable rendering of the same
    # thing, never a second source of truth. Two trials sharing identity
    # are duplicates regardless of how their label happens to be spelled;
    # keeping "last" (newest) is what actually lets a later, cleaner
    # re-run supersede an earlier, cosmetically-broken one.
    trials = trials.sort_values("logged_at").drop_duplicates(
        subset=["ticker", "tf_label", "_effective_strategy_id"], keep="last")

    scored = trials[trials["verdict"] == "SCORED"].copy()

    # Both hoisted above the main loop — real, already-existing signals
    # (revalidate_combos.py's own decay check, this module's own manual-
    # invalidation log), computed once rather than per combo. See
    # derive_lifecycle_state's own docstring for how each feeds in.
    decayed_keys = _decayed_combo_keys()
    retired_keys = (set(zip(trials.loc[trials["invalidated_reason"].notna(), "ticker"],
                             trials.loc[trials["invalidated_reason"].notna(), "tf_label"],
                             trials.loc[trials["invalidated_reason"].notna(), "strategy"]))
                    if "invalidated_reason" in trials.columns else set())

    rows = []
    auto_retired_rows = []  # merged into invalidated_combos below, alongside the manual ones
    for (ticker, tf_label, strategy), g in scored.groupby(["ticker", "tf_label", "strategy"]):
        survivors = g[g["survived"] == True]  # noqa: E712 — pandas bool column, not a real bool
        # FAIL CLOSED on identity (2026-10-07): "validated" (what
        # bot_engine.py's validated_leaderboard() filters WHERE validated=1
        # — i.e. live-tradeable) now requires a survivor with a REAL,
        # STORED strategy_id — the one run_deep_backtest_rr itself computed
        # and wrote at the moment it actually ran, never the dedup step's
        # own _effective_strategy_id (that field is a best-effort
        # RETROACTIVE guess for collapsing an old/new duplicate pair, fine
        # for bookkeeping, NOT a real recorded identity). A combo whose
        # only survivor(s) predate strategy_id tagging is legacy/unversioned
        # evidence — real, worth showing on a dashboard, but never silently
        # promoted to live-tradeable just because it happens to be the
        # highest-scoring row in its group. It becomes eligible again the
        # moment a real re-validation (revalidate_combos.py,
        # validate_target_variants.py, any fresh run_deep_backtest_rr call)
        # produces a properly-identified survivor for it.
        identified_survivors = survivors[survivors["strategy_id"].notna()]
        validated = bool(len(identified_survivors) > 0)
        # best_row still reflects the single strongest candidate for
        # DISPLAY (best_label/best_mean_return_* etc. — a dashboard showing
        # "nothing validated here" with zero context isn't more honest,
        # it's just less useful) — identified survivors win that choice
        # whenever any exist (so every "best_*" field for a VALIDATED
        # combo is internally consistent with what's actually live-
        # eligible); the broader survivor/all-scored set is the display
        # fallback ONLY when nothing here is validated, and `validated`
        # itself never depends on which branch this took.
        if validated:
            best = identified_survivors
        elif len(survivors):
            best = survivors
        else:
            best = g
        best_row = best.loc[best["mean_return_train"].idxmax()]
        locked_mean = best_row.get("locked_mean_return")
        locked_n = best_row.get("locked_n")
        has_positive_locked_result = bool(locked_mean > 0) if pd.notna(locked_mean) else None
        # Automatic retirement on a real, big-enough-to-trust locked-slice
        # loss — see _AUTO_RETIRE_MIN_LOCKED_N's own docstring for why this
        # exists and why it still refuses to fire on a thin sample even
        # though it's now automatic. Only ever considered for a combo that
        # actually has a real identified survivor (validated) — there is
        # no live-tradeable status to retire from otherwise.
        auto_retired_reason = None
        if (validated and pd.notna(locked_mean) and pd.notna(locked_n)
                and locked_n >= _AUTO_RETIRE_MIN_LOCKED_N and locked_mean <= 0):
            auto_retired_reason = (
                f"AUTO-RETIRED: the genuinely untouched locked slice averaged {locked_mean:+.3f}R over "
                f"{int(locked_n)} events, contradicting this combo's own research "
                f"({best_row.get('mean_return_train'):+.3f}R train) and validation "
                f"({best_row.get('mean_return_holdout'):+.3f}R holdout) result. Automatic — "
                f"re-evaluated fresh every rebuild_db() call; a later re-sweep with a positive locked "
                f"result un-retires this combo on its own, no manual action either way."
            )
            auto_retired_rows.append({"ticker": ticker, "tf_label": tf_label, "strategy": strategy,
                                       "n_trials_affected": int(len(g)), "reason": auto_retired_reason})
        is_retired = (ticker, tf_label, strategy) in retired_keys or auto_retired_reason is not None
        lifecycle_state = derive_lifecycle_state(
            any_bh_significant=bool(g["bh_significant"].fillna(False).any()),
            any_survived=bool(len(survivors) > 0), has_identified_survivor=validated,
            is_decayed=(ticker, tf_label, strategy) in decayed_keys,
            is_retired=is_retired,
            has_positive_locked_result=has_positive_locked_result,
            # strategy_id_matches_live: always unknown from the research
            # side alone (see derive_lifecycle_state's own docstring) —
            # this ceilings the STORED lifecycle_state at PROSPECTIVE;
            # only a live-side caller with real identity-match knowledge
            # (bot_engine.py) can ever confirm LIVE.
            strategy_id_matches_live=None,
        )
        rows.append({
            "ticker": ticker, "tf_label": tf_label, "strategy": strategy,
            "n_settings_tried": int(trials[(trials["ticker"] == ticker) & (trials["tf_label"] == tf_label)
                                            & (trials["strategy"] == strategy)].shape[0]),
            "n_scored": int(len(g)),
            "n_survived_bh": int(len(survivors)),
            "n_survived_identified": int(len(identified_survivors)),
            "pct_same_sign": float(g["same_sign"].fillna(False).mean()),
            "pct_holdout_passed": float((g["holdout_verdict"] == "PASSED").mean()),
            "best_mean_return_train": float(best_row["mean_return_train"]),
            "best_mean_return_holdout": float(best_row["mean_return_holdout"])
            if pd.notna(best_row["mean_return_holdout"]) else None,
            # Feeds bot_engine.py's uncertainty-aware Kelly sizing
            # (_kelly_fraction_lcb) — a lower confidence bound on
            # mean_return_holdout needs the holdout's own spread and
            # sample size, not just its point estimate. None (never 0 or
            # a guessed default) whenever a trial predates std_return_
            # holdout being a persisted field — an honest "can't compute a
            # bound from this," which the live side's own fallback
            # (flat-risk, never a fabricated bound) is built to handle.
            "best_std_return_holdout": float(best_row.get("std_return_holdout"))
            if pd.notna(best_row.get("std_return_holdout")) else None,
            "best_n_holdout": int(best_row.get("n_holdout"))
            if pd.notna(best_row.get("n_holdout")) else None,
            "best_label": str(best_row["label"]),
            # Only ever the REAL stored value, and only when validated —
            # never _effective_strategy_id (that's the dedup's own
            # best-effort reconstruction, explicitly not a real identity
            # per this block's own docstring above).
            "best_strategy_id": str(best_row["strategy_id"]) if validated and pd.notna(best_row["strategy_id"]) else None,
            # 1.0 (never widened) whenever there's no real identified best
            # row to read it from — the honest default, matching what every
            # pre-strategy_id trial actually tested and what an unvalidated
            # combo must never claim to have independently earned.
            "best_target_widen_mult": (float(best_row.get("target_widen_mult"))
                                        if validated and pd.notna(best_row.get("target_widen_mult")) else 1.0),
            "validated": validated,
            "lifecycle_state": lifecycle_state,
            "last_tested_at": str(g["logged_at"].max()),
        })
    strategy_ticker_tf = pd.DataFrame(rows)

    overview_rows = []
    for (ticker, tf_label), g in strategy_ticker_tf.groupby(["ticker", "tf_label"]):
        validated = g[g["validated"]]
        best = validated if len(validated) else g
        best_row = best.loc[best["best_mean_return_train"].idxmax()] if len(best) else None
        overview_rows.append({
            "ticker": ticker, "tf_label": tf_label,
            "n_strategies_tested": int(len(g)),
            "n_strategies_validated": int(len(validated)),
            "best_strategy": str(best_row["strategy"]) if best_row is not None else None,
            "best_return_train": float(best_row["best_mean_return_train"]) if best_row is not None else None,
            "best_return_holdout": (float(best_row["best_mean_return_holdout"])
                                     if best_row is not None and best_row["best_mean_return_holdout"] is not None
                                     else None),
        })
    ticker_overview = pd.DataFrame(overview_rows)

    # Combos someone manually flipped holdout_verdict to FAILED on, with a
    # reason attached (e.g. the 2026-10-06 discovery-gate retrofit) — a real,
    # deliberate removal from the tradeable universe, as opposed to a combo
    # that simply never passed. One row per (ticker, tf_label, strategy),
    # since invalidated_reason is written identically across every trial row
    # for that combo. Kept as its OWN table (not a column tacked onto
    # strategy_ticker_tf) because it answers a different question — "what
    # got pulled and why" — not "what's currently valid."
    invalidated = trials[trials["invalidated_reason"].notna()] if "invalidated_reason" in trials.columns else trials.iloc[0:0]
    invalidated_rows = []
    if len(invalidated):
        for (ticker, tf_label, strategy), g in invalidated.groupby(["ticker", "tf_label", "strategy"]):
            invalidated_rows.append({
                "ticker": ticker, "tf_label": tf_label, "strategy": strategy,
                "n_trials_affected": int(len(g)),
                "reason": str(g["invalidated_reason"].iloc[0]),
            })
    # Auto-retired combos (see _AUTO_RETIRE_MIN_LOCKED_N's own docstring)
    # merge into the SAME table — one real "what's pulled and why" answer,
    # regardless of whether a human or the locked-slice check pulled it.
    # A manual reason always wins on overlap (a deliberate human action
    # should never be silently relabeled by an automatic one); it would
    # also be redundant — the auto check only considers VALIDATED combos,
    # and a manually-invalidated trial's own strategy_id can still show up
    # as "validated" only if a DIFFERENT, unretired trial also survives
    # for that combo, which is a real, separate situation worth keeping
    # the manual reason for.
    manual_keys = {(r["ticker"], r["tf_label"], r["strategy"]) for r in invalidated_rows}
    for r in auto_retired_rows:
        if (r["ticker"], r["tf_label"], r["strategy"]) not in manual_keys:
            invalidated_rows.append(r)
    # Explicit columns even when empty — pd.DataFrame([]) has ZERO columns,
    # and to_sql on a zero-column frame emits a malformed "CREATE TABLE
    # foo ()" (sqlite3.OperationalError). Pre-existing latent bug, never
    # triggered in production (the real trial log already has invalidated
    # combos in its history) but hit immediately by the first real test of
    # this function against a small/fresh log with none.
    invalidated_combos = pd.DataFrame(invalidated_rows,
                                       columns=["ticker", "tf_label", "strategy", "n_trials_affected", "reason"])

    # Built in a TEMP file and atomically swapped in only on full success
    # (2026-10-07) — a migration/rebuild must never leave the real db
    # half-updated. Three separate to_sql(if_exists="replace") calls each
    # do their own DROP+CREATE+INSERT; if the SECOND one raised (the exact
    # empty-DataFrame bug caught and fixed earlier on invalidated_combos,
    # or any future one), the FIRST would already have replaced
    # strategy_ticker_tf on the REAL file — a genuinely corrupted, half-
    # migrated db, with the error easy to miss since rebuild_db() is
    # usually called from an unattended script. Writing everything to
    # f"{_DB_PATH}.tmp" and only replacing the real file with
    # os.replace() (atomic on POSIX) after every write AND both indexes
    # succeed means a failure anywhere leaves the real, already-working db
    # completely untouched — the affected migration stops, the exception
    # propagates with its real cause, nothing silently mutates.
    tmp_path = f"{_DB_PATH}.tmp"
    con = sqlite3.connect(tmp_path)
    try:
        strategy_ticker_tf.to_sql("strategy_ticker_tf", con, if_exists="replace", index=False)
        ticker_overview.to_sql("ticker_overview", con, if_exists="replace", index=False)
        invalidated_combos.to_sql("invalidated_combos", con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS idx_stt_ticker ON strategy_ticker_tf(ticker)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_ov_ticker ON ticker_overview(ticker)")
        con.commit()
    except Exception:
        con.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    else:
        con.close()
        os.replace(tmp_path, _DB_PATH)


def ticker_report(ticker):
    """Everything ticker_behavior.db knows about one ticker: the overview
    (per timeframe, best validated strategy if any) and the full
    strategy-by-strategy breakdown underneath it. Two DataFrames."""
    con = sqlite3.connect(_DB_PATH)
    try:
        overview = pd.read_sql(
            "SELECT * FROM ticker_overview WHERE ticker = ? ORDER BY tf_label", con, params=(ticker,))
        detail = pd.read_sql(
            "SELECT * FROM strategy_ticker_tf WHERE ticker = ? ORDER BY tf_label, validated DESC, "
            "best_mean_return_train DESC", con, params=(ticker,))
    finally:
        con.close()
    return overview, detail


def validated_leaderboard():
    """Every (ticker, timeframe, strategy) that has ever survived full
    BH-correction — the "what actually works, anywhere" view, across the
    whole project's history."""
    con = sqlite3.connect(_DB_PATH)
    try:
        return pd.read_sql(
            "SELECT ticker, tf_label, strategy, n_scored, n_survived_bh, best_mean_return_train, "
            "best_mean_return_holdout, best_std_return_holdout, best_n_holdout, best_label, "
            "best_strategy_id, best_target_widen_mult, lifecycle_state, "
            "last_tested_at FROM strategy_ticker_tf "
            "WHERE validated = 1 ORDER BY best_mean_return_train DESC", con)
    finally:
        con.close()


def all_tested_combos():
    """Every (ticker, timeframe, strategy) ever tested, WITHOUT the
    `validated = 1` filter — real bug found live 2026-10-08: bot_engine.py's
    own _close_stale_positions looks up a combo's max_bars by matching
    against validated_leaderboard(), on the theory that an already-OPEN
    position's own exit rule must be enforced "even if this combo has
    since decayed" (that function's own docstring) — decay doesn't
    remove a row from validated_leaderboard(), so that theory held. But
    the item-10 fail-closed identity gate (2026-10-07) DOES remove a row
    entirely once it has zero identified survivors — a combo can be
    validated=0 (no longer eligible for NEW entries) while a position
    opened under its EARLIER validated=1 state is still genuinely open.
    validated_leaderboard() alone then has nothing to look up, and the
    position's own max_bars time-stop silently stops being enforced —
    confirmed live: NZDJPY=X 1h clean_retracement_resumption_fixed_rr
    dropped to validated=0 on 2026-10-07 (legacy/unidentified survivors
    only), and a position opened on 2026-10-05 ran 5 bars past its own
    60-bar limit with zero enforcement, zero error, zero visibility —
    before this function existed to be read from.

    Risk management on a position already open must never depend on
    whether that combo is STILL eligible for a new one — this is the
    one place that holds regardless of validated/decayed/lifecycle_state,
    deliberately unfiltered."""
    con = sqlite3.connect(_DB_PATH)
    try:
        return pd.read_sql(
            "SELECT ticker, tf_label, strategy, best_label, validated, lifecycle_state "
            "FROM strategy_ticker_tf", con)
    finally:
        con.close()


def recently_invalidated():
    """Combos deliberately pulled from the tradeable universe (holdout_verdict
    flipped to FAILED with a reason), newest table write — not a live
    re-derivation, since the reason text itself is the only record of why;
    recomputing it from scratch each call would mean re-scanning the whole
    113k-row trial log for a table this small and this rarely-changing."""
    con = sqlite3.connect(_DB_PATH)
    try:
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", con)["name"].tolist()
        if "invalidated_combos" not in tables:
            return pd.DataFrame(columns=["ticker", "tf_label", "strategy", "n_trials_affected", "reason"])
        return pd.read_sql("SELECT * FROM invalidated_combos ORDER BY ticker, tf_label", con)
    finally:
        con.close()


def retired_combo_keys():
    """{(ticker, tf_label, strategy)} manually pulled from the tradeable
    universe (this module's own invalidated_combos table) — the RETIRED
    half of objective 8's lifecycle ladder, read as a live-usable key set
    rather than a display DataFrame. Deliberately built ON TOP of
    recently_invalidated() (one real query, not a second copy of it) —
    bot_engine.py's own live loop uses this to exclude RETIRED combos
    from trading, the same way it already excludes DECAYED ones via
    _decayed_combo_keys(); a combo manually retired after already
    earning validated=1 must not keep trading just because nothing
    else has changed about its own survivor evidence."""
    df = recently_invalidated()
    return set(zip(df["ticker"], df["tf_label"], df["strategy"]))


if __name__ == "__main__":
    rebuild_db()
    ov = pd.read_sql("SELECT COUNT(*) AS n FROM ticker_overview", sqlite3.connect(_DB_PATH))
    print(f"ticker_behavior.db rebuilt at {_DB_PATH} — {ov['n'].iloc[0]} (ticker, timeframe) rows")
