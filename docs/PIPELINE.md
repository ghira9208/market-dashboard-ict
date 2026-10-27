# How evidence is judged

Every number the live bot trusts passes through this sequence. Nothing here is a suggestion; each step exists because
skipping it once produced a false edge.

1. **One definition of the trade.** `_simulate_fixed_rr`: next-bar-open entry, a stop of `atr_mult_stop` x ATR, a target at
   `rr_multiple` x that distance, time exit at `max_bars`, stop first when a bar touches both, round-trip cost in basis
   points (2.0 by default; real index-CFD costs are far higher, see `studies/validate_cost_model.py`). The live engine
   shares the stop/target arithmetic (`compute_risk_unit`, `compute_stop_target`, `resolve_entry_pos`) so the two cannot drift.
2. **Time split 60 / 20 / 20.** Research, validation, locked. Parameters are chosen on research, confirmed on validation;
   the locked slice is scored but never used to choose anything.
3. **Permutation test against a direction-flipped null.** Is calling the real direction better than the opposite call on
   the same entries? 5,000 draws.
4. **Benjamini-Hochberg across the whole ledger.** The ledger's size is the denominator, which is why exploratory work
   gets its own ledger.
5. **Holdout and same sign.** Passing validation with the same sign as research is required.
6. **Fail-closed identity.** A combo is `validated` only if a survivor has a stored `strategy_id`; legacy rows without one
   are shown on the dashboard but cannot trade.
7. **Lifecycle.** EXPERIMENTAL -> TRAIN_SIGNIFICANT -> HOLDOUT_VALIDATED -> PROSPECTIVE -> LIVE, with DECAYED and RETIRED
   as overrides. A validated combo whose locked slice turns negative over 10+ trades is retired automatically and
   un-retired automatically if a later result recovers.
8. **Weekly decay check** (`revalidate_combos.py`) and a per-combo **backtest-vs-live** comparison on the dashboard.
9. **Forward tests for anything new.** Frozen spec, data that did not exist when the idea was chosen, judged once.

## Reading a result honestly

- Report dollars next to R, and the live account's own numbers (`user_id`), never pooled test users.
- Day-cluster any pooled statistic: the same news day hits every combo at once.
- Stress costs (6 bps) before believing a small edge.
- An overlay that skips trades must be judged on total R as well as mean R per trade.
