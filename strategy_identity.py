"""One function, one job: turn every P&L-relevant component of a strategy
into a short, stable identity string — so a trial, a leaderboard row, and a
live decision can all answer "were these actually produced by the same
strategy" instead of silently assuming it from a (ticker, tf_label, strategy)
triple and a label string that can drift underneath them.

Deliberately excludes sizing (sizing_mode/kelly_fraction): a bet-size choice
doesn't change what the backtest evidence says about the trade's own entry/
exit mechanics, and account-level sizing changes are already captured by
db.py's own config_changed event log (etoro-dashboard) — minting a new
strategy identity on every risk_pct tweak would be noise, not rigor.

Deliberately excludes scale_out_enabled from the HASH itself too, for a
sharper reason: _simulate_fixed_rr never models a partial exit — every trial
this project has ever scored was simulated as one full-size position to one
target, scale-out or not. If scale_out_enabled were hashed in, a scale-out-
enabled account could never match ANY trial (none has ever been scored with
it True), silently blocking every signal the moment live-side enforcement
ships. Recorded instead as its own plain trial-dict field (always False at
trial time, honestly reflecting what was actually simulated) for the ledger
— informational, not part of the identity match. Scale-out's own mechanics
(the near/far split, the breakeven move) are validated separately by
partial_scale_out_backtest.py, outside the BH/trial-log pipeline entirely;
folding it into this hash isn't the fix for that gap.

Deliberately NOT derived from source-hashing the detector/data-fetch code:
that's real sophistication this project doesn't need yet (see the project's
own standing discipline against hypothesis-testing-for-its-own-sake). Instead
DATA_PIPELINE_VERSION (data.py) and DETECTOR_LOGIC_VERSIONS (experiments.py)
are small, manually-bumped integers — bump one by hand when the corresponding
code changes in a way that could shift a backtested result, same spirit as
any other breaking-change version bump."""
import hashlib
import json
import subprocess

_git_commit_cache = {}


def git_commit(repo_dir):
    """Short real commit hash for the ledger's "code/version identifier"
    field — cached per (process, repo_dir) since it can't change during a
    single run. None (never raises) if the dir isn't a real git checkout or
    the git binary isn't available — the research ledger stays honest about
    a missing value rather than guessing one."""
    if repo_dir not in _git_commit_cache:
        try:
            out = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=repo_dir,
                                  capture_output=True, text=True, timeout=5, check=True)
            _git_commit_cache[repo_dir] = out.stdout.strip() or None
        except Exception:
            _git_commit_cache[repo_dir] = None
    return _git_commit_cache[repo_dir]


def compute_strategy_id(family, settings, target_widen_mult,
                         data_pipeline_version, detector_logic_version, cost_bps):
    """family: the strategy family name (e.g. "clean_expansion_fixed_rr").
    settings: the dict a trial was actually scored under — already carries
    the entry rule (min_streak/retr_window_bars/...), the stop rule
    (atr_mult_stop), the target rule (rr_multiple) and the time-exit rule
    (max_bars) as its own keys, so hashing it whole covers all four without
    a separate parameter per rule. target_widen_mult: the live-only target
    widening applied ON TOP of settings["rr_multiple"] (1.0 for a base trial
    that tests rr_multiple directly, >1.0 only for a trial from
    validate_target_variants.py that's explicitly testing a widened
    distance). cost_bps: the round-trip cost assumption the trial/live
    decision was scored/sized under.

    scale_out_enabled is deliberately NOT a parameter here — see this
    module's own docstring for why folding it into the hash would silently
    break live matching for every scale-out account the moment enforcement
    ships. Record it as a separate, plain field on the trial/decision
    instead of passing it in.

    Returns a 12-hex-char sha256 prefix — short enough to read in a label,
    long enough that an accidental collision between two genuinely different
    configurations is not a real concern at this project's scale (a handful
    of strategy families × a few thousand settings combos, nowhere near
    birthday-bound territory for 48 bits)."""
    canonical = {
        "family": family,
        "settings": {k: settings[k] for k in sorted(settings.keys()) if not k.startswith("_")},
        "target_widen_mult": round(float(target_widen_mult), 6),
        "data_pipeline_version": int(data_pipeline_version),
        "detector_logic_version": int(detector_logic_version),
        "cost_bps": round(float(cost_bps), 6),
    }
    blob = json.dumps(canonical, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
