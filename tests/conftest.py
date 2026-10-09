"""Shared test setup for the research side (experiments.py / ticker_behavior.py
and friends) — first real pytest infrastructure for this half of the project
(2026-10-07); nothing here existed before.

CONFIRMED REAL BUG, caught writing the first tests that ever called
load_experiment_trials()/rebuild_db() more than once in the same process:
_load_experiment_trials_cached is @st.cache_data(ttl=_CACHE_TTL)'d keyed ONLY
on (mtime, size) of _TRIALS_PATH — never the path itself. In real production
use _TRIALS_PATH never changes within a process lifetime, so this is a
perfectly fine cache key there. But a test suite that monkeypatches
_TRIALS_PATH to a fresh tmp_path file per test can easily produce two
DIFFERENT small fixture files with the SAME (mtime, size) signature (plausible
whenever two tests' fixture content happens to be a similar size and they run
within the same wall-clock second) — Streamlit's cache then silently serves
one test's trial data to a completely different test, with no error. This
autouse fixture clears that cache before every test in this directory so the
production code (correctly designed for its real use case) never needs to
change — only the test isolation does."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import experiments as ex


@pytest.fixture(autouse=True)
def _clear_trial_log_cache():
    ex._load_experiment_trials_cached.clear()
    yield
    ex._load_experiment_trials_cached.clear()
