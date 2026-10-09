"""load_experiment_trials()'s cache actually invalidates when the trial log
changes — pins a real, severe, previously-undetected bug found 2026-10-07
while writing the first tests that ever called this function twice in one
process.

_load_experiment_trials_cached(_mtime, _size) used to spell BOTH parameters
with a leading underscore. Streamlit's own documented st.cache_data
convention treats an underscore-prefixed parameter as explicitly EXCLUDED
from the cache key (it exists to let you pass something unhashable, like a
live DB connection, without cache_data choking on it trying to hash it).
With both of this function's only parameters underscore-prefixed, its cache
key was effectively constant — every call, regardless of the real mtime/size
passed in, hit the exact same cache entry and silently returned whatever was
computed on the FIRST call, for the rest of that process's lifetime.

No single `python3 -c` script this session ever exposed it (each only ever
called load_experiment_trials() once). Any long-running process that calls
it more than once would have: the real Streamlit app foremost among them —
a user clicking survivors_ui.py's "Refresh from trial log" button would see
literally nothing change, ever, until the server itself restarted."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import experiments as ex


def _write(path, n_events):
    with open(path, "a") as f:
        f.write(json.dumps({
            "logged_at": "2026-01-01T00:00:00+00:00", "ticker": "X", "tf_label": "5m",
            "label": "L", "settings": {}, "n_events": n_events, "verdict": "INSUFFICIENT_DATA",
        }) + "\n")


def test_a_second_call_after_the_file_changes_sees_the_new_row(tmp_path, monkeypatch):
    trials_path = tmp_path / "trials.jsonl"
    monkeypatch.setattr(ex, "_TRIALS_PATH", str(trials_path))

    _write(trials_path, n_events=1)
    first = ex.load_experiment_trials()
    assert len(first) == 1

    _write(trials_path, n_events=2)
    second = ex.load_experiment_trials()
    # The real regression: before the fix, this stayed len==1 forever —
    # the appended second row was silently invisible for the rest of the
    # process, no error, no warning.
    assert len(second) == 2


def test_a_third_call_with_even_more_rows_keeps_seeing_fresh_data(tmp_path, monkeypatch):
    # Not just "it works once after the fix" — confirms the cache
    # genuinely tracks file state call over call, not just a one-time
    # fluke of two specific (mtime, size) pairs happening to differ.
    trials_path = tmp_path / "trials.jsonl"
    monkeypatch.setattr(ex, "_TRIALS_PATH", str(trials_path))

    for n in range(1, 6):
        _write(trials_path, n_events=n)
        assert len(ex.load_experiment_trials()) == n


def test_the_cached_functions_own_parameters_are_not_underscore_prefixed():
    # A signature-level regression guard — if someone "cleans up" these
    # names back to _mtime/_size later, this fails immediately and
    # explains why, instead of silently reintroducing the exact bug above.
    import inspect
    params = list(inspect.signature(ex._load_experiment_trials_cached).parameters)
    assert not any(p.startswith("_") for p in params), (
        f"an underscore-prefixed parameter here is EXCLUDED from Streamlit's cache key "
        f"(st.cache_data's own documented convention) — got {params}"
    )
