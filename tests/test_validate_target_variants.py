"""_base_label — makes validate_target_variants.py idempotent against an
already-widened leaderboard. Real bug caught live 2026-10-07: re-running
this script after a widen variant had already won a combo's leaderboard row
appended " · widen=X" a SECOND time onto the already-widened best_label
("...widen=2.0" -> "...widen=2.0 · widen=2.0"), producing a brand-new label
that never matched, let alone superseded, the original — so the first run's
trial (written before target_widen_mult was even being persisted as a field)
stayed the leaderboard's own best_row forever, no matter how many times the
script re-ran. Stripping back to the true base label first makes repeated
runs converge instead of compound."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validate_target_variants import _base_label


def test_strips_a_single_widen_suffix():
    assert _base_label("clean_expansion_fixed_rr · rr=3.0 · atr=1.5 · maxbars=60 · widen=2.0") == \
        "clean_expansion_fixed_rr · rr=3.0 · atr=1.5 · maxbars=60"


def test_a_label_with_no_widen_suffix_is_unchanged():
    base = "clean_expansion_fixed_rr · rr=3.0 · atr=1.5 · maxbars=60"
    assert _base_label(base) == base


def test_is_idempotent_applied_twice():
    base = "clean_expansion_fixed_rr · rr=3.0 · atr=1.5 · maxbars=60"
    once = _base_label(f"{base} · widen=3.0")
    twice = _base_label(once)
    assert once == base
    assert twice == base


def test_handles_a_decimal_widen_value():
    assert _base_label("X · widen=1.5") == "X"


def test_only_strips_a_trailing_widen_not_one_mid_label():
    # Defensive: a real label could never legitimately contain "widen=" as
    # anything but this script's own trailing suffix, but confirms the
    # regex is anchored to the end, not a blind string replace.
    weird = "X · widen=2.0 · atr=1.5"
    assert _base_label(weird) == weird  # no trailing widen= here, left alone


def test_strips_two_stacked_widen_suffixes_in_one_call():
    # REAL REGRESSION, caught live 2026-10-08: a single _sub() call on an
    # end-anchored pattern only ever strips ONE match, so an input that
    # had ALREADY compounded to a double suffix (some historical trial
    # won best_row with "...widen=2.0 · widen=4.0" already attached)
    # only lost its outer layer, then had a THIRD layer appended on top —
    # "...widen=2.0 · widen=4.0 · widen=4.0" in the real leaderboard, 28
    # of 106 combos affected. Must collapse ALL the way back to the true
    # base in a single call, not leave one layer for the next bug to
    # compound further.
    base = "clean_expansion_fixed_rr · rr=3.0 · atr=1.5 · maxbars=60"
    assert _base_label(f"{base} · widen=2.0 · widen=4.0") == base


def test_strips_three_stacked_widen_suffixes_in_one_call():
    base = "clean_expansion_fixed_rr · rr=3.0 · atr=1.5 · maxbars=60"
    assert _base_label(f"{base} · widen=1.5 · widen=2.0 · widen=4.0") == base


def test_is_idempotent_even_starting_from_an_already_stacked_label():
    base = "clean_expansion_fixed_rr · rr=3.0 · atr=1.5 · maxbars=60"
    once = _base_label(f"{base} · widen=2.0 · widen=4.0")
    twice = _base_label(once)
    assert once == base
    assert twice == base
