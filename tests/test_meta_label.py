"""meta_label.py — the from-scratch tree/forest must actually learn, and must be
able to OVERFIT (memorise noise) when allowed to — otherwise the overfit-vs-
restrained comparison it exists to show would be meaningless."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import meta_label as ml


def test_a_stump_finds_the_obvious_split():
    X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
    y = np.array([0, 0, 0, 5, 5, 5], float)
    t = ml.Tree(max_depth=1, min_leaf=1).fit(X, y)
    assert t.predict([[1.5], [11.5]]).tolist() == [0.0, 5.0]


def test_the_split_threshold_sits_between_the_classes():
    X = np.arange(10, dtype=float).reshape(-1, 1)
    y = (X[:, 0] > 5.5).astype(float)
    t = ml.Tree(max_depth=1).fit(X, y)
    j, thr = t.root[0], t.root[1]
    assert j == 0 and 5 < thr < 6


def test_min_leaf_is_respected():
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = np.zeros(20); y[0] = 100.0  # one outlier; with min_leaf=5 it can't be isolated
    t = ml.Tree(min_leaf=5).fit(X, y)
    assert t.predict([[0.0]])[0] < 100.0


def test_a_deep_tree_memorises_pure_noise_a_shallow_one_does_not():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(300, 5)), rng.normal(size=300)
    deep = ml.Tree(min_leaf=1).fit(X, y)
    shallow = ml.Tree(max_depth=2, min_leaf=40).fit(X, y)
    assert np.corrcoef(deep.predict(X), y)[0, 1] > 0.99          # perfect recall of noise
    assert abs(np.corrcoef(shallow.predict(X), y)[0, 1]) < 0.5   # can't memorise it


def test_overfit_forest_looks_brilliant_in_sample_and_useless_out_of_sample_on_noise():
    rng = np.random.default_rng(1)
    Xtr, ytr = rng.normal(size=(400, 6)), rng.normal(size=400)
    Xte, yte = rng.normal(size=(400, 6)), rng.normal(size=400)
    f = ml.Forest(n_trees=25, max_depth=None, min_leaf=1, seed=0).fit(Xtr, ytr)
    in_sample = ml.top_share_result(f.predict(Xtr), ytr)["taken_mean_r"]
    out = ml.top_share_result(f.predict(Xte), yte)["taken_mean_r"]
    assert in_sample > 0.5 and abs(out) < 0.25   # the whole point: in-sample glory, out-of-sample nothing


def test_the_forest_learns_a_real_signal_out_of_sample():
    rng = np.random.default_rng(2)
    def make(n):
        X = rng.normal(size=(n, 4)); y = 1.0 * (X[:, 0] > 0) + rng.normal(0, 0.5, n)
        return X, y
    Xtr, ytr = make(500); Xte, yte = make(500)
    f = ml.Forest(n_trees=30, max_depth=3, min_leaf=20, seed=0).fit(Xtr, ytr)
    res = ml.top_share_result(f.predict(Xte), yte)
    assert res["taken_mean_r"] > res["all_mean_r"] + 0.3
    assert ml.auc(f.predict(Xte), yte > 0.5) > 0.7


def test_auc_basics():
    assert ml.auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert ml.auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert ml.auc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == pytest.approx(0.5)
    assert np.isnan(ml.auc([1, 2], [1, 1]))


def test_top_share_picks_the_highest_scores():
    r = np.array([1.0, 2.0, 3.0, 4.0])
    out = ml.top_share_result([0.1, 0.2, 0.3, 0.4], r, share=0.5)
    assert out["taken_mean_r"] == pytest.approx(3.5) and out["n_taken"] == 2 and out["all_mean_r"] == pytest.approx(2.5)


def test_adjacent_floats_never_produce_an_empty_side_or_a_nan_prediction():
    # two feature values one ulp apart: their midpoint rounds onto the larger value
    lo = 1.0
    hi = float(np.nextafter(lo, 2.0))
    X = np.array([[lo]] * 10 + [[hi]] * 10, dtype=float)
    y = np.array([0.0] * 10 + [1.0] * 10)
    t = ml.Tree(min_leaf=1).fit(X, y)
    p = t.predict(X)
    assert not np.isnan(p).any()
    assert p[:10].mean() < p[10:].mean()  # and it still separates them


def test_a_deep_forest_on_messy_float_features_never_returns_nan():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(250, 4))
    X = np.round(base, 12) + np.where(rng.random((250, 4)) < 0.3, 1e-15, 0.0)  # many near-ties
    y = rng.normal(size=250)
    p = ml.Forest(40, None, 1, 0.8, seed=0).fit(X, y).predict(X)
    assert not np.isnan(p).any()
