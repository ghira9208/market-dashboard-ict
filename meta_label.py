"""Meta-labeling experiment: can a model that looks at the surrounding market
decide WHICH of the existing strategies' signals are worth taking?

No strategy is invented here. Every signal already exists; the model only
scores it (from information available at entry) and we keep the top share.
Two models are trained on the same data and judged on the same untouched
future: a deliberately OVERFIT one (deep trees that can memorise the training
trades) and a RESTRAINED one (shallow trees, big leaves). The gap between a
model's in-sample result and its out-of-sample result is the overfit, made
visible.

Pure numpy (no scikit-learn dependency): a CART regression tree and a bagged
forest of them.
"""
import numpy as np


class Tree:
    def __init__(self, max_depth=None, min_leaf=1, n_features=None, rng=None):
        self.max_depth, self.min_leaf, self.n_features = max_depth, min_leaf, n_features
        self.rng = rng or np.random.default_rng(0)
        self.root = None

    def fit(self, X, y):
        self.root = self._grow(np.asarray(X, float), np.asarray(y, float), 0)
        return self

    def _grow(self, X, y, depth):
        if len(y) < 2 * self.min_leaf or (self.max_depth is not None and depth >= self.max_depth) or y.min() == y.max():
            return float(y.mean())
        best = None
        feats = np.arange(X.shape[1])
        if self.n_features and self.n_features < X.shape[1]:
            feats = self.rng.choice(feats, self.n_features, replace=False)
        for j in feats:
            order = np.argsort(X[:, j], kind="stable")
            xs, ys = X[order, j], y[order]
            csum, csq = np.cumsum(ys), np.cumsum(ys ** 2)
            n = len(ys)
            lo = self.min_leaf
            idx = np.arange(lo, n - lo + 1)
            idx = idx[xs[idx - 1] < xs[np.minimum(idx, n - 1)]]  # only split between different values
            if len(idx) == 0:
                continue
            nl, nr = idx, n - idx
            sl, sr = csum[idx - 1], csum[-1] - csum[idx - 1]
            ql, qr = csq[idx - 1], csq[-1] - csq[idx - 1]
            sse = (ql - sl ** 2 / nl) + (qr - sr ** 2 / nr)
            k = int(np.argmin(sse))
            if best is None or sse[k] < best[0]:
                lo_v, hi_v = float(xs[idx[k] - 1]), float(xs[idx[k]])
                mid = (lo_v + hi_v) / 2
                # two adjacent floats can have a midpoint that rounds UP onto the larger one, which would put
                # both on the left and leave the right side empty (a NaN leaf); split on the smaller one then
                best = (float(sse[k]), j, mid if mid < hi_v else lo_v)
        if best is None:
            return float(y.mean())
        _, j, thr = best
        m = X[:, j] <= thr
        if m.all() or not m.any():
            return float(y.mean())
        return (j, thr, self._grow(X[m], y[m], depth + 1), self._grow(X[~m], y[~m], depth + 1))

    def predict(self, X):
        X = np.asarray(X, float)
        out = np.empty(len(X))
        for i, row in enumerate(X):
            node = self.root
            while isinstance(node, tuple):
                node = node[2] if row[node[0]] <= node[1] else node[3]
            out[i] = node
        return out


class Forest:
    def __init__(self, n_trees=100, max_depth=None, min_leaf=1, feature_frac=0.5, seed=0):
        self.n_trees, self.max_depth, self.min_leaf, self.feature_frac, self.seed = n_trees, max_depth, min_leaf, feature_frac, seed
        self.trees = []

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        rng = np.random.default_rng(self.seed)
        nf = max(1, int(round(X.shape[1] * self.feature_frac)))
        self.trees = []
        for _ in range(self.n_trees):
            idx = rng.integers(0, len(y), len(y))
            self.trees.append(Tree(self.max_depth, self.min_leaf, nf, np.random.default_rng(rng.integers(1 << 30))).fit(X[idx], y[idx]))
        return self

    def predict(self, X):
        return np.mean([t.predict(X) for t in self.trees], axis=0)


def auc(score, positive):
    """P(a random winner is scored above a random loser) — 0.5 = no skill."""
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    pos, neg = score[positive], score[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    from scipy.stats import rankdata
    ranks = rankdata(np.concatenate([pos, neg]))  # average rank for ties: a model that gives everything the same score has AUC 0.5, not 0
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def top_share_result(score, r, share=0.5):
    """Mean R of the trades the model would TAKE (highest-scored `share`) vs all trades."""
    score, r = np.asarray(score, float), np.asarray(r, float)
    k = max(1, int(len(r) * share))
    take = np.argsort(-score, kind="stable")[:k]
    return {"taken_mean_r": float(r[take].mean()), "all_mean_r": float(r.mean()), "n_taken": int(k), "n": int(len(r)),
            "taken_total_r": float(r[take].sum())}
