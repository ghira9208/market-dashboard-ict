"""
Benjamini-Hochberg FDR correction — turns "how many things did we try" into
an actual adjusted significance threshold instead of eyeballing one raw
p-value in isolation. Bonferroni (control the probability of ANY false
positive across all trials) is the more familiar sibling but gets
punishingly conservative as trial count grows into the hundreds, which an
adaptive search agent reaches fast; BH instead controls the expected FALSE
DISCOVERY RATE — the standard choice for "we're screening many candidates
and want most of what we call significant to actually be real," which is
exactly this situation. Same underlying reasoning as
research/experiment_log.py's own docstring: a p-value only means something
in light of how many other things were tried alongside it — this is that
reasoning turned into an actual threshold instead of a comment.
"""

import numpy as np


def benjamini_hochberg(p_values, alpha=0.05):
    """p_values: array-like of raw p-values from every trial ever logged,
    not just the current batch (edge_lab/agent.py always passes the full
    cumulative log). Returns (q_values, significant_mask), both aligned to
    the input order. significant_mask follows BH's actual step-up rule
    (the largest rank k whose own p-value clears k/n*alpha, with every
    rank below k also significant) — not "q_value < alpha" applied as an
    afterthought; the two happen to coincide once q_values are computed the
    standard way, which is what this does."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([]), np.array([], dtype=bool)

    order = np.argsort(p)
    ranked = p[order]
    ranks = np.arange(1, n + 1)

    # Standard step-up adjustment: per-rank raw q, then a running minimum
    # from the largest rank down so q is monotone non-decreasing with rank
    # — skipping that pass would let a small p-value see an inflated q
    # merely because a worse p-value happened to sit just below it.
    raw_q = ranked * n / ranks
    q_sorted = np.minimum.accumulate(raw_q[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)

    q_values = np.empty(n)
    q_values[order] = q_sorted
    significant = q_values < alpha
    return q_values, significant
