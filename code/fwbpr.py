"""FW-BPR: Frank-Wolfe-inspired bipartite-ranking meta-algorithm.

Wraps any weight-accepting scorer. Iteration 0 fits with balanced class
weights (n- per positive, n+ per negative) -- the first Frank-Wolfe
subproblem of the pairwise dual (cor:fw_bridge in the paper). Each subsequent
iteration reweights every example by its count of violated pairs under the
current scores: a positive is weighted by the number of negatives scored
within `margin` above it, a negative by the number of positives scored within
`margin` below it (margin=0 gives pure misordering counts; margin=2 on SVM
decision scores matches the margin-2 pairwise hinge exactly).

Iterate selection is by AUC on a validation set if one is given, else on the
training set.
"""

import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score


def violation_counts(scores, y, margin=0.0):
    """Per-example counts of violated pairs: pair (i+, j-) is violated iff
    s_i - s_j < margin. O(N log N) via sorting."""
    s_pos = np.sort(scores[y == 1])
    s_neg = np.sort(scores[y == 0])
    n_pos, n_neg = s_pos.size, s_neg.size
    counts = np.zeros(y.size)
    # positive i: #{j : s_j > s_i - margin}
    counts[y == 1] = n_neg - np.searchsorted(
        s_neg, scores[y == 1] - margin, side="right")
    # negative j: #{i : s_i < s_j + margin}
    counts[y == 0] = np.searchsorted(
        s_pos, scores[y == 0] + margin, side="left")
    return counts


def _score(model, X):
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    p = model.predict_proba(X)[:, 1]
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


class FWBPRanker:
    """Iteratively reweighted bipartite ranker around a base estimator.

    Parameters
    ----------
    estimator : sklearn estimator supporting fit(X, y, sample_weight)
    n_iter    : maximum number of reweighting rounds (>=1; 1 = balanced fit)
    margin    : pair-violation margin (0 = misordering counts)
    """

    def __init__(self, estimator, n_iter=10, margin=0.0):
        self.estimator = estimator
        self.n_iter = n_iter
        self.margin = margin

    def fit(self, X, y, X_val=None, y_val=None):
        y = np.asarray(y)
        n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
        N = y.size
        # balanced initialization = first FW subproblem: every pair violated
        counts = np.where(y == 1, float(n_neg), float(n_pos))
        self.models_, self.history_ = [], []
        best_auc, best_model = -np.inf, None
        prev_counts = None
        for it in range(self.n_iter):
            w = counts * (N / counts.sum())
            model = clone(self.estimator)
            model.fit(X, y, sample_weight=w)
            s_train = _score(model, X)
            if X_val is not None:
                auc = roc_auc_score(y_val, _score(model, X_val))
            else:
                auc = roc_auc_score(y, s_train)
            self.models_.append(model)
            self.history_.append(dict(iter=it, auc=auc,
                                      violated=counts.sum() / 2.0))
            if auc > best_auc:
                best_auc, best_model = auc, model
            counts = violation_counts(s_train, y, self.margin)
            if counts.sum() == 0:  # perfect ranking on train
                break
            if prev_counts is not None and np.array_equal(counts, prev_counts):
                break  # weight pattern converged
            prev_counts = counts.copy()
        self.best_model_ = best_model
        self.best_auc_ = best_auc
        return self

    def decision_function(self, X):
        return _score(self.best_model_, X)
