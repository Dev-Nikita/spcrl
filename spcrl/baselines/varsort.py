"""Varsortability and the sortnregress baseline (Reisach et al., NeurIPS 2021).

Simulated additive-noise benchmarks often leak the causal order into the
marginal variances, so that simply sorting variables by variance and running a
sparse regression recovers much of the graph. A benchmark on which
`sortnregress` scores highly is not evidence about causal discovery. We measure
both quantities on our generator and report them.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LassoLarsIC, LinearRegression


def varsortability(X: np.ndarray, A: np.ndarray, tol: float | None = None) -> float:
    """Fraction of directed paths along which the marginal variance increases.

    1.0 means the causal order is fully recoverable from variances alone;
    0.5 is chance.
    """
    d = X.shape[1]
    var = X.var(0)
    if tol is None:
        # relative tolerance: on standardised data the variances differ only by
        # floating-point noise, and an absolute tolerance would score that noise
        # as a real ordering.
        tol = 1e-6 * float(np.mean(var))
    A = (np.asarray(A) != 0).astype(float)
    reach = np.eye(d)
    n_inc = n_tot = 0
    cur = A.copy()
    for _ in range(d - 1):
        if cur.sum() == 0:
            break
        idx = np.argwhere(cur > 0)
        for i, j in idx:
            n_tot += 1
            if var[j] > var[i] + tol:
                n_inc += 1
        reach = reach + cur
        cur = (cur @ A > 0).astype(float) * (reach == 0)
    return n_inc / n_tot if n_tot else 0.5


def r2_sortability(X: np.ndarray, A: np.ndarray) -> float:
    """Same statistic computed on the fraction of variance explained by the
    other variables, which standardisation does not remove."""
    d = X.shape[1]
    r2 = np.zeros(d)
    for i in range(d):
        others = [j for j in range(d) if j != i]
        lr = LinearRegression().fit(X[:, others], X[:, i])
        r2[i] = lr.score(X[:, others], X[:, i])
    return varsortability(np.sqrt(np.maximum(r2, 0))[None, :] * np.ones((2, d))
                          + np.arange(2)[:, None] * 0, A) if False else _sort_stat(r2, A)


def _sort_stat(score: np.ndarray, A: np.ndarray, tol: float | None = None) -> float:
    d = len(score)
    if tol is None:
        tol = 1e-6 * max(float(np.mean(np.abs(score))), 1e-12)
    A = (np.asarray(A) != 0).astype(float)
    reach = np.eye(d)
    n_inc = n_tot = 0
    cur = A.copy()
    for _ in range(d - 1):
        if cur.sum() == 0:
            break
        for i, j in np.argwhere(cur > 0):
            n_tot += 1
            if score[j] > score[i] + tol:
                n_inc += 1
        reach = reach + cur
        cur = (cur @ A > 0).astype(float) * (reach == 0)
    return n_inc / n_tot if n_tot else 0.5


def sortnregress(X: np.ndarray) -> np.ndarray:
    """Order variables by increasing marginal variance, then regress each on its
    predecessors with LassoLarsIC. Returns a weighted adjacency W[j, i] for j->i.
    """
    n, d = X.shape
    W = np.zeros((d, d))
    order = np.argsort(X.var(0))
    for k in range(1, d):
        i = order[k]
        pred = order[:k]
        lam = LassoLarsIC(criterion="bic")
        # weight the candidate predictors by an OLS fit, as in the original code
        lr = LinearRegression().fit(X[:, pred], X[:, i])
        w = np.abs(lr.coef_) + 1e-12
        lam.fit(X[:, pred] * w, X[:, i])
        W[pred, i] = np.abs(lam.coef_ * w)
    return W
