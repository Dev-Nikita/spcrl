"""Classical / non-SP baselines for causal structure recovery."""
from __future__ import annotations
import itertools
import numpy as np
from scipy import stats


# ---------------------------------------------------------------- utilities
def _residualise_lag(X: np.ndarray) -> np.ndarray:
    """Remove each node's own AR(1) component so that instantaneous structure
    is what remains -- gives the static baselines a fair chance."""
    Xc, Xl = X[1:], X[:-1]
    R = np.empty_like(Xc)
    for i in range(X.shape[1]):
        b = np.polyfit(Xl[:, i], Xc[:, i], 1)
        R[:, i] = Xc[:, i] - np.polyval(b, Xl[:, i])
    return R


def fisher_z(C: np.ndarray, i: int, j: int, S: tuple, n: int):
    idx = [i, j] + list(S)
    sub = C[np.ix_(idx, idx)]
    try:
        P = np.linalg.pinv(sub)
    except np.linalg.LinAlgError:
        return 1.0, 0.0
    r = -P[0, 1] / np.sqrt(P[0, 0] * P[1, 1] + 1e-12)
    r = np.clip(r, -0.999999, 0.999999)
    z = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(max(n - len(S) - 3, 1))
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return p, abs(r)


# ---------------------------------------------------------------- baselines
def pc_skeleton(X: np.ndarray, alpha: float = 0.05, max_cond: int = 3,
                prior_mask: np.ndarray | None = None):
    """PC skeleton + Meek-free v-structure orientation. Returns (B, score)."""
    R = _residualise_lag(X)
    n, d = R.shape
    C = np.corrcoef(R.T)
    adj = np.ones((d, d), dtype=bool) & ~np.eye(d, dtype=bool)
    if prior_mask is not None:
        allowed = (np.asarray(prior_mask) > 0.5)
        adj &= (allowed | allowed.T)
    sepset: dict = {}
    strength = np.zeros((d, d))
    for k in range(0, max_cond + 1):
        for i, j in itertools.combinations(range(d), 2):
            if not adj[i, j]:
                continue
            nb = [x for x in range(d) if x != i and x != j and (adj[i, x] or adj[j, x])]
            if len(nb) < k:
                continue
            for S in itertools.combinations(nb, k):
                p, r = fisher_z(C, i, j, S, n)
                if p > alpha:
                    adj[i, j] = adj[j, i] = False
                    sepset[(i, j)] = sepset[(j, i)] = set(S)
                    break
                strength[i, j] = strength[j, i] = max(strength[i, j], r)
    # v-structure orientation
    B = np.zeros((d, d), dtype=np.int8)
    oriented = np.zeros((d, d), dtype=bool)
    for j in range(d):
        nbrs = np.where(adj[j])[0]
        for i, k in itertools.combinations(nbrs, 2):
            if not adj[i, k] and j not in sepset.get((i, k), {j}):
                B[i, j] = B[k, j] = 1
                oriented[i, j] = oriented[k, j] = True
    for i, j in itertools.combinations(range(d), 2):
        if adj[i, j] and not (oriented[i, j] or oriented[j, i]):
            B[i, j] = 1        # arbitrary orientation of undirected remainder
    return B, strength


def partial_corr_scores(X: np.ndarray) -> np.ndarray:
    """|precision matrix| as a (symmetric) association score."""
    R = _residualise_lag(X)
    C = np.corrcoef(R.T)
    P = np.linalg.pinv(C + 1e-6 * np.eye(C.shape[0]))
    dnorm = np.sqrt(np.outer(np.diag(P), np.diag(P)))
    S = np.abs(-P / (dnorm + 1e-12))
    np.fill_diagonal(S, 0)
    return S


def notears_linear(X: np.ndarray, lambda1: float = 0.05, max_iter: int = 100,
                   h_tol: float = 1e-8, rho_max: float = 1e16,
                   mask: np.ndarray | None = None) -> np.ndarray:
    """Linear NOTEARS (Zheng et al. 2018) on lag-residualised data."""
    import scipy.optimize as sopt
    R = _residualise_lag(X)
    R = (R - R.mean(0)) / (R.std(0) + 1e-8)
    n, d = R.shape
    Msk = None if mask is None else (np.asarray(mask) > 0.5).astype(float)

    def _adj(w):
        return (w[: d * d] - w[d * d:]).reshape([d, d])

    def _loss(W):
        Wm = W if Msk is None else W * Msk
        Rr = R @ Wm - R
        loss = 0.5 / n * (Rr ** 2).sum()
        G = -1.0 / n * R.T @ (R - R @ Wm)
        if Msk is not None:
            G = G * Msk
        return loss, G

    def _h(W):
        E = np.linalg.matrix_power(np.eye(d) + W * W / d, d - 1)
        h = (E.T * (np.eye(d) + W * W / d)).sum() - d
        G = E.T * W * 2
        return h, G

    def _func(w):
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G = G_loss + (rho * h + alpha) * G_h
        return obj, np.concatenate((G + lambda1, -G + lambda1), axis=None)

    w_est = np.zeros(2 * d * d)
    rho, alpha, h = 1.0, 0.0, np.inf
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    for _ in range(max_iter):
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method="L-BFGS-B", jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W = _adj(w_est)
    return W if Msk is None else W * Msk
