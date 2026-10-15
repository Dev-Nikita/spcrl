"""Surrogate SCM built on an *estimated* causal graph.

The same downstream pipeline is used for every method, so that differences in
root-cause accuracy and intervention regret are attributable to the estimated
graph rather than to the downstream model.
"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from ..benchmarks.synthetic import is_dag, topological_order


def top_k_dag(P: np.ndarray, k: int) -> np.ndarray:
    """Keep the k strongest edges, then break cycles by removing weakest."""
    d = P.shape[0]
    A = np.abs(np.asarray(P, float)).copy()
    np.fill_diagonal(A, 0)
    flat = np.argsort(-A, axis=None)
    B = np.zeros((d, d), dtype=np.int8)
    for f in flat[: max(k, 1)]:
        B[np.unravel_index(f, A.shape)] = 1
    while not is_dag(B):
        idx = np.argwhere(B == 1)
        w = A[B == 1]
        i, j = idx[np.argmin(w)]
        B[i, j] = 0
    return B


class SurrogateSCM:
    def __init__(self, B: np.ndarray, kind: str = "gb"):
        self.B = (np.asarray(B) != 0).astype(np.int8)
        self.kind = kind
        self.models: dict[int, object] = {}
        self._order = None

    @property
    def order(self):
        """Topological order, computed only when a rollout needs it.

        Diagnosis fits one regression per node and never needs a global order,
        so a parent set with cycles is legitimate there; `rollout` is the only
        operation that requires acyclicity and it raises if the graph has none.
        """
        if self._order is None:
            if not is_dag(self.B):
                raise ValueError("rollout requires an acyclic parent structure; "
                                 "use top_k_dag to project before rolling out")
            self._order = topological_order(self.B)
        return self._order

    def _design(self, X, i):
        pa = np.where(self.B[:, i] == 1)[0]
        Xc, Xl = X[1:], X[:-1]
        return np.c_[Xc[:, pa], Xl[:, i]], Xc[:, i], pa

    def fit(self, X: np.ndarray):
        d = X.shape[1]
        for i in range(d):
            Z, y, _ = self._design(X, i)
            m = (HistGradientBoostingRegressor(max_iter=80, max_depth=4,
                                               random_state=0)
                 if self.kind == "gb" else Ridge(alpha=1.0))
            m.fit(Z, y)
            self.models[i] = m
        self.resid_scale = self.residuals(X)
        return self

    def residuals(self, X: np.ndarray) -> np.ndarray:
        d = X.shape[1]
        out = np.zeros(d)
        for i in range(d):
            Z, y, _ = self._design(X, i)
            out[i] = float(np.abs(self.models[i].predict(Z) - y).mean())
        return out

    def rollout(self, x0: np.ndarray, steps: int, clamp: dict | None = None,
                offsets: np.ndarray | None = None, warmup: int = 0) -> np.ndarray:
        """Mean state over the steps after `warmup`, so that the transient from
        the starting state does not contaminate the counterfactual estimate."""
        d = len(x0)
        prev = x0.copy()
        acc = np.zeros(d)
        n_acc = 0
        for step in range(steps):
            cur = prev.copy()
            for i in self.order:
                pa = np.where(self.B[:, i] == 1)[0]
                z = np.r_[cur[pa], prev[i]][None, :]
                v = float(self.models[i].predict(z)[0])
                if offsets is not None:
                    v += offsets[i]
                cur[i] = clamp[i] if (clamp and i in clamp) else np.clip(v, -60, 60)
            prev = cur
            if step >= warmup:
                acc += cur
                n_acc += 1
        return acc / max(n_acc, 1)


def rca_from_surrogate(sur: SurrogateSCM, X_fault: np.ndarray,
                       ref_resid: np.ndarray) -> np.ndarray:
    """Conditional-deviation root-cause score: how much of a node's residual
    growth is NOT explained by its estimated causal parents."""
    r = sur.residuals(X_fault)
    return (r - ref_resid) / (ref_resid + 1e-6)


def signed_offsets(sur, X_fault: np.ndarray) -> np.ndarray:
    """Mean *signed* residual of each node on the fault window.

    This is the model's estimate of how each node's own mechanism has shifted.
    Rolling the surrogate forward with these offsets reproduces the *faulty*
    system, which is what a counterfactual repair has to be evaluated against.
    """
    d = X_fault.shape[1]
    out = np.zeros(d)
    for i in range(d):
        Z, y, _ = sur._design(X_fault, i)
        out[i] = float((y - sur.models[i].predict(Z)).mean())
    return out


def anomaly_scores(X_fault: np.ndarray, X_ref: np.ndarray) -> np.ndarray:
    mu, sd = X_ref.mean(0), X_ref.std(0) + 1e-8
    return np.abs(X_fault.mean(0) - mu) / sd + np.abs(X_fault.std(0) - sd) / sd


def parents_from_scores(P: np.ndarray, k_per_node: int = 3,
                        min_score: float = 0.0) -> np.ndarray:
    """Select the k strongest candidate parents of each node, without forcing a
    global DAG.

    `top_k_dag` exists for the counterfactual rollout, which needs a topological
    order. The diagnostic surrogate does not: it fits one regression per node and
    only needs that node's parents. Forcing acyclicity there is actively harmful
    on a near-binary prior, where the tie-breaking cycle removal strips most
    edges - on the Kubernetes prior it left 16 of 83 edges and 28 of 40 nodes
    with no parents at all, which silently reduces the model to the empty graph.
    """
    A = np.abs(np.asarray(P, float)).copy()
    np.fill_diagonal(A, -np.inf)
    d = A.shape[0]
    B = np.zeros((d, d), dtype=np.int8)
    for i in range(d):
        col = A[:, i]
        order = np.argsort(-col)[:k_per_node]
        for j in order:
            if np.isfinite(col[j]) and col[j] > min_score:
                B[j, i] = 1
    return B
