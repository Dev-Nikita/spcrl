"""Downstream tasks: root-cause diagnosis and counterfactual intervention."""
from __future__ import annotations
import numpy as np
import torch

from ..benchmarks.synthetic import simulate, topological_order, is_dag
from ..causal.model import window_features


# ------------------------------------------------------------------ residuals
def residuals(model, info, X: np.ndarray, S: np.ndarray | None) -> np.ndarray:
    """Standardised absolute residuals per node on window X (T, d)."""
    mu, sd = info["mu"], info["sd"]
    Xs = (X - mu) / sd
    d = X.shape[1]
    feats = torch.tensor(window_features(X))
    Xc = torch.tensor(Xs[1:], dtype=torch.float32)
    Xl = torch.tensor(Xs[:-1], dtype=torch.float32)
    St = torch.zeros(d, d) if S is None else torch.tensor(np.asarray(S, np.float32))
    Mt = None if info.get("M") is None else torch.tensor(np.asarray(info["M"], np.float32))
    with torch.no_grad():
        pred = model(Xc, Xl, feats, St, Mt).numpy()
    r = np.abs(pred - Xs[1:]).mean(0)
    return r


def rca_scores(r_fault: np.ndarray, r_ref: np.ndarray, A: np.ndarray,
               structural: bool = True) -> np.ndarray:
    """Root-cause score.

    Deviation of a node's residual from its nominal level, discounted by the
    deviation already explained by its estimated causal parents.  Without the
    causal term this collapses to plain anomaly ranking.
    """
    dev = (r_fault - r_ref) / (r_ref + 1e-6)
    if not structural:
        return dev
    A = np.abs(A).copy()
    np.fill_diagonal(A, 0)
    w = A / (A.sum(0, keepdims=True) + 1e-9)      # column-normalised parents
    parent_dev = w.T @ dev
    return dev - parent_dev


# ------------------------------------------------------ counterfactual rollout
def learned_rollout(model, info, X_hist: np.ndarray, S, steps: int,
                    clamp: dict | None = None) -> np.ndarray:
    """Roll the learned SCM forward `steps` steps from the last state of X_hist,
    optionally clamping nodes (an intervention).  Returns mean state."""
    mu, sd = info["mu"], info["sd"]
    d = X_hist.shape[1]
    A = np.abs(info["A_raw"]).copy()
    np.fill_diagonal(A, 0)
    B = (A > np.quantile(A[A > 0], 0.5)).astype(np.int8) if (A > 0).any() else np.zeros_like(A, np.int8)
    if not is_dag(B):
        idx = np.argwhere(B == 1)
        idx = idx[np.argsort(A[B == 1])]
        for (i, j) in idx:
            B[i, j] = 0
            if is_dag(B):
                break
    order = topological_order(B)
    feats = torch.tensor(window_features(X_hist))
    St = torch.zeros(d, d) if S is None else torch.tensor(np.asarray(S, np.float32))
    Mt = None if info.get("M") is None else torch.tensor(np.asarray(info["M"], np.float32))
    prev = (X_hist[-1] - mu) / sd
    acc = np.zeros(d)
    clamp_s = {} if clamp is None else {k: (v - mu[k]) / sd[k] for k, v in clamp.items()}
    for t in range(steps):
        cur = prev.copy()
        for i in order:
            xin = torch.tensor(cur[None, :], dtype=torch.float32)
            xlag = torch.tensor(prev[None, :], dtype=torch.float32)
            with torch.no_grad():
                out = model(xin, xlag, feats, St, Mt).numpy()[0]
            cur[i] = clamp_s[i] if i in clamp_s else np.clip(out[i], -8, 8)
        prev = cur
        acc += cur
    return acc / steps * sd + mu


def intervention_experiment(sysinst, model, info, S, fault_node: int,
                            kpi: int, rng, candidates, fault_mag=3.0,
                            T=200, n_true_reps=5):
    """Compare model-selected repair action against the true interventional
    outcome computed by re-simulating the ground-truth SCM."""
    scm = sysinst.scm
    X_nom = simulate(scm, T, np.random.default_rng(int(rng.integers(1e9))))
    nominal = X_nom.mean(0)
    X_f = simulate(scm, T, np.random.default_rng(int(rng.integers(1e9))),
                   fault_node=fault_node, fault_t=0, fault_mag=fault_mag)

    true_loss = {}
    for a in candidates:
        vals = []
        for rep in range(n_true_reps):
            Xa = simulate(scm, T, np.random.default_rng(int(rng.integers(1e9))),
                          fault_node=fault_node, fault_t=0, fault_mag=fault_mag,
                          clamp={int(a): float(nominal[a])})
            vals.append(abs(Xa[:, kpi].mean() - nominal[kpi]))
        true_loss[int(a)] = float(np.mean(vals))

    pred_loss = {}
    for a in candidates:
        s_hat = learned_rollout(model, info, X_f, S, steps=6,
                                clamp={int(a): float(nominal[a])})
        pred_loss[int(a)] = float(abs(s_hat[kpi] - nominal[kpi]))

    a_sel = min(pred_loss, key=pred_loss.get)
    best = min(true_loss.values())
    worst = max(true_loss.values())
    regret = true_loss[a_sel] - best
    norm = (worst - best) if worst > best else 1.0
    return {"selected": int(a_sel), "regret": regret, "norm_regret": regret / norm,
            "oracle_loss": best, "true_loss": true_loss, "pred_loss": pred_loss,
            "X_fault": X_f, "nominal": nominal}
