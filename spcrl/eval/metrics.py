"""Causal-recovery, diagnosis and intervention metrics."""
from __future__ import annotations
import numpy as np


def shd(B_est: np.ndarray, B_true: np.ndarray) -> int:
    """Structural Hamming Distance (reversal counts as 1)."""
    E, T = (B_est != 0).astype(int), (B_true != 0).astype(int)
    d = E.shape[0]
    cost = 0
    for i in range(d):
        for j in range(i + 1, d):
            e = (E[i, j], E[j, i])
            t = (T[i, j], T[j, i])
            if e == t:
                continue
            if e[0] + e[1] > 0 and t[0] + t[1] > 0:
                cost += 1          # wrong orientation
            else:
                cost += 1          # missing or extra edge
    return cost


def edge_prf(B_est, B_true):
    E, T = (B_est != 0), (B_true != 0)
    tp = int((E & T).sum())
    fp = int((E & ~T).sum())
    fn = int((~E & T).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def orientation_f1(B_est, B_true):
    """F1 over correctly oriented edges among skeleton-correct pairs."""
    E, T = (B_est != 0), (B_true != 0)
    skel_e = E | E.T
    skel_t = T | T.T
    common = np.argwhere(np.triu(skel_e & skel_t, 1))
    if len(common) == 0:
        return 0.0
    correct = sum(1 for i, j in common if (E[i, j] and T[i, j]) or (E[j, i] and T[j, i]))
    prec = correct / max(1, int(E.sum()))
    rec = correct / max(1, int(T.sum()))
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def topk_and_mrr(scores: np.ndarray, true_node: int, ks=(1, 3)):
    order = np.argsort(-scores)
    rank = int(np.where(order == true_node)[0][0]) + 1
    out = {f"top{k}": float(rank <= k) for k in ks}
    out["mrr"] = 1.0 / rank
    out["rank"] = rank
    return out
