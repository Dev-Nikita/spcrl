"""Pairwise and node features used by the amortised SP-CRL model.

All features are computed from an *observational* window only; the structural
prior enters through a separate, explicitly labelled block of features so that
prior-free ablations can zero it out exactly.
"""
from __future__ import annotations
import numpy as np

PAIR_FEATURES = [
    "corr", "abs_corr", "pcorr", "lag_corr_ji", "lag_corr_ij",
    "anm_res_ji", "anm_res_ij", "anm_gain_ji", "anm_gain_ij",
    "r2_ji", "r2_ij", "var_ratio", "kurt_j", "kurt_i", "invd",
]
PRIOR_FEATURES = ["S_ji", "S_ij", "S_sym", "sp_ji", "sp_ij",
                  "deg_out_j", "deg_in_i", "common_nb"]
NODE_FEATURES = ["mean", "std", "lag1", "skew", "kurt", "range",
                 "prior_indeg", "prior_outdeg"]


def _standardise(X):
    """Per-variable standardisation that tolerates constant channels.

    Real telemetry contains variables that never move in nominal operation -
    error rates and retry counters, for example. Dividing them by their zero
    standard deviation yields NaNs that propagate into the correlation matrix
    and make its pseudo-inverse fail, so constant channels are mapped to zero
    and carry no linear information, which is exactly what they contain.
    """
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd


def _shortest_paths(S: np.ndarray, cap: int = 4) -> np.ndarray:
    d = S.shape[0]
    A = (S > 0.5).astype(np.int8)
    reach = np.full((d, d), cap + 1, dtype=np.float32)
    np.fill_diagonal(reach, 0)
    cur = A.copy()
    for k in range(1, cap + 1):
        reach = np.where((cur > 0) & (reach > k), k, reach)
        cur = (cur @ A > 0).astype(np.int8)
    return reach


def pair_features(X: np.ndarray, S: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Return (F_pair, F_node).

    F_pair: (d, d, n_pair_feat) with entry [j, i] describing the hypothesis j->i.
    F_node: (d, n_node_feat).
    """
    T, d = X.shape
    Z = _standardise(X)
    Zc, Zl = Z[1:], Z[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(Z.T)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(C, 1.0)
    try:
        P = np.linalg.pinv(C + 1e-6 * np.eye(d))
    except np.linalg.LinAlgError:
        # a rank-deficient correlation matrix still has a usable ridge inverse
        P = np.linalg.inv(C + 1e-2 * np.eye(d))
    dn = np.sqrt(np.outer(np.diag(P), np.diag(P)))
    PC = -P / (dn + 1e-12)
    LC = (Zl.T @ Zc) / max(1, len(Zc))            # LC[j, i] = corr(x_j(t-1), x_i(t))

    # additive-noise-model asymmetry with a cubic fit (vectorised per pair)
    kurt = ((Z ** 4).mean(0) - 3.0)
    skew = (Z ** 3).mean(0)
    var = Z.var(0)

    anm_res = np.zeros((d, d), dtype=np.float32)   # [j,i]: dep. of residual(i|j) on j
    r2 = np.zeros((d, d), dtype=np.float32)
    gain = np.zeros((d, d), dtype=np.float32)
    Phi = np.stack([Zc, Zc ** 2, Zc ** 3], axis=-1)     # (n, d, 3)
    n = Zc.shape[0]
    for j in range(d):
        B = np.concatenate([Phi[:, j, :], np.ones((n, 1))], axis=1)   # (n, 4)
        coef, *_ = np.linalg.lstsq(B, Zc, rcond=None)                 # (4, d)
        pred = B @ coef
        res = Zc - pred
        r2[j] = 1.0 - res.var(0) / (Zc.var(0) + 1e-9)
        anm_res[j] = np.abs((res * (Zc[:, [j]] ** 2 - (Zc[:, j] ** 2).mean())).mean(0))
        gain[j] = np.abs(coef[0])
    np.fill_diagonal(anm_res, 0); np.fill_diagonal(r2, 0); np.fill_diagonal(gain, 0)

    inv_d = np.full((d, d), 1.0 / d, dtype=np.float32)
    stack = [
        C, np.abs(C), PC, LC, LC.T,
        anm_res, anm_res.T, gain, gain.T,
        r2, r2.T,
        np.log((var[:, None] + 1e-6) / (var[None, :] + 1e-6)),
        np.broadcast_to(np.clip(kurt, -10, 10)[:, None], (d, d)),
        np.broadcast_to(np.clip(kurt, -10, 10)[None, :], (d, d)),
        inv_d,
    ]
    F_pair = np.stack([np.asarray(s, dtype=np.float32) for s in stack], axis=-1)

    if S is None:
        Sf = np.zeros((d, d, len(PRIOR_FEATURES)), dtype=np.float32)
        prior_indeg = np.zeros(d, np.float32); prior_outdeg = np.zeros(d, np.float32)
    else:
        S = np.asarray(S, dtype=np.float32)
        Ssym = np.maximum(S, S.T)
        sp = _shortest_paths(S)
        cn = ((S > 0.5).astype(np.float32) @ (S > 0.5).astype(np.float32))
        Sf = np.stack([S, S.T, Ssym, sp / 5.0, sp.T / 5.0,
                       np.broadcast_to(S.sum(1)[:, None] / d, (d, d)),
                       np.broadcast_to(S.sum(0)[None, :] / d, (d, d)),
                       cn / d], axis=-1).astype(np.float32)
        prior_indeg = (S.sum(0) / d).astype(np.float32)
        prior_outdeg = (S.sum(1) / d).astype(np.float32)

    F_pair = np.concatenate([F_pair, Sf], axis=-1)
    F_pair = np.nan_to_num(F_pair, nan=0.0, posinf=0.0, neginf=0.0)
    lag1 = (Zc * Zl).mean(0)
    F_node = np.stack([X.mean(0), X.std(0), lag1, skew, np.clip(kurt, -10, 10),
                       X.max(0) - X.min(0), prior_indeg, prior_outdeg],
                      axis=1).astype(np.float32)
    return np.nan_to_num(F_pair), np.nan_to_num(F_node)


N_PAIR_FEAT = len(PAIR_FEATURES) + len(PRIOR_FEATURES)
N_NODE_FEAT = len(NODE_FEATURES)
PRIOR_SLICE = slice(len(PAIR_FEATURES), N_PAIR_FEAT)

# The node features also end in prior-derived entries (the admissible in- and
# out-degree). They must be zeroed for the prior-blind ablation exactly as the
# pair block is: leaving them in gives a "prior-blind" control that still sees
# aggregate prior information, which is a leak, not an ablation.
NODE_PRIOR_SLICE = slice(NODE_FEATURES.index("prior_indeg"), N_NODE_FEAT)
