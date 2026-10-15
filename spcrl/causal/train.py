"""Augmented-Lagrangian trainer for SP-CRL and its unconstrained/hard variants."""
from __future__ import annotations
import numpy as np
import torch
from .model import SPCRL, window_features


def fit_spcrl(X: np.ndarray, M: np.ndarray | None, *, mode: str = "soft",
              lambda_struct: float = 0.05, lambda_l1: float = 0.02,
              hidden: int = 12, emb: int = 16, use_encoder: bool = True,
              use_lag: bool = True, max_outer: int = 6, inner_iters: int = 250,
              lr: float = 0.02, seed: int = 0, device: str = "cpu",
              rho_init: float = 1.0, rho_max: float = 1e8, h_tol: float = 1e-8,
              verbose: bool = False):
    """Fit the model on *observational* data X (T, d). Returns (model, info)."""
    torch.manual_seed(seed)
    T, d = X.shape
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mu) / sd
    feats = torch.tensor(window_features(X), device=device)
    Xc = torch.tensor(Xs[1:], dtype=torch.float32, device=device)
    Xl = torch.tensor(Xs[:-1], dtype=torch.float32, device=device)
    if M is None:
        Mt = None
        St = torch.zeros(d, d, device=device)
    else:
        Mt = torch.tensor(np.asarray(M, dtype=np.float32), device=device)
        St = Mt
    model = SPCRL(d, hidden=hidden, emb=emb, n_feat=feats.shape[1], mode=mode,
                  use_encoder=use_encoder, use_lag=use_lag).to(device)

    rho, alpha, h_prev = rho_init, 0.0, np.inf
    for outer in range(max_outer):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for it in range(inner_iters):
            opt.zero_grad()
            pred = model(Xc, Xl, feats, St, Mt)
            mse = ((pred - Xc) ** 2).mean()
            h = model.acyclicity()
            loss = mse + 0.5 * rho * h * h + alpha * h + lambda_l1 * model.l1() / d
            if mode == "soft" and Mt is not None:
                loss = loss + lambda_struct * model.structural_penalty(Mt) / d
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        with torch.no_grad():
            h_val = float(model.acyclicity())
        if verbose:
            print(f"  outer {outer} h={h_val:.3e} mse={float(mse):.4f} rho={rho:.1e}")
        alpha += rho * h_val
        if h_val > 0.25 * h_prev:
            rho *= 10
        h_prev = h_val
        if h_val <= h_tol or rho >= rho_max:
            break

    with torch.no_grad():
        A = model.adjacency().cpu().numpy()
        if mode == "hard" and M is not None:
            A = A * (np.asarray(M) > 0.5)
        pred = model(Xc, Xl, feats, St, Mt)
        mse = float(((pred - Xc) ** 2).mean())
    info = {"A_raw": A, "h": h_prev, "mse": mse, "mu": mu, "sd": sd,
            "feats": feats.cpu().numpy(), "M": None if M is None else np.asarray(M)}
    return model, info


def threshold_dag(A: np.ndarray, thresh: float = 0.3) -> np.ndarray:
    """Threshold then greedily remove smallest edges until acyclic."""
    from ..benchmarks.synthetic import is_dag
    B = (np.abs(A) > thresh).astype(np.int8)
    np.fill_diagonal(B, 0)
    if is_dag(B):
        return B
    order = np.argsort(np.abs(A[B == 1]))
    idx = np.argwhere(B == 1)
    idx = idx[np.argsort(np.abs(A[B == 1]))]
    for (i, j) in idx:
        B[i, j] = 0
        if is_dag(B):
            break
    return B


def auto_threshold(A: np.ndarray, target_edges: int | None = None) -> np.ndarray:
    """Threshold chosen without ground truth: keep edges above the knee of the
    sorted magnitude curve (or `target_edges` strongest, if supplied)."""
    a = np.abs(A.copy())
    np.fill_diagonal(a, 0)
    vals = np.sort(a[a > 0])[::-1]
    if len(vals) == 0:
        return np.zeros_like(A, dtype=np.int8)
    if target_edges is not None:
        k = min(target_edges, len(vals))
        t = vals[k - 1] - 1e-12
    else:
        t = max(0.3 * vals[0], np.quantile(a[a > 0], 0.5))
    return threshold_dag(a, t)
