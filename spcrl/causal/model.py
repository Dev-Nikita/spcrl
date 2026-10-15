"""SP-CRL: Structural-Prior Causal Representation Learning.

A differentiable causal-discovery model over multivariate time series in which
an *uncertain* structural prior M in [0,1]^{d x d} regularises

  (i)  the latent node representation (prior-biased attention encoder), and
  (ii) the estimated causal adjacency (hard mask OR soft penalty).

Three constraint modes are supported and are the central comparison of the
paper:
  mode="none"  -> unconstrained differentiable causal learner (NOTEARS-MLP)
  mode="hard"  -> admissibility used as a binary mask on the adjacency
  mode="soft"  -> admissibility used as a weighted penalty (SP-CRL)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def matrix_exp_trace_h(A: torch.Tensor) -> torch.Tensor:
    """NOTEARS acyclicity: h(A) = tr(exp(A o A)) - d."""
    d = A.shape[0]
    return torch.trace(torch.matrix_exp(A * A)) - d


class PriorAttentionEncoder(nn.Module):
    """Node encoder whose attention logits are biased by the structural prior.

        alpha_uv = softmax_u( e_uv + beta * S_uv )

    beta is learned, so the model can *ignore* the prior if it is unhelpful.
    """

    def __init__(self, n_feat: int, emb: int = 16, layers: int = 2):
        super().__init__()
        self.inp = nn.Linear(n_feat, emb)
        self.q = nn.ModuleList([nn.Linear(emb, emb) for _ in range(layers)])
        self.k = nn.ModuleList([nn.Linear(emb, emb) for _ in range(layers)])
        self.v = nn.ModuleList([nn.Linear(emb, emb) for _ in range(layers)])
        self.o = nn.ModuleList([nn.Linear(2 * emb, emb) for _ in range(layers)])
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.layers = layers
        self.emb = emb

    def forward(self, feats: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        # feats: (d, n_feat); S: (d, d) prior confidence in [0,1]
        z = torch.tanh(self.inp(feats))
        for l in range(self.layers):
            q, k, v = self.q[l](z), self.k[l](z), self.v[l](z)
            e = q @ k.t() / np.sqrt(self.emb)
            logits = e + self.beta * S
            alpha = torch.softmax(logits, dim=1)
            msg = alpha @ v
            z = torch.tanh(self.o[l](torch.cat([z, msg], dim=1)))
        return z


class SPCRL(nn.Module):
    def __init__(self, d: int, hidden: int = 12, emb: int = 16, n_feat: int = 6,
                 mode: str = "soft", use_encoder: bool = True,
                 use_lag: bool = True, enc_layers: int = 2):
        super().__init__()
        self.d, self.hidden, self.mode = d, hidden, mode
        self.use_encoder, self.use_lag = use_encoder, use_lag
        self.W1 = nn.Parameter(torch.randn(d, d, hidden) * 0.05)
        self.b1 = nn.Parameter(torch.zeros(d, hidden))
        self.W2 = nn.Parameter(torch.randn(d, hidden) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(d))
        self.w_lag = nn.Parameter(torch.randn(d, hidden) * 0.05)
        if use_encoder:
            self.enc = PriorAttentionEncoder(n_feat, emb, enc_layers)
            self.V = nn.Parameter(torch.randn(d, emb, hidden) * 0.05)
        self.register_buffer("eye", torch.eye(d))

    # ---- adjacency -------------------------------------------------------
    def adjacency(self) -> torch.Tensor:
        """A[j, i] = strength of edge j -> i (column i = parents of i)."""
        W = self.W1 * (1 - self.eye).unsqueeze(-1)
        A = torch.sqrt((W ** 2).sum(-1) + 1e-12)         # (i, j) -> node i from j
        A = A.t()                                        # (j, i)
        return A * (1 - self.eye)

    def masked_W1(self, M: torch.Tensor | None) -> torch.Tensor:
        # never let node i predict itself from its own contemporaneous value
        W = self.W1 * (1 - self.eye).unsqueeze(-1)
        if self.mode == "hard" and M is not None:
            hard = (M.t() > 0.5).float()                 # hard[i, j]: j admissible parent of i
            W = W * hard.unsqueeze(-1)
        return W

    # ---- forward ---------------------------------------------------------
    def forward(self, X: torch.Tensor, Xlag: torch.Tensor,
                feats: torch.Tensor, S: torch.Tensor,
                M: torch.Tensor | None) -> torch.Tensor:
        """X: (n, d) contemporaneous inputs; Xlag: (n, d); returns (n, d) preds."""
        W1 = self.masked_W1(M)
        # h[n, i, hidden] = sum_j x[n, j] * W1[i, j, :]
        h = torch.einsum("nj,ijh->nih", X, W1) + self.b1.unsqueeze(0)
        if self.use_lag:
            h = h + Xlag.unsqueeze(-1) * self.w_lag.unsqueeze(0)
        if self.use_encoder:
            z = self.enc(feats, S)                               # (d, emb)
            h = h + torch.einsum("de,deh->dh", z, self.V).unsqueeze(0)
        h = torch.tanh(h)
        out = (h * self.W2.unsqueeze(0)).sum(-1) + self.b2.unsqueeze(0)
        return out

    # ---- losses ----------------------------------------------------------
    def structural_penalty(self, M: torch.Tensor) -> torch.Tensor:
        A = self.adjacency()
        return (A * (1.0 - M)).sum()

    def l1(self) -> torch.Tensor:
        return self.adjacency().sum()

    def acyclicity(self) -> torch.Tensor:
        return matrix_exp_trace_h(self.adjacency())


def window_features(X: np.ndarray) -> np.ndarray:
    """Per-node summary features fed to the encoder (d, 6)."""
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Z = (X - mu) / sd
    lag1 = (Z[1:] * Z[:-1]).mean(0)
    sk = ((Z) ** 3).mean(0)
    ku = ((Z) ** 4).mean(0) - 3.0
    rng_ = X.max(0) - X.min(0)
    return np.stack([mu, sd, lag1, sk, np.clip(ku, -10, 10), rng_], axis=1).astype(np.float32)
