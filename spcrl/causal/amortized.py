"""Amortised SP-CRL: a single model, trained across many systems, that maps
(observational window, structural prior) -> causal adjacency posterior.

Because the model is amortised it can be applied to systems whose interaction
graph, size and mechanisms were never seen during training, which is what makes
the graph- and mechanism-shift experiments meaningful.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from ..priors.features import N_PAIR_FEAT, N_NODE_FEAT, PRIOR_SLICE, NODE_PRIOR_SLICE


class PriorAttention(nn.Module):
    def __init__(self, emb: int):
        super().__init__()
        self.q, self.k, self.v = (nn.Linear(emb, emb) for _ in range(3))
        self.o = nn.Linear(2 * emb, emb)
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.emb = emb

    def forward(self, z, S):
        q, k, v = self.q(z), self.k(z), self.v(z)
        e = q @ k.t() / np.sqrt(self.emb)
        a = torch.softmax(e + self.beta * S, dim=1)
        return torch.tanh(self.o(torch.cat([z, a @ v], dim=1)))


class AmortizedSPCRL(nn.Module):
    """mode: 'none' | 'hard' | 'soft'  (see paper Methods)."""

    def __init__(self, emb: int = 32, hidden: int = 64, layers: int = 2,
                 mode: str = "soft", use_encoder: bool = True,
                 reliability_gate: bool = False):
        """reliability_gate: the ablation baseline of the prior-attention
        encoder -- a single learned scalar gamma in (0,1) scales the whole
        prior block, so the model can learn *how much* to trust the prior but
        not *where* it applies."""
        super().__init__()
        self.mode, self.use_encoder = mode, use_encoder
        self.reliability_gate = reliability_gate
        if reliability_gate:
            self.gate = nn.Sequential(nn.Linear(N_NODE_FEAT + 2, 16), nn.ReLU(),
                                      nn.Linear(16, 1))
        self.node_in = nn.Linear(N_NODE_FEAT, emb)
        self.gnn = nn.ModuleList([PriorAttention(emb) for _ in range(layers)])
        in_dim = N_PAIR_FEAT + (3 * emb if use_encoder else 0)
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))
        self.emb = emb

    def forward(self, Fp: torch.Tensor, Fn: torch.Tensor, S: torch.Tensor):
        """Fp: (d, d, F) pair features; Fn: (d, Fn); S: (d, d) prior in [0,1].
        Returns logits (d, d) for j -> i."""
        d = Fn.shape[0]
        # Out-of-place only: an in-place write into a slice of a tensor that
        # autograd already saw breaks the backward pass.
        if self.mode == "none":
            Fp = torch.cat([Fp[..., :PRIOR_SLICE.start],
                            torch.zeros_like(Fp[..., PRIOR_SLICE])], dim=-1)
            Fn = torch.cat([Fn[..., :NODE_PRIOR_SLICE.start],
                            torch.zeros_like(Fn[..., NODE_PRIOR_SLICE])], dim=-1)
            S = torch.zeros_like(S)
        if self.reliability_gate and self.mode != "none":
            summ = torch.cat([Fn.mean(0),
                              S.mean().reshape(1), (S > 0.5).float().mean().reshape(1)])
            gamma = torch.sigmoid(self.gate(summ))
            Fp = torch.cat([Fp[..., :PRIOR_SLICE.start],
                            Fp[..., PRIOR_SLICE] * gamma], dim=-1)
            # the node-level prior features are part of the prior block and are
            # scaled by the same reliability scalar
            Fn = torch.cat([Fn[..., :NODE_PRIOR_SLICE.start],
                            Fn[..., NODE_PRIOR_SLICE] * gamma], dim=-1)
        parts = [Fp]
        if self.use_encoder:
            z = torch.tanh(self.node_in(Fn))
            for layer in self.gnn:
                z = layer(z, S)
            zj = z.unsqueeze(1).expand(d, d, self.emb)
            zi = z.unsqueeze(0).expand(d, d, self.emb)
            parts += [zj, zi, zj * zi]
        h = torch.cat(parts, dim=-1)
        logit = self.head(h).squeeze(-1)
        eye = torch.eye(d, device=logit.device, dtype=torch.bool)
        logit = logit.masked_fill(eye, -30.0)
        if self.mode == "hard":
            logit = logit.masked_fill(S <= 0.5, -30.0)
        return logit


def build_variant(name: str) -> "AmortizedSPCRL":
    """The five model variants compared in the paper.

    none        : prior-blind control (prior block zeroed, no attention bias)
    hard        : prior as a binary admissibility mask on the logits
    soft        : full SP-CRL -- prior features + prior-biased attention
    soft_noenc  : prior features only, no prior-biased attention  (baseline A)
    soft_gate   : prior features scaled by one learned reliability scalar,
                  no prior-biased attention                        (baseline B)
    """
    cfg = {"none": dict(mode="none", use_encoder=True, reliability_gate=False),
           "hard": dict(mode="hard", use_encoder=True, reliability_gate=False),
           "soft": dict(mode="soft", use_encoder=True, reliability_gate=False),
           "soft_noenc": dict(mode="soft", use_encoder=False, reliability_gate=False),
           "soft_gate": dict(mode="soft", use_encoder=False, reliability_gate=True)}
    if name not in cfg:
        raise ValueError(name)
    return AmortizedSPCRL(**cfg[name])


def predict_adjacency(model, Fp, Fn, S) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        lg = model(torch.tensor(Fp), torch.tensor(Fn),
                   torch.zeros(Fn.shape[0], Fn.shape[0]) if S is None
                   else torch.tensor(np.asarray(S, np.float32)))
        return torch.sigmoid(lg).numpy()
