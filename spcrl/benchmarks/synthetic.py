"""Synthetic structural causal systems with known ground truth.

Generates:
  * a ground-truth instantaneous DAG G over d variables (4 graph families),
  * an *admissible* structural graph S (superset of G) representing the
    structural prior that an engineer/topology would supply,
  * nonlinear temporal SCM data with self-lags,
  * interventional/fault regimes with a known ground-truth root cause.

Everything is seeded and reproducible.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Literal, Dict, Any

GraphFamily = Literal["er", "sf", "modular", "hier"]
Mechanism = Literal["linear", "additive_nl", "mlp", "hetero", "cascade"]


# --------------------------------------------------------------------------
# graph generation
# --------------------------------------------------------------------------
def _random_permutation_dag(adj_upper: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Take an upper-triangular 0/1 matrix and permute node labels -> DAG."""
    d = adj_upper.shape[0]
    perm = rng.permutation(d)
    P = np.eye(d)[perm]
    return (P.T @ adj_upper @ P).astype(np.int8)


def sample_dag(d: int, family: GraphFamily, rng: np.random.Generator,
               expected_degree: float = 2.0) -> np.ndarray:
    """Return DAG adjacency A with A[i, j] == 1 meaning i -> j."""
    if family == "er":
        p = min(1.0, expected_degree / (d - 1))
        U = np.triu((rng.random((d, d)) < p).astype(np.int8), k=1)
    elif family == "sf":  # Barabasi-Albert style preferential attachment on an ordering
        m = max(1, int(round(expected_degree / 2)))
        U = np.zeros((d, d), dtype=np.int8)
        deg = np.ones(d)
        for j in range(1, d):
            k = min(m, j)
            w = deg[:j] / deg[:j].sum()
            parents = rng.choice(j, size=k, replace=False, p=w)
            U[parents, j] = 1
            deg[parents] += 1
            deg[j] += 1
    elif family == "modular":  # dense within blocks, sparse across
        n_blocks = max(2, d // 10)
        blocks = np.array_split(np.arange(d), n_blocks)
        U = np.zeros((d, d), dtype=np.int8)
        p_in = min(1.0, expected_degree / max(1, (len(blocks[0]) - 1)))
        p_out = 0.15 * p_in
        block_of = np.zeros(d, dtype=int)
        for bi, b in enumerate(blocks):
            block_of[b] = bi
        for i in range(d):
            for j in range(i + 1, d):
                p = p_in if block_of[i] == block_of[j] else p_out
                if rng.random() < p:
                    U[i, j] = 1
    elif family == "hier":  # layered / tiered topology (gateway -> service -> store)
        n_layers = max(3, int(np.ceil(np.log2(d))))
        layers = np.array_split(np.arange(d), n_layers)
        layer_of = np.zeros(d, dtype=int)
        for li, L in enumerate(layers):
            layer_of[L] = li
        U = np.zeros((d, d), dtype=np.int8)
        p_next = min(1.0, expected_degree / max(1, len(layers[0])))
        for i in range(d):
            for j in range(i + 1, d):
                gap = layer_of[j] - layer_of[i]
                if gap == 1 and rng.random() < p_next:
                    U[i, j] = 1
                elif gap == 0 and rng.random() < 0.3 * p_next:
                    U[i, j] = 1
                elif gap == 2 and rng.random() < 0.2 * p_next:
                    U[i, j] = 1
    else:
        raise ValueError(family)
    A = _random_permutation_dag(U, rng)
    assert is_dag(A), "generated graph is not acyclic"
    return A


def is_dag(A: np.ndarray) -> bool:
    d = A.shape[0]
    M = (A != 0).astype(np.int64)
    P = np.eye(d, dtype=np.int64)
    for _ in range(d):
        P = P @ M
        if np.trace(P) != 0:
            return False
    return True


# --------------------------------------------------------------------------
# structural prior (admissibility) -- a *superset* of the true DAG
# --------------------------------------------------------------------------
def build_admissible(A_true: np.ndarray, rng: np.random.Generator,
                     superset_ratio: float = 5.0) -> np.ndarray:
    """Structural prior: everything the topology says *could* interact.

    Contains all true edges plus `superset_ratio-1` times as many spurious
    admissible pairs (both orientations of a true edge count as admissible,
    which is what a topology / call-graph really tells you).
    """
    d = A_true.shape[0]
    S = A_true.copy().astype(np.int8)
    S = np.maximum(S, A_true.T)            # topology is undirected knowledge
    n_true = int(A_true.sum())
    n_extra = int(round((superset_ratio - 1.0) * n_true))
    off = [(i, j) for i in range(d) for j in range(d) if i != j and S[i, j] == 0]
    if n_extra > 0 and off:
        idx = rng.choice(len(off), size=min(n_extra, len(off)), replace=False)
        for k in idx:
            i, j = off[k]
            S[i, j] = 1
    np.fill_diagonal(S, 0)
    return S


def corrupt_prior(S: np.ndarray, A_true: np.ndarray, c: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Corrupt the structural prior at level c in [0,1].

    Half of the corruption budget removes *true* edges from the admissible
    set (making them inadmissible -> actively harmful), half adds spurious
    admissible pairs.
    """
    S = S.copy()
    d = S.shape[0]
    true_edges = np.argwhere(A_true == 1)
    n_rm = int(round(c * len(true_edges)))
    if n_rm > 0:
        sel = rng.choice(len(true_edges), size=n_rm, replace=False)
        for k in sel:
            i, j = true_edges[k]
            S[i, j] = 0
    n_add = int(round(c * len(true_edges)))
    zeros = np.argwhere((S == 0) & (~np.eye(d, dtype=bool)))
    if n_add > 0 and len(zeros):
        sel = rng.choice(len(zeros), size=min(n_add, len(zeros)), replace=False)
        for k in sel:
            i, j = zeros[k]
            S[i, j] = 1
    return S


def corrupt_prior_op(S: np.ndarray, A_true: np.ndarray, c: float, op: str,
                     rng: np.random.Generator) -> np.ndarray:
    """Corrupt the prior with ONE named operator, at level c.

    The default `corrupt_prior` applies deletions and additions together, which
    is realistic but confounds two defects with opposite character: a deletion
    makes a true relation inadmissible and is therefore unrecoverable under a
    hard mask, while an addition merely enlarges the hypothesis space. The
    operators are separated here so that the crossover can be attributed.

    op:
      "delete"  remove a fraction c of the *true* relations from the admissible
                set (the harmful direction)
      "add"     add spurious admissible pairs equal to a fraction c of the true
                edge count (the permissive direction)
      "reverse" flip a fraction c of the true relations, so that the admissible
                set asserts the wrong direction and forbids the right one
      "both"    the default: half the budget on delete, half on add
    """
    S = np.asarray(S, dtype=np.float32).copy()
    d = S.shape[0]
    true_edges = np.argwhere(A_true == 1)
    n = int(round(c * len(true_edges)))
    if n <= 0:
        return S
    if op in ("delete", "both"):
        sel = rng.choice(len(true_edges), size=min(n, len(true_edges)), replace=False)
        for k in sel:
            i, j = true_edges[k]
            S[i, j] = 0
    if op in ("add", "both"):
        zeros = np.argwhere((S == 0) & (~np.eye(d, dtype=bool)))
        if len(zeros):
            sel = rng.choice(len(zeros), size=min(n, len(zeros)), replace=False)
            for k in sel:
                i, j = zeros[k]
                S[i, j] = 1
    if op == "reverse":
        sel = rng.choice(len(true_edges), size=min(n, len(true_edges)), replace=False)
        for k in sel:
            i, j = true_edges[k]
            S[i, j] = 0
            S[j, i] = 1
    if op not in ("delete", "add", "reverse", "both"):
        raise ValueError(f"unknown corruption operator: {op}")
    return S


def prior_missing(S: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Withhold the prior for a fraction of the variables.

    Real structural descriptions are rarely complete: a deployment manifest
    covers the services it declares and says nothing about a sidecar, a managed
    database or a component added after the manifest was written. That is not
    the same defect as an incorrect prior. A variable with no prior information
    must be treated as *unconstrained* - every relation into and out of it is
    admissible - rather than as inadmissible, which would be a silent deletion.

    Returns a copy of S in which the rows and columns of the selected variables
    are set to 1 (all relations admissible), except the diagonal.
    """
    S = np.asarray(S, dtype=np.float32).copy()
    d = S.shape[0]
    k = int(round(frac * d))
    if k <= 0:
        return S
    sel = rng.choice(d, size=min(k, d), replace=False)
    S[sel, :] = 1.0
    S[:, sel] = 1.0
    np.fill_diagonal(S, 0.0)
    return S


def prior_variant(S_full: np.ndarray, A_true: np.ndarray, kind: str,
                  node_type: np.ndarray | None = None) -> np.ndarray:
    """Prior-type ablation: P1..P5 (see paper Section 'prior types')."""
    if kind == "none":
        d = S_full.shape[0]
        return np.ones((d, d), dtype=np.float32) - np.eye(d, dtype=np.float32)
    if kind == "topology":          # P1: undirected admissibility only
        return np.maximum(S_full, S_full.T).astype(np.float32)
    if kind == "topology_dir":      # P2: + partial knowledge of direction
        return S_full.astype(np.float32)
    if kind == "topology_type":     # P3: + node-type constraints
        S = np.maximum(S_full, S_full.T).astype(np.float32)
        if node_type is not None:
            # sinks (type 2) may not cause sources (type 0)
            for i in range(S.shape[0]):
                for j in range(S.shape[1]):
                    if node_type[i] == 2 and node_type[j] == 0:
                        S[i, j] = 0.0
        return S
    if kind == "full":              # P5
        return S_full.astype(np.float32)
    raise ValueError(kind)


# --------------------------------------------------------------------------
# mechanisms
# --------------------------------------------------------------------------
def _make_mlp(fan_in: int, rng: np.random.Generator, hidden: int = 8):
    W1 = rng.uniform(-2.0, 2.0, size=(fan_in, hidden))
    W1[np.abs(W1) < 0.5] += 0.5 * np.sign(W1[np.abs(W1) < 0.5] + 1e-9)
    W2 = rng.uniform(-2.0, 2.0, size=(hidden,))
    return W1, W2


@dataclass
class SCM:
    A: np.ndarray                 # true DAG, A[i,j]=1 => i->j
    d: int
    mechanism: Mechanism
    order: np.ndarray             # topological order
    params: Dict[int, Any] = field(default_factory=dict)
    self_lag: np.ndarray = None   # AR(1) coefficient per node
    noise_scale: np.ndarray = None
    node_type: np.ndarray = None
    driver_load: np.ndarray = None   # alpha_i: loading on the shared exogenous
                                     # driver L_t (None = no driver). See
                                     # docs/preregistration/PREREGISTRATION_E16.md.
    driver_rho: float = 0.9          # AR(1) coefficient of L_t

    def f(self, i: int, pa_vals: np.ndarray) -> np.ndarray:
        """pa_vals: (n, k) values of parents of i."""
        if pa_vals.shape[1] == 0:
            return np.zeros(pa_vals.shape[0])
        p = self.params[i]
        if self.mechanism == "linear":
            return pa_vals @ p["w"]
        if self.mechanism == "additive_nl":
            out = np.zeros(pa_vals.shape[0])
            for k in range(pa_vals.shape[1]):
                a, b, kind = p["a"][k], p["b"][k], p["kind"][k]
                x = pa_vals[:, k]
                if kind == 0:
                    out += a * np.tanh(b * x)
                elif kind == 1:
                    out += a * np.sin(b * x)
                else:
                    out += a * (x ** 2) * 0.3 + 0.3 * a * x
            return out
        if self.mechanism == "cascade":
            # amplifying, mildly nonlinear mechanisms: upstream deviations grow
            g, s_ = p["g"], p["s"]
            out = np.zeros(pa_vals.shape[0])
            for k in range(pa_vals.shape[1]):
                x = pa_vals[:, k]
                out += g[k] * (x + s_[k] * np.sign(x) * np.minimum(x ** 2, 9.0) * 0.1)
            return out / max(1.0, np.sqrt(pa_vals.shape[1]))
        if self.mechanism in ("mlp", "hetero"):
            W1, W2 = p["W1"], p["W2"]
            return np.tanh(pa_vals @ W1) @ W2
        raise ValueError(self.mechanism)


def build_scm(A: np.ndarray, mechanism: Mechanism, rng: np.random.Generator,
              mech_seed_offset: int = 0, mismatch: float = 0.0,
              driver_spread: float = 0.0) -> SCM:
    """Build the structural equations.

    `mismatch` in [0, 1) adds a shared, unobserved exogenous driver L_t whose
    loading is set so that it contributes that share of each variable's
    exogenous variance: alpha_i = noise_scale_i * sqrt(m / (1 - m)). At m = 0
    the system is exactly the one used everywhere else in the paper. The driver
    is deliberately *not* part of the structural prior, which is what makes the
    prior describe a different relation than the data realise.
    """
    d = A.shape[0]
    mrng = np.random.default_rng(rng.integers(0, 2**31 - 1) + mech_seed_offset)
    order = topological_order(A)
    params: Dict[int, Any] = {}
    for i in range(d):
        pa = np.where(A[:, i] == 1)[0]
        k = len(pa)
        if mechanism == "linear":
            w = mrng.uniform(0.5, 2.0, size=k) * mrng.choice([-1, 1], size=k)
            params[i] = {"w": w}
        elif mechanism == "cascade":
            params[i] = {"g": mrng.uniform(1.0, 2.0, size=k) * mrng.choice([-1, 1], size=k),
                         "s": mrng.uniform(0.0, 1.0, size=k)}
        elif mechanism == "additive_nl":
            params[i] = {"a": mrng.uniform(0.7, 1.8, size=k) * mrng.choice([-1, 1], size=k),
                         "b": mrng.uniform(0.7, 1.6, size=k),
                         "kind": mrng.integers(0, 3, size=k)}
        else:
            W1, W2 = _make_mlp(max(k, 1), mrng)
            params[i] = {"W1": W1, "W2": W2}
    in_deg = A.sum(0)
    out_deg = A.sum(1)
    node_type = np.where(in_deg == 0, 0, np.where(out_deg == 0, 2, 1))
    noise_scale = mrng.uniform(0.4, 1.0, size=d)
    if not 0.0 <= mismatch < 1.0:
        raise ValueError("mismatch must be in [0, 1)")
    driver_load = None
    if mismatch > 0:
        driver_load = noise_scale * np.sqrt(mismatch / (1.0 - mismatch))
        if driver_spread > 0:
            # Heterogeneous sensitivity to the driver. With identical loadings
            # the driver contributes a rank-1 term that is nearly constant
            # across pairs, so it shifts every correlation by about the same
            # amount and leaves the ordering of pairs - and therefore the
            # compatibility statistic - largely intact. Real components respond
            # to a shared driver with different sensitivities, and it is the
            # resulting pattern, not the shift, that competes with the causal
            # structure. driver_spread is the standard deviation of a lognormal
            # multiplier with unit mean; 0 reproduces the homogeneous case.
            u = mrng.lognormal(mean=-0.5 * driver_spread ** 2,
                               sigma=driver_spread, size=d)
            driver_load = driver_load * u
    return SCM(A=A, d=d, mechanism=mechanism, order=order, params=params,
               self_lag=mrng.uniform(0.1, 0.5, size=d),
               noise_scale=noise_scale,
               node_type=node_type, driver_load=driver_load)


def attach_driver(scm: SCM, mismatch: float, rng: np.random.Generator,
                  T: int = 600, spread: float = 0.0) -> SCM:
    """Set the driver loadings so that L_t explains `mismatch` of each
    variable's TOTAL variance.

    Calibrating against the total variance rather than the noise variance is
    what makes `mismatch` mean what it says. The parent terms in these
    mechanisms carry weights of 0.5-2.0 against noise scales of 0.4-1.0, so a
    driver sized as a share of the noise leaves the structural signal dominant
    even at m = 0.95, and the dependence ordering - hence the compatibility
    statistic - barely moves. This function simulates the system once with the
    driver off, measures each variable's standard deviation v_i, and sets

        alpha_i = v_i * sqrt(m / (1 - m)) * u_i,   E[u_i] = 1,

    so that the driver contributes the intended share of total variance. `u_i`
    is a lognormal multiplier with standard deviation `spread`, giving the
    components different sensitivities to the shared driver; spread = 0 makes
    every loading proportional to the variable's own scale.
    """
    if mismatch <= 0:
        scm.driver_load = None
        return scm
    if not 0.0 < mismatch < 1.0:
        raise ValueError("mismatch must be in [0, 1)")
    off = SCM(A=scm.A, d=scm.d, mechanism=scm.mechanism, order=scm.order,
              params=scm.params, self_lag=scm.self_lag,
              noise_scale=scm.noise_scale, node_type=scm.node_type,
              driver_load=None, driver_rho=scm.driver_rho)
    v = simulate(off, T, np.random.default_rng(rng.integers(0, 2 ** 31 - 1))).std(0)
    v = np.where(v < 1e-9, 1.0, v)
    load = v * np.sqrt(mismatch / (1.0 - mismatch))
    if spread > 0:
        load = load * rng.lognormal(mean=-0.5 * spread ** 2, sigma=spread,
                                    size=scm.d)
    scm.driver_load = load
    return scm


def topological_order(A: np.ndarray) -> np.ndarray:
    d = A.shape[0]
    indeg = A.sum(0).astype(int).copy()
    order, stack = [], [i for i in range(d) if indeg[i] == 0]
    A_ = A.copy()
    while stack:
        i = stack.pop()
        order.append(i)
        for j in np.where(A_[i] == 1)[0]:
            A_[i, j] = 0
            indeg[j] -= 1
            if indeg[j] == 0:
                stack.append(j)
    assert len(order) == d
    return np.array(order)


# --------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------
def simulate(scm: SCM, T: int, rng: np.random.Generator,
             fault_node: int | None = None, fault_t: int | None = None,
             fault_mag: float = 3.0, noise_dist: str = "gauss",
             burn_in: int = 50, clamp: dict | None = None,
             fault_type: str = "offset",
             return_driver: bool = False):
    """Return X of shape (T, d). A fault is a soft intervention that shifts the
    mechanism of `fault_node` from time `fault_t` onwards."""
    d = scm.d
    X = np.zeros((T + burn_in, d))
    prev = np.zeros(d)
    # Shared exogenous driver: a slow AR(1) process standardised to unit
    # variance, so that the loadings alone set its contribution. A slow process
    # rather than white noise, because the thing it stands for - a workload, an
    # ambient condition, a duty cycle - moves on a timescale above the sampling
    # interval, and that is what makes it look like structure.
    L = np.zeros(T + burn_in)
    if scm.driver_load is not None:
        rho = scm.driver_rho
        innov = np.sqrt(max(1e-12, 1.0 - rho ** 2))
        for t in range(1, T + burn_in):
            L[t] = rho * L[t - 1] + innov * rng.normal()
    for t in range(T + burn_in):
        if noise_dist == "gauss":
            eps = rng.normal(0, 1, size=d) * scm.noise_scale
        elif noise_dist == "laplace":
            eps = rng.laplace(0, 1 / np.sqrt(2), size=d) * scm.noise_scale
        elif noise_dist == "student":
            eps = rng.standard_t(4, size=d) / np.sqrt(2) * scm.noise_scale
        elif noise_dist == "uniform":
            eps = rng.uniform(-np.sqrt(3), np.sqrt(3), size=d) * scm.noise_scale
        else:
            raise ValueError(noise_dist)
        xt = np.zeros(d)
        for i in scm.order:
            pa = np.where(scm.A[:, i] == 1)[0]
            val = scm.f(i, xt[pa][None, :])[0] if len(pa) else 0.0
            val = val + scm.self_lag[i] * prev[i]
            if scm.mechanism == "hetero":
                val = val + eps[i] * (0.5 + 0.5 * np.abs(val))
            else:
                val = val + eps[i]
            if scm.driver_load is not None:
                val = val + scm.driver_load[i] * L[t]
            if fault_node is not None and i == fault_node and (t - burn_in) >= (fault_t or 0):
                if fault_type == "offset":
                    val = val + fault_mag
                elif fault_type == "gain":
                    val = val * (1.0 + fault_mag)
                elif fault_type == "noise":
                    val = val + rng.normal(0, fault_mag) * scm.noise_scale[i]
                elif fault_type == "stuck":
                    val = fault_mag
                else:
                    raise ValueError(fault_type)
            if clamp is not None and i in clamp and (t - burn_in) >= 0:
                val = clamp[i]
            xt[i] = np.clip(val, -50, 50)
        X[t] = xt
        prev = xt
    if return_driver:
        return X[burn_in:], L[burn_in:]
    return X[burn_in:]


@dataclass
class SystemInstance:
    graph_id: str
    d: int
    family: str
    mechanism: str
    A_true: np.ndarray
    S_admissible: np.ndarray
    scm: SCM
    meta: Dict[str, Any] = field(default_factory=dict)

    def json_meta(self):
        m = dict(self.meta)
        m.update(graph_id=self.graph_id, d=self.d, family=self.family,
                 mechanism=self.mechanism, n_edges=int(self.A_true.sum()),
                 n_admissible=int(self.S_admissible.sum()))
        return m


def make_system(graph_id: str, d: int, family: GraphFamily, mechanism: Mechanism,
                seed: int, expected_degree: float = 2.0,
                superset_ratio: float = 5.0, mech_seed_offset: int = 0) -> SystemInstance:
    rng = np.random.default_rng(seed)
    A = sample_dag(d, family, rng, expected_degree)
    S = build_admissible(A, rng, superset_ratio)
    scm = build_scm(A, mechanism, rng, mech_seed_offset=mech_seed_offset)
    return SystemInstance(graph_id=graph_id, d=d, family=family, mechanism=mechanism,
                          A_true=A, S_admissible=S, scm=scm,
                          meta={"seed": seed, "expected_degree": expected_degree,
                                "superset_ratio": superset_ratio,
                                "mech_seed_offset": mech_seed_offset})
