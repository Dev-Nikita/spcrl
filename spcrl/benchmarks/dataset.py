"""Construction of the train / validation / OOD-test system pools."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from .synthetic import make_system, simulate, corrupt_prior
from ..priors.features import pair_features

TRAIN_FAMILIES = ["er", "sf", "modular"]
TRAIN_MECHS = ["linear", "additive_nl", "mlp"]
OOD_FAMILY = "hier"
OOD_MECHS = ["cascade", "hetero"]


@dataclass
class Sample:
    sysinst: object
    X: np.ndarray
    Fp: np.ndarray
    Fn: np.ndarray
    S: np.ndarray
    y: np.ndarray
    tag: str


def build_sample(sysinst, T: int, seed: int, noise_dist: str = "gauss",
                 corruption: float = 0.0, prior_kind: str = "full",
                 tag: str = "", standardize: bool = False,
                 corrupt_op: str = "both", missing: float = 0.0,
                 driver_observed: bool = False) -> Sample:
    rng = np.random.default_rng(seed)
    if driver_observed:
        if getattr(sysinst.scm, "driver_load", None) is None:
            raise ValueError("driver_observed=True but this system has no driver")
        X, L = simulate(sysinst.scm, T, rng, noise_dist=noise_dist,
                        return_driver=True)
    else:
        X = simulate(sysinst.scm, T, rng, noise_dist=noise_dist)
    S = sysinst.S_admissible.astype(np.float32)
    if corruption > 0:
        from .synthetic import corrupt_prior_op
        S = corrupt_prior_op(S, sysinst.A_true, corruption, corrupt_op,
                             np.random.default_rng(seed + 77)).astype(np.float32)
    if missing > 0:
        from .synthetic import prior_missing
        S = prior_missing(S, missing, np.random.default_rng(seed + 131))
    from .synthetic import prior_variant
    if prior_kind != "full":
        S = prior_variant(S, sysinst.A_true, prior_kind, sysinst.scm.node_type)
    if standardize:
        # Reisach et al. (2021): simulated additive-noise benchmarks leak the
        # causal order into marginal variances. Per-variable standardisation
        # removes that channel, at the cost of removing genuine scale
        # information as well; we report results with and without it.
        sd = X.std(0)
        sd = np.where(sd < 1e-12, 1.0, sd)   # never add an epsilon to the
        X = (X - X.mean(0)) / sd             # denominator: it would leave the
                                             # variance ordering intact at 1e-8
    A = sysinst.A_true.astype(np.float32)
    if driver_observed:
        # The driver becomes variable index d. A real manifest does not mention
        # it, so its prior row and column are unconstrained (every relation
        # admissible) rather than inadmissible - the same convention as a
        # variable the description omits. The true graph gains the edges the
        # driver actually has: L -> i for every variable it loads on.
        d0 = X.shape[1]
        X = np.concatenate([X, L[:, None]], axis=1).astype(np.float32)
        S2 = np.ones((d0 + 1, d0 + 1), np.float32)
        S2[:d0, :d0] = S
        np.fill_diagonal(S2, 0.0)
        S = S2
        A2 = np.zeros((d0 + 1, d0 + 1), np.float32)
        A2[:d0, :d0] = A
        A2[d0, :d0] = (np.abs(sysinst.scm.driver_load) > 1e-9).astype(np.float32)
        A = A2
    Fp, Fn = pair_features(X, S)
    return Sample(sysinst, X, Fp, Fn, S, A, tag)


def make_pool(n: int, seed0: int, families, mechs, d_range=(15, 30),
              T: int = 400, noise: str = "gauss", tag: str = "",
              corruption: float = 0.0, expected_degree: float = 2.0,
              prior_kind: str = "full", superset_ratio: float = 5.0,
              standardize: bool = False, corrupt_op: str = "both",
              missing: float = 0.0, mismatch: float = 0.0,
              driver_spread: float = 0.0, driver_observed: bool = False):
    rng = np.random.default_rng(seed0)
    out = []
    for k in range(n):
        d = int(rng.integers(d_range[0], d_range[1] + 1))
        fam = families[k % len(families)]
        mech = mechs[k % len(mechs)]
        gid = f"{tag}-{k:04d}"
        s = make_system(gid, d, fam, mech, seed=int(rng.integers(1e9)),
                        expected_degree=expected_degree,
                        superset_ratio=superset_ratio)
        if mismatch > 0:
            from .synthetic import attach_driver
            attach_driver(s.scm, mismatch,
                          np.random.default_rng(int(rng.integers(1e9))),
                          spread=driver_spread)
        out.append(build_sample(s, T, seed=int(rng.integers(1e9)),
                                noise_dist=noise, corruption=corruption,
                                prior_kind=prior_kind, tag=tag,
                                standardize=standardize, corrupt_op=corrupt_op,
                                missing=missing,
                                driver_observed=driver_observed))
    return out
