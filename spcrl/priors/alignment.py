"""Controls for the F2 confirmatory campaign (docs/preregistration/PREREGISTRATION_F2.md).

The F1 contrast that motivated F2 compared the semantic prior P2 with a random
prior at P2's density (P3b). That control matches density and nothing else, so
"alignment" in F1 could also be P2's service-level topology, its split into
within- and between-service relations, or its degree structure. The controls
here keep all of those fixed and break ONE thing: which measured channel each
relation is attached to.

  channel_permuted   For a chosen set of services, the four channel labels of
                     each service are deranged (every channel moves) and every
                     relation of P2 is carried along. Density, the service-level
                     relation counts (how many relations run from service a to
                     service b, including a = b) and, per service, the multiset
                     of channel in- and out-degrees are preserved exactly; what
                     changes is whether "net_tx of the caller -> net_rx of the
                     callee" still points at those channels.
  semantic_wrong_id  P2 built by the same rule from a call graph whose
                     non-entry services have been relabelled (a derangement):
                     right channel semantics, wrong relation identities.

Every matrix is a function of the topology file and a seed only -- never of
telemetry or of the campaign log -- and the SHA-256 of every matrix is committed
in the pre-registration before any F2 data exist.
"""
from __future__ import annotations
import hashlib
import numpy as np

from .semantic import semantic_prior, deranged_depends_on, _service_of, _signal_of

SEED_F2 = 20261117           # fixed in PREREGISTRATION_F2.md before any F2 data
E2_LEVELS = (0, 1, 2, 5, 7, 9)   # number of services whose channels are deranged
K_DRAWS = 10                 # independent draws per misalignment level
K_WRONG_ID = 5               # independent call-graph derangements


def _derangement(n, rng):
    for _ in range(10_000):
        p = rng.permutation(n)
        if np.all(p != np.arange(n)):
            return p
    raise RuntimeError("no derangement")


def channel_permuted(P2, variables, services, rng):
    """Derange the channel labels of `services` and carry P2's relations along."""
    svc = np.array(_service_of(variables))
    mapping = np.arange(len(variables))
    for s in services:
        idx = np.where(svc == s)[0]
        mapping[idx] = idx[_derangement(len(idx), rng)]
    P = np.asarray(P2, np.float32)
    out = np.zeros_like(P)
    out[np.ix_(mapping, mapping)] = P          # out[m(i), m(j)] = P[i, j]
    np.fill_diagonal(out, 0.0)
    return out, mapping


def semantic_wrong_id(variables, depends_on, entry, kind, seed):
    dep = deranged_depends_on(depends_on, entry, seed=seed)
    return semantic_prior(variables, dep, kind)


def build_f2_priors(P2, variables, depends_on, entry, kind):
    """Every prior the F2 analysis may use, keyed by a stable name."""
    services = sorted(set(_service_of(variables)))
    out = {"P2_semantic": np.asarray(P2, np.float32)}
    for level in E2_LEVELS[1:]:
        for k in range(K_DRAWS):
            rng = np.random.default_rng([SEED_F2, level, k])
            chosen = sorted(rng.choice(services, size=level, replace=False).tolist())
            S, _ = channel_permuted(P2, variables, chosen, rng)
            out[f"CH{level}_d{k}"] = S
    for k in range(K_WRONG_ID):
        out[f"WID_d{k}"] = semantic_wrong_id(variables, depends_on, entry, kind,
                                             seed=SEED_F2 + 100 + k)
    return out


def sha256(S):
    return hashlib.sha256(np.ascontiguousarray(S, np.float32).tobytes()).hexdigest()


def invariants(S, P2, variables):
    """What a control preserves relative to P2, as machine-checkable numbers."""
    svc = np.array(_service_of(variables))
    sig = np.array(_signal_of(variables))
    names = sorted(set(svc.tolist()))
    B, R = (np.asarray(S) > 0.5), (np.asarray(P2) > 0.5)

    def block(M):
        return np.array([[int(M[np.ix_(svc == a, svc == b)].sum()) for b in names]
                         for a in names])

    def degmultiset(M):
        return {s: sorted(zip(M[svc == s].sum(1).tolist(), M[:, svc == s].sum(0).tolist()))
                for s in names}

    same_channel = int((B & R).sum())
    return {
        "n_relations": int(B.sum()),
        "density_equal": bool(B.sum() == R.sum()),
        "diagonal_zero": bool(not np.diag(B).any()),
        "binary": bool(set(np.unique(S).tolist()) <= {0.0, 1.0}),
        "service_blocks_equal": bool((block(B) == block(R)).all()),
        "within_service_relations": int(sum(B[np.ix_(svc == s, svc == s)].sum() for s in names)),
        "degree_multiset_per_service_equal": degmultiset(B) == degmultiset(R),
        "relations_shared_with_P2": same_channel,
        "fraction_relations_moved": round(1 - same_channel / max(1, int(R.sum())), 4),
    }
