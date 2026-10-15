"""Controls that separate WHAT in a deployment prior carries information.

Committed with docs/preregistration/PREREGISTRATION_EXTERNAL.md, before any data from the
campaigns it pre-registers existed. The question these controls answer is the
one Section 2.10 leaves open: when a deployment description fails to help, is
that because of

  its topology          which components call which (P1 against P6),
  its density           how many relations it admits at all (P1 against P3),
  its degree structure  which components are hubs (P1 against P4), or
  its semantics         whether "A calls B" is translated into the right
                        channel-level relations (P1 against P2)?

Every variant is built from the deployment description and a fixed seed, never
from telemetry, so none of them can encode the answer.

  P1  topology         the deployment prior exactly as the collector builds it
  P2  semantic         only the channel relations that a call physically implies,
                       with direction (load flows to the callee; latency and
                       errors flow back to the caller)
  P3  random           uniformly random admissibility at P1's density
  P4  degree-shuffled  P1 rewired by double edge swaps, preserving every
                       variable's admissible in- and out-degree
  P5  none             the prior-blind model (no variant needed here)
  P6  wrong topology   the call graph with the non-entry services' identities
                       deranged, so the description has the right shape and
                       every relation attached to the wrong component
"""
from __future__ import annotations
import collections
import numpy as np

SEED = 20260921   # fixed before any external-campaign data existed


# ---------------------------------------------------------------- helpers
def _service_of(variables):
    return [v.split("::")[0] for v in variables]


def _signal_of(variables):
    return [v.split("::")[1] if "::" in v else v for v in variables]


def binarise(S, thr=0.5):
    B = (np.asarray(S) > thr).astype(np.float32)
    np.fill_diagonal(B, 0.0)
    return B


# ---------------------------------------------------------------- P2
# Channel-level consequences of "caller A calls callee B", written down from
# how request/response systems behave, not from any data. Keys are signal
# names; `call` relations are (signal of source, signal of target, which side
# is the source), `within` relations hold inside one component.
SEMANTICS = {
    "microservice": {
        "call": [  # (source side, source signal, target signal)
            ("caller", "throughput", "throughput"),   # requests flow down
            ("callee", "latency_p95", "latency_p95"), # slowness flows up
            ("callee", "latency_p95", "inflight"),    # slow callee holds requests
            ("callee", "latency_p95", "retries"),     # timeouts trigger retries
            ("callee", "error_rate", "error_rate"),   # failures flow up
            ("callee", "error_rate", "retries"),
        ],
        "within": [
            ("throughput", "inflight"),
            ("throughput", "latency_p95"),
            ("inflight", "latency_p95"),
            ("retries", "latency_p95"),
            ("retries", "error_rate"),
        ],
    },
    # free5GC network functions expose container-level signals only (cAdvisor)
    # plus the prober; a call moves bytes out of the caller into the callee and
    # back, and work on the callee costs it CPU.
    "nf": {
        "call": [
            ("caller", "net_tx", "net_rx"),
            ("callee", "net_tx", "net_rx"),
            ("caller", "net_tx", "cpu"),
            ("callee", "latency_p95", "latency_p95"),
            ("callee", "error_rate", "error_rate"),
        ],
        "within": [
            ("net_rx", "cpu"),
            ("cpu", "net_tx"),
            ("cpu", "latency_p95"),
            ("net_rx", "memory"),
        ],
    },
}


def semantic_prior(variables, depends_on, kind="microservice"):
    """P2. `depends_on[a]` is the set of services a calls."""
    svc, sig = _service_of(variables), _signal_of(variables)
    idx = {(s, g): i for i, (s, g) in enumerate(zip(svc, sig))}
    spec = SEMANTICS[kind]
    d = len(variables)
    S = np.zeros((d, d), np.float32)
    for a, callees in depends_on.items():
        for b in callees:
            for side, src_sig, dst_sig in spec["call"]:
                src, dst = (a, b) if side == "caller" else (b, a)
                i, j = idx.get((src, src_sig)), idx.get((dst, dst_sig))
                if i is not None and j is not None:
                    S[i, j] = 1.0
    for s in set(svc):
        for src_sig, dst_sig in spec["within"]:
            i, j = idx.get((s, src_sig)), idx.get((s, dst_sig))
            if i is not None and j is not None:
                S[i, j] = 1.0
    np.fill_diagonal(S, 0.0)
    return S


# ---------------------------------------------------------------- P3
def random_prior(S1, seed=SEED):
    B = binarise(S1)
    d = B.shape[0]
    off = ~np.eye(d, dtype=bool)
    dens = float(B[off].mean())
    rng = np.random.default_rng(seed + 3)
    R = (rng.random((d, d)) < dens).astype(np.float32)
    np.fill_diagonal(R, 0.0)
    return R


# ---------------------------------------------------------------- P4
def degree_shuffled_prior(S1, seed=SEED, swaps_per_edge=20):
    """Directed double edge swaps: (a->b, c->d) becomes (a->d, c->b) when that
    creates no self-loop and no duplicate. Every variable keeps its exact
    admissible in- and out-degree."""
    B = binarise(S1).astype(bool)
    rng = np.random.default_rng(seed + 4)
    edges = [tuple(e) for e in np.argwhere(B)]
    E = set(edges)
    n_try = swaps_per_edge * len(edges)
    for _ in range(n_try):
        (a, b), (c, dd) = (edges[k] for k in rng.choice(len(edges), 2, replace=False))
        if len({a, b, c, dd}) < 4 or (a, dd) in E or (c, b) in E:
            continue
        E -= {(a, b), (c, dd)}
        E |= {(a, dd), (c, b)}
        i1, i2 = edges.index((a, b)), edges.index((c, dd))
        edges[i1], edges[i2] = (a, dd), (c, b)
    out = np.zeros_like(B, dtype=np.float32)
    for a, b in E:
        out[a, b] = 1.0
    return out


# ---------------------------------------------------------------- P6
def deranged_depends_on(depends_on, entry, seed=SEED):
    """Relabel every non-entry service to a different one (a derangement), so
    the call graph keeps its shape and every relation lands on the wrong
    component."""
    names = sorted(set(depends_on) | {x for v in depends_on.values() for x in v})
    movable = [n for n in names if n != entry]
    rng = np.random.default_rng(seed + 6)
    for _ in range(10_000):
        perm = list(rng.permutation(movable))
        if all(p != m for p, m in zip(perm, movable)):
            break
    else:
        raise RuntimeError("no derangement found")
    rel = dict(zip(movable, perm)); rel[entry] = entry
    return {rel[a]: {rel[b] for b in bs} for a, bs in depends_on.items()}


def topology_prior(variables, depends_on):
    """P1's construction, reproduced here so that P6 can be built by exactly
    the same rule from a relabelled call graph (mirrors collect.build_prior)."""
    names = _service_of(variables)
    adj = collections.defaultdict(set)
    for u, ds in depends_on.items():
        for v in ds:
            adj[u].add(v); adj[v].add(u)
    hop = {}
    for src in set(names):
        dist = {src: 0}; q = collections.deque([src])
        while q:
            x = q.popleft()
            for y in adj[x]:
                if y not in dist:
                    dist[y] = dist[x] + 1; q.append(y)
        hop[src] = dist
    d = len(variables)
    S = np.zeros((d, d), np.float32)
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            a, b = names[i], names[j]
            if a == b:
                S[i, j] = 1.0; continue
            if b in depends_on.get(a, ()):
                S[i, j] = 0.5
            if a in depends_on.get(b, ()):
                S[i, j] = 1.0
            elif hop.get(a, {}).get(b, 99) == 2:
                S[i, j] = max(S[i, j], 0.25)
    return S


def random_prior_at_density(density, d, seed=SEED):
    rng = np.random.default_rng(seed + 33)
    R = (rng.random((d, d)) < density).astype(np.float32)
    np.fill_diagonal(R, 0.0)
    return R


def variants(S1, variables, depends_on, entry, kind="microservice"):
    """All pre-registered priors, P5 excepted (it is a model, not a prior).

    P2 is far sparser than P1, so a P2 advantage could be sparsity rather than
    semantics; P3b is the random control at P2's density that separates them.
    """
    P2 = semantic_prior(variables, depends_on, kind)
    d = len(variables)
    off = ~np.eye(d, dtype=bool)
    return {
        "P1_topology": np.asarray(S1, np.float32),
        "P2_semantic": P2,
        "P3_random": random_prior(S1),
        "P3b_random_at_P2_density": random_prior_at_density(
            float(P2[off].mean()), d),
        "P4_degree_shuffled": degree_shuffled_prior(S1),
        "P6_wrong_topology": topology_prior(
            variables, deranged_depends_on(depends_on, entry)),
    }
