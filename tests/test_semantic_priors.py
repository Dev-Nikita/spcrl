"""The pre-registered prior controls must be what their names claim."""
import json, os
import numpy as np
import yaml
from spcrl.priors import semantic as sem

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _k8s():
    topo = yaml.safe_load(open(os.path.join(
        ROOT, "benchmarks/microservices/k8s/topology.yaml")))
    dep = {s["name"]: set(s["depends_on"]) for s in topo["services"]}
    sigs = ["latency_p95", "error_rate", "throughput", "inflight", "retries"]
    variables = [f"{s['name']}::{g}" for s in topo["services"] for g in sigs]
    return topo, dep, variables


def test_topology_prior_matches_collector():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "collect", os.path.join(ROOT, "benchmarks/microservices/scripts/collect.py"))
    col = importlib.util.module_from_spec(spec); spec.loader.exec_module(col)
    topo, dep, variables = _k8s()
    assert np.allclose(col.build_prior(topo, variables),
                       sem.topology_prior(variables, dep))


def test_random_prior_keeps_density():
    topo, dep, variables = _k8s()
    S1 = sem.topology_prior(variables, dep)
    R = sem.random_prior(S1)
    d = len(variables); off = ~np.eye(d, dtype=bool)
    assert abs(R[off].mean() - sem.binarise(S1)[off].mean()) < 0.03


def test_degree_shuffle_preserves_degrees_and_changes_edges():
    topo, dep, variables = _k8s()
    S1 = sem.binarise(sem.topology_prior(variables, dep))
    P4 = sem.degree_shuffled_prior(S1)
    assert np.array_equal(P4.sum(0), S1.sum(0))
    assert np.array_equal(P4.sum(1), S1.sum(1))
    assert np.diag(P4).sum() == 0
    assert (P4 != S1).mean() > 0.1


def test_wrong_topology_is_a_derangement_with_the_same_shape():
    topo, dep, variables = _k8s()
    W = sem.deranged_depends_on(dep, "gateway")
    assert sorted(len(v) for v in W.values()) == sorted(len(v) for v in dep.values())
    assert W != dep
    assert W["gateway"] != dep["gateway"]   # the entry keeps its name, not its callees


def test_semantic_prior_is_directional_and_sparser():
    topo, dep, variables = _k8s()
    P2 = sem.semantic_prior(variables, dep)
    i = variables.index("gateway::throughput"); j = variables.index("auth::throughput")
    k = variables.index("auth::latency_p95"); l = variables.index("gateway::latency_p95")
    assert P2[i, j] == 1 and P2[j, i] == 0      # load flows down
    assert P2[k, l] == 1 and P2[l, k] == 0      # latency flows up
    assert P2.sum() < sem.binarise(sem.topology_prior(variables, dep)).sum()
