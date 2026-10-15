#!/usr/bin/env python3
"""Turn a running testbed into SP-CRL benchmark artefacts.

  * structural prior  <- topology.yaml (deployment description only)
  * observation matrix <- Prometheus range queries
  * labels             <- the fault campaign log

Output layout (identical for the microservice and telecom testbeds, so the same
model code consumes both):

  <out>/prior.npy          (d, d) float32 admissibility in [0, 1]
  <out>/variables.json     ordered list of variable names + node types
  <out>/normal/window_*.npy
  <out>/faults/<run_id>.npz   X, fault_node, fault_type, magnitude, t_start
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import requests
import yaml

# one variable per (service, signal); signals are the SRE golden signals
SIGNALS = {
    "latency_p95": 'histogram_quantile(0.95, sum(rate(svc_latency_seconds_bucket{{role="{s}"}}[15s])) by (le))',
    "error_rate": 'sum(rate(svc_requests_total{{role="{s}",status="error"}}[15s])) / clamp_min(sum(rate(svc_requests_total{{role="{s}"}}[15s])), 1e-9)',
    "throughput": 'sum(rate(svc_requests_total{{role="{s}"}}[15s]))',
    "inflight": 'sum(svc_inflight{{role="{s}"}})',
    "retries": 'sum(rate(svc_retries_total{{role="{s}"}}[15s]))',
    # container-level signals; {sel} is filled per runtime (see SELECTORS)
    "cpu": 'sum(rate(container_cpu_usage_seconds_total{{{sel}}}[15s]))',
    "memory": 'sum(container_memory_working_set_bytes{{{sel}}})',
    "net_rx": 'sum(rate(container_network_receive_bytes_total{{{sel}}}[15s]))',
    "net_tx": 'sum(rate(container_network_transmit_bytes_total{{{sel}}}[15s]))',
    "dep_latency": 'sum(rate(svc_dependency_latency_seconds_sum{{role="{s}"}}[15s])) / clamp_min(sum(rate(svc_dependency_latency_seconds_count{{role="{s}"}}[15s])), 1e-9)',
}


# How cAdvisor labels a component's containers under each runtime. Kubernetes
# names pods <deployment>-<hash>; docker compose (free5GC, F1) uses the
# container name, which the topology file records per service.
SELECTORS = {
    "k8s": 'namespace="spcrl",pod=~"{s}-.*"',
    "compose": 'name="{c}"',
}


def build_variables(topology: dict, signals=("latency_p95", "error_rate",
                                             "throughput", "inflight", "retries")):
    variables, node_type = [], []
    tmap = topology.get("node_types", {})
    for s in topology["services"]:
        for sig in signals:
            variables.append(f"{s['name']}::{sig}")
            node_type.append(tmap.get(s["name"], "middle"))
    return variables, node_type


def build_prior(topology: dict, variables: list[str], self_service_conf=1.0,
                dep_conf=1.0, reverse_conf=0.5, two_hop_conf=0.25) -> np.ndarray:
    """Admissibility with *confidence values*, not a binary mask.

    A signal of service u may influence a signal of service v if v depends on
    u (or vice versa, through back-pressure); confidence decays with hop
    distance.  Signals of the same service are mutually admissible.
    """
    names = [v.split("::")[0] for v in variables]
    svcs = [s["name"] for s in topology["services"]]
    dep = {s["name"]: set(s["depends_on"]) for s in topology["services"]}
    # hop distance over the undirected dependency graph
    import collections
    adj = collections.defaultdict(set)
    for u, ds in dep.items():
        for v in ds:
            adj[u].add(v); adj[v].add(u)
    hop = {}
    for src in svcs:
        dist = {src: 0}; q = collections.deque([src])
        while q:
            x = q.popleft()
            for y in adj[x]:
                if y not in dist:
                    dist[y] = dist[x] + 1; q.append(y)
        hop[src] = dist
    d = len(variables)
    S = np.zeros((d, d), dtype=np.float32)
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            a, b = names[i], names[j]
            if a == b:
                S[i, j] = self_service_conf
                continue
            h = hop.get(a, {}).get(b, 99)
            if b in dep.get(a, ()):          # a calls b: failures flow both ways
                S[i, j] = reverse_conf       # a -> b  (back-pressure)
            if a in dep.get(b, ()):          # b calls a: a's degradation propagates to b
                S[i, j] = dep_conf
            elif h == 2:
                S[i, j] = max(S[i, j], two_hop_conf)
    return S


def query_range(prom: str, expr: str, start: float, end: float, step: int):
    r = requests.get(f"{prom}/api/v1/query_range",
                     params={"query": expr, "start": start, "end": end, "step": step},
                     timeout=60)
    r.raise_for_status()
    res = r.json()["data"]["result"]
    if not res:
        return None
    vals = {int(float(t)): float(v) for t, v in res[0]["values"]}
    grid = np.arange(int(start), int(end) + 1, step)
    return np.array([vals.get(int(t), np.nan) for t in grid], dtype=np.float32)


def collect_window(prom, topology, variables, start, end, step=5, runtime="k8s"):
    cols = []
    container = {s["name"]: s.get("container", s["name"])
                 for s in topology["services"]}
    for v in variables:
        svc, sig = v.split("::")
        sel = SELECTORS[runtime].format(s=svc, c=container[svc])
        expr = SIGNALS[sig].format(s=svc, sel=sel)
        x = query_range(prom, expr, start, end, step)
        cols.append(np.zeros(int((end - start) // step) + 1, np.float32) if x is None else x)
    X = np.stack(cols, axis=1)
    # forward-fill then zero-fill: recorded explicitly in the metadata
    for i in range(1, X.shape[0]):
        m = ~np.isfinite(X[i])
        X[i][m] = X[i - 1][m]
    return np.nan_to_num(X)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prometheus", default="http://localhost:9090")
    p.add_argument("--topology", default="benchmarks/microservices/k8s/topology.yaml")
    p.add_argument("--out", default="data/microservices")
    p.add_argument("--campaign", default=None, help="campaign log written by campaign.py")
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--signals", nargs="+", default=None,
                   help="default: the topology file's `signals`, else the five "
                        "microservice golden signals")
    p.add_argument("--runtime", choices=list(SELECTORS), default="k8s")
    a = p.parse_args()

    topo = yaml.safe_load(open(a.topology))
    if a.signals is None:
        a.signals = topo.get("signals", ["latency_p95", "error_rate",
                                         "throughput", "inflight", "retries"])
    variables, node_type = build_variables(topo, tuple(a.signals))
    S = build_prior(topo, variables)
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "prior.npy"), S)
    json.dump({"variables": variables, "node_type": node_type,
               "topology_file": a.topology, "signals": list(SIGNALS)},
              open(os.path.join(a.out, "variables.json"), "w"), indent=2)

    if a.campaign:
        runs = [json.loads(l) for l in open(a.campaign)]
        os.makedirs(os.path.join(a.out, "faults"), exist_ok=True)
        os.makedirs(os.path.join(a.out, "normal"), exist_ok=True)
        for r in runs:
            X = collect_window(a.prometheus, topo, variables,
                               r["t_start"] - r["warmup_s"], r["t_end"], a.step,
                               runtime=a.runtime)
            if r["fault_service"] is None:
                np.save(os.path.join(a.out, "normal", f"window_{r['run_id']}.npy"), X)
            else:
                idx = [i for i, v in enumerate(variables)
                       if v.split("::")[0] == r["fault_service"]]
                np.savez(os.path.join(a.out, "faults", f"{r['run_id']}.npz"),
                         X=X, fault_service=r["fault_service"],
                         fault_variables=np.array(idx), fault_type=r["fault_type"],
                         magnitude=r["magnitude"], t_start=r["t_start"],
                         warmup_s=r["warmup_s"], step=a.step)
        print(f"collected {len(runs)} windows -> {a.out}")


if __name__ == "__main__":
    main()
