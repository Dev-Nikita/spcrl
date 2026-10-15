"""H-X6 of docs/preregistration/PREREGISTRATION_EXTERNAL.md (descriptive).

Across the external campaigns and their prior variants, the Spearman
correlation between the compatibility statistic C -- computable from the data
and the prior alone -- and the measured soft-prior contrast against the
prior-blind model. On synthetic data C anti-predicted the prior's usefulness;
this asks whether it does so on running systems. It supports no method either
way, and the pre-registration states in advance that it is descriptive.

    python analysis/external_hx6.py <campaign>=<data dir> [...]

Only campaigns whose nominal telemetry is available can enter: C is computed
from the data, not from the result records. The first Kubernetes campaign is
therefore absent -- its raw telemetry was lost, and the paper says so.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.load import load                                   # noqa: E402
from spcrl.benchmarks.real import load_validated                 # noqa: E402
from spcrl.priors.compatibility import compatibility             # noqa: E402
from spcrl.priors import semantic as sem                         # noqa: E402

CAMPAIGNS = {
    "K1": ("e10_fixed.jsonl", "k1ctl_", "benchmarks/microservices/k8s/topology.yaml"),
    "K2b": ("e10_k2b.jsonl", "k2bctl_", "benchmarks/microservices/k8s/topology.yaml"),
    "F1": ("e10_f1.jsonl", "f1ctl_", "benchmarks/telecom/f1_topology.yaml"),
}
VARIANTS = ["P1_topology", "P2_semantic", "P3_random",
            "P3b_random_at_P2_density", "P4_degree_shuffled", "P6_wrong_topology"]


def contrast(main_f, prefix, variant):
    """soft(variant) - prior-blind, paired over faults, corruption 0."""
    blind = load(main_f)
    blind = blind[(blind.method == "spcrl_none") & (blind.corruption == 0.0)]
    blind = blind.groupby("run_id_fault").top1.mean()
    fn = main_f if variant == "P1_topology" else f"{prefix}{variant}.jsonl"
    d = load(fn)
    s = d[(d.method == "spcrl_soft") & (d.corruption == 0.0)]
    s = s.groupby("run_id_fault").top1.mean().reindex(blind.index)
    return float((s - blind).mean())


def main(*args):
    rows = []
    pairs = [a.split("=", 1) for a in args if "=" in a]
    for name, data in pairs:
        if name not in CAMPAIGNS or not os.path.isdir(data):
            raise SystemExit(f"unknown campaign or missing directory: {name}={data}")
        main_f, prefix, topo_path = CAMPAIGNS[name]
        S0, meta, normals, _, _ = load_validated(data)
        X = np.concatenate(normals, 0)
        variables = meta["variables"]
        topo = yaml.safe_load(open(topo_path))
        dep = {x["name"]: set(x["depends_on"]) for x in topo["services"]}
        entry = topo.get("entrypoint", topo["services"][0]["name"])
        kind = topo.get("semantics", "microservice")
        priors = sem.variants(S0, variables, dep, entry, kind)
        for v in VARIANTS:
            S = priors[v]
            c = compatibility(S, X)["C"]
            try:
                g = contrast(main_f, prefix, v)
            except FileNotFoundError:
                continue
            rows.append({"campaign": name, "variant": v, "C": float(c),
                         "soft_minus_blind": g})
            print(f"{name:4s} {v:26s} C={c:.3f}  soft-P5={g:+.3f}")
    from scipy.stats import spearmanr
    C = [r["C"] for r in rows]
    G = [r["soft_minus_blind"] for r in rows]
    rho, p = spearmanr(C, G)
    out = {"rows": rows, "spearman_rho": float(rho), "p_value": float(p),
           "n": len(rows)}
    per = {}
    for name in {r["campaign"] for r in rows}:
        sub = [r for r in rows if r["campaign"] == name]
        if len(sub) > 2:
            r_, p_ = spearmanr([x["C"] for x in sub],
                               [x["soft_minus_blind"] for x in sub])
            per[name] = {"rho": float(r_), "p": float(p_), "n": len(sub)}
    out["per_campaign"] = per
    json.dump(out, open("results/hx6.json", "w"), indent=2)
    print(f"\npooled Spearman rho = {rho:+.3f} (p = {p:.3f}, n = {len(rows)})")
    for k, v in per.items():
        print(f"  {k:4s} rho = {v['rho']:+.3f} (p = {v['p']:.3f}, n = {v['n']})")


if __name__ == "__main__":
    main(*sys.argv[1:])
