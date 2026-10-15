"""Analysis of the pre-registered prior controls (PREREGISTRATION_EXTERNAL.md).

For one campaign, given one E10 results file per prior variant, reports for the
soft and hard models the paired Top-1 contrast of each prior against the
prior-blind model (P5) and the pre-registered pairwise contrasts
P1 - P6 (topology identity) and P2 - P3b (semantics beyond sparsity), with 95%
bootstrap intervals over faults; the three seeds are averaged within fault.

    python analysis/external_controls.py <campaign-name> <P1 file> <dir with k1ctl_*.jsonl-style files> <prefix>

The prior-blind model does not depend on the prior, so P5 is taken from the P1
run; its identity across files is checked, and a difference is reported as an
error rather than averaged away.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.load import load  # noqa: E402

B = 10000
VARS = ["P2_semantic", "P3_random", "P3b_random_at_P2_density",
        "P4_degree_shuffled", "P6_wrong_topology", "driver_declared"]


def per_fault(d, method):
    s = d[(d.method == method) & (d.corruption == 0.0)]
    return s.groupby("run_id_fault").top1.mean()


def boot(diff, rng):
    x = diff.values
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))


def main(name, p1_file, prefix):
    rng = np.random.default_rng(0)
    runs = {"P1_topology": load(p1_file)}
    for v in VARS:
        f = f"{prefix}{v}.jsonl"
        if os.path.exists(os.path.join("results", f)):
            runs[v] = load(f)
    blind = per_fault(runs["P1_topology"], "spcrl_none")
    anomaly = per_fault(runs["P1_topology"], "anomaly_no_structure")
    rep = {"campaign": name, "n_faults": int(len(blind)), "variants": list(runs),
           "p5_consistent": True, "vs_blind": {}, "pairwise": {}}
    for v, d in runs.items():
        b = per_fault(d, "spcrl_none")
        if not np.allclose(b.reindex(blind.index).values, blind.values):
            rep["p5_consistent"] = False
    rep["blind_vs_anomaly"] = boot(blind - anomaly, rng)
    for mode in ("soft", "hard"):
        for v, d in runs.items():
            s = per_fault(d, f"spcrl_{mode}").reindex(blind.index)
            rep["vs_blind"][f"{mode}|{v}"] = {"top1": float(s.mean()),
                                             "diff": boot(s - blind, rng)}
        for a, b in (("P1_topology", "P6_wrong_topology"),
                     ("P2_semantic", "P3b_random_at_P2_density"),
                     ("P1_topology", "P3_random"),
                     ("P1_topology", "P4_degree_shuffled")):
            if a in runs and b in runs:
                sa = per_fault(runs[a], f"spcrl_{mode}").reindex(blind.index)
                sb = per_fault(runs[b], f"spcrl_{mode}").reindex(blind.index)
                rep["pairwise"][f"{mode}|{a}-{b}"] = boot(sa - sb, rng)
    out = os.path.join("results", f"controls_{name}.json")
    json.dump(rep, open(out, "w"), indent=2)
    print(f"{name}: {rep['n_faults']} faults, P5 identical across runs: "
          f"{rep['p5_consistent']}")
    print(f"  blind - anomaly  {rep['blind_vs_anomaly'][0]:+.3f} "
          f"[{rep['blind_vs_anomaly'][1]:+.3f}, {rep['blind_vs_anomaly'][2]:+.3f}]")
    for k, v in rep["vs_blind"].items():
        m, lo, hi = v["diff"]
        print(f"  {k:36s} Top-1 {v['top1']:.3f}  vs P5 {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    for k, (m, lo, hi) in rep["pairwise"].items():
        print(f"  {k:48s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main(*sys.argv[1:4])
