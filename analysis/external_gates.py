"""Validity gates of docs/preregistration/PREREGISTRATION_EXTERNAL.md, checked on nominal data
before any model is run on a new campaign.

  G1  every window non-empty
  G2  at least 3,000 nominal samples
  G3  median out-of-sample cross-variable R^2 over live channels, excluding
      throughput-type channels, at least 0.10
  G4  (K2 only) mean |rho| between the throughputs of service pairs NOT
      connected in the call graph below 0.5 - the manipulation check

A campaign that fails a gate is reported as failed and not analysed further.

    python analysis/external_gates.py <data-dir> <topology.yaml> [--k2]
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spcrl.benchmarks.real import load_validated          # noqa: E402
from analysis.campaign_diagnostics import _ridge_r2       # noqa: E402

THROUGHPUT_LIKE = ("throughput", "net_rx", "net_tx")


def gates(data, topology, k2=False, seed=0):
    S, meta, normals, faults, report = load_validated(data)
    V = meta["variables"]
    Xn = np.concatenate(normals, 0)
    out = {"data": os.path.abspath(data)}

    out["G1_nonempty"] = bool(report["normal_usable"] == report["normal_total"]
                              and report["fault_usable"] == report["fault_total"])
    out["n_nominal"] = int(Xn.shape[0])
    out["G2_samples"] = bool(Xn.shape[0] >= 3000)

    sd = Xn.std(0)
    live = [k for k in range(len(V)) if sd[k] >= 1e-12]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Xn)); h = len(Xn) // 2
    Xtr, Xte = Xn[perm[:h]], Xn[perm[h:]]
    r2 = [_ridge_r2(Xtr[:, live], Xte[:, live], j)
          for j, k in enumerate(live)
          if not V[k].split("::")[1].startswith(THROUGHPUT_LIKE)]
    r2 = [x for x in r2 if x is not None]
    out["median_r2_excl_throughput"] = float(np.median(r2)) if r2 else None
    out["G3_structure"] = bool(r2 and np.median(r2) >= 0.10)

    if k2:
        topo = yaml.safe_load(open(topology))
        dep = {s["name"]: set(s["depends_on"]) for s in topo["services"]}
        thr = {v.split("::")[0]: i for i, v in enumerate(V)
               if v.endswith("::throughput")}
        names = sorted(thr)
        rhos = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if b in dep.get(a, ()) or a in dep.get(b, ()):
                    continue
                x, y = Xn[:, thr[a]], Xn[:, thr[b]]
                if x.std() > 0 and y.std() > 0:
                    rhos.append(abs(float(np.corrcoef(x, y)[0, 1])))
        out["mean_abs_rho_unconnected_throughput"] = float(np.mean(rhos)) if rhos else None
        out["G4_manipulation"] = bool(rhos and np.mean(rhos) < 0.5)

    out["passed"] = all(v for k, v in out.items() if k.startswith("G"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("topology")
    ap.add_argument("--k2", action="store_true")
    a = ap.parse_args()
    res = gates(a.data, a.topology, a.k2)
    print(json.dumps(res, indent=2))
    name = os.path.basename(os.path.normpath(a.data))
    json.dump(res, open(os.path.join("results", f"gates_{name}.json"), "w"), indent=2)
    sys.exit(0 if res["passed"] else 1)
