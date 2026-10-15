"""Leave-one-graph-family-out (E15): the prior advantage per fold.

Each fold trains on three graph families and tests on the fourth. The paired
advantage over the prior-blind control is reported for the held-out family and,
for comparison within the same fold, for the three training families.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.load import load  # noqa: E402

B = 4000


def main(path="e15_fixed.jsonl", out="results/e15_summary.json"):
    d = load(path)
    rng = np.random.default_rng(0)
    rep = {"n_rows": int(len(d)),
           "seeds": sorted(float(s) for s in d.seed.dropna().unique()),
           "folds": sorted(d.held_out.unique()), "cells": {}}
    for f in rep["folds"]:
        for regime in ("held_out", "in_dist"):
            sub = d[(d.held_out == f) & (d.regime == regime)]
            w = sub.pivot_table(index="graph_id", columns="method",
                                values="auprc").dropna()
            for mode in ("soft", "hard"):
                a, b = f"spcrl_amortized_{mode}", "spcrl_amortized_none"
                if a not in w.columns:
                    continue
                diff = (w[a] - w[b]).values
                bs = diff[rng.integers(0, len(diff), (B, len(diff)))].mean(1)
                rep["cells"][f"{f}|{regime}|{mode}"] = {
                    "mean": float(diff.mean()),
                    "ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                    "auprc": float(w[a].mean()), "auprc_none": float(w[b].mean()),
                    "n": int(len(diff))}
    held = [v["mean"] for k, v in rep["cells"].items()
            if k.endswith("|held_out|soft")]
    rep["held_out_soft_min"] = float(min(held))
    rep["held_out_soft_max"] = float(max(held))
    json.dump(rep, open(out, "w"), indent=2)
    print(f"rows={rep['n_rows']} seeds={rep['seeds']}")
    for key, v in rep["cells"].items():
        star = "*" if (v["ci"][0] > 0 or v["ci"][1] < 0) else " "
        print(f"{star} {key:<26} {v['mean']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]"
              f"   {v['auprc']:.3f} vs {v['auprc_none']:.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
