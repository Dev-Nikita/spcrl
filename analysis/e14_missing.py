"""Partial prior coverage (E14): paired advantage over the prior-blind control.

Withholding the prior for a fraction of the variables is not the same defect as
an incorrect prior: the withheld variables are declared unconstrained, so the
description becomes more permissive rather than wrong. The analysis therefore
also reports the admissible fraction rho, which links this experiment to the
coarseness axis of Section 2.8.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.load import load  # noqa: E402

B = 4000


def main(path="e14_fixed.jsonl", out="results/e14_summary.json"):
    d = load(path)
    rng = np.random.default_rng(0)
    rep = {"n_rows": int(len(d)),
           "seeds": sorted(float(s) for s in d.seed.dropna().unique()),
           "missing": sorted(float(m) for m in d.missing.dropna().unique()),
           "cells": {}, "rho": {}}
    for regime in sorted(d.regime.unique()):
        for m in rep["missing"]:
            sub = d[(d.regime == regime) & (d.missing == m)]
            rep["rho"][f"{regime}|m={m}"] = float(sub.rho.mean())
            w = sub.pivot_table(index="graph_id", columns="method", values="auprc").dropna()
            for mode in ("soft", "hard"):
                a, b = f"spcrl_amortized_{mode}", "spcrl_amortized_none"
                if a not in w.columns or b not in w.columns:
                    continue
                diff = (w[a] - w[b]).values
                bs = diff[rng.integers(0, len(diff), (B, len(diff)))].mean(1)
                rep["cells"][f"{regime}|m={m}|{mode}"] = {
                    "mean": float(diff.mean()),
                    "ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                    "n": int(len(diff)),
                    "auprc": float(w[a].mean()), "auprc_none": float(w[b].mean())}
    json.dump(rep, open(out, "w"), indent=2)
    print(f"rows={rep['n_rows']} seeds={rep['seeds']}")
    for key, v in rep["cells"].items():
        star = "*" if (v["ci"][0] > 0 or v["ci"][1] < 0) else " "
        regime, mm, _ = key.split("|")
        print(f"{star} {key:<28} {v['mean']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]"
              f"  rho={rep['rho'][regime + '|' + mm]:.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
