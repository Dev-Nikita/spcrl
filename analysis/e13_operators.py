"""Which corruption operator causes the hard-mask crossover? (E13)

Reports, per operator and protocol, the mean AUPRC of each constraint mode and
the paired advantage over the prior-blind control, with bootstrap intervals over
systems. The question is whether `delete` and `reverse` - the operators that
remove a true relation from the admissible set - reproduce the crossover of
Section 2.3 while `add`, which only enlarges the hypothesis space, does not.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.load import load  # noqa: E402

B = 4000


def _paired(d, method, op, c, protocol, regime):
    sub = d[(d.op == op) & (d.corruption == c) & (d.protocol == protocol) &
            (d.regime == regime) &
            (d.method.isin([method, "spcrl_amortized_none"]))]
    w = sub.pivot_table(index="graph_id", columns="method", values="auprc").dropna()
    if method not in w.columns or "spcrl_amortized_none" not in w.columns:
        return np.array([])
    return (w[method] - w["spcrl_amortized_none"]).values


def main(path="e13_fixed.jsonl", out="results/e13_summary.json"):
    d = load(path)
    rng = np.random.default_rng(0)
    rep = {"n_rows": int(len(d)),
           "seeds": sorted(float(s) for s in d.seed.dropna().unique()),
           "ops": sorted(d.op.dropna().unique()),
           "corruptions": sorted(float(c) for c in d.corruption.dropna().unique()),
           "cells": {}}
    for protocol in sorted(d.protocol.unique()):
        for regime in sorted(d.regime.unique()):
            for op in rep["ops"]:
                for c in rep["corruptions"]:
                    for mode in ("hard", "soft"):
                        m = f"spcrl_amortized_{mode}"
                        diff = _paired(d, m, op, c, protocol, regime)
                        if not len(diff):
                            continue
                        bs = diff[rng.integers(0, len(diff), (B, len(diff)))].mean(1)
                        rep["cells"][f"{protocol}|{regime}|{op}|c={c}|{mode}"] = {
                            "mean": float(diff.mean()),
                            "ci": [float(np.quantile(bs, .025)),
                                   float(np.quantile(bs, .975))],
                            "n": int(len(diff))}
    json.dump(rep, open(out, "w"), indent=2)
    print(f"rows={rep['n_rows']} seeds={rep['seeds']} ops={rep['ops']}")
    for key, v in rep["cells"].items():
        star = "*" if (v["ci"][0] > 0 or v["ci"][1] < 0) else " "
        print(f"{star} {key:<44} {v['mean']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")


if __name__ == "__main__":
    main(*sys.argv[1:])
