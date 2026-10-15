"""Two-dimensional prior-quality envelope: corruption x coarseness.

For each coarseness level r = |admissible| / |true edges| we report the mean
paired advantage of the soft and hard prior over the prior-blind control at each
corruption level, and the corruption level at which each advantage crosses zero,
obtained by linear interpolation between the bracketing levels. Bootstrap
intervals resample systems.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.load import load  # noqa: E402

B = 4000


def _paired(d, method, r, c):
    sub = d[(d.superset_ratio == r) & (d.corruption == c) &
            (d.method.isin([method, "spcrl_amortized_none"]))]
    w = sub.pivot_table(index="graph_id", columns="method", values="auprc").dropna()
    return (w[method] - w["spcrl_amortized_none"]).values


def _cross(cs, means):
    """First corruption level at which the advantage crosses zero."""
    for a, b, ma, mb in zip(cs[:-1], cs[1:], means[:-1], means[1:]):
        if ma > 0 >= mb:
            return float(a + (b - a) * ma / (ma - mb))
    return None if means[0] <= 0 else float("inf")


def main():
    d = load("e8_fixed.jsonl")
    rs = sorted(d.superset_ratio.unique())
    cs = sorted(d.corruption.unique())
    rng = np.random.default_rng(0)
    out = {"ratios": rs, "corruptions": cs, "n_cells": int(len(d.groupby(
        ["seed", "superset_ratio", "corruption"]))), "grid": {}, "crossover": {}}
    for method in ("spcrl_amortized_soft", "spcrl_amortized_hard"):
        grid, boots = [], []
        for r in rs:
            row, rowb = [], []
            for c in cs:
                diff = _paired(d, method, r, c)
                row.append(float(diff.mean()))
                rowb.append(diff)
            grid.append(row)
            boots.append(rowb)
        out["grid"][method] = {
            "mean": grid,
            "ci": [[[float(np.quantile(
                [x[rng.integers(0, len(x), len(x))].mean() for _ in range(1000)], q))
                for q in (.025, .975)] for x in rowb] for rowb in boots]}
        # A crossover is only meaningful where the prior has an advantage to
        # lose: at r = 20 the clean prior is already worth nothing, so an
        # interpolated "crossover" there would be interpolation through noise.
        cross = {}
        for i, r in enumerate(rs):
            lo = out["grid"][method]["ci"][i][0][0]
            cross[str(r)] = None if lo <= 0 else _cross(cs, grid[i])
        out["crossover"][method] = cross
    json.dump(out, open("results/e8_envelope.json", "w"), indent=2)
    for method in out["grid"]:
        print(f"\n== {method} - prior-blind (mean AUPRC advantage)")
        print("   r  " + "".join(f"{c:>8.2f}" for c in cs) + "   crossover c")
        for i, r in enumerate(rs):
            x = out["crossover"][method][str(r)]
            tag = "never useful" if x is None else (
                ">0.5" if x == float("inf") else f"{x:.3f}")
            print(f"{r:5.0f}  " + "".join(f"{v:+8.3f}" for v in out['grid'][method]['mean'][i])
                  + f"   {tag}")


if __name__ == "__main__":
    main()
