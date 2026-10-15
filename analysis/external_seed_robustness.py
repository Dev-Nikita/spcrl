"""Seed-aware uncertainty for the external campaigns.

The pre-registered analysis averages the three training seeds within a fault and
resamples faults; that is the right unit for the comparison, but it reports no
uncertainty over the seeds themselves. On a campaign where the model is close to
chance, the seed-to-seed spread can be as large as the effect. This script
resamples faults and seeds together (a two-way bootstrap, both with replacement)
and reports the resulting interval next to the pre-registered one. It is a
robustness check, not a replacement: the registered analysis stands as written.

    python analysis/external_seed_robustness.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.load import load  # noqa: E402

B = 10000
CAMPAIGNS = {"K1": ("e10_fixed.jsonl", "k1ctl_"),
             "K2b": ("e10_k2b.jsonl", "k2bctl_"),
             "F1": ("e10_f1.jsonl", "f1ctl_")}


def matrix(fn, method):
    """faults x seeds matrix of Top-1 at corruption 0."""
    d = load(fn)
    s = d[(d.method == method) & (d.corruption == 0.0)]
    m = s.pivot_table(index="run_id_fault", columns="seed", values="top1")
    return m


def two_way(a, b, rng):
    """Bootstrap the paired difference over faults and seeds together."""
    A, B_ = a.values, b.reindex(a.index).values
    D = A - B_
    nf, ns = D.shape
    fi = rng.integers(0, nf, (B, nf))
    si = rng.integers(0, ns, (B, ns))
    draws = np.array([D[np.ix_(f, s)].mean() for f, s in zip(fi, si)])
    return float(D.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def main():
    rng = np.random.default_rng(0)
    out = {}
    for c, (main_f, prefix) in CAMPAIGNS.items():
        if not os.path.exists(os.path.join("results", main_f)):
            continue
        blind = matrix(main_f, "spcrl_none")
        rows = {"soft|P1 - P5": (matrix(main_f, "spcrl_soft"), blind),
                "hard|P1 - P5": (matrix(main_f, "spcrl_hard"), blind),
                "P5 - anomaly": (blind, matrix(main_f, "anomaly_no_structure"))}
        p2 = os.path.join("results", f"{prefix}P2_semantic.jsonl")
        p3b = os.path.join("results", f"{prefix}P3b_random_at_P2_density.jsonl")
        p6 = os.path.join("results", f"{prefix}P6_wrong_topology.jsonl")
        if os.path.exists(p2) and os.path.exists(p3b):
            rows["soft|P2 - P3b"] = (matrix(f"{prefix}P2_semantic.jsonl", "spcrl_soft"),
                                     matrix(f"{prefix}P3b_random_at_P2_density.jsonl", "spcrl_soft"))
        if os.path.exists(p6):
            rows["soft|P1 - P6"] = (matrix(main_f, "spcrl_soft"),
                                    matrix(f"{prefix}P6_wrong_topology.jsonl", "spcrl_soft"))
        for name, (x, y) in rows.items():
            m, lo, hi = two_way(x, y, rng)
            sd = float(np.nanstd(x.mean(0)))          # spread of the seed means
            out[f"{c}|{name}"] = {"mean": m, "ci_lo": lo, "ci_hi": hi,
                                  "seed_mean_sd": sd}
            print(f"{c:4s} {name:16s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                  f"   seed-mean sd {sd:.3f}")
    json.dump(out, open("results/seed_robustness.json", "w"), indent=2)


if __name__ == "__main__":
    main()
