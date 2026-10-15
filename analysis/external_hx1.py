"""H-X1 of docs/preregistration/PREREGISTRATION_EXTERNAL.md: the soft-prior deficit against the
prior-blind model shrinks from K1 to K2(b).

    (soft - P5)_K2b - (soft - P5)_K1

with faults resampled independently within each campaign, as registered.
Supported if the interval lies above zero.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.load import load  # noqa: E402

B = 10000


def contrast(f):
    d = load(f)
    d = d[(d.method.isin(["spcrl_soft", "spcrl_none"])) & (d.corruption == 0.0)]
    g = d.groupby(["run_id_fault", "method"]).top1.mean().unstack()
    return (g["spcrl_soft"] - g["spcrl_none"]).values


def main(k1="e10_fixed.jsonl", k2="e10_k2b.jsonl"):
    rng = np.random.default_rng(0)
    a, b = contrast(k1), contrast(k2)
    diff = b.mean() - a.mean()
    bs = (b[rng.integers(0, len(b), (B, len(b)))].mean(1)
          - a[rng.integers(0, len(a), (B, len(a)))].mean(1))
    lo, hi = float(np.quantile(bs, .025)), float(np.quantile(bs, .975))
    out = {"k1_soft_minus_blind": float(a.mean()),
           "k2b_soft_minus_blind": float(b.mean()),
           "difference": float(diff), "ci_lo": lo, "ci_hi": hi,
           "supported": bool(lo > 0), "n_k1": len(a), "n_k2b": len(b)}
    json.dump(out, open("results/hx1.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:3])
