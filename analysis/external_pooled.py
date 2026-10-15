"""H-X2 and H-X3 of docs/preregistration/PREREGISTRATION_EXTERNAL.md.

Both are registered as pooled tests over the second Kubernetes campaign and F1,
with campaign as a stratum in the bootstrap:

    H-X2  P1 > P6   (the identity of the relations carries information)
    H-X3  P2 > P3b  (channel semantics carries information beyond sparsity)

Faults are resampled within campaign; the statistic is the mean over the pooled
faults. The per-campaign contrasts are printed as well, because a pooled mean
that hides a sign flip between campaigns would be a misleading summary.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.load import load  # noqa: E402

B = 10000
CAMPAIGNS = {"K2b": ("e10_k2b.jsonl", "k2bctl_"), "F1": ("e10_f1.jsonl", "f1ctl_")}
VAR_FILE = {"P1_topology": None}  # P1 lives in the campaign's own e10 file


def per_fault(fn, method):
    d = load(fn)
    s = d[(d.method == method) & (d.corruption == 0.0)]
    return s.groupby("run_id_fault").top1.mean()


def variant_series(campaign, variant, mode):
    main, prefix = CAMPAIGNS[campaign]
    fn = main if variant == "P1_topology" else f"{prefix}{variant}.jsonl"
    return per_fault(fn, f"spcrl_{mode}")


def pooled(mode, a, b, rng):
    diffs, per_campaign = [], {}
    for c in CAMPAIGNS:
        sa, sb = variant_series(c, a, mode), variant_series(c, b, mode)
        sb = sb.reindex(sa.index)
        d = (sa - sb).values
        per_campaign[c] = (float(d.mean()), len(d))
        diffs.append(d)
    draws = np.zeros(B)
    n_tot = sum(len(d) for d in diffs)
    for d in diffs:                      # stratified: resample within campaign
        idx = rng.integers(0, len(d), (B, len(d)))
        draws += d[idx].sum(1)
    draws /= n_tot
    point = float(np.concatenate(diffs).mean())
    return point, float(np.quantile(draws, .025)), float(np.quantile(draws, .975)), per_campaign


def main():
    rng = np.random.default_rng(0)
    out = {}
    for name, (a, b) in {"H-X2": ("P1_topology", "P6_wrong_topology"),
                         "H-X3": ("P2_semantic", "P3b_random_at_P2_density")}.items():
        for mode in ("soft", "hard"):
            m, lo, hi, per = pooled(mode, a, b, rng)
            out[f"{name}|{mode}"] = {"contrast": f"{a} - {b}", "mean": m,
                                     "ci_lo": lo, "ci_hi": hi,
                                     "per_campaign": per,
                                     "supported": bool(lo > 0)}
            print(f"{name} ({mode}) {a} - {b}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                  f"  supported={lo > 0}")
            for c, (v, n) in per.items():
                print(f"    {c:4s} n={n:2d}  {v:+.3f}")
    json.dump(out, open("results/pooled_hx2_hx3.json", "w"), indent=2)


if __name__ == "__main__":
    main()
