"""What part of a description is worth something: the five registered contrasts.

The prior controls of PREREGISTRATION_EXTERNAL.md are not a list of ablations;
together they separate the things a structural description supplies, each
against a control that holds the others fixed:

    total      P1 - P5    a description at all, against none
    identity   P1 - P6    these components related, against other plausible ones
    semantics  P2 - P3b   relations aligned with the channels, against random
                          relations of the same density
    density    P1 - P3    this sparsity pattern, against random at same density
    degree     P1 - P4    this degree structure, against its own rewiring

Each is a paired Top-1 contrast over the faults of a campaign, three training
seeds averaged within fault, 95% bootstrap intervals over faults.

    python analysis/prior_decomposition.py
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
PARTS = [("total", "P1_topology", "P5"),
         ("identity", "P1_topology", "P6_wrong_topology"),
         ("semantics", "P2_semantic", "P3b_random_at_P2_density"),
         ("density", "P1_topology", "P3_random"),
         ("degree", "P1_topology", "P4_degree_shuffled")]


def series(campaign, variant, mode):
    main, prefix = CAMPAIGNS[campaign]
    if variant == "P5":
        fn, method = main, "spcrl_none"
    else:
        fn = main if variant == "P1_topology" else f"{prefix}{variant}.jsonl"
        method = f"spcrl_{mode}"
    d = load(fn)
    s = d[(d.method == method) & (d.corruption == 0.0)]
    return s.groupby("run_id_fault").top1.mean()


def boot(diff, rng):
    x = diff.values
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))


def main(mode="soft"):
    rng = np.random.default_rng(0)
    out = {}
    print(f"mode: {mode}")
    for c in CAMPAIGNS:
        if not os.path.exists(os.path.join("results", CAMPAIGNS[c][0])):
            continue
        print(f"  {c}")
        for part, a, b in PARTS:
            try:
                sa, sb = series(c, a, mode), series(c, b, mode)
            except FileNotFoundError:
                continue
            m, lo, hi = boot(sa - sb.reindex(sa.index), rng)
            out[f"{c}|{mode}|{part}"] = {"mean": m, "ci_lo": lo, "ci_hi": hi,
                                         "contrast": f"{a} - {b}"}
            print(f"    {part:10s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    path = os.path.join("results", f"prior_decomposition_{mode}.json")
    json.dump(out, open(path, "w"), indent=2)


if __name__ == "__main__":
    main(*sys.argv[1:2])
