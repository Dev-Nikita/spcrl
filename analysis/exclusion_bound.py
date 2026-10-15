"""How much of the hard mask's damage is the representational minimum?

Proposition 1 says a true relation the prior excludes cannot be recovered. That
gives a ceiling rather than a prediction: if the prior excludes a fraction q of
the true relations, any estimator restricted to the admissible set satisfies

    recall <= 1 - q,          FN >= q |E*|,        SHD >= q |E*|,

and, if exclusions fall on recoverable and unrecoverable relations alike, the
recall of a model that was reaching r0 without the defect should land near
r0 (1 - q). The deletion operator removes exactly a fraction c of the true
relations, so q = c by construction and the two references can be computed.

This script compares them with what the hard-constrained model actually does,
which says whether exclusion costs only what it must, or whether it costs more
because the excluded parent's influence is misattributed elsewhere.

    python analysis/exclusion_bound.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.load import load  # noqa: E402

HARD = "spcrl_amortized_hard"


def main(fn="e13_fixed.jsonl"):
    d = load(fn)
    d = d[d.method == HARD]
    clean = d[d.op == "clean"]
    r0 = float(clean.recall.mean())
    a0 = float(clean.auprc.mean())
    out = {"recall_clean": r0, "auprc_clean": a0, "levels": {}}
    print(f"hard mask, clean prior: recall {r0:.3f}, AUPRC {a0:.3f}")
    print(f"{'c':>5} {'ceiling':>8} {'expected':>9} {'observed':>9} {'excess':>8}"
          f" | {'AUPRC exp':>9} {'AUPRC obs':>9} {'excess':>8}")
    for c in sorted(x for x in d[d.op == "delete"].corruption.unique()):
        s = d[(d.op == "delete") & (d.corruption == c)]
        r = float(s.recall.mean())
        a = float(s.auprc.mean())
        ceiling = 1.0 - c
        expected = r0 * (1.0 - c)
        a_expected = a0 * (1.0 - c)
        out["levels"][str(c)] = {
            "recall_ceiling": ceiling, "recall_expected": expected,
            "recall_observed": r, "recall_excess": r - expected,
            "auprc_expected": a_expected, "auprc_observed": a,
            "auprc_excess": a - a_expected, "n": int(len(s))}
        print(f"{c:5.2f} {ceiling:8.3f} {expected:9.3f} {r:9.3f} {r - expected:+8.3f}"
              f" | {a_expected:9.3f} {a:9.3f} {a - a_expected:+8.3f}")
    json.dump(out, open("results/exclusion_bound.json", "w"), indent=2)


if __name__ == "__main__":
    main(*sys.argv[1:2])
