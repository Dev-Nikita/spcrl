"""Does the hard mask actually exclude, or only make an edge very unlikely?

The per-system learner re-applies the mask when the adjacency is extracted
(`spcrl/causal/train.py`), so its exclusion is exact by construction. The
amortised learner instead drives the logit of an inadmissible pair to -30
(`spcrl/causal/amortized.py`), a probability of about 9.4e-14, and the graph is
then read off by taking the |E*| strongest scores. That is exclusion only as
long as no admissible pair ever scores lower, which is an empirical question,
not a definitional one.

This script extends the audit of Methods Sec. 4.3 (960 deletion rows, ceilings
computed per system from the regenerated pools) to the reverse operator as well,
from the deposited records alone and without rerunning a model.

It is a NECESSARY condition, not a sufficient one. Exceeding the ceiling would
prove that the mask leaked; respecting it does not prove that it did not, because
an excluded pair can be selected while an admissible true relation is missed, and
a selected inadmissible pair that is not a true relation moves no recall at all.
The sufficient test is `forbidden_in_graph` below, a direct count over the
estimated adjacency, which the deposited records do not carry -- they store
metrics, not matrices -- so it can only be applied to future runs. Under the deletion and reverse operators the corruption makes
n = round(c |E*|) true relations inadmissible, so an estimator that respects the
mask cannot exceed recall 1 - n/|E*|. The edge count |E*| is not stored in a
record, so a record counts as a violation only when NO plausible edge count
reproduces its recall within the ceiling -- a deliberately conservative test.

    python analysis/check_hard_mask.py <records-dir>
"""
import json, os, sys

MAX_EDGES = 120


def consistent(recall: float, c: float) -> bool:
    for m in range(3, MAX_EDGES + 1):
        tp = recall * m
        if abs(tp - round(tp)) > 1e-6:
            continue
        if recall <= 1 - round(c * m) / m + 1e-12:
            return True
    return False


def forbidden_in_graph(A_hat, S, tau: float = 0.5) -> int:
    """The direct test: how many selected pairs are inadmissible under the mask.

    For any run of the hard variant this must be 0. Call it wherever an
    estimated adjacency is available alongside the prior it was given.
    """
    import numpy as np
    A = np.asarray(A_hat) != 0
    return int((A & (np.asarray(S) <= tau)).sum())


def main(d: str) -> int:
    path = os.path.join(d, "e13_corruption_ops.jsonl")
    rows = [json.loads(l) for l in open(path)]
    sel = [r for r in rows
           if r["op"] in ("delete", "reverse")
           and r["method"] == "spcrl_amortized_hard"
           and r["corruption"] > 0]
    bad = [r for r in sel if not consistent(r["recall"], r["corruption"])]
    print(f"records checked               : {len(sel)}")
    print(f"recall above the exclusion ceiling: {len(bad)}")
    for r in bad[:10]:
        print("   ", r["graph_id"], r["op"], r["corruption"], r["recall"])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "results"))
