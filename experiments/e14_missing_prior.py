"""E14: what happens when the structural description covers only part of the system?

Corruption (E2, E13) asks what happens when the description is wrong.
Coarseness (E7, E8) asks what happens when it is too permissive. Neither asks
the question a practitioner meets first: the manifest covers the services it
declares and says nothing about the rest. A variable the description does not
mention must be treated as unconstrained - every relation into and out of it
admissible - because treating it as inadmissible would be a silent deletion of
whatever true relations it has.

We withhold the prior for a fraction m of the variables and measure the
structure-recovery advantage that survives. The interesting question is whether
the advantage degrades gracefully (proportional to the covered fraction) or
collapses, and whether the hard mask, which cannot be outvoted on the covered
part, degrades differently from the soft prior.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np

from spcrl.benchmarks.dataset import (make_pool, TRAIN_FAMILIES, TRAIN_MECHS,
                                      OOD_FAMILY)
from spcrl.causal.train_amortized import train_amortized, score_pool
from spcrl.util import JsonlWriter
from experiments.e1_main import metrics_from_scores

MISSING = [0.0, 0.25, 0.5, 0.75]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=250)
    ap.add_argument("--n-val", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--out", default="results/e14_missing_prior.jsonl")
    args = ap.parse_args()
    W = JsonlWriter(args.out, meta=vars(args), append=True)

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                r = json.loads(line)
            except Exception:
                continue
            done.add((r.get("seed"), r.get("missing"), r.get("method")))
    print("run", W.run_id, f"({len(done)} cells complete)", flush=True)

    pools = {m: {
        "iid": make_pool(args.n_test, 5000 + int(m * 100), TRAIN_FAMILIES,
                         TRAIN_MECHS, tag="iid", T=args.T, missing=m),
        "graph_ood": make_pool(args.n_test, 6000 + int(m * 100), [OOD_FAMILY],
                               TRAIN_MECHS, tag="graph_ood", T=args.T, missing=m)}
        for m in MISSING}
    print("test pools ready", flush=True)

    for seed in args.seeds:
        for m in MISSING:
            for mode in ["none", "hard", "soft"]:
                method = f"spcrl_amortized_{mode}"
                if (seed, m, method) in done:
                    continue
                t = time.time()
                tr = make_pool(args.n_train, 1000 + seed * 17 + int(m * 100),
                               TRAIN_FAMILIES, TRAIN_MECHS, tag="train",
                               T=args.T, missing=m)
                va = make_pool(args.n_val, 2000 + seed * 17 + int(m * 100),
                               TRAIN_FAMILIES, TRAIN_MECHS, tag="val",
                               T=args.T, missing=m)
                model, _ = train_amortized(tr, va, mode=mode, epochs=40, seed=seed)
                for regime, pool in pools[m].items():
                    for row in score_pool(model, pool):
                        s = [x for x in pool if x.sysinst.graph_id == row["graph_id"]][0]
                        met = metrics_from_scores(row["P"], s.y.astype(np.int8))
                        if met:
                            W.write({"missing": m, "regime": regime, "method": method,
                                     "seed": seed, "graph_id": row["graph_id"],
                                     "d": row["d"], "family": row["family"],
                                     "mechanism": row["mechanism"],
                                     "rho": float(np.asarray(s.S).mean()),
                                     "train_sec": time.time() - t, **met})
                print(f"seed={seed} missing={m} {mode} ({time.time()-t:.0f}s)", flush=True)
    print("E14 complete", flush=True)


if __name__ == "__main__":
    main()
