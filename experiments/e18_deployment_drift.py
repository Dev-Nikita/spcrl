"""E18: the deployment condition - trained clean, deployed into a driven system
with an incomplete prior.

E16 and E17 eliminated two explanations of §2.10 under the *matched* protocol: a
shared driver alone increases the prior's advantage, and a driver crossed with
deletions still leaves the soft prior above the prior-blind control in all 18
cells. But the matched protocol is not what happened in the deployment. The
amortised model applied to the Kubernetes telemetry was trained on synthetic
systems with clean priors and no shared driver, and then met, all at once, a
prior that excludes true relations and data whose dependence is dominated by a
workload driver. That is the drift protocol, and §2.3 already reports the soft
prior failing there on deletions alone.

This experiment trains one model per (mode, seed) on the paper's ordinary clean
training distribution and scores it, without retraining, across the grid of
mismatch and deletion levels. It is the closest synthetic analogue of the
deployment condition that we can construct, and it is a prediction that can be
wrong: if the soft prior stays above the prior-blind control here too, then none
of our candidate explanations survives and §2.10 must say so.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np

from spcrl.benchmarks.dataset import (make_pool, TRAIN_FAMILIES, TRAIN_MECHS,
                                      OOD_FAMILY)
from spcrl.causal.train_amortized import train_amortized, score_pool
from spcrl.priors.compatibility import compatibility
from spcrl.util import JsonlWriter
from experiments.e1_main import metrics_from_scores

MS = [0.0, 0.6, 0.95]
CS = [0.0, 0.2, 0.5]
MODES = ["none", "hard", "soft"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=250)
    ap.add_argument("--n-val", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--spread", type=float, default=0.8)
    ap.add_argument("--out", default="results/e18_deployment_drift.jsonl")
    args = ap.parse_args()
    W = JsonlWriter(args.out, meta=vars(args), append=True)

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                r = json.loads(line)
            except Exception:
                continue
            done.add((r.get("seed"), r.get("method")))
    print("run", W.run_id, f"({len(done)} model cells complete)", flush=True)

    pools = {}
    for m in MS:
        for c in CS:
            pools[(m, c)] = {
                "iid": make_pool(args.n_test, 5300 + int(m * 100) + int(c * 7),
                                 TRAIN_FAMILIES, TRAIN_MECHS, tag="p", T=args.T,
                                 mismatch=m, driver_spread=args.spread,
                                 corruption=c, corrupt_op="delete"),
                "graph_ood": make_pool(args.n_test, 6300 + int(m * 100) + int(c * 7),
                                       [OOD_FAMILY], TRAIN_MECHS, tag="p", T=args.T,
                                       mismatch=m, driver_spread=args.spread,
                                       corruption=c, corrupt_op="delete")}
            for regime, ps in pools[(m, c)].items():
                for s in ps:
                    cc = compatibility(s.S, s.X)
                    W.write({"method": "_compatibility", "mismatch": m,
                             "corruption": c, "regime": regime,
                             "graph_id": s.sysinst.graph_id, "d": s.sysinst.d,
                             **{f"comp_{k}": v for k, v in cc.items()}})
    print("test pools ready", flush=True)

    # One clean training distribution, exactly the paper's: no driver, no
    # corruption. This is what "trained on synthetic systems only" means.
    for seed in args.seeds:
        tr = make_pool(args.n_train, 1000 + seed * 17, TRAIN_FAMILIES,
                       TRAIN_MECHS, tag="train", T=args.T)
        va = make_pool(args.n_val, 2000 + seed * 17, TRAIN_FAMILIES,
                       TRAIN_MECHS, tag="val", T=args.T)
        for mode in MODES:
            method = f"spcrl_amortized_{mode}"
            if (seed, method) in done:
                continue
            t = time.time()
            model, _ = train_amortized(tr, va, mode=mode, epochs=40, seed=seed)
            for (m, c), byreg in pools.items():
                for regime, ps in byreg.items():
                    for row in score_pool(model, ps):
                        s = [x for x in ps if x.sysinst.graph_id == row["graph_id"]][0]
                        met = metrics_from_scores(row["P"], s.y.astype(np.int8))
                        if met:
                            W.write({"mismatch": m, "corruption": c,
                                     "regime": regime, "method": method,
                                     "seed": seed, "graph_id": row["graph_id"],
                                     "d": row["d"], "family": row["family"],
                                     "mechanism": row["mechanism"],
                                     "train_sec": time.time() - t, **met})
            print(f"seed={seed} {mode} scored on {len(pools)} cells "
                  f"({time.time()-t:.0f}s)", flush=True)
    print("E18 complete", flush=True)


if __name__ == "__main__":
    main()
