"""E16: relation mismatch - a shared exogenous driver the prior does not describe.

Pre-registered in docs/preregistration/PREREGISTRATION_E16.md before this file existed. The
generator adds an unobserved driver L_t that contributes a share `mismatch` of
each variable's total variance; the structural prior is unchanged and remains
correct about the mechanism, so the prior and the data describe different
relations without the prior containing a single wrong edge.

Two arms, matching the Kubernetes regimes the campaign can be re-run in:
  driver_observed = False  the driver is latent (regime K1, what we measured)
  driver_observed = True   the driver is an observed, unconstrained variable
                           (regime K3, the cheap fix a practitioner would try)

Models are trained on data with the same mismatch level they are tested on,
which is the analogue of the matched protocol: a learner deployed in a system
with a shared driver sees that driver's effect during training too.
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

MS = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
MODES = ["none", "hard", "soft", "soft_gate"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=250)
    ap.add_argument("--n-val", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--spread", type=float, default=0.8)
    ap.add_argument("--driver-observed", action="store_true")
    ap.add_argument("--out", default="results/e16_mismatch.jsonl")
    args = ap.parse_args()
    W = JsonlWriter(args.out, meta=vars(args), append=True)

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                r = json.loads(line)
            except Exception:
                continue
            done.add((r.get("seed"), r.get("mismatch"), r.get("method"),
                      r.get("driver_observed")))
    print("run", W.run_id, f"({len(done)} cells complete)", flush=True)

    def pool(n, seed0, fams, m):
        return make_pool(n, seed0, fams, TRAIN_MECHS, tag="p", T=args.T,
                         mismatch=m, driver_spread=args.spread,
                         driver_observed=(args.driver_observed and m > 0))

    pools = {}
    for m in MS:
        pools[m] = {"iid": pool(args.n_test, 5200 + int(m * 100),
                                TRAIN_FAMILIES, m),
                    "graph_ood": pool(args.n_test, 6200 + int(m * 100),
                                      [OOD_FAMILY], m)}
        # The compatibility statistic is a property of the data and the prior,
        # not of any model, so it is recorded once per test pool.
        for regime, ps in pools[m].items():
            for s in ps:
                c = compatibility(s.S, s.X)
                W.write({"method": "_compatibility", "mismatch": m,
                         "regime": regime, "graph_id": s.sysinst.graph_id,
                         "d": s.sysinst.d, "spread": args.spread,
                         "driver_observed": bool(args.driver_observed and m > 0),
                         **{f"comp_{k}": v for k, v in c.items()}})
    print("test pools ready", flush=True)

    for seed in args.seeds:
        for m in MS:
            for mode in MODES:
                method = f"spcrl_amortized_{mode}"
                key = (seed, m, method, bool(args.driver_observed and m > 0))
                if key in done:
                    continue
                t = time.time()
                tr = pool(args.n_train, 1000 + seed * 17 + int(m * 100),
                          TRAIN_FAMILIES, m)
                va = pool(args.n_val, 2000 + seed * 17 + int(m * 100),
                          TRAIN_FAMILIES, m)
                model, _ = train_amortized(tr, va, mode=mode, epochs=40, seed=seed)
                for regime, ps in pools[m].items():
                    for row in score_pool(model, ps):
                        s = [x for x in ps if x.sysinst.graph_id == row["graph_id"]][0]
                        y = s.y.astype(np.int8)
                        met = metrics_from_scores(row["P"], y)
                        if not met:
                            continue
                        rec = {"mismatch": m, "regime": regime, "method": method,
                               "seed": seed, "graph_id": row["graph_id"],
                               "d": row["d"], "family": row["family"],
                               "mechanism": row["mechanism"],
                               "spread": args.spread,
                               "driver_observed": bool(args.driver_observed and m > 0),
                               "train_sec": time.time() - t, **met}
                        # With the driver observed the matrix gains a row, so a
                        # second metric restricted to the observed-variable
                        # block keeps the two arms comparable.
                        if args.driver_observed and m > 0:
                            d0 = y.shape[0] - 1
                            blk = metrics_from_scores(
                                np.asarray(row["P"])[:d0, :d0], y[:d0, :d0])
                            if blk:
                                rec["auprc_block"] = blk["auprc"]
                                rec["shd_block"] = blk["shd"]
                        W.write(rec)
                print(f"seed={seed} m={m} {mode} ({time.time()-t:.0f}s)", flush=True)
    print("E16 complete", flush=True)


if __name__ == "__main__":
    main()
