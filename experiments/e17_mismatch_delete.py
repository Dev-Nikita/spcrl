"""E17: relation mismatch crossed with an incomplete prior.

E16 refuted the hypothesis that a shared exogenous driver makes a structural
prior useless: with a *correct* prior the advantage grows as the driver takes
over, because a latent common cause is a confounder and an admissibility
constraint is exactly what rejects the spurious pairs it creates.

That leaves the Kubernetes failure unexplained, and points at what the two
settings do not share. The deployment prior is a call graph over services,
expanded to channels; it is not admissibility-complete for the relations the
channels actually have, and 67.6% of the strongest observed dependencies there
are inadmissible under it. E13 showed deletion - a true relation declared
inadmissible - is the one defect a prior cannot recover from. This experiment
crosses the two: a driver the prior does not describe, and a prior that also
excludes true relations.

The question is whether the combination, not either factor alone, reproduces the
pattern the deployment shows: a soft prior below the prior-blind control.


The
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

MS = [0.0, 0.6, 0.95]
CS = [0.0, 0.2, 0.5]
MODES = ["none", "hard", "soft"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=250)
    ap.add_argument("--n-val", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--spread", type=float, default=0.8)
    ap.add_argument("--driver-observed", action="store_true")
    ap.add_argument("--out", default="results/e17_mismatch_delete.jsonl")
    args = ap.parse_args()
    W = JsonlWriter(args.out, meta=vars(args), append=True)

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                r = json.loads(line)
            except Exception:
                continue
            done.add((r.get("seed"), r.get("mismatch"), r.get("corruption"),
                      r.get("method")))
    print("run", W.run_id, f"({len(done)} cells complete)", flush=True)

    def pool(n, seed0, fams, m, c):
        return make_pool(n, seed0, fams, TRAIN_MECHS, tag="p", T=args.T,
                         mismatch=m, driver_spread=args.spread,
                         corruption=c, corrupt_op="delete")

    pools = {}
    for m in MS:
        for c in CS:
            pools[(m, c)] = {
                "iid": pool(args.n_test, 5200 + int(m * 100) + int(c * 7),
                            TRAIN_FAMILIES, m, c),
                "graph_ood": pool(args.n_test, 6200 + int(m * 100) + int(c * 7),
                                  [OOD_FAMILY], m, c)}
            for regime, ps in pools[(m, c)].items():
                for s in ps:
                    cc = compatibility(s.S, s.X)
                    W.write({"method": "_compatibility", "mismatch": m,
                             "corruption": c, "regime": regime,
                             "graph_id": s.sysinst.graph_id, "d": s.sysinst.d,
                             "spread": args.spread,
                             **{f"comp_{k}": v for k, v in cc.items()}})
    print("test pools ready", flush=True)

    for seed in args.seeds:
        for m in MS:
          for c in CS:
            for mode in MODES:
                method = f"spcrl_amortized_{mode}"
                key = (seed, m, c, method)
                if key in done:
                    continue
                t = time.time()
                tr = pool(args.n_train, 1000 + seed * 17 + int(m * 100) + int(c * 7),
                          TRAIN_FAMILIES, m, c)
                va = pool(args.n_val, 2000 + seed * 17 + int(m * 100) + int(c * 7),
                          TRAIN_FAMILIES, m, c)
                model, _ = train_amortized(tr, va, mode=mode, epochs=40, seed=seed)
                for regime, ps in pools[(m, c)].items():
                    for row in score_pool(model, ps):
                        s = [x for x in ps if x.sysinst.graph_id == row["graph_id"]][0]
                        y = s.y.astype(np.int8)
                        met = metrics_from_scores(row["P"], y)
                        if not met:
                            continue
                        rec = {"mismatch": m, "corruption": c,
                               "regime": regime, "method": method,
                               "seed": seed, "graph_id": row["graph_id"],
                               "d": row["d"], "family": row["family"],
                               "mechanism": row["mechanism"],
                               "spread": args.spread,
                               "train_sec": time.time() - t, **met}
                        W.write(rec)
                print(f"seed={seed} m={m} c={c} {mode} ({time.time()-t:.0f}s)", flush=True)
    print("E17 complete", flush=True)


if __name__ == "__main__":
    main()
