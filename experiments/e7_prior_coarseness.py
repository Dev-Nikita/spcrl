"""E7: how coarse may the structural prior be before it stops paying off?

The admissible set is inflated from 2x to 20x the true edge set. A reviewer's
first objection to any structure-informed method is that the structure is doing
the work; this experiment measures exactly how much structure is needed.
"""
from __future__ import annotations
import argparse, numpy as np
from spcrl.benchmarks.dataset import make_pool, TRAIN_FAMILIES, TRAIN_MECHS, OOD_FAMILY
from spcrl.causal.train_amortized import train_amortized, score_pool
from spcrl.util import JsonlWriter
from experiments.e1_main import metrics_from_scores

RATIOS = [2.0, 5.0, 10.0, 20.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=250)
    ap.add_argument("--n-val", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out", default="results/e7_coarseness.jsonl")
    a = ap.parse_args()
    W = JsonlWriter(a.out, meta=vars(a))
    print("run", W.run_id, flush=True)
    for seed in a.seeds:
        for r in RATIOS:
            tr = make_pool(a.n_train, 1000 + seed, TRAIN_FAMILIES, TRAIN_MECHS,
                           tag="train", superset_ratio=r)
            va = make_pool(a.n_val, 2000 + seed, TRAIN_FAMILIES, TRAIN_MECHS,
                           tag="val", superset_ratio=r)
            pools = {"iid": make_pool(a.n_test, 9101, TRAIN_FAMILIES, TRAIN_MECHS,
                                      tag="iid", superset_ratio=r),
                     "graph_ood": make_pool(a.n_test, 9103, [OOD_FAMILY], TRAIN_MECHS,
                                            tag="graph_ood", superset_ratio=r)}
            for mode in ["none", "hard", "soft"]:
                model, _ = train_amortized(tr, va, mode=mode, epochs=40, seed=seed)
                for tag, pool in pools.items():
                    for row in score_pool(model, pool):
                        s = [x for x in pool if x.sysinst.graph_id == row["graph_id"]][0]
                        m = metrics_from_scores(row["P"], s.y.astype(np.int8))
                        if m:
                            W.write({"superset_ratio": r, "regime": tag, "seed": seed,
                                     "method": f"spcrl_amortized_{mode}",
                                     "graph_id": row["graph_id"], "d": row["d"],
                                     "adm_over_true": float(np.asarray(s.S).sum() / s.y.sum()),
                                     "rho": float((np.asarray(s.S) > .5).sum() / row["d"] ** 2),
                                     **m})
                # prior-only reference at this coarseness
                if mode == "soft":
                    for tag, pool in pools.items():
                        for s in pool:
                            m = metrics_from_scores(np.asarray(s.S, float), s.y.astype(np.int8))
                            if m:
                                W.write({"superset_ratio": r, "regime": tag, "seed": seed,
                                         "method": "prior_only", "graph_id": s.sysinst.graph_id,
                                         "d": s.sysinst.d,
                                         "adm_over_true": float(np.asarray(s.S).sum() / s.y.sum()),
                                         **m})
            print(f"seed={seed} ratio={r} done", flush=True)
    print("E7 complete", flush=True)


if __name__ == "__main__":
    main()
