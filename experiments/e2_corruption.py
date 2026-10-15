"""E2: how much structural-prior corruption can the model tolerate?

Protocol A (matched): the corrupted prior is what the model sees at train and
test time.  Protocol B (drift): the model is trained on a clean prior and the
prior degrades after deployment.
"""
from __future__ import annotations
import argparse, time, numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from spcrl.benchmarks.dataset import (make_pool, TRAIN_FAMILIES, TRAIN_MECHS,
                                      OOD_FAMILY)
from spcrl.causal.train_amortized import train_amortized, score_pool
from spcrl.eval.metrics import shd, edge_prf, orientation_f1
from spcrl.util import JsonlWriter
from experiments.e1_main import metrics_from_scores

CS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-val", type=int, default=50)
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--out", default="results/e2_corruption.jsonl")
    args = ap.parse_args()
    W = JsonlWriter(args.out, meta=vars(args))
    print("run", W.run_id, flush=True)

    test_pools = {}
    for c in CS:
        test_pools[("iid", c)] = make_pool(args.n_test, 5000 + int(c * 100),
                                           TRAIN_FAMILIES, TRAIN_MECHS,
                                           tag="iid", T=args.T, corruption=c)
        test_pools[("graph_ood", c)] = make_pool(args.n_test, 6000 + int(c * 100),
                                                 [OOD_FAMILY], TRAIN_MECHS,
                                                 tag="graph_ood", T=args.T, corruption=c)
    print("test pools ready", flush=True)

    for seed in args.seeds:
        clean_models = {}
        for c in CS:
            tr = make_pool(args.n_train, 1000 + seed * 17 + int(c * 100), TRAIN_FAMILIES,
                           TRAIN_MECHS, tag="train", T=args.T, corruption=c)
            va = make_pool(args.n_val, 2000 + seed * 17 + int(c * 100), TRAIN_FAMILIES,
                           TRAIN_MECHS, tag="val", T=args.T, corruption=c)
            for mode in ["none", "hard", "soft"]:
                t = time.time()
                model, inf = train_amortized(tr, va, mode=mode, epochs=40, seed=seed)
                if c == 0.0:
                    clean_models[mode] = model
                for (tag, ct), pool in test_pools.items():
                    if ct != c:
                        continue
                    for row in score_pool(model, pool):
                        s = [x for x in pool if x.sysinst.graph_id == row["graph_id"]][0]
                        m = metrics_from_scores(row["P"], s.y.astype(np.int8))
                        if m:
                            W.write({"protocol": "matched", "corruption": c, "regime": tag,
                                     "method": f"spcrl_amortized_{mode}", "seed": seed,
                                     "graph_id": row["graph_id"], "d": row["d"],
                                     "family": row["family"], "mechanism": row["mechanism"],
                                     "train_sec": time.time() - t, **m})
                # prior-only reference at this corruption level
                if mode == "soft":
                    for (tag, ct), pool in test_pools.items():
                        if ct != c:
                            continue
                        for s in pool:
                            m = metrics_from_scores(np.asarray(s.S, float), s.y.astype(np.int8))
                            if m:
                                W.write({"protocol": "matched", "corruption": c, "regime": tag,
                                         "method": "prior_only", "seed": seed,
                                         "graph_id": s.sysinst.graph_id, "d": s.sysinst.d,
                                         "family": s.sysinst.family,
                                         "mechanism": s.sysinst.mechanism, **m})
            print(f"[matched] seed={seed} c={c} done", flush=True)

        # Protocol B: trained on clean prior, deployed with degraded prior
        for mode, model in clean_models.items():
            for (tag, c), pool in test_pools.items():
                for row in score_pool(model, pool):
                    s = [x for x in pool if x.sysinst.graph_id == row["graph_id"]][0]
                    m = metrics_from_scores(row["P"], s.y.astype(np.int8))
                    if m:
                        W.write({"protocol": "drift", "corruption": c, "regime": tag,
                                 "method": f"spcrl_amortized_{mode}", "seed": seed,
                                 "graph_id": row["graph_id"], "d": row["d"],
                                 "family": row["family"], "mechanism": row["mechanism"], **m})
        print(f"[drift] seed={seed} done", flush=True)
    print("E2 complete", flush=True)


if __name__ == "__main__":
    main()
