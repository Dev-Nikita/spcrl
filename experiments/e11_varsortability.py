"""E11: is the benchmark leaking the causal order through marginal variances?

Reisach et al. (NeurIPS 2021) show that simulated additive-noise benchmarks are
often solvable by sorting variables by variance and running a sparse regression
(`sortnregress`), which says nothing about causal discovery. This experiment

  * measures varsortability and R^2-sortability of our generator,
  * runs sortnregress as a baseline in its own right,
  * and re-runs the whole soft/hard/none comparison on per-variable
    standardised data, where varsortability is 0.5 by construction.

If the ordering of the three constraint modes survives standardisation, the
paper's claims do not rest on the artefact.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import average_precision_score

from spcrl.benchmarks.dataset import (make_pool, TRAIN_FAMILIES, TRAIN_MECHS,
                                      OOD_FAMILY, OOD_MECHS)
from spcrl.baselines.varsort import varsortability, sortnregress, _sort_stat
from spcrl.causal.train_amortized import train_amortized, score_pool
from spcrl.util import JsonlWriter
from experiments.e1_main import metrics_from_scores

VARIANTS = ["none", "hard", "soft"]


def sortability(s):
    X, A = s.X, s.y
    d = X.shape[1]
    r2 = np.array([LinearRegression().fit(np.delete(X, i, 1), X[:, i])
                   .score(np.delete(X, i, 1), X[:, i]) for i in range(d)])
    return float(varsortability(X, A)), float(_sort_stat(r2, A))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=250)
    ap.add_argument("--n-val", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results/e11_varsortability.jsonl")
    a = ap.parse_args()
    W = JsonlWriter(a.out, meta=vars(a), append=True)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line)
            except Exception:
                continue
            done.add((r.get("seed"), r.get("standardized"), r.get("regime"), r.get("method")))
    print("run", W.run_id, f"({len(done)} cells done)", flush=True)

    for std in [False, True]:
        pools = {"iid": make_pool(a.n_test, 3001, TRAIN_FAMILIES, TRAIN_MECHS,
                                  tag="iid", standardize=std),
                 "graph_ood": make_pool(a.n_test, 3003, [OOD_FAMILY], TRAIN_MECHS,
                                        tag="graph_ood", standardize=std),
                 "mech_ood": make_pool(a.n_test, 3002, TRAIN_FAMILIES, OOD_MECHS,
                                       tag="mech_ood", standardize=std)}
        # sortability diagnostics and the sortnregress baseline (seed-free)
        if (-1, std, "iid", "sortnregress") not in done:
            for tag, pool in pools.items():
                for s in pool:
                    vs, r2s = sortability(s)
                    Wsr = sortnregress(s.X)
                    m = metrics_from_scores(np.abs(Wsr), s.y.astype(np.int8))
                    row = {"standardized": std, "regime": tag, "seed": -1,
                           "graph_id": s.sysinst.graph_id, "d": s.sysinst.d,
                           "family": s.sysinst.family, "mechanism": s.sysinst.mechanism,
                           "varsortability": vs, "r2_sortability": r2s}
                    if m:
                        W.write({**row, "method": "sortnregress", **m})
                    W.write({**row, "method": "prior_only",
                             **(metrics_from_scores(np.asarray(s.S, float),
                                                    s.y.astype(np.int8)) or {})})
            print(f"std={std}: sortability + sortnregress done", flush=True)

        for seed in a.seeds:
            todo = [v for v in VARIANTS
                    if (seed, std, "iid", f"spcrl_amortized_{v}") not in done]
            if not todo:
                continue
            tr = make_pool(a.n_train, 1000 + seed, TRAIN_FAMILIES, TRAIN_MECHS,
                           tag="train", standardize=std)
            va = make_pool(a.n_val, 2000 + seed, TRAIN_FAMILIES, TRAIN_MECHS,
                           tag="val", standardize=std)
            for v in todo:
                t = time.time()
                model, _ = train_amortized(tr, va, variant=v, epochs=40, seed=seed)
                for tag, pool in pools.items():
                    for row in score_pool(model, pool):
                        s = [x for x in pool if x.sysinst.graph_id == row["graph_id"]][0]
                        m = metrics_from_scores(row["P"], s.y.astype(np.int8))
                        if m:
                            W.write({"standardized": std, "regime": tag, "seed": seed,
                                     "method": f"spcrl_amortized_{v}",
                                     "graph_id": row["graph_id"], "d": row["d"],
                                     "family": row["family"],
                                     "mechanism": row["mechanism"],
                                     "train_sec": time.time() - t, **m})
                print(f"std={std} seed={seed} {v} done", flush=True)
    print("E11 complete", flush=True)


if __name__ == "__main__":
    main()
