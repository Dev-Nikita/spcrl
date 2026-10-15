"""E5: computational cost and the search-space reduction implied by the prior."""
from __future__ import annotations
import argparse, time, numpy as np
from spcrl.benchmarks.dataset import make_pool, TRAIN_FAMILIES, TRAIN_MECHS
from spcrl.causal.train_amortized import train_amortized
from spcrl.causal.amortized import predict_adjacency
from spcrl.util import JsonlWriter
from experiments.e1_main import metrics_from_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/e5_scaling.jsonl")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    W = JsonlWriter(args.out, meta=vars(args))
    tr = make_pool(300, 1000, TRAIN_FAMILIES, TRAIN_MECHS, tag="train")
    va = make_pool(50, 2000, TRAIN_FAMILIES, TRAIN_MECHS, tag="val")
    models = {}
    for seed in args.seeds:
        for mode in ["none", "hard", "soft"]:
            models[(seed, mode)], _ = train_amortized(tr, va, mode=mode, epochs=40, seed=seed)
    for d in [20, 50, 100, 200]:
        pool = make_pool(6, 9000 + d, TRAIN_FAMILIES, TRAIN_MECHS,
                         d_range=(d, d), tag=f"d{d}")
        for s in pool:
            rho = float((np.asarray(s.S) > 0.5).sum() / (s.sysinst.d ** 2))
            for (seed, mode), mdl in models.items():
                t0 = time.time()
                P = predict_adjacency(mdl, s.Fp, s.Fn, s.S)
                infer = time.time() - t0
                m = metrics_from_scores(P, s.y.astype(np.int8))
                if m:
                    W.write({"d": d, "seed": seed, "method": f"spcrl_amortized_{mode}",
                             "rho": rho, "candidate_edges_full": s.sysinst.d ** 2,
                             "candidate_edges_prior": int((np.asarray(s.S) > 0.5).sum()),
                             "infer_sec": infer, "graph_id": s.sysinst.graph_id, **m})
        print("d", d, "done", flush=True)
    print("E5 complete", flush=True)


if __name__ == "__main__":
    main()
