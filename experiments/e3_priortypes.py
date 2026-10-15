"""E3: which kinds of structural knowledge actually help?

P0 none | P1 topology | P2 topology+direction | P3 topology+node type
| P5 full structural prior.
"""
from __future__ import annotations
import argparse, time, numpy as np
from spcrl.benchmarks.dataset import make_pool, TRAIN_FAMILIES, TRAIN_MECHS, OOD_FAMILY
from spcrl.causal.train_amortized import train_amortized, score_pool
from spcrl.util import JsonlWriter
from experiments.e1_main import metrics_from_scores

KINDS = ["none", "topology", "topology_type", "topology_dir", "full"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-val", type=int, default=50)
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--out", default="results/e3_priortypes.jsonl")
    args = ap.parse_args()
    W = JsonlWriter(args.out, meta=vars(args))
    print("run", W.run_id, flush=True)

    for seed in args.seeds:
        for kind in KINDS:
            tr = make_pool(args.n_train, 1000 + seed, TRAIN_FAMILIES, TRAIN_MECHS,
                           tag="train", T=args.T, prior_kind=kind)
            va = make_pool(args.n_val, 2000 + seed, TRAIN_FAMILIES, TRAIN_MECHS,
                           tag="val", T=args.T, prior_kind=kind)
            pools = {
                "iid": make_pool(args.n_test, 8001, TRAIN_FAMILIES, TRAIN_MECHS,
                                 tag="iid", T=args.T, prior_kind=kind),
                "graph_ood": make_pool(args.n_test, 8003, [OOD_FAMILY], TRAIN_MECHS,
                                       tag="graph_ood", T=args.T, prior_kind=kind)}
            mode = "none" if kind == "none" else "soft"
            t = time.time()
            model, inf = train_amortized(tr, va, mode=mode, epochs=40, seed=seed)
            for tag, pool in pools.items():
                for row in score_pool(model, pool):
                    s = [x for x in pool if x.sysinst.graph_id == row["graph_id"]][0]
                    m = metrics_from_scores(row["P"], s.y.astype(np.int8))
                    if m:
                        W.write({"prior_kind": kind, "mode": mode, "regime": tag,
                                 "method": f"spcrl_amortized_{mode}", "seed": seed,
                                 "graph_id": row["graph_id"], "d": row["d"],
                                 "family": row["family"], "mechanism": row["mechanism"],
                                 "train_sec": time.time() - t, **m})
            print(f"seed={seed} kind={kind} done ({time.time()-t:.0f}s)", flush=True)
    print("E3 complete", flush=True)


if __name__ == "__main__":
    main()
