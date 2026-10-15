"""E6: does robustness to a wrong prior survive all the way to the repair decision?

E2 shows that soft priors keep recovering the graph when the structural
description is wrong. E6 asks the question that matters operationally: does that
translate into better root-cause identification and lower intervention regret.
"""
from __future__ import annotations
import argparse, json, os, time, numpy as np

from spcrl.benchmarks.dataset import make_pool, TRAIN_FAMILIES, TRAIN_MECHS, OOD_FAMILY
from spcrl.causal.train_amortized import train_amortized
from spcrl.causal.amortized import predict_adjacency
from spcrl.util import JsonlWriter
import importlib
_e4 = importlib.import_module("experiments.e4_" + "downstream")

CS = [0.0, 0.2, 0.5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=250)
    ap.add_argument("--n-val", type=int, default=40)
    ap.add_argument("--n-sys", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--out", default="results/e6_corrupt_downstream.jsonl")
    args = ap.parse_args()
    W = JsonlWriter(args.out, meta=vars(args), append=True)
    print("run", W.run_id, flush=True)

    # Resumability. The container this runs in is restarted from time to time
    # and a full pass costs hours, so completed (seed, corruption, system) cells
    # are skipped rather than recomputed. A cell counts as done when the oracle
    # reference for that system has been written, which happens in the same
    # run_system call as every other method's rows.
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("method") == "oracle":
                done.add((r.get("seed"), r.get("corruption"), r.get("graph_id")))
    if done:
        print(f"resuming: {len(done)} cells already complete", flush=True)

    class Tagged:
        """Adds the corruption level to every row written by the shared pipeline."""
        def __init__(self, w, extra): self.w, self.extra = w, extra
        def write(self, row): self.w.write({**row, **self.extra})

    for seed in args.seeds:
        for c in CS:
            pools = {
                "iid": make_pool(args.n_sys, 7001, TRAIN_FAMILIES, TRAIN_MECHS,
                                 tag="iid", T=args.T, corruption=c),
                "graph_ood": make_pool(args.n_sys, 7003, [OOD_FAMILY], TRAIN_MECHS,
                                       tag="graph_ood", T=args.T, corruption=c)}
            if all((seed, c, sy.sysinst.graph_id) in done
                   for pool in pools.values() for sy in pool):
                print(f"seed={seed} c={c}: all systems done, skipping training",
                      flush=True)
                continue
            tr = make_pool(args.n_train, 1000 + seed, TRAIN_FAMILIES, TRAIN_MECHS,
                           tag="train", T=args.T, corruption=c)
            va = make_pool(args.n_val, 2000 + seed, TRAIN_FAMILIES, TRAIN_MECHS,
                           tag="val", T=args.T, corruption=c)
            models = {}
            for mode in ["none", "hard", "soft"]:
                models[mode], _ = train_amortized(tr, va, mode=mode, epochs=40, seed=seed)
            print(f"seed={seed} c={c}: models trained", flush=True)
            for regime, pool in pools.items():
                for s in pool:
                    if (seed, c, s.sysinst.graph_id) in done:
                        print(f"  [{regime}] c={c} {s.sysinst.graph_id} already done",
                              flush=True)
                        continue
                    graphs = {"oracle": s.y}
                    for mode, mdl in models.items():
                        graphs[f"spcrl_{mode}"] = predict_adjacency(mdl, s.Fp, s.Fn, s.S)
                    graphs["prior_only"] = (np.asarray(s.S, float)
                                            + 1e-6 * np.random.default_rng(0).random(s.S.shape))
                    t = time.time()
                    _e4.run_system(s, graphs, np.random.default_rng(
                        hash((seed, c, s.sysinst.graph_id)) % (2 ** 31)),
                        Tagged(W, {"corruption": c}), regime, seed, n_faults=5)
                    print(f"  [{regime}] c={c} {s.sysinst.graph_id} {time.time()-t:.0f}s", flush=True)
    print("E6 complete", flush=True)


if __name__ == "__main__":
    main()
