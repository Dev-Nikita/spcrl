"""E15: leave-one-graph-family-out.

Section 2.2 reports transfer to one unseen graph family (hierarchical). A single
held-out family cannot distinguish "the prior helps on unseen structure" from
"the prior happens to help on hierarchical graphs". This experiment rotates the
held-out family over all four, training on the other three each time, so the
graph-shift claim rests on four folds rather than one.

Everything else is held fixed: same architecture, same training budget, same
number of test systems, same seeds. Only the family partition rotates.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np

from spcrl.benchmarks.dataset import make_pool, TRAIN_MECHS
from spcrl.causal.train_amortized import train_amortized, score_pool
from spcrl.util import JsonlWriter
from experiments.e1_main import metrics_from_scores

FAMILIES = ["er", "sf", "modular", "hier"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=250)
    ap.add_argument("--n-val", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--out", default="results/e15_leave_one_family.jsonl")
    args = ap.parse_args()
    W = JsonlWriter(args.out, meta=vars(args), append=True)

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                r = json.loads(line)
            except Exception:
                continue
            done.add((r.get("seed"), r.get("held_out"), r.get("method")))
    print("run", W.run_id, f"({len(done)} cells complete)", flush=True)

    # The held-out test pool for a fold contains only the held-out family; the
    # in-distribution pool contains only the three training families, so the two
    # numbers are directly comparable within a fold.
    pools = {}
    for f in FAMILIES:
        rest = [x for x in FAMILIES if x != f]
        pools[f] = {
            "held_out": make_pool(args.n_test, 5100 + FAMILIES.index(f), [f],
                                  TRAIN_MECHS, tag="held_out", T=args.T),
            "in_dist": make_pool(args.n_test, 6100 + FAMILIES.index(f), rest,
                                 TRAIN_MECHS, tag="in_dist", T=args.T)}
    print("test pools ready", flush=True)

    for seed in args.seeds:
        for f in FAMILIES:
            rest = [x for x in FAMILIES if x != f]
            for mode in ["none", "hard", "soft"]:
                method = f"spcrl_amortized_{mode}"
                if (seed, f, method) in done:
                    continue
                t = time.time()
                tr = make_pool(args.n_train, 1000 + seed * 17, rest, TRAIN_MECHS,
                               tag="train", T=args.T)
                va = make_pool(args.n_val, 2000 + seed * 17, rest, TRAIN_MECHS,
                               tag="val", T=args.T)
                model, _ = train_amortized(tr, va, mode=mode, epochs=40, seed=seed)
                for regime, pool in pools[f].items():
                    for row in score_pool(model, pool):
                        s = [x for x in pool if x.sysinst.graph_id == row["graph_id"]][0]
                        m = metrics_from_scores(row["P"], s.y.astype(np.int8))
                        if m:
                            W.write({"held_out": f, "regime": regime,
                                     "method": method, "seed": seed,
                                     "graph_id": row["graph_id"], "d": row["d"],
                                     "family": row["family"],
                                     "mechanism": row["mechanism"],
                                     "train_sec": time.time() - t, **m})
                print(f"seed={seed} held_out={f} {mode} ({time.time()-t:.0f}s)", flush=True)
    print("E15 complete", flush=True)


if __name__ == "__main__":
    main()
