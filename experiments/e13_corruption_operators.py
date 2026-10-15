"""E13: which corruption operator causes the hard-mask crossover?

E2 corrupts the prior with deletions and additions at once. That is realistic,
but it confounds two defects of opposite character:

  delete   a true relation is removed from the admissible set. Under a hard mask
           it becomes unrecoverable no matter how strong the evidence is.
  add      a spurious pair is made admissible. The hypothesis space grows; no
           true relation is lost.
  reverse  a true relation is flipped, so the description asserts the wrong
           direction *and* forbids the right one.

If the crossover of Section 2.3 is caused by lost admissibility rather than by
prior error in general, the `delete` and `reverse` operators should reproduce it
and `add` should not. This experiment tests that, holding everything else fixed.

Both the matched protocol (the corrupted prior is present during training) and
the drift protocol (trained clean, deployed corrupted) are run, because the
distinction matters for the interpretation.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np

from spcrl.benchmarks.dataset import (make_pool, TRAIN_FAMILIES, TRAIN_MECHS,
                                      OOD_FAMILY)
from spcrl.causal.train_amortized import train_amortized, score_pool
from spcrl.util import JsonlWriter
from experiments.e1_main import metrics_from_scores

CS = [0.1, 0.2, 0.5]          # positive corruption levels, per operator
OPS = ["delete", "add", "reverse"]
# At c = 0 every operator produces the same prior, so the clean baseline is
# trained once per seed and shared; training it three times would cost three
# times as much and produce the same model.
CLEAN = 0.0


def _done_cells(path):
    done = set()
    if not os.path.exists(path):
        return done
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("protocol") == "matched":
            done.add((r.get("seed"), r.get("op"), r.get("corruption"), r.get("method")))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=250)
    ap.add_argument("--n-val", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--out", default="results/e13_corruption_ops.jsonl")
    args = ap.parse_args()
    W = JsonlWriter(args.out, meta=vars(args), append=True)
    done = _done_cells(args.out)
    print("run", W.run_id, f"({len(done)} cells already complete)", flush=True)

    # Test pools are built once per (operator, corruption) and shared across
    # seeds, so the pairing across methods and seeds is exact.
    pools = {}
    for op in OPS + ["clean"]:
        for c in (CS if op != "clean" else [CLEAN]):
            key = (op, c)
            pools[key] = {
                "iid": make_pool(args.n_test, 5000 + int(c * 100), TRAIN_FAMILIES,
                                 TRAIN_MECHS, tag="iid", T=args.T, corruption=c,
                                 corrupt_op=("both" if op == "clean" else op)),
                "graph_ood": make_pool(args.n_test, 6000 + int(c * 100), [OOD_FAMILY],
                                       TRAIN_MECHS, tag="graph_ood", T=args.T,
                                       corruption=c,
                                       corrupt_op=("both" if op == "clean" else op))}
    print("test pools ready", flush=True)

    def score(model, op, c, mode, seed, protocol, train_sec=None):
        for regime, pool in pools[(op, c)].items():
            for row in score_pool(model, pool):
                s = [x for x in pool if x.sysinst.graph_id == row["graph_id"]][0]
                m = metrics_from_scores(row["P"], s.y.astype(np.int8))
                if m:
                    W.write({"protocol": protocol, "op": op, "corruption": c,
                             "regime": regime, "method": f"spcrl_amortized_{mode}",
                             "seed": seed, "graph_id": row["graph_id"],
                             "d": row["d"], "family": row["family"],
                             "mechanism": row["mechanism"],
                             "train_sec": train_sec, **m})

    def train(op, c, mode, seed):
        pool_op = "both" if op == "clean" else op
        tr = make_pool(args.n_train, 1000 + seed * 17 + int(c * 100),
                       TRAIN_FAMILIES, TRAIN_MECHS, tag="train",
                       T=args.T, corruption=c, corrupt_op=pool_op)
        va = make_pool(args.n_val, 2000 + seed * 17 + int(c * 100),
                       TRAIN_FAMILIES, TRAIN_MECHS, tag="val",
                       T=args.T, corruption=c, corrupt_op=pool_op)
        model, _ = train_amortized(tr, va, mode=mode, epochs=40, seed=seed)
        return model

    drift_done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("protocol") == "drift":
                drift_done.add((r.get("seed"), r.get("method")))

    for seed in args.seeds:
        clean = {}
        for op, c in [("clean", CLEAN)] + [(o, cc) for o in OPS for cc in CS]:
            for mode in ["none", "hard", "soft"]:
                method = f"spcrl_amortized_{mode}"
                if (seed, op, c, method) in done:
                    continue
                t = time.time()
                model = train(op, c, mode, seed)
                if op == "clean":
                    clean[mode] = model
                score(model, op, c, mode, seed, "matched", time.time() - t)
                print(f"seed={seed} {op} c={c} {mode} ({time.time()-t:.0f}s)",
                      flush=True)
        # Drift: the clean-prior model applied to every corrupted test pool.
        # On a resumed run the clean models may not have been retrained above,
        # so retrain only those whose drift rows are still missing.
        for mode in ["none", "hard", "soft"]:
            method = f"spcrl_amortized_{mode}"
            if (seed, method) in drift_done:
                continue
            model = clean.get(mode) or train("clean", CLEAN, mode, seed)
            for op in OPS:
                for c in CS:
                    score(model, op, c, mode, seed, "drift")
            score(model, "clean", CLEAN, mode, seed, "drift")
        print(f"[drift] seed={seed} done", flush=True)
    print("E13 complete", flush=True)


if __name__ == "__main__":
    main()
