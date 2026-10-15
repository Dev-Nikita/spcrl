"""E9: is the robustness the architecture, or just training on corrupted priors?

Two referee objections, answered directly:
  A. a plain network that merely *sees* prior features, trained under the same
     corruption, might be just as robust  -> variant `soft_noenc`;
  B. a single learned reliability scalar gamma scaling the whole prior might be
     enough, making the prior-biased attention unnecessary -> `soft_gate`.
"""
from __future__ import annotations
import argparse, json, os, time, numpy as np

from spcrl.benchmarks.dataset import make_pool, TRAIN_FAMILIES, TRAIN_MECHS, OOD_FAMILY
from spcrl.causal.train_amortized import train_amortized, score_pool
from spcrl.util import JsonlWriter
from experiments.e1_main import metrics_from_scores

CS = [0.0, 0.10, 0.30, 0.50]
VARIANTS = ["none", "hard", "soft", "soft_noenc", "soft_gate"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-val", type=int, default=50)
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results/e9_architecture.jsonl")
    a = ap.parse_args()
    W = JsonlWriter(a.out, meta=vars(a), append=True)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line)
            except Exception:
                continue
            done.add((r.get("seed"), r.get("corruption"), r.get("method")))
    if done:
        print(f"resuming: {len(done)} cells already complete", flush=True)
    print("run", W.run_id, flush=True)
    for seed in a.seeds:
        for c in CS:
            todo = [v for v in VARIANTS
                    if (seed, c, f"spcrl_amortized_{v}") not in done]
            if not todo:
                print(f"seed={seed} c={c} already done", flush=True)
                continue
            tr = make_pool(a.n_train, 1000 + seed, TRAIN_FAMILIES, TRAIN_MECHS,
                           tag="train", corruption=c)
            va = make_pool(a.n_val, 2000 + seed, TRAIN_FAMILIES, TRAIN_MECHS,
                           tag="val", corruption=c)
            pools = {"iid": make_pool(a.n_test, 9301, TRAIN_FAMILIES, TRAIN_MECHS,
                                      tag="iid", corruption=c),
                     "graph_ood": make_pool(a.n_test, 9303, [OOD_FAMILY], TRAIN_MECHS,
                                            tag="graph_ood", corruption=c)}
            for v in todo:
                t = time.time()
                model, inf = train_amortized(tr, va, variant=v, epochs=40, seed=seed)
                n_par = sum(p.numel() for p in model.parameters())
                for tag, pool in pools.items():
                    for row in score_pool(model, pool):
                        s = [x for x in pool if x.sysinst.graph_id == row["graph_id"]][0]
                        m = metrics_from_scores(row["P"], s.y.astype(np.int8))
                        if m:
                            W.write({"corruption": c, "regime": tag, "seed": seed,
                                     "method": f"spcrl_amortized_{v}",
                                     "n_params": n_par, "train_sec": time.time() - t,
                                     "graph_id": row["graph_id"], "d": row["d"],
                                     "family": row["family"],
                                     "mechanism": row["mechanism"], **m})
                print(f"seed={seed} c={c} {v} done", flush=True)
    print("E9 complete", flush=True)


if __name__ == "__main__":
    main()
