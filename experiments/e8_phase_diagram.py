"""E8: the applicability envelope of an imperfect structural prior.

Two axes at once: how *wrong* the prior is (corruption c) and how *coarse* it
is (admissible pairs per true edge, r). The reported quantity is the gain a
structural prior buys over an identically trained prior-blind model,

    Gain(c, r) = AUPRC_soft(c, r) - AUPRC_none(c, r),

which maps out where structural knowledge is useful, neutral, or harmful.
"""
from __future__ import annotations
import argparse, json, os, time, numpy as np

from spcrl.benchmarks.dataset import make_pool, TRAIN_FAMILIES, TRAIN_MECHS, OOD_FAMILY
from spcrl.causal.train_amortized import train_amortized, score_pool
from spcrl.util import JsonlWriter
from experiments.e1_main import metrics_from_scores

CS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
RS = [2.0, 5.0, 10.0, 20.0]


def done_cells(path):
    """(seed, superset_ratio, corruption, method) combinations already written.

    The container this runs in can be recycled at any time, so every experiment
    is restartable: completed cells are read back from the JSONL and skipped.
    """
    out = set()
    if not os.path.exists(path):
        return out
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        out.add((r.get("seed"), r.get("superset_ratio"), r.get("corruption"),
                 r.get("method")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=250)
    ap.add_argument("--n-val", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--variants", nargs="+", default=["none", "hard", "soft"])
    ap.add_argument("--out", default="results/e8_phase.jsonl")
    a = ap.parse_args()
    W = JsonlWriter(a.out, meta=vars(a), append=True)
    done = done_cells(a.out)
    if done:
        print(f"resuming: {len(done)} cells already complete", flush=True)
    print("run", W.run_id, flush=True)
    test_cache = {}
    for seed in a.seeds:
        for r in RS:
            for c in CS:
                todo = [v for v in a.variants
                        if (seed, r, c, f"spcrl_amortized_{v}") not in done]
                if not todo:
                    print(f"seed={seed} r={r} c={c} already done", flush=True)
                    continue
                t0 = time.time()
                tr = make_pool(a.n_train, 1000 + seed, TRAIN_FAMILIES, TRAIN_MECHS,
                               tag="train", corruption=c, superset_ratio=r)
                va = make_pool(a.n_val, 2000 + seed, TRAIN_FAMILIES, TRAIN_MECHS,
                               tag="val", corruption=c, superset_ratio=r)
                if (r, c) not in test_cache:
                    test_cache[(r, c)] = {
                        "iid": make_pool(a.n_test, 9201, TRAIN_FAMILIES, TRAIN_MECHS,
                                         tag="iid", corruption=c, superset_ratio=r),
                        "graph_ood": make_pool(a.n_test, 9203, [OOD_FAMILY], TRAIN_MECHS,
                                               tag="graph_ood", corruption=c,
                                               superset_ratio=r)}
                pools = test_cache[(r, c)]
                for v in todo:
                    model, _ = train_amortized(tr, va, variant=v, epochs=40, seed=seed)
                    for tag, pool in pools.items():
                        for row in score_pool(model, pool):
                            s = [x for x in pool if x.sysinst.graph_id == row["graph_id"]][0]
                            m = metrics_from_scores(row["P"], s.y.astype(np.int8))
                            if m:
                                W.write({"corruption": c, "superset_ratio": r,
                                         "adm_over_true": float(np.asarray(s.S).sum() / s.y.sum()),
                                         "regime": tag, "seed": seed,
                                         "method": f"spcrl_amortized_{v}",
                                         "graph_id": row["graph_id"], "d": row["d"],
                                         "family": row["family"],
                                         "mechanism": row["mechanism"], **m})
                print(f"seed={seed} r={r} c={c} done ({time.time()-t0:.0f}s)", flush=True)
    print("E8 complete", flush=True)


if __name__ == "__main__":
    main()
