"""E12: does global graph accuracy predict decision quality? (hypothesis H6)

The corruption effect that dominates causal-structure recovery does not reach
the repair decision (E6). The hypothesis for why is that the residual structural
errors that distinguish the methods lie *outside* the causal paths that connect
the injected fault to the key performance variable.

This experiment tests that directly. For every (system, fault) it computes

    SHD_global    - structural Hamming distance over the whole graph,
    SHD_decision  - the same distance restricted to the decision-relevant
                    subgraph, i.e. the nodes that are simultaneously
                    descendants of the fault and ancestors of the KPI,

and the root-cause ranking error of the identical downstream pipeline. If
SHD_decision correlates with the ranking error substantially better than
SHD_global, then global reconstruction accuracy is a poor surrogate for
decision quality - a statement about evaluation practice, not about our method.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np

from spcrl.benchmarks.dataset import (make_pool, TRAIN_FAMILIES, TRAIN_MECHS,
                                      OOD_FAMILY, OOD_MECHS)
from spcrl.benchmarks.synthetic import simulate, topological_order
from spcrl.causal.train_amortized import train_amortized
from spcrl.causal.amortized import predict_adjacency
from spcrl.counterfactual.surrogate import (SurrogateSCM, top_k_dag,
                                            rca_from_surrogate, anomaly_scores)
from spcrl.eval.metrics import shd, topk_and_mrr
from spcrl.util import JsonlWriter

FAULTS = [("offset", 1.5), ("gain", 0.5), ("noise", 2.0)]
VARIANTS = ["none", "hard", "soft"]


def ancestors(A):
    d = A.shape[0]
    anc = {i: set() for i in range(d)}
    for i in topological_order(A):
        for j in np.where(A[:, i] == 1)[0]:
            anc[i] |= {int(j)} | anc[int(j)]
    return anc


def descendants(A):
    return ancestors(A.T)


def shd_on(B, T, nodes):
    """SHD restricted to the subgraph induced by `nodes`."""
    idx = np.array(sorted(nodes), dtype=int)
    if len(idx) < 2:
        return None
    return int(shd(np.asarray(B)[np.ix_(idx, idx)], np.asarray(T)[np.ix_(idx, idx)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=250)
    ap.add_argument("--n-val", type=int, default=40)
    ap.add_argument("--n-sys", type=int, default=15)
    ap.add_argument("--n-faults", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="results/e12_decision_relevant.jsonl")
    a = ap.parse_args()
    W = JsonlWriter(a.out, meta=vars(a), append=True)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line)
            except Exception:
                continue
            done.add((r.get("seed"), r.get("regime"), r.get("graph_id")))
    print("run", W.run_id, f"({len(done)} system-cells done)", flush=True)

    pools = {
        "iid": make_pool(a.n_sys, 7101, TRAIN_FAMILIES, TRAIN_MECHS, tag="iid"),
        "graph_ood": make_pool(a.n_sys, 7103, [OOD_FAMILY], TRAIN_MECHS, tag="graph_ood"),
        "mech_ood": make_pool(a.n_sys, 7102, TRAIN_FAMILIES, OOD_MECHS, tag="mech_ood"),
    }

    for seed in a.seeds:
        tr = make_pool(a.n_train, 1000 + seed, TRAIN_FAMILIES, TRAIN_MECHS, tag="train")
        va = make_pool(a.n_val, 2000 + seed, TRAIN_FAMILIES, TRAIN_MECHS, tag="val")
        models = {}
        for v in VARIANTS:
            models[v], _ = train_amortized(tr, va, variant=v, epochs=40, seed=seed)
            print(f"  seed {seed}: {v} trained", flush=True)

        for regime, pool in pools.items():
            for s in pool:
                if (seed, regime, s.sysinst.graph_id) in done:
                    continue
                t0 = time.time()
                A = s.sysinst.A_true.astype(np.int8)
                anc, desc = ancestors(A), descendants(A)
                kpi = int(max(range(s.sysinst.d), key=lambda i: len(anc[i])))
                kpi_anc = sorted(anc[kpi])
                if len(kpi_anc) < 3:
                    continue
                k_true = int(A.sum())
                rng = np.random.default_rng(abs(hash((seed, s.sysinst.graph_id))) % (2**31))
                X_ref = simulate(s.sysinst.scm, 300,
                                 np.random.default_rng(int(rng.integers(1e9))))

                graphs = {"reference": A}
                for v in VARIANTS:
                    graphs[f"spcrl_{v}"] = top_k_dag(
                        predict_adjacency(models[v], s.Fp, s.Fn, s.S), k_true)
                sur, ref = {}, {}
                for name, B in graphs.items():
                    m = SurrogateSCM(np.asarray(B, np.int8)).fit(s.X)
                    sur[name] = m
                    ref[name] = m.residuals(X_ref)

                nodes = list(rng.choice(kpi_anc, size=min(a.n_faults, len(kpi_anc)),
                                        replace=False))
                for v in nodes:
                    v = int(v)
                    ft, mag = FAULTS[int(rng.integers(len(FAULTS)))]
                    Xf = simulate(s.sysinst.scm, 300,
                                  np.random.default_rng(int(rng.integers(1e9))),
                                  fault_node=v, fault_t=0, fault_mag=mag,
                                  fault_type=ft)
                    # decision-relevant subgraph: between the fault and the KPI
                    rel = (desc[v] & anc[kpi]) | {v, kpi}
                    base = {"seed": seed, "regime": regime,
                            "graph_id": s.sysinst.graph_id, "d": s.sysinst.d,
                            "family": s.sysinst.family,
                            "mechanism": s.sysinst.mechanism,
                            "fault_node": v, "fault_type": ft, "kpi": kpi,
                            "n_relevant": len(rel), "n_edges": k_true}
                    sc = anomaly_scores(Xf, X_ref)
                    W.write({**base, "method": "anomaly_no_structure",
                             **topk_and_mrr(sc, v)})
                    for name, m in sur.items():
                        r = topk_and_mrr(rca_from_surrogate(m, Xf, ref[name]), v)
                        B = graphs[name]
                        W.write({**base, "method": name, **r,
                                 "shd_global": int(shd(np.asarray(B), A)),
                                 "shd_decision": shd_on(B, A, rel)})
                print(f"  [{regime}] {s.sysinst.graph_id} d={s.sysinst.d} "
                      f"{time.time()-t0:.0f}s", flush=True)
    print("E12 complete", flush=True)


if __name__ == "__main__":
    main()
