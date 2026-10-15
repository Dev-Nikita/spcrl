"""E4: does a better causal graph give better diagnosis and better repair?

For every method we take the *estimated* graph, plug it into one common
downstream pipeline (surrogate SCM -> conditional-deviation root-cause score ->
counterfactual repair selection) and compare against the true-graph oracle.
Intervention outcomes are verified by re-simulating the ground-truth SCM.
"""
from __future__ import annotations
import argparse, time, numpy as np

from spcrl.benchmarks.dataset import (make_pool, TRAIN_FAMILIES, TRAIN_MECHS,
                                      OOD_FAMILY, OOD_MECHS)
from spcrl.benchmarks.synthetic import simulate
from spcrl.causal.train_amortized import train_amortized
from spcrl.causal.amortized import predict_adjacency
from spcrl.baselines.classical import partial_corr_scores, notears_linear
from spcrl.counterfactual.surrogate import (SurrogateSCM, top_k_dag,
                                            rca_from_surrogate, anomaly_scores,
                                            signed_offsets)
from spcrl.eval.metrics import topk_and_mrr, shd
from spcrl.util import JsonlWriter

FAULTS = [("offset", 1.5), ("gain", 0.5), ("noise", 2.0)]


def _ancestors(A: np.ndarray) -> dict:
    """ancestors[i] = set of nodes with a directed path to i."""
    d = A.shape[0]
    anc = {i: set() for i in range(d)}
    from spcrl.benchmarks.synthetic import topological_order
    for i in topological_order(A):
        for j in np.where(A[:, i] == 1)[0]:
            anc[i] |= {int(j)} | anc[int(j)]
    return anc


def run_system(s, graphs: dict, rng, W, regime, seed, n_faults=6,
               T_fault=300, n_cand=8, n_true_reps=3):
    X_norm = s.X
    X_ref = simulate(s.sysinst.scm, 300, np.random.default_rng(int(rng.integers(1e9))))
    nominal = X_ref.mean(0)
    # The KPI is the variable with the largest ancestor set, and faults are
    # injected only at its ancestors.  Otherwise most repair actions are
    # causally irrelevant to the KPI and the intervention task degenerates into
    # ranking noise -- a subtlety that is easy to miss and that we check for by
    # recording the spread of the true losses for every case.
    anc = _ancestors(s.sysinst.A_true)
    kpi = int(max(range(s.sysinst.d), key=lambda i: len(anc[i])))
    kpi_anc = sorted(anc[kpi])
    if len(kpi_anc) < 3:
        return
    k_true = int(s.y.sum())

    sur, ref_res = {}, {}
    for name, P in graphs.items():
        B = (np.asarray(P) != 0).astype(np.int8) if name == "oracle" else top_k_dag(P, k_true)
        m = SurrogateSCM(B).fit(X_norm)
        sur[name] = (m, B)
        ref_res[name] = m.residuals(X_ref)

    nodes = list(rng.choice(kpi_anc, size=min(n_faults, len(kpi_anc)), replace=False))
    for v in nodes:
        ft, mag = FAULTS[int(rng.integers(len(FAULTS)))]
        Xf = simulate(s.sysinst.scm, T_fault,
                      np.random.default_rng(int(rng.integers(1e9))),
                      fault_node=int(v), fault_t=0, fault_mag=mag, fault_type=ft)
        # candidate repair actions: KPI ancestors in the admissible graph + random
        # candidate repair actions: the true root cause plus random components.
        # The KPI itself is never a candidate -- pinning the KPI to its nominal
        # value would trivially zero the loss without repairing anything.
        # candidates: the true cause, other ancestors of the KPI (causally
        # relevant distractors) and non-ancestors (irrelevant distractors).
        rel = [i for i in kpi_anc if i != int(v)]
        irr = [i for i in range(s.sysinst.d) if i != kpi and i != int(v) and i not in anc[kpi]]
        n_rel = min(len(rel), max(2, (n_cand - 1) // 2))
        n_irr = min(len(irr), n_cand - 1 - n_rel)
        cand = ([int(v)]
                + [int(x) for x in rng.choice(rel, size=n_rel, replace=False)]
                + [int(x) for x in rng.choice(irr, size=n_irr, replace=False)])
        true_loss = {}
        for a in cand:
            vals = [abs(simulate(s.sysinst.scm, 150,
                                 np.random.default_rng(int(rng.integers(1e9))),
                                 fault_node=int(v), fault_t=0, fault_mag=mag,
                                 fault_type=ft, clamp={int(a): float(nominal[a])}
                                 )[:, kpi].mean() - nominal[kpi]) for _ in range(n_true_reps)]
            true_loss[int(a)] = float(np.mean(vals))
        best, worst = min(true_loss.values()), max(true_loss.values())
        norm = (worst - best) if worst > best else 1.0

        base = {"regime": regime, "graph_id": s.sysinst.graph_id, "d": s.sysinst.d,
                "family": s.sysinst.family, "mechanism": s.sysinst.mechanism,
                "seed": seed, "fault_node": int(v), "fault_type": ft,
                "fault_mag": mag, "kpi": kpi,
                "n_kpi_ancestors": len(kpi_anc), "n_candidates": len(cand)}
        # structure-free anomaly baseline
        sc = anomaly_scores(Xf, X_ref)
        W.write({**base, "method": "anomaly_no_structure", **topk_and_mrr(sc, int(v)),
                 "regret": None, "norm_regret": None})
        for name, (m, B) in sur.items():
            sc = rca_from_surrogate(m, Xf, ref_res[name])
            r = topk_and_mrr(sc, int(v))
            # represent the *faulty* system before asking what a repair would do
            delta = signed_offsets(m, Xf)
            pred_loss = {}
            for a in cand:
                out = m.rollout(Xf[-1], steps=30, warmup=15,
                                clamp={int(a): float(nominal[a])}, offsets=delta)
                pred_loss[int(a)] = float(abs(out[kpi] - nominal[kpi]))
            # Ties must be broken at random.  Deciding them by candidate order
            # silently rewards a model whose predictions carry no information,
            # because the true root cause is the first candidate.
            tie_rng = np.random.default_rng(abs(hash((s.sysinst.graph_id, int(v), name))) % (2**31))
            order_ = list(pred_loss); tie_rng.shuffle(order_)
            a_sel = min(order_, key=lambda a: (pred_loss[a], tie_rng.random()))
            n_tied = int(sum(1 for a in pred_loss
                             if abs(pred_loss[a] - pred_loss[a_sel]) < 1e-9))
            regret = true_loss[a_sel] - best
            W.write({**base, "method": name, **r, "selected": int(a_sel),
                     "regret": float(regret), "norm_regret": float(regret / norm),
                     "picked_true_cause": bool(a_sel == int(v)),
                     "n_tied": n_tied,
                     "loss_spread": float(worst - best),
                     "shd": int(shd(B, s.y.astype(np.int8)))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-val", type=int, default=50)
    ap.add_argument("--n-sys", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--out", default="results/e4_downstream.jsonl")
    args = ap.parse_args()
    W = JsonlWriter(args.out, meta=vars(args))
    print("run", W.run_id, flush=True)

    pools = {
        "iid": make_pool(args.n_sys, 7001, TRAIN_FAMILIES, TRAIN_MECHS, tag="iid", T=args.T),
        "graph_ood": make_pool(args.n_sys, 7003, [OOD_FAMILY], TRAIN_MECHS,
                               tag="graph_ood", T=args.T),
        "mech_ood": make_pool(args.n_sys, 7002, TRAIN_FAMILIES, OOD_MECHS,
                              tag="mech_ood", T=args.T),
    }
    for seed in args.seeds:
        tr = make_pool(args.n_train, 1000 + seed, TRAIN_FAMILIES, TRAIN_MECHS, tag="train", T=args.T)
        va = make_pool(args.n_val, 2000 + seed, TRAIN_FAMILIES, TRAIN_MECHS, tag="val", T=args.T)
        models = {}
        for mode in ["none", "hard", "soft"]:
            models[mode], _ = train_amortized(tr, va, mode=mode, epochs=40, seed=seed)
        print(f"seed {seed}: amortized models trained", flush=True)
        for regime, pool in pools.items():
            for s in pool:
                graphs = {"oracle": s.y}
                for mode, mdl in models.items():
                    graphs[f"spcrl_{mode}"] = predict_adjacency(mdl, s.Fp, s.Fn, s.S)
                graphs["prior_only"] = np.asarray(s.S, float) + 1e-6 * np.random.default_rng(0).random(s.S.shape)
                graphs["pcorr"] = partial_corr_scores(s.X)
                graphs["notears_linear"] = np.abs(notears_linear(s.X))
                t = time.time()
                run_system(s, graphs, np.random.default_rng(hash((seed, s.sysinst.graph_id)) % (2**31)),
                           W, regime, seed)
                print(f"  [{regime}] {s.sysinst.graph_id} d={s.sysinst.d} {time.time()-t:.0f}s", flush=True)
    print("E4 complete", flush=True)


if __name__ == "__main__":
    main()
