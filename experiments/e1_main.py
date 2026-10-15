"""E1: causal structure recovery, IID and under distribution shift.

Compares amortised SP-CRL (none / hard / soft) against classical and
prior-augmented baselines on held-out systems.
"""
from __future__ import annotations
import argparse, time, numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from spcrl.benchmarks.dataset import (make_pool, TRAIN_FAMILIES, TRAIN_MECHS,
                                      OOD_FAMILY, OOD_MECHS)
from spcrl.causal.train_amortized import train_amortized, score_pool
from spcrl.baselines.classical import pc_skeleton, partial_corr_scores, notears_linear
from spcrl.causal.train import fit_spcrl, auto_threshold
from spcrl.eval.metrics import shd, edge_prf, orientation_f1
from spcrl.util import JsonlWriter


def metrics_from_scores(P, y, thresh_q=0.9):
    d = y.shape[0]; m = ~np.eye(d, dtype=bool)
    yv, pv = y[m].ravel(), np.abs(P)[m].ravel()
    if yv.sum() == 0:
        return None
    out = {"auprc": float(average_precision_score(yv, pv)),
           "auroc": float(roc_auc_score(yv, pv))}
    k = int(y.sum())
    B = np.zeros_like(y, dtype=np.int8)
    if k > 0:
        flat = np.argsort(-np.abs(P) * m.astype(float), axis=None)[:k]
        for f in flat:
            B[np.unravel_index(f, P.shape)] = 1
    p_, r_, f_ = edge_prf(B, y)
    out.update(shd=int(shd(B, y)), precision=p_, recall=r_, f1=f_,
               orient_f1=float(orientation_f1(B, y)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-val", type=int, default=60)
    ap.add_argument("--n-test", type=int, default=60)
    ap.add_argument("--n-deep", type=int, default=12, help="systems for per-system SP-CRL-D")
    ap.add_argument("--n-notears", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--out", default="results/e1_main.jsonl")
    args = ap.parse_args()

    W = JsonlWriter(args.out, meta=vars(args))
    print("run", W.run_id, flush=True)

    pools = {}
    t0 = time.time()
    pools["iid"] = make_pool(args.n_test, 3001, TRAIN_FAMILIES, TRAIN_MECHS, tag="iid", T=args.T)
    pools["mech_ood"] = make_pool(args.n_test, 3002, TRAIN_FAMILIES, OOD_MECHS, tag="mech_ood", T=args.T)
    pools["graph_ood"] = make_pool(args.n_test, 3003, [OOD_FAMILY], TRAIN_MECHS, tag="graph_ood", T=args.T)
    pools["size_ood"] = make_pool(max(20, args.n_test // 2), 3004, TRAIN_FAMILIES, TRAIN_MECHS,
                                  d_range=(50, 70), tag="size_ood", T=args.T)
    pools["noise_ood"] = make_pool(args.n_test, 3005, TRAIN_FAMILIES, TRAIN_MECHS,
                                   noise="student", tag="noise_ood", T=args.T)
    pools["joint_ood"] = make_pool(args.n_test, 3006, [OOD_FAMILY], OOD_MECHS,
                                   noise="laplace", tag="joint_ood", T=args.T)
    print(f"test pools built in {time.time()-t0:.0f}s", flush=True)

    # ---- amortised SP-CRL ----------------------------------------------
    for seed in args.seeds:
        tr = make_pool(args.n_train, 1000 + seed, TRAIN_FAMILIES, TRAIN_MECHS, tag="train", T=args.T)
        va = make_pool(args.n_val, 2000 + seed, TRAIN_FAMILIES, TRAIN_MECHS, tag="val", T=args.T)
        for mode in ["none", "hard", "soft"]:
            for use_enc in ([True, False] if mode == "soft" else [True]):
                t = time.time()
                model, inf = train_amortized(tr, va, mode=mode, use_encoder=use_enc,
                                             epochs=40, seed=seed)
                train_sec = time.time() - t
                name = f"spcrl_amortized_{mode}" + ("" if use_enc else "_noenc")
                for tag, pool in pools.items():
                    for row in score_pool(model, pool):
                        y = np.array([s for s in pool if s.sysinst.graph_id == row["graph_id"]][0].y, np.int8)
                        m = metrics_from_scores(row["P"], y)
                        if m:
                            W.write({"regime": tag, "graph_id": row["graph_id"], "d": row["d"],
                                     "family": row["family"], "mechanism": row["mechanism"],
                                     "n_edges": int(y.sum()), "seed": seed, "method": name,
                                     "train_sec": train_sec,
                                     "val_auprc": inf["best_val_auprc"], **m})
                print(f"[amortized] seed={seed} {name} done ({train_sec:.0f}s)", flush=True)
    # ---- baselines that need no training -------------------------------
    for tag, pool in pools.items():
        for si, s in enumerate(pool):
            y = s.y.astype(np.int8)
            base = {"regime": tag, "graph_id": s.sysinst.graph_id, "d": s.sysinst.d,
                    "family": s.sysinst.family, "mechanism": s.sysinst.mechanism,
                    "n_edges": int(y.sum()), "seed": -1}
            for name, P in [("prior_only", s.S),
                            ("pcorr", partial_corr_scores(s.X)),
                            ("pcorr_prior", partial_corr_scores(s.X) * (s.S > 0.5))]:
                m = metrics_from_scores(np.asarray(P, float), y)
                if m: W.write({**base, "method": name, **m})
            mc = 1 if s.sysinst.d > 40 else (2 if s.sysinst.d > 30 else 3)
            _, st = pc_skeleton(s.X, prior_mask=None, max_cond=mc)
            m = metrics_from_scores(st, y)
            if m: W.write({**base, "method": "pc", **m})
            _, st = pc_skeleton(s.X, prior_mask=s.S, max_cond=mc)
            m = metrics_from_scores(st, y)
            if m: W.write({**base, "method": "pc_prior", **m})
            if si < args.n_notears:
                t = time.time(); Wn = notears_linear(s.X)
                m = metrics_from_scores(np.abs(Wn), y)
                if m: W.write({**base, "method": "notears_linear", "fit_sec": time.time()-t, **m})
                t = time.time(); Wn = notears_linear(s.X, mask=s.S)
                m = metrics_from_scores(np.abs(Wn), y)
                if m: W.write({**base, "method": "notears_linear_hardmask", "fit_sec": time.time()-t, **m})
        print(f"[baselines] {tag} done", flush=True)

    # ---- per-system differentiable SP-CRL (deep variant) ---------------
    for tag, pool in pools.items():
        for s in pool[: args.n_deep]:
            y = s.y.astype(np.int8)
            base = {"regime": tag, "graph_id": s.sysinst.graph_id, "d": s.sysinst.d,
                    "family": s.sysinst.family, "mechanism": s.sysinst.mechanism,
                    "n_edges": int(y.sum()), "seed": 0}
            for mode in ["none", "hard", "soft"]:
                t = time.time()
                _, info = fit_spcrl(s.X, None if mode == "none" else s.S, mode=mode,
                                    lambda_struct=0.5, lambda_l1=0.02, lr=0.03,
                                    inner_iters=400, max_outer=6, seed=0)
                m = metrics_from_scores(np.abs(info["A_raw"]), y)
                if m: W.write({**base, "method": f"spcrl_direct_{mode}",
                               "fit_sec": time.time()-t, **m})
        print(f"[direct] {tag} done", flush=True)

    print("E1 complete", flush=True)


if __name__ == "__main__":
    main()
