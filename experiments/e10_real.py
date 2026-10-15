"""E10: zero-shot transfer of SP-CRL to a real testbed.

The amortised model is trained ONLY on synthetic systems and applied, without
any retraining, to telemetry collected from a running Kubernetes (or free5GC)
deployment. The structural prior comes from the deployment description; the
ground-truth root cause comes from the campaign log, which was written before
each run started.

There is no ground-truth causal graph for a real system, so structure-recovery
metrics are not reported here. What is tested is the claim that matters
operationally: does an uncertain structural prior improve root-cause
identification, and does the soft/hard ordering survive when the deployment
description is stale or wrong?
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np

from spcrl.benchmarks.dataset import make_pool, TRAIN_FAMILIES, TRAIN_MECHS
from spcrl.benchmarks.real import (load_validated, reference_statistics,
                                   deviation_scores)
from spcrl.causal.train_amortized import train_amortized
from spcrl.causal.amortized import predict_adjacency
from spcrl.priors.features import pair_features
from spcrl.counterfactual.surrogate import SurrogateSCM, parents_from_scores
from spcrl.util import JsonlWriter

VARIANTS = ["none", "hard", "soft", "soft_noenc", "soft_gate"]


def corrupt_real_prior(S: np.ndarray, c: float, rng) -> np.ndarray:
    """Same corruption operator as the synthetic benchmark, applied to the
    admissibility read off the deployment description."""
    S = S.copy()
    d = S.shape[0]
    on = np.argwhere(S > 0.5)
    off = np.argwhere((S <= 0.5) & (~np.eye(d, dtype=bool)))
    n = int(round(c * len(on)))
    if n and len(on):
        for k in rng.choice(len(on), size=min(n, len(on)), replace=False):
            i, j = on[k]; S[i, j] = 0.0
    if n and len(off):
        for k in rng.choice(len(off), size=min(n, len(off)), replace=False):
            i, j = off[k]; S[i, j] = 1.0
    return S


def service_of(variables):
    return np.array([v.split("::")[0] for v in variables])


def rank_services(scores: np.ndarray, svc: np.ndarray, agg="max",
                  mask: np.ndarray | None = None):
    """Rank services by their most deviating channel.

    `mask` restricts scoring to channels that vary in nominal operation. Error
    and retry counters are identically zero when nothing is wrong, so no model
    fitted on nominal data can explain them away, and they would otherwise decide
    every ranking on their own - typically in favour of whichever service is
    downstream of the fault rather than the faulty one.
    """
    names = sorted(set(svc.tolist()))
    f = {"max": np.max, "mean": np.mean}[agg]
    val = []
    for n in names:
        sel = (svc == n) if mask is None else ((svc == n) & mask)
        val.append(f(scores[sel]) if sel.any() else -np.inf)
    return [names[i] for i in np.argsort(-np.array(val))]


def topk_service(order, truth, ks=(1, 3)):
    r = order.index(truth) + 1 if truth in order else len(order) + 1
    out = {f"top{k}": float(r <= k) for k in ks}
    out.update(mrr=1.0 / r, rank=r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="collected environment directory")
    ap.add_argument("--env-name", default=None)
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-val", type=int, default=60)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--corruptions", type=float, nargs="+", default=[0.0, 0.2, 0.5])
    ap.add_argument("--parents-per-node", type=int, default=3)
    ap.add_argument("--live-only", action="store_true", default=True,
                    help="score only channels that vary in nominal operation")
    ap.add_argument("--prior-variant", default="deployment",
                    choices=["deployment", "random", "permissive", "within_only",
                             "transpose", "P1_topology", "P2_semantic",
                             "P3_random", "P3b_random_at_P2_density",
                             "P4_degree_shuffled", "P6_wrong_topology",
                             "driver_declared"],
                    help="which structural prior to give the model. "
                         "deployment: the one read from the deployment "
                         "description. random: a random admissibility matrix of "
                         "the same density, which carries no information but "
                         "commits to a structure just as strongly - the control "
                         "that separates 'the prior points the wrong way' from "
                         "'any sparse commitment hurts here'. permissive: every "
                         "relation admissible. within_only: only relations "
                         "inside a service. transpose: the deployment prior with "
                         "every relation reversed.")
    ap.add_argument("--topology", default=None,
                    help="topology file the prior was built from; needed for the "
                         "pre-registered P-variants and driver_declared "
                         "(docs/preregistration/PREREGISTRATION_EXTERNAL.md)")
    ap.add_argument("--driver-channels", nargs="*", default=["gateway::throughput"],
                    help="channels declared as the shared driver (K3)")
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS,
                    help="model variants to train; the prior controls need only "
                         "none, hard and soft")
    ap.add_argument("--train-corruption", type=float, default=0.2,
                    help="prior corruption the synthetic training pool is built with")
    ap.add_argument("--out", default="results/e10_real.jsonl")
    a = ap.parse_args()
    env_name = a.env_name or os.path.basename(os.path.normpath(a.data))
    W = JsonlWriter(a.out, meta=vars(a))
    print("run", W.run_id, "env", env_name, flush=True)

    S0, meta, normals, faults, report = load_validated(a.data)
    variables = meta["variables"]
    svc = service_of(variables)
    if a.prior_variant.startswith("P") or a.prior_variant == "driver_declared":
        # Pre-registered controls: built from the deployment description and a
        # fixed seed only (spcrl/priors/semantic.py).
        import yaml
        from spcrl.priors import semantic as sem
        topo_path = a.topology or meta.get("topology_file")
        topo = yaml.safe_load(open(topo_path))
        dep = {x["name"]: set(x["depends_on"]) for x in topo["services"]}
        entry = topo.get("entrypoint", topo["services"][0]["name"])
        kind = topo.get("semantics", "microservice")
        d0 = S0.shape[0]
        off = ~np.eye(d0, dtype=bool)
        if a.prior_variant == "driver_declared":
            S0 = np.asarray(S0, np.float32).copy()
            for ch in a.driver_channels:
                k = variables.index(ch)
                S0[k, :] = 1.0; S0[:, k] = 1.0
            np.fill_diagonal(S0, 0.0)
        else:
            S0 = sem.variants(S0, variables, dep, entry, kind)[a.prior_variant]
        print(f"prior variant '{a.prior_variant}': density "
              f"{float((S0 > 0.5)[off].mean()):.3f}", flush=True)
    elif a.prior_variant != "deployment":
        rngp = np.random.default_rng(20260917)
        d0 = S0.shape[0]
        off = ~np.eye(d0, dtype=bool)
        if a.prior_variant == "random":
            dens = float((S0 > 0.5)[off].mean())
            S0 = (rngp.random((d0, d0)) < dens).astype(np.float32)
            np.fill_diagonal(S0, 0.0)
        elif a.prior_variant == "permissive":
            S0 = off.astype(np.float32)
        elif a.prior_variant == "within_only":
            same = np.array([[svc[i] == svc[j] for j in range(d0)]
                             for i in range(d0)])
            S0 = (same & off).astype(np.float32)
        elif a.prior_variant == "transpose":
            S0 = np.ascontiguousarray(S0.T).astype(np.float32)
        print(f"prior variant '{a.prior_variant}': density "
              f"{float((S0 > 0.5)[off].mean()):.3f}", flush=True)
    if len(normals) < 2 or not faults:
        raise SystemExit("need at least two nominal windows and one fault window")
    X_norm = np.concatenate(normals, axis=0)
    W.write({"env": env_name, "method": "_dataset_report", **report,
             "prior_variant": a.prior_variant,
             "n_services": int(len(set(svc)))})
    live = np.array([X_norm[:, i].std() > 1e-12 for i in range(len(variables))])
    print(f"{len(variables)} variables ({int(live.sum())} with nominal variation), "
          f"{len(set(svc))} services, {len(normals)} nominal windows "
          f"({X_norm.shape[0]} samples), {len(faults)} usable fault runs", flush=True)

    for seed in a.seeds:
        tr = make_pool(a.n_train, 1000 + seed, TRAIN_FAMILIES, TRAIN_MECHS,
                       tag="train", corruption=a.train_corruption)
        va = make_pool(a.n_val, 2000 + seed, TRAIN_FAMILIES, TRAIN_MECHS,
                       tag="val", corruption=a.train_corruption)
        models = {}
        for v in a.variants:
            t = time.time()
            models[v], _ = train_amortized(tr, va, variant=v, epochs=40, seed=seed)
            print(f"  seed {seed}: {v} trained in {time.time()-t:.0f}s", flush=True)

        for c in a.corruptions:
            rng = np.random.default_rng(hash((seed, c)) % (2 ** 31))
            S = corrupt_real_prior(S0, c, rng) if c > 0 else S0
            Fp, Fn = pair_features(X_norm, S)
            n_adm = int((S > 0.5).sum())
            graphs = {f"spcrl_{v}": predict_adjacency(models[v], Fp, Fn, S)
                      for v in a.variants}
            graphs["prior_only"] = np.asarray(S, float) + 1e-6 * rng.random(S.shape)
            sur, ref = {}, {}
            for name, P in graphs.items():
                # per-node parent selection, not a global DAG projection: the
                # diagnostic fits one regression per node, and projecting a
                # near-binary prior to a DAG silently discards most of it
                B = parents_from_scores(P, a.parents_per_node)
                # Ridge, not gradient boosting: the nominal record is a few
                # hundred samples over 40 variables, and a boosted tree per node
                # would fit noise rather than the dependence we are testing.
                m = SurrogateSCM(B, kind="ridge").fit(X_norm)
                sur[name] = m
                ref[name] = reference_statistics(m, normals)

            for f in faults:
                truth = f["fault_service"]
                base = {"env": env_name, "seed": seed, "corruption": c,
                        "prior_variant": a.prior_variant,
                        "run_id_fault": f["run_id"], "fault_service": truth,
                        "fault_type": f["fault_type"], "magnitude": f["magnitude"],
                        "n_variables": len(variables), "n_live": int(live.sum()),
                        "n_admissible": n_adm, "parents_per_node": a.parents_per_node}
                # structure-free control: the same standardised deviation, but
                # of the raw observations instead of the surrogate residuals
                Rn = np.stack([X.mean(0) for X in normals])
                mu, sd = Rn.mean(0), Rn.std(0)
                pos = sd[sd > 0]
                sd = np.maximum(sd, float(np.quantile(pos, 0.25)) if len(pos) else 1e-6)
                sc = np.abs(f["X"].mean(0) - mu) / sd
                W.write({**base, "method": "anomaly_no_structure",
                         **topk_service(rank_services(sc, svc, mask=live if a.live_only else None), truth)})
                for name, m in sur.items():
                    s_ = deviation_scores(m, f["X"], *ref[name])
                    W.write({**base, "method": name,
                             **topk_service(rank_services(s_, svc, mask=live if a.live_only else None), truth)})
            print(f"  seed {seed} c={c}: {len(faults)} faults scored", flush=True)
    print("E10 complete", flush=True)


if __name__ == "__main__":
    main()
