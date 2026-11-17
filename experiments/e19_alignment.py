"""E19: the alignment experiments of PREREGISTRATION_F2.md (E1-E3).

Same pipeline as E10 (experiments/e10_real.py) -- amortised models trained only
on synthetic systems, applied unchanged to a running system, one ridge
surrogate per node, services ranked by their most deviating live channel -- with
three differences, none of which changes what is computed:

  * the models are trained once per seed and frozen to disk
    (data/f2_frozen/models/), so every prior is scored by the same models and the F2
    analysis uses exactly the models whose hashes the pre-registration commits;
  * many priors are scored in one run, read from data/f2_frozen/f2_priors.npz, whose
    matrices must match the hashes in F2_PRIORS_MANIFEST.json;
  * --relabel N re-runs selected priors with the variables in N random orders
    (E3a): channels and priors permuted consistently, so a pipeline with no
    positional dependence returns the same ranking.

Resumable: (seed, prior, relabel) combinations already in --out are skipped.

    python experiments/e19_alignment.py --data <campaign dir> --env-name f2 \
        --out results/e19_f2.jsonl
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spcrl.benchmarks.dataset import make_pool, TRAIN_FAMILIES, TRAIN_MECHS
from spcrl.benchmarks.real import load_validated, reference_statistics, deviation_scores
from spcrl.causal.train_amortized import train_amortized
from spcrl.causal.amortized import predict_adjacency, build_variant
from spcrl.priors.features import pair_features
from spcrl.priors.alignment import sha256
from spcrl.counterfactual.surrogate import SurrogateSCM, parents_from_scores
from spcrl.util import JsonlWriter
from experiments.e10_real import service_of, rank_services, topk_service

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "data", "f2_frozen", "models")
VARIANTS = ["none", "hard", "soft"]


def file_sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def frozen_models(seed, n_train, n_val, train_corruption, train=True, model_dir=MODEL_DIR):
    MODEL_DIR_ = model_dir
    os.makedirs(MODEL_DIR_, exist_ok=True)
    models, missing = {}, []
    for v in VARIANTS:
        p = os.path.join(MODEL_DIR_, f"seed{seed}_{v}.pt")
        if os.path.exists(p):
            m = build_variant(v); m.load_state_dict(torch.load(p)); m.eval()
            models[v] = m
        else:
            missing.append(v)
    if missing:
        if not train:
            raise SystemExit(f"frozen models missing for seed {seed}: {missing}")
        tr = make_pool(n_train, 1000 + seed, TRAIN_FAMILIES, TRAIN_MECHS,
                       tag="train", corruption=train_corruption)
        va = make_pool(n_val, 2000 + seed, TRAIN_FAMILIES, TRAIN_MECHS,
                       tag="val", corruption=train_corruption)
        for v in missing:
            t = time.time()
            m, _ = train_amortized(tr, va, variant=v, epochs=40, seed=seed)
            torch.save(m.state_dict(), os.path.join(MODEL_DIR_, f"seed{seed}_{v}.pt"))
            m.eval(); models[v] = m
            print(f"  seed {seed}: {v} trained in {time.time()-t:.0f}s", flush=True)
    return models


def done_keys(path):
    keys = set()
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            if r.get("method") == "_done":
                keys.add((r["seed"], r["prior"], r["relabel"]))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--env-name", required=True)
    ap.add_argument("--priors", default=os.path.join(ROOT, "data", "f2_frozen", "f2_priors.npz"))
    ap.add_argument("--manifest", default=os.path.join(
        ROOT, "docs", "preregistration", "F2_PRIORS_MANIFEST.json"))
    ap.add_argument("--only", nargs="*", default=None, help="subset of prior names")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--relabel", type=int, default=0,
                    help="E3a: number of random variable orders for --relabel-priors")
    ap.add_argument("--relabel-priors", nargs="*", default=["P2_semantic", "P1_topology"])
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-val", type=int, default=60)
    ap.add_argument("--train-corruption", type=float, default=0.2)
    ap.add_argument("--parents-per-node", type=int, default=3)
    ap.add_argument("--no-train", action="store_true",
                    help="refuse to train: use only frozen models (the F2 analysis)")
    ap.add_argument("--model-dir", default=MODEL_DIR,
                    help="debug runs MUST use a scratch dir, never the frozen one")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    man = json.load(open(a.manifest))
    bundle = np.load(a.priors)
    priors = {k: bundle[k].astype(np.float32) for k in bundle.files}
    for k, S in priors.items():                       # refuse unregistered priors
        if sha256(S) != man["priors"][k]["sha256"]:
            raise SystemExit(f"prior {k} does not match its registered hash")
    if a.only:
        priors = {k: priors[k] for k in a.only}

    S_dep, meta, normals, faults, report = load_validated(a.data)
    variables = meta["variables"]
    if variables != man["variables"]:
        raise SystemExit("campaign variables differ from the registered ones")
    dep_hash = sha256(np.asarray(S_dep, np.float32))
    if dep_hash != man["deployment_prior_sha256"]:
        raise SystemExit("deployment prior differs from the registered one")
    X_norm = np.concatenate(normals, axis=0)
    svc = service_of(variables)
    live = np.array([X_norm[:, i].std() > 1e-12 for i in range(len(variables))])

    done = done_keys(a.out)
    W = JsonlWriter(a.out, meta=vars(a), append=True)
    print(f"run {W.run_id}: {len(priors)} priors, {len(faults)} faults, "
          f"{len(normals)} nominal windows, {int(live.sum())}/{len(variables)} live", flush=True)

    jobs = [(p, 0) for p in priors]
    jobs += [(p, r) for r in range(1, a.relabel + 1) for p in a.relabel_priors if p in priors]
    for seed in a.seeds:
        models = frozen_models(seed, a.n_train, a.n_val, a.train_corruption,
                               train=not a.no_train, model_dir=a.model_dir)
        W.write({"method": "_models", "seed": seed, **{
            f"sha_{v}": file_sha(os.path.join(a.model_dir, f"seed{seed}_{v}.pt"))
            for v in VARIANTS}})
        for pname, rl in jobs:
            if (seed, pname, rl) in done:
                continue
            S = priors[pname]
            order = np.arange(len(variables))
            if rl:
                order = np.random.default_rng([20261117, seed, rl]).permutation(len(variables))
            Sx = S[np.ix_(order, order)]
            Xn = X_norm[:, order]
            norm_x = [X[:, order] for X in normals]
            svc_x, live_x = svc[order], live[order]
            # tie-breaking jitter drawn in the ORIGINAL variable order and carried
            # along, so that a consistent relabelling cannot change it (E3a)
            J = np.random.default_rng([seed, 7]).random(S.shape)[np.ix_(order, order)]
            Fp, Fn = pair_features(Xn, Sx)
            graphs = {f"spcrl_{v}": (predict_adjacency(models[v], Fp, Fn, Sx), 0.0)
                      for v in VARIANTS}
            # The masked logit of an inadmissible pair is -30, a score of 9.4e-14,
            # which the default parent selection (score > 0) still accepts: a node
            # with fewer than k admissible candidates is topped up with masked pairs
            # in index order. `spcrl_hard` keeps that behaviour for comparability
            # with F1; `spcrl_hard_strict` admits only scores above 1e-6, i.e. never
            # a masked pair (PREREGISTRATION_F2.md, Sec. 6).
            graphs["spcrl_hard_strict"] = (graphs["spcrl_hard"][0], 1e-6)
            graphs["prior_only"] = (np.asarray(Sx, float) + 1e-6 * J, 0.0)
            sur, ref = {}, {}
            for name, (P, min_score) in graphs.items():
                B = parents_from_scores(P, a.parents_per_node, min_score=min_score)
                m = SurrogateSCM(B, kind="ridge").fit(Xn)
                sur[name], ref[name] = m, reference_statistics(m, norm_x)
            Rn = np.stack([X.mean(0) for X in norm_x])
            mu, sd = Rn.mean(0), Rn.std(0)
            pos = sd[sd > 0]
            sd = np.maximum(sd, float(np.quantile(pos, 0.25)) if len(pos) else 1e-6)
            for f in faults:
                Xf = f["X"][:, order]
                base = {"env": a.env_name, "seed": seed, "prior": pname, "relabel": rl,
                        "run_id_fault": f["run_id"], "fault_service": f["fault_service"],
                        "fault_type": f["fault_type"], "magnitude": f["magnitude"],
                        "cell": f"{f['fault_service']}|{f['fault_type']}"}
                sc = np.abs(Xf.mean(0) - mu) / sd
                W.write({**base, "method": "anomaly_no_structure",
                         **topk_service(rank_services(sc, svc_x, mask=live_x), f["fault_service"])})
                for name, m in sur.items():
                    s_ = deviation_scores(m, Xf, *ref[name])
                    W.write({**base, "method": name,
                             **topk_service(rank_services(s_, svc_x, mask=live_x), f["fault_service"])})
            W.write({"method": "_done", "seed": seed, "prior": pname, "relabel": rl})
            print(f"  seed {seed}: {pname} (relabel {rl}) scored", flush=True)
    print("E19 complete", flush=True)


if __name__ == "__main__":
    main()
