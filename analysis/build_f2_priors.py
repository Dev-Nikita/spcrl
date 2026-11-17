"""Build every structural prior the F2 analysis may use, and commit to them.

    python analysis/build_f2_priors.py <campaign-dir-with-prior.npy+variables.json> \
        benchmarks/telecom/f1_topology.yaml

Writes data/f2_frozen/f2_priors.npz and docs/preregistration/F2_PRIORS_MANIFEST.json
(name, SHA-256, invariants relative to P2). Run before any F2 data exist; the
manifest's own hash goes into PREREGISTRATION_F2.md. The F2 analysis refuses to
run if a prior it rebuilds does not match its committed hash.
"""
import hashlib, json, os, sys
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spcrl.priors import semantic as sem
from spcrl.priors import alignment as al


def build(data_dir, topo_path):
    S1 = np.load(os.path.join(data_dir, "prior.npy")).astype(np.float32)
    variables = json.load(open(os.path.join(data_dir, "variables.json")))["variables"]
    topo = yaml.safe_load(open(topo_path))
    dep = {x["name"]: set(x["depends_on"]) for x in topo["services"]}
    entry = topo.get("entrypoint", topo["services"][0]["name"])
    kind = topo.get("semantics", "microservice")
    pub = sem.variants(S1, variables, dep, entry, kind)       # F1's published controls
    P2 = pub["P2_semantic"]
    priors = dict(pub)
    priors.update(al.build_f2_priors(P2, variables, dep, entry, kind))
    return priors, variables, S1


def main():
    data_dir, topo_path = sys.argv[1], sys.argv[2]
    priors, variables, S1 = build(data_dir, topo_path)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(root, "data", "f2_frozen"), exist_ok=True)
    np.savez(os.path.join(root, "data", "f2_frozen", "f2_priors.npz"), **priors)
    P2 = priors["P2_semantic"]
    man = {"seed_f2": al.SEED_F2, "e2_levels": list(al.E2_LEVELS),
           "k_draws": al.K_DRAWS, "k_wrong_id": al.K_WRONG_ID,
           "variables": variables, "deployment_prior_sha256": al.sha256(S1),
           "priors": {}}
    for name, S in priors.items():
        man["priors"][name] = {"sha256": al.sha256(S),
                               "invariants_vs_P2": al.invariants(S, P2, variables)}
    out = os.path.join(root, "docs", "preregistration", "F2_PRIORS_MANIFEST.json")
    with open(out, "w") as f:
        json.dump(man, f, indent=1, sort_keys=True)
    h = hashlib.sha256(open(out, "rb").read()).hexdigest()
    print(f"{len(priors)} priors; manifest {out}\nmanifest sha256 {h}")


if __name__ == "__main__":
    main()
