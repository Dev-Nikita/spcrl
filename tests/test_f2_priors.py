"""The F2 controls are what the pre-registration says they are.

Rebuilds every registered prior from the topology file and the F1 deployment
prior and checks (i) that each matches its committed hash, and (ii) that every
channel-permutation control preserves what it claims to preserve.
"""
import json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAN = os.path.join(ROOT, "docs", "preregistration", "F2_PRIORS_MANIFEST.json")
DATA = os.path.join(ROOT, "data", "external", "free5gc_f1")
TOPO = os.path.join(ROOT, "benchmarks", "telecom", "f1_topology.yaml")


def _build():
    import sys
    sys.path.insert(0, os.path.join(ROOT, "analysis"))
    from build_f2_priors import build
    return build(DATA, TOPO)


def test_every_prior_matches_its_registered_hash():
    from spcrl.priors.alignment import sha256
    priors, variables, S1 = _build()
    man = json.load(open(MAN))
    assert sha256(S1) == man["deployment_prior_sha256"]
    assert variables == man["variables"]
    assert set(priors) == set(man["priors"])
    for k, S in priors.items():
        assert sha256(S) == man["priors"][k]["sha256"], k


def test_channel_permutations_preserve_structure_and_move_channels():
    from spcrl.priors.alignment import invariants
    priors, variables, _ = _build()
    P2 = priors["P2_semantic"]
    for k, S in priors.items():
        if not k.startswith("CH"):
            continue
        inv = invariants(S, P2, variables)
        assert inv["density_equal"] and inv["service_blocks_equal"], k
        assert inv["degree_multiset_per_service_equal"] and inv["diagonal_zero"], k
        assert inv["fraction_relations_moved"] > 0, k
    full = [invariants(S, P2, variables)["fraction_relations_moved"]
            for k, S in priors.items() if k.startswith("CH9_")]
    assert min(full) > 0.9
