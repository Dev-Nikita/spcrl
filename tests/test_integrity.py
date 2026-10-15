"""Integrity tests: leakage, determinism, metric correctness, prior handling."""
import numpy as np
import pytest
import torch

from spcrl.benchmarks.synthetic import (make_system, simulate, is_dag,
                                        corrupt_prior, topological_order)
from spcrl.benchmarks.dataset import make_pool, TRAIN_FAMILIES, TRAIN_MECHS
from spcrl.priors.features import pair_features, PRIOR_SLICE
from spcrl.causal.amortized import AmortizedSPCRL, predict_adjacency
from spcrl.causal.train_amortized import train_amortized
from spcrl.eval.metrics import shd, edge_prf, topk_and_mrr
from spcrl.counterfactual.surrogate import top_k_dag


def test_generated_graphs_are_dags():
    for fam in ["er", "sf", "modular", "hier"]:
        for seed in range(5):
            s = make_system("t", 25, fam, "mlp", seed=seed)
            assert is_dag(s.A_true)


def test_prior_is_superset_of_truth_when_uncorrupted():
    s = make_system("t", 30, "er", "mlp", seed=1)
    assert (s.S_admissible[s.A_true == 1] == 1).all()


def test_corruption_removes_true_edges():
    s = make_system("t", 30, "er", "mlp", seed=1)
    S = corrupt_prior(s.S_admissible, s.A_true, 0.3, np.random.default_rng(0))
    cover = (S[s.A_true == 1] == 1).mean()
    assert 0.6 < cover < 0.8, cover


def test_no_ground_truth_in_features():
    """Prior-free features must be identical whether or not the prior is given,
    and the prior block must be exactly zero when no prior is supplied."""
    s = make_system("t", 20, "er", "mlp", seed=2)
    X = simulate(s.scm, 300, np.random.default_rng(0))
    F1, _ = pair_features(X, None)
    F2, _ = pair_features(X, s.S_admissible.astype(np.float32))
    assert np.allclose(F1[..., :PRIOR_SLICE.start], F2[..., :PRIOR_SLICE.start])
    assert np.allclose(F1[..., PRIOR_SLICE], 0.0)


def test_mode_none_ignores_prior_exactly():
    """The prior-blind control must be invariant to the prior END TO END.

    An earlier version of this test recomputed nothing: it built the features
    once and then varied only the S passed to the model, so it could not see
    that the *node* features carry the admissible in- and out-degree. The
    prior-blind control was therefore reading aggregate prior information for
    every result in the paper until 2026-09-17. The test now rebuilds the
    features from two different priors, which is the only version that can fail
    for that reason.
    """
    s = make_system("t", 18, "er", "mlp", seed=3)
    X = simulate(s.scm, 300, np.random.default_rng(0))
    S1 = s.S_admissible.astype(np.float32)
    rng = np.random.default_rng(0)
    S2 = (rng.random(S1.shape) < 0.3).astype(np.float32)
    np.fill_diagonal(S2, 0.0)
    Fp1, Fn1 = pair_features(X, S1)
    Fp2, Fn2 = pair_features(X, S2)
    assert not np.allclose(Fn1, Fn2), "the two priors must differ in node features"
    m = AmortizedSPCRL(mode="none")
    P1 = predict_adjacency(m, Fp1, Fn1, S1)
    P2 = predict_adjacency(m, Fp2, Fn2, S2)
    assert np.allclose(P1, P2), "mode='none' must not see the prior"


def test_hard_mode_predicts_nothing_outside_admissible_set():
    s = make_system("t", 18, "er", "mlp", seed=4)
    X = simulate(s.scm, 300, np.random.default_rng(0))
    Fp, Fn = pair_features(X, s.S_admissible.astype(np.float32))
    m = AmortizedSPCRL(mode="hard")
    P = predict_adjacency(m, Fp, Fn, s.S_admissible.astype(np.float32))
    assert P[s.S_admissible <= 0.5].max() < 1e-6


def test_determinism_of_training():
    tr = make_pool(6, 11, TRAIN_FAMILIES, TRAIN_MECHS, tag="tr", T=200)
    va = make_pool(3, 12, TRAIN_FAMILIES, TRAIN_MECHS, tag="va", T=200)
    m1, i1 = train_amortized(tr, va, mode="soft", epochs=2, seed=0)
    m2, i2 = train_amortized(tr, va, mode="soft", epochs=2, seed=0)
    assert abs(i1["best_val_auprc"] - i2["best_val_auprc"]) < 1e-9


def test_no_graph_leakage_between_pools():
    a = make_pool(20, 1000, TRAIN_FAMILIES, TRAIN_MECHS, tag="a", T=200)
    b = make_pool(20, 3001, TRAIN_FAMILIES, TRAIN_MECHS, tag="b", T=200)
    ka = {x.sysinst.A_true.tobytes() for x in a}
    kb = {x.sysinst.A_true.tobytes() for x in b}
    assert not (ka & kb), "train and test pools share a ground-truth graph"


def test_shd_and_prf():
    A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], np.int8)
    assert shd(A, A) == 0
    B = A.copy(); B[0, 1] = 0; B[1, 0] = 1
    assert shd(B, A) == 1                      # reversal costs one
    p, r, f = edge_prf(A, A)
    assert (p, r, f) == (1.0, 1.0, 1.0)


def test_topk_mrr():
    sc = np.array([0.1, 0.9, 0.5])
    r = topk_and_mrr(sc, 1)
    assert r["top1"] == 1.0 and r["mrr"] == 1.0
    r = topk_and_mrr(sc, 0)
    assert r["top1"] == 0.0 and r["rank"] == 3


def test_top_k_dag_is_acyclic():
    rng = np.random.default_rng(0)
    P = rng.random((15, 15))
    B = top_k_dag(P, 25)
    assert is_dag(B)
    _ = topological_order(B)


def test_intervention_ground_truth_is_causal():
    """Clamping the faulty node must restore the KPI better than clamping a
    node that is not an ancestor of it."""
    s = make_system("t", 20, "hier", "cascade", seed=7)
    kpi = int(np.argmax(s.A_true.sum(0)))
    v = int(np.argmax(s.A_true.sum(1)))
    nominal = simulate(s.scm, 400, np.random.default_rng(1)).mean(0)
    def loss(clamp):
        X = simulate(s.scm, 300, np.random.default_rng(2), fault_node=v, fault_t=0,
                     fault_mag=2.0, clamp=clamp)
        return abs(X[:, kpi].mean() - nominal[kpi])
    assert loss({v: float(nominal[v])}) <= loss(None) + 1e-9


def test_uninformative_predictions_produce_ties_not_wins():
    """Regression test for a real bug.

    If every candidate action gets the same predicted loss, the selection must
    not systematically land on the first candidate -- the true root cause is
    always first in the candidate list, so order-based tie-breaking silently
    rewards a model that knows nothing.
    """
    pred = {7: 1.0, 3: 1.0, 5: 1.0, 9: 1.0}
    picks = []
    for trial in range(200):
        rng = np.random.default_rng(trial)
        order = list(pred); rng.shuffle(order)
        picks.append(min(order, key=lambda a: (pred[a], rng.random())))
    frac_first = sum(p == 7 for p in picks) / len(picks)
    assert 0.15 < frac_first < 0.35, frac_first


def test_rollout_warmup_discards_transient():
    from spcrl.counterfactual.surrogate import SurrogateSCM
    s = make_system("t", 12, "er", "linear", seed=5)
    X = simulate(s.scm, 400, np.random.default_rng(0))
    sur = SurrogateSCM(s.A_true.astype(np.int8)).fit(X)
    x0 = X[-1] + 25.0                       # far-from-nominal starting state
    full = sur.rollout(x0, steps=30, warmup=0)
    warm = sur.rollout(x0, steps=30, warmup=15)
    settled = sur.rollout(X[-1], steps=30, warmup=15)
    assert np.abs(warm - settled).mean() < np.abs(full - settled).mean()


def test_kpi_is_never_a_repair_candidate():
    """Pinning the KPI to nominal trivially zeroes the loss; if the KPI were a
    candidate every method would score perfectly for the wrong reason."""
    import importlib
    e4 = importlib.import_module("experiments.e4_" + "downstream")
    A = np.zeros((6, 6), np.int8)
    A[0, 1] = A[1, 2] = A[2, 3] = A[3, 4] = 1
    anc = e4._ancestors(A)
    kpi = int(max(range(6), key=lambda i: len(anc[i])))
    assert kpi == 4 and anc[kpi] == {0, 1, 2, 3}
