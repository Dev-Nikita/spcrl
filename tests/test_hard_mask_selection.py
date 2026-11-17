"""Where the amortised hard mask is exact, and where it is not.

The amortised learner sets the logit of every inadmissible pair to -30 and the
downstream graph is read off by `top_k_dag`, which keeps the |E*| largest scores.
The tests below pin down the boundary of that implementation directly, on
constructed score matrices, instead of arguing about it:

* with admissible logits in any ordinary range, no inadmissible pair is ever
  selected (the case every reported run is in, as far as the deposited records
  can show -- see analysis/check_hard_mask.py);
* if admissible logits fall below the masking value, inadmissible pairs ARE
  selected -- the finite mask is not a strict exclusion, and this test says so;
* zeroing the inadmissible scores before selection makes the exclusion exact
  even in that adversarial case.
"""
import numpy as np

from spcrl.counterfactual.surrogate import top_k_dag

MASK = -30.0
TAU = 0.5


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _forbidden(B, S):
    return int(((B != 0) & (S <= TAU)).sum())


def _scores(S, admissible_logits, rng):
    d = S.shape[0]
    lg = np.where(S > TAU, admissible_logits, MASK)
    np.fill_diagonal(lg, MASK)
    return _sigmoid(lg)


def _prior(d, density, rng):
    S = (rng.random((d, d)) < density).astype(float)
    np.fill_diagonal(S, 0.0)
    return S


def test_ordinary_logits_never_select_a_masked_pair():
    rng = np.random.default_rng(0)
    for trial in range(200):
        d = int(rng.integers(8, 40))
        S = _prior(d, 0.3, rng)
        n_adm = int((S > TAU).sum())
        if n_adm < 2:
            continue
        lg = rng.normal(0.0, 4.0, (d, d))           # far wider than trained logits
        P = _scores(S, lg, rng)
        k = int(rng.integers(1, n_adm + 1))         # never more than the admissible set
        assert _forbidden(top_k_dag(P, k), S) == 0, trial


def test_finite_mask_leaks_when_admissible_logits_fall_below_it():
    rng = np.random.default_rng(1)
    d = 12
    S = _prior(d, 0.3, rng)
    lg = np.full((d, d), MASK - 10.0)               # every admissible pair below the mask
    P = _scores(S, lg, rng)
    k = int((S > TAU).sum())
    assert _forbidden(top_k_dag(P, k), S) > 0


def test_zeroing_masked_scores_makes_exclusion_exact():
    rng = np.random.default_rng(2)
    d = 12
    S = _prior(d, 0.3, rng)
    lg = np.full((d, d), MASK - 10.0)
    P = _scores(S, lg, rng) * (S > TAU)             # strict mask applied before selection
    k = int((S > TAU).sum())
    assert _forbidden(top_k_dag(P, k), S) == 0
