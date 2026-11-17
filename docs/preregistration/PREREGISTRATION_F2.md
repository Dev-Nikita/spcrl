# Pre-registration F2 — does channel alignment survive a matched control on new data?

**Frozen:** 17 November 2026 (UTC), before any F2 data existed.
**Authors:** Nikita Tarasov, Roman Zinko.
**Scope:** E1–E3 of the alignment programme, tested confirmatorily on a new
free5GC campaign (F2). F1 (32 fault windows, October 2026) is the
*development* set: it was used to estimate variance and to debug the new
controls, and no F1 result can enter a confirmatory claim.

## 1. Why, in one paragraph

On F1 the semantic prior P2 beat a random prior at P2's density (P3b) by
+0.406 Top-1. P3b matches density and nothing else. P2 also has a particular
service-level topology, 27 of its 81 relations inside a single service (P3b: 6
of 71), and a particular degree structure — so the F1 contrast cannot say
whether what helps is *channel semantics* (which measured signal a relation is
attached to) or any of those. F2 tests channel semantics against a control that
holds everything else fixed, on fault types the hypothesis was not formed on.

## 2. Two facts found while designing F2, recorded here because they shape it

1. **The independent unit is the cell, not the fault window.** In F1 the two
   repeats of each (target, fault type) cell returned the same soft-prior P2
   diagnosis in 16 of 16 cells. F1 therefore has 16 independent units, not 32,
   and fault-level bootstrap intervals are too narrow by roughly 40–50 %
   (recomputed at cell level, F1's alignment contrast is +0.406
   [+0.219, +0.604], identity −0.156 [−0.302, −0.021]; neither conclusion
   changes). F2 is designed around cells: one window per cell, new fault
   families to obtain new cells, inference by bootstrap over cells.
2. **The real-system hard arm admitted masked parents.** The diagnostic
   surrogate selects the 3 strongest parents per node with threshold > 0; a
   masked pair scores 9.4e-14, so a node with fewer than 3 admissible
   candidates is topped up with masked pairs in index order. Under P2, 24 of 36
   nodes have fewer than 3 admissible parents. Soft results are unaffected (no
   ties). F2 reports the hard arm under both rules: `spcrl_hard` (F1's rule,
   for comparability) and `spcrl_hard_strict` (threshold 1e-6: never a masked
   pair). Correcting the published K2b/F1 hard numbers is handled in the
   manuscript, separately from F2.

## 3. Controls (all built from the topology file and a seed; never from telemetry)

Code: `spcrl/priors/alignment.py`, `analysis/build_f2_priors.py`. Every matrix
and its invariants relative to P2 are in `F2_PRIORS_MANIFEST.json`; the analysis
refuses to run on a prior whose hash differs.

| Prior | What it is | Preserved vs P2 (machine-checked) |
|---|---|---|
| `P2_semantic` | F1's semantic prior, unchanged | — |
| `CHL_dk` | channel labels of L services deranged (every channel of a chosen service moves), relations carried along; L ∈ {1,2,5,7,9}, k = 0..9 | relation count (81), service-to-service block counts, within-service count (27), per-service multiset of channel in/out-degrees |
| `WID_dk` | P2 rebuilt on a call graph with non-entry services deranged; k = 0..4 | relation count, within-service count; *not* the service blocks (by design) |
| `P1`…`P6`, `P3b` | F1's published controls, for replication | as in F1 |

Realised fraction of P2's relations moved: CH1 0.09–0.25, CH2 0.23–0.41,
CH5 0.54–0.79, CH7 0.75–0.89, CH9 0.91–0.95.

## 4. Hypotheses and decision rules (fixed now)

Unit: cell = (fault target, fault type). Within a cell, Top-1 is averaged over
its windows and the three training seeds and, for a family of draws, over
draws. Uncertainty: two-way percentile bootstrap over cells and training seeds
(Amendment 1), B = 20 000, seed 20261117. Model: the soft-prior amortised model (`spcrl_soft`) unless stated; ten frozen training seeds (Amendment 1).

**H-F2-1 (primary, E1).** On the 24 cells of the three *new* fault families,
  Δ = Top-1(P2) − E[Top-1(CH9)] > 0.
  SUPPORTED if the 95 % CI lower bound > 0; REFUTED if the upper bound < 0;
  INCONCLUSIVE otherwise. The point estimate is reported against a smallest
  effect of interest of 0.10.

**S1.** The same contrast on the 16 replication cells and on all 40 cells
  (descriptive, with CIs).

**S2 (E2, dose–response).** Per cell, OLS slope of Top-1 on the realised
  fraction of relations moved across levels {0,1,2,5,7,9}; mean slope over the
  24 new cells. SUPPORTED if the CI upper bound < 0; REFUTED if the lower bound
  > 0. The whole curve is reported.

**S3 (identity, channel semantics held fixed).** Top-1(P2) − E[Top-1(WID)] on
  the new cells. "No meaningful identity benefit" SUPPORTED if the CI upper
  bound < 0.15; REFUTED if the lower bound > 0; else INCONCLUSIVE.

**S4 (replication).** F1's five published contrasts (total P1−P5, identity
  P1−P6, alignment P2−P3b, density P1−P3, degree P1−P4) on all 40 F2 cells, at
  cell level, reported next to F1's cell-level values.

**S5 (hard arm).** H-F2-1's contrast under `spcrl_hard_strict` and under
  `spcrl_hard`; descriptive.

**E3a (relabelling).** Variables and prior permuted consistently in 3 random
  orders for P2 and P1. Prediction: every ranking unchanged for `spcrl_soft`,
  `spcrl_none`, `spcrl_hard_strict`, `prior_only` and the structure-free score.
  `spcrl_hard` may change (its tie-filling depends on index order — Sec. 2).

**E3c (leakage).** Discharged by construction and recorded: every prior and
  every model is hashed in this document before F2 exists; the priors depend on
  the topology file and seeds only; the campaign log, which holds the ground
  truth, is never read by any prior-building code.

## 5. The campaign

* System: the same free5GC deployment, topology file, 36 channels, collector,
  scrape interval (1 s) and workload as F1 (two-gNB stabilised workload of the
  F1 amendment of 13 October 2026). The workload script's hash, image digests,
  kernel and Docker versions are recorded in `calibration.json` before the first
  run.
* Fault targets: the eight network functions (not the database).
* Families: **replication** — `cpu_saturation` (3 busy loops), `sbi_latency`
  (netem delay 180 ms), exactly as F1; **new** — `sbi_loss` (netem loss 10 %),
  `sbi_ratelimit` (netem rate = 30 % of the target's own p95 transmit rate,
  floor 16 kbit), `cpu_throttle` (cgroup quota = 30 % of the target's own p95
  CPU, floor 0.01 cores). The p95s are measured on the 10 minutes of nominal
  operation immediately before the first run (`campaign_f2.py`), never on
  fault data.
* 40 fault cells × 1 window + 6 nominal windows = 46 runs, shuffled with seed 2
  (F1: 0); 300 s warm-up, 300 s fault, 60 s cool-down; ≈ 8.4 h.
* Runner: `benchmarks/telecom/scripts/campaign_f2.py`; injector `inject_f2.sh`;
  smoke test `smoke_test_f2.sh`.

## 6. Validity gates and exclusions

* The F2 smoke test must pass before the campaign starts.
* G1–G3 exactly as for F1 (`analysis/external_gates.py`, same thresholds). A
  campaign that fails a gate is reported as failed and not analysed.
* Fault windows the loader marks unusable are excluded, as in F1. If more than
  6 of the 24 new-family windows are unusable, F2 is reported as failed.
* No other exclusion. No re-running of individual cells.

## 7. Deviations

* If the campaign dies, its log is archived as aborted and the campaign is
  restarted from scratch with the same seed; partial data are never analysed.
* Any change to this document, the code or the frozen artifacts after F2 data
  exist is a dated amendment in this file, stating whether it was made before
  or after the analysis was run.
* A development run of this exact pipeline on F1 is performed after this
  document is frozen. Its results are exploratory and reported as such; they
  cannot change any rule above.

## 8. Power, stated honestly

From F1 the cell-level SD of the closest available contrast (P2 − P3b) is
≈ 0.39. With 24 new cells the standard error is ≈ 0.08 and the effect
detectable with 80 % power at α = 0.05 is ≈ 0.22 before model-draw variance,
which the two-way bootstrap of Amendment 1 adds back, so the real detectable
effect is somewhat larger. F1's +0.406 was against a
much weaker control and was the reason for the hypothesis, so the effect
against CH9 is expected to be smaller. **INCONCLUSIVE is a realistic outcome**,
and will be reported as such.

## 9. What each outcome does to the manuscript

| H-F2-1 | Manuscript |
|---|---|
| SUPPORTED | Alignment stated as a confirmed, out-of-sample result against a control matched on density, topology, within-service structure and degree |
| INCONCLUSIVE | Alignment remains an F1 observation; the text says a matched confirmatory test did not resolve it |
| REFUTED | The alignment interpretation is withdrawn; what F1's contrast measured is restated as P2's structure, not its channel semantics |

## Amendment 1 — 17 November 2026, before any F2 data, after the F1 development run

The development run on F1 (Sec. 7) was made with models freshly trained by the
registered code, same seeds and settings as the published F1 analysis, but under
torch 2.15.0 instead of 2.14.1. The structure-free score reproduced the
published F1 rankings in 96 of 96 (seed, fault) pairs; every learned model did
not (`spcrl_soft` with P1: 28 of 96 identical ranks). Same code, same seeds, a
different numerical environment: a different set of trained models, and
contrasts that moved by up to 0.15 Top-1. The trained model is therefore itself
a random draw whose variance three seeds neither average out nor measure. Two
changes follow, both before any F2 data:

1. **Ten training seeds (0–9) instead of three.** Seeds 0–2 are the models
   frozen above; seeds 3–9 were trained the same way, in the same environment,
   and are frozen with them (hashes in Sec. 10).
2. **Two-way bootstrap.** Cells and training seeds are resampled independently
   (`analysis/f2_confirmatory.py`). A bootstrap over cells alone treats the
   model as fixed.

Hypotheses, controls, decision rules, the primary cell set and the campaign are
unchanged. The F1 development results are exploratory and are reported in the
manuscript as such, whatever F2 shows.

**What the F1 development run showed (exploratory, 16 cells × 10 seeds,
two-way bootstrap):** P2 vs CH9 under the soft prior +0.044 [−0.080, +0.173];
dose–response slope −0.035 [−0.160, +0.093], Top-1 0.378 → 0.334 from 0 % to
93 % of P2's relations moved; P2 vs P3b +0.263 [+0.081, +0.463]; P2 vs CH9
under the strict hard mask +0.162 [+0.054, +0.293]; P2 vs WID −0.053
[−0.169, +0.076]. Read together: on F1, what the semantic prior adds over a
density-matched random prior is its structure, not which channel each relation
is attached to — except under a hard mask. That reading was formed on F1 and is
not a result. Two secondary hypotheses test it on F2; the primary hypothesis is
**not** changed, because changing it after seeing development data is the
choice this protocol exists to prevent.

**S6 (structure of the semantic prior).** On the new cells,
  Top-1(P2) − Top-1(P3b) > 0 under the soft prior. SUPPORTED if the CI lower
  bound > 0; REFUTED if the upper bound < 0.

**S7 (channel semantics under exclusion).** On the new cells,
  Top-1(P2) − E[Top-1(CH9)] > 0 under `spcrl_hard_strict`. SUPPORTED if the
  CI lower bound > 0; REFUTED if the upper bound < 0.

Secondary hypotheses are not corrected for multiplicity and are reported as
secondary.

## 10. Frozen artifacts

Everything the F2 analysis depends on, hashed before F2 existed. The analysis
checks the priors and the models against these values and refuses to run
if any differs.

| Artifact | SHA-256 |
|---|---|
| `docs/preregistration/F2_PRIORS_MANIFEST.json` (61 priors, per-matrix hashes inside) | `16abe56dd82577035dc2005765b81f46b156454133f9a5a58ba0951a33bc9783` |
| `data/f2_frozen/models/seed0_hard.pt` | `f85f8f6148650c490f369f0158b209cb5cc917082b3b2e660055e4ca698c25d9` |
| `data/f2_frozen/models/seed0_none.pt` | `02acbf9cfc29a6786db2e7cc85ea40fa575302db1f4a37adaa25127d2e7c63f4` |
| `data/f2_frozen/models/seed0_soft.pt` | `b707549087b6eb90c1f21ccebbb8f2a826b0d4cf709db3f46c1ad657f002c8d1` |
| `data/f2_frozen/models/seed1_hard.pt` | `e71ca41f9c3d5ce27d143e44d6a1b0d0c9faa84600421bfb9c302d598d1fc6a2` |
| `data/f2_frozen/models/seed1_none.pt` | `a221dc1a1239b11f04d6e3432be670e6b48e60e72f1f4a12aa6c491b55ef3d1c` |
| `data/f2_frozen/models/seed1_soft.pt` | `4376676d12c1d1d124bea4c519c4e50c44859d56c44ec9ce37ed5fbfb96fb775` |
| `data/f2_frozen/models/seed2_hard.pt` | `782a4d0fa087d48e6e076a18eb615f388d3ed3ba4435b5f6386f2319d9971685` |
| `data/f2_frozen/models/seed2_none.pt` | `06fcd943ad6f2113f901bf57cfa50f070010164dd591ece0aaf7b0517c88aebf` |
| `data/f2_frozen/models/seed2_soft.pt` | `a1728aa2c8138a4ecbdc1d2c541381e72a122affd925d986dd7e2dd25bb8fd1b` |
| `data/f2_frozen/models/seed3_hard.pt` | `d09d7f7d0bb3b6cc5f9cd7a26a851fac739eb028320faba119e990d625c5ec5f` |
| `data/f2_frozen/models/seed3_none.pt` | `0058a7572776f42ba356c7d2d188d18c536b9bacdfea6ba05dabc077a6aeb154` |
| `data/f2_frozen/models/seed3_soft.pt` | `533647f7b9584e4bb2b311aee752b77cd538759bc14ce07daa2b4d2d3fee1e12` |
| `data/f2_frozen/models/seed4_hard.pt` | `32b6d153772712baeba3987e72543f54e29be804fa95bb72f8e63d45adeb5fac` |
| `data/f2_frozen/models/seed4_none.pt` | `7e6319dcdf653192f390c0003513278eee738949f07b368ebcd9e58bd9d3d856` |
| `data/f2_frozen/models/seed4_soft.pt` | `641f51ddb6967cf32975fc016bfd9ed19bb16ac9e03cc9d2222a9e43c95a1f9f` |
| `data/f2_frozen/models/seed5_hard.pt` | `eb158e49d16ef3e48d4e920ebd9cd2e3af3da963c4c7298c6da0a55d6ef9beaf` |
| `data/f2_frozen/models/seed5_none.pt` | `64547f2ce6ad86bd705cbb4d607ef1da1c55a694bad22c8d7dc4dde248532039` |
| `data/f2_frozen/models/seed5_soft.pt` | `0a1e296e1d9b685212162eaf09c8110cd0225f00c77055c1d8f21a5b887a8b7f` |
| `data/f2_frozen/models/seed6_hard.pt` | `e51e12a2ad54838140b6290fb4869b811f9cb0f560ea46be4648cba4a6b0de57` |
| `data/f2_frozen/models/seed6_none.pt` | `a08260553d30d4124cd1cb602e30cef21ff6941d440174119785fa979ec54028` |
| `data/f2_frozen/models/seed6_soft.pt` | `872cb5e0dcaddf2f750da61f3cb6fca6830b8d24b5d5aa41bfff0eeb63abafdc` |
| `data/f2_frozen/models/seed7_hard.pt` | `af2a56aa628c7f18f3eb8236a1fe1d8cbfa423ddc1046ce0094948b5d222ff82` |
| `data/f2_frozen/models/seed7_none.pt` | `4665c5b5d8c17d3d265548e3302244a2644fd4d151b12c732287b5bbc8794176` |
| `data/f2_frozen/models/seed7_soft.pt` | `16f1707399355a80633252048331a97ba2e5e5d0859e4735b5a028c42cbdb9be` |
| `data/f2_frozen/models/seed8_hard.pt` | `223611ad4699652699f83794d6bad8243bc1ca469751bdfac42011c33f66b975` |
| `data/f2_frozen/models/seed8_none.pt` | `c126ed5b9a31ae0812af07a694d644dac4267c42f86a9bf4cfe13bd0bef965e5` |
| `data/f2_frozen/models/seed8_soft.pt` | `409b78f108c8fe29390285b7f1d40b440075967e76b59d5850b8298984e3b737` |
| `data/f2_frozen/models/seed9_hard.pt` | `ab737780ee5c9a749999be90934bcf441d1f5400403d545eebbef9ac10c86de3` |
| `data/f2_frozen/models/seed9_none.pt` | `0a56f81780e71799b6b37880e85e17e9869a5a13e0e7dec415f60a4aa7ccbbaf` |
| `data/f2_frozen/models/seed9_soft.pt` | `f9a12dec09951d2421b25be30cfc9769f27dfcf01cad296ebecce611f9e2d08b` |
| `spcrl/priors/alignment.py` | `e75bd389a6cc6fdef2a13bf0a2e6707dc2865e3dc5423eedeab79bb96aa6c4b4` |
| `spcrl/priors/semantic.py` | `06b12096e80f6847968eb511a9a1b948f8beef35932e5b14514e895176979300` |
| `analysis/build_f2_priors.py` | `b610cd6e22bcda7a0f9b8a5db28f1033a42722e442376d2a66d1b3e70e7f49d5` |
| `experiments/e19_alignment.py` | `493f0e910af9f6f5c80f6e08361c1f2b21fa87cc5177ec06281b3669157b3567` |
| `experiments/e10_real.py` | `a4195e01002b4a2b4a7eedc334ccaf06b8a4e6badd96a44c414bf3446a0d90fc` |
| `analysis/f2_confirmatory.py` | `645d878cd884e2e1138adee99204fabc24b1ae2aeb962a288a8015c08eecc46a` |
| `analysis/external_gates.py` | `899bd40682291b4754e1df36daa00d373d17c830d0f79b52e31f4cfa5fdd9b6b` |
| `spcrl/benchmarks/real.py` | `2b885abea4d51841ad7d3cf6f24aba46ad06342c7865cf177b42a4029f11821b` |
| `spcrl/counterfactual/surrogate.py` | `8485c88c44badb54e015256de0e97218d7423a0ce702e8a170bff7bf231cf6b6` |
| `spcrl/priors/features.py` | `f269844be468ee8830712d5a985d7ee27786a1a84169b6012e19f850285fff15` |
| `spcrl/causal/amortized.py` | `15aa6f83fd8c045eab8f770070ef01bf035d2dce96aee46605ec191f4f879152` |
| `benchmarks/telecom/scripts/campaign_f2.py` | `c8f9160c0e22afb2c63e7b29d91c8c020f2ea53b438dbc5fc588777fc2357cb3` |
| `benchmarks/telecom/scripts/inject_f2.sh` | `33e37177cb5f11f451465e0dcae19ea89c6822243affb8c7a560a296e542c301` |
| `benchmarks/telecom/scripts/smoke_test_f2.sh` | `7e3c072307a7ec8a89e860321698f52e6f14f13fa79435bbaee3a865a9bf6591` |
| `benchmarks/telecom/f1_topology.yaml` | `66c3b8f69de5816596930d099ec69c4a73e44abe042d38d921d0334dcfbdd776` |

The models were trained (seeds 0–9; 400 synthetic training and 60 validation
systems, prior corruption 0.2, 40 epochs; torch 2.15.0+cpu) by
`e19_alignment.py` with the settings E10 used for F1, and are never retrained
for F2 (`--no-train`).
