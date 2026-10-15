# SP-CRL — Structural-Prior Causal Representation Learning

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23402003.svg)](https://doi.org/10.5281/zenodo.23402003)

Research code, pre-registrations and raw records for a study of how a learner
should use a **fallible structural description** of the system it is modelling.

The question is not whether structural knowledge helps a causal learner — with a
correct description it obviously does — but **how uncertain structural knowledge
should enter the learner, at what level of error it stops helping, and whether
the description a real deployment publishes about itself is that kind of
knowledge at all.**

The manuscript reporting these results is under review; its sources are not part
of this repository. Everything needed to rerun the experiments and to recompute
every reported number is.

## What is here

```
spcrl/            the library
  benchmarks/     synthetic SCM generator, dataset pools, real-testbed loader
  priors/         pairwise/node features, semantic prior variants (P1–P6);
                  the prior block is separable by design
  causal/         per-system and amortised learners, with trainers
  counterfactual/ surrogate SCM, root-cause scoring, interventional rollout
  eval/           SHD / precision / recall / orientation-F1 / Top-k / MRR
  baselines/      PC (Fisher-z), NOTEARS-linear, partial correlation
experiments/      e1–e18, one file per experiment, all resumable
analysis/         loaders, statistics, figures, tables, claim ledger,
                  external validity gates and prior controls
scripts/          campaign drivers (K1 controls, K2b, F1)
benchmarks/
  microservices/  Kubernetes testbed: Go service, topology, manifests, fault
                  campaign, collector, manipulation pre-check, runbook
  telecom/        free5GC testbed: topology, compose overlay, UERANSIM
                  workload, fault campaign, smoke test, runbook
tests/            leakage, determinism, prior-blindness, metric correctness
docs/
  preregistration/  the pre-registrations, with dated amendments
  CLAIM_LEDGER.csv  every quoted number, recomputed from the raw artefacts
data/
  external/       the external campaigns' telemetry, campaign logs, structural
                  priors and diagnostics, including the campaign that failed
                  its own manipulation check and is reported as failed
  result_records.tar.gz  one JSON object per model fit and per evaluated fault
```

## Reproducing the synthetic results

```bash
pip install torch numpy scipy scikit-learn pandas matplotlib pyyaml pytest
export PYTHONPATH=$PWD

pytest tests/ -q                                       # integrity tests

python experiments/e1_main.py       --seeds 0 1 2      # recovery, IID + 5 shifts
python experiments/e2_corruption.py --seeds 0 1 2      # prior corruption
python experiments/e3_priortypes.py --seeds 0 1 2      # which prior components help
python experiments/e4_downstream.py --seeds 0 1 2      # diagnosis + repair
python experiments/e5_scaling.py    --seeds 0 1 2      # transfer to larger systems

python analysis/make_tables.py
python analysis/claim_ledger.py                        # recompute every number
```

Every result row records run id, seed, graph id, size, family, mechanism,
regime, prior corruption, prior type, fit time and all metrics; each results
file carries a `.meta.json` with the git commit, working-tree cleanliness and
library versions. `analysis.load.load` keeps only rows from the run named in the
metadata, so a results file reused by an aborted run cannot silently contaminate
a table.

## Reproducing the testbed results

The Kubernetes and free5GC campaigns are not run from the test suite; they need
a cluster and, for free5GC, a dedicated Linux host with the `gtp5g` kernel
module. See `benchmarks/microservices/` and `benchmarks/telecom/`. Both produce
the same on-disk layout (`prior.npy`, `variables.json`, `normal/`, `faults/`)
that `spcrl.benchmarks.real` consumes, so no model code changes between the
synthetic and the real environments.

The campaigns were pre-registered before their data existed
(`docs/preregistration/`), with hypotheses, decision rules and four validity
gates fixed in advance. The gates are in `analysis/external_gates.py`; the
registered prior controls are in `analysis/external_controls.py`.

## Design decisions worth knowing before reading the code

* **The prior is a superset, not the answer.** In the synthetic benchmark the
  admissible set is about six times the true edge set.
* **`mode="none"` is exactly prior-blind.** The prior occupies a contiguous
  feature block that is zeroed, and the attention bias is zeroed too; a unit
  test asserts that predictions are identical under a randomised prior.
* **Splits are at the system level.** A unit test asserts that no ground-truth
  adjacency matrix appears in two pools.
* **Interventions are scored against the true SCM**, re-simulated under each
  candidate action — never against the model that chose the action.
* **Ties are broken at random.** The true root cause is the first candidate, so
  order-based tie-breaking would silently reward uninformative predictions.

## Limitations

The amortised model is trained on systems whose causal graphs are known, so it
is not comparable to unsupervised discovery algorithms; the prior-free amortised
model is the control that carries the claims. No identifiability result is
claimed. The real testbeds have ground-truth *root causes* but no ground-truth
causal graph, so structure-recovery metrics are reported on synthetic systems
only. The raw telemetry of the first Kubernetes campaign was lost to an
infrastructure failure after its analysis; its result records, campaign log,
structural prior and diagnostics are here, its raw telemetry is not.

## Archived version

Release 0.1 is archived on Zenodo: <https://doi.org/10.5281/zenodo.23402003>.
That DOI is the one cited in the manuscript's Data and Code availability
statements.

## License

MIT, see `LICENSE`. If you use the code or the data, please cite it as in
`CITATION.cff`.
