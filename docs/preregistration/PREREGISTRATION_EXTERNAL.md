# Pre-registration: the external campaigns K2, K3 and F1, and the semantic prior controls

Written and committed **before any data from K2 or F1 existed**. The repository
state at that moment is tagged `v0.4-pre-external`. Nothing in the model, its
synthetic training, the diagnosis pipeline or its parameters may change between
that tag and the analyses below; the E1–E18 results are frozen.

## Why these campaigns exist

On the Kubernetes deployment (regime K1, §2.10) amortised structure learned from
synthetic data transferred (+0.125 Top-1 over a structure-free anomaly score,
[+0.028, +0.236]), while the deployment's own description made diagnosis worse
(soft prior −0.083, [−0.167, −0.014]). The explanation pre-registered in
`PREREGISTRATION_E16.md` — a latent driver makes the description describe the
wrong relation — was falsified on synthetic data (§2.11). The mechanism is open.

A single deployment cannot separate a property of the method from a property of
that deployment. These campaigns change one thing at a time on the same
deployment (K2, K3) and repeat the question on a second, unrelated system whose
structural description comes from a standard rather than from a manifest (F1).
They are designed so that every outcome is informative, and each hypothesis
below says in advance what each outcome would mean.

## What is held fixed

- The model: the amortised SP-CRL variants of the paper, trained on synthetic
  systems only, exactly as in K1 (`experiments/e10_real.py` defaults: 400
  training systems, training corruption 0.2, three seeds 0/1/2, five variants).
  No training, fine-tuning or calibration on any real data.
- The diagnosis pipeline: per-node parent selection with 3 parents, ridge
  surrogate fitted on nominal windows, scoring restricted to channels with
  nominal variation, services ranked by their maximum standardised deviation.
- The comparisons: every prior variant is compared with the prior-blind model
  (P5) and with the structure-free anomaly score, paired over faults, with 95%
  bootstrap intervals over faults (10,000 resamples) and the three seeds
  averaged within fault.
- Metrics: root-cause Top-1 (primary) and MRR (secondary). No other metric will
  be reported as a primary outcome.

## The regimes

**K1 — shared load (already collected).** One load generator at the entry
point; every service's load is a function of one random walk. Data from the
second campaign of §2.10.

**K2 — independent excitation (new campaign).** Identical deployment, fault set
(8 services × {CPU saturation, injected latency, injected error rate}, magnitude
3, one repeat), timing (300 s warm-up, 300 s fault, 60 s cool-down), six nominal
windows and scrape interval (1 s) as K1. The single change: every service
receives its own request stream following its own random walk
(`render_manifests.py --load-mode independent`). Requests sent to a service
still propagate to everything it calls, so dependence between services should
now follow the call graph rather than one generator.

**K3 — declared driver (analysis of K1, not a campaign).** In K1 the load
generator's rate enters the system only through the entry point, so it is
observed, as `gateway::throughput`. K3 re-analyses the K1 data with the prior
row and column of that channel set to admissible everywhere — the description
now states that the load channel may influence any variable, which is what a
practitioner who knows about the shared driver would write.

**F1 — free5GC control plane (new campaign, second system).** Eight 5G core
network functions (NRF, AMF, AUSF, UDM, UDR, NSSF, PCF, SMF) plus the database,
driven by UERANSIM user equipment performing registration and PDU-session
attempts at a rate that follows a random walk. The structural prior is read from
the service-based architecture — which network function consumes which
function's services during registration and session establishment — encoded in
`benchmarks/telecom/f1_topology.yaml` and fixed with this document. Signals per
network function come from cAdvisor (CPU, memory, bytes received, bytes sent),
because free5GC exports no metrics of its own; the prober of the earlier harness
is **not** used, because a prober calling every function at a fixed rate is
itself a shared driver. Faults: CPU saturation and SBI latency (tc netem) on each
of the eight functions, two repeats: 32 fault windows and six nominal windows,
same timing as K1. Control plane only: the UPF cannot run on Docker Desktop, so
PDU sessions fail at the user-plane step and the paper will claim nothing about
user-plane behaviour.

## The prior controls

Implemented in `spcrl/priors/semantic.py` and tested in
`tests/test_semantic_priors.py`, both committed with this document. Every
variant is built from the deployment description and a fixed seed (20260921),
never from telemetry.

| | prior | what it isolates |
|---|---|---|
| P1 | the deployment description, as collected | — |
| P2 | semantic: only the channel relations a call physically implies, with direction (load flows to the callee; latency and errors flow back to the caller) | whether the *translation* of "A calls B" into channel relations is what fails |
| P3 | random at P1's density | whether P1 carries more than its density |
| P3b | random at P2's density | whether P2 carries more than its sparsity |
| P4 | P1 rewired by double edge swaps, every variable's in- and out-degree preserved | whether P1 carries more than which components are hubs |
| P5 | none (the prior-blind model) | the reference every prior is measured against |
| P6 | the call graph with the non-entry components' identities deranged: right shape, every relation on the wrong component | whether the *identity* of the relations matters, or only the shape |

On K1 these densities are: P1 335 admissible channel pairs (0.215), P2 82
(0.053), P3 325, P3b at P2's density, P4 335, P6 335.

## Validity gates — checked on nominal data before any model is run

A campaign that fails a gate is reported as failed, with the gate it failed, and
no hypothesis is tested on it.

- **G1** every window non-empty (the collector's empty-window check).
- **G2** at least 3,000 nominal samples.
- **G3** the data carry structure at all: median out-of-sample cross-variable R²
  over live channels, excluding throughput-type channels, at least 0.10.
- **G4 (K2 only — the manipulation check)** the shared driver is actually
  broken: the mean |ρ| between the throughputs of service pairs that are *not*
  connected in the call graph falls below 0.5 (it is above 0.99 in K1). If G4
  fails, K2 did not create the condition it exists to create, and nothing about
  K2 is interpreted.

## Hypotheses

Each is stated with its direction, its test, and what each outcome would mean.

**H-X1 (the deployment effect of §2.10 is caused by shared excitation).**
Prediction: the soft-prior deficit against P5 shrinks from K1 to K2.
Test: the difference of the two paired Top-1 contrasts,
(soft − P5)\_K2 − (soft − P5)\_K1, with a bootstrap interval resampling faults
within each campaign independently; supported if the interval lies above zero.
If supported, §2.10 gains a mechanism tested by intervention on the deployment
itself. If not, shared excitation is ruled out as the cause and the paper says
so.

**H-X2 (topology identity carries information).** Prediction: P1 > P6 in K2 and
in F1. Test: paired Top-1 contrast of the soft model under P1 against P6, pooled
over the K2 and F1 faults with campaign as a stratum in the bootstrap. If
refuted, the identity of the relations in a deployment description carries no
information for this learner beyond its shape — a stronger negative result than
§2.10 and one we would report as the headline of the external section.

**H-X3 (semantics carries information beyond sparsity).** Prediction: P2 > P3b
in K2 and in F1, same test as H-X2. If supported, the failure of a call graph
used naively is a *translation* failure, and the description becomes useful when
turned into channel relations. If refuted, being sparser is all P2 does.

**H-X4 (a second system reproduces the transfer of learned structure).**
Prediction: P5 > anomaly score in F1. Test: paired Top-1 contrast on the F1
faults. This is the replication of the positive half of §2.10.

**H-X5 (a declared driver helps).** Prediction: in K3, soft − P5 on K1 data is
higher with the driver declared than with P1. Test: paired over the K1 faults.
Because K1 motivated this analysis, a positive result is reported as suggestive
and not as a test; it becomes a test only if it also holds on K2 with the entry
streams declared, which is the confirmatory version.

**H-X6 (descriptive, no threshold).** Across the campaigns and prior variants,
the Spearman correlation between the compatibility statistic C of
`PREREGISTRATION_E16.md` and the soft − P5 contrast. On synthetic data C
anti-predicted prior usefulness; this reports whether the same holds on real
systems. It supports no method either way.

## Power, stated in advance

K2 has 24 faults and F1 32. With binary Top-1 and three seeds averaged within
fault, the smallest paired effect these campaigns can reliably separate from
zero is of the order of 0.1 Top-1 — the size of the K1 effect. Smaller effects
will be reported with their intervals and not interpreted.

## What will be reported regardless of outcome

All six prior variants on every campaign that passes its gates, with intervals;
every gate result; the K1 prior controls re-run on the fixed code and labelled
exploratory, because K1 is where the question came from; and any deviation from
this document, as a dated amendment below.

## What is explicitly not a test

- Any analysis of K1 (including the prior controls and K3): K1 motivated these
  questions and cannot test them.
- The compatibility statistic on any real campaign (H-X6 is descriptive).

## Amendments

**Amendment 1 (2026-09-24, after the K2 data were collected and gated, before
any model was run on them).** K2 as executed fails gate G4: the throughputs of
services that do not call each other correlate at mean |rho| 0.756, against the
pre-registered threshold of 0.5 (connected pairs 0.910; K1 was above 0.99
throughout). Per this document K2 is therefore reported as a failed campaign
and is not analysed further, and no model is run on it. The threshold is not
changed.

The diagnosis is a property of the call graph rather than of the generator: two
services that do not call each other are still both called by a third, so the
stream their common parent receives propagates to both. Independent per-service
excitation removes the single global generator but not this fan-out.

A replacement campaign **K2b** is registered here, before its data exist. It is
K2 with one change: the per-service request rate falls with depth from the entry
point (entry rate/8, its callees rate/3, everything deeper rate), and the walks
run on three different periods, so that the traffic each service receives
directly dominates the traffic its callers propagate to it
(`render_manifests.py --load-mode decorrelated`). Campaign length, fault set,
magnitudes, scrape interval, topology and prior are unchanged, and K2b inherits
every hypothesis, gate and analysis registered above for K2. Before the campaign
is run, `scripts/precheck_k2.py` measures G4 on ten minutes of nominal traffic;
if it fails, the campaign is not run and the failure is reported. Should K2b
also fail G4, no further redesign will be attempted: the result reported will be
that the shared-driver confound could not be removed from this testbed, and the
K2/K2b question will be answered, if at all, by F1, whose driver is the UE
population rather than a request generator.

**Note on the first K2b pre-check (2026-09-24).** The first run of the ten-minute
pre-check reported mean |ρ| 0.798 over unconnected pairs, but the K2b load
generator was not running at the time: its pod stayed `Pending` while the K2
generator from the previous regime kept running, so the measurement is of K2,
not of K2b, and it is recorded here as void rather than as a K2b gate result.
The harness now refuses to run the pre-check or a campaign unless the generator
pod actually running carries the requested regime.

**Amendment 2 (2026-10-13, before any analysed F1 data existed).** Engineering
validation before the F1 campaign found a reproducible UERANSIM v3.3.0 nr-gnb
segmentation fault under high-frequency UE attach/release churn (exit 139, not
an out-of-memory kill). The scientific design is unchanged — the same core, 40
provisioned subscribers, UE population K = 2..20 on the same bounded random
walk, registration and PDU-session attempts, eight monitored network functions,
CPU-saturation and SBI-latency faults, 32 fault and six nominal windows, scrape
interval, topology, priors and analyses. Only the access-side workload
implementation changes: two independent UERANSIM gNB processes share each
sampled population, and each population is held for 300 s with a 30 s
transition before the next random-walk step, so that the implementation defect
is not repeatedly triggered. Campaigns collected before this amendment are
engineering diagnostics, are marked aborted, and are excluded from analysis.
The campaign ran in a VirtualBox guest (Ubuntu, kernel 6.8) rather than on bare
metal; this is recorded rather than corrected, and reported in Methods.

