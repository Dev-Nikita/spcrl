# K2b runbook: the Kubernetes campaign with the shared load driver broken

Pre-registered in `docs/preregistration/PREREGISTRATION_EXTERNAL.md`, Amendment 1.

**What happened to K2.** The first attempt (`LOAD_MODE=independent`) gave every
service its own request stream, and that was not enough: services that do not
call each other still correlated at mean |ρ| 0.756, because a service that calls
two others propagates its own stream to both. The gate threshold is 0.5, so that
campaign is reported as failed and is not analysed. Nothing about it was wrong
on your side — the design was.

**What K2b changes.** One thing: the request rate now falls with depth from the
entry point (gateway gets 1/s, its callees 4/s, everything deeper 12/s), and the
streams run on three different periods. Traffic a service receives directly then
dominates traffic its callers pass on. Campaign length, faults, magnitudes,
scrape interval and topology are unchanged, so K1 and K2b stay comparable.

**Why the first K2b attempt measured nothing.** In your run the new load
generator never started: `loadgen-78c7fdd965-p5t5f 0/1 Pending`, while the K2
pod from the previous regime (`loadgen-6c88c84459-gl5xd`, 5h47m old) kept
running. Kubernetes reported the namespace as available because that old pod
was serving, so the pre-check sampled K2 traffic and returned 0.798 — close to
K2's own 0.756, which is the giveaway. That number is void, not a K2b result.
The harness now uses a `Recreate` rollout (the old generator is stopped before
the new one starts), stamps the regime into the pod, fails the deploy if the
rollout does not complete, and refuses to run the pre-check or the campaign
unless the pod actually running is the one you asked for.

**Run the ten-minute check before the campaign.** It measures the same gate on
nominal traffic. If it fails, stop — do not spend five hours on a campaign that
cannot be analysed.

```bash
tar xzf spcrl_testbed_v4.tar.gz && cd microservices

# 1. redeploy with depth-scaled independent load; this now stops the old
#    generator first and fails loudly if the new one cannot schedule
LOAD_MODE=decorrelated ./run_local.sh deploy

# 2. in a SECOND terminal, for everything below
./run_local.sh forward

# 3. back in the first terminal
./run_local.sh status                 # service targets 'up'
./run_local.sh precheck 10            # ~10 min; prints G4 PASS or G4 FAIL
```

`deploy` and `precheck` both print `load generator running: decorrelated`.
If instead they stop with an error about the running generator, the cluster
could not fit the new pod; the message gives the two commands that clear it
(`kubectl -n spcrl delete pod -l app=loadgen --wait=false`, then
`kubectl -n spcrl rollout status deploy/loadgen`). `kubectl top pods -n spcrl`
tells you whether the node is simply full.

`precheck` prints the mean |ρ| over unconnected pairs and the five worst pairs.

- **G4 PASS** → run the campaign (step 4).
- **G4 FAIL and the machine looked saturated** (fans loud, `kubectl top pods`
  near the CPU limits): retry once with lighter load,
  `RATE=8 LOAD_MODE=decorrelated ./run_local.sh deploy`, then `precheck` again.
  CPU contention is itself a shared driver.
- **G4 FAIL otherwise** → send me the precheck output and stop. Per the
  pre-registration we do not redesign a third time; the result is then that this
  testbed cannot be freed of the confound, and free5GC answers the question.

```bash
# 4. the campaign, ~5.5 h, into its own directory
caffeinate -dims &
DATA=data/k2b ./run_local.sh campaign long

# 5. IMMEDIATELY afterwards (Prometheus keeps only 6 h)
DATA=data/k2b ./run_local.sh collect
DATA=data/k2b ./run_local.sh package
```

Send me the tarball `package` prints, the `precheck` output and the line
`collect` prints (`variables=40 ... all windows contain telemetry`). Keep the
K2 data you already produced; it is reported as the failed manipulation.
