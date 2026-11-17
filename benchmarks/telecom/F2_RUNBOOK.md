# F2 runbook — the confirmatory free5GC campaign

Pre-registered in `docs/preregistration/PREREGISTRATION_F2.md`. `mac$` runs on
your MacBook, `vm$` inside the Ubuntu VM where F1 ran. Copy a block, paste it,
read what it prints. Stop at the first thing that does not look like the
expected output and send it to me — do not improvise around a failed check.

Time: ~30 min of checks, then ~8.4 h unattended, then 10 min of collection.

---

## Step 0 (Mac) — publish the pre-registration BEFORE any F2 data exist

This is what makes F2 confirmatory: a public, dated commit and a Zenodo DOI for
the exact protocol, priors, models and code, made before the campaign starts.

```bash
mac$ cd ~/"Science Article 2025-/Structural Priors Enable Generalizable Causal Diagnosis and Counterfactual Recovery in Dynamic Systems/spcrl-public"
mac$ git add -A
mac$ git status                      # only code/protocol files; no manuscript
mac$ git commit -m "F2 pre-registration: protocol, priors manifest, frozen models, campaign harness"
mac$ git tag -a f2-prereg -m "F2 pre-registration, frozen before any F2 data"
mac$ git push && git push origin f2-prereg
```

Then on GitHub: **Releases → Draft a new release → choose tag `f2-prereg` →
title "F2 pre-registration" → Publish**. Zenodo archives it automatically and
issues a DOI. Send me the DOI; it goes into the paper.

## Step 1 (VM) — state of the machine

```bash
vm$ uname -r; timedatectl | grep -i synchronized; nproc; free -h | head -2; df -h ~ | tail -1
vm$ docker ps --format '{{.Names}}\t{{.Status}}' | sort
vm$ lsmod | grep gtp5g
vm$ ls ~/telecom/scripts ~/microservices/scripts 2>&1 | head -30
```

Expected: kernel 6.x, clock synchronized, ≥4 CPUs, the free5GC containers
(amf ausf udm udr smf pcf nssf nrf mongodb upf ueransim prometheus cadvisor …)
`Up`, a `gtp5g` line, and the F1 harness folders.

If the containers are not up, start the core exactly as for F1:

```bash
vm$ cd ~/free5gc-compose
vm$ docker compose -f docker-compose.yaml -f docker-compose.spcrl.yaml up -d
vm$ sleep 60; docker ps --format '{{.Names}}\t{{.Status}}' | sort
```

## Step 2 (VM) — fetch the pre-registered code

```bash
vm$ cd ~ && rm -rf spcrl-f2 && git clone --branch f2-prereg --depth 1 https://github.com/Dev-Nikita/spcrl spcrl-f2
vm$ cd ~/spcrl-f2 && git log -1 --format='%H %ci'          # send me this line
vm$ cp ~/spcrl-f2/benchmarks/telecom/scripts/{campaign_f2.py,inject_f2.sh,smoke_test_f2.sh} ~/telecom/scripts/
vm$ chmod +x ~/telecom/scripts/*.sh
vm$ diff <(python3 -c "import yaml;[print(s['name'],sorted(s['depends_on'])) for s in yaml.safe_load(open('$HOME/telecom/f1_topology.yaml'))['services']]") \
         <(python3 -c "import yaml;[print(s['name'],sorted(s['depends_on'])) for s in yaml.safe_load(open('$HOME/spcrl-f2/benchmarks/telecom/f1_topology.yaml'))['services']]") \
     && echo "depends_on identical to the registered topology"
```

The last line must print `depends_on identical …`. Container names may differ
from the repository (they were adjusted for F1); the call graph may not.

## Step 3 (VM) — the workload, exactly as in F1

The workload must be the stabilised two-gNB workload F1 used (amendment of
13 October 2026). Show me what is running and the script itself:

```bash
vm$ docker exec ueransim sh -c 'ps -eo pid,etime,args | grep -E "nr-gnb|nr-ue|workload" | grep -v grep'
vm$ docker exec ueransim sh -c 'ls -la /workload*.sh; sha256sum /workload*.sh'
vm$ docker exec ueransim sh -c 'cat /workload*.sh' > ~/f2_workload_script.txt; wc -l ~/f2_workload_script.txt
```

If the workload is not running, start it with **the same command you used for
F1** (it is in your shell history: `history | grep workload`). Then let the
system run with the workload and **no faults for at least 15 minutes** — the
calibration reads the last 10 minutes of nominal operation.

Before the campaign: close everything else on the host, disable sleep and
automatic updates for the night (host load reaches every container at once and
looks like a shared driver).

## Step 4 (VM) — smoke test, must pass

```bash
vm$ cd ~/telecom && ./scripts/smoke_test_f2.sh http://localhost:9090 2>&1 | tee ~/f2_smoke.out
```

It runs the F1 checks, then the F2 ones (loss, rate and CPU-throttle injected
and removed on `ausf`, UEs still registering under 10 % loss, calibration
series present). It ends with `F2 SMOKE TEST PASSED`. **If it fails anywhere,
stop and send me `~/f2_smoke.out`.** Do not start on a failed smoke test; a fix
made now is a dated amendment before data, which is fine — a fix made after is
not.

## Step 5 (VM) — the campaign (~8.4 h)

```bash
vm$ sudo apt -y install tmux >/dev/null 2>&1; tmux new -s f2
vm$ cd ~/telecom && python3 scripts/campaign_f2.py --dry-run          # "46 runs (40 faults: 24 in new families)"
vm$ python3 scripts/campaign_f2.py --prometheus http://localhost:9090 2>&1 | tee ~/f2_campaign.out
```

It first prints the calibration (throttle and rate limit per function) and then
one line per run. Detach with `Ctrl-b d`; check any time with
`tail -5 ~/f2_campaign.out`; reattach with `tmux attach -t f2`.

If it dies: send me `~/f2_campaign.out`. The rule is a restart from scratch with
the same seed (the runner archives the partial log itself) — do not edit the log.

## Step 6 (VM) — collect, straight after it finishes

```bash
vm$ cd ~ && python3 microservices/scripts/collect.py \
       --topology telecom/f1_topology.yaml --runtime compose \
       --campaign telecom/data/f2/campaign.jsonl \
       --out data/f2 --prometheus http://localhost:9090 --step 1 2>&1 | tee ~/f2_collect.out
vm$ cp telecom/data/f2/calibration.json data/f2/
vm$ tar czf ~/spcrl_f2_data.tar.gz data/f2 telecom/data/f2 f2_campaign.out f2_collect.out f2_smoke.out f2_workload_script.txt
vm$ ls -la ~/spcrl_f2_data.tar.gz; sha256sum ~/spcrl_f2_data.tar.gz
```

The collector's last line should say that all windows contain telemetry.

## Step 7 — bring the archive to the Mac

Whichever way you used for F1 (scp, shared folder, or uploading from the VM's
browser). With scp, from the Mac:

```bash
mac$ cd ~/"Science Article 2025-/Structural Priors Enable Generalizable Causal Diagnosis and Counterfactual Recovery in Dynamic Systems"
mac$ mkdir -p f2_campaign_archives && scp <vm>:~/spcrl_f2_data.tar.gz f2_campaign_archives/
```

Then tell me it is there. I run the gates first; a failed gate means F2 is
reported as failed and no model is run. Only then the registered analysis
(`experiments/e19_alignment.py --no-train`, `analysis/f2_confirmatory.py`).

Keep the VM as it is until I confirm the gates passed.
