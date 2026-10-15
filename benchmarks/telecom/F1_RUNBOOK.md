# F1 runbook: free5GC control plane on Docker Desktop

Pre-registered in `docs/preregistration/PREREGISTRATION_EXTERNAL.md`. Read the "validity gates"
there before you start: a campaign that fails a gate is reported as failed, and
the smoke test below exists so that you find out in ten minutes, not after
eight hours.

What is **verified** in this repository: the campaign runner, the collector's
compose mode, the prior and its controls, and the analysis. What **depends on
your machine and your free5gc-compose version** and is therefore checked by the
smoke test rather than assumed: container and service names, whether `tc` exists
in the NF images, whether cAdvisor on Docker Desktop labels containers by name,
the gNB–AMF link, and subscriber provisioning.

Estimated wall clock: set-up 1–2 h the first time; campaign
38 runs × 11 min ≈ **7 h**; collection 10 min.

## 1. Bring up the core, the monitoring, and UERANSIM

```bash
cd ~/free5gc-compose                                  # your checkout
cp ~/spcrl/benchmarks/telecom/compose/*.y*ml .
docker compose -f docker-compose.yaml -f docker-compose.spcrl.yaml config >/dev/null \
  && echo "overlay OK"                                # fix service names if not
docker compose -f docker-compose.yaml -f docker-compose.spcrl.yaml up -d
docker ps --format '{{.Names}}'                       # note the real names
```

If the names printed differ from `amf, ausf, udm, udr, nssf, pcf, smf, nrf,
mongodb, ueransim`, edit the `container:` fields in
`~/spcrl/benchmarks/telecom/f1_topology.yaml` (not `depends_on`).

If an NF image has no `tc` (`docker exec amf which tc` prints nothing), the SBI
latency fault cannot run in that container. Tell me which ones; the fallback is a
netem sidecar sharing the NF's network namespace, which I would add as a dated
amendment before any data is taken.

## 2. The gNB

free5gc-compose starts the gNB inside the `ueransim` container. Check:

```bash
docker logs ueransim 2>&1 | grep -i "ng setup" | tail -3     # expect: NG Setup procedure is successful
```

If NG Setup fails on Docker Desktop, it is almost always SCTP. Stop there and
send me the log lines.

## 3. Subscribers

The workload needs **40 subscribers with consecutive IMSIs**, starting from the
IMSI in the UE config the compose file ships (commonly `imsi-208930000000001`;
check `ueransim/config/uecfg.yaml`, field `supi`), with the same key, OPc and
slice as that config.

Use the web console at http://localhost:5000 (default login in the free5GC
guide "Create Subscriber via webconsole"). If your console version offers a
field for the number of subscribers to create, set it to 40; otherwise the
console's REST route is documented in free5gc/free5gc issue #539. I could not
verify that route from here, so I have not scripted it — a script that silently
provisions nothing would fail at the smoke test anyway, but I would rather not
hand you one.

## 4. Start the workload

```bash
docker cp ~/spcrl/benchmarks/telecom/scripts/workload_ueransim.sh ueransim:/workload.sh
docker exec -d ueransim sh -c \
  'bash /workload.sh /ueransim/config/uecfg.yaml 208930000000001 40 2 20 > /tmp/workload.log 2>&1'
sleep 90; docker exec ueransim tail -5 /tmp/workload.log   # expect registered=k lines
```

(Adjust the config path and first IMSI to what step 3 found.)

## 5. Smoke test — must pass

```bash
cd ~/spcrl/benchmarks/telecom
chmod +x scripts/*.sh
./scripts/smoke_test_f1.sh http://localhost:9090
```

It stops at the first failed assumption and says which one. Send me its output
if it fails.

## 6. Campaign

```bash
cd ~/spcrl/benchmarks/telecom
python3 scripts/campaign_compose.py --dry-run          # 38 runs, ~7 h
nohup python3 scripts/campaign_compose.py > f1_campaign.out 2>&1 &
tail -f f1_campaign.out                                # Ctrl-C only stops the tail
```

Keep the machine awake (`caffeinate -dims` in another terminal on macOS).

## 7. Collect — immediately after the campaign

```bash
cd ~/spcrl
python3 benchmarks/microservices/scripts/collect.py \
  --topology benchmarks/telecom/f1_topology.yaml --runtime compose \
  --campaign benchmarks/telecom/data/f1/campaign.jsonl \
  --out data/f1 --prometheus http://localhost:9090 --step 1
tar czf spcrl_f1_data.tar.gz data/f1
```

Send me `spcrl_f1_data.tar.gz`, `f1_campaign.out` and the smoke-test output.
