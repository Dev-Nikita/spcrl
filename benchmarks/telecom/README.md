# Cloud-native telecom validation environment (free5GC)

This is the third validation domain, not the contribution. What it has to show
is narrow and specific: that the soft/hard/none ordering measured on synthetic
systems still holds when the structural prior is a real deployment description
and the faults are injected into a real 5G core.

## Read this before you start: what will and will not work

**The free5GC UPF requires the `gtp5g` Linux kernel module** — the project
states it explicitly, and pins the supported version (0.9.5 at the time of
writing). `gtp5g` is an out-of-tree module that must be built against the
running kernel's headers.

Consequences:

| Host | Control plane (NRF, AMF, AUSF, UDM, UDR, PCF, NSSF, SMF, MongoDB) | UPF / user plane (N3, GTP-U) |
|---|---|---|
| Docker Desktop on macOS | works | **does not work** — the LinuxKit VM's kernel has no `gtp5g` and you cannot practically insmod it |
| Linux host / VM with kernel headers (Ubuntu 22.04, kernel 5.15) | works | works after building `gtp5g` |

So on a Mac you can run **Option B** below and nothing more. That is still a
usable experiment — the dependency structure being tested is the
service-based interface graph between network functions, which is entirely in
the control plane — but the paper must then say "control-plane validation" and
must not claim user-plane fault propagation.

Also check the image architecture before you spend an evening on it:

```bash
docker manifest inspect free5gc/upf:latest | grep -A2 architecture
docker manifest inspect free5gc/amf:latest | grep -A2 architecture
```

If only `amd64` is published and you are on Apple Silicon, everything runs
under emulation: expect it to be slow, and expect MongoDB to be the first
thing that breaks (free5GC's own notes warn about MongoDB > 4.4 and AVX).

## Telemetry: free5GC has no Prometheus endpoints

The network functions do not export metrics. Per-NF observables therefore come
from two sources, and the collector treats them identically:

* **cAdvisor** — CPU seconds, working-set memory, network bytes and errors per
  container;
* **`probe/probe.py`** — an SBI prober that calls each NF's service-based
  interface at a fixed rate and exports latency histograms, error counts and
  last status code. Standard library only; runs on `python:3.12-alpine` with no
  build step.

That gives roughly six signals per NF, so a nine-NF core yields ~54 variables —
the same order as the microservice testbed.

## Option A (recommended first): do the microservice testbed instead

`benchmarks/microservices/run_local.sh` runs end to end on Docker Desktop with
no kernel modules and no emulation. It answers the same question — does the
ordering survive on a real distributed system — for a fraction of the effort.
Do it first; if the ordering does not reproduce there, free5GC will not save it.

## Option B: free5GC control plane on Docker Desktop

```bash
git clone https://github.com/free5gc/free5gc-compose.git
cd free5gc-compose
docker compose pull

# bring up only the control plane and the database
docker compose up -d db nrf amf ausf nssf pcf smf udm udr webui

# add the probe (adjust the SBI ports to your compose file)
docker run -d --name spcrl-probe --network free5gc-compose_privnet \
  -v "$PWD/../spcrl/benchmarks/telecom/probe:/probe" -p 9110:9110 \
  python:3.12-alpine python3 /probe/probe.py --port 9110 --interval 1 \
  --targets nrf=http://nrf:8000/nnrf-nfm/v1/nf-instances \
            amf=http://amf:8000/namf-comm/v1/subscriptions \
            ausf=http://ausf:8000/nausf-auth/v1/ue-authentications \
            udm=http://udm:8000/nudm-sdm/v2/shared-data \
            udr=http://udr:8000/nudr-dr/v2/subscription-data \
            pcf=http://pcf:8000/npcf-am-policy-control/v1/policies \
            nssf=http://nssf:8000/nnssf-nsselection/v2/network-slice-information \
            smf=http://smf:8000/nsmf-pdusession/v1/sm-contexts
```

A 4xx from an NF is fine and is counted as a successful probe: it proves the
function answered. Only 5xx and timeouts count as errors.

Fault injection uses `scripts/inject.sh`, which needs `NET_ADMIN` on the
target container — add to each service in your compose override:

```yaml
    cap_add: [NET_ADMIN]
```

## Option C: full free5GC including the user plane

Needs a Linux host or VM whose kernel headers you control:

```bash
sudo apt install -y linux-headers-$(uname -r) build-essential
git clone -b v0.9.5 https://github.com/free5gc/gtp5g.git
cd gtp5g && make && sudo make install && sudo modprobe gtp5g && lsmod | grep gtp5g
```

Then run the full `docker compose up` and add UERANSIM for gNB/UE simulation.
Only in this configuration are UPF latency and packet-loss faults meaningful.

## Fault classes

| class | injection | affected NF | needs |
|---|---|---|---|
| SBI latency | `inject.sh amf sbi-delay 200ms` | any control-plane NF | NET_ADMIN |
| discovery degradation | `inject.sh nrf sbi-delay 300ms` | NRF | NET_ADMIN |
| control-plane overload | `inject.sh amf cpu 2` | any NF | — |
| NF restart | `inject.sh udm crash` | any NF | — |
| database latency | `inject.sh db sbi-delay 150ms` | MongoDB | NET_ADMIN |
| user-plane latency | `inject.sh upf netem 120ms` | UPF | Option C only |
| packet loss | `inject.sh upf loss 3%` | UPF | Option C only |

## Collecting

The campaign runner and collector are shared with the microservice testbed;
only the topology file differs.

```bash
python3 ../microservices/scripts/campaign.py \
    --topology k8s/topology.yaml --namespace free5gc \
    --log data/telecom/campaign.jsonl \
    --warmup 90 --duration 150 --repeats 1

python3 ../microservices/scripts/collect.py \
    --topology k8s/topology.yaml --campaign data/telecom/campaign.jsonl \
    --out data/telecom --prometheus http://localhost:9090 \
    --signals latency_p95 error_rate throughput cpu memory
```

`data/telecom/` then has the same layout as the microservice and synthetic
data and is consumed by `spcrl.benchmarks.real` and `experiments/e10_real.py`.

## What the paper may claim from this

Root-cause identification at the network-function level, and whether the
soft/hard/none ordering survives when the deployment description is stale or
wrong. Not causal-structure recovery — there is no ground-truth causal graph
for a real system — and, under Option B, not user-plane behaviour.
