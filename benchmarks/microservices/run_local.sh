#!/usr/bin/env bash
# One-shot driver for the Kubernetes testbed on Docker Desktop (or kind/k3s).
#
#   ./run_local.sh build      build the service image into the cluster's docker
#   ./run_local.sh deploy     namespace + services + load generator + Prometheus
#   ./run_local.sh status     is everything up and is Prometheus scraping?
#   ./run_local.sh smoke      inject one fault, verify it shows up in Prometheus
#   ./run_local.sh campaign   run the fault campaign      (quick|long|full)
#   ./run_local.sh collect    Prometheus -> benchmark files
#   ./run_local.sh package    tar the collected data for analysis
#   ./run_local.sh down       delete everything
set -euo pipefail
cd "$(dirname "$0")"
NS=${NS:-spcrl}
IMAGE=${IMAGE:-spcrl/svc:latest}
DATA=${DATA:-data/microservices}
PROM_PORT=${PROM_PORT:-9090}

need() { command -v "$1" >/dev/null || { echo "missing: $1"; exit 1; }; }

case "${1:-}" in
build)
  need docker
  docker build -t "$IMAGE" svc/
  echo "built $IMAGE"
  ;;

deploy)
  need kubectl
  python3 scripts/render_manifests.py --image "$IMAGE" --namespace "$NS" \
      --out k8s/services.yaml --rate "${RATE:-12}" --load-mode "${LOAD_MODE:-shared}"
  kubectl apply -f k8s/services.yaml
  kubectl apply -f k8s/prometheus.yaml
  echo "waiting for pods..."
  kubectl -n "$NS" wait --for=condition=available --timeout=300s deployment --all
  # The load generator is the one deployment whose OLD pod keeps the namespace
  # 'available' while the new one is still Pending, which silently leaves the
  # previous load regime running. Fail loudly instead.
  if ! kubectl -n "$NS" rollout status deploy/loadgen --timeout=300s; then
    echo "ERROR: the load generator did not roll out - the previous regime is"
    echo "still running. 'kubectl -n $NS describe pod -l app=loadgen' will say"
    echo "why (usually Insufficient cpu); free the resources and deploy again."
    exit 1
  fi
  kubectl -n "$NS" get pods
  scripts/check_load_mode.sh "$NS" "${LOAD_MODE:-shared}"
  ;;

status)
  kubectl -n "$NS" get pods -o wide
  echo "--- prometheus targets (needs port-forward, see 'forward') ---"
  curl -s "http://localhost:$PROM_PORT/api/v1/targets" \
    | python3 -c 'import sys,json;d=json.load(sys.stdin)["data"]["activeTargets"];print(len(d),"targets");[print(" ",t["labels"].get("app",t["labels"].get("job")),t["health"]) for t in d]' \
    || echo "no port-forward on :$PROM_PORT — run './run_local.sh forward' in another terminal"
  ;;

forward)
  echo "forwarding prometheus to http://localhost:$PROM_PORT  (leave this running)"
  exec kubectl -n "$NS" port-forward svc/prometheus "$PROM_PORT":9090
  ;;

smoke)
  set -x
  kubectl -n "$NS" exec deploy/loadgen -- curl -s "http://gateway:8080/work"; echo
  kubectl -n "$NS" exec deploy/loadgen -- curl -s "http://payment:8080/fault?latency_ms=300"; echo
  sleep 45
  curl -s --get "http://localhost:$PROM_PORT/api/v1/query" \
       --data-urlencode 'query=histogram_quantile(0.95, sum(rate(svc_latency_seconds_bucket[1m])) by (le, role))' \
    | python3 -m json.tool | head -40
  kubectl -n "$NS" exec deploy/loadgen -- curl -s "http://payment:8080/fault?latency_ms=0"; echo
  set +x
  echo "if gateway p95 rose along with payment, propagation works"
  ;;

precheck)
  # gate G4 before the campaign, not after it - and only once the generator
  # actually running is the one that was asked for
  scripts/check_load_mode.sh "$NS" "${LOAD_MODE:-decorrelated}" || exit 1
  python3 scripts/precheck_k2.py --prometheus "http://localhost:$PROM_PORT" \
          --minutes "${2:-10}"
  ;;

campaign)
  mkdir -p "$DATA"
  # A campaign always starts from a fresh log: appending to an old one makes
  # collect.py re-query windows whose data has already left Prometheus's
  # retention, and those come back as all-zero blocks that look like real runs.
  if [ -f "$DATA/campaign.jsonl" ]; then
    mv "$DATA/campaign.jsonl" "$DATA/campaign.$(date +%Y%m%d-%H%M%S).jsonl.bak"
    echo "archived the previous campaign log"
  fi
  case "${2:-long}" in
    quick) ARGS="--warmup 120 --duration 180 --cooldown 30 --repeats 1 --normal-runs 4 --magnitudes 3.0" ;;
    long)  ARGS="--warmup 300 --duration 300 --cooldown 60 --repeats 1 --normal-runs 6 --magnitudes 3.0" ;;
    full)  ARGS="--warmup 300 --duration 300 --cooldown 60 --repeats 2 --normal-runs 8 --magnitudes 1.0 3.0" ;;
    *) echo "campaign quick|long|full"; exit 1 ;;
  esac
  python3 scripts/campaign.py --namespace "$NS" --log "$DATA/campaign.jsonl" $ARGS
  ;;

collect)
  python3 scripts/collect.py --prometheus "http://localhost:$PROM_PORT" \
      --topology k8s/topology.yaml --campaign "$DATA/campaign.jsonl" --out "$DATA" --step 1
  python3 - "$DATA" <<'PY'
import sys, glob, json, numpy as np, os
d = sys.argv[1]
S = np.load(os.path.join(d, "prior.npy"))
meta = json.load(open(os.path.join(d, "variables.json")))
nf = len(glob.glob(os.path.join(d, "faults", "*.npz")))
nn = len(glob.glob(os.path.join(d, "normal", "*.npy")))
print(f"variables={S.shape[0]}  admissible={int((S>0.5).sum())}  "
      f"density={(S>0.5).mean():.3f}  normal_windows={nn}  fault_runs={nf}")
if nf == 0 or nn == 0:
    print("WARNING: nothing collected -- is the port-forward running?")
empty_f = sum(1 for p in glob.glob(os.path.join(d, "faults", "*.npz"))
              if float(np.abs(np.load(p, allow_pickle=True)["X"]).max()) == 0.0)
empty_n = sum(1 for p in glob.glob(os.path.join(d, "normal", "*.npy"))
              if float(np.abs(np.load(p)).max()) == 0.0)
if empty_f or empty_n:
    print(f"WARNING: {empty_f} fault and {empty_n} nominal windows came back "
          f"EMPTY. Either the port-forward dropped during the campaign, or the "
          f"data left Prometheus's retention window before collection. "
          f"Re-run the campaign and collect immediately afterwards.")
else:
    print("all windows contain telemetry")
PY
  ;;

package)
  OUT="spcrl_$(basename "$DATA")_data.tar.gz"
  tar -czf "$OUT" -C "$(dirname "$DATA")" "$(basename "$DATA")"
  ls -la "$OUT"
  echo "send this file back for analysis"
  ;;

down)
  kubectl delete namespace "$NS" --ignore-not-found
  ;;

*)
  sed -n '2,12p' "$0"
  exit 1
  ;;
esac
