#!/bin/bash
# Verify that the load generator POD now running is the one the deployment asks
# for. A rollout that could not schedule leaves the previous regime running
# while every 'kubectl get' command still looks healthy, and a campaign run in
# that state measures the wrong system.
set -u
NS="${1:-spcrl}"; WANT="${2:-}"
pods=$(kubectl -n "$NS" get pods -l app=loadgen \
        --field-selector=status.phase=Running -o name 2>/dev/null)
n=$(printf '%s\n' "$pods" | grep -c . || true)
if [ "$n" -ne 1 ]; then
  echo "ERROR: $n running load generator pods (expected 1)."
  kubectl -n "$NS" get pods -l app=loadgen
  exit 1
fi
run=$(kubectl -n "$NS" get "$pods" \
        -o jsonpath='{.spec.containers[0].args[0]}' | \
      sed -n 's/.*spcrl-load-mode: \([a-z_]*\).*/\1/p' | head -1)
dep=$(kubectl -n "$NS" get deploy loadgen \
        -o jsonpath='{.spec.template.spec.containers[0].args[0]}' | \
      sed -n 's/.*spcrl-load-mode: \([a-z_]*\).*/\1/p' | head -1)
echo "load generator running: ${run:-unknown}   deployment: ${dep:-unknown}"
if [ -n "$WANT" ] && [ "$run" != "$WANT" ]; then
  echo "ERROR: the running generator is '${run:-unknown}', not '$WANT'."
  echo "Delete it so the new one can schedule:"
  echo "  kubectl -n $NS delete pod -l app=loadgen --wait=false"
  echo "  kubectl -n $NS rollout status deploy/loadgen"
  exit 1
fi
if [ "$run" != "$dep" ]; then
  echo "ERROR: running pod ('$run') does not match the deployment ('$dep')."
  exit 1
fi
