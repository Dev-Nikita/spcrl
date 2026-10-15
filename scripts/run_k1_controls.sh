#!/bin/bash
# Exploratory: the pre-registered prior controls on K1 (fixed code). K1 motivated
# these questions, so nothing here is a test (PREREGISTRATION_EXTERNAL.md).
# P1 is the deployment prior itself and is already in e10_fixed.jsonl.
cd "$(dirname "$0")/.."; export PYTHONPATH=.
D="${1:-${K1_DATA:?usage: run_k1_controls.sh <data-dir>}}"
for v in P2_semantic P3_random P3b_random_at_P2_density P4_degree_shuffled P6_wrong_topology driver_declared; do
  [ -f results/.done_k1_$v ] && continue
  rm -f results/k1ctl_$v.jsonl
  python3 experiments/e10_real.py --data "$D" \
    --topology benchmarks/microservices/k8s/topology.yaml \
    --prior-variant $v --corruptions 0.0 --variants none hard soft \
    --out results/k1ctl_$v.jsonl >> results/k1ctl.log 2>&1 && touch results/.done_k1_$v
done
echo K1CTL DONE >> results/k1ctl.log
