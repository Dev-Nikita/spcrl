#!/bin/bash
# Pre-registered analysis of K2b (docs/preregistration/PREREGISTRATION_EXTERNAL.md, Amendment 1).
# The campaign passed G1-G4 before any of this was run.
cd "$(dirname "$0")/.."; export PYTHONPATH=.
D="${1:-${K2B_DATA:?usage: run_k2b.sh <data-dir>}}"
T=benchmarks/microservices/k8s/topology.yaml
if [ ! -f results/.done_k2b_P1 ]; then
  python3 experiments/e10_real.py --data $D --env-name k2b --topology $T \
    --prior-variant P1_topology --corruptions 0.0 0.2 0.5 \
    --out results/e10_k2b.jsonl >> results/k2b.log 2>&1 && touch results/.done_k2b_P1
fi
for v in P2_semantic P3_random P3b_random_at_P2_density P4_degree_shuffled P6_wrong_topology driver_declared; do
  [ -f results/.done_k2b_$v ] && continue
  rm -f results/k2bctl_$v.jsonl
  python3 experiments/e10_real.py --data $D --env-name k2b --topology $T \
    --prior-variant $v --corruptions 0.0 --variants none hard soft \
    --out results/k2bctl_$v.jsonl >> results/k2b.log 2>&1 && touch results/.done_k2b_$v
done
echo K2B DONE >> results/k2b.log
