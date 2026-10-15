#!/bin/bash
# Pre-registered analysis of the free5GC campaign F1
# (docs/preregistration/PREREGISTRATION_EXTERNAL.md).
#
#   scripts/run_f1.sh <data-dir>      e.g. scripts/run_f1.sh ~/campaigns/f1
#
# The gates run first and the script stops if any of them fails: a campaign that
# fails a gate is reported as failed and no model is run on it. G4 is the K2
# manipulation check and does not apply here, so it is not requested.
set -u
cd "$(dirname "$0")/.."; export PYTHONPATH=.
D="${1:?usage: run_f1.sh <data-dir>}"
T=benchmarks/telecom/f1_topology.yaml

python3 analysis/external_gates.py "$D" "$T" || {
  echo "F1 failed its validity gates; reporting the failure, running no model."
  exit 1
}

if [ ! -f results/.done_f1_P1 ]; then
  python3 experiments/e10_real.py --data "$D" --env-name f1 --topology $T \
    --prior-variant P1_topology --corruptions 0.0 0.2 0.5 \
    --out results/e10_f1.jsonl >> results/f1.log 2>&1 && touch results/.done_f1_P1
fi
for v in P2_semantic P3_random P3b_random_at_P2_density P4_degree_shuffled P6_wrong_topology; do
  [ -f results/.done_f1_$v ] && continue
  rm -f results/f1ctl_$v.jsonl
  python3 experiments/e10_real.py --data "$D" --env-name f1 --topology $T \
    --prior-variant $v --corruptions 0.0 --variants none hard soft \
    --out results/f1ctl_$v.jsonl >> results/f1.log 2>&1 && touch results/.done_f1_$v
done
python3 analysis/external_controls.py F1 e10_f1.jsonl f1ctl_ | tee results/f1_controls.txt
echo F1 DONE >> results/f1.log
