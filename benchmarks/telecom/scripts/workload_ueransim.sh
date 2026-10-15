#!/usr/bin/env bash
# F1 workload: UE registrations whose concurrency follows a random walk.
#
# Runs INSIDE the UERANSIM container (the gNB must already be up and connected
# to the AMF). Every cycle it starts K UEs with consecutive IMSIs, lets them
# register and attempt a PDU session, and kills them 20 s later; K moves by a
# random step each cycle. Registration exercises AMF -> AUSF -> UDM -> UDR,
# AMF -> NSSF, AMF -> PCF and AMF -> UDM; the session attempt exercises
# AMF -> SMF -> UDM / PCF. The varying K is the excitation: the NF call graph,
# not a prober, decides which functions move together.
#
# Usage:  workload_ueransim.sh <ue-config.yaml> <first-imsi-number> <n-subscribers> [kmin] [kmax]
#   e.g.  workload_ueransim.sh /ueransim/config/uecfg.yaml 208930000000001 40 2 20
# The IMSIs first..first+n-1 must be provisioned as subscribers (README step 3).
set -u
CFG="$1"; FIRST="$2"; NSUB="$3"; KMIN="${4:-2}"; KMAX="${5:-20}"
K=$(( (KMIN + KMAX) / 2 ))
RANDOM=4242
while true; do
  STEP=$(( (RANDOM % 5) - 2 )); K=$(( K + STEP ))
  [ $K -lt $KMIN ] && K=$KMIN; [ $K -gt $KMAX ] && K=$KMAX
  [ $K -gt $NSUB ] && K=$NSUB
  OFF=$(( RANDOM % (NSUB - K + 1) ))
  IMSI="imsi-$(( FIRST + OFF ))"
  nr-ue -c "$CFG" -n "$K" -i "$IMSI" > /tmp/nr-ue.log 2>&1 &
  PID=$!
  sleep 20
  kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null
  echo "$(date +%s) k=$K first=$IMSI registered=$(grep -ci 'registration is successful' /tmp/nr-ue.log)"
  sleep 2
done
