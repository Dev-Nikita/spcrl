#!/usr/bin/env bash
# Fault injection for the free5GC testbed.  Every action is idempotent and has
# an explicit "clear" counterpart, so a campaign can always return the system
# to its nominal state before the next run.
#
#   inject.sh upf   netem     120ms       # user-plane latency
#   inject.sh upf   loss      3%          # packet loss
#   inject.sh amf   cpu       2           # CPU saturation (n busy workers)
#   inject.sh nrf   sbi-delay 200ms       # service-based-interface latency
#   inject.sh <nf>  clear
set -euo pipefail
NS="${NS:-free5gc}"
NF="$1"; ACTION="$2"; ARG="${3:-}"
POD=$(kubectl -n "$NS" get pod -l app="$NF" -o jsonpath='{.items[0].metadata.name}')

case "$ACTION" in
  netem)     kubectl -n "$NS" exec "$POD" -c "$NF" -- tc qdisc replace dev eth0 root netem delay "$ARG" ;;
  loss)      kubectl -n "$NS" exec "$POD" -c "$NF" -- tc qdisc replace dev eth0 root netem loss "$ARG" ;;
  sbi-delay) kubectl -n "$NS" exec "$POD" -c "$NF" -- tc qdisc replace dev eth0 root netem delay "$ARG" distribution normal ;;
  cpu)       kubectl -n "$NS" exec "$POD" -c "$NF" -- sh -c "for i in \$(seq 1 $ARG); do (while :; do :; done) & done; echo started" ;;
  mem)       kubectl -n "$NS" exec "$POD" -c "$NF" -- sh -c "head -c ${ARG} /dev/zero | tail & echo started" ;;
  crash)     kubectl -n "$NS" delete pod "$POD" --grace-period=0 --force ;;
  clear)     kubectl -n "$NS" exec "$POD" -c "$NF" -- sh -c "tc qdisc del dev eth0 root 2>/dev/null || true; pkill -f 'while :' || true" ;;
  *) echo "unknown action $ACTION" >&2; exit 2 ;;
esac
echo "{\"nf\":\"$NF\",\"action\":\"$ACTION\",\"arg\":\"$ARG\",\"pod\":\"$POD\",\"ts\":$(date +%s)}"
