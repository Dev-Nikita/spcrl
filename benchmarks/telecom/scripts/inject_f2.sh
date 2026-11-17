#!/usr/bin/env bash
# F2 fault injector (docs/preregistration/PREREGISTRATION_F2.md). A superset of
# inject_compose.sh, which stays untouched so that F1 remains reproducible.
#   inject_f2.sh amf sbi-delay 180ms
#   inject_f2.sh amf loss 10%
#   inject_f2.sh amf rate 96kbit
#   inject_f2.sh amf cpu 3                 # three busy loops in the container
#   inject_f2.sh amf cpu-throttle 0.02     # cgroup CPU quota, in cores
#   inject_f2.sh amf clear                 # removes every one of the above
set -euo pipefail
NF="$1"; ACTION="$2"; ARG="${3:-}"
C=$(docker ps --filter "name=^${NF}\$" --format '{{.Names}}' | head -1)
[ -n "$C" ] || C=$(docker ps --filter "name=$NF" --format '{{.Names}}' | head -1)
[ -n "$C" ] || { echo "no running container matching $NF" >&2; exit 1; }
case "$ACTION" in
  sbi-delay|netem) docker exec "$C" tc qdisc replace dev eth0 root netem delay "$ARG" ;;
  loss)            docker exec "$C" tc qdisc replace dev eth0 root netem loss "$ARG" ;;
  rate)            docker exec "$C" tc qdisc replace dev eth0 root netem rate "$ARG" ;;
  cpu)             docker exec -d "$C" sh -c "for i in \$(seq 1 $ARG); do (while :; do :; done) & done" ;;
  cpu-throttle)    docker update --cpus "$ARG" "$C" >/dev/null ;;
  clear)           docker exec "$C" sh -c "tc qdisc del dev eth0 root 2>/dev/null || true; pkill -f 'while :' || true"
                   docker update --cpus 0 "$C" >/dev/null ;;
  *) echo "unknown action $ACTION" >&2; exit 2 ;;
esac
echo "{\"nf\":\"$NF\",\"container\":\"$C\",\"action\":\"$ACTION\",\"arg\":\"$ARG\",\"ts\":$(date +%s)}"
