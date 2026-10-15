#!/usr/bin/env bash
# Same fault classes as inject.sh, but against docker compose instead of
# Kubernetes (Option B on Docker Desktop).
#   inject_compose.sh amf sbi-delay 200ms
#   inject_compose.sh amf clear
set -euo pipefail
NF="$1"; ACTION="$2"; ARG="${3:-}"
C=$(docker ps --filter "name=$NF" --format '{{.Names}}' | head -1)
[ -n "$C" ] || { echo "no running container matching $NF" >&2; exit 1; }
case "$ACTION" in
  sbi-delay|netem) docker exec "$C" tc qdisc replace dev eth0 root netem delay "$ARG" ;;
  loss)            docker exec "$C" tc qdisc replace dev eth0 root netem loss "$ARG" ;;
  cpu)             docker exec -d "$C" sh -c "for i in \$(seq 1 $ARG); do (while :; do :; done) & done" ;;
  crash)           docker restart "$C" >/dev/null ;;
  clear)           docker exec "$C" sh -c "tc qdisc del dev eth0 root 2>/dev/null || true; pkill -f 'while :' || true" ;;
  *) echo "unknown action $ACTION" >&2; exit 2 ;;
esac
echo "{\"nf\":\"$NF\",\"container\":\"$C\",\"action\":\"$ACTION\",\"arg\":\"$ARG\",\"ts\":$(date +%s)}"
