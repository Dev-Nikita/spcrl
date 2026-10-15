#!/usr/bin/env bash
# F1 smoke test. Run it AFTER the core, cAdvisor, Prometheus and the workload
# are up and BEFORE the campaign. It checks every assumption the campaign
# depends on and stops at the first one that fails. A campaign started on a
# failed smoke test is not a pre-registered campaign.
#
#   ./smoke_test_f1.sh [prometheus-url]
set -u
PROM="${1:-http://localhost:9090}"
TOPO="$(dirname "$0")/../f1_topology.yaml"
fail () { echo "FAIL: $*"; exit 1; }
ok ()   { echo "ok   $*"; }

echo "== 1. containers named in f1_topology.yaml are running"
NAMES=$(docker ps --format '{{.Names}}')
for c in $(python3 -c "import yaml;print(' '.join(s.get('container',s['name']) for s in yaml.safe_load(open('$TOPO'))['services']))"); do
  echo "$NAMES" | grep -qx "$c" || fail "container '$c' not running - edit 'container:' in f1_topology.yaml (docker ps shows: $(echo $NAMES | tr '\n' ' '))"
done
ok "all containers present"

echo "== 2. fault injection works (NET_ADMIN, tc present)"
C=$(python3 -c "import yaml;print([s.get('container',s['name']) for s in yaml.safe_load(open('$TOPO'))['services'] if s['name']=='ausf'][0])")
docker exec "$C" sh -c "command -v tc" >/dev/null || fail "no 'tc' in $C: install iproute2 in the image or use a sidecar"
docker exec "$C" tc qdisc replace dev eth0 root netem delay 1ms || fail "tc netem refused in $C: add cap_add: [NET_ADMIN]"
docker exec "$C" tc qdisc del dev eth0 root
ok "tc netem on $C"

echo "== 3. UEs register"
REG=$(docker logs --since 120s ueransim 2>&1 | grep -ci "registration is successful")
[ "$REG" -gt 0 ] || fail "no successful registrations in the last 120 s of the ueransim logs - check subscriber provisioning (README step 3) and the gNB-AMF NGAP link"
ok "$REG successful registrations in the last 120 s"

echo "== 4. Prometheus sees cAdvisor series for every container"
for c in $(python3 -c "import yaml;print(' '.join(s.get('container',s['name']) for s in yaml.safe_load(open('$TOPO'))['services']))"); do
  for m in container_cpu_usage_seconds_total container_network_receive_bytes_total container_network_transmit_bytes_total container_memory_working_set_bytes; do
    n=$(curl -s "$PROM/api/v1/query" --data-urlencode "query=count($m{name=\"$c\"})" | python3 -c "import sys,json;r=json.load(sys.stdin)['data']['result'];print(r[0]['value'][1] if r else 0)")
    [ "$n" != "0" ] || fail "no $m for name=\"$c\" - is cAdvisor scraped, and does it label containers by name?"
  done
done
ok "cAdvisor series present"

echo "== 5. the workload moves the network functions (net_rx varies)"
for c in amf ausf udm; do
  cc=$(python3 -c "import yaml;print([s.get('container',s['name']) for s in yaml.safe_load(open('$TOPO'))['services'] if s['name']=='$c'][0])")
  v=$(curl -s "$PROM/api/v1/query" --data-urlencode "query=stddev_over_time(rate(container_network_receive_bytes_total{name=\"$cc\"}[15s])[5m:5s])" | python3 -c "import sys,json;r=json.load(sys.stdin)['data']['result'];print(float(r[0]['value'][1]) if r else 0.0)")
  python3 -c "import sys; sys.exit(0 if $v > 0 else 1)" || fail "$c received traffic does not vary - is the workload running?"
done
ok "workload visible in amf/ausf/udm traffic"
echo "SMOKE TEST PASSED - the campaign may start"
