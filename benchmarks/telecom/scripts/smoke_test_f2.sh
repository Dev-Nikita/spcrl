#!/usr/bin/env bash
# F2 smoke test. Runs the F1 smoke test, then checks every assumption the NEW
# fault families depend on. It stops at the first failure. A campaign started on
# a failed smoke test is not the pre-registered campaign.
#
#   ./scripts/smoke_test_f2.sh [prometheus-url]
set -u
PROM="${1:-http://localhost:9090}"
HERE="$(cd "$(dirname "$0")" && pwd)"
INJ="$HERE/inject_f2.sh"
fail () { echo "FAIL: $*"; "$INJ" ausf clear >/dev/null 2>&1; exit 1; }
ok ()   { echo "ok   $*"; }

echo "######## F1 checks"
"$HERE/smoke_test_f1.sh" "$PROM" || fail "the F1 smoke test failed (see above)"

echo "######## F2 checks (on ausf; every injection is cleared again)"
C=$(docker ps --filter "name=^ausf\$" --format '{{.Names}}' | head -1); [ -n "$C" ] || C=ausf

echo "== 6. netem loss is accepted and removed"
"$INJ" ausf loss 10% >/dev/null || fail "netem loss refused"
docker exec "$C" tc qdisc show dev eth0 | grep -qi "loss 10%" || fail "loss not visible in tc qdisc"
"$INJ" ausf clear >/dev/null
docker exec "$C" tc qdisc show dev eth0 | grep -q netem && fail "netem still present after clear"
ok "loss 10% injected and cleared"

echo "== 7. netem rate is accepted and removed"
"$INJ" ausf rate 64kbit >/dev/null || fail "netem rate refused"
docker exec "$C" tc qdisc show dev eth0 | grep -qi "rate 64kbit" || fail "rate not visible in tc qdisc"
"$INJ" ausf clear >/dev/null
docker exec "$C" tc qdisc show dev eth0 | grep -q netem && fail "netem still present after clear"
ok "rate 64kbit injected and cleared"

echo "== 8. CPU throttle is applied and fully removed"
before=$(docker inspect -f '{{.HostConfig.NanoCpus}}' "$C")
[ "$before" = "0" ] || fail "ausf already has a CPU limit ($before); remove it from the compose file first"
"$INJ" ausf cpu-throttle 0.05 >/dev/null || fail "docker update --cpus refused"
during=$(docker inspect -f '{{.HostConfig.NanoCpus}}' "$C")
[ "$during" = "50000000" ] || fail "throttle not applied (NanoCpus=$during)"
"$INJ" ausf clear >/dev/null
after=$(docker inspect -f '{{.HostConfig.NanoCpus}}' "$C")
[ "$after" = "0" ] || fail "throttle not removed by clear (NanoCpus=$after)"
ok "throttle 0.05 cores applied and removed (NanoCpus 0 -> 50000000 -> 0)"

echo "== 9. the system survives the harshest new fault: UEs keep registering under 10% loss on ausf"
"$INJ" ausf loss 10% >/dev/null
sleep 90
REG=$(docker logs --since 90s ueransim 2>&1 | grep -ci "registration is successful")
"$INJ" ausf clear >/dev/null
[ "$REG" -gt 0 ] || fail "no registrations under 10% loss on ausf - the fault is too strong; stop and send this output"
ok "$REG registrations in 90 s under 10% loss on ausf"

echo "== 10. calibration series exist for every fault target (10-min p95 of CPU and transmit rate)"
python3 - "$PROM" "$HERE/../f1_topology.yaml" <<'PY' || fail "calibration query failed"
import sys, json, urllib.parse, urllib.request, yaml
prom, topo = sys.argv[1], yaml.safe_load(open(sys.argv[2]))
cont = {s["name"]: s.get("container", s["name"]) for s in topo["services"]}
bad = []
for t in topo["fault_targets"]:
    c = cont[t]
    for m in ("container_cpu_usage_seconds_total", "container_network_transmit_bytes_total"):
        q = f'quantile_over_time(0.95, sum(rate({m}{{name="{c}"}}[15s]))[10m:5s])'
        with urllib.request.urlopen(prom + "/api/v1/query?" + urllib.parse.urlencode({"query": q}), timeout=30) as r:
            res = json.load(r)["data"]["result"]
        v = float(res[0]["value"][1]) if res else float("nan")
        print(f"   {t:<5} {m.split('_')[2]:<8} p95 {v:.5g}")
        if not v > 0: bad.append((t, m))
sys.exit(1 if bad else 0)
PY
ok "calibration inputs present"
echo "F2 SMOKE TEST PASSED - the F2 campaign may start"
