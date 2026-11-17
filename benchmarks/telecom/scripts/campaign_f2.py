#!/usr/bin/env python3
"""F2 fault campaign: the confirmatory free5GC campaign of PREREGISTRATION_F2.md.

Same contract as F1 (campaign_compose.py, left untouched): every run is appended
to the JSONL log BEFORE it starts, so the ground-truth root cause is committed
before any telemetry exists. The log format is identical, so the collector and
the analysis consume it unchanged.

What is new relative to F1, and why:
  * three fault families F1 did not have (sbi_loss, sbi_ratelimit,
    cpu_throttle). F1's repeats of one (service, fault type) cell returned the
    same diagnosis almost every time, so the independent units are cells, not
    repeats; new families are the only way to obtain new cells on this system.
  * one run per cell (F1 had two), 8 targets x 5 families = 40 cells, plus six
    nominal windows: 46 runs, about 8.4 h;
  * a different shuffle seed from F1 (F1: 0, F2: 2);
  * a calibration phase BEFORE the first run: magnitudes of the two
    load-relative faults are set from each function's own nominal load, by a
    rule fixed in the pre-registration, and written to calibration.json, whose
    hash every run record carries;
  * provenance captured before the first run: image digests, the workload
    script's hash, kernel and Docker versions.

If the campaign dies, it is restarted from scratch with the same seed; a partial
log is archived, never resumed and never analysed (pre-registration Sec. 7).

    python3 scripts/campaign_f2.py --dry-run
    python3 scripts/campaign_f2.py --prometheus http://localhost:9090
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, os, random, subprocess, time, uuid
import urllib.parse, urllib.request
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
INJECT = os.path.join(HERE, "inject_f2.sh")
SEED = 2
REPLICATION = ["cpu_saturation", "sbi_latency"]           # F1's families
NEW = ["sbi_loss", "sbi_ratelimit", "cpu_throttle"]       # out-of-sample families
LOSS = "10%"
FRACTION = 0.30          # load-relative faults cap a function at 30 % of its own p95
CPU_FLOOR = 0.01         # cores
RATE_FLOOR_KBIT = 16


def fault_action(ftype, target, calib):
    if ftype == "cpu_saturation":
        return "cpu", "3"
    if ftype == "sbi_latency":
        return "sbi-delay", "180ms"
    if ftype == "sbi_loss":
        return "loss", LOSS
    if ftype == "cpu_throttle":
        return "cpu-throttle", f"{calib[target]['cpu_limit_cores']:.4f}"
    if ftype == "sbi_ratelimit":
        return "rate", f"{calib[target]['rate_limit_kbit']}kbit"
    raise ValueError(ftype)


def prom(url, expr):
    q = urllib.parse.urlencode({"query": expr})
    with urllib.request.urlopen(f"{url}/api/v1/query?{q}", timeout=30) as r:
        res = json.load(r)["data"]["result"]
    return float(res[0]["value"][1]) if res else float("nan")


def calibrate(url, containers, window="10m"):
    """p95 of each function's own CPU and transmit rate over the last `window`
    of NOMINAL operation (no fault may be active: the caller clears first)."""
    out = {}
    for name, c in containers.items():
        cpu = prom(url, f'quantile_over_time(0.95, sum(rate(container_cpu_usage_seconds_total{{name="{c}"}}[15s]))[{window}:5s])')
        tx = prom(url, f'quantile_over_time(0.95, sum(rate(container_network_transmit_bytes_total{{name="{c}"}}[15s]))[{window}:5s])')
        if not (cpu == cpu and tx == tx):          # NaN check
            raise SystemExit(f"calibration: no series for {c}; run the smoke test")
        out[name] = {"p95_cpu_cores": cpu, "p95_tx_bytes_per_s": tx,
                     "cpu_limit_cores": max(CPU_FLOOR, FRACTION * cpu),
                     "rate_limit_kbit": max(RATE_FLOOR_KBIT, int(FRACTION * tx * 8 / 1000))}
    return out


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.stdout or r.stderr).strip()


def provenance():
    return {
        "kernel": sh("uname -r"),
        "docker": sh("docker version --format '{{.Server.Version}}'"),
        "images": sh("docker ps --format '{{.Names}} {{.Image}}' | sort"),
        "image_digests": sh("docker ps -q | xargs docker inspect --format '{{.Name}} {{.Image}}' | sort"),
        "workload_sha256": sh("docker exec ueransim sh -c 'sha256sum /workload*.sh 2>/dev/null'"),
        "workload_procs": sh("docker exec ueransim sh -c 'ps -eo args | grep -E \"nr-gnb|workload\" | grep -v grep'"),
        "host": sh("hostname"), "time": time.time(),
    }


def inject(container, action, arg=""):
    r = subprocess.run([INJECT, container, action, arg], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"inject {container} {action} failed: {r.stderr.strip()[:300]}")
    return r.stdout.strip()


def clear_all(containers):
    for c in containers:
        try:
            inject(c, "clear")
        except RuntimeError as e:
            print("  clear:", e, flush=True)


def plan_runs(targets, normal_runs, seed):
    plan = [(None, None)] * normal_runs
    plan += list(itertools.product(targets, REPLICATION + NEW))
    random.Random(seed).shuffle(plan)
    return plan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--topology", default=os.path.join(HERE, "..", "f1_topology.yaml"))
    p.add_argument("--out-dir", default="data/f2")
    p.add_argument("--prometheus", default="http://localhost:9090")
    p.add_argument("--duration", type=int, default=300)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--cooldown", type=int, default=60)
    p.add_argument("--normal-runs", type=int, default=6)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    topo = yaml.safe_load(open(a.topology))
    cont = {s["name"]: s.get("container", s["name"]) for s in topo["services"]}
    targets = topo["fault_targets"]
    plan = plan_runs(targets, a.normal_runs, SEED)
    total = len(plan) * (a.warmup + a.duration + a.cooldown)
    n_new = sum(1 for _, f in plan if f in NEW)
    print(f"{len(plan)} runs ({len(plan)-a.normal_runs} faults: {n_new} in new families), "
          f"~{total/3600:.1f} h wall clock", flush=True)
    if a.dry_run:
        for r in plan[:12]:
            print("  ", r)
        return

    os.makedirs(a.out_dir, exist_ok=True)
    log_path = os.path.join(a.out_dir, "campaign.jsonl")
    if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
        os.rename(log_path, log_path + f".{int(time.time())}.aborted")
        print("archived a previous (aborted) F2 log; restarting from scratch", flush=True)
    fault_containers = [cont[s] for s in targets]
    clear_all(fault_containers)
    print("calibrating on the last 10 min of nominal operation ...", flush=True)
    calib = calibrate(a.prometheus, {t: cont[t] for t in targets})
    calib_doc = {"rule": {"fraction_of_p95": FRACTION, "cpu_floor_cores": CPU_FLOOR,
                          "rate_floor_kbit": RATE_FLOOR_KBIT, "loss": LOSS},
                 "targets": calib, "provenance": provenance()}
    cpath = os.path.join(a.out_dir, "calibration.json")
    with open(cpath, "w") as f:
        json.dump(calib_doc, f, indent=1)
    chash = hashlib.sha256(open(cpath, "rb").read()).hexdigest()
    for t, v in calib.items():
        print(f"  {t:<5} p95 cpu {v['p95_cpu_cores']:.4f} cores -> throttle {v['cpu_limit_cores']:.4f};"
              f"  p95 tx {v['p95_tx_bytes_per_s']:.0f} B/s -> rate {v['rate_limit_kbit']} kbit", flush=True)

    with open(log_path, "a") as log:
        for n, (svc, ft) in enumerate(plan, 1):
            run_id = uuid.uuid4().hex[:10]
            clear_all(fault_containers)
            t_start = time.time() + a.warmup
            action, arg = fault_action(ft, svc, calib) if ft else (None, None)
            rec = {"run_id": run_id, "fault_service": svc, "fault_type": ft,
                   "magnitude": 1.0 if ft else 0.0, "t_start": t_start,
                   "t_end": t_start + a.duration, "warmup_s": a.warmup,
                   "cooldown_s": a.cooldown, "seed": SEED, "campaign": "F2",
                   "family": ("new" if ft in NEW else "replication") if ft else None,
                   "params": {"action": action, "arg": arg} if ft else {},
                   "calibration_sha256": chash,
                   "logged_at": time.time(), "topology": a.topology}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            time.sleep(a.warmup)
            if svc is not None:
                out = inject(cont[svc], action, arg)
                print(f"[{n}/{len(plan)}] {run_id} {svc} {ft} {arg} -> {out}", flush=True)
            else:
                print(f"[{n}/{len(plan)}] {run_id} nominal", flush=True)
            time.sleep(a.duration)
            clear_all(fault_containers)
            time.sleep(a.cooldown)
    print("campaign complete", flush=True)


if __name__ == "__main__":
    main()
