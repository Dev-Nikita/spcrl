#!/usr/bin/env python3
"""Reproducible fault-injection campaign against the Kubernetes testbed.

Every run is appended to the JSONL log *before* it starts, so the ground-truth
root cause is committed before any telemetry is seen and cannot be edited
afterwards.

Requests reach the services through `kubectl exec` into the load-generator pod,
so no port-forwarding of the services themselves is needed.
"""
from __future__ import annotations
import argparse, itertools, json, random, subprocess, sys, time, uuid
import yaml

FAULTS = {
    "cpu_saturation":    lambda m: {"cpu_ms": int(round(3 * m))},
    "latency_injection": lambda m: {"latency_ms": int(round(60 * m))},
    "error_injection":   lambda m: {"fail_rate": round(min(0.9, 0.08 * m), 4)},
}
CLEAR = {"cpu_ms": 0, "latency_ms": 0, "fail_rate": 0.0}


def kexec(ns, via, url, timeout=20):
    cmd = ["kubectl", "-n", ns, "exec", via, "--", "curl", "-s", "--max-time", "5", url]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"kubectl exec failed: {r.stderr.strip()[:300]}")
    return r.stdout.strip()


def apply(ns, via, svc, params):
    q = "&".join(f"{k}={v}" for k, v in params.items())
    return kexec(ns, via, f"http://{svc}:8080/fault?{q}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--topology", default="k8s/topology.yaml")
    p.add_argument("--namespace", default="spcrl")
    p.add_argument("--via", default="deploy/loadgen",
                   help="pod/deployment used as the in-cluster HTTP client")
    p.add_argument("--log", default="data/microservices/campaign.jsonl")
    p.add_argument("--duration", type=int, default=150, help="seconds of fault per run")
    p.add_argument("--warmup", type=int, default=90, help="seconds of nominal traffic before the fault")
    p.add_argument("--cooldown", type=int, default=30)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--normal-runs", type=int, default=6)
    p.add_argument("--magnitudes", type=float, nargs="+", default=[1.0, 3.0])
    p.add_argument("--services", nargs="*", default=None, help="restrict to these services")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    topo = yaml.safe_load(open(a.topology))
    svcs = a.services or [s["name"] for s in topo["services"]]
    rng = random.Random(a.seed)

    plan = [(None, None, 0.0)] * a.normal_runs
    plan += [(s, f, m) for _ in range(a.repeats)
             for s, f, m in itertools.product(svcs, FAULTS, a.magnitudes)]
    rng.shuffle(plan)
    total = len(plan) * (a.warmup + a.duration + a.cooldown)
    print(f"{len(plan)} runs, ~{total/3600:.1f} h wall clock", flush=True)
    if a.dry_run:
        for r in plan[:10]:
            print("  ", r)
        return

    import os
    os.makedirs(os.path.dirname(a.log) or ".", exist_ok=True)
    with open(a.log, "a") as log:
        for n, (svc, ft, mag) in enumerate(plan, 1):
            run_id = uuid.uuid4().hex[:10]
            for s in svcs:
                apply(a.namespace, a.via, s, CLEAR)
            t_start = time.time() + a.warmup
            rec = {"run_id": run_id, "fault_service": svc, "fault_type": ft,
                   "magnitude": mag, "t_start": t_start,
                   "t_end": t_start + a.duration, "warmup_s": a.warmup,
                   "cooldown_s": a.cooldown, "seed": a.seed,
                   "params": FAULTS[ft](mag) if ft else {},
                   "logged_at": time.time(), "topology": a.topology}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            time.sleep(a.warmup)
            if svc is not None:
                state = apply(a.namespace, a.via, svc, FAULTS[ft](mag))
                print(f"[{n}/{len(plan)}] {run_id} {svc} {ft} x{mag} -> {state}", flush=True)
            else:
                print(f"[{n}/{len(plan)}] {run_id} nominal", flush=True)
            time.sleep(a.duration)
            for s in svcs:
                apply(a.namespace, a.via, s, CLEAR)
            time.sleep(a.cooldown)
    print("campaign complete", flush=True)


if __name__ == "__main__":
    main()
