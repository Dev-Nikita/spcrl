#!/usr/bin/env python3
"""F1 fault campaign against free5GC running under docker compose.

Same contract as benchmarks/microservices/scripts/campaign.py: every run is
appended to the JSONL log BEFORE it starts, so the ground-truth root cause is
committed before any telemetry exists and cannot be edited afterwards. The log
format is identical, so collect.py and experiments/e10_real.py consume it
unchanged.

Faults (pre-registered in docs/preregistration/PREREGISTRATION_EXTERNAL.md):
  cpu_saturation   N busy loops inside the network function's container
  sbi_latency      tc netem delay on the container's interface

Both are injected and cleared through inject_compose.sh, which needs NET_ADMIN
on the target containers (see the compose override in the README).
"""
from __future__ import annotations
import argparse, itertools, json, os, random, subprocess, time, uuid
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
INJECT = os.path.join(HERE, "inject_compose.sh")

FAULTS = {
    "cpu_saturation": lambda m: ("cpu", str(max(1, int(round(m))))),
    "sbi_latency":    lambda m: ("sbi-delay", f"{int(round(60 * m))}ms"),
}


def inject(container, action, arg=""):
    r = subprocess.run([INJECT, container, action, arg], capture_output=True,
                       text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"inject {container} {action} failed: {r.stderr.strip()[:300]}")
    return r.stdout.strip()


def clear_all(containers):
    for c in containers:
        try:
            inject(c, "clear")
        except RuntimeError as e:          # a restarted container has no qdisc
            print("  clear:", e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--topology", default=os.path.join(HERE, "..", "f1_topology.yaml"))
    p.add_argument("--log", default="data/f1/campaign.jsonl")
    p.add_argument("--duration", type=int, default=300)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--cooldown", type=int, default=60)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--normal-runs", type=int, default=6)
    p.add_argument("--magnitudes", type=float, nargs="+", default=[3.0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    topo = yaml.safe_load(open(a.topology))
    cont = {s["name"]: s.get("container", s["name"]) for s in topo["services"]}
    targets = topo["fault_targets"]
    rng = random.Random(a.seed)
    plan = [(None, None, 0.0)] * a.normal_runs
    plan += [(s, f, m) for _ in range(a.repeats)
             for s, f, m in itertools.product(targets, FAULTS, a.magnitudes)]
    rng.shuffle(plan)
    total = len(plan) * (a.warmup + a.duration + a.cooldown)
    print(f"{len(plan)} runs, ~{total/3600:.1f} h wall clock", flush=True)
    if a.dry_run:
        for r in plan[:12]:
            print("  ", r)
        return

    if os.path.exists(a.log) and os.path.getsize(a.log) > 0:
        # never mix two campaigns in one log: that lost half of the first
        # Kubernetes campaign (Supplementary Note 3)
        os.rename(a.log, a.log + f".{int(time.time())}.bak")
        print("archived previous campaign log", flush=True)
    os.makedirs(os.path.dirname(a.log) or ".", exist_ok=True)
    fault_containers = [cont[s] for s in targets]
    with open(a.log, "a") as log:
        for n, (svc, ft, mag) in enumerate(plan, 1):
            run_id = uuid.uuid4().hex[:10]
            clear_all(fault_containers)
            t_start = time.time() + a.warmup
            action, arg = FAULTS[ft](mag) if ft else (None, None)
            rec = {"run_id": run_id, "fault_service": svc, "fault_type": ft,
                   "magnitude": mag, "t_start": t_start,
                   "t_end": t_start + a.duration, "warmup_s": a.warmup,
                   "cooldown_s": a.cooldown, "seed": a.seed,
                   "params": {"action": action, "arg": arg} if ft else {},
                   "logged_at": time.time(), "topology": a.topology}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            time.sleep(a.warmup)
            if svc is not None:
                out = inject(cont[svc], action, arg)
                print(f"[{n}/{len(plan)}] {run_id} {svc} {ft} x{mag} -> {out}", flush=True)
            else:
                print(f"[{n}/{len(plan)}] {run_id} nominal", flush=True)
            time.sleep(a.duration)
            clear_all(fault_containers)
            time.sleep(a.cooldown)
    print("campaign complete", flush=True)


if __name__ == "__main__":
    main()
