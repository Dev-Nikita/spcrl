"""Pre-flight check of gate G4 (docs/preregistration/PREREGISTRATION_EXTERNAL.md) for K2/K2b.

Ten minutes of nominal traffic is enough to tell whether the load generator has
actually broken the shared driver. Run it before spending five hours on a
campaign whose manipulation check will fail.

    python3 scripts/precheck_k2.py [--minutes 10] [--prometheus http://localhost:9090]

Reports the mean |rho| between the throughputs of services that do NOT call each
other, at the same 1 s resolution the campaign collects. The gate is < 0.5; it
is the pre-registered threshold and is not adjusted here.
"""
from __future__ import annotations
import argparse, json, time, urllib.parse, urllib.request
import numpy as np, yaml

EXPR = 'sum(rate(svc_requests_total{{role="{s}"}}[15s]))'


def query_range(prom, expr, start, end, step):
    url = prom + "/api/v1/query_range?" + urllib.parse.urlencode(
        {"query": expr, "start": start, "end": end, "step": step})
    r = json.load(urllib.request.urlopen(url, timeout=30))
    res = r["data"]["result"]
    if not res:
        return None
    grid = np.arange(int(start), int(end) + 1, step)
    v = dict((int(float(t)), float(x)) for t, x in res[0]["values"])
    return np.array([v.get(int(t), np.nan) for t in grid], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default="k8s/topology.yaml")
    ap.add_argument("--prometheus", default="http://localhost:9090")
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--step", type=int, default=1)
    a = ap.parse_args()

    t = yaml.safe_load(open(a.topology))
    dep = {s["name"]: set(s["depends_on"]) for s in t["services"]}
    names = sorted(dep)
    end = time.time()
    print(f"sampling {a.minutes:.0f} min of nominal traffic; leave the load "
          f"generator running and inject no faults", flush=True)
    time.sleep(a.minutes * 60)
    end = time.time()
    start = end - a.minutes * 60
    X = {}
    for s in names:
        x = query_range(a.prometheus, EXPR.format(s=s), start, end, a.step)
        if x is None or np.all(np.isnan(x)):
            raise SystemExit(f"no throughput series for {s} - is the "
                             "port-forward up and the load generator running?")
        X[s] = np.nan_to_num(x)
    conn, unc = [], []
    for i, u in enumerate(names):
        for v in names[i + 1:]:
            r = float(np.corrcoef(X[u], X[v])[0, 1])
            (conn if (v in dep[u] or u in dep[v]) else unc).append((abs(r), u, v))
    mu = float(np.mean([r for r, _, _ in unc]))
    mc = float(np.mean([r for r, _, _ in conn]))
    print(f"connected pairs   mean |rho| {mc:.3f}  (n={len(conn)})")
    print(f"unconnected pairs mean |rho| {mu:.3f}  (n={len(unc)})   gate G4 < 0.5")
    for r, u, v in sorted(unc, reverse=True)[:5]:
        print(f"    {u:9s} {v:9s} |rho| {r:.3f}")
    print("G4 PASS - run the campaign" if mu < 0.5 else
          "G4 FAIL - do not run the campaign; see K2_RUNBOOK.md")
    raise SystemExit(0 if mu < 0.5 else 1)


if __name__ == "__main__":
    main()
