#!/usr/bin/env python3
"""Render Kubernetes manifests from topology.yaml.

topology.yaml is the single source of truth for both the deployment and the
structural prior: collect.py reads `depends_on` from the same file, so the
prior is derived from the deployment description and never from the telemetry.
"""
import argparse, pathlib, yaml

SVC = """---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
  namespace: {ns}
  labels: {{app: {name}, spcrl: "true"}}
spec:
  replicas: {replicas}
  selector: {{matchLabels: {{app: {name}}}}}
  template:
    metadata:
      labels: {{app: {name}}}
      annotations: {{prometheus.io/scrape: "true", prometheus.io/port: "8080"}}
    spec:
      containers:
      - name: svc
        image: {image}
        imagePullPolicy: {pull}
        env:
        - {{name: SVC_ROLE, value: "{name}"}}
        - {{name: SVC_DEPS, value: "{deps}"}}
        - {{name: SVC_BASE_LATENCY_MS, value: "{lat}"}}
        ports: [{{containerPort: 8080}}]
        readinessProbe:
          httpGet: {{path: /healthz, port: 8080}}
          initialDelaySeconds: 2
          periodSeconds: 5
        resources:
          requests: {{cpu: "{cpu_req}", memory: "64Mi"}}
          limits: {{cpu: "{cpu_lim}", memory: "192Mi"}}
---
apiVersion: v1
kind: Service
metadata: {{name: {name}, namespace: {ns}}}
spec:
  selector: {{app: {name}}}
  ports: [{{port: 8080, targetPort: 8080}}]
"""

LOAD = """---
apiVersion: apps/v1
kind: Deployment
metadata: {{name: loadgen, namespace: {ns}}}
spec:
  replicas: {replicas}
  strategy: {{type: Recreate}}
  selector: {{matchLabels: {{app: loadgen}}}}
  template:
    metadata: {{labels: {{app: loadgen}}}}
    spec:
      containers:
      - name: loadgen
        image: {image}
        imagePullPolicy: {pull}
        command: ["/bin/sh", "-c"]
        args:
        - |
          # spcrl-load-mode: shared
          # The request rate follows a slow random walk between {rate_min} and
          # {rate_max} requests per second. A constant rate pins the system at a
          # single operating point, where the only variation is noise and the
          # dependencies between services are unobservable: every throughput
          # series becomes the same series. Driving the system through different
          # load levels is what makes the structure visible at all.
          R={rate}
          while true; do
            STEP=$(( (RANDOM % 5) - 2 ))
            R=$(( R + STEP ))
            [ $R -lt {rate_min} ] && R={rate_min}
            [ $R -gt {rate_max} ] && R={rate_max}
            for i in $(seq 1 $R); do
              curl -s -o /dev/null --max-time 3 http://{entry}:8080/work &
            done
            sleep 1
          done
        resources:
          requests: {{cpu: "50m", memory: "32Mi"}}
          limits: {{cpu: "500m", memory: "128Mi"}}
"""

LOAD_INDEP = """---
apiVersion: apps/v1
kind: Deployment
metadata: {{name: loadgen, namespace: {ns}}}
spec:
  replicas: 1
  strategy: {{type: Recreate}}
  selector: {{matchLabels: {{app: loadgen}}}}
  template:
    metadata: {{labels: {{app: loadgen}}}}
    spec:
      containers:
      - name: loadgen
        image: {image}
        imagePullPolicy: {pull}
        command: ["/bin/sh", "-c"]
        args:
        - |
          # spcrl-load-mode: {mode}
          # Regime K2: independent excitation. Every service receives its own
          # request stream, each following its own random walk, so the load a
          # service sees is no longer one global signal. Requests sent to a
          # service still propagate to everything it calls; what changes is
          # that the call graph, not a shared generator, now decides which
          # services move together.
{loops}
          wait
        resources:
          requests: {{cpu: "100m", memory: "32Mi"}}
          limits: {{cpu: "1000m", memory: "192Mi"}}
"""

# One independent random walk per service. Each walk is seeded differently so
# the streams are not copies of each other; the shell's RANDOM is re-seeded by
# assignment.
LOOP = """          ( R={rate}; RANDOM={seed}
            while true; do
              STEP=$(( (RANDOM % {span}) - {half} )); R=$(( R + STEP ))
              [ $R -lt {rmin} ] && R={rmin}; [ $R -gt {rmax} ] && R={rmax}
              for i in $(seq 1 $R); do
                curl -s -o /dev/null --max-time 3 http://{svc}:8080/work &
              done
              sleep {period}
            done ) &"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default="k8s/topology.yaml")
    ap.add_argument("--image", default="spcrl/svc:latest")
    ap.add_argument("--namespace", default="spcrl")
    ap.add_argument("--out", default="k8s/services.yaml")
    ap.add_argument("--replicas", type=int, default=1)
    ap.add_argument("--rate", type=int, default=12, help="requests/s from the load generator")
    ap.add_argument("--pull", default="IfNotPresent")
    ap.add_argument("--load-mode",
                    choices=["shared", "independent", "decorrelated"],
                    default="shared",
                    help="shared: one generator at the entry point (regime K1). "
                         "independent: one random-walk stream per service (K2). "
                         "decorrelated: per-service streams whose rate falls "
                         "with depth from the entry point, so the traffic a "
                         "service receives directly dominates the traffic its "
                         "callers propagate to it (regime K2b).")
    a = ap.parse_args()

    t = yaml.safe_load(open(a.topology))
    entry = t.get("entrypoint", t["services"][0]["name"])
    chunks = [f"---\napiVersion: v1\nkind: Namespace\nmetadata: {{name: {a.namespace}}}\n"]
    for s in t["services"]:
        deps = ",".join(f"http://{d}:8080" for d in s["depends_on"])
        chunks.append(SVC.format(name=s["name"], ns=a.namespace, deps=deps,
                                 lat=s["base_latency_ms"], image=a.image,
                                 replicas=a.replicas, pull=a.pull,
                                 cpu_req=s.get("cpu_request", "50m"),
                                 cpu_lim=s.get("cpu_limit", "400m")))
    if a.load_mode == "shared":
        chunks.append(LOAD.format(ns=a.namespace, image=a.image, entry=entry,
                                  rate=a.rate, rate_min=max(2, a.rate // 3),
                                  rate_max=a.rate * 3, pull=a.pull, replicas=1))
    else:
        # K2 failed its manipulation check because two services that do not
        # call each other still move together when they are both called by a
        # third: the parent propagates its own stream to both. K2b removes that
        # shared component by making the stream a service receives directly
        # much larger than anything its callers can add, which means driving
        # the entry point least and the leaves most. The walks also run on
        # different periods, so no two streams share a time constant.
        depth, front, seen = {entry: 0}, [entry], {entry}
        dep = {x["name"]: list(x["depends_on"]) for x in t["services"]}
        while front:
            nxt = []
            for n in front:
                for d in dep.get(n, ()):
                    if d not in seen:
                        seen.add(d); depth[d] = depth[n] + 1; nxt.append(d)
            front = nxt
        loops = []
        for k, s in enumerate(t["services"]):
            dv = depth.get(s["name"], 1)
            if a.load_mode == "decorrelated":
                r = max(1, a.rate // 8) if dv == 0 else (
                    max(2, a.rate // 3) if dv == 1 else a.rate)
                per, span = 1 + (k % 3), 7
            else:
                r = a.rate if s["name"] == entry else max(2, a.rate // 2)
                per, span = 1, 5
            loops.append(LOOP.format(svc=s["name"], rate=r, seed=1000 + 17 * k,
                                     rmin=1, rmax=r * 3, period=per,
                                     span=span, half=span // 2))
        chunks.append(LOAD_INDEP.format(ns=a.namespace, image=a.image,
                                        pull=a.pull, mode=a.load_mode,
                                        loops="\n".join(loops)))
    pathlib.Path(a.out).write_text("".join(chunks))
    print(f"wrote {a.out}: {len(t['services'])} services + loadgen "
          f"({a.load_mode} load), entrypoint={entry}")


if __name__ == "__main__":
    main()
