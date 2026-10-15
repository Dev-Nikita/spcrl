#!/usr/bin/env python3
"""SBI probe exporter for a free5GC control plane.

free5GC network functions do not expose Prometheus metrics, so per-NF
observables are produced two ways:

  * from cAdvisor  -- CPU, memory, network bytes/errors per container;
  * from this probe -- request latency and error rate against each NF's
    service-based interface, sampled at a fixed rate.

Targets are given as NAME=URL pairs. Standard library only, so the container
is a plain python:3-alpine with no build step.

    python3 probe.py --targets nrf=http://nrf:8000/nnrf-nfm/v1/nf-instances \
                               amf=http://amf:8000/namf-comm/v1/... \
                     --interval 1 --port 9110
"""
from __future__ import annotations
import argparse, json, threading, time, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer

STATE: dict[str, dict] = {}
LOCK = threading.Lock()
BUCKETS = [.005, .01, .02, .04, .08, .16, .32, .64, 1.28, 2.56]


def probe_once(name: str, url: str, timeout: float):
    t0 = time.perf_counter()
    ok = False
    code = 0
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            code = r.status
            r.read(2048)
            ok = code < 500
    except urllib.error.HTTPError as e:
        code = e.code
        ok = e.code < 500          # a 404 still proves the NF answered
    except Exception:
        ok = False
    dt = time.perf_counter() - t0
    with LOCK:
        s = STATE.setdefault(name, {"n": 0, "err": 0, "sum": 0.0,
                                    "hist": [0] * (len(BUCKETS) + 1), "last_code": 0})
        s["n"] += 1
        s["sum"] += dt
        s["last_code"] = code
        if not ok:
            s["err"] += 1
        i = 0
        while i < len(BUCKETS) and dt > BUCKETS[i]:
            i += 1
        s["hist"][i] += 1


def loop(targets, interval, timeout):
    while True:
        start = time.time()
        threads = [threading.Thread(target=probe_once, args=(n, u, timeout), daemon=True)
                   for n, u in targets]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout + 0.5)
        time.sleep(max(0.0, interval - (time.time() - start)))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/healthz"):
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok"); return
        out = ["# TYPE sbi_probe_requests_total counter",
               "# TYPE sbi_probe_errors_total counter",
               "# TYPE sbi_probe_latency_seconds histogram"]
        with LOCK:
            snap = json.loads(json.dumps(STATE))
        for name, s in snap.items():
            out.append(f'sbi_probe_requests_total{{nf="{name}"}} {s["n"]}')
            out.append(f'sbi_probe_errors_total{{nf="{name}"}} {s["err"]}')
            out.append(f'sbi_probe_last_status{{nf="{name}"}} {s["last_code"]}')
            cum = 0
            for i, b in enumerate(BUCKETS):
                cum += s["hist"][i]
                out.append(f'sbi_probe_latency_seconds_bucket{{nf="{name}",le="{b}"}} {cum}')
            cum += s["hist"][-1]
            out.append(f'sbi_probe_latency_seconds_bucket{{nf="{name}",le="+Inf"}} {cum}')
            out.append(f'sbi_probe_latency_seconds_sum{{nf="{name}"}} {s["sum"]}')
            out.append(f'sbi_probe_latency_seconds_count{{nf="{name}"}} {s["n"]}')
        body = ("\n".join(out) + "\n").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", required=True, help="NAME=URL pairs")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=2.0)
    ap.add_argument("--port", type=int, default=9110)
    a = ap.parse_args()
    targets = [tuple(t.split("=", 1)) for t in a.targets]
    print("probing:", targets, flush=True)
    threading.Thread(target=loop, args=(targets, a.interval, a.timeout), daemon=True).start()
    HTTPServer(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
