// One generic instrumented service. Its role and downstream dependencies come
// from environment variables, so the whole topology lives in the Kubernetes
// manifests and one binary is deployed everywhere.
//
// Deliberately dependency-free (standard library only): the Prometheus text
// exposition format is emitted by hand, so `docker build` needs no module
// downloads and there is no go.sum to go stale.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

var (
	role   = env("SVC_ROLE", "generic")
	deps   = splitDeps(env("SVC_DEPS", ""))
	baseMs = envInt("SVC_BASE_LATENCY_MS", 5)

	mu       sync.RWMutex
	extraMs  int
	failRate float64
	cpuMs    int

	m       = newMetrics()
	client  = &http.Client{Timeout: 900 * time.Millisecond}
	buckets = []float64{.001, .002, .004, .008, .016, .032, .064, .128, .256, .512, 1.024, 2.048, 4.096}
)

func env(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}
func envInt(k string, d int) int {
	if v, err := strconv.Atoi(os.Getenv(k)); err == nil {
		return v
	}
	return d
}
func splitDeps(s string) []string {
	out := []string{}
	for _, d := range strings.Split(s, ",") {
		if d = strings.TrimSpace(d); d != "" {
			out = append(out, d)
		}
	}
	return out
}

// ---------------------------------------------------------------- metrics
type metrics struct {
	sync.Mutex
	reqOK, reqErr, retries uint64
	inflight               int64
	hist                   []uint64
	sum                    float64
	count                  uint64
	depErr                 map[string]uint64
	depLatSum              map[string]float64
	depLatN                map[string]uint64
}

func newMetrics() *metrics {
	return &metrics{hist: make([]uint64, len(buckets)+1),
		depErr: map[string]uint64{}, depLatSum: map[string]float64{},
		depLatN: map[string]uint64{}}
}

func (x *metrics) observe(sec float64, ok bool) {
	x.Lock()
	defer x.Unlock()
	x.sum += sec
	x.count++
	i := sort.SearchFloat64s(buckets, sec)
	x.hist[i]++
	if ok {
		x.reqOK++
	} else {
		x.reqErr++
	}
}

func (x *metrics) dep(name string, sec float64, ok bool) {
	x.Lock()
	defer x.Unlock()
	x.depLatSum[name] += sec
	x.depLatN[name]++
	if !ok {
		x.depErr[name]++
	}
}

func (x *metrics) render(w http.ResponseWriter) {
	x.Lock()
	defer x.Unlock()
	f := func(format string, a ...any) { fmt.Fprintf(w, format, a...) }
	f("# TYPE svc_requests_total counter\n")
	f("svc_requests_total{role=%q,status=\"ok\"} %d\n", role, x.reqOK)
	f("svc_requests_total{role=%q,status=\"error\"} %d\n", role, x.reqErr)
	f("# TYPE svc_retries_total counter\nsvc_retries_total{role=%q} %d\n", role, x.retries)
	f("# TYPE svc_inflight gauge\nsvc_inflight{role=%q} %d\n", role, x.inflight)
	f("# TYPE svc_latency_seconds histogram\n")
	var cum uint64
	for i, b := range buckets {
		cum += x.hist[i]
		f("svc_latency_seconds_bucket{role=%q,le=\"%g\"} %d\n", role, b, cum)
	}
	cum += x.hist[len(buckets)]
	f("svc_latency_seconds_bucket{role=%q,le=\"+Inf\"} %d\n", role, cum)
	f("svc_latency_seconds_sum{role=%q} %g\n", role, x.sum)
	f("svc_latency_seconds_count{role=%q} %d\n", role, x.count)
	f("# TYPE svc_dependency_latency_seconds_sum counter\n")
	for k, v := range x.depLatSum {
		f("svc_dependency_latency_seconds_sum{role=%q,dep=%q} %g\n", role, k, v)
		f("svc_dependency_latency_seconds_count{role=%q,dep=%q} %d\n", role, k, x.depLatN[k])
	}
	f("# TYPE svc_dependency_errors_total counter\n")
	for k, v := range x.depErr {
		f("svc_dependency_errors_total{role=%q,dep=%q} %d\n", role, k, v)
	}
	f("# TYPE svc_fault_state gauge\n")
	mu.RLock()
	f("svc_fault_state{role=%q,kind=\"latency_ms\"} %d\n", role, extraMs)
	f("svc_fault_state{role=%q,kind=\"fail_rate\"} %g\n", role, failRate)
	f("svc_fault_state{role=%q,kind=\"cpu_ms\"} %d\n", role, cpuMs)
	mu.RUnlock()
}

// ---------------------------------------------------------------- handlers
func work(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	m.Lock()
	m.inflight++
	m.Unlock()
	defer func() { m.Lock(); m.inflight--; m.Unlock() }()

	mu.RLock()
	burn, extra, fr := cpuMs, extraMs, failRate
	mu.RUnlock()

	if burn > 0 { // real CPU work, not a sleep
		deadline := time.Now().Add(time.Duration(burn) * time.Millisecond)
		x := 0.0
		for time.Now().Before(deadline) {
			for i := 0; i < 5000; i++ {
				x += math.Sqrt(float64(i))
			}
		}
		_ = x
	}
	time.Sleep(time.Duration(baseMs+rand.Intn(baseMs+1)) * time.Millisecond)
	if extra > 0 {
		time.Sleep(time.Duration(extra) * time.Millisecond)
	}

	ok := rand.Float64() >= fr
	if ok {
		ctx, cancel := context.WithTimeout(r.Context(), 800*time.Millisecond)
		defer cancel()
		for _, d := range deps {
			name := d[strings.LastIndex(d, "/")+1:]
			t0 := time.Now()
			good := false
			for attempt := 0; attempt < 3 && !good; attempt++ {
				if attempt > 0 {
					m.Lock()
					m.retries++
					m.Unlock()
				}
				req, _ := http.NewRequestWithContext(ctx, "GET", d+"/work", nil)
				resp, err := client.Do(req)
				if err == nil {
					resp.Body.Close()
					good = resp.StatusCode < 500
				}
			}
			m.dep(name, time.Since(t0).Seconds(), good)
			if !good {
				ok = false
				break
			}
		}
	}
	m.observe(time.Since(start).Seconds(), ok)
	if !ok {
		http.Error(w, "downstream failure", http.StatusInternalServerError)
		return
	}
	fmt.Fprintln(w, "ok")
}

// /fault?latency_ms=..&fail_rate=..&cpu_ms=..
// The only fault-injection path, so every campaign records exactly what was
// applied and can always clear it.
func fault(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	mu.Lock()
	if v := q.Get("latency_ms"); v != "" {
		extraMs, _ = strconv.Atoi(v)
	}
	if v := q.Get("fail_rate"); v != "" {
		failRate, _ = strconv.ParseFloat(v, 64)
	}
	if v := q.Get("cpu_ms"); v != "" {
		cpuMs, _ = strconv.Atoi(v)
	}
	st := map[string]any{"role": role, "latency_ms": extraMs, "fail_rate": failRate, "cpu_ms": cpuMs}
	mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(st)
}

func main() {
	http.HandleFunc("/work", work)
	http.HandleFunc("/fault", fault)
	http.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) { fmt.Fprintln(w, "ok") })
	http.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4")
		m.render(w)
	})
	log.Printf("role=%s deps=%v base=%dms", role, deps, baseMs)
	log.Fatal(http.ListenAndServe(":8080", nil))
}
