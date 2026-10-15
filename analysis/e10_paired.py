"""Paired comparison of the diagnosis methods on the real campaign.

The unit of analysis is the injected fault (24 conditions), and each method is
scored on the same fault with the same seed, so the comparison is paired.  The
confidence intervals are bootstrap intervals over faults, resampling the fault
rather than the (fault, seed) row so that the three seeds of one fault are not
treated as independent evidence.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.load import load  # noqa: E402

B = 10000


def _boot(diff_by_fault, seed=0):
    rng = np.random.default_rng(seed)
    keys = list(diff_by_fault)
    m = np.array([np.mean(diff_by_fault[k]) for k in keys])
    draws = m[rng.integers(0, len(m), (B, len(m)))].mean(1)
    return float(m.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def compare(d, a, b, metric, corruption=0.0):
    sub = d[(d.corruption == corruption) & (d.method.isin([a, b]))]
    w = sub.pivot_table(index=["run_id_fault", "seed"], columns="method", values=metric)
    w = w.dropna()
    out = {}
    for (fault, _seed), row in w.iterrows():
        out.setdefault(fault, []).append(row[a] - row[b])
    mean, lo, hi = _boot(out)
    return {"contrast": f"{a} - {b}", "metric": metric, "corruption": corruption,
            "n_faults": len(out), "n_rows": len(w),
            "mean": mean, "ci_lo": lo, "ci_hi": hi,
            "significant": bool(lo > 0 or hi < 0)}


def main():
    d = load("e10_fixed.jsonl")
    d = d[d.method != "_dataset_report"]
    methods = sorted(d.method.unique())
    rep = {"methods": methods,
           "seeds": sorted(float(s) for s in d.seed.dropna().unique()),
           "corruptions": sorted(float(c) for c in d.corruption.dropna().unique()),
           "marginal": {}, "contrasts": []}
    for m in methods:
        s = d[d.method == m]
        rep["marginal"][m] = {
            "top1": float(s.top1.mean()), "top3": float(s.top3.mean()),
            "mrr": float(s.mrr.mean()), "n": int(len(s))}
    pairs = [("spcrl_soft", "spcrl_none"), ("spcrl_hard", "spcrl_none"),
             ("spcrl_none", "anomaly_no_structure"), ("prior_only", "anomaly_no_structure"),
             ("spcrl_soft", "prior_only"), ("spcrl_soft_gate", "spcrl_soft"),
             ("spcrl_soft", "spcrl_soft_noenc")]
    for a, b in pairs:
        if a in methods and b in methods:
            for metric in ("top1", "mrr"):
                rep["contrasts"].append(compare(d, a, b, metric))
    # does corruption of the prior change anything for the prior-informed variants?
    for m in ("spcrl_soft", "spcrl_hard", "prior_only"):
        if m not in methods:
            continue
        s = d[d.method == m]
        rep.setdefault("corruption_sensitivity", {})[m] = {
            str(c): float(s[s.corruption == c].top1.mean()) for c in rep["corruptions"]}
    json.dump(rep, open("results/e10_paired.json", "w"), indent=2)
    print(json.dumps({k: rep[k] for k in ("marginal", "corruption_sensitivity")}, indent=2))
    for c in rep["contrasts"]:
        star = "*" if c["significant"] else " "
        print(f"{star} {c['contrast']:<38} {c['metric']:<5} "
              f"{c['mean']:+.3f} [{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}]  n_faults={c['n_faults']}")


if __name__ == "__main__":
    main()
