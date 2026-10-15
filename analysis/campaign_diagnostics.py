"""Traceable quality diagnostics for a collected real-environment campaign.

Every number §2.10 of the manuscript reports about the Kubernetes campaign is
produced here and written to results/campaign_diagnostics.json, so the claim
ledger can point at a raw artifact rather than at a console transcript.

Diagnostics computed
--------------------
sample counts          nominal windows, fault windows, samples per window
load variation         coefficient of variation of the per-service throughput,
                       pooled over nominal windows (a constant-rate generator
                       pins the deployment at one operating point)
cross-variable R^2     out-of-sample ridge regression of each variable on all
                       others, fitted on a random half of the nominal record and
                       scored on the other half; reported as a median over
                       variables, with and without the throughput channels
degenerate channels    variables whose nominal standard deviation is numerically
                       zero (error and retry counters are identically zero when
                       nothing is wrong, so no model fitted on nominal data can
                       explain them away)
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spcrl.benchmarks.real import load_environment, load_normal, load_faults  # noqa: E402


def _ridge_r2(Xtr, Xte, j, lam=1e-2):
    idx = [k for k in range(Xtr.shape[1]) if k != j]
    A, b = Xtr[:, idx], Xtr[:, j]
    mu, sd = A.mean(0), A.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    A = (A - mu) / sd
    Ate = (Xte[:, idx] - mu) / sd
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (b - b.mean()))
    pred = Ate @ w + b.mean()
    ss_res = float(((Xte[:, j] - pred) ** 2).sum())
    ss_tot = float(((Xte[:, j] - Xte[:, j].mean()) ** 2).sum())
    if ss_tot < 1e-12:
        return None
    return 1.0 - ss_res / ss_tot


def diagnose(path: str, seed: int = 0) -> dict:
    S, meta = load_environment(path)
    variables = meta["variables"]
    normal, faults = load_normal(path), load_faults(path)
    Xn = np.concatenate(normal, 0)

    sd = Xn.std(0)
    degenerate = [v for v, s in zip(variables, sd) if s < 1e-12]
    live = [k for k, s in enumerate(sd) if s >= 1e-12]

    thr = [k for k, v in enumerate(variables) if v.endswith("::throughput")]
    cv = []
    for k in thr:
        col = Xn[:, k]
        if abs(col.mean()) > 1e-12:
            cv.append(float(col.std() / abs(col.mean())))

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Xn))
    half = len(Xn) // 2
    Xtr, Xte = Xn[perm[:half]], Xn[perm[half:]]
    r2 = {}
    for j in live:
        v = _ridge_r2(Xtr, Xte, j)
        if v is not None:
            r2[variables[j]] = float(v)
    vals = np.array(list(r2.values()))
    no_thr = np.array([v for k, v in r2.items() if not k.endswith("::throughput")])

    return {
        "path": os.path.abspath(path),
        "n_variables": len(variables),
        "n_services": len(set(v.split("::")[0] for v in variables)),
        "n_normal_windows": len(normal),
        "n_fault_windows": len(faults),
        "samples_per_window": int(min(x.shape[0] for x in normal + [f["X"] for f in faults])),
        "n_nominal_samples": int(Xn.shape[0]),
        "prior_edges": int((S > 0.5).sum()),
        "prior_density": float((S > 0.5).mean()),
        "throughput_cv_median": float(np.median(cv)) if cv else None,
        "throughput_cv_min": float(np.min(cv)) if cv else None,
        "throughput_cv_max": float(np.max(cv)) if cv else None,
        "degenerate_channels": degenerate,
        "n_degenerate": len(degenerate),
        "n_live": len(live),
        "r2_median_all": float(np.median(vals)),
        "r2_median_excl_throughput": float(np.median(no_thr)),
        "r2_per_variable": r2,
    }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SPCRL_CAMPAIGN_DIR", "data/campaign")
    out = sys.argv[2] if len(sys.argv) > 2 else "results/campaign_diagnostics.json"
    rep = diagnose(path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rep, open(out, "w"), indent=2)
    slim = {k: v for k, v in rep.items() if k not in ("r2_per_variable", "degenerate_channels")}
    print(json.dumps(slim, indent=2))
    print("degenerate:", ", ".join(rep["degenerate_channels"]))
