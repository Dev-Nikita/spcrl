"""Loader that turns a collected testbed directory into SP-CRL Samples.

The interface is identical to the synthetic benchmark, so no model or
evaluation code needs to know which environment produced the data.
"""
from __future__ import annotations
import glob, json, os
import numpy as np

from ..priors.features import pair_features
from .dataset import Sample


class _RealSystem:
    def __init__(self, graph_id, d, variables, node_type):
        self.graph_id, self.d = graph_id, d
        self.family, self.mechanism = "real", "real"
        self.variables, self.node_type = variables, node_type
        self.A_true = None          # unknown for a real system


def load_environment(path: str):
    S = np.load(os.path.join(path, "prior.npy")).astype(np.float32)
    meta = json.load(open(os.path.join(path, "variables.json")))
    return S, meta


def load_normal(path: str):
    return [np.load(f) for f in sorted(glob.glob(os.path.join(path, "normal", "*.npy")))]


def load_faults(path: str):
    out = []
    for f in sorted(glob.glob(os.path.join(path, "faults", "*.npz"))):
        z = np.load(f, allow_pickle=True)
        out.append({"X": z["X"], "fault_service": str(z["fault_service"]),
                    "fault_variables": z["fault_variables"],
                    "fault_type": str(z["fault_type"]),
                    "magnitude": float(z["magnitude"]), "run_id": os.path.basename(f)[:-4]})
    return out


def as_sample(path: str, X: np.ndarray, tag: str = "real") -> Sample:
    S, meta = load_environment(path)
    d = X.shape[1]
    Fp, Fn = pair_features(X, S)
    sysinst = _RealSystem(os.path.basename(path), d, meta["variables"], meta["node_type"])
    return Sample(sysinst, X, Fp, Fn, S, np.zeros((d, d), np.float32), tag)


# ---------------------------------------------------------------------------
# Validation of a collected environment
# ---------------------------------------------------------------------------
def _is_empty(X: np.ndarray) -> bool:
    """A window for which Prometheus returned nothing.

    The collector forward-fills and then zero-fills, so a range query that
    returned no series becomes an all-zero, zero-variance block rather than an
    error. Those windows must be dropped, not analysed: silently keeping them
    halves the effective sample and biases every ranking toward whichever
    variables happen to be non-zero.
    """
    X = np.asarray(X)
    return (not np.isfinite(X).all()) or float(np.abs(X).max()) == 0.0 \
        or float(X.std(0).max()) == 0.0


def load_validated(path: str, verbose: bool = True):
    """Load an environment and drop windows with no telemetry.

    Returns (S, meta, normals, faults, report).
    """
    S, meta = load_environment(path)
    normals_all = load_normal(path)
    faults_all = load_faults(path)
    normals = [X for X in normals_all if not _is_empty(X)]
    faults = [f for f in faults_all if not _is_empty(f["X"])]
    report = {
        "normal_total": len(normals_all), "normal_usable": len(normals),
        "fault_total": len(faults_all), "fault_usable": len(faults),
        "n_variables": S.shape[0],
        "window_len": int(normals[0].shape[0]) if normals else 0,
        "conditions": sorted({(f["fault_service"], f["fault_type"]) for f in faults}),
    }
    if verbose:
        print(f"[real] {path}: {report['normal_usable']}/{report['normal_total']} "
              f"nominal and {report['fault_usable']}/{report['fault_total']} fault "
              f"windows usable; {report['n_variables']} variables, "
              f"{report['window_len']} samples per window")
        if report["fault_usable"] < report["fault_total"]:
            print(f"[real] dropped {report['fault_total']-report['fault_usable']} "
                  f"fault windows with no telemetry (expired retention or a "
                  f"dropped port-forward during the campaign)")
    return S, meta, normals, faults, report


def reference_statistics(sur, normals):
    """Mean and spread of each node's residual across nominal windows.

    Using the spread across independent nominal windows, rather than a relative
    growth over a single reference, is what makes the score usable on channels
    that are identically zero in nominal operation - error rates and retry
    counters - which would otherwise divide by an arbitrary epsilon and
    dominate every ranking.
    """
    R = np.stack([sur.residuals(X) for X in normals])
    mean, std = R.mean(0), R.std(0)
    pos = std[std > 0]
    floor = float(np.quantile(pos, 0.25)) if len(pos) else 1e-6
    return mean, np.maximum(std, floor)


def deviation_scores(sur, X_fault, ref_mean, ref_std):
    """Standardised residual growth of each node on a fault window."""
    return (sur.residuals(X_fault) - ref_mean) / ref_std
