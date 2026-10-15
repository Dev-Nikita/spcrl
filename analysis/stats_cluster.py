"""Cluster-robust statistics: the experimental unit is the SYSTEM, not the run.

The earlier analysis paired at (system, seed), which counts the same held-out
system three times and is pseudoreplication: three seeds are three predictions
about the *same* system, not three independent systems. Everything reported in
the manuscript is now computed here instead:

  * per-system aggregation over seeds  ->  n = number of held-out systems,
  * paired Wilcoxon / Friedman on those system means,
  * bootstrap resampled by SYSTEM (cluster bootstrap), never by row,
  * a variance decomposition showing how much of the total variance sits
    between systems versus between seeds, which is what justifies the choice.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

from .stats import cliffs_delta, rank_biserial


def system_means(df: pd.DataFrame, metric: str, unit=("graph_id",),
                 method_col="method") -> pd.DataFrame:
    """Average over seeds (and any other within-system replication) first."""
    w = df.pivot_table(index=list(unit), columns=method_col, values=metric,
                       aggfunc="mean")
    return w.dropna(axis=0, how="any")


def cluster_bootstrap_ci(w: pd.DataFrame, col: str, n=10000, alpha=0.05, seed=0):
    x = w[col].values.astype(float)
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    bs = x[idx].mean(1)
    return float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def variance_decomposition(df: pd.DataFrame, metric: str, method: str,
                           method_col="method", unit="graph_id", rep="seed"):
    """One-way random-effects decomposition of a single method's scores."""
    d = df[df[method_col] == method]
    g = d.groupby(unit)[metric]
    k = g.count()
    if len(k) < 2 or k.max() < 2:
        return None
    grand = d[metric].mean()
    between = float((k * (g.mean() - grand) ** 2).sum() / max(len(k) - 1, 1))
    within = float(((d[metric] - d[unit].map(g.mean())) ** 2).sum()
                   / max(len(d) - len(k), 1))
    k0 = float(k.mean())
    var_sys = max((between - within) / k0, 0.0)
    icc = var_sys / (var_sys + within) if (var_sys + within) > 0 else np.nan
    return {"n_systems": int(len(k)), "reps_per_system": k0,
            "var_between_systems": var_sys, "var_within_system": within,
            "icc": float(icc)}


def compare(df: pd.DataFrame, metric: str, methods=None, unit=("graph_id",),
            reference: str | None = None, greater_is_better=True,
            n_boot=10000):
    """Paired comparison at the system level, with Holm-corrected Wilcoxon."""
    if methods is not None:
        df = df[df.method.isin(methods)]
    w = system_means(df, metric, unit)
    ms = list(w.columns)
    if len(ms) < 2 or len(w) < 3:
        return {"error": "insufficient systems", "n_systems": int(len(w))}
    out = {"n_systems": int(len(w)), "unit": list(unit), "metric": metric,
           "mean": {m: float(w[m].mean()) for m in ms},
           "ci95_cluster": {m: cluster_bootstrap_ci(w, m, n=n_boot) for m in ms}}
    if len(ms) >= 3:
        chi2, p = stats.friedmanchisquare(*[w[m].values for m in ms])
        out["friedman"] = {"chi2": float(chi2), "p": float(p)}
    ranks = w.rank(axis=1, ascending=not greater_is_better).mean()
    out["avg_rank"] = {m: float(ranks[m]) for m in ms}
    ref = reference or (max(ms, key=lambda m: w[m].mean()) if greater_is_better
                        else min(ms, key=lambda m: w[m].mean()))
    out["reference"] = ref
    tests = {}
    for m in ms:
        if m == ref:
            continue
        try:
            st, p = stats.wilcoxon(w[ref].values, w[m].values)
        except ValueError:
            st, p = np.nan, 1.0
        d = w[ref].values - w[m].values
        rng = np.random.default_rng(0)
        bd = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
        tests[m] = {"p_raw": float(p),
                    "mean_diff": float(d.mean()),
                    "diff_ci95": (float(np.quantile(bd, .025)),
                                  float(np.quantile(bd, .975))),
                    "rank_biserial": rank_biserial(w[ref].values, w[m].values),
                    "cliffs_delta": cliffs_delta(w[ref].values, w[m].values)}
    order = sorted(tests, key=lambda m: tests[m]["p_raw"])
    prev = 0.0
    for i, m in enumerate(order):
        adj = min(1.0, max(prev, (len(order) - i) * tests[m]["p_raw"]))
        tests[m]["p_holm"] = adj
        prev = adj
    out["posthoc"] = tests
    return out
