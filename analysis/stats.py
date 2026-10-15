"""Statistical analysis used for every comparison reported in the paper.

Paired, per-system comparisons; Friedman omnibus over methods; Wilcoxon
signed-rank post-hoc with Holm correction; bootstrap CIs; rank-biserial and
Cliff's delta effect sizes.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_ci(x, n=10000, alpha=0.05, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = sum((a[:, None] > b[None, :]).sum(axis=None) for _ in [0])
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def rank_biserial(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def paired_matrix(df: pd.DataFrame, metric: str, block=("graph_id", "seed"),
                  method_col="method") -> pd.DataFrame:
    """Wide matrix: rows = blocks present for every method, cols = methods."""
    w = df.pivot_table(index=list(block), columns=method_col, values=metric,
                       aggfunc="mean")
    return w.dropna(axis=0, how="any")


def friedman_holm(df: pd.DataFrame, metric: str, block=("graph_id", "seed"),
                  reference: str | None = None, greater_is_better=True):
    W = paired_matrix(df, metric, block)
    methods = list(W.columns)
    if len(methods) < 2 or len(W) < 3:
        return {"error": "insufficient paired blocks", "n_blocks": len(W)}
    out = {"n_blocks": int(len(W)), "methods": methods,
           "mean": {m: float(W[m].mean()) for m in methods},
           "ci95": {m: bootstrap_ci(W[m].values) for m in methods}}
    if len(methods) >= 3:
        chi2, p = stats.friedmanchisquare(*[W[m].values for m in methods])
        out["friedman"] = {"chi2": float(chi2), "p": float(p)}
    ranks = W.rank(axis=1, ascending=not greater_is_better).mean()
    out["avg_rank"] = {m: float(ranks[m]) for m in methods}
    ref = reference or (max(methods, key=lambda m: W[m].mean()) if greater_is_better
                        else min(methods, key=lambda m: W[m].mean()))
    out["reference"] = ref
    tests = {}
    for m in methods:
        if m == ref:
            continue
        try:
            st, p = stats.wilcoxon(W[ref].values, W[m].values)
        except ValueError:
            st, p = np.nan, 1.0
        if not np.isfinite(p):
            # Every paired difference is exactly zero: the two methods made the
            # same decision on every block. scipy returns nan for that case;
            # the correct reading is "no evidence of a difference", p = 1.
            p = 1.0
        tests[m] = {"p_raw": float(p), "stat": float(st),
                    "rank_biserial": rank_biserial(W[ref].values, W[m].values),
                    "cliffs_delta": cliffs_delta(W[ref].values, W[m].values),
                    "mean_diff": float(W[ref].mean() - W[m].mean())}
    order = sorted(tests, key=lambda m: tests[m]["p_raw"])
    k = len(order)
    prev = 0.0
    for i, m in enumerate(order):
        adj = min(1.0, max(prev, (k - i) * tests[m]["p_raw"]))
        tests[m]["p_holm"] = adj
        prev = adj
    out["posthoc"] = tests
    return out
