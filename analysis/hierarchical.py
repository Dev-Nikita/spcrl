"""Two-way (system and training seed) bootstrap as a robustness check.

The primary analysis averages over training seeds within a system and treats
the system as the experimental unit (analysis/stats_cluster.py). A referee may
reasonably ask whether the conclusions depend on that choice, so the same
comparisons are repeated here under a model that treats BOTH the system and the
training seed as random effects:

    y_ism = mu + method_m + u_i + v_s + e_ism

Rather than fitting the mixed model with an extra dependency, we resample
systems and seeds independently (a two-way cluster bootstrap), which targets
the same variance structure and makes no normality assumption. A conclusion
that survives both the system-level paired test and this interval is not an
artefact of the unit of analysis.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def variance_components(df: pd.DataFrame, metric: str, method: str,
                        method_col="method", unit="graph_id", rep="seed"):
    """Crude one-way decomposition per random factor, for reporting only."""
    d = df[df[method_col] == method]
    if d.empty:
        return None
    grand = d[metric].mean()
    out = {}
    for factor in (unit, rep):
        g = d.groupby(factor)[metric]
        k = g.count()
        if len(k) < 2:
            continue
        between = float((k * (g.mean() - grand) ** 2).sum() / max(len(k) - 1, 1))
        within = float(((d[metric] - d[factor].map(g.mean())) ** 2).sum()
                       / max(len(d) - len(k), 1))
        k0 = float(k.mean())
        var = max((between - within) / k0, 0.0)
        out[factor] = {"var": var, "residual": within,
                       "share": var / (var + within) if var + within > 0 else np.nan}
    return out


def twoway_bootstrap_diff(df: pd.DataFrame, metric: str, method_a: str,
                          method_b: str, unit="graph_id", rep="seed",
                          n=4000, seed=0):
    """Bootstrap the mean difference a - b, resampling systems and seeds."""
    d = df[df.method.isin([method_a, method_b])]
    w = d.pivot_table(index=[unit, rep], columns="method", values=metric,
                      aggfunc="mean").dropna()
    if method_a not in w or method_b not in w or len(w) < 4:
        return None
    w = w.reset_index()
    diff = (w[method_a] - w[method_b]).values
    systems = w[unit].unique()
    seeds = w[rep].unique()
    idx_by = {(s, r): np.where((w[unit].values == s) & (w[rep].values == r))[0]
              for s in systems for r in seeds}
    rng = np.random.default_rng(seed)
    boots = np.empty(n)
    for b in range(n):
        ss = rng.choice(systems, size=len(systems), replace=True)
        rr = rng.choice(seeds, size=len(seeds), replace=True)
        take = np.concatenate([idx_by[(s, r)] for s in ss for r in rr
                               if len(idx_by[(s, r)])])
        boots[b] = diff[take].mean()
    lo, hi = float(np.quantile(boots, .025)), float(np.quantile(boots, .975))
    return {"mean_diff": float(diff.mean()), "ci95": (lo, hi),
            "excludes_zero": bool(lo > 0 or hi < 0),
            "n_systems": int(len(systems)), "n_seeds": int(len(seeds))}
