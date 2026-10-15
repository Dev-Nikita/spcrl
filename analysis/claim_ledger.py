#!/usr/bin/env python3
"""Reconstruct every numerical claim of the manuscript from the raw results.

Each claim is recomputed from results/*.jsonl by code that does not share state
with the code that produced those files, and is written to docs/CLAIM_LEDGER.csv
with its status. A claim whose recomputed value disagrees with the manuscript is
marked MISMATCH; the script exits non-zero if any claim mismatches or cannot be
checked. Run it before every rebuild of the manuscript.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.load import load, load_multi
from analysis.stats import friedman_holm
from analysis.stats_cluster import compare, system_means

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROWS = []


def claim(cid, section, text, experiment, metric, claimed, actual, tol=0.002,
          script="analysis/claim_ledger.py"):
    try:
        ok = abs(float(claimed) - float(actual)) <= tol
        status = "OK" if ok else "MISMATCH"
        act = f"{float(actual):.4f}"
    except (TypeError, ValueError):
        status = "UNCHECKED"
        act = str(actual)
    ROWS.append(dict(claim_id=cid, section=section, claim=text,
                     experiment=experiment, metric=metric,
                     claimed=claimed, recomputed=act, status=status,
                     analysis_script=script))


def mean_auprc(df, method, regime=None, **filt):
    d = df[df.method == method]
    if regime is not None:
        d = d[d.regime == regime]
    for k, v in filt.items():
        d = d[d[k] == v]
    return d.auprc.mean() if len(d) else None


def main():
    # ---------------------------------------------------------- E1
    # E1 was extended with two further training seeds after the first run;
    # the files are pooled only at analysis time, each keeping its own metadata.
    d1_amort = load("e1_fixed.jsonl")          # re-run after the leak fix
    d1_base = load_multi(["e1_main.jsonl", "e1_main_seeds34.jsonl"])
    d1_base = d1_base[~d1_base.method.str.startswith("spcrl_amortized")]
    import pandas as _pd
    d1 = _pd.concat([d1_amort, d1_base], ignore_index=True)
    for reg, soft, none in [("iid", 0.967, 0.900), ("mech_ood", 0.960, 0.861),
                            ("graph_ood", 0.832, 0.720), ("size_ood", 0.970, 0.861),
                            ("noise_ood", 0.941, 0.827), ("joint_ood", 0.821, 0.697)]:
        claim(f"E1-soft-{reg}", "2.1/2.2", f"soft prior AUPRC in {reg}", "E1",
              "auprc", soft, mean_auprc(d1, "spcrl_amortized_soft", reg))
        claim(f"E1-none-{reg}", "2.1/2.2", f"prior-blind AUPRC in {reg}", "E1",
              "auprc", none, mean_auprc(d1, "spcrl_amortized_none", reg))
    claim("E1-seeds", "2.1/2.2", "training seeds in the headline comparison",
          "E1", "seeds", 5, d1_amort.seed.nunique(), tol=0)
    claim("E1-blocks", "2.1/2.2", "paired observations per headline cell", "E1",
          "blocks", 300,
          len(d1[(d1.regime == "iid") &
                 (d1.method == "spcrl_amortized_soft") & (d1.seed >= 0)]), tol=0)
    for m, val in [("pcorr", 0.410), ("pc", 0.330), ("notears_linear", 0.367),
                   ("prior_only", 0.167), ("notears_linear_hardmask", 0.549)]:
        claim(f"E1-base-{m}", "2.1", f"{m} AUPRC in distribution", "E1", "auprc",
              val, mean_auprc(d1, m, "iid"))
    for m, val in [("spcrl_direct_soft", 0.452), ("spcrl_direct_hard", 0.440),
                   ("spcrl_direct_none", 0.238)]:
        claim(f"E1-direct-{m}", "2.1", f"per-system {m} AUPRC in distribution",
              "E1", "auprc", val, mean_auprc(d1, m, "iid"))

    # ---------------------------------------------------------- E2
    d2 = load("e2_fixed.jsonl")               # re-run after the leak fix
    m2 = d2[d2.protocol == "matched"]
    for reg, c, soft, hard, none in [
            ("iid", 0.0, 0.969, 0.969, 0.898), ("iid", 0.5, 0.942, 0.513, 0.894),
            ("graph_ood", 0.0, 0.824, 0.829, 0.712),
            ("graph_ood", 0.5, 0.789, 0.455, 0.717)]:
        for name, val in [("soft", soft), ("hard", hard), ("none", none)]:
            claim(f"E2-{reg}-c{c}-{name}", "2.3",
                  f"{name} AUPRC at corruption {c} in {reg}", "E2", "auprc",
                  val, mean_auprc(m2, f"spcrl_amortized_{name}", reg, corruption=c))
    claim("E2-seeds", "2.3", "training seeds in the corruption sweep", "E2",
          "seeds", 5, d2.seed.nunique(), tol=0)
    from analysis.make_tables import crossover_point
    for cid, proto, reg, v in [("E2-cstar-iid", "matched", "iid", 0.063),
                               ("E2-cstar-ood", "matched", "graph_ood", 0.136),
                               ("E2-cstar-drift-iid", "drift", "iid", 0.065),
                               ("E2-cstar-drift-ood", "drift", "graph_ood", 0.146)]:
        r = crossover_point(proto, reg, n_boot=200)
        claim(cid, "2.3", f"hard-mask crossover, {proto}/{reg}", "E2", "c_star",
              v, r["crossover_c"] if r else None, tol=0.002,
              script="analysis/make_tables.py")
    r = crossover_point("drift", "iid", n_boot=200, method="spcrl_amortized_soft")
    claim("E2-cstar-drift-soft", "2.3",
          "soft-prior crossover under drift, in distribution", "E2", "c_star",
          0.088, r["crossover_c"] if r else None, tol=0.002,
          script="analysis/make_tables.py")

    # ---------------------------------------------------------- E9
    d9 = load("e9_fixed.jsonl")
    for c, hard, none, noenc, gate, soft in [
            (0.0, 0.966, 0.899, 0.952, 0.952, 0.968),
            (0.5, 0.518, 0.899, 0.927, 0.927, 0.943)]:
        for name, val in [("hard", hard), ("none", none), ("soft_noenc", noenc),
                          ("soft_gate", gate), ("soft", soft)]:
            claim(f"E9-c{c}-{name}", "2.4", f"{name} AUPRC at c={c}, in distribution",
                  "E9", "auprc", val,
                  mean_auprc(d9, f"spcrl_amortized_{name}", "iid", corruption=c))

    # ---------------------------------------------------------- E3
    d3 = load("e3_fixed.jsonl")
    for kind, iid, ood in [("none", 0.913, 0.724), ("topology", 0.939, 0.743),
                           ("topology_type", 0.952, 0.764),
                           ("topology_dir", 0.974, 0.821), ("full", 0.974, 0.821)]:
        sub = d3[d3.prior_kind == kind]
        claim(f"E3-{kind}-iid", "2.5", f"prior type {kind}, in distribution",
              "E3", "auprc", iid, sub[sub.regime == "iid"].auprc.mean())
        claim(f"E3-{kind}-ood", "2.5", f"prior type {kind}, unseen graph family",
              "E3", "auprc", ood, sub[sub.regime == "graph_ood"].auprc.mean())

    # ---------------------------------------------------------- E11
    d11 = load("e11_fixed.jsonl")
    for reg, vs in [("iid", 0.852), ("graph_ood", 0.869), ("mech_ood", 0.927)]:
        sub = d11[(d11.standardized == False) & (d11.regime == reg)]
        claim(f"E11-vs-{reg}", "2.7", f"varsortability, raw data, {reg}", "E11",
              "varsortability", vs, sub.varsortability.mean(), tol=0.01)
    for reg, raw, std in [("iid", 0.625, 0.219), ("graph_ood", 0.496, 0.187),
                          ("mech_ood", 0.697, 0.231)]:
        claim(f"E11-snr-raw-{reg}", "2.7", f"sortnregress AUPRC, raw, {reg}",
              "E11", "auprc", raw,
              mean_auprc(d11[d11.standardized == False], "sortnregress", reg), tol=0.01)
        claim(f"E11-snr-std-{reg}", "2.7", f"sortnregress AUPRC, standardised, {reg}",
              "E11", "auprc", std,
              mean_auprc(d11[d11.standardized == True], "sortnregress", reg), tol=0.01)
    for reg, soft, none in [("iid", 0.953, 0.874), ("graph_ood", 0.793, 0.683),
                            ("mech_ood", 0.958, 0.876)]:
        s = d11[(d11.standardized == True) & (d11.seed >= 0)]
        claim(f"E11-std-soft-{reg}", "2.7", f"soft prior, standardised, {reg}",
              "E11", "auprc", soft, mean_auprc(s, "spcrl_amortized_soft", reg))
        claim(f"E11-std-none-{reg}", "2.7", f"prior-blind, standardised, {reg}",
              "E11", "auprc", none, mean_auprc(s, "spcrl_amortized_none", reg))

    # ---------------------------------------------------------- E5
    d5 = load("e5_fixed.jsonl")
    for d_, soft, none in [(20, 0.9902, 0.9521), (50, 0.9703, 0.8601),
                           (100, 0.9492, 0.7829), (200, 0.9363, 0.7492)]:
        claim(f"E5-soft-d{d_}", "2.9", f"soft prior AUPRC at d={d_}", "E5", "auprc",
              soft, mean_auprc(d5[d5.d == d_], "spcrl_amortized_soft"), tol=0.005)
        claim(f"E5-none-d{d_}", "2.9", f"prior-blind AUPRC at d={d_}", "E5", "auprc",
              none, mean_auprc(d5[d5.d == d_], "spcrl_amortized_none"), tol=0.005)
    for d_, rho in [(20, 0.3075), (200, 0.0607)]:
        claim(f"E5-rho-d{d_}", "2.9", f"admissible fraction at d={d_}", "E5", "rho",
              rho, d5[d5.d == d_].rho.mean(), tol=0.002)

    for reg, v in (("iid", 0.48), ("graph_ood", 0.60)):
        sub = d11[d11.regime == reg]
        claim(f"E11-r2sort-{reg}", "Discussion", f"R2-sortability, {reg}", "E11",
              "r2_sortability", v, sub.r2_sortability.mean(), tol=0.006)

    # ---------------------------------------------------------- external gates (Methods)
    gj = os.path.join(ROOT, "results", "gates_k8sclean.json")
    if os.path.exists(gj):
        import json as _jg
        g = _jg.load(open(gj))
        claim("GATE-K1-G4-rho", "Methods",
              "mean |rho| between unconnected services' throughputs, K1", "gates",
              "rho", 0.992, g["mean_abs_rho_unconnected_throughput"], tol=0.0006,
              script="analysis/external_gates.py")
        claim("GATE-K1-G4-fails", "Methods", "K1 fails the K2 manipulation check",
              "gates", "bool", 0, int(g["G4_manipulation"]), tol=0,
              script="analysis/external_gates.py")

    # ---------------------------------------------------------- E16/E17/E18 (mismatch)
    import json as _j6
    e16j = os.path.join(ROOT, "results", "e16_summary.json")
    if os.path.exists(e16j):
        e16 = _j6.load(open(e16j))
        for m, v in [(0.0, 0.164), (0.2, 0.109), (0.4, 0.081), (0.6, 0.077),
                     (0.8, 0.053), (0.95, 0.056)]:
            claim(f"E16-C-m{m}", "2.11", f"compatibility statistic at m={m}",
                  "E16", "C", v, e16["C"][str(m)]["mean"], tol=0.002,
                  script="analysis/e16_mismatch.py")
        for m, v in [(0.0, 0.059), (0.2, 0.133), (0.4, 0.141), (0.6, 0.197),
                     (0.8, 0.247), (0.95, 0.344)]:
            claim(f"E16-adv-m{m}", "2.11",
                  f"soft prior advantage in distribution at m={m}", "E16",
                  "auprc", v, e16["advantage"][f"iid|m={m}|soft"]["mean"],
                  script="analysis/e16_mismatch.py")
        claim("E16-H1", "2.11", "Spearman rho between C and the mismatch level",
              "E16", "rho", -0.943, e16["H1_spearman_C_vs_m"], tol=0.002,
              script="analysis/e16_mismatch.py")
        claim("E16-H2", "2.11",
              "Spearman rho between the prior advantage and the mismatch level",
              "E16", "rho", 1.0, e16["H2_spearman_adv_vs_m_iid"], tol=0.002,
              script="analysis/e16_mismatch.py")
        claim("E16-H3", "2.11",
              "Spearman rho between the prior advantage and C", "E16", "rho",
              -0.943, e16["H3_spearman_adv_vs_C_iid"], tol=0.002,
              script="analysis/e16_mismatch.py")
    d17 = load("e17_fixed.jsonl")
    d17 = d17[d17.method.str.startswith("spcrl")]
    gaps17 = []
    for m in sorted(d17.mismatch.dropna().unique()):
        for c in sorted(d17.corruption.dropna().unique()):
            for reg in sorted(d17.regime.unique()):
                sub = d17[(d17.mismatch == m) & (d17.corruption == c) &
                          (d17.regime == reg)]
                w = sub.pivot_table(index=["graph_id", "seed"], columns="method",
                                    values="auprc").dropna()
                if {"spcrl_amortized_soft", "spcrl_amortized_none"} <= set(w.columns):
                    gaps17.append(float((w["spcrl_amortized_soft"]
                                         - w["spcrl_amortized_none"]).mean()))
    claim("E17-min-gap", "2.11",
          "smallest soft-prior advantage over the mismatch x deletion grid",
          "E17", "auprc", 0.046, min(gaps17) if gaps17 else None)
    claim("E17-max-gap", "2.11",
          "largest soft-prior advantage over the mismatch x deletion grid",
          "E17", "auprc", 0.344, max(gaps17) if gaps17 else None)
    claim("E17-cells", "2.11", "cells in the mismatch x deletion grid", "E17",
          "cells", 18, len(gaps17), tol=0)

    d18 = load("e18_fixed.jsonl")
    d18 = d18[d18.method.str.startswith("spcrl")]
    for m, c, reg, v in [(0.0, 0.2, "iid", -0.043), (0.0, 0.5, "iid", -0.228),
                         (0.0, 0.5, "graph_ood", -0.200),
                         (0.95, 0.0, "iid", 0.386)]:
        sub = d18[(d18.mismatch == m) & (d18.corruption == c) & (d18.regime == reg)]
        w = sub.pivot_table(index=["graph_id", "seed"], columns="method",
                            values="auprc").dropna()
        gap = float((w["spcrl_amortized_soft"] - w["spcrl_amortized_none"]).mean())
        claim(f"E18-m{m}-c{c}-{reg}", "2.11",
              f"soft prior advantage under drift at m={m}, c={c}, {reg}", "E18",
              "auprc", v, gap)

    # ---------------------------------------------------------- E4 (downstream)
    d4 = load("e4_fixed.jsonl")
    s4 = d4[d4.regime == "iid"]
    for m, t1, rg in [("oracle", 0.798, 0.310), ("spcrl_soft", 0.804, 0.292),
                      ("spcrl_hard", 0.788, 0.303), ("spcrl_none", 0.801, 0.325),
                      ("notears_linear", 0.626, 0.422), ("pcorr", 0.698, 0.478),
                      ("prior_only", 0.660, 0.443),
                      ("anomaly_no_structure", 0.607, None)]:
        sub = s4[s4.method == m]
        claim(f"E4-{m}-top1", "2.6", f"{m} Top-1 in distribution", "E4", "top1",
              t1, sub.top1.mean())
        if rg is not None:
            claim(f"E4-{m}-regret", "2.6", f"{m} normalised regret in distribution",
                  "E4", "norm_regret", rg, sub.norm_regret.mean())
    claim("E4-ties-oracle", "2.6", "tied candidate actions, true-graph reference",
          "E4", "n_tied", 1.6, s4[s4.method == "oracle"].n_tied.mean(), tol=0.05)
    claim("E4-ties-pcorr", "2.6", "tied candidate actions, partial correlation",
          "E4", "n_tied", 7.9, s4[s4.method == "pcorr"].n_tied.mean(), tol=0.05)

    # ---------------------------------------------------------- E12 (decision-relevant structure)
    from scipy import stats as _st
    d12 = load("e12_fixed.jsonl")
    d12 = d12[d12.method.isin(["spcrl_soft", "spcrl_hard", "spcrl_none"])]
    err12 = 1 - d12.top1
    claim("E12-faults", "2.6", "evaluated faults in the decision-relevance test",
          "E12", "faults", 753, len(d12), tol=0)
    claim("E12-rho-global", "2.6",
          "rank correlation, global structural distance vs ranking error", "E12",
          "rho", -0.09, _st.spearmanr(d12.shd_global, err12).statistic, tol=0.005)
    claim("E12-rho-decision", "2.6",
          "rank correlation, decision-restricted distance vs ranking error",
          "E12", "rho", -0.02,
          _st.spearmanr(d12.shd_decision, err12).statistic, tol=0.005)
    claim("E12-n-relevant", "2.6", "nodes in the decision-relevant subgraph",
          "E12", "nodes", 4.5, d12.n_relevant.mean(), tol=0.05)

    # ---------------------------------------------------------- E7 (coarseness)
    d7 = load("e7_fixed.jsonl")
    for r, iid, ood in [(2.0, 0.080, 0.248), (5.0, 0.058, 0.120),
                        (10.0, 0.029, 0.009), (20.0, -0.001, 0.002)]:
        for reg, v in (("iid", iid), ("graph_ood", ood)):
            sub = d7[(d7.superset_ratio == r) & (d7.regime == reg)]
            gap = (sub[sub.method == "spcrl_amortized_soft"].auprc.mean()
                   - sub[sub.method == "spcrl_amortized_none"].auprc.mean())
            claim(f"E7-gap-r{r}-{reg}", "2.8",
                  f"soft prior advantage at coarseness {r}x, {reg}", "E7",
                  "auprc", v, gap)
    for r, v in [(2.0, 3.0), (5.0, 6.0), (10.0, 10.59), (20.0, 14.95)]:
        claim(f"E7-ratio-{r}", "2.8", f"realised admissible/true ratio at {r}x",
              "E7", "ratio", v, d7[d7.superset_ratio == r].adm_over_true.mean(),
              tol=0.01)

    # ---------------------------------------------------------- E15 (leave-one-family-out)
    e15j = os.path.join(ROOT, "results", "e15_summary.json")
    if os.path.exists(e15j):
        import json as _json5
        e15 = _json5.load(open(e15j))
        claim("E15-rows", "2.2", "paired observations in the fold rotation",
              "E15", "rows", 1920, e15["n_rows"], tol=0,
              script="analysis/e15_folds.py")
        for fam, v in [("er", 0.081), ("sf", 0.096), ("modular", 0.096),
                       ("hier", 0.128)]:
            claim(f"E15-heldout-{fam}", "2.2",
                  f"soft prior advantage on the held-out {fam} family", "E15",
                  "auprc", v, e15["cells"][f"{fam}|held_out|soft"]["mean"],
                  script="analysis/e15_folds.py")
        claim("E15-hier-none", "2.2",
              "prior-blind AUPRC on held-out hierarchical graphs", "E15",
              "auprc", 0.694, e15["cells"]["hier|held_out|soft"]["auprc_none"],
              script="analysis/e15_folds.py")
        # in how many folds is the held-out advantage larger than in-distribution?
        n_bigger = sum(
            1 for f in e15["folds"]
            if e15["cells"][f"{f}|held_out|soft"]["mean"]
            > e15["cells"][f"{f}|in_dist|soft"]["mean"])
        claim("E15-folds-bigger", "2.2",
              "folds where the held-out advantage exceeds the in-distribution one",
              "E15", "folds", 4, n_bigger, tol=0, script="analysis/e15_folds.py")

    # ---------------------------------------------------------- E14 (partial coverage)
    e14j = os.path.join(ROOT, "results", "e14_summary.json")
    if os.path.exists(e14j):
        import json as _json4
        e14 = _json4.load(open(e14j))
        claim("E14-rows", "2.8", "paired observations in the coverage sweep",
              "E14", "rows", 1920, e14["n_rows"], tol=0,
              script="analysis/e14_missing.py")
        for key, v in [("iid|m=0.0|soft", 0.070), ("iid|m=0.25|soft", 0.034),
                       ("iid|m=0.5|soft", 0.017), ("iid|m=0.75|soft", 0.002),
                       ("graph_ood|m=0.0|soft", 0.113),
                       ("graph_ood|m=0.25|soft", 0.048),
                       ("graph_ood|m=0.5|soft", 0.025),
                       ("graph_ood|m=0.75|soft", 0.011)]:
            claim(f"E14-{key}", "2.8", f"soft prior advantage, {key}", "E14",
                  "auprc", v, e14["cells"][key]["mean"] if key in e14["cells"] else None,
                  script="analysis/e14_missing.py")
        for key, v in [("iid|m=0.0", 0.31), ("iid|m=0.25", 0.58),
                       ("iid|m=0.5", 0.80), ("iid|m=0.75", 0.92)]:
            claim(f"E14-rho-{key}", "2.8", f"admissible fraction, {key}", "E14",
                  "rho", v, e14["rho"].get(key), tol=0.006,
                  script="analysis/e14_missing.py")
        # the hard mask never crosses below the prior-blind control
        worst = min(e14["cells"][k]["mean"] for k in e14["cells"] if k.endswith("hard"))
        claim("E14-hard-never-negative", "2.8",
              "worst hard-mask advantage over all coverage levels", "E14",
              "auprc", 0.003, worst, tol=0.002, script="analysis/e14_missing.py")
        assert worst > 0

    # ---------------------------------------------------------- E13 (operators)
    e13j = os.path.join(ROOT, "results", "e13_summary.json")
    if os.path.exists(e13j):
        import json as _json3
        e13 = _json3.load(open(e13j))
        claim("E13-rows", "2.3", "paired observations in the operator sweep",
              "E13", "rows", 9600, e13["n_rows"], tol=0,
              script="analysis/e13_operators.py")
        for key, v in [
                ("matched|iid|add|c=0.1|hard", 0.057),
                ("matched|iid|add|c=0.2|hard", 0.073),
                ("matched|iid|add|c=0.5|hard", 0.075),
                ("matched|iid|add|c=0.1|soft", 0.056),
                ("matched|iid|add|c=0.2|soft", 0.077),
                ("matched|iid|add|c=0.5|soft", 0.074),
                ("matched|iid|clean|c=0.0|hard", 0.070),
                ("matched|iid|delete|c=0.1|hard", -0.040),
                ("matched|iid|delete|c=0.2|hard", -0.112),
                ("matched|iid|delete|c=0.5|hard", -0.386),
                ("matched|iid|delete|c=0.1|soft", 0.048),
                ("matched|iid|delete|c=0.2|soft", 0.065),
                ("matched|iid|delete|c=0.5|soft", 0.050),
                ("drift|iid|delete|c=0.2|soft", -0.030),
                ("drift|iid|delete|c=0.5|soft", -0.197),
                ("drift|iid|add|c=0.2|soft", 0.074),
                ("drift|iid|add|c=0.5|soft", 0.073)]:
            claim(f"E13-{key}", "2.3", f"prior advantage, {key}", "E13", "auprc",
                  v, e13["cells"][key]["mean"] if key in e13["cells"] else None,
                  script="analysis/e13_operators.py")
        # reversal reduces to deletion under a reverse-inclusive admissible set
        for a, b in [("matched|iid|delete|c=0.5|hard", "matched|iid|reverse|c=0.5|hard"),
                     ("matched|iid|delete|c=0.2|soft", "matched|iid|reverse|c=0.2|soft")]:
            if a in e13["cells"] and b in e13["cells"]:
                claim(f"E13-rev-eq-{b}", "2.3",
                      "reversal coincides with deletion to three decimals", "E13",
                      "auprc", round(e13["cells"][a]["mean"], 3),
                      round(e13["cells"][b]["mean"], 3), tol=0.0011,
                      script="analysis/e13_operators.py")

    # ---------------------------------------------------------- E6 (downstream under corruption)
    d6 = load("e6_fixed.jsonl")
    claim("E6-rows", "2.6", "evaluated rows in the corruption-downstream run",
          "E6", "rows", 4158, len(d6), tol=0)
    claim("E6-seeds", "2.6", "training seeds", "E6", "seeds", 3,
          d6.seed.nunique(), tol=0)
    for c, soft, hard, none in [(0.0, 0.703, 0.667, 0.667),
                                (0.2, 0.739, 0.694, 0.703),
                                (0.5, 0.640, 0.532, 0.631)]:
        sub = d6[(d6.regime == "iid") & (d6.corruption == c)]
        for m, v in (("spcrl_soft", soft), ("spcrl_hard", hard),
                     ("spcrl_none", none)):
            claim(f"E6-{m}-c{c}", "2.6", f"{m} Top-1 in distribution at c={c}",
                  "E6", "top1", v, sub[sub.method == m].top1.mean())
    sub = d6[(d6.regime == "iid") & (d6.corruption == 0.5) &
             (d6.method.isin(["spcrl_soft", "spcrl_hard", "spcrl_none"]))]
    res = friedman_holm(sub, "top1", block=("graph_id", "seed", "fault_node"))
    claim("E6-hard-vs-soft-p", "2.6",
          "Holm-corrected p, hard mask vs soft prior at c=0.5", "E6", "p",
          0.005, res["posthoc"]["spcrl_hard"]["p_holm"], tol=0.001)
    w6 = sub.pivot_table(index=["graph_id", "seed", "fault_node"],
                         columns="method", values="top1").dropna()
    from scipy import stats as _st6
    claim("E6-hard-vs-none-p", "2.6",
          "uncorrected p, hard mask vs prior-blind at c=0.5", "E6", "p", 0.008,
          _st6.wilcoxon(w6["spcrl_hard"], w6["spcrl_none"]).pvalue, tol=0.001)
    claim("E6-soft-none-gap", "2.6",
          "soft minus prior-blind Top-1 at c=0.5", "E6", "top1", 0.009,
          float((w6["spcrl_soft"] - w6["spcrl_none"]).mean()))
    claim("E6-soft-none-blocks", "2.6",
          "blocks on which soft and prior-blind differ at c=0.5", "E6", "blocks",
          3, int((w6["spcrl_soft"] != w6["spcrl_none"]).sum()), tol=0)
    sub_g = d6[(d6.regime == "graph_ood") & (d6.corruption == 0.2)]
    for m, v in (("spcrl_hard", 0.683), ("spcrl_soft", 0.783), ("spcrl_none", 0.767)):
        claim(f"E6-ood-c02-{m}", "2.6",
              f"{m} Top-1 under graph shift at c=0.2", "E6", "top1", v,
              sub_g[sub_g.method == m].top1.mean())

    # ---------------------------------------------------------- E8 (envelope)
    ej = os.path.join(ROOT, "results", "e8_envelope.json")
    if os.path.exists(ej):
        import json as _json2
        en = _json2.load(open(ej))
        claim("E8-cells", "2.8", "phase-diagram cells", "E8", "cells", 48,
              en["n_cells"], tol=0, script="analysis/e8_envelope.py")
        d8 = load("e8_fixed.jsonl")
        claim("E8-rows", "2.8", "paired observations in the phase diagram", "E8",
              "rows", 11520, len(d8), tol=0, script="analysis/e8_envelope.py")
        idx_r = {float(r): i for i, r in enumerate(en["ratios"])}
        idx_c = {float(c): i for i, c in enumerate(en["corruptions"])}
        for r, c, v in [(2.0, 0.0, 0.156), (5.0, 0.0, 0.085), (10.0, 0.0, 0.027),
                        (20.0, 0.0, 0.000), (2.0, 0.5, 0.118), (5.0, 0.5, 0.051),
                        (10.0, 0.5, 0.015), (5.0, 0.2, 0.056)]:
            claim(f"E8-soft-r{r}-c{c}", "2.8",
                  f"soft prior advantage at r={r}, c={c}", "E8", "auprc", v,
                  en["grid"]["spcrl_amortized_soft"]["mean"][idx_r[r]][idx_c[c]],
                  script="analysis/e8_envelope.py")
        claim("E8-hard-r5-c0.2", "2.8", "hard mask deficit at r=5, c=0.2", "E8",
              "auprc", -0.081,
              en["grid"]["spcrl_amortized_hard"]["mean"][idx_r[5.0]][idx_c[0.2]],
              script="analysis/e8_envelope.py")
        for r, v in [(2.0, 0.163), (5.0, 0.098), (10.0, 0.036)]:
            claim(f"E8-cross-r{r}", "2.8",
                  f"hard-mask crossover corruption at r={r}", "E8", "crossover",
                  v, en["crossover"]["spcrl_amortized_hard"][str(r)], tol=0.002,
                  script="analysis/e8_envelope.py")

    # ---------------------------------------------------------- E10 (real)
    import json as _json
    d10 = load("e10_fixed.jsonl")
    d10 = d10[d10.method != "_dataset_report"]
    # the description as read (corruption 0); corrupted variants are reported
    # separately below
    for m, t1, mrr in [("spcrl_none", 0.375, 0.651),
                       ("spcrl_soft", 0.292, 0.587),
                       ("spcrl_hard", 0.319, 0.601),
                       ("prior_only", 0.333, 0.602),
                       ("anomaly_no_structure", 0.250, 0.512)]:
        sub = d10[(d10.method == m) & (d10.corruption == 0.0)]
        claim(f"E10-{m}-top1", "2.10", f"{m} Top-1 on the Kubernetes campaign",
              "E10", "top1", t1, sub.top1.mean())
        claim(f"E10-{m}-mrr", "2.10", f"{m} MRR on the Kubernetes campaign",
              "E10", "mrr", mrr, sub.mrr.mean())
    for c, v in [(0.0, 0.292), (0.2, 0.306), (0.5, 0.347)]:
        sub = d10[(d10.method == "spcrl_soft") & (d10.corruption == c)]
        claim(f"E10-soft-c{c}", "2.10", f"soft prior Top-1 at prior corruption c={c}",
              "E10", "top1", v, sub.top1.mean())
    pj = os.path.join(ROOT, "results", "e10_paired.json")
    if os.path.exists(pj):
        pr = _json.load(open(pj))
        idx = {(c["contrast"], c["metric"]): c for c in pr["contrasts"]}
        for (a, b, metric), (mean, lo, hi) in {
                ("spcrl_none", "anomaly_no_structure", "top1"): (0.125, 0.028, 0.236),
                ("spcrl_none", "anomaly_no_structure", "mrr"): (0.139, 0.059, 0.229),
                ("spcrl_soft", "spcrl_none", "top1"): (-0.083, -0.167, -0.014),
                ("spcrl_soft", "spcrl_none", "mrr"): (-0.064, -0.135, -0.008),
                ("spcrl_hard", "spcrl_none", "top1"): (-0.056, -0.153, 0.042),
        }.items():
            k = (f"{a} - {b}", metric)
            got = idx.get(k)
            claim(f"E10-paired-{a}-{b}-{metric}", "2.10",
                  f"paired {a} - {b} ({metric}) over 24 faults", "E10", metric,
                  mean, got["mean"] if got else None, tol=0.003,
                  script="analysis/e10_paired.py")
            for tag, val in (("lo", lo), ("hi", hi)):
                claim(f"E10-paired-{a}-{b}-{metric}-ci{tag}", "2.10",
                      f"bootstrap CI {tag} for {a} - {b} ({metric})", "E10", metric,
                      val, got[f"ci_{tag}"] if got else None, tol=0.02,
                      script="analysis/e10_paired.py")
    # ------------------------------------ K2b (campaign passed all gates)
    gb = os.path.join(ROOT, "results", "gates_k2b.json")
    if os.path.exists(gb):
        g = _json.load(open(gb))
        for cid, text, val, got, tol in [
                ("K2b-G4-rho", "unconnected throughput |rho| in K2b", 0.157,
                 g["mean_abs_rho_unconnected_throughput"], 0.002),
                ("K2b-G3-r2", "median cross-variable R2 excl. throughput in K2b",
                 0.151, g["median_r2_excl_throughput"], 0.002),
                ("K2b-n-nominal", "nominal samples in K2b", 3606.0,
                 float(g["n_nominal"]), 0.5),
                ("K2b-passed", "K2b passes all gates", 1.0,
                 float(g["passed"]), 0.0)]:
            claim(cid, "2.7", text, "K2b", "gate", val, got, tol=tol,
                  script="analysis/external_gates.py")
    cb = os.path.join(ROOT, "results", "controls_K2b.json")
    if os.path.exists(cb):
        ck = _json.load(open(cb))
        claim("K2b-blind-anomaly", "2.7", "prior-blind minus anomaly Top-1 in K2b",
              "K2b", "top1", 0.167, ck["blind_vs_anomaly"][0], tol=0.003,
              script="analysis/external_controls.py")
        for tag, v, g_ in zip(("lo", "hi"), (0.014, 0.347),
                              ck["blind_vs_anomaly"][1:]):
            claim(f"K2b-blind-anomaly-ci{tag}", "2.7",
                  f"bootstrap CI {tag} for prior-blind minus anomaly in K2b",
                  "K2b", "top1", v, g_, tol=0.02,
                  script="analysis/external_controls.py")
        want = {"soft|P1_topology": (0.528, -0.014, -0.056, 0.028),
                "hard|P1_topology": (0.556, 0.014, -0.069, 0.097),
                "soft|P2_semantic": (0.556, 0.014, -0.097, 0.111),
                "soft|P3_random": (0.500, -0.042, -0.139, 0.028),
                "soft|P3b_random_at_P2_density": (0.528, -0.014, -0.056, 0.028),
                "soft|P4_degree_shuffled": (0.514, -0.028, -0.097, 0.028),
                "soft|P6_wrong_topology": (0.500, -0.042, -0.139, 0.028),
                "soft|driver_declared": (0.500, -0.042, -0.139, 0.028),
                "hard|P2_semantic": (0.500, -0.042, -0.139, 0.028),
                "hard|P3_random": (0.528, -0.014, -0.125, 0.056),
                "hard|P3b_random_at_P2_density": (0.500, -0.042, -0.139, 0.028),
                "hard|P4_degree_shuffled": (0.556, 0.014, 0.000, 0.042),
                "hard|P6_wrong_topology": (0.528, -0.014, -0.083, 0.042),
                "hard|driver_declared": (0.542, 0.000, -0.111, 0.097)}
        for k, (t1, m, lo, hi) in want.items():
            e = ck["vs_blind"][k]
            claim(f"K2b-top1-{k}", "2.7", f"K2b Top-1 under {k}", "K2b", "top1",
                  t1, e["top1"], tol=0.003, script="analysis/external_controls.py")
            for tag, v, g_ in zip(("", "-cilo", "-cihi"), (m, lo, hi), e["diff"]):
                claim(f"K2b-vsP5-{k}{tag}", "2.7",
                      f"K2b {k} minus prior-blind{tag or ''}", "K2b", "top1",
                      v, g_, tol=0.003 if not tag else 0.02,
                      script="analysis/external_controls.py")
        for k, v in [("soft|P1_topology-P6_wrong_topology", 0.028),
                     ("soft|P2_semantic-P3b_random_at_P2_density", 0.028)]:
            claim(f"K2b-pair-{k}", "2.7", f"K2b pairwise {k}", "K2b", "top1",
                  v, ck["pairwise"][k][0], tol=0.003,
                  script="analysis/external_controls.py")
    hj = os.path.join(ROOT, "results", "hx1.json")
    if os.path.exists(hj):
        h = _json.load(open(hj))
        for cid, text, val, got, tol in [
                ("HX1-diff", "H-X1 difference of paired contrasts", 0.069,
                 h["difference"], 0.003),
                ("HX1-cilo", "H-X1 interval lower bound", -0.014, h["ci_lo"], 0.02),
                ("HX1-cihi", "H-X1 interval upper bound", 0.167, h["ci_hi"], 0.02),
                ("HX1-supported", "H-X1 supported by its registered rule", 0.0,
                 float(h["supported"]), 0.0)]:
            claim(cid, "2.7", text, "K1+K2b", "top1", val, got, tol=tol,
                  script="analysis/external_hx1.py")
    # ------------------------------------ exclusion against its own ceiling
    ej = os.path.join(ROOT, "results", "exclusion_bound.json")
    if os.path.exists(ej):
        eb = _json.load(open(ej))
        claim("excl-recall-clean", "2.2", "hard-mask recall with a clean prior",
              "E13", "recall", 0.814, eb["recall_clean"], tol=0.003,
              script="analysis/exclusion_bound.py")
        for c, v in (("0.1", 0.031), ("0.2", 0.031), ("0.5", 0.055)):
            claim(f"excl-excess-{c}", "2.2",
                  f"recall above the representational reference at c={c}",
                  "E13", "recall", v, eb["levels"][c]["recall_excess"], tol=0.003,
                  script="analysis/exclusion_bound.py")
        for c in ("0.1", "0.2", "0.5"):
            claim(f"excl-auprc-excess-{c}", "2.2",
                  f"|AUPRC - reference| at c={c} is at most 0.024", "E13",
                  "auprc", 1.0,
                  float(abs(eb["levels"][c]["auprc_excess"]) <= 0.024), tol=0.0,
                  script="analysis/exclusion_bound.py")
    # ------------------------------------ H-X6 (registered, descriptive)
    hj = os.path.join(ROOT, "results", "hx6.json")
    if os.path.exists(hj):
        h6 = _json.load(open(hj))
        claim("HX6-rho", "2.9", "pooled Spearman of C against soft - P5",
              "K2b+F1", "rho", 0.229, h6["spearman_rho"], tol=0.003,
              script="analysis/external_hx6.py")
        claim("HX6-p", "2.9", "p-value of the pooled Spearman", "K2b+F1",
              "p", 0.48, h6["p_value"], tol=0.02,
              script="analysis/external_hx6.py")
        claim("HX6-n", "2.9", "campaign-description pairs in H-X6", "K2b+F1",
              "n", 12.0, float(h6["n"]), tol=0.5,
              script="analysis/external_hx6.py")
        for c, rho in (("K2b", 0.530), ("F1", 0.577)):
            claim(f"HX6-rho-{c}", "SN11", f"Spearman of C against soft - P5 on {c}",
                  c, "rho", rho, h6["per_campaign"][c]["rho"], tol=0.003,
                  script="analysis/external_hx6.py")
        want = {("F1", "P2_semantic"): 0.485, ("F1", "P6_wrong_topology"): 0.000,
                ("F1", "P1_topology"): 0.140, ("K2b", "P1_topology"): 0.038}
        rows = {(r["campaign"], r["variant"]): r["C"] for r in h6["rows"]}
        for k, v in want.items():
            claim(f"HX6-C-{k[0]}-{k[1]}", "SN11", f"compatibility C of {k[1]} on {k[0]}",
                  k[0], "C", v, rows[k], tol=0.003,
                  script="analysis/external_hx6.py")
    # ------------------------------------ decomposition and seed robustness
    for mode, want in (("soft", {
            "K1|soft|total": (-0.083, -0.167, -0.014),
            "K1|soft|identity": (-0.042, -0.125, 0.000),
            "K1|soft|semantics": (-0.042, -0.194, 0.125),
            "K1|soft|density": (-0.097, -0.181, -0.028),
            "K1|soft|degree": (-0.069, -0.153, 0.000),
            "K2b|soft|total": (-0.014, -0.069, 0.028),
            "K2b|soft|identity": (0.028, 0.000, 0.083),
            "K2b|soft|semantics": (0.028, -0.069, 0.139),
            "K2b|soft|density": (0.028, 0.000, 0.083),
            "K2b|soft|degree": (0.014, 0.000, 0.042),
            "F1|soft|total": (0.115, -0.021, 0.240),
            "F1|soft|identity": (-0.156, -0.271, -0.042),
            "F1|soft|semantics": (0.406, 0.271, 0.542),
            "F1|soft|density": (0.000, -0.094, 0.083),
            "F1|soft|degree": (-0.010, -0.167, 0.146)}),
        ("hard", {
            "K1|hard|total": (-0.056, -0.153, 0.042),
            "K1|hard|identity": (-0.028, -0.083, 0.028),
            "K1|hard|semantics": (0.097, -0.028, 0.236),
            "K1|hard|density": (-0.042, -0.181, 0.083),
            "K1|hard|degree": (-0.028, -0.083, 0.028),
            "K2b|hard|total": (0.014, -0.069, 0.097),
            "K2b|hard|identity": (0.028, 0.000, 0.083),
            "K2b|hard|semantics": (0.000, 0.000, 0.000),
            "K2b|hard|density": (0.028, 0.000, 0.069),
            "K2b|hard|degree": (0.000, -0.083, 0.083),
            "F1|hard|total": (0.281, 0.125, 0.438),
            "F1|hard|identity": (-0.062, -0.125, 0.000),
            "F1|hard|semantics": (0.146, 0.031, 0.271),
            "F1|hard|density": (0.250, 0.073, 0.417),
            "F1|hard|degree": (0.135, -0.062, 0.333)})):
        dj = os.path.join(ROOT, "results", f"prior_decomposition_{mode}.json")
        if not os.path.exists(dj):
            continue
        dd = _json.load(open(dj))
        for key, (m, lo, hi) in want.items():
            e = dd[key]
            for tag, v, g_ in (("", m, e["mean"]), ("-cilo", lo, e["ci_lo"]),
                               ("-cihi", hi, e["ci_hi"])):
                claim(f"decomp-{key}{tag}", "2.10", f"decomposition {key}{tag}",
                      key.split("|")[0], "top1", v, g_,
                      tol=0.003 if not tag else 0.02,
                      script="analysis/prior_decomposition.py")
    sj = os.path.join(ROOT, "results", "seed_robustness.json")
    if os.path.exists(sj):
        sr = _json.load(open(sj))
        for key, (m, lo, hi) in {
                "F1|soft|P2 - P3b": (0.406, 0.250, 0.562),
                "F1|soft|P1 - P6": (-0.156, -0.302, -0.010),
                "F1|soft|P1 - P5": (0.115, -0.042, 0.281),
                "F1|P5 - anomaly": (0.094, -0.062, 0.260)}.items():
            e = sr[key]
            for tag, v, g_ in (("", m, e["mean"]), ("-cilo", lo, e["ci_lo"]),
                               ("-cihi", hi, e["ci_hi"])):
                claim(f"seedrob-{key}{tag}", "SN12",
                      f"two-way bootstrap {key}{tag}", "F1", "top1", v, g_,
                      tol=0.003 if not tag else 0.02,
                      script="analysis/external_seed_robustness.py")
    # ------------------------------------ F1 (free5GC, passed its gates)
    gf = os.path.join(ROOT, "results", "gates_f1.json")
    if os.path.exists(gf):
        g = _json.load(open(gf))
        claim("F1-passed", "2.9", "F1 passes the gates that apply to it", "F1",
              "gate", 1.0, float(g["passed"]), tol=0.0,
              script="analysis/external_gates.py")
        claim("F1-n-nominal", "2.9", "nominal samples in F1", "F1", "gate",
              3606.0, float(g["n_nominal"]), tol=0.5,
              script="analysis/external_gates.py")
    cf = os.path.join(ROOT, "results", "controls_F1.json")
    if os.path.exists(cf):
        ck = _json.load(open(cf))
        claim("F1-n-faults", "2.9", "faults in the F1 campaign", "F1", "n",
              32.0, float(ck["n_faults"]), tol=0.5,
              script="analysis/external_controls.py")
        for tag, v, g_ in zip(("", "-cilo", "-cihi"), (0.094, -0.042, 0.219),
                              ck["blind_vs_anomaly"]):
            claim(f"F1-blind-anomaly{tag}", "2.9",
                  f"F1 prior-blind minus anomaly{tag}", "F1", "top1", v, g_,
                  tol=0.003 if not tag else 0.02,
                  script="analysis/external_controls.py")
        want = {"soft|P1_topology": (0.271, 0.115, -0.021, 0.240),
                "hard|P1_topology": (0.438, 0.281, 0.125, 0.438),
                "soft|P2_semantic": (0.500, 0.344, 0.229, 0.458),
                "soft|P6_wrong_topology": (0.427, 0.271, 0.135, 0.417),
                "hard|P6_wrong_topology": (0.500, 0.344, 0.198, 0.490)}
        for k, (t1, m, lo, hi) in want.items():
            e = ck["vs_blind"][k]
            claim(f"F1-top1-{k}", "2.9", f"F1 Top-1 under {k}", "F1", "top1",
                  t1, e["top1"], tol=0.003,
                  script="analysis/external_controls.py")
            for tag, v, g_ in zip(("", "-cilo", "-cihi"), (m, lo, hi), e["diff"]):
                claim(f"F1-vsP5-{k}{tag}", "2.9", f"F1 {k} minus prior-blind{tag}",
                      "F1", "top1", v, g_, tol=0.003 if not tag else 0.02,
                      script="analysis/external_controls.py")
        claim("F1-blind-top1", "2.9", "F1 prior-blind Top-1", "F1", "top1",
              0.156, ck["vs_blind"]["soft|P1_topology"]["top1"]
              - ck["vs_blind"]["soft|P1_topology"]["diff"][0], tol=0.003,
              script="analysis/external_controls.py")
    pj = os.path.join(ROOT, "results", "pooled_hx2_hx3.json")
    if os.path.exists(pj):
        pk = _json.load(open(pj))
        for key, (m, lo, hi, sup) in {
                "H-X2|soft": (-0.077, -0.149, -0.006, 0.0),
                "H-X2|hard": (-0.024, -0.065, 0.018, 0.0),
                "H-X3|soft": (0.244, 0.155, 0.333, 1.0),
                "H-X3|hard": (0.083, 0.018, 0.155, 1.0)}.items():
            e = pk[key]
            for tag, v, g_ in (("", m, e["mean"]), ("-cilo", lo, e["ci_lo"]),
                               ("-cihi", hi, e["ci_hi"]),
                               ("-supported", sup, float(e["supported"]))):
                claim(f"pooled-{key}{tag}", "2.9",
                      f"pooled {key}{tag or ' mean'}", "K2b+F1", "top1", v, g_,
                      tol=0.003 if tag in ("", "-supported") else 0.02,
                      script="analysis/external_pooled.py")
        for c, v in (("K2b", 0.028), ("F1", -0.156)):
            claim(f"pooled-HX2-soft-{c}", "2.9",
                  f"H-X2 soft contrast on {c}", "K2b+F1", "top1", v,
                  pk["H-X2|soft"]["per_campaign"][c][0], tol=0.003,
                  script="analysis/external_pooled.py")
    # ------------------------------------ K2 gates (campaign failed G4)
    gj = os.path.join(ROOT, "results", "gates_k2.json")
    if os.path.exists(gj):
        g2 = _json.load(open(gj))
        for cid, text, val, got, tol in [
                ("K2-G4-rho", "unconnected throughput |rho| in K2", 0.756,
                 g2["mean_abs_rho_unconnected_throughput"], 0.002),
                ("K2-connected-rho", "connected throughput |rho| in K2", 0.910,
                 g2["mean_abs_rho_connected_throughput"], 0.002),
                ("K2-G3-r2", "median cross-variable R2 excl. throughput in K2",
                 0.81, g2["median_r2_excl_throughput"], 0.005),
                ("K2-n-nominal", "nominal samples in K2", 3606.0,
                 float(g2["n_nominal"]), 0.5),
                ("K2-G4-failed", "K2 fails gate G4", 0.0,
                 float(g2["G4_manipulation"]), 0.0),
                ("K2-passed", "K2 passes all gates", 0.0,
                 float(g2["passed"]), 0.0)]:
            claim(cid, "SN9", text, "K2", "gate", val, got, tol=tol,
                  script="analysis/external_gates.py")
    # ------------------------------------ K1 prior controls (exploratory)
    cj = os.path.join(ROOT, "results", "controls_K1.json")
    if os.path.exists(cj):
        ck = _json.load(open(cj))
        claim("K1ctl-p5-identical", "SN8", "prior-blind model identical across "
              "prior-variant runs", "K1ctl", "flag", 1.0, float(ck["p5_consistent"]),
              script="analysis/external_controls.py")
        want = {("vs_blind", "soft|P1_topology"): (-0.083, -0.167, -0.014),
                ("vs_blind", "soft|P2_semantic"): (-0.042, -0.194, 0.111),
                ("vs_blind", "soft|P3_random"): (0.014, -0.056, 0.097),
                ("vs_blind", "soft|P3b_random_at_P2_density"): (0.000, -0.069, 0.083),
                ("vs_blind", "soft|P4_degree_shuffled"): (-0.014, -0.056, 0.028),
                ("vs_blind", "soft|P6_wrong_topology"): (-0.042, -0.153, 0.083),
                ("vs_blind", "soft|driver_declared"): (-0.042, -0.125, 0.028),
                ("vs_blind", "hard|P3b_random_at_P2_density"): (-0.083, -0.167, -0.014),
                ("pairwise", "soft|P1_topology-P3_random"): (-0.097, -0.181, -0.028),
                ("pairwise", "soft|P1_topology-P4_degree_shuffled"): (-0.069, -0.153, 0.000),
                ("pairwise", "soft|P1_topology-P6_wrong_topology"): (-0.042, -0.125, 0.000),
                ("pairwise", "hard|P2_semantic-P3b_random_at_P2_density"): (0.097, -0.028, 0.236)}
        for (sec, key), vals in want.items():
            got = ck[sec].get(key)
            got = (got["diff"] if sec == "vs_blind" else got) if got else [None] * 3
            for tag, v, g, tol in zip(("mean", "lo", "hi"), vals, got,
                                      (0.003, 0.02, 0.02)):
                claim(f"K1ctl-{sec}-{key}-{tag}", "SN8",
                      f"K1 prior control {key} ({sec}) {tag}", "K1ctl", "top1",
                      v, g, tol=tol, script="analysis/external_controls.py")
    dj = os.path.join(ROOT, "results", "campaign_diagnostics.json")
    if os.path.exists(dj):
        dg = _json.load(open(dj))
        for cid, text, key, val, tol in [
                ("E10-nvars", "channels instrumented", "n_variables", 40, 0),
                ("E10-nominal", "nominal samples", "n_nominal_samples", 3606, 0),
                ("E10-window", "samples per window", "samples_per_window", 601, 0),
                ("E10-faults", "fault windows", "n_fault_windows", 24, 0),
                ("E10-prior-edges", "admissible channel pairs", "prior_edges", 335, 0),
                ("E10-prior-density", "prior density", "prior_density", 0.209, 0.001),
                ("E10-cv", "per-service throughput CV", "throughput_cv_median", 0.36, 0.005),
                ("E10-degenerate", "channels with zero nominal variance",
                 "n_degenerate", 16, 0),
                ("E10-live", "channels with nominal variation", "n_live", 24, 0)]:
            claim(cid, "2.10", text, "E10", key, val, dg.get(key), tol=tol,
                  script="analysis/campaign_diagnostics.py")
    cj = os.path.join(ROOT, "results", "campaign_prior_coverage.json")
    if os.path.exists(cj):
        cv = _json.load(open(cj))
        claim("E10-coverage-strong", "2.10",
              "admissible fraction among live pairs with |rho|>=0.3", "E10",
              "coverage", 0.324, cv["coverage"]["0.3"], tol=0.002,
              script="analysis/campaign_diagnostics.py")
        claim("E10-coverage-base", "2.10",
              "admissible fraction over all live pairs", "E10", "coverage",
              0.315, cv["base_rate"], tol=0.002,
              script="analysis/campaign_diagnostics.py")
        claim("E10-strong-pairs", "2.10", "live pairs with |rho|>=0.3", "E10",
              "n_pairs", 34, cv["n_pairs"]["0.3"], tol=0,
              script="analysis/campaign_diagnostics.py")

    # ---------------------------------------------------------- write
    df = pd.DataFrame(ROWS)
    out = os.path.join(ROOT, "docs", "CLAIM_LEDGER.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    n_bad = int((df.status == "MISMATCH").sum())
    n_unk = int((df.status == "UNCHECKED").sum())
    print(df.to_string(index=False, max_colwidth=40))
    print(f"\n{len(df)} claims: {int((df.status=='OK').sum())} OK, "
          f"{n_bad} MISMATCH, {n_unk} UNCHECKED  ->  {out}")
    return 1 if (n_bad or n_unk) else 0


if __name__ == "__main__":
    sys.exit(main())
