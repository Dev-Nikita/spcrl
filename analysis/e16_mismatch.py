"""Relation mismatch (E16): does the prior advantage fall as the prior's
relation stops being the relation the data realise?

Pre-registered hypotheses are in docs/preregistration/PREREGISTRATION_E16.md. This script
reports, per mismatch level, the compatibility statistic C (a property of the
data and the prior, not of any model) and the paired advantage of each prior
mode over the prior-blind control, with bootstrap intervals over systems, and
the rank correlation between C and the advantage.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.load import load  # noqa: E402

B = 4000


def main(path="e16_fixed.jsonl", out="results/e16_summary.json"):
    d = load(path)
    comp = d[d.method == "_compatibility"]
    mod = d[d.method.str.startswith("spcrl_")]
    rng = np.random.default_rng(0)
    ms = sorted(float(m) for m in mod.mismatch.dropna().unique())
    rep = {"n_rows": int(len(d)), "mismatch_levels": ms,
           "seeds": sorted(float(s) for s in mod.seed.dropna().unique()),
           "C": {}, "auprc": {}, "advantage": {}}
    for m in ms:
        rep["C"][str(m)] = {
            "mean": float(comp[comp.mismatch == m].comp_C.mean()),
            "sd": float(comp[comp.mismatch == m].comp_C.std())}
        for regime in sorted(mod.regime.unique()):
            sub = mod[(mod.mismatch == m) & (mod.regime == regime)]
            w = sub.pivot_table(index=["graph_id", "seed"], columns="method",
                                values="auprc").dropna()
            for mode in ("soft", "hard", "soft_gate"):
                a, b = f"spcrl_amortized_{mode}", "spcrl_amortized_none"
                if a not in w.columns or b not in w.columns:
                    continue
                diff = (w[a] - w[b]).values
                bs = diff[rng.integers(0, len(diff), (B, len(diff)))].mean(1)
                rep["advantage"][f"{regime}|m={m}|{mode}"] = {
                    "mean": float(diff.mean()),
                    "ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                    "auprc": float(w[a].mean()), "auprc_none": float(w[b].mean()),
                    "n": int(len(diff))}
    # H1: does C fall with m?  H2: does the advantage fall with m?
    cs = [rep["C"][str(m)]["mean"] for m in ms]
    rep["H1_spearman_C_vs_m"] = float(stats.spearmanr(ms, cs).statistic)
    for regime in sorted(mod.regime.unique()):
        adv = [rep["advantage"][f"{regime}|m={m}|soft"]["mean"] for m in ms
               if f"{regime}|m={m}|soft" in rep["advantage"]]
        if len(adv) == len(ms):
            rep[f"H2_spearman_adv_vs_m_{regime}"] = float(
                stats.spearmanr(ms, adv).statistic)
            rep[f"H3_spearman_adv_vs_C_{regime}"] = float(
                stats.spearmanr(cs, adv).statistic)
    json.dump(rep, open(out, "w"), indent=2)
    print(f"rows={rep['n_rows']} seeds={rep['seeds']}")
    print("\n   m      C     | soft-none (iid)          | hard-none (iid)         "
          " | soft-none (graph_ood)")
    for m in ms:
        def cell(regime, mode):
            k = f"{regime}|m={m}|{mode}"
            if k not in rep["advantage"]:
                return " " * 24
            v = rep["advantage"][k]
            return f"{v['mean']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]"
        print(f"{m:5.2f}  {rep['C'][str(m)]['mean']:.3f}  | {cell('iid','soft')} |"
              f" {cell('iid','hard')} | {cell('graph_ood','soft')}")
    print("\nH1 rho(C, m)          =", round(rep["H1_spearman_C_vs_m"], 3))
    for k in rep:
        if k.startswith("H2_") or k.startswith("H3_"):
            print(f"{k:34s}=", round(rep[k], 3))


if __name__ == "__main__":
    main(*sys.argv[1:])
