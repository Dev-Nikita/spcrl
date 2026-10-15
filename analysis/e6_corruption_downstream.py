"""Does the corruption effect of Section 2.3 reach the repair decision? (E6)

The comparison is paired at the level of the evaluated fault: every method sees
the same system, the same injected fault and the same candidate repair set, and
only the estimated graph differs. Methods are compared within a corruption
level, and the whole analysis is repeated per regime, because the prior-blind
control is expected to behave differently under graph shift.

Written to results/e6_summary.json so that every number quoted in Section 2.6
is reproducible from the raw rows by code that did not produce them.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.load import load          # noqa: E402
from analysis.stats import friedman_holm  # noqa: E402

METHODS = ["spcrl_soft", "spcrl_hard", "spcrl_none", "oracle", "prior_only",
           "anomaly_no_structure"]


def main(path="e6_corrupt_downstream.jsonl", out="results/e6_summary.json"):
    d = load(path)
    d = d[d.method.isin(METHODS)]
    rep = {"n_rows": int(len(d)),
           "seeds": sorted(float(s) for s in d.seed.dropna().unique()),
           "corruptions": sorted(float(c) for c in d.corruption.dropna().unique()),
           "regimes": sorted(d.regime.dropna().unique()),
           "top1": {}, "regret": {}, "tests": {}}
    for regime in rep["regimes"]:
        for c in rep["corruptions"]:
            sub = d[(d.regime == regime) & (d.corruption == c)]
            key = f"{regime}|c={c}"
            rep["top1"][key] = {m: float(sub[sub.method == m].top1.mean())
                                for m in METHODS if (sub.method == m).any()}
            if "norm_regret" in sub.columns:
                rep["regret"][key] = {
                    m: float(sub[sub.method == m].norm_regret.mean())
                    for m in METHODS if (sub.method == m).any()}
            three = sub[sub.method.isin(["spcrl_soft", "spcrl_hard", "spcrl_none"])]
            res = friedman_holm(three, "top1", block=("graph_id", "seed", "fault_node")
                                if "fault_node" in three.columns else ("graph_id", "seed"))
            rep["tests"][key] = {
                k: v for k, v in res.items()
                if k in ("n_blocks", "friedman", "reference", "mean", "avg_rank")}
            if "posthoc" in res:
                rep["tests"][key]["p_holm"] = {
                    m: res["posthoc"][m]["p_holm"] for m in res["posthoc"]}
                rep["tests"][key]["min_p_holm"] = float(
                    min(res["posthoc"][m]["p_holm"] for m in res["posthoc"]))
    json.dump(rep, open(out, "w"), indent=2)
    print(f"rows={rep['n_rows']} seeds={rep['seeds']}")
    for key in rep["top1"]:
        t = rep["top1"][key]
        ph = rep["tests"][key].get("min_p_holm")
        print(f"\n{key}  n_blocks={rep['tests'][key].get('n_blocks')}"
              f"  min p(Holm)={ph if ph is None else round(ph, 3)}")
        print("  Top-1 " + "  ".join(f"{m.replace('spcrl_','')}={t[m]:.3f}" for m in t))
        if key in rep["regret"]:
            r = rep["regret"][key]
            print("  regret " + "  ".join(f"{m.replace('spcrl_','')}={r[m]:.3f}" for m in r))


if __name__ == "__main__":
    main(*sys.argv[1:])
