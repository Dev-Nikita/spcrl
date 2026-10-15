"""All numbers reported in the manuscript are produced here, from results/*.jsonl."""
import json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.load import load, load_multi, LABELS, REGIME
from analysis.stats import friedman_holm, bootstrap_ci

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "tables")
os.makedirs(OUT, exist_ok=True)


def _fmt(m, lo, hi):
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def table_main():
    df = load("e1_fixed.jsonl")
    regs = [r for r in ["iid", "mech_ood", "graph_ood", "size_ood", "noise_ood",
                        "joint_ood"] if r in set(df.regime)]
    rows = []
    for m in df.method.unique():
        row = {"method": LABELS.get(m, m)}
        for r in regs:
            v = df[(df.method == m) & (df.regime == r)].auprc.values
            if len(v) == 0:
                row[REGIME[r]] = "--"
            else:
                lo, hi = bootstrap_ci(v)
                row[REGIME[r]] = _fmt(float(np.mean(v)), lo, hi)
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "table1_main_auprc.csv"), index=False)
    stats = {}
    for r in regs:
        sub = df[(df.regime == r) & df.method.str.startswith("spcrl_amortized")]
        stats[r] = friedman_holm(sub, "auprc", block=("graph_id", "seed"))
    json.dump(stats, open(os.path.join(OUT, "table1_stats.json"), "w"),
              indent=2, default=float)
    return t


def table_ood_gap():
    df = load("e1_fixed.jsonl")
    base = df[df.regime == "iid"].groupby("method").auprc.mean()
    rows = []
    for m in df.method.unique():
        row = {"method": LABELS.get(m, m), "iid": round(float(base.get(m, np.nan)), 3)}
        for r in ["mech_ood", "graph_ood", "size_ood", "noise_ood", "joint_ood"]:
            v = df[(df.method == m) & (df.regime == r)].auprc.mean()
            row[f"delta_{r}"] = round(float(base.get(m, np.nan) - v), 3) if np.isfinite(v) else None
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "table2_ood_gap.csv"), index=False)
    return t


def crossover_point(protocol="matched", regime="iid", n_boot=2000, seed=0,
                    method="spcrl_amortized_hard", ref="spcrl_amortized_none"):
    """Corruption level at which the hard-mask model falls below the prior-free
    model, estimated by linear interpolation of the paired difference, with a
    bootstrap CI over systems."""
    df = load("e2_fixed.jsonl")
    df = df[(df.protocol == protocol) & (df.regime == regime)]
    cs = np.array(sorted(df.corruption.unique()))
    piv = df.pivot_table(index=["graph_id", "seed", "corruption"],
                         columns="method", values="auprc").reset_index()
    if method not in piv or ref not in piv:
        return None
    piv["diff"] = piv[method] - piv[ref]
    rng = np.random.default_rng(seed)
    ids = piv.graph_id.unique()

    def cross(d):
        means = np.array([d[d.corruption == c]["diff"].mean() for c in cs])
        if not np.isfinite(means).all() or means[0] <= 0:
            return np.nan
        for k in range(1, len(cs)):
            if means[k] <= 0:
                x0, x1, y0, y1 = cs[k - 1], cs[k], means[k - 1], means[k]
                return float(x0 + (x1 - x0) * y0 / (y0 - y1))
        return np.nan

    point = cross(piv)
    boots = []
    for _ in range(n_boot):
        s = rng.choice(ids, size=len(ids), replace=True)
        d = pd.concat([piv[piv.graph_id == g] for g in s])
        boots.append(cross(d))
    boots = np.array([b for b in boots if np.isfinite(b)])
    ci = (float(np.quantile(boots, .025)), float(np.quantile(boots, .975))) if len(boots) > 50 else (np.nan, np.nan)
    return {"protocol": protocol, "regime": regime, "method": method, "ref": ref,
            "crossover_c": point,
            "ci95": ci, "n_boot_valid": int(len(boots))}


def table_corruption():
    df = load("e2_fixed.jsonl")
    t = df.pivot_table(index=["protocol", "regime", "corruption"], columns="method",
                       values="auprc", aggfunc="mean").round(3)
    t.to_csv(os.path.join(OUT, "table3_corruption.csv"))
    res = {}
    for proto in df.protocol.unique():
        for reg in df.regime.unique():
            for c in sorted(df.corruption.unique()):
                sub = df[(df.protocol == proto) & (df.regime == reg) & (df.corruption == c)
                         & df.method.str.startswith("spcrl_amortized")]
                if sub.method.nunique() >= 2:
                    res[f"{proto}|{reg}|c={c}"] = friedman_holm(sub, "auprc",
                                                                block=("graph_id", "seed"))
    cross = {f"{p}|{r}|{m}": crossover_point(p, r, method=m)
             for p in df.protocol.unique() for r in df.regime.unique()
             for m in ["spcrl_amortized_hard", "spcrl_amortized_soft"]}
    json.dump({"posthoc": res, "crossover": cross},
              open(os.path.join(OUT, "table3_stats.json"), "w"), indent=2, default=float)
    return t, cross


def table_downstream():
    df = load("e4_fixed.jsonl")
    t = df.groupby(["regime", "method"])[["top1", "top3", "mrr", "norm_regret", "shd"]] \
          .mean().round(3)
    t.to_csv(os.path.join(OUT, "table4_downstream.csv"))
    stats = {}
    for r in df.regime.unique():
        sub = df[df.regime == r]
        stats[f"{r}|top1"] = friedman_holm(sub, "top1", block=("graph_id", "fault_node", "seed"))
        s2 = sub[sub.norm_regret.notna()]
        stats[f"{r}|regret"] = friedman_holm(s2, "norm_regret",
                                             block=("graph_id", "fault_node", "seed"),
                                             greater_is_better=False)
    json.dump(stats, open(os.path.join(OUT, "table4_stats.json"), "w"), indent=2, default=float)
    return t


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["main", "ood_gap", "corruption", "downstream"]):
        try:
            r = globals()[f"table_{name}"]()
            print(f"=== {name} ===")
            print(r if not isinstance(r, tuple) else r[0])
            if isinstance(r, tuple):
                print("crossover:", json.dumps(r[1], indent=2, default=float))
        except FileNotFoundError:
            print("skip", name, "(no results)")
        except Exception as e:
            print("skip", name, type(e).__name__, e)
