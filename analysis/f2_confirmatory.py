"""The registered analysis of F2 (docs/preregistration/PREREGISTRATION_F2.md).

Written, and run on F1 as a development check, before any F2 data existed.
Nothing here may be changed after F2 data exist except by a dated amendment.

    python analysis/f2_confirmatory.py results/e19_f2.jsonl  [--label F2]

Unit of analysis: the cell (fault target x fault type), crossed with the
training seed. Within a (cell, seed), Top-1 is averaged over the cell's fault
windows and, for a prior family with K independent draws (CHk, WID), over draws,
so the control is the EXPECTED Top-1 under that kind of misalignment. The point
estimate is the mean over cells and seeds. Uncertainty: two-way percentile
bootstrap -- cells and training seeds resampled independently -- B = 20 000,
seed 20261117 (Amendment 1: the trained model is itself a random draw, and a
bootstrap over cells alone treats it as fixed).
"""
from __future__ import annotations
import argparse, collections, json, random, statistics as st

REPLICATION = {"cpu_saturation", "sbi_latency"}
NEW = {"sbi_loss", "sbi_ratelimit", "cpu_throttle"}
LEVELS = (0, 1, 2, 5, 7, 9)
B, BSEED = 20_000, 20261117
MESOI = 0.10          # smallest effect of interest, Top-1


def load(path):
    rows = [json.loads(l) for l in open(path)]
    return [r for r in rows if "top1" in r and r.get("relabel", 0) == 0]


def cell_table(rows, method, prior_pred):
    """(cell, seed) -> mean Top-1 over faults and draws of priors matching prior_pred."""
    acc = collections.defaultdict(list)
    for r in rows:
        if r["method"] == method and prior_pred(r["prior"]):
            acc[(r["cell"], r["seed"])].append(r["top1"])
    return {k: st.mean(v) for k, v in acc.items()}


def family(cell):
    ft = cell.split("|")[1]
    return "new" if ft in NEW else "replication" if ft in REPLICATION else "?"


def boot_ci(values, rng):
    """values: {(cell, seed): x}. Two-way bootstrap of the grand mean."""
    cells = sorted({c for c, _ in values})
    seeds = sorted({s for _, s in values})
    if len(cells) < 2:
        return (float("nan"), float("nan"))
    draws = []
    for _ in range(B):
        cs = [cells[rng.randrange(len(cells))] for _ in cells]
        ss = [seeds[rng.randrange(len(seeds))] for _ in seeds]
        v = [values[(c, s)] for c in cs for s in ss if (c, s) in values]
        draws.append(sum(v) / len(v))
    draws.sort()
    return draws[int(0.025 * B)], draws[int(0.975 * B) - 1]


def contrast(rows, cells, method, a, b, rng, method_b=None):
    A = cell_table(rows, method, a)
    Bt = cell_table(rows, method_b or method, b)
    keys = [k for k in A if k in Bt and k[0] in set(cells)]
    if not keys:
        return None
    d = {k: A[k] - Bt[k] for k in keys}
    lo, hi = boot_ci(d, rng)
    return {"n_cells": len({k[0] for k in keys}), "n_seeds": len({k[1] for k in keys}),
            "est": st.mean(d.values()), "lo": lo, "hi": hi}


def decide_directional(r):
    if r is None:
        return "not computable"
    if r["lo"] > 0:
        return "SUPPORTED"
    if r["hi"] < 0:
        return "REFUTED (opposite sign)"
    return "INCONCLUSIVE"


def is_(name):
    return lambda p: p == name


def family_of(prefix):
    return lambda p: p.startswith(prefix + "_d")


def slopes(rows, cells, method, x_of_level):
    """(cell, seed) -> OLS slope of Top-1 on the realised fraction of relations moved."""
    tabs = {L: (cell_table(rows, method, is_("P2_semantic")) if L == 0 else
                cell_table(rows, method, family_of(f"CH{L}"))) for L in LEVELS}
    out = {}
    for k in tabs[0]:
        if k[0] not in set(cells):
            continue
        pts = [(x_of_level[L], tabs[L][k]) for L in LEVELS if k in tabs[L]]
        if len(pts) < 3:
            continue
        xs, ys = zip(*pts)
        mx, my = st.mean(xs), st.mean(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        out[k] = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    return out, tabs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("--manifest", default="docs/preregistration/F2_PRIORS_MANIFEST.json")
    ap.add_argument("--label", default="F2")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    rng = random.Random(BSEED)
    rows = load(a.records)
    man = json.load(open(a.manifest))
    cells_all = sorted({r["cell"] for r in rows})
    fam = {c: family(c) for c in cells_all}
    sets = {"new": [c for c in cells_all if fam[c] == "new"],
            "replication": [c for c in cells_all if fam[c] == "replication"],
            "all": cells_all}
    x_of_level = {0: 0.0}
    for L in LEVELS[1:]:
        fr = [v["invariants_vs_P2"]["fraction_relations_moved"]
              for k, v in man["priors"].items() if k.startswith(f"CH{L}_d")]
        x_of_level[L] = st.mean(fr)

    res = {"label": a.label, "cells": {k: len(v) for k, v in sets.items()}}
    soft, P2, CH9 = "spcrl_soft", is_("P2_semantic"), family_of("CH9")
    primary_set = "new" if sets["new"] else "all"

    # H-F2-1 (primary): alignment against the matched channel-permutation control
    r = contrast(rows, sets[primary_set], soft, P2, CH9, rng)
    res["H-F2-1"] = {**(r or {}), "cells_used": primary_set, "decision": decide_directional(r)}

    # S1: same contrast on the other cell sets
    for k in ("replication", "all"):
        if sets[k] and k != primary_set:
            res[f"S1_align_{k}"] = contrast(rows, sets[k], soft, P2, CH9, rng)

    # S2: dose-response (E2)
    sl, tabs = slopes(rows, sets[primary_set], soft, x_of_level)
    lo, hi = boot_ci(sl, rng) if sl else (float("nan"), float("nan"))
    curve = {}
    for L in LEVELS:
        v = [x for (c, s_), x in tabs[L].items() if c in set(sets[primary_set])]
        curve[str(L)] = {"x": round(x_of_level[L], 3), "top1": st.mean(v) if v else None}
    res["S2_dose_slope"] = {"n": len(sl), "est": st.mean(sl.values()) if sl else None,
                            "lo": lo, "hi": hi,
                            "decision": ("SUPPORTED" if sl and hi < 0 else
                                         "REFUTED (opposite sign)" if sl and lo > 0 else "INCONCLUSIVE"),
                            "curve": curve}

    # S3: relation identity with channel semantics preserved
    r3 = contrast(rows, sets[primary_set], soft, P2, family_of("WID"), rng)
    if r3:
        r3["decision"] = ("no meaningful identity benefit (SUPPORTED)" if r3["hi"] < 0.15 else
                          "identity matters (REFUTED)" if r3["lo"] > 0 else "INCONCLUSIVE")
    res["S3_identity"] = r3

    # S4: replication of the published decomposition (F1 Table 2), all cells
    pub = {"total P1-P5": ("spcrl_soft", is_("P1_topology"), "spcrl_none", is_("P1_topology")),
           "identity P1-P6": ("spcrl_soft", is_("P1_topology"), "spcrl_soft", is_("P6_wrong_topology")),
           "alignment P2-P3b": ("spcrl_soft", P2, "spcrl_soft", is_("P3b_random_at_P2_density")),
           "density P1-P3": ("spcrl_soft", is_("P1_topology"), "spcrl_soft", is_("P3_random")),
           "degree P1-P4": ("spcrl_soft", is_("P1_topology"), "spcrl_soft", is_("P4_degree_shuffled"))}
    res["S4_published_contrasts"] = {}
    for name, (m1, p1, m2, p2) in pub.items():
        r4 = contrast(rows, cells_all, m1, p1, p2, rng, method_b=m2)
        if r4:
            res["S4_published_contrasts"][name] = r4

    # S5: the hard arm, both parent-selection rules
    res["S5_hard"] = {m: contrast(rows, sets[primary_set], m, P2, CH9, rng)
                      for m in ("spcrl_hard_strict", "spcrl_hard")}

    # S6, S7 (Amendment 1): registered after the F1 development run, before F2
    r6 = contrast(rows, sets[primary_set], soft, P2, is_("P3b_random_at_P2_density"), rng)
    res["S6_structure_P2_vs_P3b"] = {**(r6 or {}), "decision": decide_directional(r6)}
    r7 = contrast(rows, sets[primary_set], "spcrl_hard_strict", P2, CH9, rng)
    res["S7_channels_under_strict_hard"] = {**(r7 or {}), "decision": decide_directional(r7)}

    # E3a: invariance under a consistent relabelling of the variables
    allr = [json.loads(l) for l in open(a.records)]
    base = {(r["seed"], r["prior"], r["run_id_fault"], r["method"]): r["rank"]
            for r in allr if "rank" in r and r.get("relabel", 0) == 0}
    inv = collections.defaultdict(lambda: [0, 0])
    for r in allr:
        if "rank" in r and r.get("relabel", 0) > 0:
            k = (r["seed"], r["prior"], r["run_id_fault"], r["method"])
            if k in base:
                inv[r["method"]][0] += 1
                inv[r["method"]][1] += int(base[k] == r["rank"])
    res["E3a_relabel_invariance"] = {m: f"{s}/{n} rankings unchanged" for m, (n, s) in inv.items()}

    print(json.dumps(res, indent=1, default=lambda x: round(x, 4) if isinstance(x, float) else str(x)))
    if a.json_out:
        json.dump(res, open(a.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
