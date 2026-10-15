"""Figure 1: the ways a structural prior can be imperfect, and what each does.

Panel a draws each defect as the relation the prior admits against the relation
the system has. Panel b gives the measured effect of each on the learner, read
from the analysis summaries rather than typed in, so the figure cannot drift
from the ledger. The last column is the running system: a different metric
(root-cause Top-1 rather than edge AUPRC) and the only defect we could not
reproduce, which is why it is set apart.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
R = lambda f: json.load(open(os.path.join(ROOT, "results", f)))

INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#d9d9d9"
TRUE_C, ADM_C, BAD_C, DRV_C = "#1f1f1f", "#1b6ca8", "#c1440e", "#7a5aa6"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans",
                     "savefig.bbox": "tight", "figure.dpi": 200})


def values():
    from analysis.load import load
    e1 = load("e1_fixed.jsonl")
    s = e1[e1.regime == "iid"]
    m = lambda k: s[s.method == f"spcrl_amortized_{k}"].auprc.mean()
    correct = (m("soft") - m("none"), m("hard") - m("none"))
    e8 = R("e8_envelope.json"); r = e8["ratios"].index(20.0)
    coarse = (e8["grid"]["spcrl_amortized_soft"]["mean"][r][0],
              e8["grid"]["spcrl_amortized_hard"]["mean"][r][0])
    e13 = R("e13_summary.json")["cells"]
    wrong = (e13["matched|iid|delete|c=0.2|soft"]["mean"],
             e13["matched|iid|delete|c=0.2|hard"]["mean"])
    e14 = R("e14_summary.json")["cells"]
    partial = (e14["iid|m=0.75|soft"]["mean"], e14["iid|m=0.75|hard"]["mean"])
    e16 = R("e16_summary.json")["advantage"]
    driver = (e16["iid|m=0.95|soft"]["mean"], e16["iid|m=0.95|hard"]["mean"])
    k8 = {c["contrast"]: c for c in R("e10_paired.json")["contrasts"]
          if c["metric"] == "top1"}
    real = (k8["spcrl_soft - spcrl_none"]["mean"],
            k8["spcrl_hard - spcrl_none"]["mean"])
    sig_real = (k8["spcrl_soft - spcrl_none"]["ci_hi"] < 0,
                k8["spcrl_hard - spcrl_none"]["ci_hi"] < 0)
    return [correct, coarse, wrong, partial, driver, real], sig_real


def mini(ax, x0, kind):
    """A four-node system: black arrows are true relations, blue bands are
    what the prior admits, red marks a true relation the prior excludes."""
    pos = {0: (x0 + .18, .70), 1: (x0 + .62, .70), 2: (x0 + .18, .30),
           3: (x0 + .62, .30)}
    true = [(0, 1), (0, 2), (1, 3)]
    adm = {"correct": [(0, 1), (0, 2), (1, 3)],
           "coarse": [(0, 1), (0, 2), (1, 3), (2, 3), (1, 2), (0, 3)],
           "wrong": [(0, 2), (1, 3), (2, 3)],
           "partial": [(0, 1), (0, 2), (1, 3), (1, 2), (0, 3), (2, 3)],
           "driver": [(0, 1), (0, 2), (1, 3)],
           "real": [(0, 1), (0, 2), (1, 3)]}[kind]
    for a, b in adm:
        (xa, ya), (xb, yb) = pos[a], pos[b]
        ax.plot([xa, xb], [ya, yb], color=ADM_C, lw=5.5, alpha=.22,
                solid_capstyle="round", zorder=1)
    for a, b in true:
        (xa, ya), (xb, yb) = pos[a], pos[b]
        missing = (a, b) not in adm and (b, a) not in adm
        ax.add_patch(FancyArrowPatch((xa, ya), (xb, yb), arrowstyle="-|>",
                     mutation_scale=7, lw=1.1, shrinkA=6, shrinkB=6,
                     color=BAD_C if missing else TRUE_C,
                     linestyle="--" if missing else "-", zorder=3))
    for i, (x, y) in pos.items():
        unmentioned = kind == "partial" and i == 3
        ax.add_patch(Circle((x, y), .055, facecolor="white",
                     edgecolor=MUTED if unmentioned else INK,
                     linestyle=":" if unmentioned else "-", lw=1, zorder=4))
    if kind in ("driver", "real"):
        L = (x0 + .40, .96)
        ax.add_patch(Circle(L, .05, facecolor=DRV_C, edgecolor="none",
                     alpha=.9, zorder=4))
        ax.text(*L, "L", ha="center", va="center", color="white",
                fontsize=6.5, fontweight="bold", zorder=5)
        for i in pos:
            ax.add_patch(FancyArrowPatch(L, pos[i], arrowstyle="-|>",
                         mutation_scale=5, lw=.7, color=DRV_C, alpha=.75,
                         shrinkA=5, shrinkB=6, zorder=2))


def main(out=os.path.join(ROOT, "figures", "fig_concept.png")):
    vals, sig_real = values()
    cols = [("correct", "correct"),
            ("coarse", "too permissive\n(20× coarseness)"),
            ("wrong", "excludes true\nrelations (c = 0.2)"),
            ("partial", "silent on part\n(75% uncovered)"),
            ("driver", "latent common\ndriver (m = 0.95)"),
            ("real", "running system\n(Kubernetes)")]
    fig = plt.figure(figsize=(7.2, 3.9))
    ax = fig.add_axes([0.11, 0.43, 0.88, 0.52]); ax.axis("off")
    ax.set_xlim(0, 6.25); ax.set_ylim(-.05, 1.25)
    for k, (kind, name) in enumerate(cols):
        x0 = k * 1.02 + (0.14 if kind == "real" else 0)
        if kind == "real":
            ax.add_patch(FancyBboxPatch((x0 - .04, -.02), .88, 1.25,
                         boxstyle="round,pad=0.01,rounding_size=0.04",
                         facecolor="#f4f4f4", edgecolor=GRID, lw=.8, zorder=0))
        mini(ax, x0, kind)
        ax.text(x0 + .40, .06, name, ha="center", va="center", fontsize=6.8,
                color=INK, linespacing=1.15)
    ax.text(-0.72, 1.22, "a", fontsize=11, fontweight="bold", va="top")
    ax.text(-0.52, 1.22, "what the prior says against what the system does",
            fontsize=8, va="top", color=INK)
    ax.text(-0.52, 1.08, "→ true relation", fontsize=6.5, color=TRUE_C, va="top")
    ax.plot([0.40, 0.58], [1.045, 1.045], color=ADM_C, lw=5, alpha=.25)
    ax.text(0.62, 1.08, "admitted by the prior", fontsize=6.5, color=MUTED,
            va="top")
    ax.text(1.72, 1.08, "- - → true relation the prior excludes", fontsize=6.5,
            color=BAD_C, va="top")
    ax.text(3.62, 1.08, "L  latent common driver", fontsize=6.5, color=DRV_C,
            va="top")

    bx = fig.add_axes([0.11, 0.07, 0.88, 0.27])
    M = np.array(vals).T                          # rows: soft, hard
    lim = 0.40
    cmap = plt.get_cmap("RdBu")
    for j in range(M.shape[1]):
        for i in range(2):
            v = M[i, j]
            face = cmap(0.5 + 0.5 * np.clip(v / lim, -1, 1))
            x = j * 1.02 + (0.14 if j == 5 else 0) - 0.06
            bx.add_patch(FancyBboxPatch((x + .04, 1 - i + .08), .84, .84,
                         boxstyle="round,pad=0,rounding_size=0.06",
                         facecolor=face, edgecolor="white", lw=2))
            txt = f"{v:+.3f}"
            if j == 5 and not sig_real[i]:
                txt += "\n(n.s.)"
            bx.text(x + .46, 1 - i + .5, txt, ha="center", va="center",
                    fontsize=7.5, color=INK if abs(v) < .25 else "white")
    bx.set_xlim(0, 6.25); bx.set_ylim(0, 2)
    bx.set_yticks([1.5, .5]); bx.set_yticklabels(["soft prior", "hard mask"])
    bx.set_xticks([]); bx.tick_params(length=0)
    for sp in bx.spines.values():
        sp.set_visible(False)
    bx.text(-0.72, 2.35, "b", fontsize=11, fontweight="bold", va="bottom")
    bx.text(-0.52, 2.35, "effect against the prior-blind control", fontsize=8,
            va="bottom", color=INK)
    bx.text(2.44, -.28, "edge AUPRC, synthetic, in distribution", ha="center",
            fontsize=6.5, color=MUTED)
    bx.text(5.64, -.28, "root-cause Top-1, 24 faults", ha="center", fontsize=6.5,
            color=MUTED)
    fig.savefig(out); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
