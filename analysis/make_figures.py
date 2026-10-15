"""Figures for the manuscript. Every figure is produced from results/*.jsonl."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.load import load, load_multi, LABELS, REGIME

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200,
                     "savefig.bbox": "tight", "font.family": "DejaVu Sans"})
C = {"spcrl_amortized_soft": "#1b6ca8", "spcrl_amortized_hard": "#c1440e",
     "spcrl_amortized_none": "#6b6b6b", "prior_only": "#9b8bbf"}


def _ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2: return 0.0
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def fig_corruption(metric="auprc", fname="fig_corruption.png"):
    """Two protocols x two regimes. The left column is the matched protocol
    (the model is trained under the same prior quality it is deployed with);
    the right column is deployment drift (trained on a clean prior)."""
    df = load("e2_fixed.jsonl")
    protos = [p for p in ["matched", "drift"] if p in set(df.protocol)]
    regs = [r for r in ["iid", "graph_ood"] if r in set(df.regime)]
    fig, axes = plt.subplots(len(protos), len(regs),
                             figsize=(3.1 * len(regs), 2.4 * len(protos)),
                             sharey="row", sharex=True)
    axes = np.atleast_2d(axes)
    for i, proto in enumerate(protos):
        for j, reg in enumerate(regs):
            ax = axes[i, j]
            sub = df[(df.protocol == proto) & (df.regime == reg)]
            for m in ["spcrl_amortized_soft", "spcrl_amortized_hard",
                      "spcrl_amortized_none", "prior_only"]:
                s_ = sub[sub.method == m]
                if s_.empty:
                    continue
                g = s_.groupby("corruption")[metric]
                x = np.array(sorted(g.groups))
                y = g.mean().reindex(x).values
                e = g.apply(lambda v: _ci(v.values)).reindex(x).values
                ax.errorbar(x, y, yerr=e, marker="o", ms=3.2, lw=1.4, capsize=2,
                            color=C.get(m), label=LABELS[m])
            ax.grid(alpha=.25, lw=.5)
            if i == 0:
                ax.set_title(REGIME.get(reg, reg), fontsize=8.5)
            if i == len(protos) - 1:
                ax.set_xlabel("structural-prior corruption $c$")
            if j == 0:
                ax.set_ylabel(("matched prior\n" if proto == "matched"
                               else "prior drift after training\n") +
                              {"auprc": "edge AUPRC", "shd": "SHD",
                               "f1": "edge F1"}[metric], fontsize=7.5)
    axes[0, -1].legend(frameon=False, fontsize=6.3, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname)); plt.close(fig)
    print("wrote", fname)


def fig_main(fname="fig_main.png"):
    df = load("e1_fixed.jsonl")
    order = ["iid", "mech_ood", "graph_ood", "size_ood", "noise_ood", "joint_ood"]
    order = [o for o in order if o in df.regime.unique()]
    methods = [m for m in ["spcrl_amortized_soft", "spcrl_amortized_hard",
                           "spcrl_amortized_none", "notears_linear_hardmask",
                           "pc_prior", "pcorr", "prior_only"] if m in df.method.unique()]
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    w = 0.8 / len(methods)
    for k, m in enumerate(methods):
        y = [df[(df.regime == r) & (df.method == m)].auprc.mean() for r in order]
        e = [_ci(df[(df.regime == r) & (df.method == m)].auprc.values) for r in order]
        ax.bar(np.arange(len(order)) + k * w, y, w, yerr=e, capsize=1.5,
               label=LABELS.get(m, m), color=C.get(m))
    ax.set_xticks(np.arange(len(order)) + 0.4 - w / 2)
    ax.set_xticklabels([REGIME.get(o, o) for o in order], rotation=12)
    ax.set_ylabel("edge AUPRC"); ax.grid(axis="y", alpha=.25, lw=.5)
    ax.legend(frameon=False, fontsize=6.2, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, 1.28))
    fig.savefig(os.path.join(FIG, fname)); plt.close(fig)
    print("wrote", fname)


def fig_priortypes(fname="fig_priortypes.png"):
    df = load("e3_fixed.jsonl")
    kinds = ["none", "topology", "topology_type", "topology_dir", "full"]
    kinds = [k for k in kinds if k in df.prior_kind.unique()]
    fig, ax = plt.subplots(figsize=(4.2, 2.5))
    for reg, mk in [("iid", "o"), ("graph_ood", "s")]:
        s = df[df.regime == reg]
        if s.empty: continue
        y = [s[s.prior_kind == k].auprc.mean() for k in kinds]
        e = [_ci(s[s.prior_kind == k].auprc.values) for k in kinds]
        ax.errorbar(range(len(kinds)), y, yerr=e, marker=mk, ms=4, lw=1.4,
                    capsize=2, label=REGIME.get(reg, reg))
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels(["none", "topology", "+ node type", "+ direction", "full"],
                       rotation=12)
    ax.set_ylabel("edge AUPRC"); ax.grid(alpha=.25, lw=.5)
    ax.legend(frameon=False, fontsize=6.5)
    fig.savefig(os.path.join(FIG, fname)); plt.close(fig)
    print("wrote", fname)


def fig_downstream(fname="fig_downstream.png"):
    df = load("e4_fixed.jsonl")
    methods = [m for m in ["oracle", "spcrl_soft", "spcrl_hard", "spcrl_none",
                           "notears_linear", "pcorr", "prior_only",
                           "anomaly_no_structure"] if m in df.method.unique()]
    regs = [r for r in ["iid", "mech_ood", "graph_ood"] if r in df.regime.unique()]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    w = 0.8 / len(methods)
    for k, m in enumerate(methods):
        vals = [df[(df.regime == r) & (df.method == m)].top1.values for r in regs]
        axes[0].bar(np.arange(len(regs)) + k * w, [v.mean() for v in vals], w,
                    yerr=[_ci(v) for v in vals], capsize=1.5,
                    error_kw={"lw": .7}, label=LABELS.get(m, m))
        s = df[(df.method == m) & df.norm_regret.notna()]
        if not s.empty:
            v2 = [s[s.regime == r].norm_regret.values for r in regs]
            axes[1].bar(np.arange(len(regs)) + k * w, [v.mean() for v in v2], w,
                        yerr=[_ci(v) for v in v2], capsize=1.5, error_kw={"lw": .7})
    for ax, lab in zip(axes, ["root-cause Top-1 accuracy", "normalised intervention regret"]):
        ax.set_xticks(np.arange(len(regs)) + 0.4 - w / 2)
        ax.set_xticklabels([REGIME.get(r, r) for r in regs], rotation=10)
        ax.set_ylabel(lab); ax.grid(axis="y", alpha=.25, lw=.5)
    axes[0].legend(frameon=False, fontsize=6, ncol=3, loc="upper center",
                   bbox_to_anchor=(1.1, 1.3))
    fig.savefig(os.path.join(FIG, fname)); plt.close(fig)
    print("wrote", fname)


def fig_scaling(fname="fig_scaling.png"):
    df = load("e5_fixed.jsonl")
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.5))
    for m in df.method.unique():
        s = df[df.method == m].groupby("d")
        axes[0].plot(s.auprc.mean().index, s.auprc.mean().values, marker="o", ms=3.5,
                     label=LABELS.get(m, m), color=C.get(m))
        axes[1].plot(s.infer_sec.mean().index, s.infer_sec.mean().values, marker="o",
                     ms=3.5, color=C.get(m))
    g = df.groupby("d").rho.mean()
    ax2 = axes[1].twinx(); ax2.plot(g.index, g.values, "k--", lw=1, label=r"prior density $\rho$")
    ax2.set_ylabel(r"admissible fraction $\rho$")
    axes[0].set_xlabel("number of variables $d$"); axes[0].set_ylabel("edge AUPRC")
    axes[1].set_xlabel("number of variables $d$"); axes[1].set_ylabel("inference time (s)")
    for a in axes: a.grid(alpha=.25, lw=.5)
    axes[0].legend(frameon=False, fontsize=6.5)
    fig.savefig(os.path.join(FIG, fname)); plt.close(fig)
    print("wrote", fname)


def fig_coarseness(fname="fig_coarseness.png"):
    """How coarse may the structural prior be before it stops paying off?"""
    df = load("e7_fixed.jsonl")
    regs = [r for r in ["iid", "graph_ood"] if r in set(df.regime)]
    fig, axes = plt.subplots(1, len(regs), figsize=(3.2 * len(regs), 2.6), sharey=True)
    axes = np.atleast_1d(axes)
    xs = sorted(df.superset_ratio.unique())
    xlab = [round(df[df.superset_ratio == r].adm_over_true.mean(), 1) for r in xs]
    for ax, reg in zip(axes, regs):
        sub = df[df.regime == reg]
        for m in ["spcrl_amortized_soft", "spcrl_amortized_hard",
                  "spcrl_amortized_none", "prior_only"]:
            s = sub[sub.method == m]
            if s.empty:
                continue
            g = s.groupby("superset_ratio")["auprc"]
            y = g.mean().reindex(xs).values
            e = g.apply(lambda v: _ci(v.values)).reindex(xs).values
            ax.errorbar(range(len(xs)), y, yerr=e, marker="o", ms=3.5, lw=1.4,
                        capsize=2, color=C.get(m), label=LABELS[m])
        ax.set_xticks(range(len(xs))); ax.set_xticklabels(xlab)
        ax.set_xlabel("admissible pairs / true edges")
        ax.set_title(REGIME.get(reg, reg)); ax.grid(alpha=.25, lw=.5)
    axes[0].set_ylabel("edge AUPRC")
    axes[-1].legend(frameon=False, fontsize=6.3, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname)); plt.close(fig)
    print("wrote", fname)


def fig_varsortability(fname="fig_varsortability.png"):
    """Raw versus standardised data: the benchmark artefact and what survives it."""
    df = load("e11_fixed.jsonl")
    regs = [r for r in ["iid", "mech_ood", "graph_ood"] if r in set(df.regime)]
    methods = ["spcrl_amortized_soft", "spcrl_amortized_hard",
               "spcrl_amortized_none", "sortnregress", "prior_only"]
    methods = [m for m in methods if m in set(df.method)]
    lbl = dict(LABELS); lbl["sortnregress"] = "sortnregress (variance order)"
    col = dict(C); col["sortnregress"] = "#c9a227"; col["prior_only"] = "#9b8bbf"
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    w = 0.8 / len(methods)
    for ax, std, title in zip(axes, [False, True],
                              ["raw data (varsortability 0.84-0.93)",
                               "standardised (varsortability 0)"]):
        sub = df[df.standardized == std]
        for k, m in enumerate(methods):
            vals = [sub[(sub.regime == r) & (sub.method == m)].auprc.values for r in regs]
            ax.bar(np.arange(len(regs)) + k * w, [v.mean() if len(v) else np.nan for v in vals],
                   w, yerr=[_ci(v) if len(v) else 0 for v in vals], capsize=1.5,
                   error_kw={"lw": .7}, color=col.get(m), label=lbl.get(m, m))
        ax.set_xticks(np.arange(len(regs)) + 0.4 - w / 2)
        ax.set_xticklabels([REGIME.get(r, r) for r in regs], rotation=10)
        ax.set_title(title, fontsize=8); ax.grid(axis="y", alpha=.25, lw=.5)
    axes[0].set_ylabel("edge AUPRC")
    axes[1].legend(frameon=False, fontsize=6.2, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname)); plt.close(fig)
    print("wrote", fname)


def fig_envelope(fname="fig_envelope.png"):
    """The two-dimensional prior-quality envelope (E8).

    Left and centre: mean AUPRC advantage over the prior-blind control as a
    function of prior corruption and prior coarseness. Right: the corruption at
    which each mode's advantage crosses zero, as a function of coarseness.
    """
    import json
    path = os.path.join(os.path.dirname(FIG), "results", "e8_envelope.json")
    env = json.load(open(path))
    rs, cs = env["ratios"], env["corruptions"]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.7))
    vmax = max(abs(np.array(env["grid"][m]["mean"])).max() for m in env["grid"])
    for ax, m, name in zip(axes[:2], ("spcrl_amortized_soft", "spcrl_amortized_hard"),
                           ("soft prior", "hard mask")):
        M = np.array(env["grid"][m]["mean"])
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
                       origin="lower")
        ax.set_xticks(range(len(cs))); ax.set_xticklabels([f"{c:g}" for c in cs])
        ax.set_yticks(range(len(rs))); ax.set_yticklabels([f"{r:g}x" for r in rs])
        ax.set_xlabel("prior corruption $c$")
        ax.set_ylabel("coarseness $r$ = admissible / true")
        ax.set_title(f"{name} $-$ prior-blind")
        for i in range(len(rs)):
            for j in range(len(cs)):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                        fontsize=6, color="black")
        # the line the paper is about: where the prior stops paying for itself
        # Only where the sign change is real: at 20x coarseness the soft
        # prior's values oscillate around zero by less than half a percent, and
        # a contour drawn through that would be a picture of noise.
        if (M < -0.02).any() and (M > 0.02).any():
            ax.contour(M, levels=[0.0], colors="k", linewidths=1.1,
                       linestyles="--")
            ax.plot([], [], "k--", lw=1.1, label="$\\Delta$ = 0")
            ax.legend(frameon=False, loc="lower left", fontsize=6)
    fig.colorbar(im, ax=axes[1], fraction=0.046, label="$\\Delta$ AUPRC")
    ax = axes[2]
    for m, name in (("spcrl_amortized_soft", "soft prior"),
                    ("spcrl_amortized_hard", "hard mask")):
        xs, ys = [], []
        for r in rs:
            v = env["crossover"][m][str(r)]
            if v is None:
                ax.plot([r], [0.0], "x", color=C[m], ms=6)
                continue
            xs.append(r); ys.append(min(v, 0.55))
        ax.plot(xs, ys, "o-", color=C[m], label=name)
    ax.text(11.0, 0.03, "no clean-prior\nadvantage to lose", fontsize=6, color="0.4")
    ax.axhline(0.5, ls=":", c="0.5", lw=0.8)
    ax.text(2.2, 0.515, "no crossover observed up to $c$ = 0.5", fontsize=6,
            color="0.4")
    ax.set_xscale("log"); ax.set_xticks(rs)
    ax.set_xticklabels([f"{r:g}x" for r in rs])
    ax.set_xlabel("coarseness $r$"); ax.set_ylabel("crossover corruption $c$")
    ax.set_title("corruption tolerance falls with coarseness")
    ax.legend(frameon=False, loc="center left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname)); plt.close(fig)
    print("wrote", fname)


if __name__ == "__main__":
    which = sys.argv[1:] or ["corruption", "main", "priortypes", "downstream",
                             "scaling", "coarseness", "varsortability", "envelope"]
    for w in which:
        try:
            globals()[f"fig_{w}"]()
        except FileNotFoundError as e:
            print("skip", w, "(no results yet)")
        except Exception as e:
            print("skip", w, type(e).__name__, e)
