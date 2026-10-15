#!/usr/bin/env python3
"""Assemble the manuscript from docs/sections/*.md into one markdown file.

Numbers are never typed here: they come from the section files, which are
themselves checked against results/*.jsonl by the analysis scripts.
"""
import os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = lambda f: open(os.path.join(ROOT, "docs", "sections", f)).read()
TITLE = "When Structural Priors Help — and Hurt — Causal Learning under Distribution Shift"

# Each figure is placed at the END of the section that owns it - immediately
# before the next heading - so that it follows the text that cites it. An
# earlier version inserted figures right after a heading, which put every
# figure one section late and gave Section 2.10 the scaling figure of 2.9.
# npj Artificial Intelligence allows at most ten display items; with two tables
# in the Results that leaves eight figures. The coarseness sweep (formerly
# Figure 7) is the c = 0 column of the envelope and is not shown separately.
FIGS = [
    ("# Introduction", "fig_concept.png",
     "**Figure 1** | Ways a structural prior can be imperfect, and what each "
     "does to a causal learner. **a**, the relation a prior admits (shaded) "
     "against the relation the system has (arrows); the running system's true "
     "graph is unknown, and its panel shows the measured situation - "
     "dependence dominated by one load driver. **b**, effect on the learner "
     "against the prior-blind control: edge AUPRC for the synthetic defects, "
     "root-cause Top-1 for the running system."),
    ("## 2.1 ", "fig_main.png",
     "**Figure 2** | Causal structure recovery, all methods, six regimes."),
    ("## 2.1 ", "fig_scaling.png",
     "**Figure 3** | Transfer to unseen system sizes."),
    ("## 2.2 ", "fig_corruption.png",
     "**Figure 4** | Prior corruption under the matched and drift protocols."),
    ("## 2.3 ", "fig_priortypes.png",
     "**Figure 5** | What in the prior carries the information."),
    ("## 2.4 ", "fig_envelope.png",
     "**Figure 6** | The prior-quality envelope: corruption against coarseness."),
    ("## 2.5 ", "fig_varsortability.png",
     "**Figure 7** | Raw versus standardised data: the benchmark artefact and "
     "what survives it."),
    ("## 2.6 ", "fig_downstream.png",
     "**Figure 8** | Downstream diagnosis and counterfactual repair."),
]


def _place(body, prefix, block):
    """Insert `block` at the end of the section whose heading starts with
    `prefix`, i.e. immediately before the next heading of the same or higher
    level."""
    start = body.find("\n" + prefix) if not body.startswith(prefix) else -1
    start = 0 if body.startswith(prefix) else start + 1
    if start <= 0 and not body.startswith(prefix):
        return body, False
    level = prefix.split(" ")[0]
    rest = body[start + len(prefix):]
    m = re.search(r"\n(#{1,%d}) " % len(level), rest)
    end = start + len(prefix) + (m.start() if m else len(rest))
    return body[:end].rstrip() + "\n\n" + block + "\n\n" + body[end:].lstrip("\n"), True


TAIL = """

# Data availability

The synthetic benchmark is fully specified by the generator and regenerated
deterministically from the seeds recorded in every result row, so no data file
is needed to reproduce it. All raw result records --- one JSON object per model
fit and per evaluated fault --- and their provenance metadata are released with
the code. The collected telemetry of the second Kubernetes
campaign --- six nominal and 24 fault windows, the campaign log with the
ground-truth origin of every run, and the structural prior derived from the
deployment description --- is deposited with it, together with the telemetry of
the campaign that failed its manipulation check. The raw telemetry of the first
Kubernetes campaign was lost to an infrastructure failure after its analysis;
its result records, campaign log, structural prior and diagnostics are
deposited, and the one analysis in this paper that needs nominal telemetry
rather than result records (the compatibility statistic of §2.9) therefore
covers the other two campaigns only. The free5GC campaign of
§2.9 is deposited on the same terms --- six nominal and 32 fault windows, the
campaign log with the ground truth of every run, the structural prior and its
five registered controls, the dated amendment to the workload, the result
records with their checksums, and the environment record of the machine that
produced them.

# Code availability

Benchmark generator, models, baselines, experiment drivers, statistical
analysis, figure scripts and the integrity test suite are released under an open
licence, together with the Kubernetes and free5GC harnesses.
"""


def main():
    abstract = D("abstract.md").split("---")[0]
    abstract = re.sub(r"^# Abstract.*$", "## Abstract", abstract, flags=re.M)
    body = (D("introduction.md") + "\n\n" + D("results.md") + "\n\n"
            + D("discussion.md") + "\n\n" + D("methods.md")
            + "\n\n" + D("supplementary.md"))
    for prefix, fig, cap in FIGS:
        if not os.path.exists(os.path.join(ROOT, "figures", fig)):
            print("missing figure", fig)
            continue
        body, ok = _place(body, prefix, f"![](figures/{fig}){{width=95%}}\n\n{cap}")
        if not ok:
            print("section not found for", fig, repr(prefix))
    out = (f"---\ntitle: \"{TITLE}\"\n"
           f"subtitle: \"Manuscript draft for npj Artificial Intelligence (Article)\"\n"
           f"date: \"All reported numbers are generated from results/*.jsonl\"\n---\n\n"
           + abstract + "\n\n" + body + TAIL)
    path = os.path.join(ROOT, "docs", "build", "manuscript_full.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(out)
    core = re.sub(r"\|.*\|", "",
                  D("introduction.md") + D("results.md") + D("discussion.md"))
    print(f"wrote {path}")
    print(f"Introduction+Results+Discussion: {len(core.split())} words "
          f"(npj AI guideline 4000-4500)")
    print(f"total: {len(out.split())} words")


if __name__ == "__main__":
    main()
