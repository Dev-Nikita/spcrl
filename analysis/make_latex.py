"""Build the single-file Springer Nature LaTeX submission from the manuscript.

    python analysis/make_latex.py [outdir]

Writes <outdir>/spcrl_npjai.tex, copies sn-jnl.cls, the bst files, the figures
and the bibliography, so the directory compiles with pdflatex + bibtex.
The manuscript markdown remains the single source: this is a converter, not a
second copy of the text.
"""
from __future__ import annotations
import os, re, shutil, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The Springer Nature template files are kept in manuscript/ so that the build
# does not depend on the downloaded template package still being around.
TEMPLATE = os.environ.get("SN_TEMPLATE", os.path.join(ROOT, "manuscript"))

MAIN_TABLE_CAPTIONS = [
    ("tab:shift", "Structure recovery with and without a structural prior, by "
                  "shift regime. Edge AUPRC over 60 held-out systems per "
                  "regime, five training seeds averaged within system; "
                  "$p$-values from paired Wilcoxon tests with Holm correction."),
    ("tab:attribution", "What part of a description is worth something. Paired "
                        "Top-1 contrasts on each external campaign, soft "
                        "prior, three training seeds averaged within fault, "
                        "95% bootstrap intervals over faults. Each row varies "
                        "one property of the supplied description and holds "
                        "the others fixed."),
]

UNI = [("−", "$-$"), ("—", "---"), ("–", "--"), ("×", "$\\times$"),
       ("≤", "$\\le$"), ("≥", "$\\ge$"), ("±", "$\\pm$"), ("·", "$\\cdot$"),
       ("→", "$\\rightarrow$"), ("↑", "$\\uparrow$"), ("↓", "$\\downarrow$"),
       ("Δ", "$\\Delta$"), ("ρ", "$\\rho$"), ("δ", "$\\delta$"),
       ("§", "Sec.~"), ("10⁻⁶", "$10^{-6}$"), ("10⁻⁵", "$10^{-5}$"),
       ("⁻⁶", "$^{-6}$"), ("⁻¹⁰", "$^{-10}$"), ("²", "$^2$"), ("ᵀ", "$^{T}$"),
       ("⁶", "$^{6}$"), ("⁵", "$^{5}$"), ("¹", "$^{1}$"), ("⁰", "$^{0}$"),
       ("ő", "\\H{o}"), ("é", "\\'e")]


def inline(s: str) -> str:
    """Markdown inline markup and stray characters to LaTeX; math is left alone.

    Anything that has to become LaTeX (unicode symbols, plain-text
    superscripts) is parked in a placeholder before the escaping pass, so that
    escaping never touches a backslash this function itself introduced.
    """
    parts = re.split(r"(\$[^$]*\$)", s)
    out = []
    for k, p in enumerate(parts):
        if k % 2:
            out.append(p)
            continue
        keep = []

        def park(tex):
            keep.append(tex)
            return "\x00%d\x00" % (len(keep) - 1)

        p = p.replace("10\u207b\u2076", park("$10^{-6}$"))
        p = p.replace("10\u207b\u2075", park("$10^{-5}$"))
        for a, b in UNI:
            if a in p:
                p = p.replace(a, park(b))
        p = re.sub(r"(?<=[A-Za-z0-9])\^(\{[^}]*\}|[\w.+-]+)",
                   lambda m: park("$^{%s}$" % m.group(1).strip("{}")), p)
        p = p.replace("\\", "\\textbackslash{}")
        p = re.sub(r"([%&#_])", r"\\\1", p)
        p = p.replace("~", "\\textasciitilde{}").replace("^", "\\^{}")
        p = p.replace("<", "$<$").replace(">", "$>$")
        def _cite(m):
            """Pandoc-style group: keys only, any prose in it is dropped."""
            keys = re.findall(r"@([A-Za-z][\w:.-]*)", m.group(0))
            return "\\cite{" + ",".join(keys) + "}"

        p = re.sub(r"\[@[^\]]*\]", _cite, p)
        p = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", p)
        p = re.sub(r"(?<![\w*])\*([^*]+)\*(?![\w*])", r"\\emph{\1}", p)
        p = re.sub(r"`([^`]+)`", r"\\texttt{\1}", p)
        p = re.sub(r"\x00(\d+)\x00", lambda m: keep[int(m.group(1))], p)
        out.append(p)
    return "".join(out)


def table(rows, caption, label):
    caption = caption.replace("\\%", "%").replace("%", "\\%")
    head, body = rows[0], rows[2:]
    n = len(head)
    widths = [max(len(r[j]) for r in rows[2:] + [head]) for j in range(n)]
    widths = [max(w, 5) for w in widths]
    tot = float(sum(widths))
    weights = [max(0.45, n * w / tot) for w in widths]
    scale = n / sum(weights)
    weights = [w * scale for w in weights]
    cols = "".join(("A{%.3f}" % weights[0]) if j == 0 else ("B{%.3f}" % weights[j])
                   for j in range(n))
    widest = max(sum(len(c) for c in r) + 3 * (len(r) - 1) for r in [head] + rows[2:])
    size = "\\small" if widest <= 62 else ("\\footnotesize" if widest <= 78 else "\\scriptsize")
    t = ["\\begin{table}[htbp]", "\\caption{%s}" % caption, "\\label{%s}" % label,
         "\\centering",
         "%s\\setlength{\\tabcolsep}{3.5pt}%%" % size,
         "\\begin{tabularx}{\\textwidth}{@{}%s@{}}" % cols, "\\toprule",
         " & ".join("\\textbf{%s}" % inline(c) for c in head) + " \\\\",
         "\\midrule"]
    for r in body:
        t.append(" & ".join(inline(c) for c in r) + " \\\\")
    t += ["\\bottomrule", "\\end{tabularx}", "\\end{table}", ""]
    return t


def split_row(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def convert(md: str, supplementary=False):
    lines = md.split("\n")
    out, i, n_tab, n_fig = [], 0, 0, 0
    buf = []
    section = [""]

    def flush():
        if buf:
            out.append(inline(" ".join(buf).strip()))
            out.append("")
            buf.clear()

    while i < len(lines):
        L = lines[i]
        m = re.match(r"^!\[\]\((figures/[^)]+)\)", L)
        if m:
            flush()
            path = m.group(1)
            cap = ""
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].startswith("**Figure"):
                cap_lines = []
                while j < len(lines) and lines[j].strip():
                    cap_lines.append(lines[j]); j += 1
                cap = " ".join(cap_lines)
                i = j
            else:
                i += 1
            cap = re.sub(r"^\*\*Figure (\d+)\*\*\s*\|\s*", "", cap)
            n_fig += 1
            base = os.path.basename(path)
            out += ["\\begin{figure}[htbp]", "\\centering",
                    "\\includegraphics[width=0.95\\textwidth]{figures/%s}" % base,
                    "\\caption{%s}" % inline(cap),
                    "\\label{fig:%s}" % base.replace(".png", "").replace("_", "-"),
                    "\\end{figure}", ""]
            continue
        if L.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[-: |]+\|$", lines[i + 1]):
            flush()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(split_row(lines[i])); i += 1
            if supplementary:
                n_tab += 1
                cap, lab = ("%s." % (section[0] or "Supplementary table %d" % n_tab),
                            "tab:s%d" % n_tab)
            else:
                cap, lab = MAIN_TABLE_CAPTIONS[n_tab][1], MAIN_TABLE_CAPTIONS[n_tab][0]
                n_tab += 1
            out += table(rows, cap, lab)
            continue
        if L.strip() == "$$":
            flush()
            i += 1
            eq = []
            while i < len(lines) and lines[i].strip() != "$$":
                eq.append(lines[i]); i += 1
            i += 1
            out += ["\\begin{equation*}"] + eq + ["\\end{equation*}", ""]
            continue
        if L.startswith("#"):
            flush()
            lvl = len(L) - len(L.lstrip("#"))
            title = L.lstrip("# ").strip()
            title = re.sub(r"^\d+\.\d+\s+", "", title)
            section[0] = title
            cmd = {1: "section", 2: "subsection", 3: "subsubsection"}[min(lvl, 3)]
            out += ["\\%s{%s}" % (cmd, inline(title)), ""]
            i += 1
            continue
        if not L.strip():
            flush(); i += 1; continue
        if L.startswith("- "):
            flush()
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(lines[i][2:]); i += 1
            out += ["\\begin{itemize}"] + ["\\item %s" % inline(x) for x in items] + \
                   ["\\end{itemize}", ""]
            continue
        buf.append(L.strip())
        i += 1
    flush()
    return "\n".join(out)


PREAMBLE = r"""%% Single-file Springer Nature submission, generated by analysis/make_latex.py
%% from docs/*.md. Do not edit by hand: edit the markdown and regenerate.
%% sn-basic compiles cleanly with the bibliography as it stands. The Nature
%% Portfolio style (sn-nature) additionally requires publisher/address fields in
%% every @inproceedings entry; switch to it once those are filled in from the
%% publishers' records.
%%\documentclass[pdflatex,lineno,sn-nature]{sn-jnl}
\documentclass[pdflatex,lineno,sn-basic]{sn-jnl}

\usepackage{graphicx}
\usepackage{tabularx}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\newcolumntype{R}{>{\raggedleft\arraybackslash}X}
\newcolumntype{A}[1]{>{\hsize=#1\hsize\raggedright\arraybackslash}X}
\newcolumntype{B}[1]{>{\hsize=#1\hsize\raggedleft\arraybackslash}X}
\usepackage{multirow}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{amsthm}
\usepackage{mathrsfs}
\usepackage[title]{appendix}
\usepackage{xcolor}
\usepackage{textcomp}
\usepackage{manyfoot}
\usepackage{booktabs}
\usepackage{algorithm}
\usepackage{algorithmicx}
\usepackage{algpseudocode}

\theoremstyle{thmstyleone}
\newtheorem{theorem}{Theorem}
\newtheorem{proposition}[theorem]{Proposition}
\theoremstyle{thmstyletwo}
\newtheorem{example}{Example}
\theoremstyle{thmstylethree}
\newtheorem{definition}{Definition}
\raggedbottom

\begin{document}

\title[When Structural Priors Help---and Hurt---Causal Learning]{%TITLE%}

\author*[1]{\fnm{Nikita} \sur{Tarasov}}\email{dev.nikita@outlook.com}
\author[1]{\fnm{Roman} \sur{Zinko}}\email{roman.v.zinko@lpnu.ua}
\affil[1]{\orgname{Lviv Polytechnic National University},
\orgaddress{\city{Lviv}, \country{Ukraine}}}

\abstract{%ABSTRACT%}

\keywords{causal discovery, structural priors, distribution shift,
amortised inference, root-cause analysis, pre-registration}

\maketitle
"""

TAIL = r"""
\section*{Declarations}
%DECLARATIONS%

\bibliography{spcrl-refs}

\end{document}
"""


def main(outdir=None):
    outdir = outdir or os.path.join(ROOT, "manuscript")
    os.makedirs(outdir, exist_ok=True)
    md = open(os.path.join(ROOT, "docs", "build", "manuscript_full.md")).read()

    title = re.search(r'(?m)^title:\s*"(.+)"\s*$', md).group(1).strip()
    abstract = re.search(r"## Abstract\n(.*?)\n#", md, re.S).group(1).strip()
    abstract = inline(" ".join(x.strip() for x in abstract.split("\n") if x.strip()))

    body = md[md.index("# Introduction"):]
    main_md, supp_md = body.split("# Supplementary Notes")
    tex = PREAMBLE.replace("%TITLE%", inline(title)).replace("%ABSTRACT%", abstract)
    tex += "\n" + convert(main_md) + "\n"
    tex += "\\begin{appendices}\n\\section{Supplementary Notes}\n"
    tex += convert(supp_md, supplementary=True).replace("\\section{", "\\subsection*{")
    tex += "\n\\end{appendices}\n" + TAIL

    decl = open(os.path.join(ROOT, "docs", "submission",
                             "declarations.md")).read()
    decl = decl[decl.index("## Author contributions"):]
    tex = tex.replace("%DECLARATIONS%", convert(decl).replace(
        "\\subsection{", "\\subsection*{"))
    open(os.path.join(outdir, "spcrl_npjai.tex"), "w").write(tex)
    for f in ("sn-jnl.cls", "sn-basic.bst", "sn-nature.bst"):
        src = os.path.join(TEMPLATE, "bst", f)
        if not os.path.exists(src):
            src = os.path.join(TEMPLATE, f)
        if os.path.abspath(src) != os.path.abspath(os.path.join(outdir, f)):
            shutil.copy(src, outdir)
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    used = set(re.findall(r"figures/([\w.-]+\.png)", md))
    for f in sorted(used):
        shutil.copy(os.path.join(ROOT, "figures", f), figdir)
    shutil.copy(os.path.join(ROOT, "docs", "refs.bib"),
                os.path.join(outdir, "spcrl-refs.bib"))
    print("wrote", os.path.join(outdir, "spcrl_npjai.tex"))


if __name__ == "__main__":
    main(*sys.argv[1:2])
