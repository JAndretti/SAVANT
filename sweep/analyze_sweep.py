#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_sweep.py — analyse the runs produced by sweep/run_sweep.sh.

Reads sweep/results/<study>/<tag>.{meta,log,csv}, rebuilds each run's option
dict from the recorded command line, and produces:

    sweep/figures/*.png     one figure per study + an overview
    sweep/report.tex        the report: what each knob does, the figures,
                            the paired statistics, and the winning command
    sweep/report.pdf        compiled with pdflatex, when it is installed

Every run of a study uses the same (n, m, seed), so instance k is byte-identical
across runs (cw.c:2064 seeds instance k with seed+k). Comparisons are therefore
*paired*: we compare cost_i(A) with cost_i(B) on the same instance and report the
mean of the differences, its 95 % CI, and a distribution-free sign test. That is
far tighter than comparing two means with independent standard deviations.

The recommended configuration at the end is *derived*, not hand-picked: a knob
enters it only if its best setting beat its own default significantly. That
combination is then re-run, on the sweep seed and on a seed the sweep never saw,
because tuning knobs one at a time does not make them additive (see the `tuned`
study, where the naive combination is worse than the defaults).

Usage:
    uv run sweep/analyze_sweep.py
    uv run sweep/analyze_sweep.py --no-verify      # skip the confirmation runs
    uv run sweep/analyze_sweep.py --no-pdf         # emit .tex only
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import NullFormatter

# --------------------------------------------------------------- palette
# Categorical slots in fixed order (never cycled); sequential = one hue,
# light -> dark; diverging = blue <-> red with a neutral grey midpoint.
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
       "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
GRID = "#e3e2de"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list(
    "div_br", ["#104281", "#2a78d6", "#86b6ef", "#f0efec",
               "#f2a3a2", "#e34948", "#99201f"])

# Font sizes are set for a figure that will be scaled to \linewidth on a
# landscape A4 page (~25.7 cm): a 12.5 in figure lands at ~0.8x, so 10 pt here
# reads as ~8 pt on the page.
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "figure.facecolor": "white",
    "axes.facecolor": "white", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
    "axes.edgecolor": INK3, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK2, "legend.frameon": False, "legend.fontsize": 9,
    "lines.linewidth": 2.0, "lines.markersize": 5,
})


def style(ax, ygrid=True, xgrid=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(ygrid, axis="y")
    ax.grid(xgrid, axis="x")
    return ax


# ----------------------------------------------------------- cw defaults
# Only what the sweep varies. Kept in one place so a run that leaves an option
# out is still comparable with one that sets it explicitly.
FLAGS = {"--no-sa", "--exact", "--2opt", "--round", "--random",
         "--per-instance", "-q", "--check"}
DEFAULTS = {
    "n": 100, "m": 1000, "seed": 42, "sa-steps": 1000, "init": "cw",
    "ops": "1,1,1,0", "or-max": 3, "sa-knn": 20, "restarts": 1,
    "split": "off", "split-every": 0, "split-tour": "both",
    "pick": 2, "pick-crit": "lb", "pick-eps": 0.3,
    "t-accept": 0.001, "t-decades": 2, "lambda": 1.0, "mu": 0.0,
    "knn": 0, "threads": 0, "cw-rand": "perturb", "cw-alpha": 0.03,
    "no-sa": False, "exact": False, "2opt": False,
}
NUMERIC = {"n", "m", "seed", "sa-steps", "or-max", "sa-knn", "restarts",
           "split-every", "pick", "t-decades", "knn", "threads"}
FLOAT = {"pick-eps", "t-accept", "lambda", "mu", "cw-alpha"}

LOG_PATTERNS = (
    ("cost_before", r"mean cost before annealing\s*:\s*([\d.]+)"),
    ("cost_after", r"mean cost after annealing\s*:\s*([\d.]+)"),
    ("std", r"std deviation\s*:\s*([\d.]+)"),
    ("split_gain", r"mean Split gain\s*:\s*([\d.eE+-]+)"),
    ("accept_pct", r"annealing acceptance rate\s*:\s*([\d.]+) %"),
    ("t0", r"mean calibrated T0\s*:\s*([\d.eE+-]+)\)"),
    ("routes", r"routes \(mean\)\s*:\s*([\d.]+)"),
    ("wall_s", r"total time\s*:\s*([\d.]+) s\s*\(wall\)"),
    ("cpu_s", r"total time\s*:.*?([\d.]+) s \(cumulative CPU\)"),
    ("ms_per_inst", r"time / instance\s*:\s*([\d.]+) ms"),
    ("throughput", r"throughput\s*:\s*([\d.]+)"),
    ("infeasible", r"instances\s*:\s*\d+\s*\((\d+) infeasible\)"),
)


def parse_cmd(cmd: str) -> dict:
    """Rebuild the option dict from the recorded command line."""
    o = dict(DEFAULTS)
    toks = cmd.split()[1:]
    i = 0
    while i < len(toks):
        t = toks[i]
        if not t.startswith("-"):
            i += 1
            continue
        if t in FLAGS:
            o[t.lstrip("-")] = True
            i += 1
            continue
        key, val = t.lstrip("-"), (toks[i + 1] if i + 1 < len(toks) else "")
        if key == "csv":
            i += 2
            continue
        if key in NUMERIC:
            val = int(val)
        elif key in FLOAT:
            val = float(val)
        o[key] = val
        i += 2
    if o.get("no-sa"):
        o["sa-steps"] = 0
    return o


class Run:
    __slots__ = ("study", "tag", "opts", "ok", "shell_s", "log", "cost",
                 "cost_init", "time_ms", "routes_v", "feasible")

    def __init__(self, study, tag, opts, ok, shell_s, log, arrays):
        self.study, self.tag, self.opts, self.ok = study, tag, opts, ok
        self.shell_s, self.log = shell_s, log
        (self.cost, self.cost_init, self.time_ms,
         self.routes_v, self.feasible) = arrays

    # instances are identical iff (n, m, seed) match -> pairing key
    @property
    def instkey(self):
        return (self.opts["n"], self.opts["m"], self.opts["seed"])

    @property
    def mean(self):
        return float(self.cost.mean())

    def __repr__(self):
        return f"<Run {self.study}/{self.tag} mean={self.mean:.5f}>"


def read_run(base, study, tag_default):
    meta = {}
    with open(base + ".meta", encoding="utf-8") as f:
        for line in f:
            k, _, v = line.rstrip("\n").partition("=")
            meta[k] = v
    ok = meta.get("exit") == "0"
    opts = parse_cmd(meta.get("cmd", ""))
    log = {}
    if os.path.exists(base + ".log"):
        txt = open(base + ".log", encoding="utf-8", errors="replace").read()
        for key, pat in LOG_PATTERNS:
            mm = re.search(pat, txt)
            if mm:
                log[key] = float(mm.group(1))
    cols = [[], [], [], [], []]
    if ok and os.path.exists(base + ".csv"):
        with open(base + ".csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cols[0].append(float(row["cost_annealed"]))
                cols[1].append(float(row["cost_init"]))
                cols[2].append(float(row["time_ms"]))
                cols[3].append(float(row["routes"]))
                cols[4].append(int(row["feasible"]))
    arrays = tuple(np.asarray(c, dtype=float) for c in cols)
    if ok and arrays[0].size == 0:
        ok = False
    return Run(study, meta.get("tag", tag_default), opts, ok,
               float(meta.get("wall_s", "nan")), log, arrays)


def load(results_dir: str) -> list[Run]:
    runs = []
    for study in sorted(os.listdir(results_dir)):
        sdir = os.path.join(results_dir, study)
        if not os.path.isdir(sdir):
            continue
        for fn in sorted(os.listdir(sdir)):
            if fn.endswith(".meta"):
                runs.append(read_run(os.path.join(sdir, fn[:-5]), study, fn[:-5]))
    return [r for r in runs if r.ok]


# ------------------------------------------------------------ statistics
def paired(a: Run, b: Run) -> dict:
    """b relative to a, on the shared instance set. Negative = b is better."""
    if a is None or b is None or a.instkey != b.instkey or a.cost.size != b.cost.size:
        return {}
    d = b.cost - a.cost
    n = d.size
    mean = float(d.mean())
    sem = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    half = 1.96 * sem
    wins = int((d < -1e-12).sum())      # b strictly better
    losses = int((d > 1e-12).sum())
    ties = n - wins - losses
    k = wins + losses
    if k:
        z = (wins - k / 2) / math.sqrt(k / 4)
        p = math.erfc(abs(z) / math.sqrt(2))
    else:
        p = 1.0
    base = float(a.cost.mean())
    return {
        "delta": mean, "pct": 100 * mean / base,
        "lo": mean - half, "hi": mean + half,
        "pct_lo": 100 * (mean - half) / base, "pct_hi": 100 * (mean + half) / base,
        "wins": wins, "losses": losses, "ties": ties, "p": p,
        "sig": abs(mean) > half and p < 0.05,
    }


def sel(runs, study, **filt):
    out = []
    for r in runs:
        if r.study != study:
            continue
        if all(r.opts.get(k) == v for k, v in filt.items()):
            out.append(r)
    return out


def one(runs, study, tag):
    for r in runs:
        if r.study == study and r.tag == tag:
            return r
    return None


# ------------------------------------------------------------ LaTeX layer
# Labels are shared with the matplotlib figures, which are happy with unicode;
# pdflatex with inputenc is not. Mapped to plain names rather than math so the
# result is still safe inside \texttt{}.
UNI = {"α": "alpha", "λ": "lambda", "μ": "mu", "σ": "sigma", "Δ": "delta",
       "×": "x", "·": " ", "→": "->", "★": "*", "—": "--", "–": "-",
       "≤": "<=", "≥": ">=", "⁵": "5", "⁶": "6"}


def esc(s) -> str:
    """Escape the LaTeX active characters that occur in our fields."""
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("_", r"\_"), ("#", r"\#"), ("$", r"\$"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")):
        s = s.replace(a, b)
    for a, b in UNI.items():
        s = s.replace(a, b)
    # `--` ligatures into an en-dash even inside \texttt, which silently turns
    # `--lambda` into `-lambda`. Must come after the brace escaping above.
    s = s.replace("--", "-{}-")
    # anything still outside ASCII would be a fatal pdflatex error
    return s.encode("ascii", "replace").decode("ascii")


def tt(s) -> str:
    """Monospace, escaped — for option names, tags and values."""
    return r"\texttt{%s}" % esc(s)


class Doc:
    """Minimal LaTeX body builder. Everything dynamic goes through esc()."""

    def __init__(self):
        self.body = []

    def raw(self, s):
        self.body.append(s)

    def sec(self, title, label=None):
        # the label must follow \section, or it binds to the previous number
        self.body.append("\n\\section{%s}%s\n"
                         % (title, ("\\label{%s}" % label) if label else ""))

    def sub(self, title):
        self.body.append("\n\\subsection{%s}\n" % title)

    def p(self, text):
        self.body.append(text.strip() + "\n")

    def note(self, text):
        self.body.append("\\begin{quote}\\small %s\\end{quote}\n" % text.strip())

    def fig(self, name, caption):
        self.body.append(
            "\\begin{figure}[H]\\centering\n"
            "  \\includegraphics[width=\\linewidth,height=0.74\\textheight,keepaspectratio]{figures/%s}\n"
            "  \\caption{%s}\n\\end{figure}\n" % (name, caption))

    def verb(self, text):
        self.body.append("\\begin{verbatim}\n%s\n\\end{verbatim}\n" % text)

    def table(self, header, rows, caption, align=None, long=False):
        if not rows:
            return
        ncol = len(header)
        align = align or ("l" * ncol)
        head = " & ".join(r"\textbf{%s}" % h for h in header) + r" \\"
        body = "\n".join("  " + " & ".join(str(c) for c in r) + r" \\" for r in rows)
        if long:
            self.body.append(
                "{\\small\\begin{longtable}{%s}\n"
                "\\caption{%s}\\\\\n\\toprule\n%s\n\\midrule\n\\endfirsthead\n"
                "\\toprule\n%s\n\\midrule\n\\endhead\n%s\n\\bottomrule\n"
                "\\end{longtable}}\n" % (align, caption, head, head, body))
        else:
            self.body.append(
                "\\begin{table}[H]\\centering\\small\n\\begin{tabular}{%s}\n"
                "\\toprule\n%s\n\\midrule\n%s\n\\bottomrule\n\\end{tabular}\n"
                "\\caption{%s}\n\\end{table}\n" % (align, head, body, caption))


DHEAD = (r"$\Delta$ (\%)", r"95\,\% CI", "win/loss")


def stat_cells(s):
    """A paired stat as three table cells: delta, CI, win/loss."""
    if not s:
        return ("--", "--", "--")
    d = "%+.3f" % s["pct"]
    if s["sig"]:
        d = r"\textbf{%s}" % d
    return (d, "[%+.3f,\\;%+.3f]" % (s["pct_lo"], s["pct_hi"]),
            "%d/%d" % (s["wins"], s["losses"]))


# --------------------------------------------------------------- helpers
def bar_delta(ax, labels, stats, title, xlabel="Δ mean cost vs baseline (%)"):
    """Horizontal bars of a paired delta, diverging colour, CI whiskers."""
    y = np.arange(len(labels))
    pct = np.array([s["pct"] for s in stats])
    lo = np.array([s["pct_lo"] for s in stats])
    hi = np.array([s["pct_hi"] for s in stats])
    lim = max(1e-9, float(np.abs(np.concatenate([lo, hi])).max()))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    colors = [DIV(norm(v)) for v in pct]
    ax.barh(y, pct, color=colors, height=0.68, edgecolor="white", linewidth=0.8)
    ax.errorbar(pct, y, xerr=[pct - lo, hi - pct], fmt="none",
                ecolor=INK2, elinewidth=1.0, capsize=2.5)
    ax.axvline(0, color=INK3, lw=1.0)
    ax.set_yticks(y, labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    style(ax, ygrid=False, xgrid=True)


def heat(ax, mat, rowlab, collab, title, cbar_label, diverging=True, fmt="{:+.2f}"):
    mat = np.asarray(mat, dtype=float)
    finite = mat[np.isfinite(mat)]
    if diverging:
        lim = max(1e-9, float(np.abs(finite).max()) if finite.size else 1.0)
        im = ax.imshow(mat, cmap=DIV, aspect="auto",
                       norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim))
    else:
        im = ax.imshow(mat, cmap=SEQ, aspect="auto")
    ax.set_xticks(range(len(collab)), collab, fontsize=9)
    ax.set_yticks(range(len(rowlab)), rowlab, fontsize=9)
    ax.set_title(title, loc="left")
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isfinite(mat[i, j]):
                continue
            v = mat[i, j]
            rel = abs(v) / (max(abs(finite.min()), abs(finite.max())) + 1e-12)
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                    color="white" if rel > 0.62 else INK)
    cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(cbar_label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)
    return im


def savefig(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure  {os.path.relpath(path)}")
    return path


def equiv_budget(target, steps, costs):
    """Smallest budget whose cost reaches `target`, interpolated in log(steps).

    Returns None if the series never gets there within the sampled range.
    """
    for i, c in enumerate(costs):
        if c <= target:
            if i == 0:
                return float(steps[0])
            c0, l0 = costs[i - 1], math.log(steps[i - 1])
            l1 = math.log(steps[i])
            if c0 <= c:
                return float(steps[i])
            t = (c0 - target) / (c0 - c)
            return math.exp(l0 + t * (l1 - l0))
    return None


OPNAMES = ("rel", "swap", "2opt", "or")


def ops_label(ops: str) -> str:
    bits = [int(float(x) > 0) for x in ops.split(",")]
    on = [OPNAMES[i] for i, b in enumerate(bits) if b]
    return "+".join(on) if on else "none"


# =========================================================== the studies
def study_init(runs, out, doc):
    rs = sel(runs, "init")
    if not rs:
        return
    ns = sorted({r.opts["n"] for r in rs})

    doc.sec("Initialisation: where the annealing starts from", "sec:init")
    doc.p(r"""
\textbf{What it controls.} \texttt{-{}-init} chooses the solution the simulated
annealing starts from. \texttt{cw} (the default) runs the Clarke \& Wright
savings construction: every pair of customers gets a score
$s(i,j) = d_{0i} + d_{0j} - \lambda\,d_{ij} + \mu\,|d_{0i}-d_{0j}|$, measuring how
much is saved by serving $i$ and $j$ on one route instead of two, and merges
route endpoints greedily in decreasing score while capacity allows.
\texttt{random} throws that away: it shuffles the customers and cuts the
permutation into routes at the capacity limit (first fit). Under
\texttt{-{}-restarts} each restart draws its own permutation.
""")
    doc.p(r"""
\textbf{The question.} A random start is $100$--$380\,\%$ worse before annealing.
Does the annealing erase that, and how much budget does it cost to do so?
""")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, n in zip(axes.ravel(), ns):
        for k, init in enumerate(("cw", "random")):
            sub = sorted(sel(runs, "init", n=n, init=init),
                         key=lambda r: r.opts["sa-steps"])
            if not sub:
                continue
            ax.plot([r.opts["sa-steps"] for r in sub], [r.mean for r in sub],
                    "o-", color=CAT[k], label=f"--init {init}")
        ax.set_xscale("log")
        ax.set_title(f"n = {n}", loc="left")
        ax.set_xlabel("SA steps")
        ax.set_ylabel("mean cost")
        style(ax)
        ax.legend(loc="upper right")
    fig.tight_layout()
    savefig(fig, out, "init_curves.png")
    doc.fig("init_curves.png",
            "Mean cost against the annealing budget, for both initialisations, "
            "at four dimensions. Same budget on both sides of each panel.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    rows = []
    for k, n in enumerate(ns):
        cw = {r.opts["sa-steps"]: r for r in sel(runs, "init", n=n, init="cw")}
        rd = {r.opts["sa-steps"]: r for r in sel(runs, "init", n=n, init="random")}
        steps = sorted(set(cw) & set(rd))
        pct, lo, hi = [], [], []
        for s in steps:
            st = paired(cw[s], rd[s])
            pct.append(st["pct"]); lo.append(st["pct_lo"]); hi.append(st["pct_hi"])
        ax.plot(steps, pct, "o-", color=CAT[k], label=f"n = {n}")
        ax.fill_between(steps, lo, hi, color=CAT[k], alpha=0.15, linewidth=0)
        hit = next((s for s, h in zip(steps, hi) if h <= 0), None)
        rows.append([n, "%+.2f" % pct[0], "%+.2f" % pct[-1],
                     ("%s" % f"{hit:,}") if hit else "not within $10^6$"])
    ax.axhline(0, color=INK3, lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("SA steps")
    ax.set_ylabel("excess cost of random init (%)")
    ax.set_title("Penalty for starting from a random solution", loc="left")
    ax.legend()
    style(ax)

    ax = axes[1]
    brows = []
    for k, n in enumerate(ns):
        cw = sorted(sel(runs, "init", n=n, init="cw"), key=lambda r: r.opts["sa-steps"])
        rd = sorted(sel(runs, "init", n=n, init="random"), key=lambda r: r.opts["sa-steps"])
        if not cw or not rd:
            continue
        rsteps = [r.opts["sa-steps"] for r in rd]
        rcost = [r.mean for r in rd]
        xs, ys = [], []
        for r in cw:
            eq = equiv_budget(r.mean, rsteps, rcost)
            if eq:
                xs.append(r.opts["sa-steps"]); ys.append(eq / r.opts["sa-steps"])
        if xs:
            ax.plot(xs, ys, "o-", color=CAT[k], label=f"n = {n}")
            brows.append([n, "%.0f$\\times$" % ys[0],
                          "%.0f$\\times$" % ys[len(ys) // 2],
                          "%.0f$\\times$" % ys[-1]])
    ax.axhline(1, color=INK3, lw=1.0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("C&W budget S (steps)")
    ax.set_ylabel("random-start budget needed / S")
    ax.set_title("Cost of throwing away the construction", loc="left")
    ax.legend()
    style(ax)
    fig.tight_layout()
    savefig(fig, out, "init_gap.png")
    doc.fig("init_gap.png",
            "Left: paired excess of the random start over Clarke \\& Wright at "
            "equal budget, with a 95\\,\\% band. Right: how many steps the random "
            "start needs to reach what Clarke \\& Wright reaches at $S$ steps.")

    doc.table([r"$n$", r"excess at $10^3$ (\%)", r"excess at $10^6$ (\%)",
               "steps to catch up"], rows,
              "Penalty for a random start, paired against the C\\&W start at the "
              "same budget. `Catch up' is the first budget at which the upper end "
              "of the 95\\,\\% CI reaches zero.", align="rrrr")
    doc.table([r"$n$", "at smallest $S$", "mid-range $S$", "at largest $S$"], brows,
              "Budget multiplier: steps the random start needs to match the C\\&W "
              "start, relative to the C\\&W budget. A ratio of 1 means the head "
              "start has been fully repaid.", align="rrrr")
    doc.p(r"""
\textbf{Reading.} The random start converges to parity but never overtakes: it
buys no extra diversity that the annealing can exploit. The gap closes late and
costs dearly along the way --- at $n=200$ it still needs several times the budget
at $10^5$ steps. Keep the construction.
""")


def study_ops(runs, out, doc):
    rs = sel(runs, "ops")
    if not rs:
        return
    base = one(runs, "ops", "sub_1110")           # the stock default 1,1,1,0

    doc.sec("Move operators", "sec:ops")
    doc.p(r"""
\textbf{What it controls.} \texttt{-{}-ops r,s,t,o} sets the relative
probabilities of the four neighbourhood moves the annealing draws from. Each
move first picks a vertex $u$ (Section~\ref{sec:pick}) and a partner $v$ near it
(Section~\ref{sec:knn}), then:
\begin{itemize}\itemsep2pt
  \item \textbf{relocate} (\texttt{mv\_relocate}, \texttt{cw.c:980}) --- take $u$
        out of its route and re-insert it next to $v$, possibly in another route.
  \item \textbf{swap} (\texttt{mv\_swap}, \texttt{cw.c:1017}) --- exchange the
        positions of $u$ and $v$.
  \item \textbf{2-opt} (\texttt{mv\_2opt}, \texttt{cw.c:1137}) --- replace the two
        edges carried by $u$ and $v$ by the other pairing; inside one route this
        reverses a segment, across two routes it exchanges their tails.
  \item \textbf{or-opt} (\texttt{mv\_oropt}, \texttt{cw.c:1082}) --- move a whole
        run of 2 to \texttt{-{}-or-max} consecutive customers starting at $u$,
        optionally reversed, next to $v$.
\end{itemize}
Every move is accepted by the Metropolis rule on its exact cost delta. The
default is \texttt{1,1,1,0}: or-opt is present in the code but switched off.
""")
    doc.p(r"""
\textbf{The question.} Is the default subset the right one, is the uniform
weighting right, and is or-opt off for a good reason?
""")

    subs = sorted([r for r in rs if r.tag.startswith("sub_")], key=lambda r: r.mean)
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1])
    ax = fig.add_subplot(gs[0, :])
    stats = [paired(base, r) for r in subs]
    bar_delta(ax, [ops_label(r.opts["ops"]) for r in subs], stats,
              "Operator subsets vs the default rel+swap+2opt (--ops 1,1,1,0)")

    ax = fig.add_subplot(gs[1, 0])
    ws = sorted([r for r in rs if r.tag.startswith("w_")], key=lambda r: r.mean)
    if ws:
        bar_delta(ax, [r.opts["ops"] for r in ws], [paired(base, r) for r in ws],
                  "Unbalanced operator weights")

    ax = fig.add_subplot(gs[1, 1])
    om = sorted([r for r in rs if r.tag.startswith("ormax_")],
                key=lambda r: r.opts["or-max"])
    orows = []
    if om:
        ref = om[0]
        st = [paired(ref, r) for r in om]
        x = [r.opts["or-max"] for r in om]
        ax.plot(x, [s["pct"] for s in st], "o-", color=CAT[2])
        ax.fill_between(x, [s["pct_lo"] for s in st], [s["pct_hi"] for s in st],
                        color=CAT[2], alpha=0.15, linewidth=0)
        ax.axhline(0, color=INK3, lw=1.0)
        ax.set_xlabel("--or-max (max or-opt segment length)")
        ax.set_ylabel(f"Δ vs or-max {ref.opts['or-max']} (%)")
        ax.set_title("Or-opt segment length (with --ops 1,1,1,1)", loc="left")
        style(ax)
        orows = [[r.opts["or-max"], "%.5f" % r.mean, *stat_cells(s)]
                 for r, s in zip(om, st)]
    fig.tight_layout()
    savefig(fig, out, "ops.png")
    doc.fig("ops.png",
            "Top: all 15 non-empty operator subsets. Bottom left: unbalanced "
            "weightings. Bottom right: the or-opt segment length cap. Bars are "
            "paired deltas against the default with 95\\,\\% CIs.")

    doc.table(["subset", "mean cost", *DHEAD],
              [[ops_label(r.opts["ops"]), "%.5f" % r.mean, *stat_cells(s)]
               for r, s in zip(subs, stats)],
              "All 15 non-empty operator subsets, paired against the default "
              "\\texttt{-{}-ops 1,1,1,0}. Bold = significant at 95\\,\\% and by "
              "sign test.", align="lrrrr")
    doc.table(["weights", "mean cost", *DHEAD],
              [[tt(r.opts["ops"]), "%.5f" % r.mean, *stat_cells(paired(base, r))]
               for r in ws],
              "Unbalanced weightings, against the same default.", align="lrrrr")
    if orows:
        doc.table([r"\texttt{-{}-or-max}", "mean cost", *DHEAD], orows,
                  "Or-opt segment length, with or-opt enabled "
                  "(\\texttt{-{}-ops 1,1,1,1}). Valid range is 2--8 "
                  "(\\texttt{cw.c:2042}).", align="rrrrr")
    doc.p(r"""
\textbf{Reading.} The default subset wins: every other subset is worse, and
turning or-opt on costs a small but significant amount. Or-opt overlaps with
relocate (a length-1 relocate is the same move) while being more expensive per
draw, so at a fixed step count it buys less. \texttt{-{}-or-max} is then
irrelevant --- its whole range sits inside the noise. Leaving or-opt off is the
right default.
""")


def study_knn(runs, out, doc):
    rs = sel(runs, "knn")
    if not rs:
        return
    ns = sorted({r.opts["n"] for r in rs})
    Ks = sorted({r.opts["sa-knn"] for r in rs})

    doc.sec("Candidate neighbourhood", "sec:knn")
    doc.p(r"""
\textbf{What it controls.} Once a move has picked the vertex $u$, it needs a
partner $v$. \texttt{sa\_cand} (\texttt{cw.c:971}) draws $v$ uniformly from the
$K$ nearest neighbours of $u$, with $K=$ \texttt{-{}-sa-knn}; at $K=0$ it draws
uniformly from all customers. This is what keeps a move geometrically local, so
that the cost delta has a chance of being negative. $K$ is clamped to $n-1$
(\texttt{cw.c:1399}).
""")
    doc.note(r"""
$K$ does double duty: at $K=0$ the kNN lists are not built, and
\texttt{cw.c:1400} then \emph{silently} forces uniform vertex selection as well,
whatever \texttt{-{}-pick} says. So $K=0$ disables two mechanisms, not one ---
see Section~\ref{sec:pick}.
""")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4))
    rows = []
    for k, n in enumerate(ns):
        by = {r.opts["sa-knn"]: r for r in sel(runs, "knn", n=n)}
        avail = [K for K in Ks if K in by]
        best = min(by[K].mean for K in avail)
        axes[0].plot([Ks.index(K) for K in avail],
                     [100 * (by[K].mean - best) / best for K in avail],
                     "o-", color=CAT[k], label=f"n = {n}")
        axes[1].plot([Ks.index(K) for K in avail],
                     [float(np.median(by[K].time_ms)) for K in avail],
                     "o-", color=CAT[k], label=f"n = {n}")
        bestK = min(avail, key=lambda K: by[K].mean)
        ref = by[20] if 20 in by else by[avail[0]]
        for K in avail:
            rows.append([n, K, (min(K, n - 1) if K else "uniform"),
                         "%.5f" % by[K].mean, *stat_cells(paired(ref, by[K])),
                         "%.3f" % float(np.median(by[K].time_ms)),
                         r"$\star$" if K == bestK else ""])
    for ax, ttl, yl in ((axes[0], "Solution quality", "excess over the best K at this n (%)"),
                        (axes[1], "Cost per instance", "median time / instance (ms)")):
        ax.set_xticks(range(len(Ks)), [("unif" if K == 0 else str(K)) for K in Ks])
        ax.set_xlabel("--sa-knn")
        ax.set_ylabel(yl)
        ax.set_title(ttl, loc="left")
        ax.legend()
        style(ax)

    mat, rl = [], []
    for n in ns:
        by = {r.opts["sa-knn"]: r for r in sel(runs, "knn", n=n)}
        ref = by.get(20) or by[sorted(by)[0]]
        mat.append([paired(ref, by[K])["pct"] if K in by else np.nan for K in Ks])
        rl.append(f"n={n}")
    heat(axes[2], mat, rl, [("unif" if K == 0 else str(K)) for K in Ks],
         "Δ vs K = 20 (%)", "Δ cost (%)")
    fig.tight_layout()
    savefig(fig, out, "knn.png")
    doc.fig("knn.png",
            "Quality (left, as excess over the best $K$ at that $n$), cost per "
            "instance (middle) and the paired delta against $K=20$ (right).")

    doc.table([r"$n$", "$K$", "effective $K$", "mean cost", *DHEAD, "ms/inst", ""],
              rows,
              "\\texttt{-{}-sa-knn} across dimension, paired against $K=20$. "
              "The effective $K$ is $\\min(K, n-1)$, which is why the rows for "
              "$n=20$ at $K \\ge 20$ are exactly identical runs.",
              align="rrrrrrrrl", long=True)
    doc.p(r"""
\textbf{Reading.} $K=20$ is the optimum at every dimension tested, and the curve
is a clean U: too small a list starves the move of good partners, too large a
one dilutes it with distant ones that will be rejected. The default needs no
change --- and notably it does not need to grow with $n$.
""")


def study_timing(runs, out, doc):
    rs = sel(runs, "timing")
    if not rs:
        return
    nosa = {r.opts["n"]: r for r in rs if r.tag.endswith("_nosa")}
    sa = {r.opts["n"]: r for r in rs if r.tag.endswith("_sa")}
    sa10 = {r.opts["n"]: r for r in rs if r.tag.endswith("_sa10")}
    ns = sorted(set(nosa) & set(sa) & set(sa10))

    doc.sec("Compute cost", "sec:timing")
    doc.p(r"""
\textbf{What is measured.} Two phases: the Clarke \& Wright construction, which
builds a savings list ($O(n^2)$ exactly, for $n \le 1500$) and sorts it, and the
annealing, which does a fixed number of $O(1)$ steps regardless of $n$. Instances
are solved in parallel by OpenMP, one instance per thread, with no interaction
between them.
""")
    doc.note(r"""
\textbf{Metric: cumulative CPU time per instance, from the run log --- not the
\texttt{time\_ms} column of the CSV.} That column is wall time measured
\emph{inside} the 24-thread parallel region, so it absorbs memory contention: at
$n=1000$ it overstates the cost by about $2.5\times$ and is not additive, which
makes the annealing look roughly $10\times$ cheaper than it is. Use it for
latency distributions (third panel), never for cost accounting.
""")
    doc.p(r"""
The annealing cost is isolated as $(t(10S) - t(S))/9$ at equal $n$: both runs pay
the same construction, so it cancels exactly. Construction is then $t(S)$ minus
that, cross-checked against an independent \texttt{-{}-no-sa} run. A $2\times$
budget delta was tried first and is not enough at $n=1000$, where the run-to-run
spread on the construction alone ($\pm 3\,\%$, measured over three repetitions)
exceeds the entire annealing cost.
""")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4))
    ax = axes[0]
    if ns:
        def cpu(r):
            return 1000.0 * r.log.get("cpu_s", float("nan")) / max(r.opts["m"], 1)

        tot = np.array([cpu(sa[n]) for n in ns])
        tot10 = np.array([cpu(sa10[n]) for n in ns])
        raw = np.array([cpu(nosa[n]) for n in ns])
        annealing = np.clip((tot10 - tot) / 9.0, 1e-9, None)
        c = np.clip(tot - annealing, 1e-9, None)
        ax.plot(ns, c, "o-", color=CAT[0], label="construction (by subtraction)")
        ax.plot(ns, annealing, "o-", color=CAT[1],
                label=f"annealing, {sa[ns[0]].opts['sa-steps']:,} steps")
        ax.plot(ns, tot, "o--", color=INK3, label="total", linewidth=1.4)
        ax.plot(ns, raw, "^:", color=CAT[2], linewidth=1.4, markersize=4,
                label="--no-sa run (cross-check)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("n (customers)")
        ax.set_ylabel("CPU time / instance (ms)")
        ax.set_title("Where the time goes", loc="left")
        ax.legend(fontsize=8)
        style(ax)
        sl_c = np.polyfit(np.log(ns), np.log(c), 1)[0]
        sl_a = np.polyfit(np.log(ns), np.log(annealing), 1)[0]
        doc.table([r"$n$", "construction", "annealing", "total", "annealing share",
                   r"\texttt{-{}-no-sa}"],
                  [[n, "%.4f" % c[i], "%.4f" % annealing[i], "%.4f" % tot[i],
                    "%.1f\\,\\%%" % (100 * annealing[i] / tot[i]), "%.4f" % raw[i]]
                   for i, n in enumerate(ns)],
                  "CPU time per instance (ms). Empirical scaling on this grid: "
                  "construction $\\sim n^{%.2f}$, annealing $\\sim n^{%.2f}$ at a "
                  "fixed step count --- the step count does not depend on $n$, so "
                  "that exponent is pure cache behaviour." % (sl_c, sl_a),
                  align="rrrrrr")

    ax = axes[1]
    th = sorted([r for r in rs if r.tag.startswith("threads_")],
                key=lambda r: r.opts["threads"])
    if th:
        t = [r.opts["threads"] for r in th]
        w = [r.log.get("wall_s", np.nan) for r in th]
        sp = [w[0] / v if v else np.nan for v in w]
        ax.plot(t, sp, "o-", color=CAT[0], label="measured")
        ax.plot(t, t, "--", color=INK3, linewidth=1.2, label="linear")
        ax.set_xlabel("--threads")
        ax.set_ylabel(f"speed-up vs {t[0]} thread(s)")
        ax.set_title("Thread scaling", loc="left")
        ax.legend()
        style(ax)
        doc.table(["threads", "wall (s)", "speed-up", "efficiency"],
                  [[t[i], "%.3f" % w[i], "%.2f$\\times$" % sp[i],
                    "%.0f\\,\\%%" % (100 * sp[i] / (t[i] / t[0]))]
                   for i in range(len(t))],
                  "Thread scaling at $n=100$. Instances are independent, so the "
                  "loss at high thread counts is memory bandwidth and clock "
                  "throttling, not synchronisation.", align="rrrr")

    ax = axes[2]
    for k, n in enumerate(ns):
        v = np.sort(sa[n].time_ms)
        ax.plot(v, np.linspace(0, 100, v.size), color=CAT[k % 8], label=f"n = {n}")
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("time / instance (ms)")
    ax.set_ylabel("percentile")
    ax.set_title("Per-instance time distribution", loc="left")
    ax.legend()
    style(ax)
    fig.tight_layout()
    savefig(fig, out, "timing.png")
    doc.fig("timing.png",
            "Left: construction and annealing cost against $n$ (log-log). "
            "Middle: thread scaling. Right: distribution of per-instance wall "
            "time inside the parallel region.")
    doc.p(r"""
\textbf{Reading.} Up to $n \approx 200$ the annealing is essentially the whole
cost and the construction is free; past $n \approx 500$ the $O(n^2)$ savings list
takes over and dominates. If large instances ever become the target, the
construction is where to look --- \texttt{-{}-knn} truncates that list, and
Section~\ref{sec:construct} shows the annealing absorbs the loss of quality.
""")


def study_restarts(runs, out, doc):
    rs = sel(runs, "restarts")
    if not rs:
        return

    doc.sec("Restarts", "sec:restarts")
    doc.p(r"""
\textbf{What it controls.} \texttt{-{}-restarts R} builds $R$ initial solutions
per instance, anneals each one, and keeps the best. Restart 0 is always the
deterministic Clarke \& Wright; the others are diversified by
\texttt{-{}-cw-rand}: \texttt{perturb} multiplies every saving by
$1 + \alpha\,U(-1,1)$ with $\alpha =$ \texttt{-{}-cw-alpha}, which reshuffles the
order in which pairs are merged; \texttt{param} redraws $\lambda$ and $\mu$;
\texttt{both} does both; \texttt{off} makes all restarts identical apart from
the annealing seed.
""")
    doc.note(r"""
\textbf{$R$ multiplies the work by $R$.} Comparing $R$ restarts against one at
the same \texttt{-{}-sa-steps} answers nothing useful --- of course more compute
helps. The study therefore also holds $R \times \texttt{sa-steps}$ constant, which
asks the question that matters: given a fixed number of annealing steps, is it
better to spend them on one long run or several short ones?
""")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4))
    fx = sorted([r for r in rs if r.tag.startswith("fixed_")],
                key=lambda r: r.opts["restarts"])
    iso = sorted([r for r in rs if r.tag.startswith("iso_")],
                 key=lambda r: r.opts["restarts"])
    ax = axes[0]
    for grp, col, lab in ((fx, CAT[0], "fixed steps/restart"),
                          (iso, CAT[1], "equal total budget")):
        if not grp:
            continue
        ref = grp[0]
        st = [paired(ref, r) for r in grp]
        x = [r.opts["restarts"] for r in grp]
        ax.plot(x, [s["pct"] for s in st], "o-", color=col, label=lab)
        ax.fill_between(x, [s["pct_lo"] for s in st], [s["pct_hi"] for s in st],
                        color=col, alpha=0.15, linewidth=0)
    ax.axhline(0, color=INK3, lw=1.0)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("--restarts R")
    ax.set_ylabel("Δ mean cost vs R = 1 (%)")
    ax.set_title("Raw gain vs gain at equal cost", loc="left")
    ax.legend()
    style(ax)

    ax = axes[1]
    isor = sorted([r for r in rs if r.tag.startswith("isorand_")],
                  key=lambda r: r.opts["restarts"])
    if iso and isor:
        for k, (grp, lab) in enumerate(((iso, "--init cw"), (isor, "--init random"))):
            ref = grp[0]
            ax.plot([r.opts["restarts"] for r in grp],
                    [paired(ref, r)["pct"] for r in grp], "o-",
                    color=CAT[k], label=lab)
        ax.axhline(0, color=INK3, lw=1.0)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("--restarts R (equal total budget)")
        ax.set_ylabel("Δ vs R = 1 of the same init (%)")
        ax.set_title("Does a random start profit more?", loc="left")
        ax.legend()
        style(ax)

    ax = axes[2]
    div = [r for r in rs if r.tag.startswith(("cwrand_", "alpha_"))]
    ref = one(runs, "restarts", "cwrand_perturb")
    if div and ref:
        div.sort(key=lambda r: r.mean)
        lab = [r.tag.replace("cwrand_", "mode ").replace("alpha_", "α ") for r in div]
        bar_delta(ax, lab, [paired(ref, r) for r in div],
                  "Restart diversity (R = 8)", "Δ vs --cw-rand perturb (%)")
    fig.tight_layout()
    savefig(fig, out, "restarts.png")
    doc.fig("restarts.png",
            "Left: the same restart count judged two ways. Middle: restarts "
            "crossed with the initialisation. Right: how the diversity between "
            "restarts is generated.")

    if iso:
        doc.table([r"$R$", "steps/restart", "mean cost", *DHEAD, "wall (s)"],
                  [[r.opts["restarts"], "%s" % f"{r.opts['sa-steps']:,}",
                    "%.5f" % r.mean, *stat_cells(paired(iso[0], r)),
                    "%.2f" % r.log.get("wall_s", float("nan"))] for r in iso],
                  "Restarts at \\emph{equal total budget} "
                  "($R \\times \\texttt{sa-steps} = 320\\,000$), paired against "
                  "$R=1$.", align="rrrrrrr")
    if fx:
        doc.table([r"$R$", "mean cost", *DHEAD, "wall (s)"],
                  [[r.opts["restarts"], "%.5f" % r.mean,
                    *stat_cells(paired(fx[0], r)),
                    "%.2f" % r.log.get("wall_s", float("nan"))] for r in fx],
                  "Restarts at fixed steps per restart --- i.e.\\ at $R$ times the "
                  "compute. Shown for completeness; the table above is the "
                  "meaningful one.", align="rrrrrr")
    if div and ref:
        doc.table(["variant", "mean cost", *DHEAD],
                  [[tt(l), "%.5f" % r.mean, *stat_cells(paired(ref, r))]
                   for l, r in zip(lab, div)],
                  "How restart diversity is produced, at $R=8$, against the "
                  "default \\texttt{perturb} with $\\alpha=0.03$.", align="lrrrr")
    doc.p(r"""
\textbf{Reading.} At equal budget, restarts are flat: splitting $320\,000$ steps
into 2, 4, \dots, 32 independent anneals changes nothing significant. The
annealing is not getting trapped in a way that a fresh start would fix, so the
whole budget may as well go into one run. The random start is the exception ---
it degrades sharply as its share of the budget shrinks, which is the same result
as Section~\ref{sec:init} seen from another angle. On diversity, the default \texttt{perturb}
at $\alpha=0.03$ is the best of the options: too much perturbation
($\alpha \ge 0.3$) damages every restart, and \texttt{off} is worse still.
""")


def study_split(runs, out, doc):
    rs = sel(runs, "split")
    if not rs:
        return
    modes = ["off", "cw", "end", "both"]
    evs = [0, 100, 1000, 10000]
    grid = {(r.opts["split"], r.opts["split-every"]): r
            for r in rs if r.tag.startswith("m")}
    ref = grid.get(("off", 0))

    doc.sec("Optimal Split", "sec:split")
    doc.p(r"""
\textbf{What it controls.} Split (Vidal, 2016) takes the current solution,
concatenates its routes into one giant tour ignoring capacity, then re-cuts that
tour into routes optimally by dynamic programming in $O(n)$. Because the current
partition is one of the candidates, the result can never be worse.
\texttt{-{}-split} says where to apply it: \texttt{cw} right after the
construction, \texttt{end} after the annealing, \texttt{both}, or \texttt{off}.
\texttt{-{}-split-every N} additionally applies it every $N$ annealing steps ---
an independent code path (\texttt{cw.c:1440} versus \texttt{cw.c:1640/1658}), so
\texttt{off} combined with a period is a meaningful cell. \texttt{-{}-split-tour}
chooses how the giant tour is built: routes in their current order,
\texttt{sweep} by polar angle, or \texttt{both} keeping the better.
""")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4))
    if ref:
        mat = [[paired(ref, grid[(m, e)])["pct"] if (m, e) in grid else np.nan
                for e in evs] for m in modes]
        heat(axes[0], mat, [f"--split {m}" for m in modes],
             [("never" if e == 0 else f"every {e}") for e in evs],
             "Δ mean cost vs no Split at all (%)", "Δ cost (%)")

    ax = axes[1]
    tours = sorted([r for r in rs if r.tag.startswith("tour_")], key=lambda r: r.mean)
    if tours and ref:
        bar_delta(ax, [r.opts["split-tour"] for r in tours],
                  [paired(ref, r) for r in tours],
                  "--split-tour (with --split both --split-every 1000)",
                  "Δ vs no Split (%)")

    ax = axes[2]
    big = [r for r in rs if r.tag.startswith("n")]
    if big:
        bns = sorted({r.opts["n"] for r in big})
        w = 0.35
        for k, e in enumerate((0, 1000)):
            vals = []
            for n in bns:
                b = [r for r in big if r.opts["n"] == n and r.opts["split"] == "off"
                     and r.opts["split-every"] == e]
                t = [r for r in big if r.opts["n"] == n and r.opts["split"] == "both"
                     and r.opts["split-every"] == e]
                vals.append(paired(b[0], t[0])["pct"] if b and t else np.nan)
            ax.bar(np.arange(len(bns)) + (k - 0.5) * w, vals, w, color=CAT[k],
                   edgecolor="white", linewidth=0.8,
                   label=("no periodic Split" if e == 0 else f"+ every {e} steps"))
        ax.axhline(0, color=INK3, lw=1.0)
        ax.set_xticks(range(len(bns)), [f"n={n}" for n in bns])
        ax.set_ylabel("Δ of --split both vs off (%)")
        ax.set_title("Split at larger n", loc="left")
        ax.legend()
        style(ax)
    fig.tight_layout()
    savefig(fig, out, "split.png")
    doc.fig("split.png",
            "Left: the full mode $\\times$ period grid. Middle: giant-tour "
            "construction. Right: the same comparison at larger $n$, where there "
            "are more routes to repartition.")

    if ref:
        rows = []
        for m in modes:
            for e in evs:
                r = grid.get((m, e))
                if not r:
                    continue
                g = r.log.get("split_gain")   # cw prints no gain line when off
                rows.append([tt(m), (e or "never"), "%.5f" % r.mean,
                             *stat_cells(paired(ref, r)),
                             ("%.4f" % g) if g is not None else "--",
                             "%.2f" % r.log.get("wall_s", float("nan"))])
        doc.table([r"\texttt{-{}-split}", r"\texttt{-{}-split-every}", "mean cost",
                   *DHEAD, "Split gain", "wall (s)"], rows,
                  "Split mode against periodic Split, paired against no Split at "
                  "all. `Split gain' is the total improvement cw attributes to "
                  "Split internally --- note it is large where the paired delta "
                  "is not, i.e.\\ Split keeps recovering cost the annealing had "
                  "just spent.", align="llrrrrrr", long=True)
    doc.p(r"""
\textbf{Reading.} Split never hurts --- \texttt{-{}-split end} improves 53
instances and worsens exactly zero, which is the theoretical guarantee showing up
in the data --- but on this instance family it barely helps either: the best cell
is about $-0.02\,\%$, inside the noise. Its internal `gain' counter is large
while the net effect is nil, meaning it mostly undoes damage the annealing did
rather than finding new structure. Applying it every 100 steps is actively bad:
it costs double the wall time and drags the search back to a repartitioned
solution before the annealing can exploit its own moves.
""")


def study_pick(runs, out, doc):
    rs = sel(runs, "pick")
    if not rs:
        return
    Ts = [0, 2, 3, 4, 8, 16, 32]
    crits = ["lb", "rem", "remnorm", "raw"]
    grid = {(r.opts["pick"], r.opts["pick-crit"]): r
            for r in rs if re.fullmatch(r"T\d+_\w+", r.tag)}
    ref = grid.get((2, "lb"))                       # the stock default
    unif = one(runs, "pick", "T1_uniform")

    doc.sec("Vertex selection", "sec:pick")
    doc.p(r"""
\textbf{What it controls.} Before a move can be built, the annealing must pick
the vertex $u$ to disturb (\texttt{pick\_u}, \texttt{cw.c:952}). Uniform choice
wastes draws on customers that are already well placed, so the code biases the
choice by a \emph{regret} measure. With $\texttt{-{}-pick } T \ge 2$ it draws $T$
customers uniformly and keeps the one with the largest regret (a tournament);
$T=1$ is plain uniform; $T=0$ samples exactly proportional to
$\max(\text{regret},0) + \varepsilon$ through a Fenwick tree, at $O(\log n)$ per
draw and per update.
""")
    doc.p(r"""
\textbf{-{}-pick-crit} defines the regret, with
$\mathrm{inc}[u] = d(\mathrm{prv}[u],u) + d(u,\mathrm{nxt}[u])$ the cost of the
two edges $u$ currently carries:
\begin{itemize}\itemsep2pt
  \item \texttt{lb} (default) --- $\mathrm{inc}[u] - \mathrm{lb}[u]$, the gap to
        the two shortest edges available to $u$ in its kNN list.
  \item \texttt{rem} --- $\mathrm{inc}[u] - d(p,q)$, the removal gain: exactly
        what relocating $u$ could recover at best.
  \item \texttt{remnorm} --- the removal gain divided by $\mathrm{lb}[u]$,
        normalising by the local density.
  \item \texttt{raw} --- $\mathrm{inc}[u]$ itself.
\end{itemize}
\texttt{-{}-pick-eps} sets how peaked the $T=0$ sampler is.
""")
    doc.note(r"""
The code's own comment (\texttt{cw.c:911--916}) notes that \texttt{lb} ``can stay
large for an isolated customer even when nothing can improve it any further'',
whereas \texttt{rem} is zero as soon as $u$ sits well. That suggests the default
wastes draws on structurally irreducible vertices. The grid below was run
specifically to test that prediction.
""")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    if ref:
        mat = [[paired(ref, grid[(t, c)])["pct"] if (t, c) in grid else np.nan
                for c in crits] for t in Ts]
        heat(axes[0], mat, [("0 (Fenwick)" if t == 0 else f"T={t}") for t in Ts],
             crits, "Δ vs the default (T=2, lb) (%)", "Δ cost (%)")

    ax = axes[1]
    eps = sorted([r for r in rs if r.tag.startswith("eps_")],
                 key=lambda r: r.opts["pick-eps"])
    if eps:
        e0 = eps[0]
        st = [paired(e0, r) for r in eps]
        x = [r.opts["pick-eps"] for r in eps]
        ax.plot(x, [s["pct"] for s in st], "o-", color=CAT[2])
        ax.fill_between(x, [s["pct_lo"] for s in st], [s["pct_hi"] for s in st],
                        color=CAT[2], alpha=0.15, linewidth=0)
        ax.set_xscale("log")
        ax.axhline(0, color=INK3, lw=1.0)
        ax.set_xlabel("--pick-eps (with --pick 0 --pick-crit rem)")
        ax.set_ylabel(f"Δ vs eps = {e0.opts['pick-eps']} (%)")
        ax.set_title("Peakedness of the proportional sampler", loc="left")
        style(ax)

    ax = axes[2]
    inter = [r for r in rs if r.tag.startswith("inter_")]
    irows = []
    if inter:
        Ks = sorted({r.opts["sa-knn"] for r in inter})
        Ti = sorted({r.opts["pick"] for r in inter})
        w = 0.8 / len(Ti)
        for k, t in enumerate(Ti):
            vals = []
            for K in Ks:
                g = [r for r in inter if r.opts["pick"] == t and r.opts["sa-knn"] == K]
                b = [r for r in inter if r.opts["pick"] == 1 and r.opts["sa-knn"] == K]
                st = paired(b[0], g[0]) if g and b else {}
                vals.append(st.get("pct", np.nan))
                if g and b:
                    irows.append([K, t, "%.5f" % g[0].mean, *stat_cells(st)])
            ax.bar(np.arange(len(Ks)) + (k - (len(Ti) - 1) / 2) * w, vals, w,
                   color=CAT[k], edgecolor="white", linewidth=0.8, label=f"--pick {t}")
        ax.axhline(0, color=INK3, lw=1.0)
        ax.set_xticks(range(len(Ks)), [("K=0 (uniform)" if K == 0 else f"K={K}") for K in Ks])
        ax.set_ylabel("Δ vs --pick 1 at the same K (%)")
        ax.set_title("At K = 0 the selection rule is ignored", loc="left")
        ax.legend()
        style(ax)
    fig.tight_layout()
    savefig(fig, out, "pick.png")
    doc.fig("pick.png",
            "Left: tournament size against regret criterion. Middle: peakedness "
            "of the proportional sampler. Right: \\texttt{-{}-pick} crossed with "
            "\\texttt{-{}-sa-knn}; the flat $K=0$ group is the silent fallback.")

    if ref:
        rows = []
        if unif:
            rows.append(["1 (uniform)", "--", "%.5f" % unif.mean,
                         *stat_cells(paired(ref, unif))])
        for t in Ts:
            for c in crits:
                r = grid.get((t, c))
                if r:
                    rows.append([("0 (Fenwick)" if t == 0 else t), tt(c),
                                 "%.5f" % r.mean, *stat_cells(paired(ref, r))])
        doc.table([r"\texttt{-{}-pick}", r"\texttt{-{}-pick-crit}", "mean cost", *DHEAD],
                  rows, "Tournament size against regret criterion, paired against "
                        "the default $T=2$ with \\texttt{lb}.",
                  align="llrrrr", long=True)
    if irows:
        doc.table([r"\texttt{-{}-sa-knn}", r"\texttt{-{}-pick}", "mean cost", *DHEAD],
                  irows,
                  "\\texttt{-{}-pick} crossed with \\texttt{-{}-sa-knn}, paired "
                  "against \\texttt{-{}-pick 1} at the same $K$. The $K=0$ rows are "
                  "exactly zero because \\texttt{cw.c:1400} forces uniform "
                  "selection there --- while the run header still prints the "
                  "requested rule.", align="rrrrrr")
    doc.p(r"""
\textbf{Reading.} The prediction was wrong. At $n=100$ with $10^5$ steps the
entire grid --- every tournament size, every criterion, and the Fenwick sampler
--- sits inside $\pm 0.08\,\%$ of the default. The regret machinery is
measurable in the code but not in the result: whatever it saves in draw quality
it appears to give back in draw cost. It is only at $T \ge 16$ with \texttt{lb}
or \texttt{raw} that the choice starts to hurt, from over-concentrating on a few
vertices. The one solid finding here is the silent fallback at $K=0$: the header
reports a rule the code is not using.
""")


def study_temp(runs, out, doc):
    rs = sel(runs, "temp")
    if not rs:
        return
    As = sorted({r.opts["t-accept"] for r in rs})
    Ds = sorted({r.opts["t-decades"] for r in rs})
    grid = {(r.opts["t-accept"], r.opts["t-decades"]): r for r in rs}
    ref = grid.get((0.001, 2))

    doc.sec("Annealing schedule", "sec:temp")
    doc.p(r"""
\textbf{What it controls.} The initial temperature $T_0$ is not given directly:
\texttt{calibrate\_T0} samples worsening moves on the actual instance and solves
for the $T_0$ whose acceptance probability matches \texttt{-{}-t-accept} (default
$0.001$, i.e.\ $0.1\,\%$ of worsening moves accepted at the start). The
temperature then decays geometrically to $T_0 \cdot 10^{-D}$ over the whole run,
with $D =$ \texttt{-{}-t-decades} (default 2). Together they fix both ends of the
schedule; \texttt{-{}-t0} and \texttt{-{}-tend} override the calibration.
""")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    if ref:
        mat = [[paired(ref, grid[(a, d)])["pct"] if (a, d) in grid else np.nan
                for d in Ds] for a in As]
        heat(axes[0], mat, ["%g" % a for a in As], [f"{d} dec" for d in Ds],
             "Δ vs the default (accept 1e-3, 2 decades) (%)", "Δ cost (%)")
        axes[0].set_ylabel("--t-accept")
    ax = axes[1]
    for k, d in enumerate(Ds):
        pts = [(grid[(a, d)].log.get("accept_pct", np.nan), grid[(a, d)].mean)
               for a in As if (a, d) in grid]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                    color=CAT[k], label=f"{d} decades")
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("realised acceptance rate (%)")
    ax.set_ylabel("mean cost")
    ax.set_title("Cost against the acceptance actually achieved", loc="left")
    ax.legend()
    style(ax)
    fig.tight_layout()
    savefig(fig, out, "temp.png")
    doc.fig("temp.png",
            "Left: the full \\texttt{-{}-t-accept} $\\times$ "
            "\\texttt{-{}-t-decades} grid. Right: the same runs plotted against "
            "the acceptance rate they actually realised.")

    if ref:
        doc.table([r"\texttt{-{}-t-accept}", r"\texttt{-{}-t-decades}", "mean cost",
                   r"accept.\ (\%)", *DHEAD],
                  [["%g" % a, d, "%.5f" % grid[(a, d)].mean,
                    "%.2f" % grid[(a, d)].log.get("accept_pct", float("nan")),
                    *stat_cells(paired(ref, grid[(a, d)]))]
                   for a in As for d in Ds if (a, d) in grid],
                  "Annealing schedule, paired against the default "
                  "($\\texttt{t-accept}=10^{-3}$, 2 decades).",
                  align="rrrrrrr", long=True)
    doc.p(r"""
\textbf{Reading.} This is the largest effect in the whole sweep. Narrowing the
temperature range to a single decade is worth about $-0.37\,\%$ --- more than
every operator, neighbourhood and selection setting combined. The interpretation
is that at $10^5$ steps the default schedule spends too long at temperatures so
low that nothing is accepted: the run is effectively over before the step budget
is. Fewer decades keeps the search alive for more of the run. This also implies
the right number of decades depends on the budget, so it should be re-checked
when \texttt{-{}-sa-steps} changes substantially.
""")


def study_construct(runs, out, doc):
    rs = sel(runs, "construct")
    if not rs:
        return

    doc.sec("Construction quality", "sec:construct")
    doc.p(r"""
\textbf{What it controls.} Three separate things about the Clarke \& Wright
stage. The savings formula
$s(i,j) = d_{0i} + d_{0j} - \lambda\,d_{ij} + \mu\,|d_{0i}-d_{0j}|$ has two shape
parameters: \texttt{-{}-lambda} sets how much weight the direct link between two
customers carries against their two depot legs (the classical generalisation of
Clarke \& Wright, $\lambda > 1$ favouring longer, less radial routes), and
\texttt{-{}-mu} adds an asymmetry term penalising the merge of two customers at
very different distances from the depot. \texttt{-{}-knn K} truncates the savings
list to each customer's $K$ nearest neighbours instead of all $O(n^2)$ pairs
(\texttt{-{}-exact} forces the full list; the default is exact for
$n \le 1500$). \texttt{-{}-2opt} runs an intra-route 2-opt pass after the
construction.
""")
    doc.p(r"""
\textbf{The question.} Every one of these improves or degrades the \emph{starting}
solution. The interesting question is whether that survives the annealing, so each
grid is run twice: with \texttt{-{}-no-sa} and with the full budget.
""")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for i, (pref, ttl) in enumerate((("nosa_l", "Clarke & Wright alone (--no-sa)"),
                                     ("sa_l", "after 10⁵ SA steps"))):
        grp = [r for r in rs if r.tag.startswith(pref)]
        if not grp:
            continue
        Ls = sorted({r.opts["lambda"] for r in grp})
        Ms = sorted({r.opts["mu"] for r in grp})
        g = {(r.opts["lambda"], r.opts["mu"]): r for r in grp}
        ref = g.get((1.0, 0.0))
        if not ref:
            continue
        mat = [[paired(ref, g[(l, m)])["pct"] if (l, m) in g else np.nan for m in Ms]
               for l in Ls]
        heat(axes[i], mat, [f"λ={l:g}" for l in Ls], [f"μ={m:g}" for m in Ms],
             f"{ttl}\nΔ vs λ=1, μ=0 (%)", "Δ cost (%)")
        doc.table([r"$\lambda$", r"$\mu$", "mean cost", *DHEAD],
                  [["%g" % l, "%g" % m, "%.5f" % g[(l, m)].mean,
                    *stat_cells(paired(ref, g[(l, m)]))]
                   for l in Ls for m in Ms if (l, m) in g],
                  "Savings parameters, %s, paired against $\\lambda=1$, $\\mu=0$."
                  % ("construction only" if i == 0 else "after annealing"),
                  align="rrrrrr", long=True)

    ax = axes[2]
    rows, labs, stats = [], [], []
    for pref, tag in (("nosa_n200_", "--no-sa"), ("sa_n200_", "with SA")):
        grp = [r for r in rs if r.tag.startswith(pref)]
        ref = next((r for r in grp if r.tag.endswith("_exact")), None)
        if not ref:
            continue
        for r in sorted(grp, key=lambda r: r.tag):
            k = r.tag[len(pref):]
            labs.append(f"{tag} · {k}")
            s = paired(ref, r)
            stats.append(s)
            rows.append([tt(tag), tt(k), "%.5f" % r.mean, *stat_cells(s),
                         "%.2f" % r.log.get("wall_s", float("nan"))])
    if stats:
        bar_delta(ax, labs, stats, "Savings-list truncation, n = 200",
                  "Δ vs --exact (%)")
    fig.tight_layout()
    savefig(fig, out, "construct.png")
    doc.fig("construct.png",
            "Left and middle: the same $\\lambda \\times \\mu$ grid before and "
            "after annealing --- note the colour scales differ by an order of "
            "magnitude. Right: truncating the savings list, with and without "
            "annealing.")

    if rows:
        doc.table(["stage", "variant", "mean cost", *DHEAD, "wall (s)"], rows,
                  "Savings-list truncation at $n=200$, paired against "
                  "\\texttt{-{}-exact}. \\texttt{knn0} is the automatic setting, "
                  "which is exact at this size.", align="llrrrrr")
    rows = []
    for pref, lab in (("nosa", "construction only"), ("sa", "after annealing")):
        off = one(runs, "construct", f"{pref}_2opt_off")
        on = one(runs, "construct", f"{pref}_2opt_on")
        if off and on:
            rows.append([lab, "%.5f" % off.mean, "%.5f" % on.mean,
                         *stat_cells(paired(off, on))])
    doc.table(["stage", r"without \texttt{-{}-2opt}", r"with \texttt{-{}-2opt}", *DHEAD],
              rows, "Intra-route 2-opt after the construction.", align="lrrrrr")
    doc.p(r"""
\textbf{Reading.} Two opposite lessons. The savings \emph{shape} matters and
survives: $\lambda = 1.2$--$1.4$ with $\mu \approx 0.2$--$0.5$ is worth about
$-0.29\,\%$ after annealing, second only to the schedule. The savings
\emph{list}, by contrast, does not survive at all: truncating it to the 5 nearest
neighbours costs $+7.7\,\%$ on the raw construction and essentially nothing once
the annealing has run. That is a useful lever for large $n$, where
Section~\ref{sec:timing} showed the $O(n^2)$ list is the dominant cost --- the annealing repairs
what the truncation breaks. \texttt{-{}-2opt} is likewise absorbed: it improves
the construction and changes nothing afterwards.
""")


def study_tuned(runs, out, doc):
    rs = sel(runs, "tuned")
    if not rs:
        return
    ns = sorted({r.opts["n"] for r in rs})
    variants = [("oropt", "or-opt enabled"), ("critrem", "pick-crit rem"),
                ("split", "Split both + every 1000"), ("all", "the three combined"),
                ("all_r8", "the three, over 8 restarts")]

    doc.sec("Do the knobs combine?", "sec:tuned")
    doc.p(r"""
\textbf{What is tested.} Tuning one knob at a time says nothing about tuning
several at once. These runs take three plausible changes, apply them
individually and together at equal SA budget, and compare against the stock
defaults. \texttt{all\_r8} additionally splits the same budget over 8 restarts.
""")

    fig, ax = plt.subplots(figsize=(10, 4.2))
    w = 0.8 / len(variants)
    rows = []
    for k, (v, lab) in enumerate(variants):
        vals, los, his = [], [], []
        for n in ns:
            b = one(runs, "tuned", f"n{n}_default")
            r = one(runs, "tuned", f"n{n}_{v}")
            s = paired(b, r) if b and r else {}
            vals.append(s.get("pct", np.nan))
            los.append(s.get("pct_lo", np.nan)); his.append(s.get("pct_hi", np.nan))
            if s:
                rows.append([n, esc(lab), "%.5f" % r.mean, *stat_cells(s),
                             "%.2f" % r.log.get("wall_s", float("nan"))])
        x = np.arange(len(ns)) + (k - (len(variants) - 1) / 2) * w
        ax.bar(x, vals, w, color=CAT[k], edgecolor="white", linewidth=0.8, label=v)
        ax.errorbar(x, vals, yerr=[np.array(vals) - np.array(los),
                                   np.array(his) - np.array(vals)],
                    fmt="none", ecolor=INK2, elinewidth=0.9, capsize=2)
    ax.axhline(0, color=INK3, lw=1.0)
    ax.set_xticks(range(len(ns)), [f"n = {n}" for n in ns])
    ax.set_ylabel("Δ mean cost vs defaults (%)")
    ax.set_title("Combined settings against the stock defaults", loc="left")
    ax.legend(ncol=len(variants))
    style(ax)
    fig.tight_layout()
    savefig(fig, out, "tuned.png")
    doc.fig("tuned.png", "Candidate combinations against the stock defaults at "
                         "equal SA budget, with 95\\,\\% CIs.")
    doc.table([r"$n$", "variant", "mean cost", *DHEAD, "wall (s)"], rows,
              "Combinations against the stock defaults.", align="rlrrrrr", long=True)
    doc.p(r"""
\textbf{Reading.} The combination is worse than its parts: \texttt{all} loses to
the defaults at every $n$, even though one of its three ingredients helps on its
own. Gains found one knob at a time do not add up, which is exactly why the
recommendation in Section~\ref{sec:best} is confirmed by re-running it rather
than assembled on paper.
""")


# ------------------------------------------------------- knobs & overview
# Which cw options constitute each knob. A knob enters the recommended command
# only as a whole: if its winner differs from the default in two options, both
# are taken.
KNOB_KEYS = {
    "operators": ["ops"],
    "or-opt length": ["or-max"],
    "neighbourhood": ["sa-knn"],
    "vertex choice": ["pick", "pick-crit", "pick-eps"],
    "schedule": ["t-accept", "t-decades"],
    "Split": ["split", "split-every", "split-tour"],
    "restarts (equal budget)": ["restarts"],
    "savings shape": ["lambda", "mu"],
    "initialisation": ["init"],
}


def knobs(runs):
    """One record per knob: (label, ref run, best run, paired stat)."""
    out = []

    def add(label, ref, cands):
        cands = [c for c in cands if c and ref and c.instkey == ref.instkey]
        if not ref or not cands:
            return
        best = min(cands, key=lambda r: r.mean)
        s = paired(ref, best)
        if s:
            out.append((label, ref, best, s))

    add("operators", one(runs, "ops", "sub_1110"),
        [r for r in sel(runs, "ops") if r.tag.startswith(("sub_", "w_"))])
    add("or-opt length", one(runs, "ops", "ormax_3"),
        [r for r in sel(runs, "ops") if r.tag.startswith("ormax_")])
    add("neighbourhood", one(runs, "knn", "n100_K20"), sel(runs, "knn", n=100))
    add("vertex choice", one(runs, "pick", "T2_lb"),
        [r for r in sel(runs, "pick") if re.fullmatch(r"T\d+_\w+", r.tag)])
    add("schedule", next((r for r in sel(runs, "temp")
                          if r.opts["t-accept"] == 0.001 and r.opts["t-decades"] == 2),
                         None), sel(runs, "temp"))
    add("Split", next((r for r in sel(runs, "split")
                       if r.opts["split"] == "off" and r.opts["split-every"] == 0
                       and r.opts["n"] == 100), None),
        [r for r in sel(runs, "split") if r.tag.startswith("m")])
    add("restarts (equal budget)", one(runs, "restarts", "iso_R1"),
        [r for r in sel(runs, "restarts") if r.tag.startswith("iso_")])
    add("savings shape", one(runs, "construct", "sa_l1.0_m0"),
        [r for r in sel(runs, "construct") if r.tag.startswith("sa_l")])
    add("initialisation", next((r for r in sel(runs, "init")
                                if r.opts["n"] == 100 and r.opts["init"] == "cw"
                                and r.opts["sa-steps"] == 100000), None),
        [r for r in sel(runs, "init")
         if r.opts["n"] == 100 and r.opts["sa-steps"] == 100000])
    return out


def overview(runs, out, doc, ks):
    if not ks:
        return
    items = sorted(ks, key=lambda t: t[3]["pct"])
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(items) + 1.8))
    bar_delta(ax, [f"{l}\n→ {b.tag}" for l, _, b, _ in items],
              [s for _, _, _, s in items],
              "Value of each knob at its best setting (all else default)",
              "best achievable Δ mean cost vs default (%)")
    fig.tight_layout()
    savefig(fig, out, "overview.png")

    doc.sec("Overview: what each knob is worth")
    doc.p(r"""
Each row below varies one knob with everything else left at its default, and
reports the best setting the sweep found for it. Negative is better. The sections
that follow explain what each knob does and give the full grids.
""")
    doc.note(r"""
\textbf{Read the confidence intervals, not the point estimates.} A setting is
listed \emph{because} it came out lowest, so its $\Delta$ is optimistically
biased --- the more values a row sweeps, the stronger the bias. A row whose CI
straddles zero is a knob that did nothing here. This is why
Section~\ref{sec:best} re-runs the winners on a seed the sweep never saw.
""")
    doc.fig("overview.png",
            "Best achievable improvement per knob, with 95\\,\\% CIs. Bars that "
            "cross zero are knobs with no measurable effect on this instance "
            "family at this budget.")
    doc.table(["knob", "best setting found", *DHEAD],
              [[esc(l), tt(b.tag), *stat_cells(s)] for l, _, b, s in items],
              "Each knob at its best setting, paired against its own default.",
              align="llrrr")


# ------------------------------------------------------ the winning config
def flags_for(run, keys):
    """The cw flags by which `run` differs from the defaults, over `keys`."""
    out = []
    for k in keys:
        v = run.opts.get(k)
        if v == DEFAULTS.get(k):
            continue
        if isinstance(v, bool):
            if v:
                out.append("--" + k)
        else:
            out += ["--" + k, "%g" % v if isinstance(v, float) else str(v)]
    return out


def best_config(runs, ks):
    """Flags from every knob whose winner beat its own default significantly."""
    flags, kept, dropped = [], [], []
    for label, ref, best, s in ks:
        keys = KNOB_KEYS.get(label, [])
        if s["sig"] and s["pct"] < 0 and keys:
            f = flags_for(best, keys)
            if f:
                flags += f
                kept.append((label, best, s, f))
                continue
        dropped.append((label, best, s))
    # --or-max is inert unless or-opt carries a positive weight
    if "--or-max" in flags:
        ops = flags[flags.index("--ops") + 1] if "--ops" in flags else DEFAULTS["ops"]
        if float(ops.split(",")[3]) <= 0:
            i = flags.index("--or-max")
            del flags[i:i + 2]
            kept = [k for k in kept if k[0] != "or-opt length"]
    return flags, kept, dropped


# Every option the sweep touched, grouped as in cw's own --help. The final
# command is printed in full, defaults included, so that it is self-documenting:
# nothing about the run depends on what cw's built-in defaults happen to be.
CMD_GROUPS = (
    ("initial solution", ["init"]),
    ("construction", ["knn", "lambda", "mu"]),
    ("annealing budget and moves", ["sa-steps", "ops", "or-max"]),
    ("temperature schedule", ["t-accept", "t-decades"]),
    ("move sampling", ["sa-knn", "pick", "pick-crit", "pick-eps"]),
    ("restarts", ["restarts", "cw-rand", "cw-alpha"]),
    ("Split", ["split", "split-every", "split-tour"]),
)


def resolve(flags, steps):
    """Default option values, overridden by `flags`."""
    vals = dict(DEFAULTS)
    vals["sa-steps"] = steps
    i = 0
    while i < len(flags):
        k = flags[i].lstrip("-")
        if i + 1 < len(flags) and not flags[i + 1].startswith("-"):
            v = flags[i + 1]
            vals[k] = int(v) if k in NUMERIC else float(v) if k in FLOAT else v
            i += 2
        else:                                   # a bare boolean flag
            vals[k] = True
            i += 1
    return vals


def fmtv(v):
    return "%g" % v if isinstance(v, float) else str(v)


def full_command(source, vals):
    """The complete command line, one group per continuation line."""
    lines = ["./cw " + source + " \\"]
    for _, keys in CMD_GROUPS:
        lines.append("     " + " ".join("--%s %s" % (k, fmtv(vals[k])) for k in keys)
                     + " \\")
    lines[-1] = lines[-1].rstrip(" \\")
    return "\n".join(lines)


def run_cw(binary, args, base):
    """Run cw once, write <base>.{log,csv,meta}, return the Run."""
    cmd = [binary] + args + ["--csv", base + ".csv"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    with open(base + ".log", "w", encoding="utf-8") as f:
        f.write(p.stdout + (("\n" + p.stderr) if p.stderr else ""))
    with open(base + ".meta", "w", encoding="utf-8") as f:
        f.write("study=best\ntag=%s\ncmd=%s\nexit=%d\nwall_s=nan\n"
                % (os.path.basename(base), " ".join(cmd), p.returncode))
    return read_run(base, "best", os.path.basename(base))


def verify(runs, ks, results_dir, doc, binary, do_run=True):
    flags, kept, dropped = best_config(runs, ks)
    tmpl = one(runs, "tuned", "n100_default")
    if tmpl is None:
        return
    m, seed, steps = tmpl.opts["m"], tmpl.opts["seed"], tmpl.opts["sa-steps"]

    doc.sec("The configuration to use", "sec:best")
    if not kept:
        doc.p("No knob beat its own default significantly. The stock defaults "
              "are the recommendation.")
        return

    doc.p(r"""
A knob enters the command below only if its best setting beat \emph{its own}
default significantly --- both the 95\,\% CI excluding zero and the sign test
rejecting at 5\,\%. Everything else stays at the default, deliberately: a knob
whose interval straddles zero has not earned a change.
""")
    doc.table(["knob", "setting", *DHEAD, "flags"],
              [[esc(l), tt(b.tag), *stat_cells(s), tt(" ".join(f))]
               for l, b, s, f in kept],
              "Knobs that earned their place, with the improvement each showed "
              "on its own.", align="llrrrl")
    if dropped:
        doc.table(["knob", "best setting tried", *DHEAD],
                  [[esc(l), tt(b.tag), *stat_cells(s)] for l, b, s in dropped],
                  "Knobs left at their default: no significant improvement.",
                  align="llrrr")

    vals = resolve(flags, steps)
    changed = [k for _, keys in CMD_GROUPS for k in keys
               if k != "sa-steps" and vals[k] != DEFAULTS[k]]

    doc.sub("The command")
    doc.p(r"""
Written out in full, defaults included, so the run does not depend on what
\texttt{cw}'s built-in defaults happen to be:
""")
    doc.verb(full_command("--bundle data/cvrp_100.cvrpb", vals))
    doc.p("On generated instances instead of a bundle, swap the first line for "
          "\\texttt{-{}-random -n 100 -m 1000 -{}-seed 42}:")
    doc.verb(full_command("--random -n 100 -m 1000 --seed 42", vals))
    def andlist(items):
        items = list(items)
        return items[0] if len(items) == 1 else \
            ", ".join(items[:-1]) + " and " + items[-1]

    doc.p("Only %s differ from the stock defaults (%s). Every other value above "
          "is what \\texttt{cw} would have used anyway, written out so the run "
          "is reproducible against a future version whose defaults have moved."
          % (andlist(tt("--" + k) for k in changed),
             andlist("%s was %s" % (tt("--" + k), tt(fmtv(DEFAULTS[k])))
                     for k in changed)))

    doc.sub("What the three changed options do")
    doc.p(r"""
\textbf{\texttt{-{}-t-decades 1}} (default 2) --- the width of the temperature
schedule. The annealing does not take $T_0$ as an argument: \texttt{calibrate\_T0}
samples worsening moves on the instance itself and solves for the $T_0$ at which
a fraction \texttt{-{}-t-accept} of them would be accepted ($0.1\,\%$ by
default). From there the temperature decays geometrically to
$T_{\text{end}} = T_0 \cdot 10^{-D}$, one factor
$\alpha = 10^{-D/(\text{steps}-1)}$ per step, with $D =$
\texttt{-{}-t-decades}. So $D$ is \emph{how many orders of magnitude the
temperature falls over the run}: $D=2$ ends a hundred times colder than it
started, $D=1$ only ten times colder.
""")
    doc.p(r"""
Lowering $D$ keeps the search warm for longer. The realised acceptance rate over
the whole run goes $0.9\,\%$ at $D=4$, $1.1\,\%$ at $D=3$, $1.5\,\%$ at the
default $D=2$, $2.6\,\%$ at $D=1$ --- and the cost falls monotonically along that
sequence. The reading is that the default schedule freezes too early: it reaches
temperatures at which essentially nothing is accepted well before the step budget
runs out, and those steps are wasted. Widening the schedule further ($D=3$, $D=4$)
makes it strictly worse, by $+0.27\,\%$ and $+0.41\,\%$.
""")
    doc.p(r"""
\textbf{\texttt{-{}-lambda 1.4} and \texttt{-{}-mu 0.2}} (defaults 1 and 0) ---
the shape of the Clarke \& Wright savings criterion,
\[
  s(i,j) \;=\; d_{0i} + d_{0j} \;-\; \lambda\, d_{ij}
                \;+\; \mu\,\lvert d_{0i} - d_{0j}\rvert .
\]
At $\lambda=1$, $\mu=0$ this is the original 1964 criterion: the distance saved
by serving $i$ and $j$ on one route rather than two separate out-and-back trips.
The construction merges route endpoints in decreasing $s$ while capacity allows,
so $s$ decides nothing but the \emph{order} in which pairs are considered ---
which is enough to determine the whole solution.
""")
    doc.p(r"""
$\lambda$ reweights the direct link $d_{ij}$ against the two depot legs. Raising
it above 1 makes the criterion more sceptical of long links, so only genuinely
close pairs are merged early; the routes that come out are more compact and less
radial. $\mu$ adds a bonus for merging two customers that sit at \emph{different}
distances from the depot, which favours routes that run outward and sweep back
rather than joining two customers on the same ring.
""")
    doc.p(r"""
The two interact, and neither is any good alone: at $\lambda=1.4$, $\mu=0$ the
construction is $+0.34\,\%$ \emph{worse} than the default, and $\mu=1$ at the
same $\lambda$ is $+1.30\,\%$ worse. It is the combination $\lambda \approx 1.4$
with a small positive $\mu$ that wins, and that is a fact about this instance
family --- uniform points on a square, where the depot is not central by
construction --- not a universal constant. On clustered instances the optimum
will move.
""")
    doc.note(r"""
\textbf{Both of these are conditional on the budget.} \texttt{-{}-t-decades} was
tuned at $\texttt{-{}-sa-steps} = %s$ and controls how fast the temperature
reaches the point where nothing is accepted; that trade-off is different at
$10^4$ or $10^6$ steps. Re-run the \texttt{temp} study if you change the budget
substantially. The savings parameters are the more portable of the two, but they
were still tuned on one instance family.
""" % f"{steps:,}")

    if not do_run:
        return

    # Confirmation. Two seeds on purpose: the sweep seed is in-sample and
    # carries the selection bias, the second one has never been seen.
    bdir = os.path.join(results_dir, "best")
    os.makedirs(bdir, exist_ok=True)
    fresh = 20260801
    rows = []
    print("  verifying the recommended configuration...")
    for label, sd in (("sweep seed (in-sample)", seed), ("fresh seed", fresh)):
        for n in (50, 100, 200):
            src = ["--random", "-n", str(n), "-m", str(m), "--seed", str(sd),
                   "--sa-steps", str(steps)]
            b = run_cw(binary, src, os.path.join(bdir, f"s{sd}_n{n}_default"))
            t = run_cw(binary, src + flags, os.path.join(bdir, f"s{sd}_n{n}_tuned"))
            if not (b.ok and t.ok):
                continue
            s = paired(b, t)
            rows.append([esc(label), n, "%.5f" % b.mean, "%.5f" % t.mean,
                         *stat_cells(s)])
    if rows:
        doc.sub("Confirmation")
        doc.p(r"""
The combination is re-run from scratch, because Section~\ref{sec:tuned} showed that knobs tuned
separately do not add up. It is checked on the sweep's own instances and on a
seed the sweep never saw --- the difference between the two is the selection bias
made visible.
""")
        doc.table(["instances", r"$n$", "default", "recommended", *DHEAD], rows,
                  "The recommended configuration against the stock defaults, at "
                  "equal SA budget. The fresh-seed rows are the honest estimate.",
                  align="lrrrrrr")


# ------------------------------------------------------------ the document
PREAMBLE = r"""% ---------------------------------------------------------------------------
% Generated by sweep/analyze_sweep.py -- do not edit by hand.
% Build:  cd sweep && pdflatex report.tex
% ---------------------------------------------------------------------------
\documentclass[10pt,a4paper,landscape]{article}
\usepackage[margin=2cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{graphicx}
\usepackage{float}
\usepackage{amsmath,amssymb}
\usepackage[dvipsnames]{xcolor}
\usepackage[colorlinks=true,linkcolor=RoyalBlue,urlcolor=RoyalBlue]{hyperref}
\setlength{\parskip}{0.5em}
\setlength{\parindent}{0pt}
\title{\texttt{cw} --- parameter study}
\author{Generated by \texttt{sweep/analyze\_sweep.py}}
\date{@DATE@}

\begin{document}
\maketitle

\section*{Setup}

@NRUNS@ runs of \texttt{./cw}, @M@ instances each, generated with the
Kool/NeuOpt law (coordinates $U[0,1]^2$, demands $U\{1,\dots,9\}$, capacity from
\texttt{default\_capacity}). Unless a study says otherwise, $n=100$ and
\texttt{-{}-sa-steps}~$=$~@STEPS@.

\textbf{Every comparison is paired.} Instance $k$ is generated from
\texttt{seed}~$+~k$ (\texttt{cw.c:2064}), so all runs of a study see
byte-identical instances. Rather than comparing two means with their independent
standard deviations, we compare $\mathrm{cost}_i(A)$ with $\mathrm{cost}_i(B)$ on
the same instance $i$ and report the mean of the @M@ differences with a
95\,\% confidence interval, plus a distribution-free sign test. This matters: the
between-instance spread ($\sigma \approx 1.9$ at $n=100$) is more than an order of
magnitude larger than the effects being measured.

In every table, $\Delta$ is that paired mean as a percentage of the baseline;
\textbf{bold} means the CI excludes zero \emph{and} the sign test rejects at
5\,\%; \texttt{win/loss} counts instances strictly improved and strictly
worsened. \textbf{Negative is better} --- these are costs.

\tableofcontents
\clearpage
"""


def main():
    ap = argparse.ArgumentParser(description="analyse the cw parameter sweep")
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap.add_argument("--results", default=os.path.join(here, "results"))
    ap.add_argument("--out", default=here)
    ap.add_argument("--bin", default=os.path.join(root, "cw"))
    ap.add_argument("--no-verify", action="store_true",
                    help="do not re-run the recommended configuration")
    ap.add_argument("--no-pdf", action="store_true", help="emit the .tex only")
    a = ap.parse_args()

    if not os.path.isdir(a.results):
        sys.exit(f"{a.results} not found — run sweep/run_sweep.sh first")
    runs = [r for r in load(a.results) if r.study != "best"]
    if not runs:
        sys.exit("no successful run found")
    per = defaultdict(int)
    for r in runs:
        per[r.study] += 1
    print(f"loaded {len(runs)} runs: " + ", ".join(f"{k}={v}" for k, v in sorted(per.items())))

    figdir = os.path.join(a.out, "figures")
    os.makedirs(figdir, exist_ok=True)
    doc = Doc()

    ks = knobs(runs)
    try:
        overview(runs, figdir, doc, ks)
    except Exception as e:
        print(f"  !! overview: {e}", file=sys.stderr)
    for fn in (study_init, study_ops, study_knn, study_timing, study_restarts,
               study_split, study_pick, study_temp, study_construct, study_tuned):
        try:
            fn(runs, figdir, doc)
        except Exception as e:                      # one broken study must not
            print(f"  !! {fn.__name__}: {e}", file=sys.stderr)   # sink the rest
    try:
        verify(runs, ks, a.results, doc, a.bin, do_run=not a.no_verify)
    except Exception as e:
        print(f"  !! verify: {e}", file=sys.stderr)

    doc.sec("Caveats")
    doc.p(r"""
\begin{itemize}\itemsep3pt
  \item \textbf{One instance family.} Uniform coordinates and uniform demands.
        Nothing here is evidence about clustered or real-world instances; run
        \texttt{tools/fetch\_neuopt.py} and point the sweep at
        \texttt{-{}-bundle} for that.
  \item \textbf{One budget.} Most grids are at $10^5$ annealing steps. The
        schedule result in particular is budget-dependent by construction.
  \item \textbf{Selection bias.} The overview picks each knob's best setting
        \emph{because} it scored lowest. The fresh-seed rows in
        Section~\ref{sec:best} are the only bias-free numbers in this document.
  \item \textbf{\texttt{-{}-sa-knn} is clamped} to $n-1$
        (\texttt{cw.c:1399}), so at $n=20$ the settings $K \ge 20$ are the same
        run; and at $K=0$ the vertex-selection rule is silently forced to
        uniform (\texttt{cw.c:1400}) while the header still prints the requested
        \texttt{-{}-pick}. That is a reporting bug worth fixing in \texttt{cw}.
\end{itemize}
""")

    import datetime
    tmpl = one(runs, "tuned", "n100_default") or runs[0]
    head = PREAMBLE                    # token substitution, not %-formatting:
    for tok, val in (                  # the preamble is full of literal LaTeX %
            ("@DATE@", datetime.date.today().isoformat()),
            ("@NRUNS@", str(len(runs))),
            ("@M@", f"{tmpl.opts['m']:,}"),
            ("@STEPS@", f"{tmpl.opts['sa-steps']:,}")):
        head = head.replace(tok, val)
    tex = head + "\n".join(doc.body) + "\n\n\\end{document}\n"
    path = os.path.join(a.out, "report.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"  report  {os.path.relpath(path)}")

    if not a.no_pdf and shutil.which("pdflatex"):
        # Build in a scratch directory under a private -jobname. Both matter:
        # -output-directory keeps the .aux/.log/.toc out of the repository, and
        # -jobname stops us colliding with an editor's build-on-save watcher, if
        # one is running -- it compiles report.tex too, and pdflatex would
        # happily read the half-written ./report.aux that watcher is producing
        # and die on it. Graphics paths stay relative to cwd.
        import tempfile
        job = "sweepreport"
        with tempfile.TemporaryDirectory(prefix="cw-sweep-tex-") as td:
            for i in range(2):      # twice, for the table of contents
                p = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                     "-output-directory", td, "-jobname", job, "report.tex"],
                    cwd=a.out, capture_output=True, text=True)
                if p.returncode != 0:
                    break
            built = os.path.join(td, job + ".pdf")
            if p.returncode == 0 and os.path.exists(built):
                shutil.copyfile(built, os.path.join(a.out, "report.pdf"))
                print(f"  pdf     "
                      f"{os.path.relpath(os.path.join(a.out, 'report.pdf'))}")
            else:
                tail = "\n".join(p.stdout.splitlines()[-25:])
                print(f"  !! pdflatex failed:\n{tail}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
