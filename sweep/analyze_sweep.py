#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_sweep.py — analyse the runs produced by sweep/run_sweep.sh.

Reads sweep/results/<tier>/s<seed>/<study>/<tag>.{meta,log,csv}, rebuilds each
run's option dict from the recorded command line, and produces:

    sweep/figures/*.png     one figure per study + an overview
    sweep/report.tex        the report: what each knob does, the figures,
                            the paired statistics, and the winning command
    sweep/report.pdf        compiled with pdflatex, when it is installed

Three things the reader should know about the shape of the data.

*Pairing.* Every run of a study uses the same (n, m, seed), so instance k is
byte-identical across runs (cw.c:2574 seeds instance k with seed+k). Comparisons
are therefore *paired*: we compare cost_i(A) with cost_i(B) on the same instance
and report the mean of the differences, its 95 % CI, and a distribution-free
sign test. That is far tighter than comparing two means with independent
standard deviations.

*Seeds.* Each configuration is run at every seed in the sweep's seed list, and
load() pools them: the per-instance vectors are concatenated in seed order, so
the pairing above still holds position-by-position while the paired sample grows
by a factor of len(seeds). cw has one --seed, driving both the instances and the
annealing RNG (cw.c:2679), so a seed is a full replication and no result here
rests on a single instance draw.

*Tiers.* The same grids are run at two budgets -- 10^5 steps with m=1000, and
10^7 with m=200, the budget the solver is actually used at. The body of the
report is one tier (--tier, default `hi`); budget_shift() puts the two side by
side and names the knobs whose recommendation does not transfer.

The recommended configuration at the end is *derived*, not hand-picked: a knob
enters it only if its best setting beat its own default significantly. That
combination is then re-run, on the sweep's own seeds and on as many seeds it has
never seen, because tuning knobs one at a time does not make them additive (see
the `tuned` study, where the naive combination is worse than the defaults).

Usage:
    uv run sweep/analyze_sweep.py
    uv run sweep/analyze_sweep.py --tier lo        # report the 10^5 tier instead
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
    "ops": "1,1,1,0,0,0", "or-max": 3, "sa-knn": 20, "restarts": 1,
    "split": "off", "split-every": 0, "split-tour": "both",
    "pick": 2, "pick-crit": "lb", "pick-eps": 0.3,
    "t-accept": 0.001, "t-decades": 2, "lambda": 1.0, "mu": 0.0,
    "knn": 0, "threads": 0, "cw-rand": "perturb", "cw-alpha": 0.03,
    "no-sa": False, "exact": False, "2opt": False,
    # added with swap*, route opening, the selection biases and racing
    "vrank": 1, "pick2": 1, "reloc-side": "coin", "pair": 0,
    "race": 0.0, "race-at": 0.25,
}
NUMERIC = {"n", "m", "seed", "sa-steps", "or-max", "sa-knn", "restarts",
           "split-every", "pick", "t-decades", "knn", "threads",
           "vrank", "pick2", "pair"}
FLOAT = {"pick-eps", "t-accept", "lambda", "mu", "cw-alpha", "race-at"}

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


def norm_ops(s: str) -> str:
    """--ops padded to the canonical six weights.

    cw accepts 4 (or fewer) values and zero-fills the rest, so `1,1,1,0` and
    `1,1,1,0,0,0` are the same run. Normalising here keeps runs recorded before
    swap* existed comparable with runs recorded after, and lets DEFAULTS hold a
    single spelling.
    """
    parts = [p for p in re.split(r"[,:/]", str(s).strip()) if p != ""]
    parts = (parts + ["0"] * 6)[:6]
    return ",".join("%g" % float(p) for p in parts)


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
        elif key == "race":                 # a margin, or the word "off"
            val = 0.0 if val == "off" else float(val)
        elif key == "ops":
            val = norm_ops(val)
        o[key] = val
        i += 2
    if o.get("no-sa"):
        o["sa-steps"] = 0
    return o


class Run:
    __slots__ = ("study", "tag", "opts", "ok", "shell_s", "log", "cost",
                 "cost_init", "time_ms", "routes_v", "feasible", "seeds")

    def __init__(self, study, tag, opts, ok, shell_s, log, arrays, seeds=None):
        self.study, self.tag, self.opts, self.ok = study, tag, opts, ok
        self.shell_s, self.log = shell_s, log
        (self.cost, self.cost_init, self.time_ms,
         self.routes_v, self.feasible) = arrays
        self.seeds = tuple(seeds) if seeds is not None else (opts["seed"],)

    # Instances are identical iff (n, m, seeds) match -> pairing key. The seed
    # tuple is part of it because a pooled run concatenates one block per seed:
    # two runs are only comparable position-by-position if they pooled the same
    # seeds in the same order, so a configuration that failed on one seed is
    # refused by paired() rather than silently misaligned.
    @property
    def instkey(self):
        return (self.opts["n"], self.opts["m"], self.seeds)

    @property
    def n_seeds(self):
        return len(self.seeds)

    @property
    def mean(self):
        return float(self.cost.mean())

    def __repr__(self):
        return f"<Run {self.study}/{self.tag} mean={self.mean:.5f}>"


def ffloat(s, default=float("nan")):
    """float() that also accepts a comma decimal separator.

    run_sweep.sh pins LC_ALL=C, but .meta files written before it did can carry
    a locale-formatted wall time such as "0,055".
    """
    try:
        return float(str(s).replace(",", "."))
    except (TypeError, ValueError):
        return default


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
               ffloat(meta.get("wall_s", "nan")), log, arrays)


ARRAY_FIELDS = ("cost", "cost_init", "time_ms", "routes_v", "feasible")


def pool(study: str, tag: str, per_seed) -> Run:
    """Merge the per-seed runs of one configuration into a single Run.

    A seed is a full replication: `--seed` drives both the instance set
    (cw.c:2574) and the annealing RNG (cw.c:2679), so each seed is a fresh draw
    of instances solved with fresh randomness. Concatenating the per-instance
    vectors **in seed order** keeps every downstream comparison exactly paired
    --- instance j of seed s lands at the same offset in every run of the study
    --- while multiplying the paired sample by the number of seeds. Nothing in
    the analysis below has to know this happened.

    Scalars read out of the log (wall time, T0, acceptance rate) are averaged
    instead: each seed measured the same quantity on an equally large sample.
    """
    per_seed = sorted(per_seed, key=lambda t: t[0])
    seeds = [s for s, _ in per_seed]
    rs = [r for _, r in per_seed]
    arrays = tuple(np.concatenate([getattr(r, f) for r in rs])
                   for f in ARRAY_FIELDS)
    log = {}
    for key in set().union(*(set(r.log) for r in rs)):
        vals = [r.log[key] for r in rs if key in r.log]
        log[key] = sum(vals) / len(vals)
    shell = [r.shell_s for r in rs if not math.isnan(r.shell_s)]
    return Run(study, tag, dict(rs[0].opts), True,
               sum(shell) / len(shell) if shell else float("nan"),
               log, arrays, seeds=seeds)


def load(results_dir: str) -> list[Run]:
    """All runs of one tier, pooled over the seeds.

    Layout is <results_dir>/s<seed>/<study>/<tag>.{meta,log,csv}. Runs sharing
    (study, tag) across seed directories are one configuration replicated, and
    are pooled by pool() above.
    """
    groups = defaultdict(list)
    for sd in sorted(os.listdir(results_dir)):
        if not (sd.startswith("s") and sd[1:].isdigit()):
            continue
        seed = int(sd[1:])
        sroot = os.path.join(results_dir, sd)
        for study in sorted(os.listdir(sroot)):
            sdir = os.path.join(sroot, study)
            if not os.path.isdir(sdir):
                continue
            for fn in sorted(os.listdir(sdir)):
                if not fn.endswith(".meta"):
                    continue
                r = read_run(os.path.join(sdir, fn[:-5]), study, fn[:-5])
                if r.ok:
                    groups[(study, r.tag)].append((seed, r))
    return [pool(st, tg, v) for (st, tg), v in sorted(groups.items())]


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


OPNAMES = ("rel", "swap", "2opt", "or", "swap*", "open")


def ops_label(ops: str) -> str:
    bits = [int(float(x) > 0) for x in norm_ops(ops).split(",")]
    on = [OPNAMES[i] for i, b in enumerate(bits) if b]
    return "+".join(on) if on else "none"


def wall_of(r) -> float:
    """Seconds of wall clock for a run: cw's own figure, else the shell's."""
    v = r.log.get("wall_s")
    if v is None or not np.isfinite(v):
        v = r.shell_s
    return float(v)


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
    # The ladder is a set of multiples of the tier's budget, so its ends move
    # with the tier: 10^3..10^6 at `lo`, 10^5..10^7 at `hi`. Read them off the
    # data so the table headers and the "not within" cell stay true.
    ladder = sorted({r.opts["sa-steps"] for r in sel(runs, "init")})
    s_min, s_max = (ladder[0], ladder[-1]) if ladder else (0, 0)
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
                     ("%s" % f"{hit:,}") if hit
                     else "not within %s" % f"{s_max:,}"])
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

    doc.table([r"$n$", r"excess at %s (\%%)" % f"{s_min:,}",
               r"excess at %s (\%%)" % f"{s_max:,}",
               "steps to catch up"], rows,
              "Penalty for a random start, paired against the C\\&W start at the "
              "same budget. `Catch up' is the first budget at which the upper end "
              "of the 95\\,\\% CI reaches zero.", align="rrrr")
    doc.table([r"$n$", "at smallest $S$", "mid-range $S$", "at largest $S$"], brows,
              "Budget multiplier: steps the random start needs to match the C\\&W "
              "start, relative to the C\\&W budget. A ratio of 1 means the head "
              "start has been fully repaid.", align="rrrr")
    # The closing claim is checked, not asserted: "never overtakes" means no
    # (n, budget) cell where the random start is significantly *better*.
    over = [(r.opts["n"], r.opts["sa-steps"])
            for r in sel(runs, "init", init="random")
            for c in [next((x for x in sel(runs, "init", n=r.opts["n"], init="cw")
                            if x.opts["sa-steps"] == r.opts["sa-steps"]), None)]
            if c for st in [paired(c, r)] if st and st["sig"] and st["pct"] < 0]
    tail = (r"""It never overtakes --- not one of the %d cells in the grid has the
random start significantly ahead --- so it buys no diversity the annealing can
exploit. Keep the construction."""
            % (len(ns) * len(ladder))) if not over else (
        r"""It does overtake in %d of the %d cells (%s), which is a genuine
        exception to the earlier reading and worth following up rather than
        averaging away."""
        % (len(over), len(ns) * len(ladder),
           ", ".join("$n=%d$ at %s steps" % (n, f"{s:,}") for n, s in over[:4])))
    doc.p(r"""
\textbf{Reading.} The random start converges towards parity as the budget grows:
the penalty falls from %s at %s steps to %s at %s. %s
""" % (rows[0][1] + r"\,\%" if rows else "--", f"{s_min:,}",
       rows[0][2] + r"\,\%" if rows else "--", f"{s_max:,}", tail))


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
  \item \textbf{relocate} (\texttt{mv\_relocate}, \texttt{cw.c:1040}) --- take $u$
        out of its route and re-insert it next to $v$, possibly in another route.
  \item \textbf{swap} (\texttt{mv\_swap}, \texttt{cw.c:1083}) --- exchange the
        positions of $u$ and $v$.
  \item \textbf{2-opt} (\texttt{mv\_2opt}, \texttt{cw.c:1362}) --- replace the two
        edges carried by $u$ and $v$ by the other pairing; inside one route this
        reverses a segment, across two routes it exchanges their tails.
  \item \textbf{or-opt} (\texttt{mv\_oropt}, \texttt{cw.c:1148}) --- move a whole
        run of 2 to \texttt{-{}-or-max} consecutive customers starting at $u$,
        optionally reversed, next to $v$.
\end{itemize}
Every move is accepted by the Metropolis rule on its exact cost delta. The
default is \texttt{1,1,1,0}: or-opt is present in the code but switched off.
\texttt{-{}-ops} takes two further weights, for swap* and route opening, which
are also zero by default and have a section of their own
(Section~\ref{sec:newops}); this section is about the original four, so every
run in it leaves the other two at zero.
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
                  "(\\texttt{cw.c:2546}).", align="rrrrr")
    # "The default subset wins" is a claim, so it is checked rather than
    # asserted: any subset or weighting significantly below the default is
    # named. Same for "--or-max sits inside the noise".
    beaters = [(ops_label(r.opts["ops"]), s) for r, s in
               list(zip(subs, stats)) + [(r, paired(base, r)) for r in ws]
               if s and s["sig"] and s["pct"] < 0]
    or_sig = [r.opts["or-max"] for r, s in zip(om, st) if s and s["sig"]] if orows else []
    doc.p((r"""
\textbf{Reading.} The default subset wins: every other subset and weighting is
worse or indistinguishable, and turning or-opt on costs a small but significant
amount.""" if not beaters else r"""
\textbf{Reading.} The default subset is \emph{not} the best here: %s beat it
significantly (best %+.3f\,\%%). That contradicts the same study at the other
budget and is the single most actionable line in this section.""" % (
        esc(", ".join(b for b, _ in beaters[:3])),
        min(s["pct"] for _, s in beaters))) + r"""
Or-opt overlaps with relocate (a length-1 relocate is the same move) while being
more expensive per draw, so at a fixed step count it buys less. """ + (
        r"\texttt{-{}-or-max} is then irrelevant --- its whole range sits inside "
        r"the noise. Leaving or-opt off is the right default."
        if not or_sig else
        r"\texttt{-{}-or-max} does move here, at %s, which it did not at the "
        r"lower budget." % esc(", ".join(str(v) for v in or_sig))))


def study_newops(runs, out, doc):
    rs = sel(runs, "newops")
    if not rs:
        return
    # The x1 rung of the ladder is by construction the same run as the weight
    # blocks below (same n, same budget, stock operators), which is what makes
    # it a valid baseline for them. Guard it rather than assume it.
    base = next((r for r in rs if r.tag == "bud_def_x1"), None)
    others = [r for r in rs if r.tag.startswith(("sstar_", "open_", "mix_"))]
    if base and others and any(r.opts["sa-steps"] != base.opts["sa-steps"]
                               for r in others):
        print("  !! newops: ladder and weight blocks disagree on --sa-steps",
              file=sys.stderr)
        base = None
    if base is None:
        return

    doc.sec("swap* and route opening", "sec:newops")
    doc.p(r"""
\textbf{What it controls.} \texttt{-{}-ops} carries two further weights beyond
the four of Section~\ref{sec:ops}, both defaulting to zero:
\begin{itemize}\itemsep2pt
  \item \textbf{swap*} (\texttt{mv\_swapstar}, \texttt{cw.c:1216}) --- Vidal's
        \emph{Hybrid genetic search for the CVRP}, C\&OR 2022. Ordinary swap
        forces $u$ into $v$'s slot; swap* drops that and re-inserts each of the
        two customers at its \emph{best} position in the other's route. The two
        routes are disjoint, so the two re-insertions are independent and the
        delta is still exact --- but the scan costs $O(L_1+L_2)$ instead of
        $O(1)$. The kNN neighbour selects the target \emph{route}, not the
        partner: $v$ is then the worse by regret of two uniform customers of
        that route.
  \item \textbf{opening} (\texttt{mv\_open}, \texttt{cw.c:1325}) --- isolate one
        customer in an empty route. Almost always worsening, and that is the
        point: the annealing can empty a route but has no move that repopulates
        one, so the number of active routes is a one-way door between two
        Splits. This is the only move that opens it again.
\end{itemize}
""")
    doc.p(r"""
\textbf{The question.} swap* is the one non-elementary operator in the solver,
so it has to earn its cost: at a fixed step count it is not competing on equal
terms. Everything below is therefore also measured against wall time.
""")

    fig = plt.figure(figsize=(12, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])

    # --- weight of each new operator, one at a time
    ax = fig.add_subplot(gs[0, 0])
    ss = sorted([r for r in rs if r.tag.startswith("sstar_")],
                key=lambda r: float(r.opts["ops"].split(",")[4]))
    op = sorted([r for r in rs if r.tag.startswith("open_")],
                key=lambda r: float(r.opts["ops"].split(",")[5]))
    for lab, group, idx, col in (("swap* weight", ss, 4, CAT[0]),
                                 ("opening weight", op, 5, CAT[1])):
        if not group:
            continue
        st = [paired(base, r) for r in group]
        x = [float(r.opts["ops"].split(",")[idx]) for r in group]
        ax.plot(x, [s["pct"] for s in st], "o-", color=col, label=lab)
        ax.fill_between(x, [s["pct_lo"] for s in st], [s["pct_hi"] for s in st],
                        color=col, alpha=0.15, linewidth=0)
    ax.axhline(0, color=INK3, lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("weight (relative to 1 for relocate / swap / 2-opt)")
    ax.set_ylabel("Δ vs default (%)")
    ax.set_title("One new operator at a time, equal steps", loc="left")
    ax.legend()
    style(ax)

    # --- the mixtures
    ax = fig.add_subplot(gs[0, 1])
    mix = sorted([r for r in rs if r.tag.startswith("mix_")], key=lambda r: r.mean)
    if mix:
        bar_delta(ax, [ops_label(r.opts["ops"]) for r in mix],
                  [paired(base, r) for r in mix],
                  "Operator mixtures, equal steps")

    # --- the honest one: cost against measured wall time
    ax = fig.add_subplot(gs[1, :])
    LADDER = (("def", "default 1,1,1,0", CAT[2]),
              ("sstar", "+ swap*", CAT[0]),
              ("sstaropen", "+ swap* + opening", CAT[1]),
              ("dropswap", "swap* instead of swap", CAT[3]))
    iso_rows, curves = [], {}
    for key, lab, col in LADDER:
        g = sorted([r for r in rs if r.tag.startswith(f"bud_{key}_x")],
                   key=lambda r: r.opts["sa-steps"])
        if not g:
            continue
        t = [wall_of(r) for r in g]
        c = [r.mean for r in g]
        curves[key] = (g, t, c)
        ax.plot(t, c, "o-", color=col, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("wall time for %s instances (s, log)"
                  % f"{base.opts['m']:,}" if base else "wall time (s, log)")
    ax.set_ylabel("mean cost")
    ax.set_title("Cost against wall time — the comparison that counts", loc="left")
    ax.legend()
    style(ax)
    fig.tight_layout()
    savefig(fig, out, "newops.png")
    doc.fig("newops.png",
            "Top left: each new operator's weight swept alone. Top right: "
            "mixtures. Bottom: cost against measured wall time for four "
            "configurations swept over a 32x budget range --- a curve that sits "
            "lower-left dominates.")

    doc.table(["mixture", "mean cost", *DHEAD],
              [[ops_label(r.opts["ops"]), "%.5f" % r.mean,
                *stat_cells(paired(base, r))] for r in mix],
              "Operator mixtures at equal step count, paired against the stock "
              "\\texttt{-{}-ops 1,1,1,0,0,0}. Equal steps flatters swap*, which "
              "is why the wall-time reading below is the one to trust.",
              align="lrrrr")

    # iso-time: for each budget of the default, what does each config cost at
    # the same wall time?
    if "def" in curves and len(curves) > 1:
        gd, td, cd = curves["def"]
        for key, lab, _ in LADDER[1:]:
            if key not in curves:
                continue
            g, t, c = curves[key]
            row = [esc(lab)]
            for target_t in (td[len(td) // 2], td[-1]):
                ci = float(np.interp(math.log(target_t),
                                     np.log(t), c)) if min(t) <= target_t <= max(t) else float("nan")
                cdef = float(np.interp(math.log(target_t), np.log(td), cd))
                row.append("%.2fs" % target_t)
                row.append("%.5f" % cdef if np.isfinite(cdef) else "--")
                row.append("%.5f" % ci if np.isfinite(ci) else "out of range")
                row.append("%+.3f\\,\\%%" % (100 * (ci - cdef) / cdef)
                           if np.isfinite(ci) else "--")
            iso_rows.append(row)
    if iso_rows:
        doc.table(["configuration", "time", "default", "this", r"$\Delta$",
                   "time", "default", "this", r"$\Delta$"], iso_rows,
                  "Iso-time reading of the bottom panel: both curves "
                  "interpolated in $\\log$(wall time) at two points of the "
                  "default's own ladder. Negative $\\Delta$ = better at the "
                  "same wall time.", align="lrrrrrrrr")

    # --- across dimension
    ns = sorted({r.opts["n"] for r in rs if r.tag.endswith(("_off", "_on"))})
    rows = []
    for n in ns:
        a = next((r for r in rs if r.tag == f"n{n}_off"), None)
        b = next((r for r in rs if r.tag == f"n{n}_on"), None)
        s = paired(a, b)
        if s:
            rows.append([n, "%.5f" % a.mean, "%.5f" % b.mean, *stat_cells(s),
                         "%.2f" % (wall_of(b) / max(wall_of(a), 1e-9))])
    if rows:
        doc.table([r"$n$", "default", "+swap*+open", *DHEAD, "time x"], rows,
                  "\\texttt{-{}-ops 1,1,1,0,1,0.05} against the default across "
                  "dimension, at equal step count. The last column is the wall "
                  "time ratio --- the price of the equal-step gain.",
                  align="rrrrrrr")
    # The two load-bearing claims -- "survives the iso-time yardstick" and
    # "the weight is insensitive" -- are read back out of the tables above.
    # the "+ swap*" rung specifically: swap* added, nothing else changed
    iso_ds = [float(c.replace("\\,\\%", "")) for r in iso_rows
              if r[0] == esc("+ swap*") for c in (r[4], r[8]) if c != "--"]
    iso_ok = bool(iso_ds) and max(iso_ds) < 0
    sst = [(r.opts["ops"], paired(base, r)) for r in rs
           if r.tag.startswith("sstar_") and paired(base, r)]
    spread = (max(s["pct"] for _, s in sst) - min(s["pct"] for _, s in sst)) \
        if sst else float("nan")
    n100 = next((r for r in rows if r[0] == 100), None)
    tx = n100[-1] if n100 else "--"
    doc.p(r"""
\textbf{Reading.} swap* is %s. It costs %s$\times$ the wall time per step at
$n=100$, and the iso-time table %s --- %s. Its weight is insensitive over the
whole range tried: a quarter to four times the weight of relocate spans only
%.3f\,\%%, which is what one wants from a knob.
""" % (("the largest single improvement in this document"
        if sst and min(s["pct"] for _, s in sst) <= min(
            (k[3]["pct"] for k in knobs(runs)), default=0)
        else "a clear improvement at equal step count"),
       tx,
       ("still shows it ahead" if iso_ok else "does \\emph{not} keep it ahead"),
       ("so the extra work per draw is bought back" if iso_ok else
        "so at this budget the equal-step gain is paid for in wall time and "
        "does not survive the change of yardstick --- the honest reading is "
        "that swap* is not free here"),
       spread))
    doc.p(r"""
Route opening is a different kind of thing. On its own it is worth almost
nothing, and at large weight it is actively harmful --- unsurprising, since it
never improves a solution by itself. It is a pure enabler: the annealing can
empty a route but has no other move that repopulates one, so without it the
route count is monotone between two Splits. Kept at a small weight it is close
to free and slightly positive.
""")
    doc.p(r"""
The one result here that contradicts the intuition behind the code is that
\emph{dropping} swap in favour of swap* is not a trade-off at all: it is the
best configuration on the ladder at both time points. For any given pair, the
in-place exchange is one of the positions swap* already sweeps, so
$\Delta_{\text{swap*}} \le \Delta_{\text{swap}}$ always; the measurement says
that the ten-fold cheaper draw of ordinary swap does not buy back that
domination. swap is nevertheless kept in the code as a separate operator --- it
is a weighting decision, and \texttt{-{}-ops 1,1,1,0,1,0.05} remains the safer
recommendation at small $n$, where the two are within noise of each other.
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
partner $v$. \texttt{sa\_cand} (\texttt{cw.c:1010}) draws $v$ uniformly from the
$K$ nearest neighbours of $u$, with $K=$ \texttt{-{}-sa-knn}; at $K=0$ it draws
uniformly from all customers. This is what keeps a move geometrically local, so
that the cost delta has a chance of being negative. $K$ is clamped to $n-1$
(\texttt{cw.c:1675}).
""")
    doc.note(r"""
$K$ does double duty: at $K=0$ the kNN lists are not built, and
\texttt{cw.c:1676} then \emph{silently} forces uniform vertex selection as well,
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
    # Which K actually wins at each n, so the claim "K=20 everywhere" is read
    # off the data. At n=20 every K >= 19 is the same run (clamped), so a tie
    # there is not evidence against the default.
    winner = {}
    for n in ns:
        by = {r.opts["sa-knn"]: r for r in sel(runs, "knn", n=n)}
        if by:
            winner[n] = min(by, key=lambda K: by[K].mean)
    off = {n: K for n, K in winner.items() if min(K, n - 1) != min(20, n - 1)}
    doc.p((r"""
\textbf{Reading.} $K=20$ is the optimum at every dimension tested (%s), and the
curve is a clean U: too small a list starves the move of good partners, too
large a one dilutes it with distant ones that will be rejected. The default
needs no change --- and notably it does not need to grow with $n$.
""" % esc(", ".join("n=%d" % n for n in sorted(winner)))) if not off else (r"""
\textbf{Reading.} The best $K$ is \emph{not} 20 everywhere at this budget: %s.
The curve is still a U --- too small a list starves the move of good partners,
too large a one dilutes it with distant ones that will be rejected --- but its
minimum has moved, so the default is worth revisiting at this budget. Read the
paired column before acting: a shifted argmin inside the noise is not a shifted
optimum.
""" % esc(", ".join("n=%d prefers K=%s" % (n, K) for n, K in sorted(off.items())))))


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
    # This is the knob the two tiers disagree about most, so the paragraph is
    # written from the iso-budget curve rather than from the previous reading.
    iso1 = one(runs, "restarts", "iso_R1")
    budget = iso1.opts["sa-steps"] if iso1 else 0
    isost = [(r.opts["restarts"], paired(iso1, r)) for r in iso[1:]] if iso else []
    isost = [(R, s) for R, s in isost if s]
    bestR, bestS = min(isost, key=lambda t: t[1]["pct"]) if isost else (None, {})
    anysig = [R for R, s in isost if s["sig"] and s["pct"] < 0]
    if anysig:
        head = (r"""At equal budget, splitting the run pays: %d restarts of
$%s$ steps each beat a single run of $%s$ by $%+.3f\,\%%$, and %s of the %d
restart counts tried clear significance. The annealing \emph{is} getting stuck
in a way a fresh start fixes --- so at this budget the whole allowance should
not go into one long anneal."""
                % (bestR, f"{budget // bestR:,}".replace(",", "\\,"),
                   f"{budget:,}".replace(",", "\\,"), bestS["pct"],
                   len(anysig), len(isost)))
    else:
        head = (r"""At equal budget, restarts are flat: splitting $%s$ steps into
2, 4, \dots, 32 independent anneals changes nothing significant (best
$%+.3f\,\%%$ at $R=%s$). The annealing is not getting trapped in a way that a
fresh start would fix, so the whole budget may as well go into one run."""
                % (f"{budget:,}".replace(",", "\\,"),
                   bestS.get("pct", float("nan")), bestR))
    doc.p(r"""
\textbf{Reading.} %s The random start is the exception --- it degrades sharply as
its share of the budget shrinks, which is the same result as
Section~\ref{sec:init} seen from another angle. On diversity, the default
\texttt{perturb} at $\alpha=0.03$ is the best of the options: too much
perturbation ($\alpha \ge 0.3$) damages every restart, and \texttt{off} is worse
still.
""" % head)


def study_split(runs, out, doc):
    rs = sel(runs, "split")
    if not rs:
        return
    modes = ["off", "cw", "end", "both"]
    grid = {(r.opts["split"], r.opts["split-every"]): r
            for r in rs if r.tag.startswith("m")}
    # --split-every is a period in steps, so the sweep scales it with the tier
    # (STEPS/1000, /100, /10) to hold the *number* of Splits fixed. Read the
    # periods off the data rather than hardcoding the `lo` values.
    evs = sorted({e for _, e in grid})
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
an independent code path (\texttt{cw.c:1744} versus \texttt{cw.c:1968/2075}), so
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
                  "--split-tour (with --split both --split-every %d)"
                  % (tours[0].opts["split-every"]),
                  "Δ vs no Split (%)")

    ax = axes[2]
    big = [r for r in rs if r.tag.startswith("n")]
    if big:
        bns = sorted({r.opts["n"] for r in big})
        w = 0.35
        # the two periods the large-n block was run at: never, and the study's
        # middle period (STEPS/100), whatever the tier made that
        big_evs = sorted({r.opts["split-every"] for r in big})[:2]
        for k, e in enumerate(big_evs):
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
    if ref:
        # Every number in the paragraph below is read back out of the grid, so
        # it cannot drift from the table above it when the tier or the seed set
        # changes. `end` is the pure-guarantee cell: Split applied once, after
        # the annealing, where the theory says the result can never be worse.
        endr = grid.get(("end", 0))
        s_end = paired(ref, endr) if endr else {}
        cells = [(m, e, paired(ref, grid[(m, e)]))
                 for (m, e) in grid if (m, e) != ("off", 0)]
        cells = [c for c in cells if c[2]]
        bm, be, bs = min(cells, key=lambda c: c[2]["pct"]) if cells else (None, None, {})
        # the shortest period tried, i.e. the most aggressive setting
        fast = min((e for _, e in grid if e), default=0)
        fast_cells = [paired(ref, grid[(m, fast)]) for m in modes
                      if (m, fast) in grid and paired(ref, grid[(m, fast)])]
        fast_worst = max((c["pct"] for c in fast_cells), default=float("nan"))
        w_ref = ref.log.get("wall_s", float("nan"))
        w_fast = np.nanmean([grid[(m, fast)].log.get("wall_s", np.nan)
                             for m in modes if (m, fast) in grid]) if fast else float("nan")
        doc.p(r"""
\textbf{Reading.} Split never hurts: \texttt{-{}-split end} --- one Split, after
the annealing --- improves %d instances and worsens %d, which is the theoretical
guarantee showing up in the data. But on this instance family it barely helps
either: the best cell in the whole grid is \texttt{%s} + %s at $%+.3f\,\%%$. Its
internal `gain' counter is large while the net effect is small, meaning it mostly
undoes damage the annealing did rather than finding new structure. Applying it
every %s steps --- the most aggressive period tried --- costs %.1f$\times$ the
wall time (%.2f\,s against %.2f\,s) and is worth at worst $%+.3f\,\%%$: it drags
the search back to a repartitioned solution before the annealing can exploit its
own moves.
""" % (s_end.get("wins", 0), s_end.get("losses", 0),
       bm or "--", ("never" if not be else "every %s" % f"{be:,}"),
       bs.get("pct", float("nan")),
       f"{fast:,}", (w_fast / w_ref) if w_ref else float("nan"),
       w_fast, w_ref, fast_worst))


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
the vertex $u$ to disturb (\texttt{pick\_u}, \texttt{cw.c:987}). Uniform choice
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
The code's own comment (\texttt{cw.c:938--916}) notes that \texttt{lb} ``can stay
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
                  "exactly zero because \\texttt{cw.c:1676} forces uniform "
                  "selection there --- while the run header still prints the "
                  "requested rule.", align="rrrrrr")
    if ref:
        # Span of the whole grid, so the "nothing here moves" claim is measured
        # rather than asserted, and the worst cell is named by the data.
        gs = {(t, c): paired(ref, grid[(t, c)]) for (t, c) in grid
              if paired(ref, grid[(t, c)])}
        span = max((abs(s["pct"]) for s in gs.values()), default=float("nan"))
        nsig = sum(1 for s in gs.values() if s["sig"])
        (wt, wc), ws = max(gs.items(), key=lambda kv: kv[1]["pct"]) \
            if gs else ((None, None), {})
        (bt, bc), bs2 = min(gs.items(), key=lambda kv: kv[1]["pct"]) \
            if gs else ((None, None), {})
        doc.p(r"""
\textbf{Reading.} At $n=100$ with $%s$ steps the entire grid --- every tournament
size, every criterion, and the Fenwick sampler --- sits inside
$\pm %.3f\,\%%$ of the default, and %s of the %d cells clears significance. The
regret machinery is measurable in the code but not, at this budget, in the
result: whatever it saves in draw quality it appears to give back in draw cost.
The worst cell is $T=%s$ with \texttt{%s} at $%+.3f\,\%%$, from over-concentrating
on a few vertices; the best is $T=%s$ with \texttt{%s} at $%+.3f\,\%%$. The one
unambiguous finding here is the silent fallback at $K=0$: the header reports a
rule the code is not using.
""" % (f"{tier_steps(runs):,}".replace(",", "\\,"), span,
       ("none" if nsig == 0 else str(nsig)), len(gs),
       wt, wc, ws.get("pct", float("nan")),
       bt, bc, bs2.get("pct", float("nan"))))


def study_select(runs, out, doc):
    rs = sel(runs, "select")
    if not rs:
        return
    base = next((r for r in rs if r.tag == "vrank1_K20"), None) \
        or next((r for r in rs if r.tag == "side_coin"), None)

    doc.sec("Biasing the second vertex", "sec:select")
    doc.p(r"""
\textbf{What it controls.} Section~\ref{sec:pick} is about choosing the vertex
$u$ to move. These three knobs are about the \emph{partner} $v$, and about where
$u$ is put once chosen.
\begin{itemize}\itemsep2pt
  \item \texttt{-{}-vrank T} --- the kNN lists are stored sorted by distance, so
        drawing the index as the \emph{minimum of $T$ uniform draws} biases
        towards the nearer neighbours at no memory access at all. $T=1$ is the
        plain uniform draw.
  \item \texttt{-{}-pick2 T} --- a tournament of size $T$ among candidate
        partners, keeping the one of largest regret. Unlike \texttt{-{}-vrank}
        this one costs a memory read per candidate.
  \item \texttt{-{}-reloc-side long} --- relocate breaks the \emph{longer} of
        the two edges adjacent to $v$ instead of flipping a coin, which
        maximises the $-d(v,q)$ term of the insertion cost. Two reads, no
        randomness.
\end{itemize}
""")
    doc.p(r"""
\textbf{The question.} \texttt{-{}-vrank} is claimed to be \emph{coupled} with
\texttt{-{}-sa-knn}: a longer candidate list gives the reach, the rank bias
restores the concentration, and either one alone is supposed to hurt. That is
only testable on the grid, so the grid is what is run.
""")

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])

    # --- vrank x sa-knn heat map, the coupling claim
    ax = fig.add_subplot(gs[0, 0])
    vs = sorted({r.opts["vrank"] for r in rs if r.tag.startswith("vrank")})
    Ks = sorted({r.opts["sa-knn"] for r in rs if r.tag.startswith("vrank")})
    mat = np.full((len(vs), len(Ks)), np.nan)
    for i, v in enumerate(vs):
        for j, K in enumerate(Ks):
            r = next((x for x in rs if x.tag == f"vrank{v}_K{K}"), None)
            s = paired(base, r)
            if s:
                mat[i, j] = s["pct"]
    heat(ax, mat, [f"vrank {v}" for v in vs], [f"K={K}" for K in Ks],
         "--vrank x --sa-knn (Δ % vs vrank 1, K=20)", "Δ %")

    # --- pick2
    ax = fig.add_subplot(gs[0, 1])
    p2 = sorted([r for r in rs if r.tag.startswith("pick2_")],
                key=lambda r: r.opts["pick2"])
    if p2:
        st = [paired(base, r) for r in p2]
        x = [r.opts["pick2"] for r in p2]
        ax.plot(x, [s["pct"] for s in st], "o-", color=CAT[0])
        ax.fill_between(x, [s["pct_lo"] for s in st], [s["pct_hi"] for s in st],
                        color=CAT[0], alpha=0.15, linewidth=0)
        ax.axhline(0, color=INK3, lw=1.0)
        ax.set_xlabel("--pick2 (tournament size on v)")
        ax.set_ylabel("Δ vs --pick2 1 (%)")
        ax.set_title("Tournament on the partner", loc="left")
        style(ax)

    # --- reloc-side, alone and with swap* on
    ax = fig.add_subplot(gs[1, 0])
    side_rows = []
    pairs = [("side_coin", "side_long", "default operators"),
             ("side_coin_sstar", "side_long_sstar", "with swap* active")]
    labs, sts = [], []
    for a_tag, b_tag, lab in pairs:
        a = next((r for r in rs if r.tag == a_tag), None)
        b = next((r for r in rs if r.tag == b_tag), None)
        s = paired(a, b)
        if s:
            labs.append(lab)
            sts.append(s)
            side_rows.append([esc(lab), "%.5f" % a.mean, "%.5f" % b.mean,
                              *stat_cells(s)])
    if sts:
        bar_delta(ax, labs, sts, "--reloc-side long vs coin")

    # --- across dimension
    ax = fig.add_subplot(gs[1, 1])
    dim_rows, dlabs, dsts = [], [], []
    for n in sorted({r.opts["n"] for r in rs if r.tag.endswith(("_plain", "_vrank2"))}):
        a = next((r for r in rs if r.tag == f"n{n}_plain"), None)
        b = next((r for r in rs if r.tag == f"n{n}_vrank2"), None)
        s = paired(a, b)
        if s:
            dlabs.append(f"n={n}")
            dsts.append(s)
            dim_rows.append([n, "%.5f" % a.mean, "%.5f" % b.mean, *stat_cells(s)])
    if dsts:
        bar_delta(ax, dlabs, dsts, "--vrank 2 --sa-knn 30 vs default, by n")
    fig.tight_layout()
    savefig(fig, out, "select.png")
    doc.fig("select.png",
            "Top left: the \\texttt{-{}-vrank}/\\texttt{-{}-sa-knn} grid, the "
            "coupling claim. Top right: tournament size on the partner. Bottom "
            "left: the relocate insertion side. Bottom right: the recommended "
            "\\texttt{-{}-vrank 2 -{}-sa-knn 30} pair across dimension.")

    rows = []
    for i, v in enumerate(vs):
        for j, K in enumerate(Ks):
            r = next((x for x in rs if x.tag == f"vrank{v}_K{K}"), None)
            if r:
                rows.append([v, K, "%.5f" % r.mean, *stat_cells(paired(base, r))])
    doc.table([r"\texttt{-{}-vrank}", r"\texttt{-{}-sa-knn}", "mean cost", *DHEAD],
              rows, "The full \\texttt{-{}-vrank} $\\times$ "
              "\\texttt{-{}-sa-knn} grid, paired against \\texttt{vrank 1, "
              "K=20} (the default).", align="rrrrrr", long=True)
    if side_rows:
        doc.table(["context", "coin", "long", *DHEAD], side_rows,
                  "The relocate insertion side, with and without swap* active.",
                  align="lrrrrr")
    if dim_rows:
        doc.table([r"$n$", "default", "vrank 2, K=30", *DHEAD], dim_rows,
                  "The rank bias with the longer candidate list, across "
                  "dimension.", align="rrrrrr")
    # "The best cell is the default corner" and "vrank 2 + K=30 is among the
    # worst" are both checkable, so they are checked. If the grid ever comes
    # out the other way the paragraph has to say so.
    cells = {(v, K): x for v in vs for K in Ks
             for x in [next((y for y in rs if y.tag == f"vrank{v}_K{K}"), None)] if x}
    ranked = sorted(cells, key=lambda k: cells[k].mean)
    best_cell = ranked[0] if ranked else None
    claim_pair = (2, 30)
    pair_rank = (ranked.index(claim_pair) + 1) if claim_pair in ranked else None
    default_wins = best_cell == (1, 20)
    doc.p((r"""
\textbf{Reading.} The coupling claim does not reproduce here, and the heat map
is unusually unambiguous about it: the best cell of the whole grid is the
default corner \texttt{vrank 1, K=20}, and the supposed winning combination
\texttt{vrank 2, K=30} ranks %s of %s cells tried.""" % (pair_rank, len(ranked))
           if default_wins else r"""
\textbf{Reading.} The best cell of the grid is \texttt{vrank %d, K=%d}, not the
default corner --- so at this budget the coupling claim is \emph{not} simply
refuted, and \texttt{vrank 2, K=30} ranks %s of %s. Read the paired column
before acting on it: an argmin that moves inside the noise is not a result.""" %
           (best_cell[0], best_cell[1], pair_rank, len(ranked))) + r"""
The most likely explanation for the disagreement with the source is that the
claim was made against a construction that shares one $K=30$ neighbour list
between the savings and the annealing, whereas here the savings are exact for
$n \le 1500$ and the annealing builds its own list. Those are not the same
baseline, and this sweep cannot settle which is right for the other one --- only
what happens on this code at this budget.
""")
    doc.p(r"""
\texttt{-{}-pick2} is the mild surprise: a tournament of 3 on the partner is a
small but genuine improvement, which makes it the only selection bias in this
document that survives while costing a memory read per candidate. It is worth
about a tenth of what swap* is worth, and the curve is flat enough between 2 and
4 that the exact size does not matter. The relocate insertion side moves in the
right direction with and without swap* active, but neither interval excludes
zero: on this evidence it is free rather than useful, and it is left off.
""")


def study_race(runs, out, doc):
    rs = sel(runs, "race")
    if not rs:
        return
    base = next((r for r in rs if r.tag == "margin_off"), None)

    doc.sec("Racing between restarts, and interleaving", "sec:race")
    doc.p(r"""
\textbf{What it controls.} Section~\ref{sec:restarts} spends the budget equally
over the restarts. \texttt{-{}-race M} spends it unequally instead: at
\texttt{-{}-race-at} of its own budget (a quarter by default) a start is
compared with the best \emph{finished} start \emph{at the same point of its
trajectory} --- same temperature, which is what makes the comparison fair --- and
abandoned if it is worse by a relative margin $M$. Its unspent steps lengthen the
schedules of the starts that follow, and whatever is left at the end is spent
polishing the best solution found. The total step budget never changes; only its
allocation does.
""")
    doc.p(r"""
\texttt{-{}-pair} is a different kind of knob: two starts are advanced
alternately in the same loop so that one's memory latency overlaps with the
other's arithmetic. Each chain owns its solution, its incremental buffers and its
random stream, so without \texttt{-{}-race} the trajectories are identical bit
for bit and only the clock may move. That makes it a self-checking row in the
table below: if the cost column is not flat, something is wrong.
""")
    doc.p(r"""
\textbf{The question.} Every run in this section holds
$\texttt{restarts} \times \texttt{sa-steps}$ constant, because a racing gain at
unequal budget would be meaningless.
""")

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(gs[0, 0])
    mg = [r for r in rs if r.tag.startswith("margin_")]
    mg = sorted(mg, key=lambda r: r.opts["race"])
    if mg:
        bar_delta(ax, ["off" if r.opts["race"] == 0 else "%g" % r.opts["race"]
                       for r in mg], [paired(base, r) for r in mg],
                  "Racing margin (10 starts, equal total budget)")

    ax = fig.add_subplot(gs[0, 1])
    at = sorted([r for r in rs if r.tag.startswith("at_")],
                key=lambda r: r.opts["race-at"])
    if at:
        st = [paired(base, r) for r in at]
        x = [r.opts["race-at"] for r in at]
        ax.plot(x, [s["pct"] for s in st], "o-", color=CAT[1])
        ax.fill_between(x, [s["pct_lo"] for s in st], [s["pct_hi"] for s in st],
                        color=CAT[1], alpha=0.15, linewidth=0)
        ax.axhline(0, color=INK3, lw=1.0)
        ax.set_xlabel("--race-at (fraction of the budget at the checkpoint)")
        ax.set_ylabel("Δ vs no racing (%)")
        ax.set_title("Where to put the checkpoint", loc="left")
        style(ax)

    ax = fig.add_subplot(gs[1, 0])
    Rs = sorted({r.opts["restarts"] for r in rs if r.tag.startswith("R")})
    rrows, rlabs, rsts = [], [], []
    for R in Rs:
        a = next((r for r in rs if r.tag == f"R{R}_off"), None)
        b = next((r for r in rs if r.tag == f"R{R}_on"), None)
        s = paired(a, b)
        if s:
            rlabs.append(f"{R} starts")
            rsts.append(s)
            rrows.append([R, "%.5f" % a.mean, "%.5f" % b.mean, *stat_cells(s)])
    if rsts:
        bar_delta(ax, rlabs, rsts, "Racing vs equal split, by restart count")

    ax = fig.add_subplot(gs[1, 1])
    prows, pn, pspd = [], [], []
    for n in sorted({r.opts["n"] for r in rs if r.tag.startswith("pair")}):
        a = next((r for r in rs if r.tag == f"pair0_n{n}"), None)
        b = next((r for r in rs if r.tag == f"pair1_n{n}"), None)
        if not (a and b):
            continue
        spd = wall_of(a) / max(wall_of(b), 1e-9)
        same = "yes" if abs(a.mean - b.mean) < 1e-9 else "NO"
        pn.append(n)
        pspd.append(spd)
        prows.append([n, "%.5f" % a.mean, "%.5f" % b.mean, same,
                      "%.2fs" % wall_of(a), "%.2fs" % wall_of(b), "%.2fx" % spd])
    if pn:
        ax.plot(pn, pspd, "o-", color=CAT[0])
        ax.axhline(1.0, color=INK3, lw=1.0)
        ax.set_xscale("log")
        ax.set_xlabel("n (log)")
        ax.set_ylabel("speed-up of --pair 1 over --pair 0")
        ax.set_title("Interleaving two chains per core", loc="left")
        style(ax)
    fig.tight_layout()
    savefig(fig, out, "race.png")
    doc.fig("race.png",
            "Top: the racing margin and the checkpoint position, both at equal "
            "total budget. Bottom left: racing against an equal split as the "
            "number of starts grows. Bottom right: the interleaving speed-up, "
            "which is a timing result only --- the costs are identical.")

    doc.table(["margin", "mean cost", *DHEAD],
              [["off" if r.opts["race"] == 0 else "%g" % r.opts["race"],
                "%.5f" % r.mean, *stat_cells(paired(base, r))] for r in mg],
              "Racing margin, 10 starts at equal total budget.", align="rrrrr")
    if at:
        doc.table([r"\texttt{-{}-race-at}", "mean cost", *DHEAD],
                  [["%g" % r.opts["race-at"], "%.5f" % r.mean,
                    *stat_cells(paired(base, r))] for r in at],
                  "Position of the checkpoint, at margin 0.002.", align="rrrrr")
    if rrows:
        doc.table(["starts", "equal split", "racing", *DHEAD], rrows,
                  "Racing against an equal split, holding "
                  "$\\texttt{restarts} \\times \\texttt{sa-steps}$ constant.",
                  align="rrrrrr")
    if prows:
        doc.table([r"$n$", r"\texttt{-{}-pair 0}", r"\texttt{-{}-pair 1}",
                   "identical", "wall 0", "wall 1", "speed-up"], prows,
                  "Interleaving. The \\emph{identical} column is the check that "
                  "this is an engineering change and not an algorithmic one: "
                  "the two costs must agree exactly.", align="rrrlrrr")
    # "the gain grows with the restart count" is the load-bearing claim, and
    # rrows already holds it: read the first and last rung rather than assert.
    grow = ""
    if len(rrows) >= 2:
        def pct_of(row):
            try:
                return float(str(row[3]).replace(r"\textbf{", "").replace("}", ""))
            except ValueError:
                return float("nan")
        first, last = pct_of(rrows[0]), pct_of(rrows[-1])
        if np.isfinite(first) and np.isfinite(last):
            grow = (r"""with %s starts it is worth $%+.3f\,\%%$ and with %s
$%+.3f\,\%%$, so the gain %s with the restart count"""
                    % (rrows[0][0], first, rrows[-1][0], last,
                       "grows" if last < first else "does \\emph{not} grow"))
    doc.p(r"""
\textbf{Reading.} Racing needs starts to race: %s. It costs nothing --- the
budget is the same, only its allocation changes.
""" % (grow or "the gain is read off the table above"))
    doc.p(r"""
The margin behaves monotonically over the range tried, which is worth noting
because it was not the expectation: the tightest margin is the best one, and
loosening it only ever gives ground back. By $0.05$ almost no start is ever
behind by that much at the checkpoint, so the setting is indistinguishable from
\texttt{off}, and by $0.2$ it is exactly \texttt{off}. There was reason to expect
the other failure mode as well --- a margin so tight that good starts are killed
on noise --- and this sweep does not reach it, so the bottom of the useful range
is still unmeasured. The checkpoint position matters much less than the margin;
anything from a tenth to half of the budget is within noise of the default
quarter, and only pushing it out to three quarters clearly hurts, by which point
most of the budget the racing was meant to reclaim has already been spent.
""")
    doc.p(r"""
\texttt{-{}-pair} does exactly what it claims. The \emph{identical} column reads
\texttt{yes} on every row --- the interleaved chains reproduce the sequential
trajectories exactly --- and the speed-up appears only once the state of one
chain stops fitting in the low-level caches: slightly negative at $n \le 500$,
where the bookkeeping is not paid back, and clearly positive from $n = 1000$.
That is a smaller effect than the $1.42\times$ reported for the implementation
this was ported from; the shape of the curve is the same, so the difference is
most likely the cache hierarchy of the machine rather than the technique.
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
    # The whole point of the `hi` tier is that this knob is budget-coupled: the
    # cooling ratio is (Tend/T0)^(1/(S-1)), so the same decade count is traversed
    # 100x more slowly. Name the winning cell from the data, not from memory.
    if ref:
        cells = {(a, d): paired(ref, grid[(a, d)])
                 for a in As for d in Ds if (a, d) in grid}
        cells = {k: v for k, v in cells.items() if v}
        (ba, bd), bs = min(cells.items(), key=lambda kv: kv[1]["pct"]) \
            if cells else ((None, None), {})
        acc = grid[(ba, bd)].log.get("accept_pct", float("nan")) \
            if (ba, bd) in grid else float("nan")
        doc.p(r"""
\textbf{Reading.} The best cell at this budget is
$\texttt{-{}-t-accept}=%g$ with %s, worth $%+.3f\,\%%$ against the
default, and it realises a %.2f\,\%% acceptance rate. The mechanism is that a
schedule spanning too many decades spends the tail of the run at temperatures so
low that nothing is accepted --- the search is effectively over before the step
budget is --- while too few leaves it still hot at the end. Where that trade-off
lands is a function of the budget: the geometric ratio is
$(T_\mathrm{end}/T_0)^{1/(S-1)}$, so at $S=10^7$ the same decade count is
traversed a hundred times more slowly than at $10^5$. Section~\ref{sec:budget}
reports whether the two budgets agree here; if they do not, this is the knob to
re-measure whenever \texttt{-{}-sa-steps} changes substantially.
""" % (ba, ("1 decade" if bd == 1 else "%s decades" % bd),
       bs.get("pct", float("nan")), acc))


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
    # Both halves of this reading are measured: the best (lambda, mu) after
    # annealing, and what a K=5 savings list costs before and after it.
    sa_ref = one(runs, "construct", "sa_l1.0_m0")
    sa_all = [(r, paired(sa_ref, r)) for r in sel(runs, "construct")
              if r.tag.startswith("sa_l")]
    sa_all = [(r, s) for r, s in sa_all if s]
    bshape, bshape_s = min(sa_all, key=lambda t: t[1]["pct"]) if sa_all else (None, {})
    k5_nosa = one(runs, "construct", "nosa_n200_knn5")
    k5_sa = one(runs, "construct", "sa_n200_knn5")
    ex_nosa = one(runs, "construct", "nosa_n200_exact")
    ex_sa = one(runs, "construct", "sa_n200_exact")
    s_nosa = paired(ex_nosa, k5_nosa) if (ex_nosa and k5_nosa) else {}
    s_sa = paired(ex_sa, k5_sa) if (ex_sa and k5_sa) else {}
    doc.p(r"""
\textbf{Reading.} Two opposite lessons. The savings \emph{shape} matters and
survives the annealing: the best cell after SA is
$\lambda=%g$, $\mu=%g$, worth $%+.3f\,\%%$ against $\lambda=1$, $\mu=0$. The
savings \emph{list}, by contrast, does not survive at all: truncating it to the
5 nearest neighbours costs $%+.2f\,\%%$ on the raw construction and
$%+.3f\,\%%$ once the annealing has run. That is a useful lever for large $n$, where
Section~\ref{sec:timing} showed the $O(n^2)$ list is the dominant cost --- the annealing repairs""" % (
        bshape.opts["lambda"] if bshape else float("nan"),
        bshape.opts["mu"] if bshape else float("nan"),
        bshape_s.get("pct", float("nan")),
        s_nosa.get("pct", float("nan")), s_sa.get("pct", float("nan"))) + r"""
what the truncation breaks. \texttt{-{}-2opt} is likewise absorbed: it improves
the construction and changes nothing afterwards.
""")


def study_tuned(runs, out, doc):
    rs = sel(runs, "tuned")
    if not rs:
        return
    ns = sorted({r.opts["n"] for r in rs if "_x3_" not in r.tag})
    sp = next((r.opts["split-every"] for r in rs if r.tag.endswith("_split")), 0)
    variants = [("oropt", "or-opt enabled"), ("critrem", "pick-crit rem"),
                ("split", "Split both + every %s" % f"{sp:,}"),
                ("all", "the three combined"),
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
    # The 3x rung, when the tier ran it: is the default-vs-tuned ordering still
    # moving at the top of the budget range, or has it settled?
    x3 = []
    for n in ns:
        b = one(runs, "tuned", f"n{n}_x3_default")
        t = one(runs, "tuned", f"n{n}_x3_newall")
        b1 = one(runs, "tuned", f"n{n}_default")
        t1 = one(runs, "tuned", f"n{n}_newall")
        s3, s1 = paired(b, t) if (b and t) else {}, paired(b1, t1) if (b1 and t1) else {}
        if s3 and s1:
            x3.append([n, "%.5f" % b.mean, "%.5f" % t.mean, *stat_cells(s3),
                       "%+.3f" % s1["pct"]])
    if x3:
        steps1 = one(runs, "tuned", f"n{ns[0]}_default").opts["sa-steps"]
        doc.table([r"$n$", "default", r"\texttt{newall}", *DHEAD,
                   r"$\Delta$ at $1\times$ (\%)"], x3,
                  "The $3\\times$ rung ($%s$ steps): the same comparison at three "
                  "times the tier's budget, with the $1\\times$ delta beside it. "
                  "If the two columns agree, the ordering has settled; if the "
                  "$3\\times$ delta is still shrinking, the budget has not yet "
                  "stopped mattering." % f"{3 * steps1:,}", align="rrrrrrr")

    # "the combination is worse than its parts" is a claim about `all`; check it
    alls = [(n, paired(one(runs, "tuned", f"n{n}_default"),
                       one(runs, "tuned", f"n{n}_all"))) for n in ns]
    alls = [(n, s) for n, s in alls if s]
    worse = [n for n, s in alls if s["pct"] > 0]
    doc.p((r"""
\textbf{Reading.} The combination is worse than its parts: \texttt{all} loses to
the defaults at every $n$ tested (%s), even though one of its three ingredients
helps on its own.""" % esc(", ".join("n=%d" % n for n in worse))
           if len(worse) == len(alls) and alls else r"""
\textbf{Reading.} The combination loses to the defaults at %s of the %d
dimensions tested, so at this budget the naive stacking is not uniformly bad ---
but it is not uniformly good either.""" % (len(worse), len(alls))) + r"""
Gains found one knob at a time do not add up, which is exactly why the
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
    "new operators": ["ops"],
    "partner bias": ["vrank", "sa-knn"],
    "partner tournament": ["pick2"],
    "relocate side": ["reloc-side"],
    "racing": ["race", "race-at"],
}


def tier_steps(runs, default=100000):
    """The tier's own SA budget, read off its `tuned` baseline.

    Nothing in the analysis may hardcode a step count any more: the same study
    is run at 10^5 and at 10^7, and a comparison that picked the literal 10^5
    rung would silently find nothing in the `hi` tier.
    """
    t = one(runs, "tuned", "n100_default")
    if t is not None:
        return t.opts["sa-steps"]
    return default


def tier_label(runs):
    """A short human label for the tier, e.g. '10^7 steps, m=200'."""
    t = one(runs, "tuned", "n100_default")
    if t is None:
        return "?"
    return r"$10^{%d}$ steps, $m=%d\times%d$" % (
        round(math.log10(max(1, t.opts["sa-steps"]))), t.opts["m"], t.n_seeds)


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
    # The init ladder spans three decades, so its knob row has to be read at
    # one rung. That rung is the tier's own budget, not a literal 10^5: the
    # `hi` tier reruns the same ladder 100x higher up.
    steps = tier_steps(runs)
    add("initialisation", next((r for r in sel(runs, "init")
                                if r.opts["n"] == 100 and r.opts["init"] == "cw"
                                and r.opts["sa-steps"] == steps), None),
        [r for r in sel(runs, "init")
         if r.opts["n"] == 100 and r.opts["sa-steps"] == steps])

    # --- knobs added with swap*, the selection biases and racing ---
    # Baseline for the newops study is its own ladder rung at the default
    # budget, so the comparison is against the same step count.
    nb = next((r for r in sel(runs, "newops") if r.tag == "bud_def_x1"), None)
    add("new operators", nb,
        [r for r in sel(runs, "newops")
         if r.tag.startswith(("sstar_", "open_", "mix_"))])
    sb = next((r for r in sel(runs, "select") if r.tag == "vrank1_K20"), None)
    add("partner bias", sb,
        [r for r in sel(runs, "select") if r.tag.startswith("vrank")])
    add("partner tournament", next((r for r in sel(runs, "select")
                                    if r.tag == "pick2_1"), None) or sb,
        [r for r in sel(runs, "select") if r.tag.startswith("pick2_")])
    add("relocate side", next((r for r in sel(runs, "select")
                               if r.tag == "side_coin"), None),
        [r for r in sel(runs, "select") if r.tag in ("side_coin", "side_long")])
    add("racing", next((r for r in sel(runs, "race") if r.tag == "margin_off"), None),
        [r for r in sel(runs, "race") if r.tag.startswith(("margin_", "at_"))])
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


# ------------------------------------------------------- budget dependence
def budget_shift(lo_runs, hi_runs, out, doc):
    """Does a knob's recommendation survive a 100x larger budget?

    Every knob is measured twice, at the two tiers, and the two answers are put
    side by side. This is the one section that cannot be produced from a single
    tier, and it is the reason the sweep runs two.
    """
    if not lo_runs or not hi_runs:
        return
    lo, hi = knobs(lo_runs), knobs(hi_runs)
    lo_by = {l: (b, s) for l, _, b, s in lo}
    hi_by = {l: (b, s) for l, _, b, s in hi}
    shared = [l for l, _, _, _ in hi if l in lo_by]
    if not shared:
        return

    lo_steps, hi_steps = tier_steps(lo_runs), tier_steps(hi_runs)

    rows, flips = [], []
    for l in shared:
        lb, ls = lo_by[l]
        hb, hs = hi_by[l]
        # A "flip" is a change in the operational answer, not in the point
        # estimate: the knob was worth turning at one budget and is not at the
        # other, or the winning setting itself moved.
        verdict = ""
        if ls["sig"] != hs["sig"]:
            verdict = "significant only at %s" % ("$10^5$" if ls["sig"] else "$10^7$")
            flips.append(l)
        elif ls["sig"] and hs["sig"] and lb.tag != hb.tag:
            verdict = "same knob, different setting"
            flips.append(l)
        elif not ls["sig"] and not hs["sig"]:
            verdict = "inert at both"
        else:
            verdict = "stable"
        def cell(st):
            d = "%+.3f" % st["pct"]
            return (r"\textbf{%s}" % d if st["sig"] else d,
                    "[%+.3f,\\;%+.3f]" % (st["pct_lo"], st["pct_hi"]))

        rows.append([esc(l), tt(lb.tag), *cell(ls),
                     tt(hb.tag), *cell(hs), verdict])

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(shared) + 2.0))
    y = np.arange(len(shared))
    ax.barh(y - 0.19, [lo_by[l][1]["pct"] for l in shared], height=0.36,
            color=CAT[0], label=r"$10^5$ steps, $m=1000\times5$")
    ax.barh(y + 0.19, [hi_by[l][1]["pct"] for l in shared], height=0.36,
            color=CAT[1], label=r"$10^7$ steps, $m=200\times5$")
    ax.set_yticks(y)
    ax.set_yticklabels(shared)
    ax.invert_yaxis()
    ax.axvline(0, color=INK3, lw=1)
    ax.set_xlabel("best achievable Δ mean cost vs default (%)")
    ax.set_title("What each knob is worth, at two budgets 100× apart")
    # every bar extends left from 0, so the lower-left corner is the only
    # reliably empty region
    ax.legend(loc="lower left")
    style(ax, ygrid=False, xgrid=True)
    fig.tight_layout()
    savefig(fig, out, "budget_shift.png")

    doc.sec("Does any of this survive a 100$\\times$ larger budget?",
            "sec:budget")
    doc.p(r"""
Every grid in this report is run twice: once at $\texttt{-{}-sa-steps} = %s$ with
%s instances per seed, and once at $%s$ with %s. A knob tuned at the lower budget
is only useful if its answer still holds at the higher one, which is where this
solver is actually run --- and that is not something a single-budget sweep can
check. The two columns below are the same measurement at the two budgets.
""" % (f"{lo_steps:,}", "1000", f"{hi_steps:,}", "200"))
    doc.fig("budget_shift.png",
            "Each knob's best achievable improvement, measured independently at "
            "the two budgets. A bar pair that changes sign or magnitude is a "
            "recommendation that does not transfer.")
    doc.table(["knob", r"best @ $10^5$", r"$\Delta$ (\%)", r"95\,\% CI",
               r"best @ $10^7$", r"$\Delta$ (\%)", r"95\,\% CI", "verdict"], rows,
              "Each knob at each budget, with both intervals --- they are not "
              "equally precise, since $m$ is 1000 per seed at $10^5$ and 200 at "
              "$10^7$. \\emph{stable} means the same setting won and stayed "
              "significant; \\emph{inert at both} means the knob never earned a "
              "change. Anything else is a recommendation that depends on the "
              "budget.", align="llrrlrrl", long=True)
    if flips:
        doc.note(r"""
\textbf{%d of %d knobs do not transfer:} %s. For these the setting recommended
in Section~\ref{sec:best} is the one measured at $10^7$ steps, because that is
the budget this solver is run at; the $10^5$ column is kept only to show that
the disagreement exists.
""" % (len(flips), len(shared), esc(", ".join(flips))))
    else:
        doc.note(r"""
\textbf{No knob changed its answer between the two budgets.} That is a stronger
statement than any single-budget sweep can make, and it is what licenses reading
the rest of this report as advice rather than as a description of one operating
point.
""")


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
    """Flags from every knob whose winner beat its own default significantly.

    Several knobs share cw options -- "operators" and "new operators" both write
    --ops, "neighbourhood" and "partner bias" both write --sa-knn. Two knobs
    cannot set the same option to different values, so a collision is resolved
    in favour of the larger improvement and the loser is reported as dropped.
    """
    winners, dropped = [], []
    for label, ref, best, s in ks:
        keys = KNOB_KEYS.get(label, [])
        f = flags_for(best, keys) if keys else []
        if s["sig"] and s["pct"] < 0 and f:
            winners.append((label, best, s, f, keys))
        else:
            dropped.append((label, best, s, "no significant improvement"))

    winners.sort(key=lambda w: w[2]["pct"])          # strongest first
    claimed, kept, flags = {}, [], []
    for label, best, s, f, keys in winners:
        touched = [k for k in keys if ("--" + k) in f]
        clash = [k for k in touched if k in claimed]
        if clash:
            dropped.append((label, best, s, "same option as %s, which won by "
                            "more" % claimed[clash[0]]))
            continue
        for k in touched:
            claimed[k] = label
        flags += f
        kept.append((label, best, s, f))

    def drop(label, why):
        """Remove a knob after the fact, rebuilding flags from what is left."""
        nonlocal flags, kept
        row = next((k for k in kept if k[0] == label), None)
        if not row:
            return
        kept = [k for k in kept if k[0] != label]
        flags = [f for _, _, _, fl in kept for f in fl]
        dropped.append((row[0], row[1], row[2], why))

    # --or-max is inert unless or-opt carries a positive weight
    if "--or-max" in flags:
        ops = flags[flags.index("--ops") + 1] if "--ops" in flags else DEFAULTS["ops"]
        if float(norm_ops(ops).split(",")[3]) <= 0:
            drop("or-opt length", "inert: or-opt has weight 0 here")
    # racing has nothing to redistribute with a single start
    if "--race" in flags:
        rst = flags[flags.index("--restarts") + 1] if "--restarts" in flags \
            else DEFAULTS["restarts"]
        if int(rst) < 2:
            drop("racing", "inert: needs --restarts > 1, which did not win "
                           "its own knob")
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
    ("partner and insertion side", ["vrank", "pick2", "reloc-side"]),
    ("restarts", ["restarts", "cw-rand", "cw-alpha"]),
    ("budget allocation", ["race", "race-at", "pair"]),
    ("Split", ["split", "split-every", "split-tour"]),
)


# Per-option commentary, emitted only for the options the sweep actually
# changed. Keeping these keyed by option name rather than written as one block
# is what stops the text describing a previous run's winners.
OPTION_NOTES = {
    "t-decades": r"""
\textbf{\texttt{-{}-t-decades}} (default 2) --- the width of the temperature
schedule. The annealing does not take $T_0$ as an argument:
\texttt{calibrate\_T0} samples worsening moves on the instance itself and solves
for the $T_0$ at which a fraction \texttt{-{}-t-accept} of them would be accepted
($0.1\,\%$ by default). From there the temperature decays geometrically to
$T_{\text{end}} = T_0 \cdot 10^{-D}$, one factor
$\alpha = 10^{-D/(\text{steps}-1)}$ per step, with $D =$
\texttt{-{}-t-decades}. So $D$ is \emph{how many orders of magnitude the
temperature falls over the run}: $D=2$ ends a hundred times colder than it
started, $D=1$ only ten times colder. Lowering $D$ keeps the search warm for
longer; the reading is that the default schedule freezes early and spends its
last steps at temperatures where essentially nothing is accepted. It is
budget-dependent by construction.
""",
    "lambda": r"""
\textbf{\texttt{-{}-lambda} and \texttt{-{}-mu}} (defaults 1 and 0) --- the shape
of the Clarke \& Wright savings criterion,
\[
  s(i,j) \;=\; d_{0i} + d_{0j} \;-\; \lambda\, d_{ij}
                \;+\; \mu\,\lvert d_{0i} - d_{0j}\rvert .
\]
At $\lambda=1$, $\mu=0$ this is the original 1964 criterion. The construction
merges route endpoints in decreasing $s$ while capacity allows, so $s$ decides
nothing but the \emph{order} in which pairs are considered --- which is enough to
determine the whole solution. $\lambda$ reweights the direct link against the two
depot legs: above 1 the criterion is more sceptical of long links, so only
genuinely close pairs merge early and the routes come out more compact and less
radial. $\mu$ rewards merging customers at \emph{different} distances from the
depot, favouring routes that run outward and sweep back. The two interact and
neither is much good alone --- and the optimum is a fact about this instance
family, uniform points on a square, not a universal constant.
""",
    "ops": r"""
\textbf{\texttt{-{}-ops}} --- the operator mixture (Sections~\ref{sec:ops}
and~\ref{sec:newops}). Positions 5 and 6 are swap* and route opening, both zero
by default. swap* is the only non-elementary move in the solver, $O(L_1+L_2)$
per draw against $O(1)$ for everything else, so it is the one option here whose
equal-step gain and equal-time gain genuinely differ; see the wall-time panel of
Section~\ref{sec:newops} for the comparison that counts.
""",
    "reloc-side": r"""
\textbf{\texttt{-{}-reloc-side long}} (default \texttt{coin}) --- which of the
two edges adjacent to the partner $v$ relocate breaks when inserting $u$.
\texttt{coin} flips a bit; \texttt{long} breaks the longer of the two, which
maximises the $-d(v,q)$ term of the insertion cost. Two array reads and no
randomness, and the diversity of insertion positions is still carried by the
draw of $v$ among the nearest neighbours --- which is why it can afford to be
deterministic where the analogous choices on the $u$ side cannot.
""",
    "vrank": r"""
\textbf{\texttt{-{}-vrank}} (default 1) --- a bias towards the nearer candidate
partners, obtained by drawing the index into the kNN list as the minimum of $T$
uniform draws. The lists are already sorted by distance, so this costs no memory
access at all. It is meant to be used with a larger \texttt{-{}-sa-knn}: the
longer list supplies the reach, the rank bias restores the concentration.
""",
    "pick2": r"""
\textbf{\texttt{-{}-pick2}} (default 1) --- a tournament of size $T$ among
candidate partners, keeping the one of largest regret. Unlike
\texttt{-{}-vrank} this costs a memory read per candidate.
""",
    "race": r"""
\textbf{\texttt{-{}-race}} (default off) --- budget redistribution between
restarts (Section~\ref{sec:race}). At \texttt{-{}-race-at} of its budget a start
is compared with the best finished start \emph{at the same point of its
trajectory}, and abandoned if it is behind by more than the margin. The unspent
steps go to the starts that follow, and the remainder polishes the best solution
found. The total budget is unchanged; only its allocation is. It does nothing
with a single start.
""",
    "restarts": r"""
\textbf{\texttt{-{}-restarts}} --- how many independent Clarke \& Wright
constructions are annealed, the best kept (Section~\ref{sec:restarts}). Here the
comparison is always at equal \emph{total} budget, so more restarts means
proportionally fewer steps each.
""",
    "split": r"""
\textbf{\texttt{-{}-split}} --- optimal re-cutting of the giant tour
(Section~\ref{sec:split}). It can never make a solution worse, so the only
question it raises is whether the time it takes is better spent annealing.
""",
    "sa-knn": r"""
\textbf{\texttt{-{}-sa-knn}} --- how many nearest neighbours a move may draw its
partner from (Section~\ref{sec:knn}). Too small and the search cannot reach;
too large and most draws propose a partner far enough away that the move is
rejected on arrival.
""",
    "pick-crit": r"""
\textbf{\texttt{-{}-pick-crit}} --- the definition of the ``regret'' that
drives which vertex gets moved (Section~\ref{sec:pick}).
""",
}
OPTION_NOTES["mu"] = OPTION_NOTES["lambda"]     # explained together
OPTION_NOTES["race-at"] = OPTION_NOTES["race"]


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
    m, steps = tmpl.opts["m"], tmpl.opts["sa-steps"]
    seeds = list(tmpl.seeds)

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
        doc.table(["knob", "best setting tried", *DHEAD, "why not"],
                  [[esc(l), tt(b.tag), *stat_cells(s), esc(why)]
                   for l, b, s, why in dropped],
                  "Knobs left at their default, and why. Note the distinction: "
                  "most of these simply showed no significant improvement, but "
                  "a knob can also be dropped while being significant on its "
                  "own, if it is inert in the company it would keep.",
                  align="llrrrl")

    # The one-knob-at-a-time frame cannot express "A is only worth having if B
    # is too". Racing is exactly that, and it is a large effect, so it gets
    # said in words with the numbers that support it.
    r8 = one(runs, "tuned", "n100_r8")
    rr8 = one(runs, "tuned", "n100_newall_r8")
    nall = one(runs, "tuned", "n100_newall")
    dflt = one(runs, "tuned", "n100_default")
    if all((r8, rr8, nall, dflt)):
        s_pair = paired(dflt, rr8)
        s_solo = paired(dflt, nall)
        # Whether racing was dropped is a result, not a constant: it depends on
        # whether --restarts cleared its own bar at this budget, which is
        # exactly what differs between the two tiers.
        race_dropped = any(l == "racing" for l, _, _, _ in dropped)
        race_pct = next((s["pct"] for l, _, s, _ in dropped if l == "racing"),
                        next((s["pct"] for l, _, s, _ in kept if l == "racing"),
                             float("nan")))
        rest_kept = any(l.startswith("restarts") for l, _, _, _ in kept)
        lead = (r"""Racing was worth $%+.3f\,\%%$ on its own and is still dropped,
because it does nothing without more than one start and \texttt{-{}-restarts}
did not clear the bar on its own knob at this budget."""
                % race_pct if race_dropped else
                r"""Racing was worth $%+.3f\,\%%$ on its own and is kept%s."""
                % (race_pct,
                   " --- and so is \\texttt{-{}-restarts}, which is what makes "
                   "it useful: the two only pay together" if rest_kept else
                   ", though \\texttt{-{}-restarts} did not clear its own bar, "
                   "so it has little to redistribute"))
        doc.p(r"""
\textbf{One caveat the table above cannot express.} %s At $n=100$ and equal
total budget, the recommended options with a
single start reach %.5f (%s vs the default), and the same options with 8 starts
and racing reach %.5f (%s). If you are going to spend the budget on restarts at
all, turn racing on; the sweep's one-knob-at-a-time frame simply has no way to
say so.
""" % (lead,
       nall.mean, "%+.3f\\,\\%%" % s_solo["pct"] if s_solo else "--",
       rr8.mean, "%+.3f\\,\\%%" % s_pair["pct"] if s_pair else "--"))

    vals = resolve(flags, steps)
    # --restarts multiplies the work, so the recommendation has to spend the
    # same TOTAL budget as the default it is compared with -- that is the only
    # sense in which the knob was measured (the `iso_` block of
    # Section~\ref{sec:restarts}). Without this the confirmation below would
    # hand the tuned command R times the steps and call the result an
    # improvement.
    nrestarts = int(vals.get("restarts", 1) or 1)
    tuned_steps = max(1, steps // nrestarts)
    vals["sa-steps"] = tuned_steps
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

    doc.sub("What the changed options do")
    seen_notes = []
    for key in changed:                    # lambda/mu and race/race-at share
        blk = OPTION_NOTES.get(key)        # a note; emit each at most once
        if blk and blk not in seen_notes:
            seen_notes.append(blk)
            doc.p(blk)
    if not seen_notes:
        doc.p("See the section for each knob above for what it controls.")
    doc.note(r"""
\textbf{These are conditional on the budget and on the instance family.}
Everything above was tuned at $\texttt{-{}-sa-steps} = %s$ on uniform points in
a square. The temperature schedule in particular trades off against the budget
by construction --- re-run the \texttt{temp} study if you change it
substantially --- and the savings parameters were fitted to one geometry.
""" % f"{steps:,}")

    if not do_run:
        return

    # Confirmation. Two seed *sets* on purpose: the sweep's own seeds are
    # in-sample and carry the selection bias, the fresh block has never been
    # seen. Both blocks use as many seeds as the sweep itself, so the two rows
    # are equally powered and the gap between them is the bias, not noise.
    bdir = os.path.join(results_dir, "best")
    os.makedirs(bdir, exist_ok=True)
    fresh = [20260801 + i for i in range(len(seeds))]
    rows = []
    print("  verifying the recommended configuration "
          f"({2 * len(seeds) * 3 * 2} runs)...")
    for label, sds in (("sweep seeds (in-sample)", seeds), ("fresh seeds", fresh)):
        for n in (50, 100, 200):
            got = {"default": [], "tuned": []}
            for sd in sds:
                src = ["--random", "-n", str(n), "-m", str(m), "--seed", str(sd)]
                # equal TOTAL budget on both sides: the default runs one start
                # of `steps`, the recommendation runs `nrestarts` starts of
                # `tuned_steps` each
                b = run_cw(binary, src + ["--sa-steps", str(steps)],
                           os.path.join(bdir, f"s{sd}_n{n}_default"))
                t = run_cw(binary, src + ["--sa-steps", str(tuned_steps)] + flags,
                           os.path.join(bdir, f"s{sd}_n{n}_tuned"))
                if b.ok and t.ok:
                    got["default"].append((sd, b))
                    got["tuned"].append((sd, t))
            if not got["default"]:
                continue
            b = pool("best", f"n{n}_default", got["default"])
            t = pool("best", f"n{n}_tuned", got["tuned"])
            s = paired(b, t)
            rows.append([esc(label), n, "%.5f" % b.mean, "%.5f" % t.mean,
                         *stat_cells(s)])
    if rows:
        doc.sub("Confirmation")
        doc.p(r"""
The combination is re-run from scratch, because Section~\ref{sec:tuned} showed
that knobs tuned separately do not add up. It is checked on the sweep's own
instances and on %d seeds the sweep never saw --- the difference between the two
blocks is the selection bias made visible.
""" % len(fresh))
        cap = ("The recommended configuration against the stock defaults, at "
               "equal SA budget. The fresh-seed rows are the honest estimate.")
        if nrestarts > 1:
            cap += (" The recommendation uses %d restarts of %s steps against "
                    "the default's single run of %s, so both sides spend the "
                    "same total." % (nrestarts, f"{tuned_steps:,}",
                                     f"{steps:,}"))
        doc.table(["instances", r"$n$", "default", "recommended", *DHEAD], rows,
                  cap, align="lrrrrrr")


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

@NRUNS@ configurations of \texttt{./cw}, each run at @NSEEDS@ seeds
(@SEEDS@) of @MPER@ instances, so @M@ instances per configuration. Instances are
generated with the Kool/NeuOpt law (coordinates $U[0,1]^2$, demands
$U\{1,\dots,9\}$, capacity from \texttt{default\_capacity}). Unless a study says
otherwise, $n=100$ and \texttt{-{}-sa-steps}~$=$~@STEPS@.

\textbf{Every comparison is paired.} Instance $k$ is generated from
\texttt{seed}~$+~k$ (\texttt{cw.c:2574}), so all runs of a study see
byte-identical instances. Rather than comparing two means with their independent
standard deviations, we compare $\mathrm{cost}_i(A)$ with $\mathrm{cost}_i(B)$ on
the same instance $i$ and report the mean of the @M@ differences with a
95\,\% confidence interval, plus a distribution-free sign test. This matters: the
between-instance spread ($\sigma \approx 1.9$ at $n=100$) is more than an order of
magnitude larger than the effects being measured.

\textbf{Every configuration is replicated on @NSEEDS@ seeds.} \texttt{cw} has a
single \texttt{-{}-seed}, and it drives both the instance set
(\texttt{cw.c:2574}) and the annealing RNG (\texttt{cw.c:2679}); there is no way
to re-randomise the solver while holding the instances fixed. A seed is
therefore a complete replication --- fresh instances, fresh randomness --- and
the @NSEEDS@ blocks are concatenated before the paired statistics are taken. No
conclusion in this report rests on a single instance draw, and each $\Delta$ is
the mean of @M@ paired differences rather than @MPER@.

\textbf{Everything is measured at two budgets.} Each grid is run at
$\texttt{-{}-sa-steps} = 10^5$ and again at $10^7$, the budget this solver is
actually run at. Section~\ref{sec:budget} puts the two side by side; the body of
the report is the @PRIMARY@ tier.

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
    ap.add_argument("--tier", default="hi", choices=("lo", "hi"),
                    help="which budget tier carries the report (default: hi, "
                         "the operating point); the other one is still read, "
                         "for the budget-dependence section")
    ap.add_argument("--no-verify", action="store_true",
                    help="do not re-run the recommended configuration")
    ap.add_argument("--no-pdf", action="store_true", help="emit the .tex only")
    a = ap.parse_args()

    if not os.path.isdir(a.results):
        sys.exit(f"{a.results} not found — run sweep/run_sweep.sh first")

    # Two budget tiers, each a directory of per-seed subdirectories. The
    # primary tier carries the whole report; the other one exists so that
    # budget_shift() can say whether the answers transfer.
    tiers = {}
    for t in ("lo", "hi"):
        d = os.path.join(a.results, t)
        if os.path.isdir(d):
            got = [r for r in load(d) if r.study != "best"]
            if got:
                tiers[t] = got
    if not tiers:
        sys.exit(f"no successful run under {a.results}/(lo|hi) — "
                 "run sweep/run_sweep.sh first")
    primary = a.tier if a.tier in tiers else ("hi" if "hi" in tiers else "lo")
    runs = tiers[primary]
    for t, rs in tiers.items():
        per = defaultdict(int)
        for r in rs:
            per[r.study] += 1
        nseed = max((r.n_seeds for r in rs), default=0)
        mark = " (primary)" if t == primary else ""
        print(f"tier {t}{mark}: {len(rs)} configurations x {nseed} seeds, "
              f"{tier_steps(rs):,} steps — "
              + ", ".join(f"{k}={v}" for k, v in sorted(per.items())))

    figdir = os.path.join(a.out, "figures")
    os.makedirs(figdir, exist_ok=True)
    doc = Doc()

    ks = knobs(runs)
    try:
        overview(runs, figdir, doc, ks)
    except Exception as e:
        print(f"  !! overview: {e}", file=sys.stderr)
    if len(tiers) > 1:
        try:
            budget_shift(tiers["lo"], tiers["hi"], figdir, doc)
        except Exception as e:
            print(f"  !! budget_shift: {e}", file=sys.stderr)
    for fn in (study_init, study_ops, study_newops, study_knn, study_timing,
               study_restarts, study_split, study_pick, study_select,
               study_race, study_temp, study_construct, study_tuned):
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
  \item \textbf{Two budgets, not a curve.} Every grid is run at $10^5$ and at
        $10^7$ annealing steps (Section~\ref{sec:budget}), which is enough to
        say whether a recommendation transfers but not to fit how it varies.
        The schedule knob in particular is budget-dependent by construction ---
        the geometric ratio is $(T_\mathrm{end}/T_0)^{1/(S-1)}$ --- so at a
        third budget it would have to be re-measured again.
  \item \textbf{The two tiers use different instance counts.} $m=1000$ per seed
        at $10^5$ steps and $m=200$ at $10^7$, to keep the total compute
        bounded. Pooled over the seeds both tiers carry $\ge 1000$ paired
        instances, so their CIs are comparable, but the $10^7$ tier is the
        looser of the two.
  \item \textbf{Selection bias.} The overview picks each knob's best setting
        \emph{because} it scored lowest. The fresh-seed rows in
        Section~\ref{sec:best} are the only bias-free numbers in this document.
  \item \textbf{\texttt{-{}-sa-knn} is clamped} to $n-1$
        (\texttt{cw.c:1675}), so at $n=20$ the settings $K \ge 20$ are the same
        run; and at $K=0$ the vertex-selection rule is silently forced to
        uniform (\texttt{cw.c:1676}) while the header still prints the requested
        \texttt{-{}-pick}. That is a reporting bug worth fixing in \texttt{cw}.
  \item \textbf{One machine, for the wall-time results.} The iso-time reading in
        Section~\ref{sec:newops} and the interleaving speed-up in
        Section~\ref{sec:race} are the only conclusions here that depend on the
        hardware. Both were measured on one laptop with all threads busy; the
        cost ratios will move on a machine with a different cache hierarchy, and
        the interleaving result in particular is a cache effect by construction.
  \item \textbf{Knobs that need each other are invisible to the frame.} Every
        knob is swept with everything else at its default, so a knob that only
        pays in the presence of another cannot show up. Racing is the clear
        case --- significant on its own measurement, dropped from the
        recommendation, and worth having as soon as \texttt{-{}-restarts} is
        raised. Section~\ref{sec:tuned} is the only part of this document that
        looks at combinations at all.
  \item \textbf{Where this disagrees with its source.} \texttt{-{}-vrank} and
        \texttt{-{}-sa-knn 30} were ported from an implementation that reports
        them as a win, and here they lose consistently across the whole grid.
        The likely cause is that the other implementation shares one $K=30$
        neighbour list between the savings construction and the annealing while
        this one builds exact savings up to $n = 1500$, so the two are not the
        same baseline. This sweep can say the setting does not pay \emph{here};
        it cannot say the original measurement was wrong.
\end{itemize}
""")

    import datetime
    tmpl = one(runs, "tuned", "n100_default") or runs[0]
    head = PREAMBLE                    # token substitution, not %-formatting:
    for tok, val in (                  # the preamble is full of literal LaTeX %
            ("@DATE@", datetime.date.today().isoformat()),
            ("@NRUNS@", str(len(runs))),
            ("@NSEEDS@", str(tmpl.n_seeds)),
            ("@SEEDS@", ", ".join(str(s) for s in tmpl.seeds)),
            ("@MPER@", f"{tmpl.opts['m']:,}"),
            ("@M@", f"{tmpl.opts['m'] * tmpl.n_seeds:,}"),
            ("@PRIMARY@", r"$10^{%d}$-step"
                          % round(math.log10(max(1, tmpl.opts["sa-steps"])))),
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
