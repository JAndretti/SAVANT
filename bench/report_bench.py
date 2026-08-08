#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""report_bench.py — the tables, from what bench/ left in results/bench/.

Writes bench/report.tex (a compilable standalone document) and, with
`--format md`, bench/report.md. Both come from the same table objects, so the
two never disagree.

  1. the generated sets (n20, n50, n100): mean / min / max cost, SAVANT
     against HGS-CVRP at the same per-instance budget;
  2. XML100: the same plus the optimality gap, then a breakdown by the four
     generator attributes in the layout of Table 4 of the XL paper
     (arXiv:2601.11467, Queiroga et al.), one column per solver;
  3. X and XL: one row per instance -- every solver's cost, gap to the
     reference, and time, plus the budget --sa-steps N gave SAVANT.

Every solver's *achieved* time is reported next to its cost, because at these
budgets none of them is matched: process launch, a dense distance matrix, a
candidate set or a JVM all sit outside the time flag. Read the time column
before the cost column.

The XML100 attribute decoding
-----------------------------
An instance is named XML100_<depot><customers><demand><route>_<rep>, four
digits. The mapping below was *measured* from the .vrp files rather than read
off the paper, because getting it silently wrong would put every row of the
breakdown in the wrong bucket while still looking plausible:

  depot     digit 1 -> depot coordinates: 2 = (500,500) central,
                       3 = (0,0) eccentric, 1 = neither, i.e. random
  customers digit 2 -> mean nearest-neighbour distance over 5 instances each:
                       1 = 52.8 (random), 2 = 31.2 (clustered),
                       3 = 47.1 (random-clustered)
  demand    digit 3 -> observed support: 1 = {1}, 2 = [1,10], 3 = [5,10],
                       4 = [1,100], 5 = [51,100], 6 = quadrant-dependent,
                       7 = mostly [1,10] with a few [50,100]
  route     digit 4 -> r = n / (total demand / capacity), 8 instances each:
                       3..4, 5..7, 8..11, 12..15, 16..23, 25..46, i.e.
                       U[3,5] U[5,8] U[8,12] U[12,16] U[16,25] U[25,50]

`--verify` re-derives the depot, demand and route-size mappings and fails if
they have moved.

Usage:
    uv run bench/report_bench.py
    uv run bench/report_bench.py --format md
    uv run bench/report_bench.py --verify
    uv run bench/report_bench.py --no-pdf
"""

import argparse
import csv
import datetime
import glob
import json
import os
import re
import shutil
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(ROOT, "results", "bench")
XMLDIR = os.path.join(ROOT, "data", "cvrplib", "XML100")

ORDER = ["n20", "n50", "n100", "XML100", "X", "XL"]
SOLVERS = ["SAVANT", "HGS", "LKH", "AILS"]
BKS_OF = {"XML100": "data/cvrplib/XML100_bks.csv",
          "X": "data/cvrplib/X_bks.csv",
          "XL": "baseline/xl_bks.csv"}

DEPOT = {"1": "R", "2": "C", "3": "E"}
CUST = {"1": "R", "2": "C", "3": "RC"}
DEMAND = {"1": "U", "2": "1--10", "3": "5--10", "4": "1--100", "5": "50--100",
          "6": "Q", "7": "SL"}
ROUTE = {"1": "VS", "2": "S", "3": "M", "4": "L", "5": "VL", "6": "UL"}
ATTRS = [("Depot pos.", 0, DEPOT), ("Customer pos.", 1, CUST),
         ("Demand dist.", 2, DEMAND), ("Route size $r$", 3, ROUTE)]


# ------------------------------------------------------------------- loading


def stem(name):
    """`X-n101-k25.vrp` -> `X-n101-k25`: --dir keeps the file name."""
    return name[:-4] if name.endswith(".vrp") else name


class Run:
    """One solver on one set: per-instance costs and times, plus the totals."""

    def __init__(self, path, cfg, rows):
        self.dir, self.cfg = path, cfg
        self.rows = rows
        self.cost = {r["name"]: r["cost"] for r in rows}
        self.ms = {r["name"]: r["ms"] for r in rows}

    @property
    def m(self):
        return len(self.rows)

    @property
    def ms_per_inst(self):
        """Achieved single-core ms per instance, from the driver's own total.

        Preferred over the mean of the per-instance column: the drivers measure
        process CPU, which includes the set-up the time flag does not cover,
        and that is the number the comparison turns on.
        """
        res = self.cfg.get("result", {}) or {}
        if res.get("cpu_s_per_instance"):
            return 1e3 * res["cpu_s_per_instance"]
        cpu, m = res.get("cpu_s") or res.get("time_cpu_s"), res.get("instances")
        if cpu and m:
            return 1e3 * cpu / m
        return statistics.mean(self.ms.values()) if self.ms else None

    @property
    def steps(self):
        """Per-instance --sa-steps, from solutions.txt (SAVANT runs only)."""
        try:
            with open(os.path.join(self.dir, "solutions.txt"), encoding="utf-8") as f:
                for line in f:
                    if line.startswith("#sa-steps"):
                        return dict(zip([r["name"] for r in self.rows],
                                        (int(v) for v in line.split()[1:])))
                    if not line.startswith("#"):
                        break
        except (OSError, ValueError):
            pass
        return {}

    @property
    def valid(self):
        try:
            with open(os.path.join(self.dir, "validation.txt"), encoding="utf-8") as f:
                m = re.search(r"(\d+) instance\(s\) checked, (\d+) error\(s\)",
                              f.read())
            if not m:
                return "?"
            return "ok" if m.group(2) == "0" else f"{m.group(2)} ERRORS"
        except OSError:
            return "--"


def load(runs_root, tag, solver):
    """The newest run of one (set, solver), or None."""
    pat = f"*_{tag}" if solver == "SAVANT" else f"*_{solver}_{tag}"
    dirs = sorted(d for d in glob.glob(os.path.join(runs_root, pat))
                  if os.path.isdir(d))
    if solver == "SAVANT":                      # *_X must not catch *_HGS_X
        dirs = [d for d in dirs
                if os.path.basename(d).split("_", 1)[-1] == tag]
    if not dirs:
        return None
    d = dirs[-1]
    try:
        with open(os.path.join(d, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        with open(os.path.join(d, "results.csv"), newline="", encoding="utf-8") as f:
            raw = list(csv.DictReader(f))
    except OSError:
        return None
    # the four drivers do not agree on column names: cw writes
    # instance/cost_annealed/time_ms, the reference drivers name/cost/wall_s
    rows = []
    for r in raw:
        try:
            name = stem(r.get("instance") or r["name"])
            cost = float(r["cost_annealed"] if r.get("cost_annealed") else r["cost"])
            ms = (float(r["time_ms"]) if r.get("time_ms")
                  else 1e3 * float(r["wall_s"]) if r.get("wall_s") else 0.0)
            rows.append({"name": name, "n": int(r["n"]), "cost": cost, "ms": ms,
                         "routes": int(r.get("routes") or 0)})
        except (KeyError, ValueError):
            continue
    # a driver records the instances it could not solve; keeping a zero cost
    # would silently make the solver look perfect on them
    rows = [r for r in rows if r["cost"] > 0]
    return Run(d, cfg, rows) if rows else None


def pair(a, b):
    """Instances shared by two runs, as a list of (row_a, row_b).

    The four drivers do not name instances the same way: cw over a bundle
    writes `cvrp_20.cvrpb#7`, run_hgs.py writes `cvrp_20_0007`, and only the
    CVRPLib sets (read with --dir) carry the real name everywhere. So pairing
    is by name when the two runs share names at all, and by position
    otherwise -- both drivers walk the set in the same sorted order.

    Either way `n` is checked on every pair. A mispairing that survived that
    would have to line up size for size across the whole set, and it is the
    failure that would otherwise be invisible: two internally consistent runs
    averaged against each other in the wrong order.
    """
    if set(x["name"] for x in a.rows) & set(x["name"] for x in b.rows):
        bi = {x["name"]: x for x in b.rows}
        out = [(x, bi[x["name"]]) for x in a.rows if x["name"] in bi]
    else:
        out = list(zip(a.rows, b.rows))
    bad = [(x, y) for x, y in out if x["n"] != y["n"]]
    if bad:
        print(f"  !! {len(bad)} pair(s) disagree on n between "
              f"{os.path.basename(a.dir)} and {os.path.basename(b.dir)} "
              f"— pairing dropped", file=sys.stderr)
        return []
    return out


def read_bks(path):
    """{name: cost} from a reference CSV, whichever column holds it."""
    out = {}
    try:
        with open(os.path.join(ROOT, path), newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                col = next((c for c in ("bks_cost", "final_bks", "cost", "opt")
                            if c in r and r[c]), None)
                if col:
                    out[stem(r["name"])] = float(r[col])
    except OSError:
        pass
    return out


# ------------------------------------------------------------- tiny document


class Table:
    """Header rows, body rows, column alignment, caption. Rendered twice."""

    def __init__(self, header, rows, align, caption, label=None, long=False,
                 groups=None):
        self.header, self.rows, self.align = header, rows, align
        self.caption, self.label, self.long = caption, label, long
        self.groups = groups            # [(title, span), ...] above the header


# Table cells and captions are author-written LaTeX: they are emitted as is,
# because half of them are deliberately \textbf{...}, $\times$ or 8.7\%, and
# escaping those a second time is what turns "\%" into "\{\}%". The only
# untrusted text in the document is the instance names, which come from the
# file system; tex_name() is the one place that escapes.
def tex_name(s):
    return str(s).replace("\\", "/").replace("_", r"\_").replace("%", r"\%")


def esc_md(s):
    """LaTeX -> something readable in the markdown rendering of the same table."""
    s = str(s)
    s = re.sub(r"\\textbf\{(.+?)\}", r"**\1**", s)
    s = re.sub(r"\\emph\{(.+?)\}", r"*\1*", s)
    s = re.sub(r"\\texttt\{(.+?)\}", r"`\1`", s)
    s = (s.replace(r"\times", "x").replace(r"\%", "%").replace(r"\_", "_")
          .replace(r"\&", "&").replace(r"\,", " ").replace("---", "—")
          .replace("$", "").replace("{", "").replace("}", ""))
    return s


class Doc:
    def __init__(self, fmt):
        self.fmt, self.body = fmt, []

    def h(self, level, text):
        if self.fmt == "md":
            self.body.append("#" * level + " " + esc_md(text) + "\n")
        else:
            cmd = {1: "section", 2: "subsection", 3: "subsubsection"}[level]
            self.body.append("\\%s{%s}\n" % (cmd, text))

    def p(self, text):
        self.body.append((esc_md(text) if self.fmt == "md" else text) + "\n")

    def table(self, t):
        self.body.append(self._md(t) if self.fmt == "md" else self._tex(t))

    def _md(self, t):
        out = ["| " + " | ".join(esc_md(h) for h in t.header) + " |",
               "|" + "|".join({"l": "---", "r": "---:", "c": ":---:"}[a]
                              for a in t.align) + "|"]
        out += ["| " + " | ".join(esc_md(c) for c in r) + " |" for r in t.rows]
        return "\n".join(out) + "\n\n*" + esc_md(t.caption) + "*\n"

    def _tex(self, t):
        head = ""
        if t.groups:
            cells, rules, col = [], [], 1
            for title, span in t.groups:
                cells.append("\\multicolumn{%d}{c}{%s}" % (span, title)
                             if span > 1 else (title or ""))
                if span > 1 and title:
                    rules.append("\\cmidrule(lr){%d-%d}" % (col, col + span - 1))
                col += span
            head += " & ".join(cells) + " \\\\\n" + "".join(rules) + "\n"
        head += " & ".join("\\textbf{%s}" % h for h in t.header) + " \\\\"
        body = "\n".join("  " + " & ".join(str(c) for c in r) + " \\\\"
                         for r in t.rows)
        cols = "".join(t.align)
        lab = "\\label{%s}" % t.label if t.label else ""
        if t.long:
            return ("{\\scriptsize\\begin{longtable}{%s}\n"
                    "\\caption{%s}%s\\\\\n\\toprule\n%s\n\\midrule\n"
                    "\\endfirsthead\n\\toprule\n%s\n\\midrule\n\\endhead\n"
                    "%s\n\\bottomrule\n\\end{longtable}}\n"
                    % (cols, t.caption, lab, head, head, body))
        size = "\\scriptsize" if len(t.header) > 8 else "\\small"
        return ("\\begin{table}[H]\\centering%s\n\\begin{tabular}{%s}\n"
                "\\toprule\n%s\n\\midrule\n%s\n\\bottomrule\n\\end{tabular}\n"
                "\\caption{%s}%s\n\\end{table}\n"
                % (size, cols, head, body, t.caption, lab))

    def render(self, title, sub):
        if self.fmt == "md":
            return f"# {title}\n\n{esc_md(sub)}\n\n" + "\n".join(self.body)
        return (PREAMBLE % (title, sub)) + "\n".join(self.body) + "\n\\end{document}\n"


PREAMBLE = r"""%% generated by bench/report_bench.py -- do not edit by hand
\documentclass[10pt]{article}
\usepackage[a4paper,margin=15mm]{geometry}
\usepackage{booktabs,longtable,float,pdflscape,amsmath,microtype}
\usepackage[T1]{fontenc}
\usepackage[hidelinks]{hyperref}
\setlength{\tabcolsep}{4pt}
\title{%s}
\date{}
\begin{document}
\maketitle
\noindent %s

"""


# ------------------------------------------------------------------ helpers


# A solver that returned a small share of the set must not be allowed to define
# the subset everyone else is scored on: LKH-3 came back with 18 of the 100 XL
# instances, all of them small, and intersecting on those would report SAVANT
# and AILS-II on the easy tail of the set and call it the comparison.
MIN_COVERAGE = 0.5


def common_keys(R, tag, have, universe):
    """(keys every well-covered solver returned, solvers excluded for coverage)."""
    keep, drop = [], []
    for sv in have:
        cov = len(set(R[(sv, tag)].cost) & set(universe)) / max(1, len(universe))
        (keep if cov >= MIN_COVERAGE else drop).append(sv)
    keys = set(universe)
    for sv in keep:
        keys &= set(R[(sv, tag)].cost)
    return sorted(keys), drop


def gap(cost, ref):
    return 100.0 * (cost - ref) / ref if ref else None


def fmt(v, spec, dash="--"):
    return format(v, spec) if v is not None else dash


def attrs(name):
    m = re.fullmatch(r"XML100_(\d)(\d)(\d)(\d)_(\d+)", name)
    return m.groups()[:4] if m else None


def parse_vrp(p):
    xy, dem, cap, sec = {}, {}, None, None
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line.startswith("CAPACITY"):
            cap = float(line.split(":")[1])
        elif line.startswith("NODE_COORD"):
            sec = "c"
        elif line.startswith("DEMAND"):
            sec = "d"
        elif line.startswith(("DEPOT", "EOF")):
            sec = None
        elif sec and line[:1].isdigit():
            a = line.split()
            (xy if sec == "c" else dem)[int(a[0])] = (
                (float(a[1]), float(a[2])) if sec == "c" else float(a[1]))
    return xy, dem, cap


def verify_xml(limit=8):
    bad = []
    for d, want in (("2", (500.0, 500.0)), ("3", (0.0, 0.0))):
        p = os.path.join(XMLDIR, f"XML100_{d}111_01.vrp")
        if os.path.exists(p):
            got = parse_vrp(p)[0][1]
            if got != want:
                bad.append(f"depot digit {d}: expected {want}, got {got}")
    for d, lo, hi in (("1", 1, 1), ("2", 1, 10), ("3", 5, 10), ("5", 51, 100)):
        p = os.path.join(XMLDIR, f"XML100_11{d}1_01.vrp")
        if not os.path.exists(p):
            continue
        v = [x for i, x in parse_vrp(p)[1].items() if i != 1]
        if not (min(v) >= lo and max(v) <= hi):
            bad.append(f"demand digit {d}: expected [{lo},{hi}], "
                       f"got [{min(v):.0f},{max(v):.0f}]")
    for d, (lo, hi) in {"1": (3, 5), "2": (5, 8), "3": (8, 12), "4": (12, 16),
                        "5": (16, 25), "6": (25, 50)}.items():
        rs = []
        for rep in range(1, limit + 1):
            p = os.path.join(XMLDIR, f"XML100_111{d}_{rep:02d}.vrp")
            if os.path.exists(p):
                _, dem, cap = parse_vrp(p)
                v = [x for i, x in dem.items() if i != 1]
                rs.append(len(v) / (sum(v) / cap))
        if rs and not (lo - 0.5 <= min(rs) and max(rs) <= hi + 0.5):
            bad.append(f"route digit {d}: expected U[{lo},{hi}], "
                       f"got {min(rs):.1f}..{max(rs):.1f}")
    return bad


# ------------------------------------------------------------------ sections


def sec_generated(R, doc):
    doc.h(1, "Generated instances")
    doc.p("10,000 instances per set, uniform points in the unit square, "
          "\\emph{continuous} distances. No reference solutions ship with them, "
          "so the cost stands on its own. LKH-3 has no float mode --- EUC\\_2D "
          "is rounded by TSPLIB rules --- so it cannot be run here without "
          "changing the problem, and AILS-II was not asked for on these sets. "
          "\\texttt{min} and \\texttt{max} are over the \\emph{instances}, not "
          "over repeated runs of one instance: they describe the spread of the "
          "set.")
    rows = []
    for tag in ("n20", "n50", "n100"):
        sav = R.get(("SAVANT", tag))
        if not sav:
            continue
        for solver in ("SAVANT", "HGS"):
            run = R.get((solver, tag))
            if not run:
                continue
            pr = pair(run, sav)
            if not pr:
                continue
            c = [x["cost"] for x, _ in pr]
            base = statistics.mean(y["cost"] for _, y in pr)
            d = [x["cost"] - y["cost"] for x, y in pr]
            steps = sav.steps
            rows.append([
                tag if solver == "SAVANT" else "",
                solver, f"{len(c):,}",
                f"{statistics.mean(c):.5f}", f"{min(c):.5f}", f"{max(c):.5f}",
                f"{statistics.stdev(c):.5f}" if len(c) > 1 else "--",
                "--" if solver == "SAVANT"
                else f"{100 * statistics.mean(d) / base:+.3f}",
                fmt(run.ms_per_inst, ".1f"),
                fmt(run.ms_per_inst / sav.ms_per_inst, ".2f"),
                f"{next(iter(steps.values())):,}" if solver == "SAVANT" and steps else "--",
                run.valid])
    doc.table(Table(
        ["set", "solver", "$m$", "mean cost", "min", "max", "sd",
         "vs SAVANT \\%", "CPU ms/inst", "$\\times$ asked", "--sa-steps", "valid"],
        rows, list("llrrrrrrrrrl"),
        "Generated sets, at SAVANT's own per-instance budget. "
        "`vs SAVANT \\%' is the paired mean difference as a percentage of "
        "SAVANT's mean; negative means the other solver is cheaper. "
        "The CPU column is what each solver actually spent.",
        "tab:gen"))


def sec_xml(R, doc):
    tag = "XML100"
    sav = R.get(("SAVANT", tag))
    if not sav:
        return
    bks = read_bks(BKS_OF[tag])
    doc.h(1, "XML100")
    doc.p("10,000 instances of exactly 100 customers, \\emph{integer} "
          "distances (TSPLIB EUC\\_2D). Every one has a \\emph{proven optimum}, "
          "so the gap is a true optimality gap rather than a bound.")
    have = [s for s in SOLVERS if (s, tag) in R]
    common, thin = common_keys(R, tag, have, sorted(bks))
    rows = []
    for solver in have:
        run = R[(solver, tag)]
        keys = ([k for k in common if k in run.cost] if solver in thin
                else common)
        c = [run.cost[k] for k in keys]
        g = [gap(run.cost[k], bks[k]) for k in keys]
        opt = sum(1 for v in g if v <= 1e-9)
        rows.append([solver, f"{len(run.cost):,}", f"{statistics.mean(c):.2f}",
                     f"{min(c):.0f}", f"{max(c):.0f}",
                     f"{statistics.stdev(c):.2f}" if len(c) > 1 else "--",
                     f"{statistics.mean(g):+.3f}", f"{statistics.median(g):+.3f}",
                     f"{max(g):+.3f}",
                     f"{opt:,} ({100 * opt / max(1, len(g)):.1f}\\%)",
                     fmt(run.ms_per_inst, ".1f"),
                     fmt(run.ms_per_inst / sav.ms_per_inst, ".2f"), run.valid])
    doc.table(Table(
        ["solver", "returned", "mean cost", "min", "max", "sd",
         "mean gap \\%", "median", "worst", "at optimum", "CPU ms/inst",
         "$\\times$ asked", "valid"],
        rows, list("lrrrrrrrrrrrl"),
        f"XML100 against the proven optima, at SAVANT's per-instance budget. "
        f"`returned' is how many of the 10,000 that solver produced a feasible "
        f"answer for; every other column is computed on the "
        f"{len(common):,} instances \\emph{{all}} of them returned, so the "
        f"rows are comparable. Averaging each solver over its own subset would "
        f"reward whichever one failed on the hardest instances."
        + (f" {', '.join(thin)} returned under {100 * MIN_COVERAGE:.0f}\\% and "
           f"is excluded from that intersection." if thin else ""),
        "tab:xml"))

    # ---- by attribute, in the layout of Table 4 of arXiv:2601.11467
    doc.h(2, "By generator attribute")
    doc.p("The layout of Table~4 of the XL paper (arXiv:2601.11467): the mean "
          "optimality gap of every subgroup defined by one attribute value. "
          "The four blocks are four different partitions of the same 10,000 "
          "instances, so each one averages back to the overall row. The "
          "decoding of the instance name's four digits was measured from the "
          "\\texttt{.vrp} files, not assumed; "
          "\\texttt{report\\_bench.py -{}-verify} re-checks it.")
    per = {}
    for s in have:
        run = R[(s, tag)]
        per[s] = {k: gap(run.cost[k], bks[k]) for k in common}
    rows = []
    for label, idx, levels in ATTRS:
        first = True
        for code, name in levels.items():
            sel = [k for k in per[have[0]] if (attrs(k) or "----")[idx] == code]
            if not sel:
                continue
            row = [label if first else "", name, f"{len(sel):,}"]
            for s in have:
                g = [per[s][k] for k in sel if k in per[s]]
                row += [f"{statistics.mean(g):+.3f}" if g else "--",
                        f"{100 * sum(1 for v in g if v <= 1e-9) / len(g):.1f}"
                        if g else "--"]
            rows.append(row)
            first = False
    row = ["\\textbf{Overall}", "", f"{len(per[have[0]]):,}"]
    for s in have:
        g = list(per[s].values())
        row += [f"\\textbf{{{statistics.mean(g):+.3f}}}",
                f"{100 * sum(1 for v in g if v <= 1e-9) / len(g):.1f}"]
    rows.append(row)
    doc.table(Table(
        ["attribute", "level", "$m$"] + sum([["gap \\%", "opt \\%"] for _ in have], []),
        rows, list("llr" + "rr" * len(have)),
        "Mean optimality gap by generator attribute, and the share of the "
        "subgroup solved to optimality. Levels: depot Random / Central / "
        "Eccentric; customers Random / Clustered / Random-Clustered; demand "
        "Unitary, four sampling ranges, Quadrant-dependent, Small-with-a-few-"
        "Large; route size Very Short (3--5) through Ultra Long (25--50).",
        "tab:xmlattr", groups=[("", 3)] + [(s, 2) for s in have]))


def sec_instances(R, tag, doc, note):
    sav = R.get(("SAVANT", tag))
    if not sav:
        return
    bks = read_bks(BKS_OF[tag])
    have = [s for s in SOLVERS if (s, tag) in R]
    steps = sav.steps
    doc.h(1, f"{tag}, instance by instance")
    doc.p(note)

    common, thin = common_keys(R, tag, have, sorted(bks))
    summary = []
    for s in have:
        run = R[(s, tag)]
        keys = [k for k in common if k in run.cost] if s in thin else common
        g = [gap(run.cost[k], bks[k]) for k in keys]
        own = [gap(run.cost[k], bks[k]) for k in run.cost if k in bks]
        summary.append([s + ("$^{\\dagger}$" if s in thin else ""),
                        f"{len(run.cost)}", f"{statistics.mean(g):+.3f}",
                        f"{statistics.median(g):+.3f}", f"{min(g):+.3f}",
                        f"{max(g):+.3f}",
                        f"{statistics.mean(own):+.3f}" if own else "--",
                        fmt(run.ms_per_inst / 1e3, ".2f"),
                        fmt(run.ms_per_inst / sav.ms_per_inst, ".2f"),
                        run.valid])
    doc.table(Table(
        ["solver", "returned", "mean gap \\%", "median", "best", "worst",
         "own subset", "CPU s/inst", "$\\times$ asked", "valid"],
        summary, list("lrrrrrrrrl"),
        f"{tag}: the first block is computed on the {len(common)} instances "
        f"\\emph{{every}} solver returned, so the rows are comparable. "
        f"`own subset' is the same mean over whatever that solver alone "
        f"managed --- the difference between the two columns is how much a "
        f"solver is flattered by the instances it failed on. "
        f"`$\\times$ asked' is achieved CPU over the budget requested: read it "
        f"before the gap columns, because a solver several times over budget is "
        f"not being compared at equal cost."
        + (f" $^{{\\dagger}}$ returned under {100 * MIN_COVERAGE:.0f}\\% of "
           f"the set, so it does not define the common subset --- its row is "
           f"its own instances only and is not comparable with the others."
           if thin else ""),
        f"tab:{tag}sum"))

    order = sorted(sav.cost, key=lambda k: (sav.rows[0] and 0,
                                            [r["n"] for r in sav.rows
                                             if r["name"] == k][0], k))
    nof = {r["name"]: r["n"] for r in sav.rows}
    rows = []
    for k in order:
        row = [tex_name(k), f"{nof[k]:,}",
               f"{bks.get(k, 0):,.0f}" if k in bks else "--"]
        for s in have:
            run = R[(s, tag)]
            c = run.cost.get(k)
            row += [f"{c:,.0f}" if c is not None else "--",
                    fmt(gap(c, bks.get(k)) if c is not None else None, "+.2f"),
                    fmt(run.ms.get(k, None) and run.ms[k] / 1e3, ".2f")]
        row.append(f"{steps.get(k, 0):,}" if steps.get(k) else "--")
        rows.append(row)
    doc.table(Table(
        ["instance", "$n$", "BKS"]
        + sum([["cost", "gap \\%", "s"] for _ in have], []) + ["--sa-steps"],
        rows, list("lrr" + "rrr" * len(have) + "r"),
        f"{tag}, one row per instance. `s' is that solver's own measured time "
        f"for that instance, single-core. The last column is the budget "
        f"\\texttt{{-{{}}-sa-steps N}} gave SAVANT, i.e. "
        f"$\\max(500000,\\,668\\,n^{{1.37}})$.",
        f"tab:{tag}", long=True,
        groups=[("", 3)] + [(s, 3) for s in have] + [("", 1)]))


# --------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--format", choices=("tex", "md"), default="tex")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    if a.verify:
        bad = verify_xml()
        for b in bad:
            print("!! " + b, file=sys.stderr)
        print("XML100 attribute mapping: "
              + ("OK, re-derived from the .vrp files" if not bad
                 else f"{len(bad)} DISAGREEMENT(S)"))
        return 1 if bad else 0

    R = {}
    for tag in ORDER:
        for s in SOLVERS:
            r = load(a.runs, tag, s)
            if r:
                R[(s, tag)] = r
    if not R:
        raise SystemExit(f"{a.runs}: no run — run bench/run_bench.py first")
    print("### loaded: " + ", ".join(f"{s}/{t}" for s, t in sorted(R)))

    doc = Doc(a.format)
    env = R[("SAVANT", ORDER[0])].cfg.get("environment", {}) if ("SAVANT", ORDER[0]) in R else {}
    sub = (
        "SAVANT at its 2026-08-07 defaults, budgeted with "
        "\\texttt{-{}-sa-steps N} so each instance's budget comes from its own "
        "dimension, $\\max(500000,\\,668\\,n^{1.37})$. The reference solvers "
        "were then given \\emph{SAVANT's own measured per-instance CPU time} on "
        "each set. None of them is matched at these budgets and none is "
        "expected to be: a process launch is a few milliseconds, HGS-CVRP "
        "builds a dense $n\\times n$ matrix, LKH-3 generates a candidate set "
        "and AILS-II starts a JVM, all outside the time flag. Every solver's "
        "\\emph{achieved} time is therefore printed next to its cost, and "
        "should be read first. Every run was checked by "
        "\\texttt{tools/validate.py}, which recomputes each solution from the "
        "coordinates independently of the solver that produced it.\\\\[2mm]"
        f"\\small {tex_name(env.get('platform', '?'))}, "
        f"{env.get('cpu_count', '?')} cores. "
        "The 10,000-instance sets ran on every core; X and XL ran "
        "single-threaded, because their tables report a per-instance time and "
        "\\texttt{time\\_ms} is measured inside SAVANT's parallel region, where "
        "it would otherwise absorb the contention between threads. "
        f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M}."
    )
    if a.format == "md":
        sub = re.sub(r"\\\\\[2mm\]|\\small |\\emph|\\texttt|\\max|\\times", "", sub)

    sec_generated(R, doc)
    sec_xml(R, doc)
    sec_instances(R, "X", doc,
                  "Integer distances. 61 of the 100 references are proven "
                  "optima and the rest are best known solutions, so the gap "
                  "bounds the optimality gap from above.")
    sec_instances(R, "XL", doc,
                  "Integer distances. The references are the \\emph{final} "
                  "BKSs of the CVRPLib BKS Challenge (arXiv:2601.11467, "
                  "Table~1), not the pre-challenge \\texttt{.sol} files --- "
                  "scoring against those would understate every gap. They were "
                  "produced by months of dedicated large-scale solvers (one "
                  "team reports about 117 CPU-years); a sub-minute run is not "
                  "in that regime, and the gap column is here to show where "
                  "the budget lands, not to claim parity.")

    out = a.out or os.path.join(HERE, f"report.{a.format}")
    with open(out, "w", encoding="utf-8") as f:
        f.write(doc.render("SAVANT on every dataset", sub))
    print(f"  report  {os.path.relpath(out, ROOT)}")

    if a.format == "tex" and not a.no_pdf and shutil.which("pdflatex"):
        for _ in range(2):
            p = subprocess.run(["pdflatex", "-interaction=nonstopmode",
                                "-halt-on-error", os.path.basename(out)],
                               cwd=HERE, capture_output=True, text=True)
        if p.returncode:
            tail = "\n".join(p.stdout.splitlines()[-25:])
            print(f"  !! pdflatex failed:\n{tail}", file=sys.stderr)
        else:
            for ext in (".aux", ".log", ".out", ".toc"):
                try:
                    os.remove(out[:-4] + ext)
                except OSError:
                    pass
            print(f"  pdf     {os.path.relpath(out[:-4] + '.pdf', ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
