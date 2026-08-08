#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""timing_profile.py — what SAVANT's budget is actually made of, per instance.

run_matched_time.py hands the reference solvers one scalar budget per set:
SAVANT's mean CPU seconds per instance. That is exact on the sets where every
instance has the same n, and a question mark on X and XL, where it does not.
This script answers the question by measuring rather than assuming, in both of
the modes run_best_config.sh can run SAVANT in:

  steps  --sa-steps N. Every instance does the same number of draws, so what
         varies is the TIME each one takes — and a flat mean budget for the
         other solvers is then a poor description of the set.
  time   --sa-time S. Every instance gets the same time, so what varies is the
         WORK it buys, i.e. the step count. The flat budget is now exact by
         construction, and the profile moves from the time column to the steps
         column.

Same driver in both: what varies across instances is the cost of one draw.

The measurement has to be single-threaded. `results.csv`'s `time_ms` is wall
time recorded *inside* the parallel region, so it absorbs memory contention
between the threads — the report's Compute cost section shows it overstating
the cost by ~2.5x at n = 1000 and being non-additive. With `--threads 1` there
is no contention and no parallel tail, so wall == CPU and the per-instance
column is a true single-core time. X and XL hold 100 instances each, so the
whole profile costs a few minutes.

What it prints, per set and per mode:

  * the profile binned by n — the question "does the cost depend on size?";
  * the same profile binned by mean route length, n / routes;
  * the correlation of the per-instance quantity with each.

The second is there because the first turns out not to be the explanation. At
a fixed --sa-steps the annealing does a fixed number of draws whatever n is
(the report measures the annealing scaling as n^0.03), and the construction is
a few ms even at n = 10000. What does vary is the cost of one draw: swap*
(`mv_swapstar`, cw.c:1311) scans both routes, O(L1+L2), while every other
operator is O(1) — and the default configuration gives it a fifth of the draws,
on top of a ruin & recreate every 100 steps whose repair is O(k(K+L)). So
SAVANT's cost per instance tracks the mean route length, not n.

Usage:
    uv run --no-project scripts/timing_profile.py              # measure, then report
    uv run --no-project scripts/timing_profile.py --report-only
    uv run --no-project scripts/timing_profile.py --sets X --bins 5
    uv run --no-project scripts/timing_profile.py --modes steps
"""

import argparse
import csv
import glob
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "timing_profile")

sys.path.insert(0, HERE)
from run_matched_time import SETS, savant_reference   # noqa: E402

DEFAULT_SETS = ["X", "XL"]
DEFAULT_MODES = ["steps", "time"]

# cw's defaults as of 2026-08-07, spelled out as in scripts/run_best_config.sh
# (`cand`) so the profile describes the configuration actually being compared.
CONFIG = """--init cw --knn 0 --lambda 1 --mu 0
--restarts 1 --ops 1,1,1,0,1,0.05 --or-max 3
--t-accept 0.001 --t-decades 1
--sa-knn 20 --pick 2 --pick-crit lb --pick-eps 0.3
--vrank 1 --pick2 2 --reloc-side coin
--cw-rand perturb --cw-alpha 0.03
--kick 100 --kick-max 10
--race 0 --race-at 0.25 --pair 0
--split off --split-every 0 --split-tour both""".split()

# the two budgets run_best_config.sh offers, and what each holds fixed
BUDGETS = {
    "steps": ["--sa-steps", "10000000"],
    "time": ["--sa-time", "0.6", "--sa-steps", "20000"],
}


def measure(set_tag, mode):
    """One single-threaded run of the set in one budget mode."""
    spec = SETS[set_tag]
    cmd = (["uv", "run", "--no-project", "tools/run.py",
            "--out", os.path.relpath(OUT, ROOT), "--name", f"{set_tag}_t1_{mode}",
            "--bundle", spec["bundle"], "--threads", "1"]
           + (["--round"] if spec["bks"] else []) + CONFIG + BUDGETS[mode])
    print("  " + " ".join(cmd))
    sys.stdout.flush()
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if p.returncode:
        print(p.stdout + p.stderr, file=sys.stderr)
        raise SystemExit(f"{set_tag}/{mode}: measurement failed")
    for line in p.stdout.splitlines():
        if ("total time" in line or "time / instance" in line
                or "steps bought" in line):
            print("  " + line.strip())


def read_steps(run_dir):
    """Per-instance step counts from solutions.txt's #sa-steps header, or [].

    Only a --sa-time run writes it: it is the record that makes such a run
    replayable at all (re-running instance k with --sa-steps <count k> gives
    the same solution back, bit for bit).
    """
    path = os.path.join(run_dir, "solutions.txt")
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#sa-steps"):
                    return [int(v) for v in line.split()[1:]]
                if not line.startswith("#"):
                    break
    except (OSError, ValueError):
        pass
    return []


def load(set_tag, mode):
    """Per-instance rows of the newest single-threaded run of a set."""
    dirs = sorted(glob.glob(os.path.join(OUT, f"*_{set_tag}_t1_{mode}")))
    if not dirs:
        return None, None
    with open(os.path.join(dirs[-1], "results.csv"), newline="",
              encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    steps = read_steps(dirs[-1])
    for i, r in enumerate(rows):
        r["n"] = int(r["n"])
        r["cpu_ms"] = float(r["time_ms"])
        r["routes"] = int(r["routes"])
        r["steps"] = steps[i] if i < len(steps) else None
        # mean route length: what swap* pays per draw, up to a factor of two
        r["route_len"] = r["n"] / r["routes"] if r["routes"] else 0.0
    return dirs[-1], rows


def bin_table(rows, key, label, field, unit, ref, bins):
    """Quantile bins of `key`, with the spread of `field` inside each."""
    rows = sorted(rows, key=lambda r: r[key])
    size = max(1, len(rows) // bins)
    print(f"    {label:<20}{'k':>4}{unit + '  min':>15}{'median':>11}{'max':>11}"
          f"{'median/mean':>13}")
    for i in range(bins):
        g = rows[i * size:(i + 1) * size] if i < bins - 1 else rows[i * size:]
        if not g:
            continue
        t = sorted(r[field] for r in g)
        lo, hi = g[0][key], g[-1][key]
        span = (f"{lo:.0f}..{hi:.0f}" if key == "n" else f"{lo:.1f}..{hi:.1f}")
        print(f"    {span:<20}{len(g):>4}{t[0]:>15.0f}{t[len(t) // 2]:>11.0f}"
              f"{t[-1]:>11.0f}{t[len(t) // 2] / ref:>12.2f}x")


def report(set_tag, mode, budget_ms, bins):
    run_dir, rows = load(set_tag, mode)
    if not rows:
        print(f"{set_tag}/{mode}: no single-threaded run under "
              f"{os.path.relpath(OUT, ROOT)} — drop --report-only")
        return
    # in steps mode the varying quantity is the time; in time mode, the work
    if mode == "time" and all(r["steps"] for r in rows):
        field, unit = "steps", "steps"
    else:
        field, unit = "cpu_ms", "CPU ms"
        if mode == "time":
            print(f"{set_tag}/{mode}: no #sa-steps header, "
                  f"falling back to the time profile")

    t = sorted(r[field] for r in rows)
    mean = sum(t) / len(t)
    print(f"\n### {set_tag} / {mode} — {len(rows)} instances, single-threaded "
          f"({os.path.basename(run_dir)})")
    if field == "cpu_ms":
        print(f"    mean {mean:.0f} ms CPU/instance;  the flat budget "
              f"run_matched_time.py hands out is {budget_ms:.0f} ms")
    else:
        cpu = sorted(r["cpu_ms"] for r in rows)
        print(f"    mean {mean:.0f} steps/instance bought by the time budget; "
              f"CPU {cpu[0]:.0f}..{cpu[-1]:.0f} ms "
              f"({cpu[-1] / cpu[0]:.1f}x — this is what is now held flat)")
    print(f"    min {t[0]:.0f}  median {t[len(t) // 2]:.0f}  max {t[-1]:.0f}"
          f"   span {t[-1] / t[0]:.1f}x\n")

    ns = [r["n"] for r in rows]
    ls = [r["route_len"] for r in rows]
    cs = [r[field] for r in rows]
    print(f"    corr(n, {unit})                 = {statistics.correlation(ns, cs):+.3f}")
    print(f"    corr(mean route length, {unit}) = {statistics.correlation(ls, cs):+.3f}\n")

    bin_table(rows, "n", "n", field, unit, mean, bins)
    print()
    bin_table(rows, "route_len", "mean route length", field, unit, mean, bins)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sets", nargs="+", default=DEFAULT_SETS,
                    choices=[s for s in SETS], metavar="SET")
    ap.add_argument("--modes", nargs="+", default=DEFAULT_MODES,
                    choices=DEFAULT_MODES, metavar="MODE",
                    help="steps (fixed --sa-steps) and/or time (fixed --sa-time)")
    ap.add_argument("--bins", type=int, default=10, help="quantile bins (default 10)")
    ap.add_argument("--report-only", action="store_true",
                    help="reuse the last measurement instead of re-running")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    for set_tag in args.sets:
        for mode in args.modes:
            if not args.report_only:
                print(f"### measuring {set_tag} single-threaded, {mode} budget")
                measure(set_tag, mode)
            try:
                budget = savant_reference(
                    set_tag, os.path.join(ROOT, "results", "best_config")
                )["budget_s"] * 1e3
            except SystemExit:
                _, rows = load(set_tag, mode)
                budget = (sum(r["cpu_ms"] for r in rows) / len(rows)) if rows else 1.0
            report(set_tag, mode, budget, args.bins)
    return 0


if __name__ == "__main__":
    sys.exit(main())
