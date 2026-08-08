#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_baselines.py — HGS-CVRP, LKH-3 and AILS-II at SAVANT's per-instance time.

Reads what run_bench.py spent on each set and hands the same per-instance
budget to the reference solvers, on the same instances, through the same
drivers and the same validator.

Which solver on which set
-------------------------
  HGS-CVRP   every set
  LKH-3      XML100, X, XL only -- it has no float mode, so the generated sets
             cannot be run without changing the problem
  AILS-II    X and XL only

The budget
----------
SAVANT's *single-core* millisecond per instance, from its own run:
`cumulative CPU / instances`. That is the only figure that makes the four
commensurable, since SAVANT spreads one instance-parallel loop over every core
while the other three are single-threaded processes run J at a time.

It will not be matched, and the report says so per row rather than hiding it.
Three reasons, none under the budget flag's control:

  * every solver has set-up the budget does not cover -- HGS builds a dense
    n x n matrix, LKH-3 generates a candidate set, AILS-II starts a JVM (~0.45 s
    at n = 100, which is more than ten times the whole budget on the generated
    sets);
  * AILS-II's -limit is wall clock, not CPU, and its JIT and GC threads run
    alongside the search, so what it costs in CPU is whatever the JVM decides;
  * a process launch is ~5 ms, which is not free against a 35 ms budget.

At these budgets the overruns are large. That is a real result about what a
sub-100-millisecond budget can buy, not a flaw in the measurement, and
`report_bench.py` prints the achieved time next to every cost.

Usage
-----
    uv run bench/run_baselines.py                       # everything
    uv run bench/run_baselines.py --dry-run
    uv run bench/run_baselines.py --sets X XL
    uv run bench/run_baselines.py --solvers hgs lkh
    uv run bench/run_baselines.py --skip lkh:XL         # the expensive one
"""

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(ROOT, "results", "bench")

sys.path.insert(0, HERE)
from run_bench import SETS, ORDER            # noqa: E402

SOLVERS = {
    "hgs": dict(driver="run_hgs.py", label="HGS",
                binary="external/HGS-CVRP/build/hgs",
                sets=set(ORDER)),
    "lkh": dict(driver="run_lkh.py", label="LKH",
                binary="external/LKH-3.0.14/LKH",
                sets={"XML100", "X", "XL"}),
    "ails": dict(driver="run_ails.py", label="AILS",
                 binary="external/AILS-II/AILSII.jar",
                 sets={"X", "XL"}),
}

# per-set parallelism: HGS holds a dense n x n double matrix (~2.3 GiB at
# n = 10000), so XL cannot run many at once. AILS-II is always 1 -- its budget
# is wall clock, so J at once would silently understate it.
JOBS = {"n20": 12, "n50": 12, "n100": 12, "XML100": 12, "X": 12, "XL": 4}
HEAP = {"X": "4g", "XL": "12g"}


def savant_budget(runs_root, tag):
    """(ms per instance, instances) SAVANT spent on this set, single-core."""
    dirs = sorted(glob.glob(os.path.join(runs_root, f"*_{tag}")))
    if not dirs:
        return None, None
    try:
        with open(os.path.join(dirs[-1], "config.json"), encoding="utf-8") as f:
            res = json.load(f).get("result", {})
    except OSError:
        return None, None
    cpu, m = res.get("time_cpu_s"), res.get("instances")
    return (1e3 * cpu / m, m) if (cpu and m) else (None, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=ORDER, choices=ORDER, metavar="SET")
    ap.add_argument("--solvers", nargs="+", default=list(SOLVERS),
                    choices=list(SOLVERS), metavar="SOLVER")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip", nargs="*", default=[], metavar="SOLVER:SET",
                    help="pairs to leave out, e.g. lkh:XL")
    ap.add_argument("--runs", default=RUNS, help="where run_bench.py left SAVANT")
    ap.add_argument("--out", default=RUNS, help="where to put the new runs")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    skip = {s.lower() for s in a.skip}
    os.makedirs(a.out, exist_ok=True)

    print("### budgets, from SAVANT's own run of each set\n")
    print(f"{'set':<8}{'instances':>11}{'SAVANT ms/inst':>16}   solvers")
    print("-" * 62)
    plan = []
    for tag in a.sets:
        ms, m = savant_budget(a.runs, tag)
        if ms is None:
            print(f"{tag:<8}{'—':>11}{'not run yet':>16}")
            continue
        todo = [s for s in a.solvers
                if tag in SOLVERS[s]["sets"] and f"{s}:{tag}".lower() not in skip]
        print(f"{tag:<8}{m:>11,}{ms:>16.1f}   "
              f"{' '.join(SOLVERS[s]['label'] for s in todo) or '—'}")
        for s in todo:
            plan.append((s, tag, ms))
    print()

    failed, skipped = [], []
    for solver, tag, ms in plan:
        sv, spec = SOLVERS[solver], SETS[tag]
        label = f"{sv['label']}/{tag}"
        if not os.path.exists(os.path.join(ROOT, sv["binary"])):
            print(f"-- {label}: skipped ({sv['binary']} not built)")
            skipped.append(label)
            continue

        jobs = 1 if solver == "ails" else JOBS.get(tag, 12)
        cmd = (["uv", "run", "--no-project", os.path.join("tools", sv["driver"])]
               + spec["src"]
               + ["--time", f"{ms / 1e3:.5f}",
                  "--out", os.path.relpath(a.out, ROOT),
                  "--name", f"{sv['label']}_{tag}",
                  "--jobs", str(jobs)]
               + (["--limit", str(a.limit)] if a.limit else [])
               + (["--heap", HEAP.get(tag, "2g")] if solver == "ails" else []))

        print(f"\n{'=' * 70}\n### {label}   {ms:.1f} ms/instance requested"
              f"   ({jobs} job{'s' if jobs > 1 else ''})\n{'=' * 70}")
        print("  " + " ".join(cmd), flush=True)
        if a.dry_run:
            continue

        t0 = time.time()
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        run = sorted(glob.glob(os.path.join(a.out, f"*_{sv['label']}_{tag}")))
        print(f"  {time.time() - t0:.0f} s wall")
        if not run:
            print(f"!! {label}: no run directory\n{p.stderr[-800:]}", file=sys.stderr)
            failed.append(label)
            continue
        run = run[-1]
        with open(os.path.join(run, "driver.log"), "w", encoding="utf-8") as f:
            f.write(p.stdout + (("\n--- stderr ---\n" + p.stderr) if p.stderr else ""))
        if p.returncode:
            # usually a partial result rather than a lost one: run_lkh.py fails
            # the run when an instance comes back capacity-penalised, and the
            # ones that did solve are still in solutions.txt
            print(f"!! {label}: driver exited {p.returncode} — "
                  f"validating the partial result", file=sys.stderr)
            failed.append(f"{label}(exit {p.returncode})")

        v = subprocess.run(["uv", "run", "--no-project", "tools/validate.py", run],
                           cwd=ROOT, capture_output=True, text=True)
        with open(os.path.join(run, "validation.txt"), "w", encoding="utf-8") as f:
            f.write(v.stdout + (("\n--- stderr ---\n" + v.stderr) if v.stderr else ""))
        if v.returncode:
            print(f"!! VALIDATION FAILED: {label}", file=sys.stderr)
            failed.append(label + "(validate)")

        if spec["bks"] and os.path.exists(os.path.join(ROOT, spec["bks"])):
            g = subprocess.run(["uv", "run", "--no-project", "tools/gap_to_bks.py",
                                run, spec["bks"], "--csv",
                                os.path.join(run, "bks_per_instance.csv")],
                               cwd=ROOT, capture_output=True, text=True)
            with open(os.path.join(run, "bks_gap.txt"), "w", encoding="utf-8") as f:
                f.write(g.stdout + g.stderr)

    if a.dry_run:
        return 0
    print("\n" + "=" * 70)
    subprocess.run(["uv", "run", "bench/report_bench.py", "--runs", a.out], cwd=ROOT)
    if skipped:
        print("\nskipped: " + ", ".join(skipped))
    if failed:
        print("\n!! problems in: " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
