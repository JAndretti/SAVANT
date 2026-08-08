#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_bench.py — the recommended configuration on every dataset in data/.

One run per set, all of them with the same command: cw's 2026-08-07 defaults
written out in full, budgeted with `--sa-steps N` so each instance's budget
comes from its own dimension (timing/report.md). `report_bench.py` turns the
runs into the tables.

Threads
-------
The 10,000-instance sets are run on every core: only their cost is reported, and
cw's OpenMP loop is over instances, so the wall time is 1/P of the work.

X and XL are run with `--threads 1`, because their tables report a per-instance
compute time. `results.csv`'s `time_ms` is wall time measured *inside* the
parallel region, so with several threads it absorbs the memory contention
between them — the sweep measures that at ~2.5x at n = 1000. Single-threaded,
wall == CPU and the column is a true single-core time. It costs about 10 extra
minutes and makes the number mean something.

Names
-----
The CVRPLib sets are read with `--dir`, not `--bundle`, so every row carries the
real instance name (`X-n101-k25`, `XML100_1111_01`). The XML100 table needs it:
the name is what encodes the generator's four attributes.

Usage
-----
    uv run bench/run_bench.py                 # everything, ~15 min
    uv run bench/run_bench.py --sets X XL
    uv run bench/run_bench.py --limit 20      # smoke test
    uv run bench/run_bench.py --dry-run
"""

import argparse
import datetime as _dt
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "bench")

# cw's defaults as of 2026-08-07, spelled out so the run does not depend on
# what the built-in defaults happen to be later (same list as `cand` in
# scripts/run_best_config.sh), plus the dimension-scaled budget.
CONFIG = """--init cw --knn 0 --lambda 1 --mu 0
--restarts 1 --sa-steps N --ops 1,1,1,0,1,0.05 --or-max 3
--t-accept 0.001 --t-decades 1
--sa-knn 20 --pick 2 --pick-crit lb --pick-eps 0.3
--vrank 1 --pick2 2 --reloc-side coin
--cw-rand perturb --cw-alpha 0.03
--kick 100 --kick-max 10
--race 0 --race-at 0.25 --pair 0
--split off --split-every 0 --split-tour both
--empty-p 0 --dlb 0 --reheat 0 --t0-trim 0
--check""".split()

# threads=1 marks the sets whose per-instance time is reported
SETS = {
    "n20":    dict(src=["--bundle", "data/cvrp_20.cvrpb"],   rnd=False, threads=0, bks=None),
    "n50":    dict(src=["--bundle", "data/cvrp_50.cvrpb"],   rnd=False, threads=0, bks=None),
    "n100":   dict(src=["--bundle", "data/cvrp_100.cvrpb"],  rnd=False, threads=0, bks=None),
    "XML100": dict(src=["--dir", "data/cvrplib/XML100"],     rnd=True,  threads=0,
                   bks="data/cvrplib/XML100_bks.csv"),
    "X":      dict(src=["--dir", "data/cvrplib/X"],          rnd=True,  threads=1,
                   bks="data/cvrplib/X_bks.csv"),
    "XL":     dict(src=["--dir", "data/cvrplib/XL"],         rnd=True,  threads=1,
                   bks="baseline/xl_bks.csv"),
}
ORDER = ["n20", "n50", "n100", "XML100", "X", "XL"]


def newest(out, name):
    d = sorted(p for p in os.listdir(out) if p.endswith("_" + name))
    return os.path.join(out, d[-1]) if d else None


def link_bundle(run, src):
    """Point validate.py at the sibling .cvrpb of a --dir run.

    validate.py re-reads the instances to recompute every cost, and it cannot
    do that from a --dir run: run.py only dumps a bundle for --random, and the
    directory itself is not recorded as one. Every CVRPLib set here ships a
    sibling bundle built from the same directory, so linking it in as
    `instances.cvrpb` -- the first thing validate.py looks for -- restores the
    check.

    The link asserts that the bundle's order matches the directory's, which is
    true because both are the sorted file order. It is not taken on trust: a
    wrong pairing would make every recomputed cost disagree with the reported
    one, and validate.py fails loudly on that. So the link is either correct or
    the validation that follows it fails.
    """
    if src[0] != "--dir":
        return
    bundle = os.path.join(ROOT, src[1] + ".cvrpb")
    dst = os.path.join(run, "instances.cvrpb")
    if os.path.exists(bundle) and not os.path.exists(dst):
        os.symlink(os.path.relpath(bundle, run), dst)


def sh(cmd, log=None):
    print("  " + " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if log:
        with open(log, "w", encoding="utf-8") as f:
            f.write(p.stdout + (("\n--- stderr ---\n" + p.stderr) if p.stderr else ""))
    return p.returncode, p.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=ORDER, choices=ORDER, metavar="SET")
    ap.add_argument("--limit", type=int, default=0, help="instances per set (0 = all)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    failed = []
    for tag in a.sets:
        spec = SETS[tag]
        src = spec["src"]
        if not os.path.exists(os.path.join(ROOT, src[1])):
            print(f"!! {tag}: {src[1]} not found — run `sh tools/setup.sh data`")
            failed.append(tag + "(missing)")
            continue

        cmd = (["uv", "run", "--no-project", "tools/run.py",
                "--out", os.path.relpath(a.out, ROOT), "--name", tag]
               + src + (["--round"] if spec["rnd"] else [])
               + (["--limit", str(a.limit)] if a.limit else [])
               + ["--threads", str(spec["threads"])] + CONFIG)

        print(f"\n{'=' * 66}\n### {tag}"
              f"{'  (single-threaded: its per-instance time is reported)' if spec['threads'] == 1 else ''}"
              f"\n{'=' * 66}")
        if a.dry_run:
            print("  " + " ".join(cmd))
            continue

        t0 = _dt.datetime.now()
        rc, _ = sh(cmd)
        run = newest(a.out, tag)
        if rc or not run:
            print(f"!! {tag}: run.py exited {rc}", file=sys.stderr)
            failed.append(f"{tag}(exit {rc})")
            if not run:
                continue
        print(f"  -> {os.path.relpath(run, ROOT)}  "
              f"({(_dt.datetime.now() - t0).total_seconds():.0f} s)")
        link_bundle(run, src)

        rc, _ = sh(["uv", "run", "--no-project", "tools/validate.py", run],
                   os.path.join(run, "validation.txt"))
        if rc:
            print(f"!! VALIDATION FAILED: {tag}", file=sys.stderr)
            failed.append(tag + "(validate)")

        if spec["bks"] and os.path.exists(os.path.join(ROOT, spec["bks"])):
            sh(["uv", "run", "--no-project", "tools/gap_to_bks.py", run,
                spec["bks"], "--csv", os.path.join(run, "bks_per_instance.csv")],
               os.path.join(run, "bks_gap.txt"))

    if a.dry_run:
        return 0
    print("\n" + "=" * 66)
    subprocess.run(["uv", "run", "bench/report_bench.py", "--runs", a.out], cwd=ROOT)
    if failed:
        print("\n!! problems in: " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
