#!/usr/bin/env python3
"""
fuzz.py — draw random option combinations, run the solver, then validate every
solution produced with validate.py (independent code).

Checks per run:
  * expected exit code (0, or 2 if the instance really is infeasible)
  * no crash, nothing written to stderr
  * incremental-cost drift below 1e-6 (--check)
  * every solution written revalidated: coverage, capacities, and cost
    recomputed from the coordinates

Usage: python3 fuzz.py [n_trials] [--asan]
"""

import glob
import os
import random
import subprocess
import sys

from _paths import BIN, BIN_DBG, DATA, EDGE, VALIDATE

BIN = BIN_DBG if "--asan" in sys.argv else BIN
N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 200
SLOW = BIN.endswith("dbg")

SOURCES = [
    ("--bundle " + f, os.path.basename(f))
    for f in sorted(glob.glob(os.path.join(EDGE, "*.cvrpb")))
] + [
    (f"--bundle {DATA}/cvrp_20.cvrpb --limit 40", "cvrp20"),
    (f"--bundle {DATA}/cvrp_200.cvrpb --limit 12", "cvrp200"),
    (f"--dir {DATA}/cvrp_200 --limit 6", "dir200"),
]


def rnd_opts(r):
    o = []
    steps = r.choice([0, 1, 2, 17, 300] + ([] if SLOW else [3000, 20000]))
    o += ["--sa-steps", str(steps)]
    w = [r.choice([0, 0, 1, 1, 2, 5]) for _ in range(6)]
    if sum(w) == 0:
        w[r.randrange(6)] = 1
    o += ["--ops", ",".join(map(str, w))]
    o += ["--or-max", str(r.randint(2, 8))]
    o += ["--pick", str(r.choice([0, 1, 2, 3, 5]))]
    o += ["--pick-crit", r.choice(["lb", "rem", "remnorm", "raw"])]
    o += ["--pick-eps", str(r.choice([0.0, 0.05, 0.3, 2.0]))]
    o += ["--sa-knn", str(r.choice([0, 1, 2, 5, 20, 200]))]
    o += ["--restarts", str(r.randint(1, 3))]
    o += ["--vrank", str(r.randint(1, 8))]
    o += ["--pick2", str(r.choice([1, 1, 2, 3, 16]))]
    o += ["--reloc-side", r.choice(["coin", "long"])]
    o += ["--pair", str(r.choice([0, 1, -1]))]
    o += ["--race", r.choice(["off", "off", "0.0005", "0.002", "0.5"])]
    o += ["--race-at", str(r.choice([0.05, 0.25, 0.9]))]
    o += ["--cw-rand", r.choice(["off", "perturb", "param", "both"])]
    o += ["--cw-alpha", str(r.choice([0.0, 0.03, 0.4, 0.9]))]
    o += ["--split", r.choice(["off", "cw", "end", "both"])]
    o += ["--split-tour", r.choice(["routes", "sweep", "both"])]
    o += ["--split-every", str(r.choice([0, 0, 1, 7, 250]))]
    if r.random() < 0.5:
        o += ["--t-accept", str(r.choice([0.001, 0.05, 0.5, 0.95]))]
        o += ["--t-decades", str(r.choice([0.5, 2, 6]))]
    else:
        t0 = r.choice([1e-9, 0.01, 1.0, 50.0])
        o += ["--t0", str(t0), "--tend", str(t0 * r.choice([1e-6, 1e-2, 1.0]))]
    if r.random() < 0.3:
        o += ["--knn", str(r.choice([1, 2, 3, 8, 5000]))]
    if r.random() < 0.2:
        o += ["--exact"]
    if r.random() < 0.3:
        o += ["--lambda", str(r.choice([0.2, 1.0, 1.6]))]
    if r.random() < 0.3:
        o += ["--mu", str(r.choice([0.0, 0.5]))]
    if r.random() < 0.2:
        o += ["--2opt"]
    if r.random() < 0.2:
        o += ["--round"]  # honoured by --validate and validate.py
    o += ["--threads", str(r.randint(1, 3))]
    o += ["--seed", str(r.randrange(10**6))]
    return o


def main():
    r = random.Random(20260729)
    fails = 0
    for t in range(N):
        src, name = r.choice(SOURCES)
        if r.random() < 0.25:
            n = r.choice([1, 2, 4, 30, 200, 900, 2500])
            src = f"--random -n {n} -m {r.randint(1, 6)} --cap {r.choice([1, 9, 30, 200, 1e5])}"
            name = f"rnd{n}"
        cmd = (
            [BIN]
            + src.split()
            + rnd_opts(r)
            + [
                "-q",
                "--check",
                "--sol",
                "/tmp/fz.txt",
                "--dump-bundle",
                "/tmp/fz.cvrpb",
            ]
        )
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        # exit code 2 is legitimate if the instance really is infeasible
        infeas = False
        if p.returncode == 2 and os.path.exists("/tmp/fz.cvrpb"):
            from validate import read_bundle

            infeas = any(
                max(ds[1:]) > cap + 1e-9
                for n_, cap, xs, ys, ds in read_bundle("/tmp/fz.cvrpb")
            )
        ok = (p.returncode == 0) or (p.returncode == 2 and infeas)
        why = []
        if not ok:
            why.append(f"exit code {p.returncode}")
        if p.stderr.strip():
            why.append("stderr: " + p.stderr.strip()[:200])
        if ok:
            v = subprocess.run(
                ["python3", VALIDATE, "/tmp/fz.cvrpb", "/tmp/fz.txt"],
                capture_output=True,
                text=True,
            )
            # on an infeasible instance, only overloads are tolerated
            bad = [
                l
                for l in v.stderr.strip().splitlines()
                if l and not (infeas and "overloaded" in l)
            ]
            if bad:
                why.append("validation: " + bad[0][:200])
        if why:
            fails += 1
            print(f"\n[FAIL {t}] {' '.join(cmd)}")
            for w in why:
                print("   ", w)
        elif t % 20 == 0:
            print(f"  {t}/{N} ok ({name})", flush=True)
    print(f"\n{N} trials, {fails} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
