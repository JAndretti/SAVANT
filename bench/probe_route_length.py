#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_route_length.py — why two XL instances of the same size differ 4x in time.

XL-n9571-k55 took 48.13 s and XL-n10001-k1570 took 12.16 s in the bench run,
although the second is the larger instance and was given more steps. The two
differ in one obvious way: 55 routes against 1570, i.e. a mean route length of
174 customers against 6.4.

This script measures where that goes, in two parts:

  1. **The whole XL set**: per-instance nanoseconds per annealing step against
     mean route length and against n, so the association is measured on 100
     instances rather than argued from two.
  2. **An ablation on four instances of near-identical n and wildly different
     route length**: each operator run *alone*, so its own cost per draw can be
     read off directly, plus the default mix with and without the kick. That
     turns "route length correlates with time" into "this operator costs this
     many nanoseconds per unit of route length".

Every run is `--threads 1` with a fixed `--sa-steps`, so nanoseconds per step
are comparable across configurations; one run at a time, for the same reason
`timing/` does it.

Usage:
    uv run bench/probe_route_length.py            # ~4 min
    uv run bench/probe_route_length.py --steps 1000000
    uv run bench/probe_route_length.py --report-only
"""

import argparse
import csv
import glob
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BIN = os.path.join(ROOT, "cw")
XLDIR = os.path.join(ROOT, "data", "cvrplib", "XL")
OUT = os.path.join(ROOT, "results", "bench", "probe")

# four XL instances with n within 5 % of each other and mean route length
# spanning 27x: the natural experiment the generated family cannot provide,
# because its capacity ladder pins route length near 40 above n = 1000
INSTANCES = ["XL-n9571-k55", "XL-n9363-k209", "XL-n9160-k379", "XL-n10001-k1570"]

BASE = """--init cw --knn 0 --lambda 1 --mu 0 --restarts 1 --or-max 3
--t-accept 0.001 --t-decades 1 --sa-knn 20 --pick 2 --pick-crit lb
--pick-eps 0.3 --vrank 1 --pick2 2 --reloc-side coin --cw-rand perturb
--cw-alpha 0.03 --race 0 --pair 0 --split off --kick-max 10
--empty-p 0 --dlb 0 --reheat 0 --t0-trim 0 --round --threads 1""".split()

# one operator at a time, then the shipped mix with and without the kick
CONFIGS = [
    ("relocate", ["--ops", "1,0,0,0,0,0", "--kick", "0"]),
    ("swap", ["--ops", "0,1,0,0,0,0", "--kick", "0"]),
    ("2-opt", ["--ops", "0,0,1,0,0,0", "--kick", "0"]),
    ("or-opt", ["--ops", "0,0,0,1,0,0", "--kick", "0"]),
    ("swap*", ["--ops", "0,0,0,0,1,0", "--kick", "0"]),
    ("opening", ["--ops", "0,0,0,0,0,1", "--kick", "0"]),
    ("kick only", ["--ops", "1,0,0,0,0,0", "--kick", "1"]),
    ("default, no kick", ["--ops", "1,1,1,0,1,0.05", "--kick", "0"]),
    ("default", ["--ops", "1,1,1,0,1,0.05", "--kick", "100"]),
]
# the shipped weights, for predicting the mix from the parts
WEIGHTS = {"relocate": 1.0, "swap": 1.0, "2-opt": 1.0, "or-opt": 0.0,
           "swap*": 1.0, "opening": 0.05}


def cell(inst, tag, steps):
    return os.path.join(OUT, f"{inst}__{tag.replace(' ', '_').replace('*', 'star')}"
                             f"__{steps}")


def run(inst, tag, flags, steps):
    base = cell(inst, tag, steps)
    if os.path.exists(base + ".csv"):
        return
    cmd = ([BIN, "--dir", XLDIR, "--sa-steps", str(steps)] + BASE + flags
           + ["--csv", base + ".csv"])
    # --dir reads the whole directory; --limit cannot select one instance by
    # name, so the run is restricted by pointing at a one-file directory
    one = base + ".d"
    os.makedirs(one, exist_ok=True)
    link = os.path.join(one, inst + ".vrp")
    if not os.path.exists(link):
        os.symlink(os.path.join(XLDIR, inst + ".vrp"), link)
    cmd[cmd.index("--dir") + 1] = one
    p = subprocess.run(cmd, capture_output=True, text=True)
    with open(base + ".log", "w", encoding="utf-8") as f:
        f.write(p.stdout + p.stderr)
    if p.returncode:
        print(f"  !! {inst} / {tag}: exit {p.returncode}", file=sys.stderr)


def read(inst, tag, steps):
    """(ns per step, mean route length reached) for one cell."""
    try:
        with open(cell(inst, tag, steps) + ".csv", newline="",
                  encoding="utf-8") as f:
            r = next(csv.DictReader(f))
        n, routes = int(r["n"]), int(r["routes"])
        ms = float(r["time_ms"])
        return 1e6 * ms / steps, (n / routes if routes else 0.0)
    except (OSError, StopIteration, KeyError, ValueError):
        return None, None


def xl_set():
    """Per-instance (n, route length, ns/step) from the bench XL run."""
    dirs = [d for d in sorted(glob.glob(os.path.join(ROOT, "results", "bench", "*_XL")))
            if not any(s in os.path.basename(d) for s in ("HGS", "LKH", "AILS"))]
    if not dirs:
        return []
    d = dirs[-1]
    with open(os.path.join(d, "results.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    steps = []
    with open(os.path.join(d, "solutions.txt"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("#sa-steps"):
                steps = [int(v) for v in line.split()[1:]]
                break
            if not line.startswith("#"):
                break
    out = []
    for i, r in enumerate(rows):
        if i >= len(steps):
            break
        n, R = int(r["n"]), int(r["routes"])
        out.append({"name": r["instance"].removesuffix(".vrp"), "n": n,
                    "L": n / R if R else 0.0, "steps": steps[i],
                    "s": float(r["time_ms"]) / 1e3,
                    "ns": 1e6 * float(r["time_ms"]) / steps[i]})
    return out


# Does swap* still earn its weight where it costs the most? Compared at equal
# WALL TIME (--sa-time), because that is the only comparison in which a cheaper
# operator mix is allowed to convert its saving into more draws.
WEIGHT_CONFIGS = [("w=1 (default)", "1,1,1,0,1,0.05"),
                  ("w=0.25", "1,1,1,0,0.25,0.05"),
                  ("w=0 (off)", "1,1,1,0,0,0.05")]


def weight_probe(secs, long_n, short_n, seed=1):
    """swap* weight against solution quality at equal time, long vs short routes.

    Configurations are run adjacent in time for the same instance, so a machine
    that is busier at one moment than another moves all three together.
    """
    data = sorted((d for d in xl_set() if d["L"] > 0), key=lambda d: d["L"])
    picks = ([("short", d) for d in data[:short_n]]
             + [("long", d) for d in data[-long_n:]])
    out = {}
    for group, d in picks:
        one = os.path.join(OUT, "w_" + d["name"] + ".d")
        os.makedirs(one, exist_ok=True)
        link = os.path.join(one, d["name"] + ".vrp")
        if not os.path.exists(link):
            os.symlink(os.path.join(XLDIR, d["name"] + ".vrp"), link)
        for tag, ops in WEIGHT_CONFIGS:
            base = os.path.join(OUT, f"w_{d['name']}__{ops}__{secs}")
            if not os.path.exists(base + ".csv"):
                cmd = ([BIN, "--dir", one, "--sa-time", str(secs),
                        "--sa-steps", "20000", "--seed", str(seed),
                        "--ops", ops, "--kick", "100"] + BASE
                       + ["--csv", base + ".csv"])
                subprocess.run(cmd, capture_output=True, text=True)
            try:
                with open(base + ".csv", newline="", encoding="utf-8") as f:
                    r = next(csv.DictReader(f))
                out[(d["name"], tag)] = float(r["cost_annealed"])
            except (OSError, StopIteration, KeyError, ValueError):
                pass
        out[(d["name"], "group")] = group
        out[(d["name"], "L")] = d["L"]
    return [d for _, d in picks], out


def pick_wide(k):
    """k XL instances spread evenly over route length, from the bench run."""
    data = [d for d in xl_set() if d["L"] > 0]
    data.sort(key=lambda d: d["L"])
    if not data:
        return []
    step = max(1, len(data) // k)
    return [d["name"] for d in data[::step]][:k]


def fit(xs, ys):
    """(intercept, slope, R^2) of a plain least-squares line."""
    n = len(xs)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    a = my - b * mx
    ss = sum((y - my) ** 2 for y in ys)
    rs = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return a, b, (1 - rs / ss if ss else 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--instances", nargs="+", default=INSTANCES)
    ap.add_argument("--wide", type=int, default=16, metavar="K",
                    help="also run swap* and relocate alone on K instances "
                         "spread over route length (0 = skip); the four-way "
                         "ablation attributes, this one measures the slope")
    ap.add_argument("--weights", type=float, default=10.0, metavar="SEC",
                    help="seconds per run for the swap*-weight experiment "
                         "(0 = skip)")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if not a.report_only:
        todo = [(i, t, f) for i in a.instances for t, f in CONFIGS]
        for k, (inst, tag, flags) in enumerate(todo, 1):
            print(f"  [{k:>2}/{len(todo)}] {inst:<18} {tag}", flush=True)
            run(inst, tag, flags, a.steps)

    data = xl_set()
    print("\n### the whole XL set, from the bench run\n")
    if data:
        ns = [d["ns"] for d in data]
        L = [d["L"] for d in data]
        n = [float(d["n"]) for d in data]
        print(f"    {len(data)} instances; ns/step {min(ns):.1f}..{max(ns):.1f} "
              f"({max(ns) / min(ns):.1f}x), route length {min(L):.1f}..{max(L):.1f}")
        print(f"    corr(n, ns/step)            = "
              f"{statistics.correlation(n, ns):+.3f}")
        print(f"    corr(route length, ns/step) = "
              f"{statistics.correlation(L, ns):+.3f}")
        aa, bb, r2 = fit(L, ns)
        print(f"    ns/step = {aa:.1f} + {bb:.3f} * L      R2 = {r2:.3f}")
        aa2, bb2, r22 = fit(n, ns)
        print(f"    ns/step = {aa2:.1f} + {bb2:.2e} * n    R2 = {r22:.3f}"
              f"   <- size alone explains nothing")

    print(f"\n### per-operator cost, {a.steps:,} steps each, single-threaded\n")
    hdr = f"    {'configuration':<18}" + "".join(f"{i.split('-')[1]:>12}"
                                                 for i in a.instances)
    print(hdr)
    Ls = {}
    for inst in a.instances:
        _, L = read(inst, "default", a.steps)
        Ls[inst] = L
    print(f"    {'mean route length':<18}" + "".join(
        f"{Ls[i]:>12.1f}" if Ls[i] else f"{'--':>12}" for i in a.instances))
    print("    " + "-" * (18 + 12 * len(a.instances)))
    table = {}
    for tag, _ in CONFIGS:
        cells = []
        for inst in a.instances:
            ns, _ = read(inst, tag, a.steps)
            table[(tag, inst)] = ns
            cells.append(f"{ns:>12.1f}" if ns else f"{'--':>12}")
        print(f"    {tag:<18}" + "".join(cells))

    print(f"\n    ns per draw. Every column is the same instance and the same "
          f"{a.steps:,} draws;\n    what changes is which operator makes them.\n")

    # does the mix add up from the parts?
    print("    predicted mix = sum(weight_i * cost_i) / sum(weight):\n")
    for inst in a.instances:
        num = sum(WEIGHTS[t] * (table.get((t, inst)) or 0.0) for t in WEIGHTS)
        pred = num / sum(WEIGHTS.values())
        got = table.get(("default, no kick", inst))
        if got:
            print(f"      {inst:<18} predicted {pred:>7.1f}   measured "
                  f"{got:>7.1f}   ({100 * (got - pred) / pred:+.1f} %)")

    # The four-instance ablation attributes the cost to an operator, but it
    # cannot pin the functional form: four instances differ in geometry and in
    # n as well as in route length. So the two operators that matter are re-run
    # across a wider spread, one configuration each.
    if not a.report_only and a.wide:
        wide = pick_wide(a.wide)
        print(f"\n### the same two configurations on {len(wide)} instances "
              f"spanning route length\n")
        for k, inst in enumerate(wide, 1):
            print(f"  [{k:>2}/{len(wide)}] {inst}", flush=True)
            for tag in ("swap*", "relocate"):
                run(inst, tag, dict(CONFIGS)[tag], a.steps)
        print(f"      {'operator':<12}{'m':>4}{'intercept':>11}"
              f"{'ns per unit L':>15}{'R2':>8}")
        for tag in ("swap*", "relocate"):
            pts = []
            for inst in wide:
                ns, L = read(inst, tag, a.steps)
                if ns and L:
                    pts.append((L, ns))
            if len(pts) >= 3:
                aa, bb, r2 = fit([p[0] for p in pts], [p[1] for p in pts])
                print(f"      {tag:<12}{len(pts):>4}{aa:>11.1f}{bb:>15.3f}{r2:>8.3f}")

    if not a.report_only and a.weights:
        print(f"\n### is swap* worth its cost where it is dearest? "
              f"{a.weights} s per run, equal wall time\n")
        picks, W = weight_probe(a.weights, 6, 6)
        print(f"    {'instance':<20}{'L':>7}" +
              "".join(f"{t:>16}" for t, _ in WEIGHT_CONFIGS))
        for grp in ("short", "long"):
            deltas = {t: [] for t, _ in WEIGHT_CONFIGS}
            for d in picks:
                if W.get((d['name'], "group")) != grp:
                    continue
                ref = W.get((d["name"], WEIGHT_CONFIGS[0][0]))
                cells = []
                for t, _ in WEIGHT_CONFIGS:
                    c = W.get((d["name"], t))
                    if c and ref:
                        deltas[t].append(100 * (c - ref) / ref)
                        cells.append(f"{c:>10.0f}"
                                     + (f"{100 * (c - ref) / ref:>+6.2f}"
                                        if t != WEIGHT_CONFIGS[0][0] else "      "))
                    else:
                        cells.append(f"{'--':>16}")
                print(f"    {d['name']:<20}{W[(d['name'], 'L')]:>7.1f}" + "".join(cells))
            print(f"    {'mean delta % (' + grp + ')':<27}" + "".join(
                f"{statistics.mean(v):>+16.2f}" if v else f"{'--':>16}"
                for v in deltas.values()))
            print()
        print("    Cost after annealing, same wall clock. Negative means the "
              "lighter mix wins:\n    it converts the saved time into more "
              "draws.\n")

    # per-unit-of-route-length slope of each operator
    print("\n    cost against route length, across the four instances:\n")
    print(f"      {'operator':<18}{'intercept':>11}{'ns per unit L':>16}{'R2':>8}")
    for tag, _ in CONFIGS:
        pts = [(Ls[i], table[(tag, i)]) for i in a.instances
               if Ls.get(i) and table.get((tag, i))]
        if len(pts) >= 3:
            aa, bb, r2 = fit([p[0] for p in pts], [p[1] for p in pts])
            print(f"      {tag:<18}{aa:>11.1f}{bb:>16.3f}{r2:>8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
