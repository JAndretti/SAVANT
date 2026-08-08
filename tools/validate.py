#!/usr/bin/env python3
"""
validate.py — check a solution file produced by `cw --sol`.

Shares no code with the solver: the instances are re-read from a .cvrpb bundle
(`cw --dump-bundle`) and every cost is recomputed in Python from the
coordinates.

Checks performed, per instance:
  * every customer in 1..n served exactly once, no index out of range
  * load of each route <= capacity
  * number of routes consistent with the header
  * recomputed cost == reported cost (relative tolerance 1e-9)

At the end of the report, the recomputed mean cost is compared with the
published references (HGS and LKH-3) read from baseline/baseline.csv, for the
size n of the validated set.

Usage:
    python3 tools/validate.py results/20260731-172744_N100_200k   # run directory
    python3 tools/validate.py data/cvrp_100.cvrpb solutions.txt    # explicit files
    python3 tools/validate.py results/<run> --tol 1e-12 -v
    python3 tools/validate.py results/<run> --baseline other.csv

Given a run directory (produced by run.py), the instances are located on their
own: `instances.cvrpb` if present, otherwise the `--bundle` source read from
`config.json`. The mean cost reported by the solver is also re-read from there
and compared with the recomputed one, which validates the chain end to end.
"""

import argparse
import csv
import json
import math
import os
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))       # tools/
ROOT = os.path.dirname(HERE)                            # repository root


def read_bundle(path):
    """Read a .cvrpb bundle: list of (n, capacity, xs, ys, demands)."""
    out = []
    with open(path, "rb") as f:
        if f.read(8) != b"CVRPBIN1":
            raise SystemExit(f"{path}: invalid .cvrpb signature")
        nb, _ = struct.unpack("<II", f.read(8))
        for _ in range(nb):
            (n,) = struct.unpack("<I", f.read(4))
            (cap,) = struct.unpack("<d", f.read(8))
            xs = struct.unpack(f"<{n + 1}d", f.read(8 * (n + 1)))
            ys = struct.unpack(f"<{n + 1}d", f.read(8 * (n + 1)))
            ds = struct.unpack(f"<{n + 1}d", f.read(8 * (n + 1)))
            out.append((n, cap, xs, ys, ds))
    return out


def read_solutions(path):
    """Read the `cw --sol` file: (list of (idx, name, n, Q, cost, routes), rounded)."""
    sols = []
    rounded = False
    with open(path) as f:
        lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if line.startswith("#round"):
            rounded = line.split()[1] == "1"
            continue
        if not line or line.startswith("#"):
            continue
        f_ = line.split()
        if f_[0] != "inst":
            raise SystemExit(f"unexpected line: {line!r}")
        idx, name, n, cap, cost, nr = (
            int(f_[1]),
            f_[2],
            int(f_[3]),
            float(f_[4]),
            float(f_[5]),
            int(f_[6]),
        )
        routes = []
        for _ in range(nr):
            if i >= len(lines):
                raise SystemExit(f"instance {idx}: truncated routes")
            routes.append([int(v) for v in lines[i].split()])
            i += 1
        sols.append((idx, name, n, cap, cost, routes))
    return sols, rounded


DEFAULT_BASELINE = os.path.join(ROOT, "baseline", "baseline.csv")


def _short(path):
    """Display a path relative to the repository root, not to the cwd.

    Keeps the report readable when the script is invoked from elsewhere, where
    os.path.relpath would otherwise produce a chain of `../`.
    """
    rel = os.path.relpath(os.path.abspath(path), ROOT)
    return path if rel.startswith("..") else rel


def resolve_run(path):
    """From a run directory: (instance bundle, solution file, config).

    The instances come from the local `instances.cvrpb` when it exists
    (`--random` source, self-contained directory), otherwise from the
    `--bundle` source recorded in `config.json`. `config` is None when it is
    missing or unreadable: validation still runs, only the cross-checks are
    skipped.
    """
    sols = os.path.join(path, "solutions.txt")
    if not os.path.exists(sols):
        raise SystemExit(f"{path}: no solutions.txt — a run directory is expected")

    cfg = None
    try:
        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        pass

    local = os.path.join(path, "instances.cvrpb")
    if os.path.exists(local):
        return local, sols, cfg

    cw_args = (cfg or {}).get("cw_args") or []
    if "--bundle" in cw_args:
        i = cw_args.index("--bundle")
        cand = cw_args[i + 1] if i + 1 < len(cw_args) else ""
        for base in (os.getcwd(), ROOT):
            p = cand if os.path.isabs(cand) else os.path.join(base, cand)
            if os.path.exists(p):
                return p, sols, cfg
        raise SystemExit(f"{path}: instance bundle {cand!r} not found")
    if "--dir" in cw_args:
        raise SystemExit(
            f"{path}: --dir source, no instance bundle recorded.\n"
            f"  re-run with --dump-bundle to make the run verifiable"
        )
    raise SystemExit(
        f"{path}: instances not found "
        f"(neither instances.cvrpb nor --bundle in config.json)"
    )


def read_baselines(path):
    """Read baseline.csv: {n: [(name, objective), ...]} for HGS and LKH-3.

    Objectives are taken from the `n<N>_obj` columns, which makes the function
    independent of how many sizes the file holds. A missing or unreadable file
    is not an error: the comparison is simply skipped.
    """
    refs = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return refs
    for row in rows:
        name = (row.get("method") or "").strip()
        if not name.upper().startswith(("HGS", "LKH")):
            continue
        for col, val in row.items():
            if (
                not col
                or not col.startswith("n")
                or not col.endswith("_obj")
                or not val
            ):
                continue
            try:
                n, obj = int(col[1:-4]), float(val)
            except ValueError:
                continue
            refs.setdefault(n, []).append((name, obj))
    return refs


def main():
    ap = argparse.ArgumentParser(
        description="Validation of cw's CVRP solutions",
        epilog="Give either a run directory, or a .cvrpb bundle and a solution file.",
    )
    ap.add_argument("target", help="run directory (results/<id>/) or .cvrpb bundle")
    ap.add_argument(
        "solutions",
        nargs="?",
        help="file produced by cw --sol (not needed when target is a directory)",
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=1e-9,
        help="relative tolerance on the cost (default 1e-9)",
    )
    ap.add_argument(
        "--baseline",
        default=DEFAULT_BASELINE,
        help="CSV of published references (default: baseline/baseline.csv)",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    cfg = None
    if args.solutions is None:
        if not os.path.isdir(args.target):
            ap.error(
                f"{args.target}: a run directory is expected — "
                f"or supply the solution file as well"
            )
        bundle, solutions, cfg = resolve_run(args.target)
        print(f"run {os.path.basename(os.path.normpath(args.target))}")
        print(f"  instances : {_short(bundle)}")
    else:
        bundle, solutions = args.target, args.solutions

    insts = read_bundle(bundle)
    sols, rounded = read_solutions(solutions)
    if rounded:
        print("  (integer distances, TSPLIB EUC_2D convention)")

    errors, worst = 0, 0.0
    total_recomputed = 0.0
    sizes = set()
    # constraint tally, reported even when nothing fails
    served = expected = duplicates = out_of_range = 0
    routes_total = overloaded = 0
    worst_ratio, worst_load = 0.0, (0.0, 0.0)

    for idx, name, n, cap, cost, routes in sols:
        if idx >= len(insts):
            print(f"instance {idx}: missing from the bundle", file=sys.stderr)
            errors += 1
            continue
        bn, bcap, xs, ys, ds = insts[idx]
        if rounded:
            d = lambda a, b: math.floor(math.hypot(xs[a] - xs[b], ys[a] - ys[b]) + 0.5)
        else:
            d = lambda a, b: math.hypot(xs[a] - xs[b], ys[a] - ys[b])

        if bn != n or abs(bcap - cap) > 1e-12:
            print(
                f"instance {idx}: inconsistent header (n={n}/{bn}, Q={cap}/{bcap})",
                file=sys.stderr,
            )
            errors += 1

        seen = [0] * (bn + 1)
        total = 0.0
        for r, route in enumerate(routes):
            if not route:
                continue
            routes_total += 1
            load = 0.0
            prev = 0
            for c in route:
                if not 1 <= c <= bn:
                    print(f"instance {idx}: customer {c} out of range", file=sys.stderr)
                    errors += 1
                    out_of_range += 1
                    continue
                if seen[c]:
                    print(f"instance {idx}: customer {c} served twice", file=sys.stderr)
                    errors += 1
                    duplicates += 1
                seen[c] = 1
                total += d(prev, c)
                load += ds[c]
                prev = c
            total += d(prev, 0)
            if bcap > 0 and load / bcap > worst_ratio:
                worst_ratio, worst_load = load / bcap, (load, bcap)
            if load > bcap + 1e-9:
                print(
                    f"instance {idx}: route {r} overloaded ({load} > {bcap})",
                    file=sys.stderr,
                )
                errors += 1
                overloaded += 1

        served += sum(seen[1:])
        expected += bn
        missing_c = [c for c in range(1, bn + 1) if not seen[c]]
        if missing_c:
            print(
                f"instance {idx}: {len(missing_c)} customer(s) not served, "
                f"e.g. {missing_c[:5]}",
                file=sys.stderr,
            )
            errors += 1

        rel = abs(total - cost) / (cost if cost > 0 else 1.0)
        worst = max(worst, rel)
        if rel > args.tol:
            print(
                f"instance {idx}: reported cost {cost:.10f}, "
                f"recomputed {total:.10f} (gap {rel:.2e})",
                file=sys.stderr,
            )
            errors += 1
        elif args.verbose:
            print(
                f"instance {idx} ({name}): {len(routes)} routes, cost {total:.6f}  OK"
            )
        total_recomputed += total
        sizes.add(bn)

    print(f"{len(sols)} instance(s) checked, {errors} error(s)")
    report_constraints(
        served, expected, duplicates, out_of_range,
        routes_total, overloaded, worst_ratio, worst_load,
    )
    print(f"  max relative gap reported / recomputed cost: {worst:.3e}")
    if sols:
        mean = total_recomputed / len(sols)
        print(f"  mean cost : {mean:.6f}")
        errors += check_against_config(mean, cfg)
        report_gaps(mean, sizes, args.baseline, rounded)
        report_timing(cfg, len(sols))
    return 1 if errors else 0


def report_constraints(
    served, expected, duplicates, out_of_range,
    routes_total, overloaded, worst_ratio, worst_load,
):
    """State the two CVRP constraints explicitly, pass or fail.

    The per-instance checks only speak up on failure, which makes a clean run
    indistinguishable from a run where nothing was checked. These two lines say
    what was actually established.
    """
    cover = f"  coverage : {served}/{expected} customers served exactly once"
    faults = []
    if served != expected:
        faults.append(f"{expected - served} missing")
    if duplicates:
        faults.append(f"{duplicates} duplicate(s)")
    if out_of_range:
        faults.append(f"{out_of_range} out of range")
    print(cover + ("   [" + ", ".join(faults) + "]" if faults else "   OK"))

    load, cap = worst_load
    if cap > 0:
        state = f"{overloaded} route(s) over" if overloaded else "OK"
        print(
            f"  capacity : {routes_total} route(s), fullest {load:g}/{cap:g} "
            f"({100 * worst_ratio:.1f} % of Q)   {state}"
        )


def report_timing(cfg, n_inst):
    """Report the solver's wall time and its single-core equivalent.

    cw records both: the wall time, and the CPU time summed over the threads,
    which is what the same work would cost on a single core. Their ratio is the
    effective parallel speedup. Nothing is printed when the run predates these
    fields or when the solver was not timed.
    """
    res = (cfg or {}).get("result") or {}
    wall, cpu = res.get("time_wall_s"), res.get("time_cpu_s")
    if wall is None or cpu is None:
        return

    threads = ""
    for line in (cfg or {}).get("resolved") or []:
        m = re.search(r"threads\s*:\s*(\d+)", line)
        if m:
            threads = f" on {m.group(1)} threads"
    speedup = f", speedup {cpu / wall:.1f}x{threads}" if wall > 0 else ""

    print(f"  solver time : {wall:.3f} s wall, {cpu:.3f} s single-core{speedup}")
    if n_inst:
        print(
            f"                {1e3 * wall / n_inst:.4f} ms/instance wall, "
            f"{1e3 * cpu / n_inst:.4f} ms/instance single-core"
        )


def check_against_config(mean, cfg):
    """Cross-check the recomputed mean cost against the one the solver reported.

    cw's summary is printed to 5 decimals, hence the tolerance. A mismatch
    mostly signals a wrong instances / solutions pairing, which the per-instance
    checks would not catch when the two sets are internally consistent.
    """
    reported = (cfg or {}).get("result", {}).get("cost_after_sa")
    if reported is None:
        return 0
    if abs(mean - reported) > 1e-4:
        print(
            f"  mean cost reported by cw: {reported:.6f} (gap {mean - reported:+.2e})",
            file=sys.stderr,
        )
        return 1
    print(f"  matches the run summary ({reported:.5f})")
    return 0


def report_gaps(mean, sizes, baseline_path, rounded=False):
    """Compare the recomputed mean cost with the published references.

    The comparison only makes sense when every validated instance has the same
    size (the published objectives are per value of n) AND under the same
    distance convention. baseline.csv holds objectives for the generated float
    sets, so quoting them against a --round run of XML100 -- which also has
    n = 100 -- produced a "+110110 %" gap against a number from a different
    problem entirely.
    """
    if rounded:
        print("  (integer distances: baseline.csv is for the float sets, "
              "no comparison)")
        return
    if len(sizes) > 1:
        print(f"  (mixed sizes {sorted(sizes)}: no comparison)")
        return
    refs = read_baselines(baseline_path).get(next(iter(sizes)), [])
    n = next(iter(sizes))
    if not refs:
        print(f"  (no reference for n={n} in {baseline_path})")
        return
    width = max(len(name) for name, _ in refs)
    for name, obj in refs:
        print(
            f"  gap to {name:<{width}} ({obj:.3f}): {(mean - obj) / obj * 100.0:+.2f} %"
        )


if __name__ == "__main__":
    sys.exit(main())
