#!/usr/bin/env python3
"""
run_hgs.py — run Vidal's HGS-CVRP over a .cvrpb bundle, in the run-directory
layout that tools/validate.py already understands.

The point is to compare SAVANT against HGS *on this machine*, on exactly the
same instances, with both sides checked by the same independent validator.
The output directory therefore mirrors what tools/run.py produces for `cw`:

    results/<stamp>_<name>/
        solutions.txt   HGS's routes in the `cw --sol` format
        results.csv     one row per instance
        config.json     command, versions, environment, aggregate result
        run.log         anything HGS wrote to stderr

so that

    python3 tools/validate.py results/<stamp>_<name>

re-reads the bundle and recomputes every cost from the coordinates, exactly as
it does for SAVANT.

Three things about HGS matter here and are handled below:

  * `-round` defaults to 1. The NeuOpt instances have coordinates in [0,1], so
    integer rounding collapses every distance to 0 or 1 and HGS reports a cost
    of 0. This script always passes `-round 0` and refuses `--round 1` unless
    the bundle really is integral.

  * `-t` is CPU time (`clock()`), not wall clock — see Genetic.cpp. A process
    therefore gets its full budget even when 12 of them share 12 cores, which
    makes the budget reproducible but means wall time exceeds `-t` under load.
    Both are recorded.

  * HGS prints the objective with 6 significant digits, which is not enough to
    validate at 1e-9. solutions.txt therefore carries the cost recomputed here
    in float64; HGS's own figure is cross-checked against it to 1e-5 relative
    and any disagreement is reported as an error.

Usage:
    python3 tools/run_hgs.py --bundle data/cvrp_100.cvrpb --time 1.0 \\
        --jobs 12 --name HGS_N100_t1
    python3 tools/run_hgs.py --bundle data/cvrp_100.cvrpb --limit 200 --it 20000
"""

import argparse
import concurrent.futures as cf
import csv
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_HGS = os.path.join(ROOT, "external", "HGS-CVRP", "build", "hgs")
RESULTS = os.path.join(ROOT, "results")

sys.path.insert(0, HERE)
from bundle_to_vrp import read_bundle, write_vrp, instance_name  # noqa: E402


ROUTE_RE = re.compile(r"^Route #(\d+)\s*:\s*(.*)$")
COST_RE = re.compile(r"^Cost\s+([0-9.eE+-]+)\s*$")


def parse_sol(path):
    """Read HGS's .sol: (routes, announced_cost).

    Customer indices are HGS's internal 1..nbClients, which coincide with the
    bundle's own numbering because write_vrp() emits TSPLIB node i+1 for
    bundle index i and HGS's parser asserts that ordering.
    """
    routes, cost = [], None
    with open(path) as f:
        for line in f:
            m = ROUTE_RE.match(line.strip())
            if m:
                routes.append([int(v) for v in m.group(2).split()])
                continue
            m = COST_RE.match(line.strip())
            if m:
                cost = float(m.group(1))
    if cost is None:
        raise ValueError(f"{path}: no Cost line")
    return routes, cost


def recompute(routes, xs, ys, ds, cap, n, rounded):
    """Cost recomputed from the coordinates, plus a feasibility verdict.

    Deliberately independent of HGS's own arithmetic: this is what lands in
    solutions.txt, and validate.py will redo it a third time.
    """
    if rounded:
        def d(a, b):
            return math.floor(math.hypot(xs[a] - xs[b], ys[a] - ys[b]) + 0.5)
    else:
        def d(a, b):
            return math.hypot(xs[a] - xs[b], ys[a] - ys[b])

    seen = [0] * (n + 1)
    total = 0.0
    problems = []
    nroutes = 0
    for r, route in enumerate(routes):
        if not route:
            continue
        nroutes += 1
        load = prev = 0
        prev = 0
        for c in route:
            if not 1 <= c <= n:
                problems.append(f"customer {c} out of range")
                continue
            if seen[c]:
                problems.append(f"customer {c} served twice")
            seen[c] = 1
            total += d(prev, c)
            load += ds[c]
            prev = c
        total += d(prev, 0)
        if load > cap + 1e-9:
            problems.append(f"route {r} overloaded ({load} > {cap})")
    missing = [c for c in range(1, n + 1) if not seen[c]]
    if missing:
        problems.append(f"{len(missing)} customer(s) unserved, e.g. {missing[:5]}")
    return total, nroutes, problems


def solve_one(job):
    """Run HGS on one instance. Returns a dict; never raises."""
    idx, name, vrp, soldir, hgs, argv = job
    out = os.path.join(soldir, name + ".sol")
    cmd = [hgs, vrp, out] + argv
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"idx": idx, "name": name, "error": f"launch failed: {e}",
                "wall": time.perf_counter() - t0}
    wall = time.perf_counter() - t0
    if p.returncode != 0 or not os.path.exists(out):
        return {"idx": idx, "name": name, "wall": wall,
                "error": f"exit {p.returncode}: "
                         f"{(p.stdout + p.stderr).strip()[:200]}"}
    try:
        routes, announced = parse_sol(out)
    except (OSError, ValueError) as e:
        return {"idx": idx, "name": name, "wall": wall, "error": str(e)}
    return {"idx": idx, "name": name, "wall": wall, "routes": routes,
            "announced": announced, "stderr": p.stderr.strip()}


def binary_fingerprint(path):
    try:
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()[:16]
        return {"path": path, "sha256_16": digest,
                "mtime": _dt.datetime.fromtimestamp(
                    os.path.getmtime(path)).isoformat(timespec="seconds"),
                "size": os.path.getsize(path)}
    except OSError:
        return {"path": path}


def hgs_version(repo):
    try:
        r = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True)
        return r.stdout.strip() or None
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser(
        description="Run HGS-CVRP over a .cvrpb bundle into a validatable run "
                    "directory",
    )
    ap.add_argument("--bundle", required=True, help="instances (.cvrpb)")
    ap.add_argument("--hgs", default=DEFAULT_HGS, help="hgs binary")
    ap.add_argument("--vrp", default=None,
                    help="directory of exported .vrp files (default: "
                         "data/vrp_<stem>, created and reused)")
    ap.add_argument("--limit", type=int, default=0,
                    help="solve only the first N instances")
    ap.add_argument("--time", type=float, default=0.0, metavar="SEC",
                    help="HGS -t: CPU-second budget per instance")
    ap.add_argument("--it", type=int, default=0,
                    help="HGS -it: iterations without improvement "
                         "(HGS's own default is 20000)")
    ap.add_argument("--seed", type=int, default=0, help="HGS -seed")
    ap.add_argument("--veh", type=int, default=0,
                    help="HGS -veh: prescribed fleet size (default: HGS's own "
                         "estimate, 1.3 x bin-packing LB + 3)")
    ap.add_argument("--round", dest="rounded", type=int, default=0,
                    choices=(0, 1),
                    help="integer distances (default 0 — the NeuOpt "
                         "coordinates live in [0,1] and MUST NOT be rounded)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count(),
                    help="parallel HGS processes (default: all cores)")
    ap.add_argument("--name", help="readable suffix for the run directory")
    ap.add_argument("--out", default=RESULTS, help="root for runs")
    ap.add_argument("--keep-sol", action="store_true",
                    help="keep HGS's raw .sol files inside the run directory")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.hgs):
        ap.error(f"{args.hgs} not found — build it with:\n"
                 f"  cmake -S external/HGS-CVRP -B external/HGS-CVRP/build "
                 f"-DCMAKE_BUILD_TYPE=Release && "
                 f"make -C external/HGS-CVRP/build bin")
    if args.time <= 0 and args.it <= 0:
        print("note: neither --time nor --it given; HGS runs to its own "
              "default of 20000 non-improving iterations (~6 s/instance at "
              "n=100 on this machine)", file=sys.stderr)

    insts = read_bundle(args.bundle, args.limit)
    if not insts:
        raise SystemExit(f"{args.bundle}: no instance")
    n_all = {n for n, _, _, _, _ in insts}

    if args.rounded and any(
        any(abs(v - round(v)) > 1e-12 for v in xs[:5] + ys[:5])
        for _, _, xs, ys, _ in insts[:5]
    ):
        ap.error("--round 1 on continuous coordinates would collapse every "
                 "distance; this is almost certainly not what you want")

    # --- export the .vrp files (reused across runs) -----------------------
    stem = os.path.splitext(os.path.basename(args.bundle))[0]
    vrpdir = args.vrp or os.path.join(ROOT, "data", "vrp_" + stem.split("_")[-1])
    os.makedirs(vrpdir, exist_ok=True)
    width = len(str(len(insts) - 1))
    exported = 0
    for k, (n, cap, xs, ys, ds) in enumerate(insts):
        name = instance_name(stem, k, width)
        path = os.path.join(vrpdir, name + ".vrp")
        if not os.path.exists(path):
            write_vrp(path, name, n, cap, xs, ys, ds)
            exported += 1

    argv = ["-round", str(args.rounded), "-log", "0", "-seed", str(args.seed)]
    if args.time > 0:
        argv += ["-t", str(args.time)]
    if args.it > 0:
        argv += ["-it", str(args.it)]
    if args.veh > 0:
        argv += ["-veh", str(args.veh)]

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", args.name).strip("-") if args.name else ""
    run_dir = os.path.join(args.out, f"{stamp}_{slug}" if slug else stamp)

    print(f"bundle    : {args.bundle}  ({len(insts)} instances, n = {sorted(n_all)})")
    print(f"vrp       : {vrpdir}" + (f"  (+{exported} exported)" if exported else ""))
    print(f"hgs       : {args.hgs} {' '.join(argv)}")
    print(f"jobs      : {args.jobs}")
    print(f"directory : {run_dir}")
    if args.dry_run:
        return 0

    os.makedirs(run_dir, exist_ok=True)
    soldir = (os.path.join(run_dir, "sol") if args.keep_sol
              else tempfile.mkdtemp(prefix="hgs-sol-"))
    os.makedirs(soldir, exist_ok=True)

    jobs = [
        (k, instance_name(stem, k, width),
         os.path.join(vrpdir, instance_name(stem, k, width) + ".vrp"),
         soldir, args.hgs, argv)
        for k in range(len(insts))
    ]

    started = _dt.datetime.now()
    t_wall0 = time.perf_counter()
    t_cpu0 = time.process_time()
    ru0 = os.times()

    results = [None] * len(jobs)
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for res in ex.map(solve_one, jobs):
            results[res["idx"]] = res
            done += 1
            if done % 50 == 0 or done == len(jobs):
                el = time.perf_counter() - t_wall0
                eta = el / done * (len(jobs) - done)
                sys.stdout.write(
                    f"\r  {done}/{len(jobs)}  {el:7.1f} s elapsed, "
                    f"{eta:7.1f} s left   ")
                sys.stdout.flush()
    print()

    wall = time.perf_counter() - t_wall0
    ru1 = os.times()
    cpu_children = (ru1.children_user - ru0.children_user
                    + ru1.children_system - ru0.children_system)
    ended = _dt.datetime.now()

    # --- recompute every cost, independently ------------------------------
    rows, errors, log = [], [], []
    total_cost = 0.0
    solved = 0
    worst_rel = 0.0
    for k, (n, cap, xs, ys, ds) in enumerate(insts):
        r = results[k]
        if r is None or "error" in r:
            msg = r["error"] if r else "no result"
            errors.append(f"instance {k}: {msg}")
            rows.append({"idx": k, "name": r["name"] if r else "", "n": n,
                         "cost": "", "routes": "", "wall_s": "",
                         "hgs_cost": "", "status": "FAILED"})
            continue
        cost, nroutes, problems = recompute(r["routes"], xs, ys, ds, cap, n,
                                            args.rounded)
        rel = abs(cost - r["announced"]) / (cost if cost > 0 else 1.0)
        worst_rel = max(worst_rel, rel)
        if rel > 1e-5:
            problems.append(
                f"HGS announced {r['announced']}, recomputed {cost:.10f} "
                f"(rel {rel:.2e})")
        if problems:
            errors.extend(f"instance {k}: {p}" for p in problems)
        if r["stderr"]:
            log.append(f"[{k}] {r['stderr']}")
        total_cost += cost
        solved += 1
        rows.append({"idx": k, "name": r["name"], "n": n,
                     "cost": f"{cost:.17g}", "routes": nroutes,
                     "wall_s": f"{r['wall']:.4f}",
                     "hgs_cost": r["announced"],
                     "status": "ok" if not problems else "CHECK"})

    mean = total_cost / solved if solved else float("nan")

    # --- solutions.txt, in the `cw --sol` format --------------------------
    solpath = os.path.join(run_dir, "solutions.txt")
    with open(solpath, "w") as f:
        f.write(f"#CWSOL 1\n#instances {len(insts)}\n")
        f.write(f"#source {args.bundle}\n")
        f.write(f"#round {args.rounded}\n")
        f.write(f"#solver HGS-CVRP {' '.join(argv)}\n")
        f.write("#format inst <idx> <name> <n> <Q> <cost> <routes>, "
                "then one line per route\n")
        for k, (n, cap, xs, ys, ds) in enumerate(insts):
            r = results[k]
            if r is None or "error" in r:
                continue
            nonempty = [rt for rt in r["routes"] if rt]
            cost = float(next(row["cost"] for row in rows if row["idx"] == k))
            f.write(f"inst {k} {r['name']} {n} {cap:.17g} {cost:.17g} "
                    f"{len(nonempty)}\n")
            for rt in nonempty:
                f.write(" ".join(str(c) for c in rt) + "\n")

    with open(os.path.join(run_dir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "name", "n", "cost", "routes",
                                          "wall_s", "hgs_cost", "status"])
        w.writeheader()
        w.writerows(rows)

    with open(os.path.join(run_dir, "run.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(log + errors) + "\n")

    config = {
        "run": {
            "id": os.path.basename(run_dir),
            "name": args.name,
            "solver": "HGS-CVRP",
            "started": started.isoformat(timespec="seconds"),
            "ended": ended.isoformat(timespec="seconds"),
            "wall_s": round(wall, 3),
            "cpu_s": round(cpu_children, 3),
            "jobs": args.jobs,
        },
        "command": f"{args.hgs} <instance> <sol> {' '.join(argv)}",
        # read by validate.py to locate the instances
        "cw_args": ["--bundle", os.path.relpath(os.path.abspath(args.bundle), ROOT)],
        "hgs_args": argv,
        "hgs_commit": hgs_version(os.path.join(ROOT, "external", "HGS-CVRP")),
        "binary": binary_fingerprint(args.hgs),
        "environment": {
            "host": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
        },
        "result": {
            "instances": len(insts),
            "solved": solved,
            "failed": len(insts) - solved,
            "mean_cost": mean,
            "checks_failed": len(errors),
            "worst_cost_mismatch_rel": worst_rel,
            "wall_s": round(wall, 3),
            "cpu_s": round(cpu_children, 3),
            "cpu_s_per_instance": round(cpu_children / len(insts), 4),
        },
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")

    if not args.keep_sol:
        shutil.rmtree(soldir, ignore_errors=True)

    print()
    print(f"  instances : {solved}/{len(insts)} solved")
    print(f"  mean cost : {mean:.6f}   (recomputed from the coordinates)")
    print(f"  wall      : {wall:.1f} s on {args.jobs} jobs")
    print(f"  cpu       : {cpu_children:.1f} s total, "
          f"{cpu_children / len(insts):.3f} s/instance")
    print(f"  cost agreement with HGS: worst {worst_rel:.2e} relative")
    if errors:
        print(f"  {len(errors)} CHECK FAILURE(S) — see run.log", file=sys.stderr)
        for e in errors[:5]:
            print(f"    {e}", file=sys.stderr)
    print(f"\n-> {run_dir}")
    print(f"   validate with: python3 tools/validate.py {os.path.relpath(run_dir, ROOT)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
