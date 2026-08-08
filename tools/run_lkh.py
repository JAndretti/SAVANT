#!/usr/bin/env python3
"""
run_lkh.py — run LKH-3 over a CVRP instance set, in the run-directory layout
tools/validate.py already understands.

Same contract as tools/run_hgs.py: one directory per run holding
`solutions.txt`, `results.csv`, `config.json` and `run.log`, so

    python3 tools/validate.py results/<run>
    python3 tools/gap_to_bks.py results/<run> data/cvrplib/X_bks.csv

work on an LKH-3 run exactly as they do on a SAVANT or HGS one.

Four things about LKH-3 shape this driver:

  * **It needs a fixed fleet size.** `VEHICLES` is not an upper bound, it is
    the number of routes LKH-3 will produce. The XL paper sets it to the `k`
    in the instance name; this script does the same when the name carries one
    (X-n101-k25 -> 25) and otherwise falls back to the bin-packing lower bound
    ceil(total demand / capacity), which is the loosest value that can still
    admit a feasible solution. Either way the choice is recorded per instance.

  * **`SPECIAL` matters enormously.** It is a shorthand for LKH-3's purpose-
    built CVRP moves (GAIN23 = NO, KICKS = 1, KICK_TYPE = 4, MAX_SWAPS = 0,
    MOVE_TYPE = 5 SPECIAL, POPULATION_SIZE = 10). Without it, LKH-3 spent 68 s
    on X-n101-k25 without ever reaching feasibility; with it, feasible in 4 s.

  * **A solution can come back infeasible.** LKH-3 reports `Cost = P_C` where
    P is a penalty: P > 0 means capacity is violated and the answer is not a
    CVRP solution at all. It still writes the file and exits 0, so the penalty
    is checked explicitly and a penalised solution is recorded as a failure
    rather than averaged into the mean.

  * **Distances are integers.** LKH-3 computes EUC_2D by TSPLIB rules, which
    round. There is no float mode, so the continuous NeuOpt instances cannot
    be run without rescaling the coordinates, and this script refuses them
    rather than silently answering a different problem.

Usage:
    python3 tools/run_lkh.py --dir data/cvrplib/X --time 10 --jobs 12
    python3 tools/run_lkh.py --dir data/cvrplib/X --limit 30 --time 10 \\
        --name LKH_X_smoke
"""

import argparse
import concurrent.futures as cf
import csv
import datetime as _dt
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")

sys.path.insert(0, HERE)
from _common import binary_fingerprint, recompute  # noqa: E402
from fetch_cvrplib import read_vrp  # noqa: E402
from bundle_to_vrp import read_bundle, write_vrp, instance_name  # noqa: E402

DEFAULT_LKH = None
for _cand in sorted(os.listdir(os.path.join(ROOT, "external")))[::-1] \
        if os.path.isdir(os.path.join(ROOT, "external")) else []:
    _p = os.path.join(ROOT, "external", _cand, "LKH")
    if os.path.exists(_p):
        DEFAULT_LKH = _p
        break
DEFAULT_LKH = DEFAULT_LKH or os.path.join(ROOT, "external", "LKH-3.0.14", "LKH")

# "X-n101-k25, Cost: 0_27591"   ->   penalty, cost
COST_RE = re.compile(r"Cost:\s*(?:(\d+)_)?(\d+)")
# "1 51 92 53 1 (#3)  Cost: 1065"
ROUTE_RE = re.compile(r"^\s*1((?:\s+\d+)+)\s+1\s*\(#\d+\)")

K_IN_NAME_RE = re.compile(r"-k(\d+)\s*$")


def fleet_size(name, cap, ds):
    """(VEHICLES, how it was chosen)."""
    m = K_IN_NAME_RE.search(name)
    if m:
        return int(m.group(1)), "name"
    return max(1, math.ceil(sum(ds) / cap)), "bin-packing LB"


def parse_mtsp(path):
    """(routes, penalty, cost) from an MTSP_SOLUTION_FILE.

    Node numbers are TSPLIB's: 1 is the depot and customer i is node i+1, so
    every index is shifted back by one to reach the numbering `solutions.txt`
    and validate.py use.
    """
    routes, penalty, cost = [], None, None
    with open(path) as f:
        for line in f:
            if cost is None:
                m = COST_RE.search(line)
                if m and "(#" not in line:
                    penalty = int(m.group(1) or 0)
                    cost = int(m.group(2))
                    continue
            m = ROUTE_RE.match(line)
            if m:
                routes.append([int(v) - 1 for v in m.group(1).split()])
    if cost is None:
        raise ValueError(f"{path}: no cost header")
    return routes, penalty, cost


def solve_one(job):
    """One LKH-3 run. Never raises."""
    idx, name, vrp, veh, workdir, lkh, opts = job
    par = os.path.join(workdir, name + ".par")
    out = os.path.join(workdir, name + ".mtsp")
    with open(par, "w") as f:
        f.write(f"PROBLEM_FILE = {os.path.abspath(vrp)}\n")
        f.write("SPECIAL\n")
        f.write(f"VEHICLES = {veh}\n")
        f.write(f"MTSP_SOLUTION_FILE = {out}\n")
        f.write("TRACE_LEVEL = 0\n")
        for k, v in opts:
            f.write(f"{k} = {v}\n")

    t0 = time.perf_counter()
    try:
        p = subprocess.run([lkh, par], capture_output=True, text=True,
                           timeout=7200)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"idx": idx, "name": name, "veh": veh,
                "wall": time.perf_counter() - t0, "error": f"launch failed: {e}"}
    wall = time.perf_counter() - t0

    if not os.path.exists(out):
        return {"idx": idx, "name": name, "veh": veh, "wall": wall,
                "error": f"no solution file (exit {p.returncode}): "
                         f"{(p.stdout + p.stderr).strip()[-200:]}"}
    try:
        routes, penalty, cost = parse_mtsp(out)
    except (OSError, ValueError) as e:
        return {"idx": idx, "name": name, "veh": veh, "wall": wall,
                "error": str(e)}
    if penalty:
        # Capacity violated: LKH-3 still writes the file and exits 0.
        return {"idx": idx, "name": name, "veh": veh, "wall": wall,
                "penalty": penalty,
                "error": f"infeasible: LKH-3 penalty {penalty} "
                         f"(capacity violated) with VEHICLES = {veh}"}
    return {"idx": idx, "name": name, "veh": veh, "wall": wall,
            "routes": routes, "announced": cost, "penalty": 0,
            "stderr": p.stderr.strip()}


def main():
    ap = argparse.ArgumentParser(
        description="Run LKH-3 over a CVRP set into a validatable run directory")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dir", dest="indir", help="directory of TSPLIB .vrp")
    src.add_argument("--bundle", help="instances (.cvrpb)")
    ap.add_argument("--lkh", default=DEFAULT_LKH, help="LKH binary")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--time", type=float, default=10.0, metavar="SEC",
                    help="LKH TIME_LIMIT, per run (default 10)")
    ap.add_argument("--runs", type=int, default=1, help="LKH RUNS (default 1)")
    ap.add_argument("--trials", type=int, default=0,
                    help="LKH MAX_TRIALS. Default: 1000000 whenever --time is "
                         "set, so that the time limit is what actually binds "
                         "(LKH's own default is DIMENSION, which on Set X ends "
                         "a run after ~2 s of a 10 s budget). Pass a value to "
                         "override, or --trials -1 for LKH's default")
    ap.add_argument("--seed", type=int, default=1, help="LKH SEED (default 1)")
    ap.add_argument("--veh", type=int, default=0,
                    help="force VEHICLES for every instance (default: the k in "
                         "the instance name, else the bin-packing lower bound)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--name", help="readable suffix for the run directory")
    ap.add_argument("--out", default=RESULTS)
    ap.add_argument("--keep-sol", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.lkh):
        ap.error(f"{args.lkh} not found — build it with:\n"
                 f"  curl -O http://webhotel4.ruc.dk/~keld/research/LKH-3/"
                 f"LKH-3.0.14.tgz\n"
                 f"  tar xzf LKH-3.0.14.tgz -C external/ && "
                 f"make -C external/LKH-3.0.14")

    # --- instances -------------------------------------------------------
    if args.indir:
        names = sorted(f[:-4] for f in os.listdir(args.indir)
                       if f.endswith(".vrp"))
        if args.limit:
            names = names[:args.limit]
        if not names:
            raise SystemExit(f"{args.indir}: no .vrp file")
        insts = [read_vrp(os.path.join(args.indir, nm + ".vrp")) for nm in names]
        vrpdir, source = args.indir, args.indir
        cand = os.path.normpath(args.indir.rstrip("/")) + ".cvrpb"
        bundle_ref = cand if os.path.exists(cand) else None
    else:
        insts = read_bundle(args.bundle, args.limit)
        stem = os.path.splitext(os.path.basename(args.bundle))[0]
        width = len(str(len(insts) - 1))
        names = [instance_name(stem, k, width) for k in range(len(insts))]
        vrpdir = os.path.join(ROOT, "data", "vrp_" + stem.split("_")[-1])
        os.makedirs(vrpdir, exist_ok=True)
        for k, (n, cap, xs, ys, ds) in enumerate(insts):
            path = os.path.join(vrpdir, names[k] + ".vrp")
            if not os.path.exists(path):
                write_vrp(path, names[k], n, cap, xs, ys, ds)
        source, bundle_ref = args.bundle, args.bundle

    integral = all(
        all(abs(v - round(v)) < 1e-9 for v in xs[:8] + ys[:8])
        for _, _, xs, ys, _ in insts[:8]
    )
    if not integral:
        ap.error("these instances have continuous coordinates. LKH-3 computes "
                 "EUC_2D by TSPLIB rules, which round to the nearest integer, "
                 "and has no float mode — running it here would answer a "
                 "different problem. Rescale the coordinates first (e.g. x1e6) "
                 "if you want an LKH-3 number on the NeuOpt sets.")

    fleets = []
    for k, (n, cap, xs, ys, ds) in enumerate(insts):
        if args.veh:
            fleets.append((args.veh, "forced"))
        else:
            fleets.append(fleet_size(names[k], cap, ds))

    opts = [("RUNS", args.runs), ("SEED", args.seed)]
    if args.time > 0:
        opts.append(("TIME_LIMIT", args.time))
    # MAX_TRIALS defaults to DIMENSION, which terminates a run long before a
    # time limit does — on Set X, 2.1 of a 10 s budget, and a mean gap of
    # +4.01 % instead of +2.90 %. A time budget is only a time budget if the
    # trial count is taken out of the way.
    trials = args.trials
    if trials == 0 and args.time > 0:
        trials = 1000000
    if trials > 0:
        opts.append(("MAX_TRIALS", trials))

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", args.name).strip("-") if args.name else ""
    run_dir = os.path.join(args.out, f"{stamp}_{slug}" if slug else stamp)

    n_all = {n for n, _, _, _, _ in insts}
    nspan = (f"{min(n_all)}..{max(n_all)}" if len(n_all) > 1
             else str(next(iter(n_all))))
    from_name = sum(1 for _, how in fleets if how == "name")
    print(f"source    : {source}  ({len(insts)} instances, n = {nspan})")
    print(f"lkh       : {args.lkh}  SPECIAL, "
          + ", ".join(f"{k} = {v}" for k, v in opts))
    print(f"vehicles  : {from_name}/{len(fleets)} taken from the instance name, "
          f"the rest from the bin-packing bound")
    print(f"jobs      : {args.jobs}")
    print(f"directory : {run_dir}")
    if args.dry_run:
        return 0

    os.makedirs(run_dir, exist_ok=True)
    workdir = (os.path.join(run_dir, "lkh") if args.keep_sol
               else tempfile.mkdtemp(prefix="lkh-"))
    os.makedirs(workdir, exist_ok=True)

    jobs = [(k, names[k], os.path.join(vrpdir, names[k] + ".vrp"),
             fleets[k][0], workdir, args.lkh, opts)
            for k in range(len(insts))]

    started = _dt.datetime.now()
    t0 = time.perf_counter()
    ru0 = os.times()
    results = [None] * len(jobs)
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for res in ex.map(solve_one, jobs):
            results[res["idx"]] = res
            done += 1
            if done % 10 == 0 or done == len(jobs):
                el = time.perf_counter() - t0
                sys.stdout.write(f"\r  {done}/{len(jobs)}  {el:7.1f} s elapsed, "
                                 f"{el / done * (len(jobs) - done):7.1f} s left   ")
                sys.stdout.flush()
    print()
    wall = time.perf_counter() - t0
    ru1 = os.times()
    cpu = (ru1.children_user - ru0.children_user
           + ru1.children_system - ru0.children_system)
    ended = _dt.datetime.now()

    rows, errors, log = [], [], []
    total = 0.0
    solved = infeasible = 0
    worst_rel = 0.0
    for k, (n, cap, xs, ys, ds) in enumerate(insts):
        r = results[k]
        if r is None or "error" in r:
            msg = r["error"] if r else "no result"
            errors.append(f"instance {k} ({names[k]}): {msg}")
            if r and r.get("penalty"):
                infeasible += 1
            rows.append({"idx": k, "name": names[k], "n": n, "cost": "",
                         "routes": "", "vehicles": fleets[k][0], "wall_s": "",
                         "lkh_cost": "", "status": "FAILED"})
            continue
        cost, nroutes, problems = recompute(r["routes"], xs, ys, ds, cap, n,
                                            rounded=True)
        rel = abs(cost - r["announced"]) / (cost if cost > 0 else 1.0)
        worst_rel = max(worst_rel, rel)
        if rel > 1e-9:
            problems.append(f"LKH announced {r['announced']}, recomputed "
                            f"{cost:.1f} (rel {rel:.2e})")
        if problems:
            errors.extend(f"instance {k} ({names[k]}): {p}" for p in problems)
        if r.get("stderr"):
            log.append(f"[{k}] {r['stderr']}")
        total += cost
        solved += 1
        rows.append({"idx": k, "name": names[k], "n": n, "cost": f"{cost:.17g}",
                     "routes": nroutes, "vehicles": fleets[k][0],
                     "wall_s": f"{r['wall']:.4f}", "lkh_cost": r["announced"],
                     "status": "ok" if not problems else "CHECK"})

    mean = total / solved if solved else float("nan")

    with open(os.path.join(run_dir, "solutions.txt"), "w") as f:
        f.write(f"#CWSOL 1\n#instances {len(insts)}\n#source {source}\n")
        f.write("#round 1\n")
        f.write(f"#solver LKH-3 SPECIAL "
                + " ".join(f"{k}={v}" for k, v in opts) + "\n")
        f.write("#format inst <idx> <name> <n> <Q> <cost> <routes>, "
                "then one line per route\n")
        for k, (n, cap, xs, ys, ds) in enumerate(insts):
            r = results[k]
            if r is None or "error" in r:
                continue
            nonempty = [rt for rt in r["routes"] if rt]
            cost = float(next(x["cost"] for x in rows if x["idx"] == k))
            f.write(f"inst {k} {names[k]} {n} {cap:.17g} {cost:.17g} "
                    f"{len(nonempty)}\n")
            for rt in nonempty:
                f.write(" ".join(str(c) for c in rt) + "\n")

    with open(os.path.join(run_dir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "name", "n", "cost", "routes",
                                          "vehicles", "wall_s", "lkh_cost",
                                          "status"])
        w.writeheader()
        w.writerows(rows)

    with open(os.path.join(run_dir, "run.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(log + errors) + "\n")

    config = {
        "run": {"id": os.path.basename(run_dir), "name": args.name,
                "solver": "LKH-3",
                "started": started.isoformat(timespec="seconds"),
                "ended": ended.isoformat(timespec="seconds"),
                "wall_s": round(wall, 3), "cpu_s": round(cpu, 3),
                "jobs": args.jobs},
        "command": f"{args.lkh} <instance>.par   [SPECIAL, "
                   + ", ".join(f"{k} = {v}" for k, v in opts) + "]",
        "cw_args": (["--bundle", os.path.relpath(os.path.abspath(bundle_ref), ROOT)]
                    if bundle_ref else ["--dir", source]),
        "lkh_par": {"SPECIAL": True, **{k: v for k, v in opts}},
        "vehicles_source": ("instance name where present, else the "
                            "bin-packing lower bound"),
        "binary": binary_fingerprint(args.lkh, root=ROOT),
        "environment": {"host": platform.node(), "platform": platform.platform(),
                        "machine": platform.machine(),
                        "cpu_count": os.cpu_count()},
        "result": {"instances": len(insts), "solved": solved,
                   "failed": len(insts) - solved, "infeasible": infeasible,
                   "mean_cost": mean, "checks_failed": len(errors),
                   "worst_cost_mismatch_rel": worst_rel,
                   "wall_s": round(wall, 3), "cpu_s": round(cpu, 3),
                   "cpu_s_per_instance": round(cpu / len(insts), 4)},
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")

    if not args.keep_sol:
        shutil.rmtree(workdir, ignore_errors=True)

    print()
    print(f"  instances : {solved}/{len(insts)} solved"
          + (f", {infeasible} infeasible (nonzero LKH penalty)"
             if infeasible else ""))
    print(f"  mean cost : {mean:.2f}   (recomputed from the coordinates)")
    print(f"  wall      : {wall:.1f} s on {args.jobs} jobs")
    print(f"  cpu       : {cpu:.1f} s total, {cpu / len(insts):.3f} s/instance")
    print(f"  cost agreement with LKH: worst {worst_rel:.2e} relative")
    if errors:
        print(f"  {len(errors)} CHECK FAILURE(S) — see run.log", file=sys.stderr)
        for e in errors[:5]:
            print(f"    {e}", file=sys.stderr)
    print(f"\n-> {run_dir}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
