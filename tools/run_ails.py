#!/usr/bin/env python3
"""
run_ails.py — run AILS-II over a CVRP instance set, in the run-directory
layout tools/validate.py already understands.

    Máximo, Cordeau, Nascimento, "AILS-II: An Adaptive Iterated Local Search
    Heuristic for the Large-scale Capacitated Vehicle Routing Problem",
    INFORMS Journal on Computing (2023).  https://github.com/INFORMSJoC/2023.0106

Same contract as run_hgs.py and run_lkh.py, so `validate.py` and
`gap_to_bks.py` work on an AILS-II run unchanged.

Setup — the repository ships Java sources, no jar:

    git clone https://github.com/INFORMSJoC/2023.0106.git external/AILS-II
    patch -p0 -d external/AILS-II < tools/ails_solution_output.patch
    cd external/AILS-II && mkdir -p build \\
      && javac -nowarn -d build $(find src -name '*.java') \\
      && (cd build && jar cfe ../AILSII.jar SearchMethod.AILSII .)

The patch adds a `-solution <file>` option. Without it AILS-II prints only a
cost line, which would make its numbers the only ones here that could not be
independently revalidated. It touches nothing but main(); see the patch header.

Three things about AILS-II shape this driver:

  * **Never pass `-best`.** It is a stopping criterion, not an annotation:
    the search halts as soon as the cost reaches it. Supplying a BKS would let
    AILS-II stop the instant it matches, making every timing meaningless. The
    default of 0 is unreachable, so the budget is what binds.

  * **`-limit` is wall-clock**, not CPU (the search polls
    System.currentTimeMillis). Running J instances at once therefore gives
    each of them less real work than a solo run, unlike HGS's `-t`, which is a
    CPU budget and is unaffected. The two are only comparable if this is kept
    in mind; both figures are recorded, and --jobs 1 is the honest setting for
    a timing comparison.

  * **`-rounded`** selects the distance convention, so unlike LKH-3 this
    solver runs the continuous NeuOpt instances natively. It is inferred from
    the coordinates unless forced, and the decision is always printed.

Usage:
    python3 tools/run_ails.py --dir data/cvrplib/X --time 10 --jobs 12
    python3 tools/run_ails.py --bundle data/cvrp_100.cvrpb --limit 200 --time 1
"""

import argparse
import concurrent.futures as cf
import csv
import datetime as _dt
import glob
import json
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
DEFAULT_JAR = os.path.join(ROOT, "external", "AILS-II", "AILSII.jar")

sys.path.insert(0, HERE)
from _common import binary_fingerprint, recompute  # noqa: E402
from fetch_cvrplib import read_vrp, read_sol  # noqa: E402
from bundle_to_vrp import read_bundle, write_vrp, instance_name  # noqa: E402


def find_java(explicit):
    """A java launcher, wherever this machine keeps one.

    external/jdk is checked first: that is where setup.sh drops a JDK when the
    system has none and there is no root to install one. After that, $JAVA_HOME,
    PATH, then the usual per-platform locations — Homebrew on macOS,
    /usr/lib/jvm on Debian/Ubuntu/Fedora, /usr/java on RHEL.
    """
    if explicit:
        return explicit
    local = os.path.join(ROOT, "external", "jdk", "bin", "java")
    if os.path.exists(local):
        return local
    home = os.environ.get("JAVA_HOME")
    if home and os.path.exists(os.path.join(home, "bin", "java")):
        return os.path.join(home, "bin", "java")
    found = shutil.which("java")
    if found:
        return found
    for pat in ("/opt/homebrew/opt/openjdk*/bin/java",     # macOS, arm64
                "/usr/local/opt/openjdk*/bin/java",        # macOS, x86_64
                "/usr/lib/jvm/*/bin/java",                 # Debian, Fedora
                "/usr/java/*/bin/java"):                   # RHEL
        cands = sorted(glob.glob(pat))
        if cands:
            return cands[-1]
    return "java"


def solve_one(job):
    idx, name, vrp, soldir, java, jar, heap, rounded, limit = job
    out = os.path.join(soldir, name + ".sol")
    cmd = [java, f"-Xmx{heap}", "-jar", jar,
           "-file", os.path.abspath(vrp),
           "-rounded", "true" if rounded else "false",
           "-stoppingCriterion", "Time", "-limit", str(limit),
           "-solution", out]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=max(600, limit * 20))
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"idx": idx, "name": name, "wall": time.perf_counter() - t0,
                "error": f"launch failed: {e}"}
    wall = time.perf_counter() - t0
    if not os.path.exists(out):
        return {"idx": idx, "name": name, "wall": wall,
                "error": f"no solution file (exit {p.returncode}): "
                         f"{(p.stdout + p.stderr).strip()[-200:]}"}
    try:
        routes, cost = read_sol(out)
    except (OSError, ValueError) as e:
        return {"idx": idx, "name": name, "wall": wall, "error": str(e)}
    if cost is None:
        return {"idx": idx, "name": name, "wall": wall,
                "error": "solution file has no Cost line"}
    return {"idx": idx, "name": name, "wall": wall, "routes": routes,
            "announced": cost, "stderr": p.stderr.strip()}


def main():
    ap = argparse.ArgumentParser(
        description="Run AILS-II over a CVRP set into a validatable run directory")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dir", dest="indir", help="directory of TSPLIB .vrp")
    src.add_argument("--bundle", help="instances (.cvrpb)")
    ap.add_argument("--jar", default=DEFAULT_JAR, help="AILSII.jar")
    ap.add_argument("--java", default=None, help="java binary")
    ap.add_argument("--heap", default="2g", help="JVM -Xmx per job (default 2g)")
    ap.add_argument("--limit", type=int, default=0,
                    help="solve only the first N instances")
    ap.add_argument("--time", type=float, default=10.0, metavar="SEC",
                    help="AILS-II -limit: WALL-clock seconds per instance")
    ap.add_argument("--round", dest="rounded", default="auto",
                    choices=("auto", "0", "1"),
                    help="integer distances; auto infers from the coordinates")
    ap.add_argument("--jobs", type=int, default=os.cpu_count(),
                    help="parallel JVMs. Note -limit is wall-clock: use 1 for "
                         "a timing comparison")
    ap.add_argument("--name", help="readable suffix for the run directory")
    ap.add_argument("--out", default=RESULTS)
    ap.add_argument("--keep-sol", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    java = find_java(args.java)
    if not os.path.exists(args.jar):
        ap.error(f"{args.jar} not found — see the build steps in this file's "
                 f"docstring (clone, patch, javac, jar)")
    try:
        subprocess.run([java, "-version"], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        ap.error(f"no working java at {java!r}. Install a JDK "
                 f"(apt install default-jdk / dnf install java-latest-openjdk-devel / "
                 f"brew install openjdk), or let `sh tools/setup.sh` drop one in "
                 f"external/jdk, or pass --java")

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
    if args.rounded == "auto":
        rounded = 1 if integral else 0
        why = ("integral coordinates -> TSPLIB EUC_2D convention" if integral
               else "continuous coordinates -> no rounding")
    else:
        rounded = int(args.rounded)
        why = "forced"
        if rounded and not integral:
            ap.error("--round 1 on continuous coordinates would collapse every "
                     "distance")

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", args.name).strip("-") if args.name else ""
    run_dir = os.path.join(args.out, f"{stamp}_{slug}" if slug else stamp)

    n_all = {n for n, _, _, _, _ in insts}
    nspan = (f"{min(n_all)}..{max(n_all)}" if len(n_all) > 1
             else str(next(iter(n_all))))
    print(f"source    : {source}  ({len(insts)} instances, n = {nspan})")
    print(f"ails      : {java} -Xmx{args.heap} -jar {args.jar}")
    print(f"rounding  : -rounded {'true' if rounded else 'false'}   ({why})")
    print(f"budget    : -limit {args.time} s of WALL clock per instance")
    print(f"jobs      : {args.jobs}"
          + ("   (wall-clock budget under contention: each job gets less real "
             "work than a solo run)" if args.jobs > 1 else ""))
    print(f"directory : {run_dir}")
    if args.dry_run:
        return 0

    os.makedirs(run_dir, exist_ok=True)
    soldir = (os.path.join(run_dir, "sol") if args.keep_sol
              else tempfile.mkdtemp(prefix="ails-"))
    os.makedirs(soldir, exist_ok=True)

    jobs = [(k, names[k], os.path.join(vrpdir, names[k] + ".vrp"), soldir,
             java, args.jar, args.heap, rounded, args.time)
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
    solved = 0
    worst_rel = 0.0
    for k, (n, cap, xs, ys, ds) in enumerate(insts):
        r = results[k]
        if r is None or "error" in r:
            errors.append(f"instance {k} ({names[k]}): "
                          f"{r['error'] if r else 'no result'}")
            rows.append({"idx": k, "name": names[k], "n": n, "cost": "",
                         "routes": "", "wall_s": "", "ails_cost": "",
                         "status": "FAILED"})
            continue
        cost, nroutes, problems = recompute(r["routes"], xs, ys, ds, cap, n,
                                            rounded)
        rel = abs(cost - r["announced"]) / (cost if cost > 0 else 1.0)
        worst_rel = max(worst_rel, rel)
        if rel > 1e-6:
            problems.append(f"AILS-II announced {r['announced']}, recomputed "
                            f"{cost:.4f} (rel {rel:.2e})")
        if problems:
            errors.extend(f"instance {k} ({names[k]}): {p}" for p in problems)
        if r.get("stderr"):
            log.append(f"[{k}] {r['stderr']}")
        total += cost
        solved += 1
        rows.append({"idx": k, "name": names[k], "n": n, "cost": f"{cost:.17g}",
                     "routes": nroutes, "wall_s": f"{r['wall']:.4f}",
                     "ails_cost": r["announced"],
                     "status": "ok" if not problems else "CHECK"})

    mean = total / solved if solved else float("nan")

    with open(os.path.join(run_dir, "solutions.txt"), "w") as f:
        f.write(f"#CWSOL 1\n#instances {len(insts)}\n#source {source}\n")
        f.write(f"#round {rounded}\n")
        f.write(f"#solver AILS-II -stoppingCriterion Time -limit {args.time}\n")
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
                                          "wall_s", "ails_cost", "status"])
        w.writeheader()
        w.writerows(rows)

    with open(os.path.join(run_dir, "run.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(log + errors) + "\n")

    config = {
        "run": {"id": os.path.basename(run_dir), "name": args.name,
                "solver": "AILS-II",
                "started": started.isoformat(timespec="seconds"),
                "ended": ended.isoformat(timespec="seconds"),
                "wall_s": round(wall, 3), "cpu_s": round(cpu, 3),
                "jobs": args.jobs},
        "command": f"{java} -Xmx{args.heap} -jar {args.jar} -file <instance> "
                   f"-rounded {'true' if rounded else 'false'} "
                   f"-stoppingCriterion Time -limit {args.time} -solution <sol>",
        "cw_args": (["--bundle", os.path.relpath(os.path.abspath(bundle_ref), ROOT)]
                    if bundle_ref else ["--dir", source]),
        "budget_is": "wall clock, not CPU",
        "patched": "tools/ails_solution_output.patch (adds -solution; "
                   "no algorithmic change)",
        "binary": binary_fingerprint(args.jar, root=ROOT),
        "environment": {"host": platform.node(), "platform": platform.platform(),
                        "machine": platform.machine(),
                        "cpu_count": os.cpu_count()},
        "result": {"instances": len(insts), "solved": solved,
                   "failed": len(insts) - solved, "mean_cost": mean,
                   "checks_failed": len(errors),
                   "worst_cost_mismatch_rel": worst_rel,
                   "wall_s": round(wall, 3), "cpu_s": round(cpu, 3),
                   "cpu_s_per_instance": round(cpu / len(insts), 4)},
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")

    if not args.keep_sol:
        shutil.rmtree(soldir, ignore_errors=True)

    print()
    print(f"  instances : {solved}/{len(insts)} solved")
    print(f"  mean cost : {mean:.2f}   (recomputed from the coordinates)")
    print(f"  wall      : {wall:.1f} s on {args.jobs} jobs")
    print(f"  cpu       : {cpu:.1f} s total, {cpu / len(insts):.3f} s/instance")
    print(f"  cost agreement with AILS-II: worst {worst_rel:.2e} relative")
    if errors:
        print(f"  {len(errors)} CHECK FAILURE(S) — see run.log", file=sys.stderr)
        for e in errors[:5]:
            print(f"    {e}", file=sys.stderr)
    print(f"\n-> {run_dir}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
