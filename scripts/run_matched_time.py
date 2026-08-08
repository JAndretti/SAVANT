#!/usr/bin/env python3
"""run_matched_time.py — HGS-CVRP, LKH-3 and AILS-II at SAVANT's own budget.

scripts/run_best_config.sh established what the recommended configuration
reaches on each dataset, and how much CPU it spent getting there. This script
hands that same budget to the three reference solvers, on the same instances,
and validates every run with the same independent checker.

The budget
----------
For each set, the reference is SAVANT's best run at the budget it was last run
at: among the runs sharing the newest one's --sa-time / --sa-steps setting, the
one with the lower mean cost. Its budget is read from config.json as

    cumulative CPU seconds / instances

and passed to each solver as its per-instance budget — what SAVANT actually
spent, not what it was asked for. CPU rather than wall clock is the only figure
that makes the four commensurable: SAVANT spreads one instance-parallel OpenMP
loop over every core, while the other three are single-threaded processes run J
at a time.

Since run_best_config.sh drives SAVANT with --sa-time, the protocol is no
longer one-directional: SAVANT is now *given* a per-instance budget like
everyone else, and the requested-vs-achieved ratio is printed for it too. That
also all but removes the caveat that used to matter most here — see below.

Three caveats this cannot paper over, all reported rather than hidden:

  * **AILS-II's -limit is wall clock**, not CPU (it polls
    currentTimeMillis). With J jobs at once each JVM gets less real work than
    the budget says, so AILS-II defaults to --ails-jobs 1 here. HGS's -t
    (clock(), Genetic.cpp) and LKH-3's TIME_LIMIT (getrusage, GetTime.c) are
    both CPU budgets and are unaffected by --jobs.
  * **The budget is still a per-set mean**, because every driver takes one
    scalar. Under --sa-time that is now nearly exact: SAVANT spends the same
    time on every instance by construction, so the mean describes the set
    instead of averaging over a 10x spread the way a fixed step count did on X
    and XL (scripts/timing_profile.py). What is left is the calibration's own
    error, which the printed spread measures.
  * **LKH-3 has no float mode**: EUC_2D is rounded by TSPLIB rules, so the
    continuous NeuOpt sets (n20, n50, n100) cannot be run without changing the
    problem. They are skipped for LKH-3 and reported as skipped.

Usage
-----
    uv run --no-project scripts/run_matched_time.py                 # everything
    uv run --no-project scripts/run_matched_time.py --ails-jobs 8   # AILS-II dominates
    uv run --no-project scripts/run_matched_time.py --sets X XL --solvers hgs lkh
    uv run --no-project scripts/run_matched_time.py --savant-config cand
    uv run --no-project scripts/run_matched_time.py --dry-run       # print, run nothing

Runs land in results/matched_time/<stamp>_<SOLVER>_<set>/, each validated into
validation.txt and, on the CVRPLib sets, scored into bks_gap.txt. Then:

    uv run --no-project scripts/summarize_matched.py
"""

import argparse
import csv
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))  # scripts/
ROOT = os.path.dirname(HERE)  # repository root
TOOLS = os.path.join(ROOT, "tools")

SAVANT_RUNS = os.path.join(ROOT, "results", "best_config")
OUT = os.path.join(ROOT, "results", "matched_time")

# --jobs is per set, not global: HGS holds a dense n x n double matrix, ~2.3 GiB
# at n = 10000, so XL cannot run 12 of them at once (same choice as
# tools/compare_all.sh). `heap` is the JVM -Xmx handed to AILS-II.
SETS = {
    "n20": dict(bundle="data/cvrp_20.cvrpb", vrpdir=None, bks=None, jobs=12, heap="2g"),
    "n50": dict(bundle="data/cvrp_50.cvrpb", vrpdir=None, bks=None, jobs=12, heap="2g"),
    "n100": dict(
        bundle="data/cvrp_100.cvrpb", vrpdir=None, bks=None, jobs=12, heap="2g"
    ),
    "n200": dict(
        bundle="data/cvrp_200.cvrpb", vrpdir=None, bks=None, jobs=12, heap="2g"
    ),
    "XML100": dict(
        bundle="data/cvrplib/XML100.cvrpb",
        vrpdir="data/cvrplib/XML100",
        bks="data/cvrplib/XML100_bks.csv",
        jobs=12,
        heap="2g",
    ),
    "X": dict(
        bundle="data/cvrplib/X.cvrpb",
        vrpdir="data/cvrplib/X",
        bks="data/cvrplib/X_bks.csv",
        jobs=12,
        heap="4g",
    ),
    "XL": dict(
        bundle="data/cvrplib/XL.cvrpb",
        vrpdir="data/cvrplib/XL",
        bks="baseline/xl_bks.csv",
        jobs=4,
        heap="12g",
    ),
}
DEFAULT_SETS = ["n20", "n50", "n100", "XML100", "X", "XL"]

SOLVERS = {
    "hgs": dict(
        driver="run_hgs.py",
        label="HGS",
        needs_integer=False,
        binary="external/HGS-CVRP/build/hgs",
    ),
    "lkh": dict(
        driver="run_lkh.py",
        label="LKH",
        needs_integer=True,
        binary="external/LKH-3.0.14/LKH",
    ),
    "ails": dict(
        driver="run_ails.py",
        label="AILS",
        needs_integer=False,
        binary="external/AILS-II/AILSII.jar",
    ),
}


# ------------------------------------------------------------------ reference


def budget_key(cfg):
    """How a SAVANT run was budgeted: ('time', S), ('steps', N) or None.

    Two runs are only comparable, and only usable as one reference, when this
    matches: 10,000,000 steps and 0.6 s per instance are different questions.
    A step count is per restart, so it is multiplied out — 625,000 x 16 and
    10,000,000 x 1 are the same budget. A time budget is not: the restarts of
    one instance share it.
    """
    argv = cfg.get("cw_args") or []
    restarts = 1
    if "--restarts" in argv:
        try:
            restarts = max(1, int(argv[argv.index("--restarts") + 1]))
        except (IndexError, ValueError):
            restarts = 1
    for flag, kind in (("--sa-time", "time"), ("--sa-steps", "steps")):
        if flag in argv:
            try:
                v = float(argv[argv.index(flag) + 1])
            except (IndexError, ValueError):
                return None
            return (kind, v * restarts) if kind == "steps" else (kind, v)
    return None


def savant_reference(set_tag, runs_root, config=None):
    """SAVANT's reference run for a set.

    The budget handed to the other solvers has to come from a run that is
    actually the one being compared against, so the choice is made in two
    steps: keep the runs that share the *newest* run's budget setting, then
    take the lowest mean cost among them (ties to the cheaper run). `config`
    forces one configuration instead. Raises when the set was never run.
    """
    cands = []
    for path in sorted(glob.glob(os.path.join(runs_root, f"*_{set_tag}_*"))):
        cfg_path = os.path.join(path, "config.json")
        if not os.path.isdir(path) or not os.path.exists(cfg_path):
            continue
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        name = cfg.get("run", {}).get("name") or ""
        if not name.startswith(set_tag + "_"):
            continue  # *_X_* must not catch *_XL_*
        tag = name.rsplit("_", 1)[1]
        if config and tag != config:
            continue
        res = cfg.get("result", {})
        cost, cpu, inst = (
            res.get("cost_after_sa"),
            res.get("time_cpu_s"),
            res.get("instances"),
        )
        if not (cost and cpu and inst):
            continue
        cands.append(dict(cost=cost, per_inst=cpu / inst, dir=path, cfg=cfg,
                          config=tag, budget=budget_key(cfg)))
    if cands:
        # sorted() walked the timestamps, so the last candidate is the newest
        latest = cands[-1]["budget"]
        same = [c for c in cands if c["budget"] == latest] or cands
        best = min(same, key=lambda c: (c["cost"], c["per_inst"]))
    else:
        best = None
    if best is None:
        raise SystemExit(
            f"{set_tag}: no usable SAVANT run under {os.path.relpath(runs_root, ROOT)}"
            f" — run scripts/run_best_config.sh first"
        )
    res = best["cfg"]["result"]
    kind, value = best["budget"] or (None, None)
    return dict(
        dir=best["dir"],
        cfg=best["cfg"],
        config=best["config"],
        mean_cost=best["cost"],
        budget_s=best["per_inst"],
        requested_s=value if kind == "time" else None,
        steps_mean=res.get("steps_mean"),
        instances=res["instances"],
    )


def time_spread(run_dir, limit):
    """(p05, median, p95, min, max) of a SAVANT run's per-instance time, in ms.

    results.csv's time_ms is wall time measured *inside* the parallel region,
    so it overstates the absolute cost (see the report's Compute cost section)
    — but its shape across instances is what says whether one mean budget is a
    fair description of the set. Only the shape is used here.

    Under --sa-time the shape is the calibration's residual error and should be
    flat; under --sa-steps it is the instances' own cost, which on X and XL
    spans an order of magnitude.
    """
    try:
        with open(
            os.path.join(run_dir, "results.csv"), newline="", encoding="utf-8"
        ) as f:
            t = [float(r["time_ms"]) for r in csv.DictReader(f)]
    except (OSError, KeyError, ValueError):
        return None
    if limit:
        t = t[:limit]
    if not t:
        return None
    t.sort()
    # the extremes of 10,000 instances are not a description of the set: one
    # instance whose calibration chain was timed badly moves min/max and
    # nothing else. The 5-95 % range is what says whether a flat budget fits.
    return (t[len(t) // 20], t[len(t) // 2], t[19 * len(t) // 20], t[0], t[-1])


# ----------------------------------------------------------------- the driver


def is_integer_set(set_tag):
    """CVRPLib sets are TSPLIB EUC_2D (integer); the NeuOpt sets are float."""
    return SETS[set_tag]["bks"] is not None


def limit_bias(set_tag, limit):
    """Warn when --limit takes a biased prefix rather than a sample, or None.

    Every driver's --limit keeps the *first* L instances of the set's own
    order, which is only a valid subsample when that order carries no
    information. It does on the NeuOpt bundles, where instance k is generated
    from seed + k and the instances are i.i.d. by construction. It does not on
    the CVRPLib sets, which are sorted by file name:

      * XML100's names encode the generator's configuration
        (XML100_<depot><customers><demand><routes>_<rep>), so the file order
        is blocked by configuration and a prefix samples a handful of them;
      * X and XL sort lexicographically, which puts X-n1001 before X-n101, so
        a prefix is not even ordered by n.
    """
    spec = SETS[set_tag]
    if not limit or not spec["vrpdir"]:
        return None  # no truncation, or an i.i.d. bundle
    vrpdir = os.path.join(ROOT, spec["vrpdir"])
    if not os.path.isdir(vrpdir):
        return None
    names = sorted(f[:-4] for f in os.listdir(vrpdir) if f.endswith(".vrp"))
    if limit >= len(names):
        return None  # the whole set is run anyway

    groups = {n.rsplit("_", 1)[0] for n in names}
    if len(groups) > 1 and len(groups) < len(names):
        seen = {n.rsplit("_", 1)[0] for n in names[:limit]}
        return (
            f"--limit {limit} keeps {limit}/{len(names)} instances, "
            f"covering {len(seen)} of the {len(groups)} generator "
            f"configurations.\n     The file order is blocked by "
            f"configuration, so this is a slice, not a sample. Use "
            f"--limit 0\n     for the whole set (feasible for HGS and "
            f"LKH-3; AILS-II would take hours)."
        )
    return (
        f"--limit {limit} keeps the first {limit}/{len(names)} instances "
        f"in name order,\n     which for this set is not a random sample."
    )


def build_cmd(solver, set_tag, budget_s, args):
    """The full driver invocation for one (solver, set), or None if skipped."""
    spec = SETS[set_tag]
    sv = SOLVERS[solver]

    if sv["needs_integer"] and not is_integer_set(set_tag):
        return None  # LKH-3 on continuous coordinates

    cmd = ["uv", "run", "--no-project", os.path.join("tools", sv["driver"])]

    # On the CVRPLib sets, --dir keeps the real instance names, which lets
    # gap_to_bks.py check the pairing by name instead of trusting the index
    # order. The drivers record the sibling .cvrpb so validate.py still works.
    if spec["vrpdir"] and os.path.isdir(os.path.join(ROOT, spec["vrpdir"])):
        cmd += ["--dir", spec["vrpdir"]]
    else:
        cmd += ["--bundle", spec["bundle"]]

    cmd += ["--time", f"{budget_s:.4f}"]
    cmd += ["--out", os.path.relpath(args.out, ROOT)]
    cmd += ["--name", f"{sv['label']}_{set_tag}"]
    if args.limit:
        cmd += ["--limit", str(args.limit)]

    if solver == "ails":
        # -limit is wall clock: J at once means each gets less real work.
        cmd += ["--jobs", str(args.ails_jobs), "--heap", spec["heap"]]
    else:
        cmd += ["--jobs", str(args.jobs if args.jobs else spec["jobs"])]

    return cmd


def estimate(refs, args):
    """Print a wall-time floor per (set, solver), before anything is launched.

    instances x budget / jobs is the time the budget alone accounts for. It is
    a *floor*, not a prediction: none of the three solvers' time flags covers
    its set-up work, and the measured overruns are ~3x for AILS-II at any size
    (JVM start and its JIT/GC threads) and 2.5-43x for everyone on XL, where
    n = 10000 makes that set-up dominate. Printed anyway, because with
    --limit 0 the difference between a 20-minute run and an overnight one is
    the thing you want to know before starting, not after.
    """
    print("### wall-time floor (instances x budget / jobs), hours:minutes\n")
    labels = [SOLVERS[s]["label"] for s in args.solvers]
    print(f"    {'set':<8}{'inst':>8}" + "".join(f"{lab:>10}" for lab in labels))
    print("    " + "-" * (16 + 10 * len(labels)))

    totals = {lab: 0.0 for lab in labels}
    for set_tag in args.sets:
        ref = refs[set_tag]
        inst = min(args.limit, ref["instances"]) if args.limit else ref["instances"]
        cells = []
        for solver in args.solvers:
            sv = SOLVERS[solver]
            if sv["needs_integer"] and not is_integer_set(set_tag):
                cells.append(f"{'-':>10}")
                continue
            jobs = (
                args.ails_jobs
                if solver == "ails"
                else (args.jobs or SETS[set_tag]["jobs"])
            )
            secs = inst * ref["budget_s"] / max(1, jobs)
            totals[sv["label"]] += secs
            cells.append(f"{int(secs // 3600):>7}:{int(secs % 3600 // 60):02d}")
        print(f"    {set_tag:<8}{inst:>8}" + "".join(cells))

    grand = sum(totals.values())
    print("    " + "-" * (16 + 10 * len(labels)))
    print(
        f"    {'total':<8}{'':>8}"
        + "".join(
            f"{int(totals[lab] // 3600):>7}:{int(totals[lab] % 3600 // 60):02d}"
            for lab in labels
        )
        + f"     -> {int(grand // 3600)}:{int(grand % 3600 // 60):02d} sequential"
    )
    print(
        "\n    A FLOOR: set-up work sits outside every solver's time flag. "
        "Measured overruns\n    were ~3x for AILS-II at n = 100 and 2.5-43x "
        "for all three on XL. Raise\n    --ails-jobs (its budget is wall "
        "clock, so J at once costs J-way contention,\n    not J-way less "
        "work) or drop --solvers ails to cut the dominant term.\n"
    )


def newest_run(out_dir, name):
    dirs = sorted(glob.glob(os.path.join(out_dir, f"*_{name}")))
    return dirs[-1] if dirs else None


def run_and_capture(cmd, log_path):
    """Run a command, tee its output to a file, return the exit code."""
    sys.stdout.flush()
    with open(log_path, "w", encoding="utf-8") as f:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        f.write(p.stdout)
        if p.stderr:
            f.write("\n--- stderr ---\n" + p.stderr)
    sys.stdout.write(p.stdout)
    if p.stderr:
        sys.stderr.write(p.stderr)
    return p.returncode


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--sets",
        nargs="+",
        default=DEFAULT_SETS,
        choices=list(SETS),
        metavar="SET",
        help=f"default: {' '.join(DEFAULT_SETS)}",
    )
    ap.add_argument(
        "--solvers",
        nargs="+",
        default=list(SOLVERS),
        choices=list(SOLVERS),
        metavar="SOLVER",
        help="hgs, lkh, ails (default: all three)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="instances per set (0 = all, the default). A limit keeps the "
        "FIRST L of the set's own order, which is a valid subsample on the "
        "NeuOpt bundles (instance k comes from seed + k, i.i.d.) but not on "
        "XML100, whose file order is blocked by generator configuration.",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="parallel processes for HGS and LKH-3 "
        "(default: per-set, 12 except 4 on XL for memory)",
    )
    ap.add_argument(
        "--ails-jobs",
        type=int,
        default=1,
        help="parallel JVMs for AILS-II (default 1: its budget is "
        "wall clock, so anything else understates it)",
    )
    ap.add_argument(
        "--savant-runs",
        default=SAVANT_RUNS,
        help="where scripts/run_best_config.sh left its runs",
    )
    ap.add_argument(
        "--savant-config",
        default=None,
        metavar="TAG",
        help="force one SAVANT configuration as the reference (e.g. cand) "
        "instead of the best at the budget it was last run at",
    )
    ap.add_argument("--out", default=OUT, help="root for the new runs")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the budgets and commands, run nothing",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    failed, skipped = [], []

    # ---------------------------------------------------------- the budgets
    refs = {}
    print("### matched budgets, from SAVANT's best run of each set\n")
    hdr = (
        f"{'set':<8} {'SAVANT cfg':<10} {'mean cost':>14} {'inst':>7} "
        f"{'asked s':>9} {'spent CPU s':>12} {'x asked':>8}   "
        f"per-instance spread (relative)"
    )
    print(hdr)
    print("-" * len(hdr))
    for set_tag in args.sets:
        ref = savant_reference(set_tag, args.savant_runs, args.savant_config)
        refs[set_tag] = ref
        spread = time_spread(ref["dir"], args.limit)
        note = ""
        if spread:
            lo, med, hi, mn, mx = spread
            ratio = hi / lo if lo else 0.0
            note = (f"{lo / med:.2f}x .. {hi / med:.2f}x of median (5-95 %), "
                    f"{mn / med:.2f}x .. {mx / med:.2f}x overall")
            if ratio > 2.0:
                note += "   <- one mean budget is a poor fit"
        asked = ref["requested_s"]
        asked_s = f"{asked:.4f}" if asked else "(steps)"
        ratio_s = f"{ref['budget_s'] / asked:.2f}" if asked else "-"
        print(
            f"{set_tag:<8} {ref['config']:<10} {ref['mean_cost']:>14.5f} "
            f"{ref['instances']:>7} {asked_s:>9} {ref['budget_s']:>12.4f} "
            f"{ratio_s:>8}   {note}"
        )
    print(
        "\n    `asked` is SAVANT's own --sa-time; `spent` is what it cost, and "
        "that is what the\n    other solvers are given. A run budgeted in steps "
        "has nothing to ask for, and its\n    spread is then the instances' own "
        "— on X and XL, a 2-10x span that no single\n    scalar budget "
        "describes.\n"
    )

    for set_tag in args.sets:
        bias = limit_bias(set_tag, args.limit)
        if bias:
            print(f"!! {set_tag}: {bias}\n")

    estimate(refs, args)

    # ------------------------------------------------------------- the runs
    for set_tag in args.sets:
        ref = refs[set_tag]
        budget = ref["budget_s"]

        for solver in args.solvers:
            sv = SOLVERS[solver]
            label = f"{sv['label']}/{set_tag}"

            cmd = build_cmd(solver, set_tag, budget, args)
            if cmd is None:
                print(
                    f"-- {label}: skipped (LKH-3 has no float mode; "
                    f"{set_tag} has continuous coordinates)"
                )
                skipped.append(label)
                continue
            if not os.path.exists(os.path.join(ROOT, sv["binary"])):
                print(
                    f"-- {label}: skipped ({sv['binary']} not built — "
                    f"see tools/setup.sh)"
                )
                skipped.append(label)
                continue

            print("\n" + "=" * 70)
            print(f"### {label}   {budget:.4f} CPU s/instance")
            print("=" * 70)
            print("  " + " ".join(cmd))
            if args.dry_run:
                continue

            driver_log = os.path.join(args.out, f"{sv['label']}_{set_tag}.log")
            rc = run_and_capture(cmd, driver_log)

            run_dir = newest_run(args.out, f"{sv['label']}_{set_tag}")
            if not run_dir:
                print(f"!! {label}: driver failed, no run directory", file=sys.stderr)
                failed.append(label)
                continue
            os.replace(driver_log, os.path.join(run_dir, "driver.log"))
            if rc:
                # A non-zero exit is usually a partial result, not a lost one:
                # run_lkh.py fails the run when an instance comes back with a
                # capacity penalty, and the instances that did solve are still
                # in solutions.txt. Validate and score what there is, and let
                # the summary show how many are missing.
                print(
                    f"!! {label}: driver exited {rc} — validating the partial result",
                    file=sys.stderr,
                )
                failed.append(label + f"(exit {rc})")

            # -- the same independent checker every SAVANT run went through --
            print(f"\n--- validate {label}")
            vpath = os.path.join(run_dir, "validation.txt")
            if run_and_capture(
                ["uv", "run", "--no-project", "tools/validate.py", run_dir], vpath
            ):
                print(f"!! VALIDATION FAILED: {label}", file=sys.stderr)
                failed.append(label + "(validate)")

            bks = SETS[set_tag]["bks"]
            if bks and os.path.exists(os.path.join(ROOT, bks)):
                print(f"\n--- gap to {bks}")
                run_and_capture(
                    [
                        "uv",
                        "run",
                        "--no-project",
                        "tools/gap_to_bks.py",
                        run_dir,
                        bks,
                        "--csv",
                        os.path.join(run_dir, "bks_per_instance.csv"),
                    ],
                    os.path.join(run_dir, "bks_gap.txt"),
                )

    if args.dry_run:
        return 0

    print("\n" + "=" * 70)
    sys.stdout.flush()  # the child writes to the same fd; keep the order
    subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "scripts/summarize_matched.py",
            "--matched",
            args.out,
            "--savant-runs",
            args.savant_runs,
            "--csv",
        ],
        cwd=ROOT,
    )

    if skipped:
        print("\nskipped: " + ", ".join(skipped))
    if failed:
        print("\n!! problems in: " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
