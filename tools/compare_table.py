#!/usr/bin/env python3
"""
compare_table.py — collect run directories into one time-quality table.

Reads the config.json of every run given (SAVANT runs produced by run.py,
HGS runs produced by run_hgs.py), and prints them sorted by CPU cost per
instance, with the mean objective and the gap to the best run in the set.

Comparing on CPU seconds per instance rather than wall clock is deliberate:
HGS's `-t` is a CPU-time budget (Genetic.cpp uses clock()), SAVANT spreads one
instance-parallel loop over OpenMP threads, and only total CPU makes the two
commensurable on the same machine.

Usage:
    python3 tools/compare_table.py results/*HGS_cvrp_100* results/*SAVANT_cvrp_100*
    python3 tools/compare_table.py --csv cmp.csv results/2026*
"""

import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_run(path):
    """One run -> dict, or None when the directory is not a finished run."""
    try:
        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return None

    run = cfg.get("run", {})
    res = cfg.get("result", {}) or {}
    solver = run.get("solver") or "SAVANT"

    # run_hgs.py writes mean_cost / cpu_s / wall_s; run.py's parse_summary
    # writes cost_after_sa / time_cpu_s / time_wall_s straight from cw's own
    # summary, which already reports cumulative CPU.
    mean = res.get("mean_cost", res.get("cost_after_sa"))
    if mean is None:
        return None

    n_inst = res.get("instances")
    wall = res.get("wall_s", res.get("time_wall_s", run.get("elapsed_s")))
    cpu = res.get("cpu_s", res.get("time_cpu_s"))

    if solver == "HGS-CVRP":
        budget = " ".join(cfg.get("hgs_args", []))
        for flag, label in (("-t", "t="), ("-it", "it=")):
            if flag in cfg.get("hgs_args", []):
                budget = label + cfg["hgs_args"][cfg["hgs_args"].index(flag) + 1]
                break
        else:
            budget = "it=20000 (default)"
    else:
        args = cfg.get("cw_args", [])
        budget = ""
        if "--sa-steps" in args:
            budget = "steps=" + args[args.index("--sa-steps") + 1]
        if "--restarts" in args:
            budget += " x" + args[args.index("--restarts") + 1]

    return {
        "run": os.path.basename(os.path.normpath(path)),
        "solver": solver,
        "budget": budget.strip(),
        "instances": n_inst,
        "mean": float(mean),
        "wall_s": wall,
        "cpu_s": cpu,
        "cpu_per_inst": (cpu / n_inst) if (cpu and n_inst) else None,
        "checks_failed": res.get("checks_failed"),
    }


def main():
    ap = argparse.ArgumentParser(description="Time-quality table across runs")
    ap.add_argument("runs", nargs="+", help="run directories")
    ap.add_argument("--csv", help="also write the table to this CSV")
    ap.add_argument("--ref", help="run to use as the 0 %% reference "
                                  "(default: the best mean in the set)")
    args = ap.parse_args()

    rows = [r for r in (load_run(p) for p in args.runs) if r]
    if not rows:
        raise SystemExit("no readable run directory among the arguments")

    sizes = {r["instances"] for r in rows if r["instances"]}
    if len(sizes) > 1:
        print(f"warning: runs cover different instance counts {sorted(sizes)} — "
              f"the means are not comparable", file=sys.stderr)

    if args.ref:
        ref = next((r for r in rows if args.ref in r["run"]), None)
        if ref is None:
            raise SystemExit(f"{args.ref}: no matching run")
    else:
        ref = min(rows, key=lambda r: r["mean"])

    rows.sort(key=lambda r: (r["solver"], r["cpu_per_inst"] or 0.0))

    w = max(len(r["run"]) for r in rows)
    print(f"reference: {ref['run']}  (mean {ref['mean']:.6f})")
    print(f"instances: {sorted(sizes)}")
    print()
    print(f"{'run':<{w}}  {'solver':<9} {'budget':<20} "
          f"{'mean':>10} {'gap %':>8} {'cpu s/inst':>11} {'wall s':>9}")
    print("-" * (w + 74))
    for r in rows:
        gap = 100.0 * (r["mean"] - ref["mean"]) / ref["mean"]
        cpi = f"{r['cpu_per_inst']:.4f}" if r["cpu_per_inst"] else "-"
        wall = f"{r['wall_s']:.1f}" if r["wall_s"] else "-"
        flag = "" if not r["checks_failed"] else f"  !! {r['checks_failed']} check(s)"
        print(f"{r['run']:<{w}}  {r['solver']:<9} {r['budget']:<20} "
              f"{r['mean']:>10.5f} {gap:>+8.3f} {cpi:>11} {wall:>9}{flag}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["gap_pct"])
            wr.writeheader()
            for r in rows:
                r = dict(r)
                r["gap_pct"] = 100.0 * (r["mean"] - ref["mean"]) / ref["mean"]
                wr.writerow(r)
        print(f"\n-> {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
