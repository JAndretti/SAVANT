#!/usr/bin/env python3
"""
gap_to_bks.py — gap of a run to the reference solutions shipped with the
CVRPLib sets (proven optima for XML100, BKS for X and XL).

These sets are worth far more than a solver-vs-solver comparison: XML100 comes
with *proven optimal* costs, so the number this prints is an optimality gap,
not a relative one. Set X and XL ship the current best known solutions, which
bound it from above.

Works on any run directory, from run.py (SAVANT) or run_hgs.py (HGS), because
both write the same `solutions.txt`. Instances are matched by index — the
bundle, `cw --dir` and `run_hgs.py --dir` all use the same sorted-by-name
order — and the names are cross-checked, so a mismatched pairing is reported
rather than silently averaged.

Usage:
    python3 tools/gap_to_bks.py results/<run> data/cvrplib/X_bks.csv
    python3 tools/gap_to_bks.py results/<run> data/cvrplib/X_bks.csv --per-instance
"""

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

sys.path.insert(0, HERE)
from validate import read_solutions  # noqa: E402


def read_bks(path, column=None):
    """{idx: (name, n, cost)} for the rows that carry a reference.

    Two schemas are accepted. `data/cvrplib/<SET>_bks.csv`, written by
    fetch_cvrplib.py, has an `idx` and a `bks_cost` taken from the distributed
    .sol files. `baseline/xl_bks.csv`, extracted from the XL paper, has no
    index and carries both `initial_bks` and `final_bks`; rows are then
    numbered by sorted instance name, which is the order everything else uses,
    and `final_bks` is the default because the archive ships the pre-challenge
    solutions and measuring against those understates every gap.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path}: empty")

    col = column
    if col is None:
        for cand in ("bks_cost", "final_bks", "initial_bks"):
            if cand in rows[0]:
                col = cand
                break
    if col is None or col not in rows[0]:
        raise SystemExit(f"{path}: no usable cost column "
                         f"(have {', '.join(rows[0])})")

    if "idx" not in rows[0]:
        rows = sorted(rows, key=lambda r: r["name"])
        for k, r in enumerate(rows):
            r["idx"] = k

    out = {}
    for row in rows:
        if not row.get(col):
            continue
        out[int(row["idx"])] = (row["name"], int(row["n"]), float(row[col]))
    return out, col


def main():
    ap = argparse.ArgumentParser(description="Gap of a run to the CVRPLib "
                                             "reference solutions")
    ap.add_argument("run", help="run directory or solutions.txt")
    ap.add_argument("bks", help="the set's *_bks.csv")
    ap.add_argument("--column", default=None,
                    help="which reference column to score against "
                         "(bks_cost, final_bks, initial_bks); default: the "
                         "first present, final_bks preferred over initial_bks")
    ap.add_argument("--per-instance", action="store_true",
                    help="one line per instance, worst gap first")
    ap.add_argument("--csv", help="write the per-instance table here")
    args = ap.parse_args()

    solp = (args.run if os.path.isfile(args.run)
            else os.path.join(args.run, "solutions.txt"))
    sols, rounded = read_solutions(solp)
    bks, col = read_bks(args.bks, args.column)

    rows, mismatched = [], []
    for idx, name, n, cap, cost, routes in sols:
        if idx not in bks:
            continue
        bname, bn, bcost = bks[idx]
        # Three naming conventions reach this point for the same instance:
        # run_hgs.py --dir writes the bare name, `cw --dir` keeps the file
        # extension, and `cw --bundle` has no name to keep and synthesises
        # "<bundle>#<idx>". Only the first two can be checked against the
        # reference; for the third the size is the only usable guard, which is
        # why the index order of the bundle has to match the sorted file names.
        stem = os.path.splitext(name)[0]
        synthetic = "#" in name and name.rsplit("#", 1)[1] == str(idx)
        if n != bn or (not synthetic and stem != bname):
            mismatched.append((idx, name, bname))
            continue
        rows.append({"idx": idx, "name": bname, "n": n, "cost": cost,
                     "bks": bcost, "gap_pct": 100.0 * (cost - bcost) / bcost,
                     "routes": len(routes)})

    if mismatched:
        print(f"{len(mismatched)} instance(s) do not line up with the reference "
              f"— wrong set, or a --limit run against a full CSV:",
              file=sys.stderr)
        for idx, a, b in mismatched[:5]:
            print(f"  idx {idx}: run has {a!r}, reference has {b!r}",
                  file=sys.stderr)
        if not rows:
            return 2

    if not rows:
        raise SystemExit("no instance in common with the reference")

    if not rounded:
        print("warning: this run used FLOAT distances, but the CVRPLib "
              "references are scored with integer rounding — the gap below is "
              "not meaningful. Re-run with --round (cw) or --round 1 "
              "(run_hgs.py).", file=sys.stderr)

    gaps = [r["gap_pct"] for r in rows]
    mean_gap = sum(gaps) / len(gaps)
    optimal = sum(1 for g in gaps if g <= 1e-9)
    worse = sorted(rows, key=lambda r: -r["gap_pct"])

    print(f"run       : {os.path.basename(os.path.normpath(args.run))}")
    print(f"reference : {os.path.relpath(args.bks, ROOT)}  [{col}]")
    print(f"matched   : {len(rows)} instance(s), "
          f"n = {min(r['n'] for r in rows)}..{max(r['n'] for r in rows)}")
    print()
    print(f"  mean gap      : {mean_gap:+.3f} %")
    print(f"  median gap    : {sorted(gaps)[len(gaps) // 2]:+.3f} %")
    print(f"  best / worst  : {min(gaps):+.3f} % / {max(gaps):+.3f} %")
    print(f"  matched ref   : {optimal}/{len(rows)}")
    print(f"  mean cost     : {sum(r['cost'] for r in rows) / len(rows):.2f}"
          f"   (reference {sum(r['bks'] for r in rows) / len(rows):.2f})")

    if args.per_instance:
        print()
        print(f"  {'instance':<20} {'n':>6} {'cost':>12} {'reference':>12} {'gap %':>8}")
        for r in worse:
            print(f"  {r['name']:<20} {r['n']:>6} {r['cost']:>12.1f} "
                  f"{r['bks']:>12.1f} {r['gap_pct']:>+8.3f}")
    else:
        print()
        print("  worst five:")
        for r in worse[:5]:
            print(f"    {r['name']:<20} n={r['n']:<6} {r['gap_pct']:+.3f} %")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n-> {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
