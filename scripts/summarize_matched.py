#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""summarize_matched.py — SAVANT against HGS, LKH-3 and AILS-II at equal budget.

Reads the runs scripts/run_matched_time.py produced and puts them next to the
SAVANT run whose budget they were given. Everything is recomputed on the
**instances the two runs actually have in common**, because the competitors are
usually run on a subsample (`--limit`) of a 10,000-instance bundle while SAVANT
ran all of it — comparing its published mean against theirs would be comparing
two different instance sets.

The comparison is paired, as everywhere else here: instance k against instance
k, the mean of the per-instance differences. Instances are matched on the index
carried in solutions.txt, which is the sorted order every driver uses, and the
size n is cross-checked on every pair so a mismatched pairing is reported
rather than averaged.

Standard library only.

Usage:
    uv run --no-project scripts/summarize_matched.py
    uv run --no-project scripts/summarize_matched.py --csv
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from validate import read_solutions          # noqa: E402
from gap_to_bks import read_bks              # noqa: E402

sys.path.insert(0, HERE)
from run_matched_time import SETS, SOLVERS, savant_reference   # noqa: E402

SET_ORDER = ["n20", "n50", "n100", "n200", "XML100", "X", "XL"]
LABELS = [SOLVERS[s]["label"] for s in ("hgs", "lkh", "ails")]


def fmt(v, spec):
    """Format a value, or a right-aligned "-" when the run did not produce it.

    The placeholder has to carry the field's width, or a row with a missing
    value shifts every column after it out of line with the header.
    """
    if v is not None:
        return format(v, spec)
    m = re.match(r"[+\- ]?(\d+)", spec)
    return "-".rjust(int(m.group(1))) if m else "-"


def set_key(tag):
    return (SET_ORDER.index(tag) if tag in SET_ORDER else len(SET_ORDER), tag)


def load_matched(root):
    """{(set, label): {dir, cfg}} for every finished run under root."""
    out = {}
    for path in sorted(glob.glob(os.path.join(root, "*"))):
        cfg_path = os.path.join(path, "config.json")
        if not os.path.isdir(path) or not os.path.exists(cfg_path):
            continue
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        name = (cfg.get("run", {}) or {}).get("name") or ""
        m = re.fullmatch(r"(" + "|".join(LABELS) + r")_(.+)", name)
        if not m:
            continue
        out[(m.group(2), m.group(1))] = {"dir": path, "cfg": cfg}
    return out


def costs_by_index(run_dir):
    """{idx: (n, cost)} from solutions.txt."""
    path = os.path.join(run_dir, "solutions.txt")
    if not os.path.exists(path):
        return {}
    sols, _ = read_solutions(path)
    return {idx: (n, cost) for idx, _name, n, _cap, cost, _routes in sols}


def paired(a, b, keys):
    """A against baseline B on `keys`, in the report's terms."""
    diffs = [a[k][1] - b[k][1] for k in keys]
    m = len(diffs)
    base = sum(b[k][1] for k in keys) / m
    mean = sum(diffs) / m
    var = sum((d - mean) ** 2 for d in diffs) / (m - 1) if m > 1 else 0.0
    half = 1.96 * math.sqrt(var / m)
    wins = sum(1 for d in diffs if d < 0)
    losses = sum(1 for d in diffs if d > 0)
    n_eff = wins + losses
    p = (math.erfc(abs(wins - n_eff / 2) / math.sqrt(n_eff / 4) / math.sqrt(2))
         if n_eff else 1.0)
    scale = 100.0 / base if base else 0.0
    return dict(m=m, delta_pct=mean * scale, lo_pct=(mean - half) * scale,
                hi_pct=(mean + half) * scale, wins=wins, losses=losses, p_sign=p)


def mean_gap_to_ref(costs, keys, bks_path):
    """Mean gap to the published references over `keys`, or None.

    Recomputed here rather than read from bks_gap.txt so that every row is
    scored on the same subset — the files were each written over whatever that
    run happened to cover.
    """
    if not bks_path:
        return None
    full = os.path.join(ROOT, bks_path)
    if not os.path.exists(full):
        return None
    try:
        bks, _col = read_bks(full)
    except SystemExit:
        return None
    gaps = []
    for k in keys:
        if k not in bks:
            continue
        n, cost = costs[k]
        bname, bn, bcost = bks[k]
        if n != bn or bcost <= 0:
            continue
        gaps.append(100.0 * (cost - bcost) / bcost)
    return sum(gaps) / len(gaps) if gaps else None


def validation(run_dir):
    path = os.path.join(run_dir, "validation.txt")
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return "-"
    m = re.search(r"(\d+) instance\(s\) checked, (\d+) error\(s\)", text)
    if not m:
        return "?"
    return "OK" if m.group(2) == "0" else f"FAIL:{m.group(2)}"


def cpu_per_instance(cfg):
    res = cfg.get("result", {}) or {}
    if res.get("cpu_s_per_instance") is not None:
        return res["cpu_s_per_instance"]
    cpu, inst = res.get("time_cpu_s"), res.get("instances")
    return cpu / inst if cpu and inst else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--matched", default=os.path.join(ROOT, "results", "matched_time"))
    ap.add_argument("--savant-runs", default=os.path.join(ROOT, "results", "best_config"))
    ap.add_argument("--savant-config", default=None, metavar="TAG",
                    help="force one SAVANT configuration as the baseline "
                         "(e.g. cand); default is the same rule "
                         "run_matched_time.py used to hand out the budget")
    ap.add_argument("--csv", action="store_true",
                    help="write <matched>/summary.csv")
    args = ap.parse_args()

    if not os.path.isdir(args.matched):
        raise SystemExit(f"{args.matched}: not found — run "
                         f"scripts/run_matched_time.py first")
    runs = load_matched(args.matched)
    if not runs:
        raise SystemExit(f"{args.matched}: no finished run")

    rows, overrun = [], []
    for set_tag in sorted({s for s, _ in runs}, key=set_key):
        ref = savant_reference(set_tag, args.savant_runs, args.savant_config)
        sav = costs_by_index(ref["dir"])
        bks = SETS.get(set_tag, {}).get("bks")
        budget = ref["budget_s"]
        asked = (f" asked {ref['requested_s']:.4f} s"
                 if ref.get("requested_s") else "")

        print(f"\n### {set_tag}   SAVANT {ref['config']}, "
              f"{budget:.4f} CPU s/instance{asked} ({validation(ref['dir'])}), "
              f"over {ref['instances']} instances")
        hdr = (f"{'solver':<7} {'m':>6} {'mean cost':>13} {'SAVANT':>13} "
               f"{'vs SAVANT %':>12} {'95 % CI':>19} {'win/loss':>10} "
               f"{'cpu s/inst':>11} {'x budget':>9} "
               f"{'gap %':>8} {'SAVANT':>8}  valid")
        print(hdr)
        print("-" * len(hdr))

        # Every row is scored on ITS OWN intersection with SAVANT, and carries
        # SAVANT's mean over exactly those instances. A solver that returned
        # fewer instances than another (LKH-3 drops the ones it could not make
        # feasible) is then still read against the right baseline, instead of
        # against a mean taken over a different set.
        for lab in LABELS:
            if (set_tag, lab) not in runs:
                continue
            r = runs[(set_tag, lab)]
            other = costs_by_index(r["dir"])
            keys = sorted(set(other) & set(sav))
            bad = [k for k in keys if other[k][0] != sav[k][0]]
            if bad:
                print(f"  !! {lab}: {len(bad)} instance(s) whose n disagrees with "
                      f"SAVANT's — pairing is wrong, skipped", file=sys.stderr)
                continue
            res = r["cfg"].get("result", {}) or {}
            if not keys:
                # A solver that returned nothing usable must still occupy a
                # row: dropping it would read as "not run" rather than as the
                # result it is (LKH-3 at a sub-second budget on XL comes back
                # with a capacity penalty on every instance).
                why = (f"{res['failed']}/{res.get('instances', '?')} unsolved"
                       if res.get("failed") else "no instance in common")
                cpu = cpu_per_instance(r["cfg"])
                print(f"{lab:<7} {0:>6} {'-':>13} {'-':>13} {'-':>12} {'-':>19} "
                      f"{'-':>10} {fmt(cpu, '11.4f')} "
                      f"{'-':>9} {'-':>8} {'-':>8}  {why}")
                rows.append(dict(set=set_tag, solver=lab, m=0, mean_cost=None,
                                 savant_cost=None, cpu_per_inst=cpu,
                                 budget_s=budget, budget_ratio=None, wall_s=res.get("wall_s"),
                                 valid=why, gap_ref=None, savant_gap_ref=None,
                                 delta_pct=None, lo_pct=None, hi_pct=None,
                                 wins=None, losses=None, p_sign=None,
                                 run=os.path.basename(r["dir"]),
                                 savant_run=os.path.basename(ref["dir"])))
                continue
            st = paired(other, sav, keys)
            cpu = cpu_per_instance(r["cfg"])
            valid = validation(r["dir"])
            if res.get("failed"):
                valid += f" ({res['failed']} unsolved)"
            row = dict(
                set=set_tag, solver=lab, m=st["m"],
                mean_cost=sum(other[k][1] for k in keys) / len(keys),
                savant_cost=sum(sav[k][1] for k in keys) / len(keys),
                cpu_per_inst=cpu,
                budget_s=budget,
                budget_ratio=(cpu / budget) if (cpu and budget) else None,
                wall_s=res.get("wall_s"),
                valid=valid,
                gap_ref=mean_gap_to_ref(other, keys, bks),
                savant_gap_ref=mean_gap_to_ref(sav, keys, bks),
                delta_pct=st["delta_pct"], lo_pct=st["lo_pct"], hi_pct=st["hi_pct"],
                wins=st["wins"], losses=st["losses"], p_sign=st["p_sign"],
                run=os.path.basename(r["dir"]),
                savant_run=os.path.basename(ref["dir"]),
            )
            if row["budget_ratio"] and row["budget_ratio"] > 1.25:
                overrun.append((set_tag, lab, row["budget_ratio"]))
            rows.append(row)
            print_row(row)

    print("\n`vs SAVANT %` is the paired mean difference as a percentage of "
          "SAVANT's mean on the same\ninstances — negative means the other "
          "solver is cheaper. `*` = the 95 % CI excludes zero\nand the sign "
          "test rejects at 5 %. `SAVANT` columns are SAVANT restricted to that "
          "row's\ninstances, so every row is read against its own baseline.")

    if overrun:
        print("\n!! budget overrun — these rows are NOT matched on CPU, they "
              "got more of it:")
        for set_tag, lab, ratio in overrun:
            print(f"     {lab}/{set_tag}: {ratio:.2f}x the requested budget "
                  f"actually spent")
        print("""
   Three causes, none of them under the budget flag's control:
     * AILS-II is budgeted in WALL clock (-limit polls currentTimeMillis) and
       has no CPU option, so what it costs in CPU is whatever the JVM decides.
       Measured at n = 100: 1.06 s wall against a 0.63 s budget, and 1.90 s CPU
       against that 1.06 s wall.
     * -limit governs the search loop only. JVM start, class loading, parsing
       and teardown sit outside it — about 0.4 s at n = 100, and it does not
       shrink when the budget does.
     * Every solver has set-up work the budget does not cover: HGS builds a
       dense n x n distance matrix, LKH-3 generates its candidate set before
       the first trial. At n = 10000 (XL) that alone outruns a one-second
       budget, which is why LKH-3 there can spend 16 CPU-s and still return a
       capacity-penalised answer.
   These rows say what the solver reaches when given several times SAVANT's
   CPU. That bounds them from the favourable side, but it is not the matched
   comparison.""")

    if args.csv and rows:
        path = os.path.join(args.matched, "summary.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"-> {os.path.relpath(path, ROOT)}")
    return 0


def print_row(r):
    sig = ((r["hi_pct"] < 0 or r["lo_pct"] > 0) and r["p_sign"] < 0.05)
    delta = f"{r['delta_pct']:+.3f}" + ("*" if sig else "")
    ci = f"[{r['lo_pct']:+.3f}, {r['hi_pct']:+.3f}]"
    wl = f"{r['wins']}/{r['losses']}"
    print(f"{r['solver']:<7} {r['m']:>6} {fmt(r['mean_cost'], '13.4f')} "
          f"{fmt(r['savant_cost'], '13.4f')} {delta:>12} {ci:>19} {wl:>10} "
          f"{fmt(r['cpu_per_inst'], '11.4f')} {fmt(r['budget_ratio'], '9.2f')} "
          f"{fmt(r['gap_ref'], '+8.3f')} {fmt(r['savant_gap_ref'], '+8.3f')}  "
          f"{r['valid']}")


if __name__ == "__main__":
    sys.exit(main())
