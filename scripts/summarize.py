#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""summarize.py — one table for everything run_best_config.sh produced.

Reads every run directory under results/best_config/ and prints:

  1. a row per run: budget, mean cost, timing, validation verdict, gap to the
     published references when the set ships them;
  2. the paired `cand` vs `prev` comparison per set, which is the point of the
     study: both get the same budget, and they differ only in the four flags
     the defaults moved on 2026-08-07 (--ops, --t-decades, --pick2, --kick).

The paired reading follows sweep/report.tex: instance k of one run is instance
k of the other (same bundle, same order), so the statistic is the mean of the
per-instance differences rather than the difference of two means. The
between-instance spread is orders of magnitude larger than the effect, and an
unpaired comparison would drown in it.

`--pair A B` compares two other configurations; the older `split16` / `single`
runs (the budget-split study, settled: restarts lose at fixed total budget on
both X and XL) are still recognised and still listed in the first table.

Standard library only, so `uv run --no-project scripts/summarize.py` works.

Usage:
    uv run --no-project scripts/summarize.py
    uv run --no-project scripts/summarize.py --out results/best_config --csv
    uv run --no-project scripts/summarize.py --pair split16 single
"""

import argparse
import csv
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))       # scripts/
ROOT = os.path.dirname(HERE)                            # repository root

SET_ORDER = ["n20", "n50", "n100", "n200", "XML100", "X", "XL"]
# cand/prev is the current study; split16/single was the budget-split one, kept
# so its runs still summarise instead of being silently dropped.
CFG_ORDER = ["cand", "prev", "split16", "single"]
DEFAULT_PAIR = ("cand", "prev")


def load_runs(out_dir, configs):
    """Every run directory under out_dir, newest kept per (set, config)."""
    runs = {}
    for entry in sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, entry)
        cfg_path = os.path.join(path, "config.json")
        if not os.path.isdir(path) or not os.path.exists(cfg_path):
            continue
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        name = cfg.get("run", {}).get("name") or ""
        if "_" not in name:
            continue
        set_tag, config = name.rsplit("_", 1)
        if config not in configs:
            continue
        # sorted() walks timestamps in order, so the last write wins
        runs[(set_tag, config)] = {"dir": path, "cfg": cfg}
    return runs


def read_validation(run_dir):
    """(ok, detail) from validation.txt, or (None, '') when it was not run."""
    path = os.path.join(run_dir, "validation.txt")
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return None, ""
    m = re.search(r"(\d+) instance\(s\) checked, (\d+) error\(s\)", text)
    if not m:
        return False, "unparsable"
    checked, errors = int(m.group(1)), int(m.group(2))
    if errors:
        return False, f"{errors} error(s)"
    return True, f"{checked} ok"


def read_bks_gap(run_dir):
    """Mean gap to the reference solutions, or None."""
    path = os.path.join(run_dir, "bks_gap.txt")
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return None
    m = re.search(r"mean gap\s*:\s*([+-][\d.]+) %", text)
    return float(m.group(1)) if m else None


def read_budget(cfg):
    """How the run was budgeted, as (label, kind), from the command line.

    A time-budgeted run and a step-budgeted one are not comparable, so the
    first table names which it was rather than leaving the reader to guess
    from the timings.
    """
    argv = cfg.get("cw_args") or []
    res = cfg.get("result", {}) or {}
    restarts = 1
    if "--restarts" in argv:
        try:
            restarts = int(argv[argv.index("--restarts") + 1])
        except (IndexError, ValueError):
            restarts = 1
    # a step count is per restart, so 625,000 x 16 and 10,000,000 x 1 are the
    # same budget and have to read as such
    mult = f" x{restarts}" if restarts > 1 else ""
    for flag, kind in (("--sa-time", "time"), ("--sa-steps", "steps")):
        if flag in argv:
            v = argv[argv.index(flag) + 1]
            if kind == "time":
                steps = res.get("steps_mean")
                mean = f" ~{steps / 1e6:.1f}M" if steps else ""
                return f"{v}s{mean}", kind
            return f"{int(v) / 1e6:.2f}M{mult}", kind
    return "-", None


def read_costs(run_dir):
    """{instance name: annealed cost} from results.csv."""
    path = os.path.join(run_dir, "results.csv")
    out = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out[row["instance"]] = float(row["cost_annealed"])
    except (OSError, KeyError, ValueError):
        return {}
    return out


def paired(a_costs, b_costs):
    """Paired comparison of A against baseline B, in the report's terms.

    Returns None when the two runs share no instance — a --limit mismatch, or
    two runs of different sets. `delta_pct` is the mean paired difference as a
    percentage of B's mean, and the interval is that same mean +- 1.96 standard
    errors, rescaled identically. `p_sign` is the two-sided sign test on
    wins against losses (normal approximation; m is in the thousands here).
    """
    keys = [k for k in a_costs if k in b_costs]
    if not keys:
        return None
    diffs = [a_costs[k] - b_costs[k] for k in keys]
    m = len(diffs)
    base = sum(b_costs[k] for k in keys) / m
    mean = sum(diffs) / m
    var = sum((d - mean) ** 2 for d in diffs) / (m - 1) if m > 1 else 0.0
    half = 1.96 * math.sqrt(var / m)

    wins = sum(1 for d in diffs if d < 0)        # A strictly cheaper
    losses = sum(1 for d in diffs if d > 0)
    n_eff = wins + losses
    if n_eff:
        z = abs(wins - n_eff / 2) / math.sqrt(n_eff / 4)
        p_sign = math.erfc(z / math.sqrt(2))
    else:
        p_sign = 1.0

    scale = 100.0 / base if base else 0.0
    return {
        "m": m,
        "mean_a": sum(a_costs[k] for k in keys) / m,
        "mean_b": base,
        "delta_pct": mean * scale,
        "lo_pct": (mean - half) * scale,
        "hi_pct": (mean + half) * scale,
        "wins": wins,
        "losses": losses,
        "p_sign": p_sign,
    }


def set_key(tag):
    return (SET_ORDER.index(tag) if tag in SET_ORDER else len(SET_ORDER), tag)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "best_config"),
                    help="root of the runs (default: results/best_config)")
    ap.add_argument("--pair", nargs=2, metavar=("A", "B"), default=DEFAULT_PAIR,
                    help=f"the two configurations to compare "
                         f"(default: {' '.join(DEFAULT_PAIR)})")
    ap.add_argument("--csv", action="store_true",
                    help="also write <out>/summary.csv and <out>/paired.csv")
    args = ap.parse_args()

    if not os.path.isdir(args.out):
        raise SystemExit(f"{args.out}: not found — run scripts/run_best_config.sh first")
    configs = CFG_ORDER + [c for c in args.pair if c not in CFG_ORDER]
    runs = load_runs(args.out, configs)
    if not runs:
        raise SystemExit(f"{args.out}: no run directory with a config.json")

    # ------------------------------------------------------------ per run
    rows = []
    for (set_tag, config) in sorted(runs, key=lambda k: (set_key(k[0]),
                                                         configs.index(k[1]))):
        r = runs[(set_tag, config)]
        res = r["cfg"].get("result", {})
        ok, detail = read_validation(r["dir"])
        budget, kind = read_budget(r["cfg"])
        inst = res.get("instances") or 0
        cpu = res.get("time_cpu_s")
        rows.append({
            "set": set_tag,
            "config": config,
            "budget": budget,
            "budget_kind": kind,
            "instances": inst,
            "mean_cost": res.get("cost_after_sa"),
            "wall_s": res.get("time_wall_s"),
            "cpu_ms_per_inst": (1e3 * cpu / inst) if (cpu and inst) else None,
            "drift_max": res.get("drift_max"),
            "valid": "OK" if ok else ("FAIL" if ok is False else "-"),
            "valid_detail": detail,
            "bks_gap_pct": read_bks_gap(r["dir"]),
            "run": os.path.basename(r["dir"]),
        })

    print(f"### runs under {os.path.relpath(args.out, ROOT)}\n")
    hdr = (f"{'set':<8} {'config':<8} {'budget':>12} {'inst':>6} {'mean cost':>14} "
           f"{'cpu ms/inst':>12} {'wall s':>9} {'drift':>10} "
           f"{'validate':<12} {'gap to ref':>11}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def f(v, spec):
            return format(v, spec) if v is not None else "-"
        gap = f"{r['bks_gap_pct']:+.3f} %" if r["bks_gap_pct"] is not None else "-"
        print(f"{r['set']:<8} {r['config']:<8} {r['budget']:>12} {r['instances']:>6} "
              f"{f(r['mean_cost'], '14.5f')} {f(r['cpu_ms_per_inst'], '12.2f')} "
              f"{f(r['wall_s'], '9.2f')} {f(r['drift_max'], '10.2e')} "
              f"{r['valid'] + ' (' + r['valid_detail'] + ')':<12} {gap:>11}")

    print("\n`budget` is what was asked for: `<x>s` a wall-clock budget per "
          "instance (with the mean\nstep count it bought), `<n>M` a fixed step "
          "count. The two are not comparable — a step\nof `cand` costs more "
          "than a step of `prev`, which is the whole reason for --sa-time.")

    # ---------------------------------------------------- paired A vs B
    a_tag, b_tag = args.pair
    print(f"\n### paired: {a_tag} against {b_tag}, same budget per instance")
    print(f"    negative delta = {a_tag} wins\n")
    hdr2 = (f"{'set':<8} {'m':>7} {a_tag:>13} {b_tag:>13} "
            f"{'delta %':>9} {'95 % CI':>20} {'win/loss':>13} {'sign p':>9}")
    print(hdr2)
    print("-" * len(hdr2))

    prows = []
    for set_tag in sorted({s for s, _ in runs}, key=set_key):
        a = runs.get((set_tag, a_tag))
        b = runs.get((set_tag, b_tag))
        if not a or not b:
            continue
        st = paired(read_costs(a["dir"]), read_costs(b["dir"]))
        if st is None:
            print(f"{set_tag:<8} (no instance in common)")
            continue
        sig = st["hi_pct"] < 0 or st["lo_pct"] > 0
        sig = sig and st["p_sign"] < 0.05
        mark = "*" if sig else " "
        ci = f"[{st['lo_pct']:+.3f}, {st['hi_pct']:+.3f}]"
        print(f"{set_tag:<8} {st['m']:>7} {st['mean_a']:>13.5f} {st['mean_b']:>13.5f} "
              f"{st['delta_pct']:>+8.3f}{mark} {ci:>20} "
              f"{str(st['wins']) + '/' + str(st['losses']):>13} {st['p_sign']:>9.2e}")
        prows.append({"set": set_tag, "a": a_tag, "b": b_tag,
                      "significant": int(sig), **st})
    if not prows:
        print(f"(no set has both {a_tag} and {b_tag} — "
              f"try --pair with one of: {', '.join(sorted({c for _, c in runs}))})")

    print("\n* = the 95 % CI excludes zero and the sign test rejects at 5 %.")

    if args.csv:
        for name, data in (("summary.csv", rows), ("paired.csv", prows)):
            if not data:
                continue
            path = os.path.join(args.out, name)
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                w.writeheader()
                w.writerows(data)
            print(f"-> {os.path.relpath(path, ROOT)}")

    return 1 if any(r["valid"] == "FAIL" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
