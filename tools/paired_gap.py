#!/usr/bin/env python3
"""
paired_gap.py — paired per-instance comparison of two runs.

Comparing two solvers by their mean objective alone throws away the fact that
both solved the *same* instances. On the NeuOpt CVRP-100 set the per-instance
standard deviation is ~1.4, so the standard error of a mean over 1000
instances is ~0.045 — about 0.3 %, which is larger than most of the gaps worth
resolving. The paired difference has a far smaller standard error, because the
instance-to-instance variation cancels.

This script therefore reports, over the instances both runs solved:
  * the mean gap B vs A, with the standard error *of the paired difference*
  * how often each side wins
  * a two-sided Wilcoxon-style sign test, so the verdict does not assume
    the differences are normal

Usage:
    python3 tools/paired_gap.py results/<run_A> results/<run_B>
"""

import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from validate import read_solutions  # noqa: E402


def costs(run):
    """{instance index: cost} from a run directory's solutions.txt."""
    path = run if os.path.isfile(run) else os.path.join(run, "solutions.txt")
    sols, _ = read_solutions(path)
    return {idx: cost for idx, _, _, _, cost, _ in sols}


def norm_sf(z):
    """Upper tail of the standard normal, via erfc."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def main():
    ap = argparse.ArgumentParser(description="Paired comparison of two runs")
    ap.add_argument("run_a", help="reference run (directory or solutions.txt)")
    ap.add_argument("run_b", help="run to compare against it")
    args = ap.parse_args()

    A, B = costs(args.run_a), costs(args.run_b)
    common = sorted(set(A) & set(B))
    if not common:
        raise SystemExit("the two runs share no instance")
    missing = (len(A) - len(common)) + (len(B) - len(common))

    diffs = [B[i] - A[i] for i in common]
    n = len(diffs)
    mean_a = sum(A[i] for i in common) / n
    mean_b = sum(B[i] for i in common) / n
    md = sum(diffs) / n
    var = sum((d - md) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)

    wins_b = sum(1 for d in diffs if d < -1e-9)
    wins_a = sum(1 for d in diffs if d > 1e-9)
    ties = n - wins_a - wins_b

    # sign test on the non-tied pairs, normal approximation
    m = wins_a + wins_b
    if m:
        z = abs(wins_b - m / 2.0) / math.sqrt(m / 4.0)
        p = 2.0 * norm_sf(z)
    else:
        z = p = float("nan")

    print(f"A = {os.path.basename(os.path.normpath(args.run_a))}")
    print(f"B = {os.path.basename(os.path.normpath(args.run_b))}")
    print(f"paired on {n} instance(s)"
          + (f"  ({missing} not shared, ignored)" if missing else ""))
    print()
    print(f"  mean A          : {mean_a:.6f}")
    print(f"  mean B          : {mean_b:.6f}")
    print(f"  mean difference : {md:+.6f}  +/- {se:.6f} (SE, paired)")
    print(f"  relative        : {100 * md / mean_a:+.4f} %  "
          f"+/- {100 * se / mean_a:.4f} %")
    lo, hi = md - 1.96 * se, md + 1.96 * se
    print(f"  95 % CI         : [{100 * lo / mean_a:+.4f} %, "
          f"{100 * hi / mean_a:+.4f} %]")
    print()
    print(f"  B better on     : {wins_b:5d} / {n}  ({100 * wins_b / n:.1f} %)")
    print(f"  A better on     : {wins_a:5d} / {n}  ({100 * wins_a / n:.1f} %)")
    print(f"  ties            : {ties:5d}")
    if m:
        print(f"  sign test       : z = {z:.2f}, p = {p:.3g}")

    # For context: the SE of each mean on its own, which is what an unpaired
    # comparison against a published figure would be stuck with.
    va = sum((A[i] - mean_a) ** 2 for i in common) / (n - 1) if n > 1 else 0.0
    print()
    print(f"  (unpaired SE of mean A alone: {math.sqrt(va / n):.6f}, "
          f"i.e. {100 * math.sqrt(va / n) / mean_a:.3f} % — the precision "
          f"floor\n   for any comparison against a number from another paper)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
