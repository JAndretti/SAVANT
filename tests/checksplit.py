#!/usr/bin/env python3
"""Validates the C O(n) Split against a naive O(n^2) dynamic program on
exactly the same giant tour."""

import math
import subprocess
import sys

from _paths import BIN
from check import read_bundle


def split_naive(tour, dem, cap, d):
    """Prins DP: P[j] = min over feasible i of P[i] + cost(i,j]"""
    n = len(tour)
    P = [float("inf")] * (n + 1)
    P[0] = 0.0
    pred = [0] * (n + 1)
    for i in range(n):
        load = 0.0
        cost = 0.0
        for j in range(i + 1, n + 1):
            load += dem[tour[j - 1]]
            if load > cap + 1e-9:
                break
            if j == i + 1:
                cost = 2 * d(0, tour[i])
            else:
                cost = (
                    cost
                    - d(tour[j - 2], 0)
                    + d(tour[j - 2], tour[j - 1])
                    + d(tour[j - 1], 0)
                )
            if P[i] + cost < P[j] - 1e-12:
                P[j] = P[i] + cost
                pred[j] = i
    return P[n]


def routes_from_sol(path):
    from validate import read_solutions

    return [t[5] for t in read_solutions(path)[0]]


if __name__ == "__main__":
    bundle, lim = sys.argv[1], int(sys.argv[2])
    extra = sys.argv[3:]  # e.g. --sa-steps 20000
    insts = read_bundle(bundle, lim)
    # reference solution WITHOUT split, routes written to /tmp/s0.txt
    subprocess.run(
        [
            BIN,
            "--bundle",
            bundle,
            "--limit",
            str(lim),
            "-q",
            "--split",
            "off",
            "--sol",
            "/tmp/s0.txt",
            "--csv",
            "/tmp/c0.csv",
        ]
        + extra,
        capture_output=True,
        text=True,
    )
    # same solution WITH a final split
    subprocess.run(
        [
            BIN,
            "--bundle",
            bundle,
            "--limit",
            str(lim),
            "-q",
            "--split",
            "end",
            "--split-tour",
            "routes",
            "--csv",
            "/tmp/c1.csv",
        ]
        + extra,
        capture_output=True,
        text=True,
    )
    R0 = routes_from_sol("/tmp/s0.txt")
    c1 = [float(l.split(",")[4]) for l in open("/tmp/c1.csv").read().splitlines()[1:]]
    worst = 0.0
    for k, (n, cap, xs, ys, ds) in enumerate(insts):
        d = lambda a, b: math.hypot(xs[a] - xs[b], ys[a] - ys[b])
        tour = [c for r in R0[k] for c in r]
        assert sorted(tour) == list(range(1, n + 1)), "incomplete giant tour"
        ref = split_naive(tour, ds, cap, d)
        rel = abs(ref - c1[k]) / ref
        worst = max(worst, rel)
        print(
            f"inst {k:2d}: split C={c1[k]:.6f}  naive DP={ref:.6f}  "
            f"{'OK' if rel < 1e-9 else 'DIFF'}"
        )
    print(f"max relative gap = {worst:.3e}")
