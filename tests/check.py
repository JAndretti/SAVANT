#!/usr/bin/env python3
"""Independent check: re-implements Clarke & Wright naively (explicit route
lists, full sort of the savings) and compares against the C binary.
Also checks the validity of the solutions written by --sol."""

import math
import struct
import subprocess
import sys

from _paths import BIN


def read_bundle(path, limit=None):
    out = []
    with open(path, "rb") as f:
        assert f.read(8) == b"CVRPBIN1"
        nb, _ = struct.unpack("<II", f.read(8))
        for k in range(nb if limit is None else min(nb, limit)):
            (n,) = struct.unpack("<I", f.read(4))
            (cap,) = struct.unpack("<d", f.read(8))
            xs = struct.unpack(f"<{n + 1}d", f.read(8 * (n + 1)))
            ys = struct.unpack(f"<{n + 1}d", f.read(8 * (n + 1)))
            ds = struct.unpack(f"<{n + 1}d", f.read(8 * (n + 1)))
            out.append((n, cap, xs, ys, ds))
    return out


def cw_naive(n, cap, xs, ys, ds):
    d = lambda a, b: math.hypot(xs[a] - xs[b], ys[a] - ys[b])
    sav = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            s = d(0, i) + d(0, j) - d(i, j)
            if s > 0:
                sav.append((s, i, j))
    sav.sort(key=lambda t: -t[0])
    route = {i: [i] for i in range(1, n + 1)}  # route id -> list
    where = {i: i for i in range(1, n + 1)}  # customer -> route id
    load = {i: ds[i] for i in range(1, n + 1)}
    for s, i, j in sav:
        ri, rj = where[i], where[j]
        if ri == rj:
            continue
        if load[ri] + load[rj] > cap + 1e-9:
            continue
        A, B = route[ri], route[rj]
        # i and j must be endpoints
        if A[-1] == i and B[0] == j:
            new = A + B
        elif A[0] == i and B[-1] == j:
            new = B + A
        elif A[-1] == i and B[-1] == j:
            new = A + B[::-1]
        elif A[0] == i and B[0] == j:
            new = A[::-1] + B
        else:
            continue
        route[ri] = new
        load[ri] += load[rj]
        for c in B:
            where[c] = ri
        del route[rj]
    cost = 0.0
    for r in route.values():
        cost += (
            d(0, r[0]) + sum(d(r[t], r[t + 1]) for t in range(len(r) - 1)) + d(r[-1], 0)
        )
    return cost, len(route)


def check_sol(solfile, insts):
    """validate the solution file produced by --sol"""
    from validate import read_solutions

    sols = {t[0]: t[5] for t in read_solutions(solfile)[0]}
    for k, (n, cap, xs, ys, ds) in enumerate(insts):
        rs = sols.get(k)
        if rs is None:
            continue
        seen = sorted(c for r in rs for c in r)
        assert seen == list(range(1, n + 1)), (
            f"instance {k}: missing/duplicated customers"
        )
        for r in rs:
            assert sum(ds[c] for c in r) <= cap + 1e-9, (
                f"instance {k}: capacity exceeded"
            )
    return len(insts)


if __name__ == "__main__":
    bundle, limit = sys.argv[1], int(sys.argv[2])
    insts = read_bundle(bundle, limit)
    # C reference, one line per instance
    out = subprocess.run(
        [
            BIN,
            "--bundle",
            bundle,
            "--limit",
            str(limit),
            "--exact",
            "--per-instance",
            "-q",
            "--sol",
            "/tmp/sol.txt",
        ],
        capture_output=True,
        text=True,
    )
    print(out.stdout.strip().splitlines()[-1] if out.stdout else out.stderr)
    csv = subprocess.run(
        [
            BIN,
            "--bundle",
            bundle,
            "--limit",
            str(limit),
            "--exact",
            "-q",
            "--csv",
            "/tmp/r.csv",
        ],
        capture_output=True,
        text=True,
    )
    ccost = [float(l.split(",")[3]) for l in open("/tmp/r.csv").read().splitlines()[1:]]
    worst = 0.0
    for k, (n, cap, xs, ys, ds) in enumerate(insts):
        c, nr = cw_naive(n, cap, xs, ys, ds)
        rel = abs(c - ccost[k]) / c
        worst = max(worst, rel)
        flag = "OK " if rel < 1e-9 else "DIFF"
        print(f"{flag} inst {k:3d}  C={ccost[k]:.6f}  python={c:.6f}  gap={rel:.2e}")
    print(f"max relative gap = {worst:.3e}")
    print(f"solutions validated: {check_sol('/tmp/sol.txt', insts)}")
