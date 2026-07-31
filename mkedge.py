#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Builds edge-case instances the random generator never produces: tiny
sizes, degenerate geometries, pathological capacities."""
import struct, sys, os, random

def write(path, insts):
    with open(path, "wb") as f:
        f.write(b"CVRPBIN1")
        f.write(struct.pack("<II", len(insts), 0))
        for n, cap, xs, ys, ds in insts:
            f.write(struct.pack("<I", n))
            f.write(struct.pack("<d", float(cap)))
            f.write(struct.pack(f"<{n+1}d", *xs))
            f.write(struct.pack(f"<{n+1}d", *ys))
            f.write(struct.pack(f"<{n+1}d", *ds))

def inst(n, cap, xs, ys, ds):
    assert len(xs) == n + 1 and len(ys) == n + 1 and len(ds) == n + 1
    return (n, cap, xs, ys, ds)

random.seed(1234)
cases = {}

# --- tiny sizes ---
for n in (1, 2, 3, 5):
    xs = [0.5] + [random.random() for _ in range(n)]
    ys = [0.5] + [random.random() for _ in range(n)]
    ds = [0.0] + [float(random.randint(1, 9)) for _ in range(n)]
    cases[f"tiny{n}"] = [inst(n, 30, xs, ys, ds)]

# --- all points coincident (every distance zero) ---
for n in (10, 800):
    cases[f"same{n}"] = [inst(n, 50, [0.3]*(n+1), [0.7]*(n+1),
                              [0.0] + [1.0]*n)]

# --- collinear points (grid degenerate in y) ---
for n in (10, 800):
    xs = [0.0] + [i/n for i in range(n)]
    cases[f"line{n}"] = [inst(n, 50, xs, [0.5]*(n+1), [0.0] + [3.0]*n)]

# --- two far-apart clusters (empty cells in between) ---
n = 600
xs = [0.5] + [random.gauss(0.02, 0.002) if i % 2 else random.gauss(0.98, 0.002)
              for i in range(n)]
ys = [0.5] + [random.gauss(0.5, 0.002) for _ in range(n)]
cases["clusters"] = [inst(n, 40, xs, ys, [0.0] + [2.0]*n)]

# --- capacity equal to the maximum demand: every customer alone ---
n = 40
xs = [0.5] + [random.random() for _ in range(n)]
ys = [0.5] + [random.random() for _ in range(n)]
cases["alone"] = [inst(n, 9, xs, ys, [0.0] + [9.0]*n)]

# --- huge capacity: a single route ---
cases["onebig"] = [inst(n, 1e6, xs, ys, [0.0] + [1.0]*n)]

# --- demand strictly above capacity: INFEASIBLE instance ---
ds = [0.0] + [1.0]*n
ds[7] = 99.0
cases["infeasible"] = [inst(n, 30, xs, ys, ds)]

# --- fractional demands ---
cases["frac"] = [inst(n, 7.5, xs, ys, [0.0] + [round(random.uniform(.1, 2.4), 3)
                                              for _ in range(n)])]

# --- very large coordinates (scale test) ---
cases["huge"] = [inst(n, 30, [1e6 * v for v in xs], [1e6 * v for v in ys],
                      [0.0] + [3.0]*n)]

# --- depot coincident with a customer ---
xs2 = list(xs); ys2 = list(ys); xs2[1] = xs2[0]; ys2[1] = ys2[0]
cases["depotdup"] = [inst(n, 30, xs2, ys2, [0.0] + [3.0]*n)]

out = sys.argv[1] if len(sys.argv) > 1 else "edge"
os.makedirs(out, exist_ok=True)
for name, insts in cases.items():
    write(os.path.join(out, name + ".cvrpb"), insts)
print(" ".join(sorted(cases)))
