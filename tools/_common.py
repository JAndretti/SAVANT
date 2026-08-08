#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Helpers shared by the external-solver runners (run_hgs.py, run_lkh.py,
run_ails.py).

Only what was literally the same code in the three of them lives here: the
independent cost recomputation and the binary fingerprint. The redundancy that
matters -- C solver vs Python checker -- is untouched: `recompute` below is
still a second implementation next to the solvers', and `validate.py` redoes
the same work a third time from the files on disk, sharing nothing with either.
What this module removes is the three-way copy inside tools/, which would
otherwise drift silently.
"""

import datetime as _dt
import hashlib
import math
import os


def recompute(routes, xs, ys, ds, cap, n, rounded=False):
    """Cost recomputed from the coordinates, plus a feasibility verdict.

    Deliberately independent of the external solver's own arithmetic: this is
    what lands in solutions.txt, and validate.py will redo it a third time.

    `rounded` selects the integer EUC_2D convention of CVRPLib (round half up)
    instead of the floating-point one.

    Returns (total_cost, n_routes, problems), `problems` being a list of
    human-readable strings; an empty list means the solution is valid.
    """
    if rounded:
        def d(a, b):
            return math.floor(math.hypot(xs[a] - xs[b], ys[a] - ys[b]) + 0.5)
    else:
        def d(a, b):
            return math.hypot(xs[a] - xs[b], ys[a] - ys[b])

    seen = [0] * (n + 1)
    total = 0.0
    problems = []
    nroutes = 0
    for r, route in enumerate(routes):
        if not route:
            continue
        nroutes += 1
        load = 0.0
        prev = 0
        for c in route:
            if not 1 <= c <= n:
                problems.append(f"customer {c} out of range")
                continue
            if seen[c]:
                problems.append(f"customer {c} served twice")
            seen[c] = 1
            total += d(prev, c)
            load += ds[c]
            prev = c
        total += d(prev, 0)
        if load > cap + 1e-9:
            problems.append(f"route {r} overloaded ({load:g} > {cap:g})")
    missing = [c for c in range(1, n + 1) if not seen[c]]
    if missing:
        problems.append(f"{len(missing)} customer(s) unserved, e.g. {missing[:5]}")
    return total, nroutes, problems


def binary_fingerprint(path, root=None, mtime=False):
    """sha256 (16 hex chars) + size of the solver binary, so a run directory
    records *which* build produced it. `root`, when given, makes the recorded
    path relative to it."""
    try:
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()[:16]
        out = {"path": os.path.relpath(path, root) if root else path,
               "sha256_16": digest}
        if mtime:
            out["mtime"] = _dt.datetime.fromtimestamp(
                os.path.getmtime(path)).isoformat(timespec="seconds")
        out["size"] = os.path.getsize(path)
        return out
    except OSError:
        return {"path": path}
