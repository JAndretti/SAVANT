#!/usr/bin/env python3
"""
bundle_to_vrp.py — export a .cvrpb bundle as one TSPLIB .vrp file per instance.

The point is to hand *exactly* the instances SAVANT solves to an external
solver (HGS-CVRP, LKH-3), rather than regenerating them from the NeuOpt
pickles: the bundle is the single source of truth, so no divergence in
coordinates or demands is possible.

Coordinates stay floating point in [0, 1] and are written with 17 significant
digits, which round-trips a float64 exactly. HGS's TSPLIB parser reads them as
doubles, so the only thing that can still change the metric is HGS's own
`-round` flag, which must be 0 for these instances.

Usage:
    python3 tools/bundle_to_vrp.py data/cvrp_100.cvrpb data/vrp_100
    python3 tools/bundle_to_vrp.py data/cvrp_100.cvrpb data/vrp_100 --limit 100
"""

import argparse
import os
import struct
import sys


def read_bundle(path, limit=0):
    """Read a .cvrpb bundle: list of (n, capacity, xs, ys, demands)."""
    out = []
    with open(path, "rb") as f:
        if f.read(8) != b"CVRPBIN1":
            raise SystemExit(f"{path}: invalid .cvrpb signature")
        nb, _ = struct.unpack("<II", f.read(8))
        if limit:
            nb = min(nb, limit)
        for _ in range(nb):
            (n,) = struct.unpack("<I", f.read(4))
            (cap,) = struct.unpack("<d", f.read(8))
            xs = struct.unpack(f"<{n + 1}d", f.read(8 * (n + 1)))
            ys = struct.unpack(f"<{n + 1}d", f.read(8 * (n + 1)))
            ds = struct.unpack(f"<{n + 1}d", f.read(8 * (n + 1)))
            out.append((n, cap, xs, ys, ds))
    return out


def write_vrp(path, name, n, cap, xs, ys, ds):
    """One TSPLIB CVRP file. Node 1 is the depot, nodes 2..n+1 the customers,
    so TSPLIB index = internal index + 1.

    The first three lines must be NAME / COMMENT / TYPE, in that order: HGS's
    parser (`InstanceCVRPLIB.cpp`) skips exactly three lines with getline()
    before it starts reading `KEY : VALUE` tokens. Dropping the COMMENT line
    makes it swallow DIMENSION instead, after which `nbClients` stays
    uninitialised and the error it reports points at an unrelated line. This is
    why `fetch_neuopt.py --tsplib`, which emits NAME/TYPE/DIMENSION, cannot be
    fed to HGS directly.
    """
    with open(path, "w") as f:
        f.write(f"NAME : {name}\nCOMMENT : from .cvrpb bundle\nTYPE : CVRP\n")
        f.write(f"DIMENSION : {n + 1}\n")
        f.write(f"EDGE_WEIGHT_TYPE : EUC_2D\nCAPACITY : {int(cap)}\n")
        f.write("NODE_COORD_SECTION\n")
        for i in range(n + 1):
            f.write(f"{i + 1} {xs[i]:.17g} {ys[i]:.17g}\n")
        f.write("DEMAND_SECTION\n")
        for i in range(n + 1):
            f.write(f"{i + 1} {int(ds[i])}\n")
        f.write("DEPOT_SECTION\n1\n-1\nEOF\n")


def instance_name(prefix, k, width):
    return f"{prefix}_{k:0{width}d}"


def main():
    ap = argparse.ArgumentParser(
        description="Export a .cvrpb bundle to TSPLIB .vrp files"
    )
    ap.add_argument("bundle", help="input .cvrpb")
    ap.add_argument("outdir", help="output directory (created if needed)")
    ap.add_argument("--limit", type=int, default=0,
                    help="export only the first N instances (default: all)")
    ap.add_argument("--prefix", default=None,
                    help="file name prefix (default: derived from the bundle)")
    ap.add_argument("--force", action="store_true",
                    help="rewrite files that already exist")
    args = ap.parse_args()

    insts = read_bundle(args.bundle, args.limit)
    if not insts:
        raise SystemExit(f"{args.bundle}: no instance")

    prefix = args.prefix or os.path.splitext(os.path.basename(args.bundle))[0]
    os.makedirs(args.outdir, exist_ok=True)
    width = len(str(len(insts) - 1))

    caps = {c for _, c, _, _, _ in insts}
    ns = {n for n, _, _, _, _ in insts}

    written = skipped = 0
    for k, (n, cap, xs, ys, ds) in enumerate(insts):
        name = instance_name(prefix, k, width)
        path = os.path.join(args.outdir, name + ".vrp")
        if os.path.exists(path) and not args.force:
            skipped += 1
            continue
        if abs(cap - int(cap)) > 1e-12:
            raise SystemExit(
                f"instance {k}: capacity {cap} is not an integer; TSPLIB "
                f"cannot express it"
            )
        write_vrp(path, name, n, cap, xs, ys, ds)
        written += 1

    print(f"{args.bundle} -> {args.outdir}")
    print(f"  instances : {len(insts)}  (n = {sorted(ns)}, Q = {sorted(caps)})")
    print(f"  written   : {written}" + (f"  (skipped {skipped}, already there)"
                                        if skipped else ""))
    print(f"  naming    : {instance_name(prefix, 0, width)}.vrp ... "
          f"{instance_name(prefix, len(insts) - 1, width)}.vrp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
