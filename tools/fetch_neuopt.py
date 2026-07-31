#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_neuopt.py — fetches the CVRP test sets from

    Ma, Y.; Cao, Z.; Chee, Y. M. (2023).
    "Learning to Search Feasible and Infeasible Regions of Routing Problems
     with Flexible Neural k-Opt", NeurIPS 2023.  arXiv:2310.18264
    https://github.com/yining043/NeuOpt   (datasets/ folder)

and converts them into a format ./cw can read directly.

Contents of the .pkl files (Nazari/Kool convention, reused by NeuOpt): a list
of instances, each one a tuple

    (depot[2], loc[n][2], demand[n], capacity)

with coordinates U[0,1]^2, integer demands U{1,...,9} and
    n =  20 -> Q = 30,  10000 instances
    n =  50 -> Q = 40,  10000 instances
    n = 100 -> Q = 50,  10000 instances
    n = 200 -> Q = 70,   1000 instances

Outputs:
  * data/cvrp_<n>.cvrpb   binary bundle (default, instant to read in C)
  * data/cvrp_<n>/*.vrp   one TSPLIB file per instance (--tsplib option)

Examples:
    python3 fetch_neuopt.py                    # all 4 sizes, binary format
    python3 fetch_neuopt.py --sizes 100 200
    python3 fetch_neuopt.py --sizes 200 --tsplib --max 50
"""

import argparse
import os
import pickle
import struct
import sys
import urllib.request

BASE = "https://raw.githubusercontent.com/yining043/NeuOpt/main/datasets/"
SIZES = [20, 50, 100, 200]
MAGIC = b"CVRPBIN1"


def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  [cache] {dest}")
        return
    print(f"  [get]   {url}")
    tmp = dest + ".part"
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                sys.stdout.write(f"\r          {100 * done // total:3d} %  ({done >> 20} Mio)")
                sys.stdout.flush()
    if total:
        sys.stdout.write("\r" + " " * 40 + "\r")
    os.replace(tmp, dest)


def write_bundle(path, instances):
    """Little-endian binary bundle:
         'CVRPBIN1' | u32 count | u32 0
         then per instance: u32 n | f64 Q | f64 x[n+1] | f64 y[n+1] | f64 d[n+1]
       (index 0 = depot)"""
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", len(instances), 0))
        for depot, loc, dem, cap in instances:
            n = len(loc)
            xs = [float(depot[0])] + [float(p[0]) for p in loc]
            ys = [float(depot[1])] + [float(p[1]) for p in loc]
            ds = [0.0] + [float(d) for d in dem]
            f.write(struct.pack("<I", n))
            f.write(struct.pack("<d", float(cap)))
            f.write(struct.pack(f"<{n + 1}d", *xs))
            f.write(struct.pack(f"<{n + 1}d", *ys))
            f.write(struct.pack(f"<{n + 1}d", *ds))


def write_tsplib(folder, instances, prefix):
    """One file per instance. Coordinates stay floating point in [0,1]: the
    solver does not round them (unless --round is given), so the cost obtained
    is directly comparable with the published values."""
    os.makedirs(folder, exist_ok=True)
    width = len(str(len(instances)))
    for k, (depot, loc, dem, cap) in enumerate(instances):
        n = len(loc)
        name = f"{prefix}_{k:0{width}d}"
        with open(os.path.join(folder, name + ".vrp"), "w") as f:
            f.write(f"NAME : {name}\nTYPE : CVRP\nDIMENSION : {n + 1}\n")
            f.write(f"EDGE_WEIGHT_TYPE : EUC_2D\nCAPACITY : {int(cap)}\n")
            f.write("NODE_COORD_SECTION\n")
            f.write(f"1 {float(depot[0]):.17g} {float(depot[1]):.17g}\n")
            for i, p in enumerate(loc):
                f.write(f"{i + 2} {float(p[0]):.17g} {float(p[1]):.17g}\n")
            f.write("DEMAND_SECTION\n1 0\n")
            for i, d in enumerate(dem):
                f.write(f"{i + 2} {int(d)}\n")
            f.write("DEPOT_SECTION\n1\n-1\nEOF\n")


def main():
    ap = argparse.ArgumentParser(description="Fetch the NeuOpt CVRP sets (NeurIPS 2023)")
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES, choices=SIZES,
                    help="sizes to fetch (default: all)")
    ap.add_argument("--out", default="data", help="output directory (default: data)")
    ap.add_argument("--tsplib", action="store_true",
                    help="also write one .vrp file per instance")
    ap.add_argument("--max", type=int, default=0,
                    help="convert only the first N instances")
    ap.add_argument("--keep-pkl", action="store_true", help="keep the downloaded .pkl files")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    raw = os.path.join(args.out, "pkl")
    os.makedirs(raw, exist_ok=True)

    for n in args.sizes:
        print(f"CVRP{n}")
        pkl = os.path.join(raw, f"cvrp_{n}.pkl")
        download(BASE + f"cvrp_{n}.pkl", pkl)
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        if args.max:
            data = data[: args.max]
        caps = {inst[3] for inst in data}
        print(f"  {len(data)} instances, n={len(data[0][1])}, Q={caps}")

        bundle = os.path.join(args.out, f"cvrp_{n}.cvrpb")
        write_bundle(bundle, data)
        print(f"  [ok]    {bundle} ({os.path.getsize(bundle) >> 20} Mio)")

        if args.tsplib:
            folder = os.path.join(args.out, f"cvrp_{n}")
            write_tsplib(folder, data, f"cvrp{n}")
            print(f"  [ok]    {folder}/ ({len(data)} fichiers .vrp)")

        if not args.keep_pkl:
            os.remove(pkl)

    if not args.keep_pkl:
        try:
            os.rmdir(raw)
        except OSError:
            pass
    print("\nExample usage:")
    print("  ./cw --bundle data/cvrp_100.cvrpb")
    print("  ./cw --dir data/cvrp_200 --2opt")


if __name__ == "__main__":
    main()
