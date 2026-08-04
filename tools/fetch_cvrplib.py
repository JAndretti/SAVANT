#!/usr/bin/env python3
"""
fetch_cvrplib.py — download the CVRPLib benchmark sets that ship reference
solutions: X, XL and XML100.

    Set X     Uchoa, Pecin, Pessoa, Poggi, Subramanian, Vidal (2017)
              100 instances, 100..1000 customers, BKS included
    Set XL    Queiroga, Sadykov, Uchoa, Vidal (2026)
              100 instances, 1047..10000 customers, BKS from the CVRPLib
              BKS Challenge
    XML100    Queiroga, Sadykov, Uchoa, Vidal (2022)
              10000 instances, 100 customers, *proven optimal* solutions

Unlike the NeuOpt sets fetched by fetch_neuopt.py, all three use the TSPLIB
EUC_2D convention: **distances are rounded to the nearest integer**. This is
not taken on faith — `--verify` recomputes the shipped reference cost both ways
and reports which convention reproduces it. Every instance checked so far
matches the rounded one exactly, and the float one never. Consequences:

    ./cw --dir data/cvrplib/X --round            <- required
    tools/run_hgs.py --dir data/cvrplib/X        <- --round defaults to 1 here

Forgetting it does not crash, it silently answers a different problem.

For each set the script writes:

    data/cvrplib/<SET>/          the .vrp and .sol files as distributed
    data/cvrplib/<SET>.cvrpb     a bundle, instances sorted by file name
    data/cvrplib/<SET>_bks.csv   idx,name,n,capacity,bks_cost,bks_routes

The bundle exists so that tools/validate.py works unchanged; its instance
order is the sorted file name order, which is also the order `cw --dir` uses,
so index k means the same instance everywhere. The CSV is the reference these
sets are actually worth measuring against — an optimum or a BKS is a far better
yardstick than another heuristic.

The archives are .7z. Extraction uses whichever reader the machine has —
bsdtar (macOS, Debian libarchive-tools), the 7-Zip CLI, or the pure-Python
py7zr — so no particular one is required; see extract().

Usage:
    python3 tools/fetch_cvrplib.py                    # all three sets
    python3 tools/fetch_cvrplib.py --sets X XL
    python3 tools/fetch_cvrplib.py --sets X --verify  # check the convention
    uv run --with py7zr tools/fetch_cvrplib.py        # no system extractor
"""

import argparse
import csv
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "cvrplib")

BASE = "https://galgos.inf.puc-rio.br/cvrplib"

# name -> (list of (url, subdirectory of the archive holding the files))
SETS = {
    "X": [(f"{BASE}/index.php/en/download/instance-set/17", "X")],
    "XL": [(f"{BASE}/index.php/en/download/instance-set/21", "XL")],
    "XML100": [(f"{BASE}/uploads/files/xml100/instances.7z", "instances"),
               (f"{BASE}/uploads/files/xml100/solutions.7z", "solutions")],
}

ROUTE_RE = re.compile(r"^Route #(\d+)\s*:\s*(.*)$")
COST_RE = re.compile(r"^Cost\s+([0-9.eE+-]+)\s*$")


def download(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "SAVANT/fetch"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                sys.stdout.write(f"\r          {100 * done // total:3d} % "
                                 f"({done >> 10} kio)")
                sys.stdout.flush()
    if total:
        sys.stdout.write("\r" + " " * 40 + "\r")


def extract(archive, into):
    """Unpack a .7z, using whatever this machine happens to have.

    Tried in order, so that a plain `python3 tools/fetch_cvrplib.py` works on a
    machine with no 7-Zip and no root: bsdtar (macOS, Debian libarchive-tools),
    the 7-Zip CLI under any of its four names, then the pure-Python py7zr.
    """
    for exe, args in (("bsdtar", ["-xf", archive, "-C", into]),
                      ("7zz", ["x", "-y", f"-o{into}", archive]),
                      ("7z", ["x", "-y", f"-o{into}", archive]),
                      ("7za", ["x", "-y", f"-o{into}", archive]),
                      ("7zr", ["x", "-y", f"-o{into}", archive])):
        if shutil.which(exe):
            subprocess.run([exe] + args, check=True,
                           stdout=subprocess.DEVNULL)
            return
    try:
        import py7zr
    except ImportError:
        raise SystemExit(
            "no .7z extractor found. Any one of these is enough:\n"
            "  Debian/Ubuntu : apt install libarchive-tools   (bsdtar)\n"
            "  Fedora/RHEL   : dnf install bsdtar\n"
            "  Arch          : pacman -S libarchive\n"
            "  macOS         : bsdtar is already there\n"
            "  no root       : uv run --with py7zr tools/fetch_cvrplib.py ...")
    with py7zr.SevenZipFile(archive, "r") as z:
        z.extractall(path=into)


def read_vrp(path):
    """(n, capacity, xs, ys, demands) with index 0 = depot.

    Tolerant of the layout differences across the sets: Set X separates its
    `KEY : VALUE` headers with tabs, XML100 with spaces, and DEPOT_SECTION is
    indented in one and not the other.
    """
    coords, dem, cap, depot = {}, {}, None, 1
    section = None
    with open(path) as f:
        for line in f:
            tok = line.replace(":", " ").split()
            if not tok:
                continue
            key = tok[0].upper()
            if key in ("NODE_COORD_SECTION", "DEMAND_SECTION",
                       "DEPOT_SECTION", "EOF"):
                section = key
                continue
            if key == "CAPACITY":
                cap = float(tok[-1])
                continue
            if key in ("NAME", "COMMENT", "TYPE", "DIMENSION",
                       "EDGE_WEIGHT_TYPE"):
                continue
            if section == "NODE_COORD_SECTION" and len(tok) >= 3:
                coords[int(tok[0])] = (float(tok[1]), float(tok[2]))
            elif section == "DEMAND_SECTION" and len(tok) >= 2:
                dem[int(tok[0])] = float(tok[1])
            elif section == "DEPOT_SECTION" and tok[0].lstrip("-").isdigit():
                v = int(tok[0])
                if v >= 1:
                    depot = v

    if cap is None or not coords:
        raise ValueError(f"{path}: incomplete instance")
    others = sorted(i for i in coords if i != depot)
    n = len(others)
    xs = [coords[depot][0]] + [coords[i][0] for i in others]
    ys = [coords[depot][1]] + [coords[i][1] for i in others]
    ds = [0.0] + [dem.get(i, 0.0) for i in others]
    return n, cap, xs, ys, ds


def read_sol(path):
    """(routes, cost) from a CVRPLib .sol; customers are 1-based already."""
    routes, cost = [], None
    with open(path) as f:
        for line in f:
            m = ROUTE_RE.match(line.strip())
            if m:
                routes.append([int(v) for v in m.group(2).split()])
                continue
            m = COST_RE.match(line.strip())
            if m:
                cost = float(m.group(1))
    return routes, cost


def route_problems(routes, ds, cap, n):
    """Feasibility faults in a reference route list, as a short list of strings.

    12 of the 10000 XML100 .sol files are damaged as distributed: customers
    missing, customers listed twice. Their `Cost` field still looks like the
    published optimum, so the cost is kept as the reference while the route
    list is marked unusable — checking this first stops a corrupt file from
    being misread as evidence about the rounding convention.
    """
    seen = [0] * (n + 1)
    dup = 0
    over = 0
    for r in routes:
        if sum(ds[c] for c in r if 1 <= c <= n) > cap + 1e-9:
            over += 1
        for c in r:
            if not 1 <= c <= n:
                continue
            dup += seen[c]
            seen[c] = 1
    missing = sum(1 for c in range(1, n + 1) if not seen[c])
    out = []
    if missing:
        out.append(f"{missing} unserved")
    if dup:
        out.append(f"{dup} duplicated")
    if over:
        out.append(f"{over} overloaded")
    return out


def route_cost(xs, ys, routes, rounded):
    if rounded:
        def d(a, b):
            return math.floor(math.hypot(xs[a] - xs[b], ys[a] - ys[b]) + 0.5)
    else:
        def d(a, b):
            return math.hypot(xs[a] - xs[b], ys[a] - ys[b])
    total = 0.0
    for r in routes:
        prev = 0
        for c in r:
            total += d(prev, c)
            prev = c
        total += d(prev, 0)
    return total


def write_bundle(path, instances):
    """Same .cvrpb layout as fetch_neuopt.py, so validate.py reads it as is."""
    with open(path, "wb") as f:
        f.write(b"CVRPBIN1")
        f.write(struct.pack("<II", len(instances), 0))
        for n, cap, xs, ys, ds in instances:
            f.write(struct.pack("<I", n))
            f.write(struct.pack("<d", float(cap)))
            f.write(struct.pack(f"<{n + 1}d", *xs))
            f.write(struct.pack(f"<{n + 1}d", *ys))
            f.write(struct.pack(f"<{n + 1}d", *ds))


def fetch_set(name, out, keep_archives, verify):
    target = os.path.join(out, name)
    os.makedirs(target, exist_ok=True)

    have = len([f for f in os.listdir(target) if f.endswith(".vrp")])
    if have:
        print(f"{name}: {have} .vrp already in {os.path.relpath(target, ROOT)}, "
              f"skipping download")
    else:
        tmp = tempfile.mkdtemp(prefix=f"cvrplib-{name}-")
        try:
            for url, sub in SETS[name]:
                print(f"{name}: downloading {url}")
                arc = os.path.join(tmp, os.path.basename(url) or f"{name}.7z")
                if not arc.endswith(".7z"):
                    arc += ".7z"
                download(url, arc)
                extract(arc, tmp)
                src = os.path.join(tmp, sub)
                if not os.path.isdir(src):
                    raise SystemExit(f"{name}: {sub}/ not found in the archive")
                for fn in os.listdir(src):
                    if fn.endswith((".vrp", ".sol")):
                        shutil.move(os.path.join(src, fn),
                                    os.path.join(target, fn))
                if keep_archives:
                    shutil.copy(arc, os.path.join(out, os.path.basename(arc)))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    names = sorted(f[:-4] for f in os.listdir(target) if f.endswith(".vrp"))
    if not names:
        raise SystemExit(f"{name}: no instance after extraction")

    instances, rows = [], []
    mismatch = corrupt = 0
    for k, base in enumerate(names):
        n, cap, xs, ys, ds = read_vrp(os.path.join(target, base + ".vrp"))
        instances.append((n, cap, xs, ys, ds))
        solp = os.path.join(target, base + ".sol")
        bks_cost = bks_routes = ""
        ref_ok = ""
        if os.path.exists(solp):
            routes, cost = read_sol(solp)
            bks_cost = cost if cost is not None else ""
            bks_routes = len([r for r in routes if r])
            faults = route_problems(routes, ds, cap, n)
            ref_ok = 0 if faults else 1
            if faults:
                corrupt += 1
                print(f"  {base:<24} reference ROUTES unusable: "
                      f"{', '.join(faults)} (cost {cost:.0f} kept)")
            elif verify and cost is not None:
                cr = route_cost(xs, ys, routes, True)
                cf = route_cost(xs, ys, routes, False)
                verdict = ("rounded" if abs(cr - cost) < 0.5
                           else "float" if abs(cf - cost) < 0.5 else "NEITHER")
                if verdict != "rounded":
                    mismatch += 1
                    print(f"  {base:<24} ref={cost:<12.0f} rounded={cr:<12.0f} "
                          f"float={cf:<14.3f} -> {verdict}")
        rows.append({"idx": k, "name": base, "n": n, "capacity": cap,
                     "bks_cost": bks_cost, "bks_routes": bks_routes,
                     "ref_ok": ref_ok})

    bundle = os.path.join(out, name + ".cvrpb")
    write_bundle(bundle, instances)
    csvp = os.path.join(out, name + "_bks.csv")
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "name", "n", "capacity",
                                          "bks_cost", "bks_routes", "ref_ok"])
        w.writeheader()
        w.writerows(rows)

    ns = [r["n"] for r in rows]
    withbks = sum(1 for r in rows if r["bks_cost"] != "")
    print(f"{name}: {len(rows)} instances, n = {min(ns)}..{max(ns)}, "
          f"{withbks} with a reference solution")
    print(f"  {os.path.relpath(target, ROOT)}/            .vrp + .sol")
    print(f"  {os.path.relpath(bundle, ROOT)}     bundle (sorted by name)")
    print(f"  {os.path.relpath(csvp, ROOT)}  reference costs")
    if corrupt:
        print(f"  !! {corrupt} of {withbks} reference route lists are damaged "
              f"as distributed (ref_ok=0 in the CSV); their cost is kept, the "
              f"routes are not usable", file=sys.stderr)
    if verify:
        checked = withbks - corrupt
        if mismatch:
            print(f"  !! {mismatch} intact instance(s) did NOT match the "
                  f"rounded convention", file=sys.stderr)
        else:
            print(f"  convention verified on {checked} intact reference "
                  f"solution(s): integer rounding, exactly")
    return len(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Download CVRPLib sets X, XL and XML100 (all rounded EUC_2D)")
    ap.add_argument("--sets", nargs="+", default=list(SETS),
                    choices=list(SETS), help="which sets (default: all)")
    ap.add_argument("--out", default=OUT, help=f"output root (default: {OUT})")
    ap.add_argument("--keep-archives", action="store_true",
                    help="keep the downloaded .7z next to the extracted files")
    ap.add_argument("--verify", action="store_true",
                    help="recompute every shipped reference cost both ways and "
                         "report which convention reproduces it")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    for name in args.sets:
        fetch_set(name, args.out, args.keep_archives, args.verify)
        print()

    print("these sets are ROUNDED — remember `--round` for cw; run_hgs.py "
          "defaults to -round 1 when given --dir/--bundle under data/cvrplib/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
