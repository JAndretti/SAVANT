"""Instance/solution I/O and an independent feasibility + cost checker.

Conventions (shared with SAVANT/cw.c):
  - index 0 is the depot, customers are 1..n in file order with the depot removed;
  - CVRPLIB .sol files number customers the same way (node id - 1, depot = node 1);
  - `rounded` = TSPLIB EUC_2D nint distances (X, XL, XML100); real distances otherwise.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@dataclass
class Instance:
    name: str
    n: int                 # customers, depot excluded
    cap: float
    xy: np.ndarray         # (n+1, 2), row 0 = depot
    dem: np.ndarray        # (n+1,), dem[0] = 0


def read_vrp(path: str | Path) -> Instance:
    path = Path(path)
    lines = path.read_text().splitlines()
    dim, cap, depot = -1, -1.0, 1
    coords: dict[int, tuple[float, float]] = {}
    demand: dict[int, float] = {}
    section = None
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        head = line.split(":")[0].strip()
        if head == "DIMENSION":
            dim = int(line.split(":")[1])
        elif head == "CAPACITY":
            cap = float(line.split(":")[1])
        elif head == "EDGE_WEIGHT_TYPE":
            if line.split(":")[1].strip() != "EUC_2D":
                raise ValueError(f"{path}: only EUC_2D is supported")
        elif line in ("NODE_COORD_SECTION", "DEMAND_SECTION", "DEPOT_SECTION"):
            section = line
        elif line == "EOF":
            break
        elif section == "NODE_COORD_SECTION":
            i, x, y = line.split()
            coords[int(i)] = (float(x), float(y))
        elif section == "DEMAND_SECTION":
            i, d = line.split()
            demand[int(i)] = float(d)
        elif section == "DEPOT_SECTION":
            v = int(line.split()[0])
            if v >= 1:
                depot = v
    if dim < 2 or cap < 0 or len(coords) != dim:
        raise ValueError(f"{path}: malformed instance")
    order = [depot] + [i for i in range(1, dim + 1) if i != depot]
    xy = np.array([coords[i] for i in order], dtype=np.float64)
    dem = np.array([demand.get(i, 0.0) for i in order], dtype=np.float64)
    dem[0] = 0.0
    return Instance(path.stem, dim - 1, cap, xy, dem)


def write_vrp(path: str | Path, name: str, xy: np.ndarray, dem: np.ndarray, cap: float,
              comment: str = "generated") -> None:
    """xy (n+1,2) and dem (n+1,) with the depot at row 0. The header keeps the
    CVRPLIB line order: HGS-CVRP skips the first three lines unread and FILO
    parses them in this order, so the COMMENT line is not optional."""
    out = [f"NAME : {name}", f"COMMENT : {comment}", "TYPE : CVRP", f"DIMENSION : {len(xy)}",
           "EDGE_WEIGHT_TYPE : EUC_2D", f"CAPACITY : {cap:g}", "NODE_COORD_SECTION"]
    out += [f"{i + 1} {x!r} {y!r}" for i, (x, y) in enumerate(xy.tolist())]
    out.append("DEMAND_SECTION")
    out += [f"{i + 1} {int(d)}" for i, d in enumerate(dem.tolist())]
    out += ["DEPOT_SECTION", "1", "-1", "EOF"]
    Path(path).write_text("\n".join(out) + "\n")


def route_cost(inst: Instance, route: list[int], rounded: bool) -> float:
    idx = np.fromiter([0, *route, 0], dtype=np.int64)
    seg = np.diff(inst.xy[idx], axis=0)
    d = np.hypot(seg[:, 0], seg[:, 1])
    if rounded:
        d = np.floor(d + 0.5)
    return float(d.sum())


def check_solution(inst: Instance, routes: list[list[int]], rounded: bool) -> tuple[float, bool, str]:
    """Returns (cost, feasible, reason). Recomputes everything from coordinates."""
    seen = np.zeros(inst.n + 1, dtype=np.int64)
    cost, msg = 0.0, ""
    for r in routes:
        if not r:
            continue
        if min(r) < 1 or max(r) > inst.n:
            return math.inf, False, "customer id out of range"
        np.add.at(seen, r, 1)
        load = inst.dem[r].sum()
        if load > inst.cap + 1e-9:
            msg = msg or f"route over capacity ({load:g} > {inst.cap:g})"
        cost += route_cost(inst, r, rounded)
    if (seen[1:] != 1).any():
        msg = msg or f"{int((seen[1:] == 0).sum())} missing, {int((seen[1:] > 1).sum())} repeated"
    return cost, msg == "", msg


def read_cvrplib_sol(path: str | Path) -> tuple[list[list[int]], float | None]:
    routes, cost = [], None
    for line in Path(path).read_text().splitlines():
        if line.startswith("Route"):
            routes.append([int(t) for t in line.split(":", 1)[1].split()])
        elif line.startswith("Cost"):
            cost = float(line.split()[1])
    return routes, cost


def read_cwsol(path: str | Path) -> dict[str, tuple[float, list[list[int]]]]:
    """SAVANT --sol file -> {instance name (no extension): (reported cost, routes)}."""
    out: dict[str, tuple[float, list[list[int]]]] = {}
    lines = [l for l in Path(path).read_text().splitlines() if l and not l.startswith("#")]
    i = 0
    while i < len(lines):
        tok = lines[i].split()
        assert tok[0] == "inst", f"{path}: unexpected line {lines[i]!r}"
        name, cost, nr = Path(tok[2]).stem, float(tok[5]), int(tok[6])
        routes = [[int(t) for t in lines[i + 1 + k].split()] for k in range(nr)]
        out[name] = (cost, routes)
        i += 1 + nr
    return out


def load_bks() -> dict[str, float]:
    bks: dict[str, float] = {}
    for f in (DATA / "cvrplib").glob("*_bks.csv"):
        with f.open() as fh:
            for row in csv.DictReader(fh):
                bks[row["name"]] = float(row["bks_cost"])
    return bks


def is_rounded(path: str | Path) -> bool:
    """X, XL, XML (and anything made with the XML generator) use integer
    coordinates and nint distances; the uniform NeuOpt/NDS/gen sets do not."""
    return Path(path).stem.startswith(("X-", "XL-", "XML"))


def physical_cpus() -> list[int]:
    """One logical CPU id per physical core, so timed jobs never share a core.
    Restricted to the CPUs this process may run on: under Slurm that is the
    job's allocation (cgroup), not the whole node."""
    import os
    import subprocess

    allowed = os.sched_getaffinity(0)
    first: dict[str, int] = {}
    for line in subprocess.run(["lscpu", "-p=CPU,CORE,SOCKET"], capture_output=True,
                               text=True, check=True).stdout.splitlines():
        if line.startswith("#"):
            continue
        cpu, core, sock = line.split(",")
        if int(cpu) in allowed:
            first.setdefault(f"{sock}:{core}", int(cpu))
    return sorted(first.values())


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, in the order of `p`. (The Phase 2
    analyses each carry their own copy; this one is for new code.)"""
    order = np.argsort(p)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def round_sig(x: float, digits: int) -> float:
    """x rounded to `digits` significant decimal digits."""
    if x == 0 or not math.isfinite(x):
        return x
    e = math.floor(math.log10(abs(x))) - (digits - 1)
    return round(x / 10 ** e) * 10 ** e


def reported_cost_ok(cost: float, reported: float, rounded: bool) -> bool:
    """Does a solver's own cost agree with the one we recomputed?

    This guards against misreading a solution file; feasibility and the cost we
    keep are always established by recomputation, never by this value.

    Some solvers print the cost with a fixed number of significant digits:
    HGS-CVRP's solution writer uses six, so on XL, where costs exceed 10^6, the
    last digits are rounded and the difference alone can exceed the tolerance
    even though the solution is perfect. A reported value that is exactly the
    recomputed cost rounded to 6..9 significant digits is therefore accepted.
    (Observed 2026-09-21: it had rejected 51 valid HGS-CVRP runs on XL.)
    """
    tol = (1e-6 if rounded else 1e-4) * max(1.0, abs(cost))
    if abs(cost - reported) <= tol:
        return True
    return any(abs(round_sig(cost, d) - reported) <= 1e-9 for d in range(6, 10))
