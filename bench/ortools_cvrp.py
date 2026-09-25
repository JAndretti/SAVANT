"""OR-Tools routing solver (guided local search), as configured in Vidal (2022):
PATH_CHEAPEST_ARC first solution + GUIDED_LOCAL_SEARCH until the time limit.

    uv run python -m bench.ortools_cvrp INSTANCE.vrp OUT.sol TIME_S [--round 0|1] [--seed S]

Writes a CVRPLIB-style .sol (customers numbered 1..n, depot excluded). OR-Tools
is deterministic; as in Vidal (2022) the seed permutes the customer order.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from bench.common import check_solution, read_vrp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("out")
    ap.add_argument("time", type=float)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t_start = time.perf_counter()                   # the trace clock: model building counts

    inst = read_vrp(a.instance)
    n = inst.n
    perm = np.arange(1, n + 1)
    if a.seed:
        np.random.default_rng(a.seed).shuffle(perm)
    order = np.concatenate([[0], perm])            # model node k -> customer order[k]
    xy = inst.xy[order]
    # integer arc costs: OR-Tools needs ints; scale real distances by 1e3
    scale = 1 if a.round else 1000
    d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    dist = (np.floor(d + 0.5) if a.round else np.rint(d * scale)).astype(np.int64)
    dem = inst.dem[order].astype(np.int64)
    fleet = n  # upper bound, unused vehicles cost nothing

    mgr = pywrapcp.RoutingIndexManager(n + 1, fleet, 0)
    routing = pywrapcp.RoutingModel(mgr)
    cb = routing.RegisterTransitCallback(
        lambda i, j: int(dist[mgr.IndexToNode(i), mgr.IndexToNode(j)]))
    routing.SetArcCostEvaluatorOfAllVehicles(cb)
    dcb = routing.RegisterUnaryTransitCallback(lambda i: int(dem[mgr.IndexToNode(i)]))
    routing.AddDimensionWithVehicleCapacity(dcb, 0, [int(inst.cap)] * fleet, True, "load")

    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.FromMilliseconds(int(a.time * 1000))

    best = [float("inf")]

    def trace() -> None:                            # every solution the search accepts
        c = routing.CostVar().Value() / scale
        if c < best[0]:
            best[0] = c
            print(f"TRACE {time.perf_counter() - t_start:.6f} {c:.10g}", flush=False)

    routing.AddAtSolutionCallback(trace)
    sol = routing.SolveWithParameters(p)
    if sol is None:
        raise SystemExit("no solution found")

    routes = []
    for v in range(fleet):
        i, r = routing.Start(v), []
        i = sol.Value(routing.NextVar(i))
        while not routing.IsEnd(i):
            r.append(int(order[mgr.IndexToNode(i)]))
            i = sol.Value(routing.NextVar(i))
        if r:
            routes.append(r)
    cost, ok, msg = check_solution(inst, routes, bool(a.round))
    with open(a.out, "w") as fh:
        for k, r in enumerate(routes, 1):
            fh.write(f"Route #{k}: {' '.join(map(str, r))}\n")
        fh.write(f"Cost {cost:g}\n")
    print(f"cost {cost} feasible {ok} {msg}")


if __name__ == "__main__":
    main()
