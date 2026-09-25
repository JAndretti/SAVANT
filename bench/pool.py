"""Shared job scheduling for every solver runner.

One single-threaded process per (instance, seed), pinned to its own physical
core; every solution is re-checked from the coordinates (bench.common) rather
than trusting the solver; results are appended to a resumable CSV. Runs that
return a convergence trace (bench.trace) also append it, one JSON line per
run, to <out>.trace.jsonl.
"""

from __future__ import annotations

import argparse
import csv
import json
import queue
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from bench import trace as tr
from bench.common import (check_solution, is_rounded, load_bks, physical_cpus,
                          read_vrp, reported_cost_ok)

FIELDS = ["solver", "tag", "extra", "instance", "n", "seed", "budget", "steps",
          "cost", "cost_reported", "feasible", "routes", "time_s", "bks", "gap_pct", "msg"]


@dataclass
class Task:
    vrp: Path
    seed: int
    budget_s: float | None      # wall-clock budget, None for step budgets
    budget: str                 # label written to the CSV
    steps: str | None = None    # step budget for SAVANT ("N" = cw's size rule)


@dataclass
class RunOutput:
    routes: list[list[int]]     # customers 1..n, depot excluded
    cost_reported: float | None = None
    steps: str = ""
    trace: tr.Trace | None = None   # (seconds, cost) on each new best, solver-reported


Runner = Callable[[Task, int, Path], RunOutput]    # (task, cpu id, private tmp dir)


def dimension(vrp: Path) -> int:
    """Number of customers, read from the header only."""
    with vrp.open() as fh:
        for line in fh:
            if line.startswith("DIMENSION"):
                return int(line.split(":")[1]) - 1
    raise ValueError(f"{vrp}: no DIMENSION")


def add_common_args(ap: argparse.ArgumentParser, out_required: bool = True) -> None:
    ap.add_argument("--set", required=True, type=Path, help="directory of .vrp files")
    ap.add_argument("--glob", default="*.vrp")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1, help="take every k-th instance")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, required=out_required)
    ap.add_argument("--jobs", type=int, default=0, help="default: one per physical core")
    ap.add_argument("--shard", default="0/1",
                    help="K/N: run only the K-th of N disjoint slices of the runs (one per cluster node)")


def add_time_args(group: argparse._MutuallyExclusiveGroup) -> None:
    group.add_argument("--time", type=float, help="wall seconds per instance")
    group.add_argument("--time-hgs", type=float,
                       help="T_max = F * n * 240/100 s (Vidal 2022 protocol; F=1 is the paper's)")


def seconds_for(a: argparse.Namespace, vrp: Path) -> float:
    return a.time if a.time is not None else a.time_hgs * dimension(vrp) * 240 / 100


def make_tasks(a: argparse.Namespace, budget_s: Callable[[Path], float | None],
               label: Callable[[Path], str]) -> list[Task]:
    files = sorted(a.set.glob(a.glob))[::a.stride]
    if a.limit > 0:
        files = files[:a.limit]
    tasks = [Task(v, s, budget_s(v), label(v))
             for v in files for s in range(a.seed0, a.seed0 + a.seeds)]
    return shard(tasks, a.shard)


def shard(tasks: list[Task], spec: str) -> list[Task]:
    """K/N slicing. Tasks are dealt round-robin after sorting by decreasing
    budget, so every shard gets a similar total run time."""
    k, n = (int(x) for x in spec.split("/"))
    if not 0 <= k < n:
        raise ValueError(f"--shard {spec}: expected K/N with 0 <= K < N")
    order = sorted(range(len(tasks)), key=lambda i: (-(tasks[i].budget_s or 0.0), i))
    return [tasks[i] for j, i in enumerate(order) if j % n == k]


def execute(tasks: list[Task], runner: Runner, *, solver: str, tag: str, extra: str,
            out: Path, jobs: int = 0) -> None:
    bks = load_bks()
    done: set[tuple[str, str, str, int]] = set()
    if out.exists() and out.stat().st_size:
        with out.open() as fh:
            done = {(r["solver"], r["tag"], r["instance"], int(r["seed"])) for r in csv.DictReader(fh)}
    todo = [t for t in tasks if (solver, tag, t.vrp.stem, t.seed) not in done]
    cpus = physical_cpus()
    jobs = min(jobs or len(cpus), len(cpus))
    print(f"{solver}/{tag}: {len(todo)} runs ({len(tasks) - len(todo)} already done), {jobs} pinned jobs")
    if not todo:
        return

    free: queue.Queue[int] = queue.Queue()
    for c in cpus[:jobs]:
        free.put(c)

    def one(t: Task) -> dict:
        cpu = free.get()
        try:
            with tempfile.TemporaryDirectory(prefix=f"{solver}_") as tmp:
                t0 = time.perf_counter()
                res = runner(t, cpu, Path(tmp))
                wall = time.perf_counter() - t0
        finally:
            free.put(cpu)
        rounded = is_rounded(t.vrp)
        inst = read_vrp(t.vrp)
        cost, ok, msg = check_solution(inst, res.routes, rounded)
        if ok and res.cost_reported is not None \
                and not reported_cost_ok(cost, res.cost_reported, rounded):
            ok, msg = False, f"reported {res.cost_reported} != recomputed {cost}"
        ref = bks.get(inst.name)
        rec = None
        if res.trace is not None:
            t_ok, t_msg = tr.check(res.trace, cost, rounded)
            rec = {"solver": solver, "tag": tag, "instance": inst.name, "seed": t.seed,
                   "budget_s": t.budget_s, "time_s": round(wall, 4), "cost": cost,
                   "trace_ok": t_ok, "trace_msg": t_msg, "trace": res.trace}
        return {"solver": solver, "tag": tag, "extra": extra, "instance": inst.name, "n": inst.n,
                "seed": t.seed, "budget": t.budget, "steps": res.steps, "cost": cost,
                "cost_reported": res.cost_reported if res.cost_reported is not None else "",
                "feasible": int(ok), "routes": sum(1 for r in res.routes if r),
                "time_s": round(wall, 4), "bks": ref if ref else "",
                "gap_pct": round(100 * (cost - ref) / ref, 4) if ref and ok else "",
                "msg": msg, "_msg": msg, "_trace": rec}

    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists() or out.stat().st_size == 0
    nbad = nbadtrace = 0
    fields = FIELDS
    if not new_file:                       # a CSV written before a column was added
        with out.open() as fh_head:        # keeps its own header: resuming stays valid
            fields = next(csv.reader(fh_head), FIELDS) or FIELDS
    with out.open("a", newline="") as fh, ThreadPoolExecutor(jobs) as pool:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new_file:
            w.writeheader()
        futs = [pool.submit(one, t) for t in todo]
        for k, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if not r["feasible"]:
                nbad += 1
                print(f"INVALID {r['instance']} seed={r['seed']}: {r['_msg']}")
            if r["_trace"] is not None:          # before the CSV row: a resumed run redoes both
                if not r["_trace"]["trace_ok"]:
                    nbadtrace += 1
                    print(f"TRACE {r['instance']} seed={r['seed']}: {r['_trace']['trace_msg']}")
                with tr.sidecar(out).open("a") as fh_tr:   # only created for timed runs
                    fh_tr.write(json.dumps(r["_trace"]) + "\n")
            w.writerow(r)
            fh.flush()
            if k % max(1, len(futs) // 20) == 0 or k == len(futs):
                print(f"  {k}/{len(futs)}")
    print("done" + (f", {nbad} INVALID solutions" if nbad else ", all solutions valid")
          + (f", {nbadtrace} unusable traces" if nbadtrace else ""))
