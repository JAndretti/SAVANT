"""Run a baseline CVRP solver under a wall-clock budget, same protocol as SAVANT.

  --solver hgs      HGS-CVRP (Vidal 2022)                -t CPU seconds, -seed
  --solver filo     FILO (Accorsi & Vigo 2021)            --time seconds (integer), --seed
  --solver filo2    FILO2 (Accorsi & Vigo 2024)           --optimization-seconds (integer), --seed
  --solver ails     AILS-II (Maximo et al. 2024)          -limit seconds; no seed option
  --solver lkh      LKH-3.0.14 (Helsgaun 2017)            TOTAL_TIME_LIMIT, SEED
  --solver ortools  OR-Tools GLS                          seed permutes the customers
  --solver nds      NDS (Hottung et al. 2025), GPU        wall seconds; seeds torch/numpy (bench.nds_cvrp)

  uv run python -m bench.run_baselines --solver hgs --set data/cvrplib/X --time-hgs 1 \
      --seeds 10 --tag tmax --out results/baselines_X.csv

Budgets are what each solver accepts; the CSV also records the measured wall
time of the whole process (start-up and preprocessing included). Every runner
also returns the convergence trace (bench.trace); FILO, FILO2 and LKH-3 print
it through a small output-only patch ("SAVANT benchmark patch" in their sources).
"""

from __future__ import annotations

import argparse
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path

from bench import trace as tr
from bench.common import ROOT, is_rounded, physical_cpus, read_cvrplib_sol, read_vrp
from bench.pool import (RunOutput, Task, add_common_args, add_time_args, execute,
                        make_tasks, seconds_for)

EXT = ROOT / "external_code"
# FILO, FILO2 and LKH-3 compute integer (rounded) arc costs: meaningless on the
# real-valued uniform sets, so they are compared on CVRPLIB sets only
ROUNDED_ONLY = ("filo", "filo2", "lkh")
HGS = EXT / "HGS-CVRP-main/build/hgs"
FILO = EXT / "filo-master/build_tl/filo"
FILO2 = EXT / "filo2-main/build_tl/filo2"
JAVA = EXT / "jdk21/bin/java"
AILS = EXT / "AILS-CVRP-main/AILSII.jar"
LKH = EXT / "LKH-3.0.14/LKH"
NDS_PY = EXT / "nds_env/.venv/bin/python"     # uv sync --project external_code/nds_env


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    if p.returncode:
        raise RuntimeError(f"{shlex.join(map(str, cmd))}\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    return p


def pin(cpu: int) -> list[str]:
    return ["taskset", "-c", str(cpu)]


def hgs(t: Task, cpu: int, tmp: Path) -> RunOutput:
    run([*pin(cpu), HGS, t.vrp, tmp / "s.sol", "-t", f"{t.budget_s:.3f}", "-seed", t.seed,
         "-round", int(is_rounded(t.vrp)), "-log", 0])
    routes, cost = read_cvrplib_sol(tmp / "s.sol")
    pg = tmp / "s.sol.PG.csv"                      # written by HGS on every new best
    return RunOutput(routes, cost, trace=tr.parse_hgs(pg.read_text()) if pg.exists() else [])


def _filo(binary: Path, token: str):
    def runner(t: Task, cpu: int, tmp: Path) -> RunOutput:
        secs = max(1, round(t.budget_s))           # both FILOs parse an integer
        p = run([*pin(cpu), binary, t.vrp, "--outpath", tmp, "--seed", t.seed, token, secs])
        sol = next(tmp.glob("*.vrp.sol"))
        routes, cost = read_cvrplib_sol(sol)
        return RunOutput(routes, cost, trace=tr.parse_prefixed(p.stdout))
    return runner


def ails(t: Task, cpu: int, tmp: Path) -> RunOutput:
    p = run([*pin(cpu), JAVA, "-Xmx2g", "-XX:ActiveProcessorCount=1", "-jar", AILS,
             "-file", t.vrp, "-rounded", str(is_rounded(t.vrp)).lower(),
             "-limit", f"{t.budget_s:.3f}", "-stoppingCriterion", "Time"])
    text = p.stdout.split("=== BEST SOLUTION ===", 1)[1]
    (tmp / "s.sol").write_text(text)
    routes, cost = read_cvrplib_sol(tmp / "s.sol")
    return RunOutput(routes, cost, trace=tr.parse_ails(p.stdout))


def lkh(t: Task, cpu: int, tmp: Path) -> RunOutput:
    if not is_rounded(t.vrp):
        raise ValueError("LKH-3 runner: real-valued instances are not supported (EUC_2D rounds)")
    inst = read_vrp(t.vrp)
    kmin = math.ceil(inst.dem.sum() / inst.cap)
    vehicles = kmin + max(2, math.ceil(0.1 * kmin))   # slack; empty routes allowed below
    lines = [f"NAME : {inst.name}", "TYPE : CVRP", f"DIMENSION : {inst.n + 1}",
             "EDGE_WEIGHT_TYPE : EUC_2D", f"CAPACITY : {inst.cap:g}", f"VEHICLES : {vehicles}",
             "NODE_COORD_SECTION"]
    lines += [f"{i + 1} {x:.10g} {y:.10g}" for i, (x, y) in enumerate(inst.xy.tolist())]
    lines += ["DEMAND_SECTION"] + [f"{i + 1} {int(d)}" for i, d in enumerate(inst.dem.tolist())]
    lines += ["DEPOT_SECTION", "1", "-1", "EOF"]
    (tmp / "p.vrp").write_text("\n".join(lines) + "\n")
    (tmp / "p.par").write_text("\n".join([
        "SPECIAL", f"PROBLEM_FILE = {tmp / 'p.vrp'}", "MTSP_MIN_SIZE = 0",
        "RUNS = 1000000",                    # runs + GA recombination until the time limit
        f"TIME_LIMIT = {t.budget_s:.3f}", f"TOTAL_TIME_LIMIT = {t.budget_s:.3f}",
        f"SEED = {t.seed}", "TRACE_LEVEL = 0", f"MTSP_SOLUTION_FILE = {tmp / 'sol.txt'}"]) + "\n")
    p = run([*pin(cpu), LKH, tmp / "p.par"])
    routes = []
    for line in (tmp / "sol.txt").read_text().splitlines()[2:]:
        ids = [int(x) for x in line.split("(#")[0].split()]
        r = [i - 1 for i in ids if i != 1]           # node id -> customer index
        if r:
            routes.append(r)
    return RunOutput(routes, trace=tr.parse_prefixed(p.stdout))


def ortools(t: Task, cpu: int, tmp: Path) -> RunOutput:
    p = run([*pin(cpu), sys.executable, "-m", "bench.ortools_cvrp", t.vrp, tmp / "s.sol",
             f"{t.budget_s:.3f}", "--round", int(is_rounded(t.vrp)), "--seed", t.seed], cwd=ROOT)
    routes, cost = read_cvrplib_sol(tmp / "s.sol")
    return RunOutput(routes, cost, trace=tr.parse_prefixed(p.stdout))


def gpu_ids() -> list[str]:
    """GPUs this process may use: Slurm's CUDA_VISIBLE_DEVICES, else every GPU of the host."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is not None:
        return [g for g in vis.split(",") if g]
    p = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    return [str(k) for k, line in enumerate(p.stdout.splitlines()) if line.startswith("GPU")]


def nds(t: Task, cpu: int, tmp: Path) -> RunOutput:
    # the pool hands out cpus[:jobs] and --jobs <= #GPUs is enforced in main(), so the
    # position of this core in the list is a job slot that owns one GPU exclusively
    gpu = gpu_ids()[physical_cpus().index(cpu)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "OMP_NUM_THREADS": "1"}
    p = run([*pin(cpu), NDS_PY, "-W", "ignore::DeprecationWarning", "-m", "bench.nds_cvrp",
             t.vrp, tmp / "s.sol", f"{t.budget_s:.3f}", "--seed", t.seed], cwd=ROOT, env=env)
    routes, cost = read_cvrplib_sol(tmp / "s.sol")
    return RunOutput(routes, cost, trace=tr.parse_prefixed(p.stdout))


RUNNERS = {"hgs": hgs, "filo": _filo(FILO, "--time"), "filo2": _filo(FILO2, "--optimization-seconds"),
           "ails": ails, "lkh": lkh, "ortools": ortools, "nds": nds}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solver", required=True, choices=sorted(RUNNERS))
    add_common_args(ap)
    add_time_args(ap.add_mutually_exclusive_group(required=True))
    a = ap.parse_args()
    tasks = make_tasks(a, lambda v: seconds_for(a, v), lambda v: f"time={seconds_for(a, v):.3f}")
    real = [t.vrp.name for t in tasks if not is_rounded(t.vrp)]
    if a.solver in ROUNDED_ONLY and real:
        ap.error(f"{a.solver} rounds arc costs: not run on real-valued instances ({real[0]}, ...)")
    if a.solver == "nds":
        ngpu = len(gpu_ids())
        if not ngpu or not 0 < a.jobs <= ngpu:
            ap.error(f"nds runs one job per GPU: pass --jobs between 1 and {ngpu} (GPUs visible)")
        if not NDS_PY.exists():
            ap.error(f"{NDS_PY} missing: uv sync --project external_code/nds_env")
    execute(tasks, RUNNERS[a.solver], solver=a.solver, tag=a.tag, extra="", out=a.out, jobs=a.jobs)


if __name__ == "__main__":
    main()
