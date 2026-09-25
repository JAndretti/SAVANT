"""Solve one CVRP instance with NDS (Hottung et al. 2025) under a wall-clock limit.

    uv run --project external_code/nds_env python -m bench.nds_cvrp <vrp> <out.sol> <seconds> \
        [--seed S] [--device cuda|cpu]

Runs the official implementation in external_code/NDS (commit ca37420) with its
pretrained CVRP model and evaluation config for the instance size
(configs/eval/cvrp_{n}.yaml). Nothing in the search is changed; this file only
  - feeds one .vrp instance instead of a .pkl set;
  - seeds Python, NumPy and PyTorch (NDS has no seed parameter);
  - lifts the config's iteration cap (nb_iterations = 100,000) so that the
    wall-clock limit is the only stopping rule: at T_max = 2.4n a fast GPU could
    reach the cap first, and NDS then aborts on an assertion (at n = 100 an
    RTX 5070 does ~44 iterations/s, i.e. ~10^4 in T_max = 240 s -- below the cap);
  - prints `TRACE <s> <cost>` on each new best (seconds since NDS's own start_time,
    i.e. from the construction of its initial solution, model loading excluded).

`Search._solve_one_instance` is copied from external_code/NDS/src/search_sa.py with
the trace lines added (marked "SAVANT benchmark"): the equivalent of the output-only
patches applied to FILO, FILO2 and LKH-3, without editing the third-party tree.

Rounded instances (CVRPLIB, integer coordinates, e.g. XML100) are translated and
scaled to the unit square, the domain NDS was trained on. NDS then optimises real
Euclidean distances on the scaled coordinates; the trace reports the *rounded* cost
of each new incumbent on the original coordinates, the measure of the set, so it
can end on the checked final cost. Its incumbent is still chosen by NDS's own
(real-valued) cost, so that trace need not be monotone.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import yaml

from bench.common import ROOT, is_rounded, read_vrp, route_cost

NDS = ROOT / "external_code" / "NDS"
MAX_ITERATIONS = 10**12            # time limit only (see docstring)


def load_config(n: int) -> dict:
    path = NDS / "configs" / "eval" / f"cvrp_{n}.yaml"
    if not path.exists():
        sys.exit(f"NDS has no CVRP model for n = {n} (only 100, 500, 1000, 2000)")
    return yaml.safe_load(path.read_text())


def make_search_class():
    sys.path.insert(0, str(NDS))
    os.chdir(NDS)       # cppimport builds its C++ module with setuptools in the cwd
    sys.modules.setdefault("wandb", types.ModuleType("wandb"))   # imported by src.trainer only
    from src.search_sa import Search

    class TracedSearch(Search):
        trace_cost = None          # set by main: solution -> cost printed in the trace

        def _solve_one_instance(self, instance_idx):
            # copied from NDS src/search_sa.py (ca37420); only the TRACE print is added
            aug_factor = self.tester_params["aug_factor"]
            max_iterations = self.tester_params["nb_iterations"]
            rollout_size = self.tester_params["rollout_size"]
            sa_config = self._init_simulated_annealing()
            start_time = time.time()
            self.env.init_instances(1, rollout_size, self.device, aug_factor)
            incumbent_cost = np.inf
            incumbent_solution = None
            iteration = 0
            while iteration < max_iterations:
                new_solutions = self._perform_sa_iteration(
                    aug_factor, rollout_size, sa_config["T"]
                )
                improved = False
                for sol in new_solutions:
                    if sol.totalCosts < incumbent_cost:
                        incumbent_cost = sol.totalCosts
                        incumbent_solution = sol
                        improved = True
                if improved:                                   # SAVANT benchmark
                    print(f"TRACE {time.time() - start_time:.6f} "
                          f"{self.trace_cost(incumbent_solution):.10g}")
                if aug_factor > 1:
                    self._synchronize_augmented_solutions(
                        new_solutions, sa_config["T"], sa_config["delta"]
                    )
                for idx, sol in enumerate(new_solutions):
                    self.env.instanceSet.set_solution(idx, sol)
                sa_config["T"] = self._update_temperature(
                    sa_config, iteration, start_time, max_iterations
                )
                iteration += 1
                if (
                    sa_config["runtime_limited"]
                    and (time.time() - start_time) > sa_config["max_runtime"]
                ):
                    break
            runtime = time.time() - start_time
            if sa_config["runtime_limited"]:
                assert (
                    runtime > sa_config["max_runtime"]
                ), "Runtime limit was set, but search terminated based on iteration count"
            return {
                "cost": incumbent_cost,
                "runtime": runtime,
                "nb_iterations": iteration,
                "solution": incumbent_solution,
            }

    return TracedSearch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("vrp", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("seconds", type=float)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    a = ap.parse_args()
    a.vrp, a.out = a.vrp.resolve(), a.out.resolve()     # the NDS import changes directory

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)                  # also seeds every CUDA device
    torch.set_num_threads(1)                   # one CPU core, as in Hottung et al.

    inst = read_vrp(a.vrp)
    rounded = is_rounded(a.vrp)
    xy = inst.xy
    if rounded:                                # CVRPLIB coordinates -> unit square
        xy = xy - xy.min()
        xy = xy / xy.max()
    cfg = load_config(inst.n)
    env_params, tp = cfg["env_params"], cfg["tester_params"]
    tp["use_cuda"] = a.device == "cuda"
    tp["cuda_device_num"] = 0                  # the runner selects the GPU via CUDA_VISIBLE_DEVICES
    tp["max_runtime"] = a.seconds
    tp["nb_iterations"] = MAX_ITERATIONS
    tp["test_data_load"]["enable"] = False
    for m in tp["model_load"]:
        m["path"] = str(NDS / m["path"])

    Search = make_search_class()

    def trace_cost(sol) -> float:
        if not rounded:
            return sol.totalCosts
        return sum(route_cost(inst, t, True) for t in sol.getTourList() if t)
    Search.trace_cost = staticmethod(trace_cost)

    search = Search(env_params=env_params, tester_params=tp)
    prob = search.env.problem                  # as ProblemCVRP.load_problem_dataset_pkl
    prob.use_saved_problems = True
    prob.dataset_depot_xy = torch.tensor(xy[:1].tolist())[None]          # (1, 1, 2)
    prob.dataset_node_xy = torch.tensor(xy[1:].tolist())[None]           # (1, n, 2)
    prob.dataset_node_demand = torch.tensor([[int(d) for d in inst.dem[1:]]])
    prob.dataset_capacity = torch.tensor([[int(inst.cap)]])
    prob.saved_index = 0
    prob.nb_instances = 1

    res = search._solve_one_instance(0)
    tours = [list(t) for t in res["solution"].getTourList() if len(t)]
    with a.out.open("w") as fh:
        for k, r in enumerate(tours, 1):
            fh.write(f"Route #{k}: {' '.join(map(str, r))}\n")
        if not rounded:                        # NDS's own cost is meaningful only unscaled
            fh.write(f"Cost {res['cost']:.10g}\n")
    print(f"iterations {res['nb_iterations']} runtime {res['runtime']:.3f}")


if __name__ == "__main__":
    main()
