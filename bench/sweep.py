"""Phase 2, step 1: one-factor-at-a-time screening around the current defaults.

Every configuration changes ONE aspect of the stock solver (cw.c defaults, as
shipped on 2026-09-15; bench.run_savant.STOCK is prepended since the defaults
changed on 2026-09-17, so the runs are reproduced exactly). Budgets are in annealing steps per customer, so that
instance sizes are funded comparably, and at a fixed *total* budget: a
configuration with R restarts gets budget/R steps per restart.

Instances: a stratified subset of the tuning set (data/tuning, disjoint from
every test set). Binary: cw_repro (bit-reproducible).

    uv run python -m bench.sweep --budget 1e4              # all configurations
    uv run python -m bench.sweep --budget 1e5 --configs default,ops_no_swapstar
    uv run python -m bench.sweep --list
"""

from __future__ import annotations

import argparse
import shlex

from bench.common import DATA, ROOT
from bench.pool import Task, dimension, execute
from bench.run_savant import STOCK, restarts_of, savant_runner

# name -> extra cw options, relative to the stock configuration: --init cw --lambda 1
# --mu 0 --t-accept 0.001 --t-decades 1 --ops 1,1,1,0,1,0.05 --or-max 3 --sa-knn 20
# --pick 2 --pick-crit lb --pick2 2 --vrank 1 --reloc-side coin --kick 100
# --kick-max 10 --split off --restarts 1 --dlb 0 --empty-p 0 --t0-trim 0
# (--reheat, studied and dropped, was removed from cw on 2026-09-17)
CONFIGS: dict[str, str] = {
    "default": "",
    # --- construction
    "init_random": "--init random",
    "lambda_0.6": "--lambda 0.6",
    "lambda_1.4": "--lambda 1.4",
    "mu_0.3": "--mu 0.3",
    "cw_2opt": "--2opt --2opt-knn",
    # --- temperature schedule
    "taccept_1e-4": "--t-accept 0.0001",
    "taccept_1e-2": "--t-accept 0.01",
    "taccept_1e-1": "--t-accept 0.1",
    "decades_0.5": "--t-decades 0.5",
    "decades_2": "--t-decades 2",
    "decades_3": "--t-decades 3",
    "t0trim_0.1": "--t0-trim 0.1",
    # --reheat is excluded (2026-09-15): its trigger (best not improved for K*n
    # steps) fires during normal high-temperature annealing and the reheats
    # compound, pinning T near T0 for the whole run. A flawed design, not a
    # parameter value; the 1e4/1e5 runs stay on disk but are not analysed.
    # --- operator mix (r,s,t,o,x,e = relocate, swap, 2-opt, or-opt, swap*, open)
    "ops_relocate_only": "--ops 1,0,0,0,0,0",
    "ops_no_swap": "--ops 1,0,1,0,1,0.05",
    "ops_no_2opt": "--ops 1,1,0,0,1,0.05",
    "ops_no_swapstar": "--ops 1,1,1,0,0,0.05",
    "ops_swapstar_x2": "--ops 1,1,1,0,2,0.05",
    "ops_oropt": "--ops 1,1,1,1,1,0.05",
    "ops_oropt_max5": "--ops 1,1,1,1,1,0.05 --or-max 5",
    "ops_no_open": "--ops 1,1,1,0,1,0",
    "ops_open_0.2": "--ops 1,1,1,0,1,0.2",
    "empty_p_0.05": "--empty-p 0.05",
    # --- move selection (the "tournament")
    "pick_uniform": "--pick 1",
    # both stages uniform: the reference the regret-guided selection is claimed
    # to beat (each stage alone is not significant at the larger budgets)
    "pick_uniform_both": "--pick 1 --pick2 1",
    "pick_3": "--pick 3",
    "pick_4": "--pick 4",
    "pick_fenwick": "--pick 0",
    "crit_rem": "--pick-crit rem",
    "crit_remnorm": "--pick-crit remnorm",
    "crit_raw": "--pick-crit raw",
    "pick2_1": "--pick2 1",
    "pick2_3": "--pick2 3",
    "vrank_2_knn30": "--vrank 2 --sa-knn 30",
    "saknn_10": "--sa-knn 10",
    "saknn_40": "--sa-knn 40",
    "saknn_uniform": "--sa-knn 0",
    "reloc_long": "--reloc-side long",
    "dlb_5": "--dlb 5",
    # --- ruin & recreate
    "kick_off": "--kick 0",
    "kick_50": "--kick 50",
    "kick_500": "--kick 500",
    "kickmax_5": "--kick-max 5",
    "kickmax_20": "--kick-max 20",
    # --- Split
    "split_end": "--split end",
    "split_both": "--split both",
    "split_every": "--split-every 100000 --split end",
    # --- restarts, at the same total budget
    "restarts_4": "--restarts 4",
    "restarts_4_param": "--restarts 4 --cw-rand both",
    "restarts_4_race": "--restarts 4 --race 0.002",
}

# (tuning subset, stride): ~180 instances, n = 100..2000
SCREEN = [("xml100", 6), ("gen100", 2), ("gen200", 2), ("gen500", 2), ("gen1000", 2),
          ("gen2000", 2), ("xlike", 1)]


def screen_tasks(budget_per_n: float, restarts: int, seeds: int) -> list[Task]:
    tasks = []
    for sub, stride in SCREEN:
        for v in sorted((DATA / "tuning" / sub).glob("*.vrp"))[::stride]:
            steps = max(1000, round(budget_per_n * dimension(v) / restarts))
            for s in range(1, seeds + 1):
                tasks.append(Task(v, s, None, f"B={budget_per_n:g}", steps=str(steps)))
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=float, help="annealing steps per customer (total over restarts)")
    ap.add_argument("--configs", default="all")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", help="default: results/phase2/screen_B<budget>.csv")
    ap.add_argument("--binary", default=str(ROOT / "cw_repro"))
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, v in CONFIGS.items():
            print(f"{k:22s} {v}")
        return
    if a.budget is None:
        ap.error("--budget is required")
    names = list(CONFIGS) if a.configs == "all" else a.configs.split(",")
    out = ROOT / (a.out or f"results/phase2/screen_B{a.budget:g}.csv")
    for name in names:
        extra = CONFIGS[name]
        options = f"{STOCK} {extra}"                 # cw: the last occurrence wins
        tasks = screen_tasks(a.budget, restarts_of(options), a.seeds)
        execute(tasks, savant_runner(a.binary, shlex.split(options)), solver="savant",
                tag=name, extra=extra, out=out, jobs=a.jobs)


if __name__ == "__main__":
    main()
