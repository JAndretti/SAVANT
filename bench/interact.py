"""Phase 2, step 2: full factorial over the factors that survived screening.

Screening varies one factor at a time and cannot see interactions (e.g. more
candidate neighbours may only pay off once restarts are on). This runs every
combination of the selected factors at their "on" level, on the same tuning
instances, budgets and seeds as bench.sweep.

    uv run python -m bench.interact --budget 1e6 --list
    uv run python -m bench.interact --budget 1e6                # all 2^k cells
    uv run python -m bench.interact --budget 1e6 --factors restarts,kick,knn

Cells are named by the factors switched on ("base" = the default configuration),
so results merge with the screening CSVs.
"""

from __future__ import annotations

import argparse
import itertools
import shlex

from bench.common import ROOT
from bench.pool import execute
from bench.run_savant import STOCK, restarts_of, savant_runner
from bench.sweep import screen_tasks
from bench.time_match import tasks_for as time_tasks

# factor -> options applied when the factor is "on"; "off" is the stock configuration
# (bench.run_savant.STOCK, prepended)
FACTORS: dict[str, str] = {
    "restarts": "--restarts 4",
    "kick": "--kick 50",
    "knn": "--sa-knn 40",
    # "crit": the regret definition only matters when the tournaments are on;
    # the configuration used for the comparisons selects uniformly (Sec. 4.4)
    "kickmax": "--kick-max 20",
}


def cell_name(on: tuple[str, ...]) -> str:
    return "base" if not on else "x_" + "+".join(on)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    b = ap.add_mutually_exclusive_group(required=False)
    b.add_argument("--budget", type=float, help="annealing steps per customer (total over restarts)")
    b.add_argument("--f", type=float,
                   help="wall budget T = f * n * 2.4 s per instance (cw --sa-wall). Retained factors "
                        "change the cost of a step, so the final choice is made at equal time")
    ap.add_argument("--factors", default="all")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--every", type=int, default=1, help="keep every K-th tuning instance (--f only)")
    ap.add_argument("--out", help="default: results/phase2/interact_{B<budget>|f<f>}.csv")
    ap.add_argument("--binary", default="", help="default: cw_repro for steps, cw for wall budgets")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    names = list(FACTORS) if a.factors == "all" else a.factors.split(",")
    unknown = set(names) - set(FACTORS)
    if unknown:
        ap.error(f"unknown factors: {sorted(unknown)}")
    cells = [c for k in range(len(names) + 1) for c in itertools.combinations(names, k)]
    if a.list:
        for c in cells:
            print(f"{cell_name(c):40s} {' '.join(FACTORS[f] for f in c)}")
        print(f"{len(cells)} cells")
        return
    if a.budget is None and a.f is None:
        ap.error("one of --budget (steps) or --f (wall time) is required")

    timed = a.f is not None
    binary = a.binary or str(ROOT / ("cw" if timed else "cw_repro"))
    out = ROOT / (a.out or (f"results/phase2/interact_f{a.f:g}.csv" if timed
                            else f"results/phase2/interact_B{a.budget:g}.csv"))
    for c in cells:
        extra = " ".join(FACTORS[f] for f in c)
        options = f"{STOCK} {extra}"
        if timed:
            tasks = time_tasks(a.f, a.seeds, a.every)
        else:
            tasks = screen_tasks(a.budget, restarts_of(options), a.seeds)
        execute(tasks, savant_runner(binary, shlex.split(options)), solver="savant",
                tag=cell_name(c), extra=extra, out=out, jobs=a.jobs)


if __name__ == "__main__":
    main()
