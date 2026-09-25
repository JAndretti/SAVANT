"""Time-matched comparison of configurations (Phase 2, control experiment).

The screening fixes the number of annealing steps, which isolates the effect of
a parameter from its effect on the cost of a step. That is the right control
when the two costs are similar, and misleading when they are not: uniform move
selection is ~18% faster per step than the regret tournament, so at equal time
it buys ~22% more steps. This driver gives every configuration the same wall
budget instead, T = f * n * 2.4 seconds (the protocol of Vidal 2022 scaled by
f), enforced by cw --sa-wall, and uses the fast binary.

    uv run python -m bench.time_match --f 0.002            # ~5 s at n=1000
    uv run python -m bench.time_match --f 0.02             # ~48 s at n=1000
"""

from __future__ import annotations

import argparse
import shlex

from bench.common import DATA, ROOT
from bench.pool import Task, dimension, execute
from bench.run_savant import STOCK, savant_runner
from bench.sweep import SCREEN

# relative to the stock configuration (bench.run_savant.STOCK, prepended)
CONFIGS: dict[str, str] = {
    "default": "",                              # tournament on both vertices
    "pick_uniform_both": "--pick 1 --pick2 1",  # neither stage
    "pick_u_only": "--pick2 1",                 # tournament on the first vertex only
    "pick_v_only": "--pick 1",                  # tournament on the second vertex only
}


def tasks_for(f: float, seeds: int, every: int = 1) -> list[Task]:
    """every > 1 keeps a fraction of the tuning subset, for budgets too long to
    run on all 181 instances (the comparison protocol uses f = 1)."""
    tasks = []
    for sub, stride in SCREEN:
        for v in sorted((DATA / "tuning" / sub).glob("*.vrp"))[::stride * every]:
            t = f * dimension(v) * 2.4
            for s in range(1, seeds + 1):
                tasks.append(Task(v, s, t, f"T={f:g}*n*2.4s"))
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--f", type=float, required=True, help="wall budget T = f * n * 2.4 seconds")
    ap.add_argument("--configs", default="all")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--every", type=int, default=1, help="keep every K-th tuning instance")
    ap.add_argument("--out", help="default: results/phase2/timematch_f<f>.csv")
    ap.add_argument("--binary", default=str(ROOT / "cw"), help="timed runs use the fast build")
    ap.add_argument("--jobs", type=int, default=0)
    a = ap.parse_args()

    names = list(CONFIGS) if a.configs == "all" else a.configs.split(",")
    out = ROOT / (a.out or f"results/phase2/timematch_f{a.f:g}.csv")
    for name in names:
        extra = CONFIGS[name]
        execute(tasks_for(a.f, a.seeds, a.every), savant_runner(a.binary, shlex.split(f"{STOCK} {extra}")),
                solver="savant", tag=name, extra=extra, out=out, jobs=a.jobs)


if __name__ == "__main__":
    main()
