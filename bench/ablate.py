"""Ablation of the configuration retained for the comparison, at equal running time.

Every cell is the retained configuration with exactly one thing changed, so the
table reads as "what each component is worth". Options are appended after the
base ones and cw lets the last occurrence win, so a cell can override any base
value. The `tournament` cell is not an ablation of a component but the control
that the regret selection of Section 3.3 is still not worth its cost once the
other retained factors are on.

    uv run python -m bench.ablate --list
    uv run python -m bench.ablate --f 0.02 --every 2          # ~1.2 h
    uv run python -m bench.ablate --f 0.1 --every 4           # longer budget
    uv run python -m bench.ablate --f 1 --every 4 --cells final,stock,tournament
        # confirmation at the comparison budget T_max (~116 core-h, ~10 h on 12 cores)
"""

from __future__ import annotations

import argparse
import shlex

from bench.common import ROOT
from bench.pool import execute
from bench.run_savant import STOCK, savant_runner
from bench.time_match import tasks_for as time_tasks

# configuration retained after the screening, the time-matched control and the
# factorial study (Sections 4.2-4.4); cw's compiled defaults since 2026-09-17,
# kept explicit so the cells do not depend on them
BASE = "--pick 1 --pick2 1 --restarts 4 --kick 50 --kick-max 20"

CELLS: dict[str, str] = {
    "final": "",
    # control: regret tournament instead of uniform selection
    "tournament": "--pick 2 --pick2 2",
    # control: the stock configuration the parameter study started from
    "stock": STOCK,
    # retained factors, removed one at a time
    "no_restarts": "--restarts 1",
    "kick_period_100": "--kick 100",
    "kick_max_10": "--kick-max 10",
    # components of the method itself
    "no_kick": "--kick 0",
    "no_swapstar": "--ops 1,1,1,0,0,0.05",
    "no_2opt": "--ops 1,1,0,0,1,0.05",
    "no_candidate_lists": "--sa-knn 0",
    "random_start": "--init random",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--f", type=float, help="wall budget T = f * n * 2.4 s per instance")
    ap.add_argument("--cells", default="all")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--every", type=int, default=1, help="keep every K-th tuning instance")
    ap.add_argument("--out", help="default: results/phase2/ablate_f<f>.csv")
    ap.add_argument("--binary", default=str(ROOT / "cw"), help="timed runs use the fast build")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    names = list(CELLS) if a.cells == "all" else a.cells.split(",")
    if a.list:
        for n in names:
            print(f"{n:20s} {(BASE + ' ' + CELLS[n]).strip()}")
        return
    if a.f is None:
        ap.error("--f is required")

    out = ROOT / (a.out or f"results/phase2/ablate_f{a.f:g}.csv")
    for name in names:
        extra = (BASE + " " + CELLS[name]).strip()
        execute(time_tasks(a.f, a.seeds, a.every), savant_runner(a.binary, shlex.split(extra)),
                solver="savant", tag=name, extra=extra, out=out, jobs=a.jobs)


if __name__ == "__main__":
    main()
