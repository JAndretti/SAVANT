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
    # additions, tested at equal time once the kNN lists were sorted (2026-10-19;
    # the screening row "K = 30, bias towards nearer ranks" turned into a gain).
    # Run with the sorted cw, `final` included, into results/p1_sorted_knn/:
    #   uv run python -m bench.ablate --f 0.02 --every 2 --cells final,knn_30,knn_30_vrank_2 \
    #       --out results/p1_sorted_knn/ablate_f0.02.csv        (and --f 0.1 --every 4)
    #   results in results/p1_sorted_knn
    "knn_30": "--sa-knn 30",
    "knn_30_vrank_2": "--sa-knn 30 --vrank 2",
    # speedups proposed in the October 2026 code review (2026-10-20), tested at equal time with the
    # cw that includes the bit-identical bnx + d0 speedups, `final` re-run:
    #   uv run python -m bench.ablate --f 0.02 --every 2 \
    #       --cells final,op_cycle,kick_early,op_cycle_kick_early \
    #       --out results/p2_speedups/ablate_f0.02.csv        (and --f 0.1 --every 4)
    #   results in results/p2_speedups
    "op_cycle": "--op-cycle",
    "kick_early": "--kick-early",
    "op_cycle_kick_early": "--op-cycle --kick-early",
    # configuration changes proposed in the code review (2026-10-20; --kick-early is
    # the default from here on, `final` re-run):
    #   uv run python -m bench.ablate --f 0.02 --every 2 \
    #       --cells final,no_swap,restarts_2,pursue_2 \
    #       --out results/p3_config/ablate_f0.02.csv          (and --f 0.1 --every 4)
    #   results in results/p3_config
    "no_swap": "--ops 1,0,1,0,1,0.05",
    "restarts_2": "--restarts 2",                      # control for pursue_2
    "pursue_2": "--restarts 2 --pursue 0.5",           # 2 starts on T/4 each, best on T/2
    # follow-up (2026-10-21): wider schedule, gain grows with the budget in results/p3_starts
    "decades_1.25": "--t-decades 1.25",
    "decades_1.5": "--t-decades 1.5",
    # variant (2026-10-21): route elimination at the end (--elim; F is a
    # share of the budget, the starts get 1 - F). Tuning set only, never X (test set).
    #   uv run python -m bench.ablate --f 0.02 --every 2 --cells final,elim_0.25,... \
    #       --out results/p4_elim/ablate_f0.02.csv            (f 0.002 every 1, f 0.1 every 4)
    #   results in results/p4_elim
    "elim_0.25": "--elim 0.25",                          # Nagata & Braysy in full, every level
    "elim_0.25_lv0": "--elim 0.25 --elim-levels 0",      # ... re-anneal the last level only
    "elim_0.1": "--elim 0.1",
    "elim_0.5": "--elim 0.5",
    "reanneal_0.25": "--elim 0.25 --elim-mode 0",        # control: the low-T re-anneal alone
    # variant (2026-10-21): kick with repair (--kick-repair), also tried with a
    # kick every 100 steps. Same commands, out
    # results/p4_repair/ablate_f*.csv
    "repair_1": "--kick-repair 1",
    "repair_2": "--kick-repair 2",
    "repair_1_k100": "--kick-repair 1 --kick 100",
    "repair_2_k100": "--kick-repair 2 --kick 100",
    # variant (2026-10-27): SISR ingredients the kick lacks (Christiaens & Vanden Berghe
    # 2020): the order in which removed customers are reinserted, and blinks.
    #   uv run python -m bench.ablate --f 0.02 --every 2 --cells final,order_random,... \
    #       --out results/p4_kick/ablate_f0.02.csv            (f 0.1 every 4 for survivors)
    #   results in results/p4_kick
    "order_random": "--kick-order 1",
    "order_demand": "--kick-order 2",
    "order_far": "--kick-order 3",
    "order_sisr": "--kick-order 5",                      # one of 1-4 per kick, weights 4:4:2:1
    "blink_0.01": "--kick-blink 0.01",
    "sisr_blink": "--kick-order 5 --kick-blink 0.01",     # both SISR ingredients
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
