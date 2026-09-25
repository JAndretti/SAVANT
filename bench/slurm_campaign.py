"""Generate a Slurm array job for a timed campaign on a CPU cluster (e.g. Jean Zay).

One array element = one exclusive node = one shard of the (instance, seed) runs,
executed by the usual pinned pool (one single-threaded run per physical core).
Every solver of a comparison must run on the same partition, with the same
number of jobs per node, so that time limits mean the same thing for all.

    uv run python -m bench.slurm_campaign --solver hgs --set data/cvrplib/X --time-hgs 1 \
        --seeds 10 --tag tmax --shards 10 --cores 40 --account abc@cpu
    uv run python -m bench.slurm_campaign --solver savant --set data/cvrplib/X --time-hgs 1 \
        --seeds 10 --tag tmax --extra "--kick 50" --shards 10 --cores 40 --account abc@cpu
    # -> slurm/tmax_hgs_X.sbatch ; on the cluster: sbatch slurm/tmax_hgs_X.sbatch
    uv run python -m bench.merge_csv results/jz/X/tmax_hgs_shard*.csv -o results/jz/X/tmax_hgs.csv

--cores must be the number of PHYSICAL cores of one node of the partition
(check with `lscpu` inside a job); --time for Slurm is estimated from the budgets.

GPU mode (NDS only): --gpus G makes each array element a G-GPU allocation on the
Jean Zay A100 partition (-C a100, 8 cores per GPU, not exclusive: A100 nodes are
shared and billed per GPU), running G concurrent runs, one per GPU:
    uv run python -m bench.slurm_campaign --solver nds --gpus 1 --set data/nds_vrp/n1000 \
        --limit 100 --time-hgs 1 --seeds 5 --tag tmax --shards 20
"""

from __future__ import annotations

import argparse
import math
import shlex

from bench.common import ROOT, is_rounded
from bench.pool import add_common_args, add_time_args, make_tasks, seconds_for

BASELINES = ("hgs", "filo", "filo2", "ails", "lkh", "ortools", "nds")
CPUS_PER_GPU = 8           # Jean Zay A100 partition: 64 cores for 8 GPUs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solver", required=True, choices=("savant", *BASELINES))
    add_common_args(ap, out_required=False)      # outputs are named per shard
    add_time_args(ap.add_mutually_exclusive_group(required=True))
    ap.add_argument("--extra", default="", help="SAVANT only: extra cw options")
    ap.add_argument("--shards", type=int, required=True, help="number of nodes (array size)")
    ap.add_argument("--cores", type=int, default=40,
                    help="physical cores per node (Jean Zay cpu_p1: 2x Xeon Gold 6248 = 40)")
    ap.add_argument("--account", default="",
                    help="Slurm account; omitted by default: pass it at submission (sbatch -A ...)")
    ap.add_argument("--partition", default="", help="e.g. cpu_p1 (omitted if empty)")
    ap.add_argument("--qos", default="", help="e.g. qos_cpu-t3 (omitted if empty)")
    ap.add_argument("--max-hours", type=float, default=20.0,
                    help="refuse shards longer than this (cpu_p1 hard limit: 100 h)")
    ap.add_argument("--repo", default="$WORK/SAVANT_pub", help="repository path on the cluster")
    ap.add_argument("--gpus", type=int, default=0,
                    help="GPU mode: GPUs per array element on the A100 partition (NDS only)")
    ap.add_argument("--constraint", default="a100", help="GPU mode: Slurm -C constraint")
    a = ap.parse_args()
    if (a.solver == "nds") != (a.gpus > 0):
        ap.error("NDS runs in GPU mode (--gpus >= 1), and only NDS does")
    if a.gpus:
        a.cores = CPUS_PER_GPU * a.gpus
        a.jobs = a.gpus                          # one run per GPU
        if a.max_hours > 20:
            ap.error("--max-hours: A100 jobs are limited to 20 h (qos_gpu_a100-t3)")
    if a.shard != "0/1":
        ap.error("--shard is set by the array job itself")
    if a.max_hours > 100:
        ap.error("--max-hours: cpu_p1 does not accept jobs longer than 100 h")
    if a.solver in ("filo", "filo2", "lkh") and not all(
            is_rounded(v) for v in a.set.glob(a.glob)):
        ap.error(f"{a.solver} rounds arc costs: not run on real-valued instances (see bench.run_baselines)")
    jpn = a.jobs or a.cores          # concurrent runs per node (--jobs; default: every core)
    if jpn > a.cores:
        ap.error("--jobs cannot exceed --cores")

    # wall-time estimate: tasks of one shard packed greedily on `jpn` slots
    worst = 0.0
    total = 0.0
    for k in range(a.shards):
        a.shard = f"{k}/{a.shards}"
        tasks = make_tasks(a, lambda v: seconds_for(a, v), lambda v: "")
        slots = [0.0] * min(jpn, max(1, len(tasks)))
        for t in tasks:                                   # already sorted by decreasing budget
            i = min(range(len(slots)), key=slots.__getitem__)
            slots[i] += t.budget_s * 1.05 + 2.0           # start-up, preprocessing, validation
        worst = max(worst, max(slots))
        total += sum(t.budget_s for t in tasks)
    hours = worst / 3600 * 1.15 + 0.25
    if hours > a.max_hours:
        ap.error(f"a shard needs ~{hours:.1f} h > --max-hours {a.max_hours}: raise --shards")
    hh = math.ceil(hours * 60)

    set_name = a.set.name
    stem = f"{a.tag}_{a.solver}_{set_name}"
    out_dir = f"results/jz/{set_name}"
    module = "bench.run_savant" if a.solver == "savant" else "bench.run_baselines"
    budget = f"--time {a.time}" if a.time is not None else f"--time-hgs {a.time_hgs}"
    cmd = [f".venv/bin/python -m {module}"]
    if a.solver != "savant":
        cmd.append(f"--solver {a.solver}")
    cmd += [f"--set {shlex.quote(str(a.set))} --glob {shlex.quote(a.glob)}",
            f"--stride {a.stride} --limit {a.limit} --seeds {a.seeds} --seed0 {a.seed0}",
            budget, f"--tag {a.tag}"]
    if a.solver == "savant":
        cmd.append(f"--extra {shlex.quote(a.extra)} --binary cw")
    cmd.append(f"--jobs {jpn} --shard ${{SLURM_ARRAY_TASK_ID}}/{a.shards} "
               f"--out {out_dir}/{stem}_shard${{SLURM_ARRAY_TASK_ID}}.csv")

    lines = ["#!/bin/bash",
             f"#SBATCH --job-name=sv-{stem}"]
    if a.account:
        lines.append(f"#SBATCH --account={a.account}")
    if a.partition:
        lines.append(f"#SBATCH --partition={a.partition}")
    if a.qos:
        lines.append(f"#SBATCH --qos={a.qos}")
    if a.gpus:
        lines += [f"#SBATCH -C {a.constraint}", f"#SBATCH --gres=gpu:{a.gpus}"]
    lines += ["#SBATCH --nodes=1", "#SBATCH --ntasks=1", f"#SBATCH --cpus-per-task={a.cores}",
              "#SBATCH --hint=nomultithread", *([] if a.gpus else ["#SBATCH --exclusive"]),
              f"#SBATCH --time={hh // 60:02d}:{hh % 60:02d}:00",
              f"#SBATCH --array=0-{a.shards - 1}",
              f"#SBATCH --output=slurm/logs/{stem}_%a.out",
              "",
              "set -euo pipefail",
              f"cd {a.repo}",
              # NDS's C++ operators were compiled by slurm/nds_smoke.sbatch under this module
              # and need its libstdc++ at run time (the system one in /lib64 is too old:
              # "GLIBCXX_3.4.32 not found", jobs 100877-100880, 2026-09-23)
              *(["module purge", "module load gcc   # same line as slurm/nds_smoke.sbatch"]
                if a.gpus else []),
              "export TMPDIR=${JOBSCRATCH:-/tmp}",
              "export OMP_NUM_THREADS=1",
              f"mkdir -p {out_dir}",
              "lscpu | grep -E 'Model name|^CPU\\(s\\)|Thread|Core|Socket'",
              *(["nvidia-smi -L"] if a.gpus else []),
              " \\\n    ".join(cmd), ""]
    (ROOT / "slurm" / "logs").mkdir(parents=True, exist_ok=True)
    path = ROOT / "slurm" / f"{stem}.sbatch"
    path.write_text("\n".join(lines))
    if a.gpus:
        print(f"wrote {path}: {a.shards} x {a.gpus} GPU, ~{hours:.1f} h per shard, "
              f"{total / 3600:.0f} GPU-hours of budget, ~{a.shards * a.gpus * hours:.0f} GPU-h billed at most")
    else:
        print(f"wrote {path}: {a.shards} nodes x {jpn} runs/node ({a.cores} cores), ~{hours:.1f} h per shard, "
              f"{total / 3600:.0f} core-hours of budget (+ overhead; exclusive nodes bill all cores)")


if __name__ == "__main__":
    main()
