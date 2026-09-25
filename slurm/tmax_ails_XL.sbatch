#!/bin/bash
#SBATCH --job-name=sv-tmax_ails_XL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --time=14:56:00
#SBATCH --array=0-2
#SBATCH --output=slurm/logs/tmax_ails_XL_%a.out

set -euo pipefail
cd $WORK/SAVANT_pub
export TMPDIR=${JOBSCRATCH:-/tmp}
export OMP_NUM_THREADS=1
mkdir -p results/jz/XL
lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket'
.venv/bin/python -m bench.run_baselines \
    --solver ails \
    --set data/cvrplib/XL --glob '*.vrp' \
    --stride 1 --limit 0 --seeds 5 --seed0 1 \
    --time-hgs 1.0 \
    --tag tmax \
    --jobs 40 --shard ${SLURM_ARRAY_TASK_ID}/3 --out results/jz/XL/tmax_ails_XL_shard${SLURM_ARRAY_TASK_ID}.csv
