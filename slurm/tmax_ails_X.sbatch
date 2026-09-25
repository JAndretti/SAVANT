#!/bin/bash
#SBATCH --job-name=sv-tmax_ails_X
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --time=04:28:00
#SBATCH --array=0-1
#SBATCH --output=slurm/logs/tmax_ails_X_%a.out

set -euo pipefail
cd $WORK/SAVANT_pub
export TMPDIR=${JOBSCRATCH:-/tmp}
export OMP_NUM_THREADS=1
mkdir -p results/jz/X
lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket'
.venv/bin/python -m bench.run_baselines \
    --solver ails \
    --set data/cvrplib/X --glob '*.vrp' \
    --stride 1 --limit 0 --seeds 10 --seed0 1 \
    --time-hgs 1.0 \
    --tag tmax \
    --jobs 40 --shard ${SLURM_ARRAY_TASK_ID}/2 --out results/jz/X/tmax_ails_X_shard${SLURM_ARRAY_TASK_ID}.csv
