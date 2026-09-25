#!/bin/bash
#SBATCH --job-name=sv-tmax_hgs_n1000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --time=10:44:00
#SBATCH --array=0-0
#SBATCH --output=slurm/logs/tmax_hgs_n1000_%a.out

set -euo pipefail
cd $WORK/SAVANT_pub
export TMPDIR=${JOBSCRATCH:-/tmp}
export OMP_NUM_THREADS=1
mkdir -p results/jz/n1000
lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket'
.venv/bin/python -m bench.run_baselines \
    --solver hgs \
    --set data/nds_vrp/n1000 --glob '*.vrp' \
    --stride 1 --limit 100 --seeds 5 --seed0 1 \
    --time-hgs 1.0 \
    --tag tmax \
    --jobs 40 --shard ${SLURM_ARRAY_TASK_ID}/1 --out results/jz/n1000/tmax_hgs_n1000_shard${SLURM_ARRAY_TASK_ID}.csv
