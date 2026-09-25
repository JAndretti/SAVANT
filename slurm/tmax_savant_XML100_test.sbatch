#!/bin/bash
#SBATCH --job-name=sv-tmax_savant_XML100_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --time=02:17:00
#SBATCH --array=0-0
#SBATCH --output=slurm/logs/tmax_savant_XML100_test_%a.out

set -euo pipefail
cd $WORK/SAVANT_pub
export TMPDIR=${JOBSCRATCH:-/tmp}
export OMP_NUM_THREADS=1
mkdir -p results/jz/XML100_test
lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket'
.venv/bin/python -m bench.run_savant \
    --set data/cvrplib/XML100_test --glob '*.vrp' \
    --stride 1 --limit 0 --seeds 10 --seed0 1 \
    --time-hgs 1.0 \
    --tag tmax \
    --extra '--pick 1 --pick2 1 --restarts 4 --kick 50 --kick-max 20' --binary cw \
    --jobs 40 --shard ${SLURM_ARRAY_TASK_ID}/1 --out results/jz/XML100_test/tmax_savant_XML100_test_shard${SLURM_ARRAY_TASK_ID}.csv
