#!/bin/bash
#SBATCH --job-name=sv-tmax_nds_n500
#SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=13:58:00
#SBATCH --array=0-2
#SBATCH --output=slurm/logs/tmax_nds_n500_%a.out

set -euo pipefail
cd $WORK/SAVANT_pub
module purge
module load gcc   # same line as slurm/nds_smoke.sbatch
export TMPDIR=${JOBSCRATCH:-/tmp}
export OMP_NUM_THREADS=1
mkdir -p results/jz/n500
lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket'
nvidia-smi -L
.venv/bin/python -m bench.run_baselines \
    --solver nds \
    --set data/nds_vrp/n500 --glob '*.vrp' \
    --stride 1 --limit 100 --seeds 1 --seed0 1 \
    --time-hgs 1.0 \
    --tag tmax \
    --jobs 1 --shard ${SLURM_ARRAY_TASK_ID}/3 --out results/jz/n500/tmax_nds_n500_shard${SLURM_ARRAY_TASK_ID}.csv
