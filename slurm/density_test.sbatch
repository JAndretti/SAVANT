#!/bin/bash
# Do concurrent single-threaded runs on one cpu_p1 node (2x Xeon Gold 6248,
# 20 cores and 6 memory channels per socket) slow each other down?
#
# SAVANT with a FIXED step budget does identical work whatever the load, so its
# run time measures throughput directly. Same (instance, seed) pairs at 40, 20
# and 1 concurrent runs, on XL (large memory footprint) and X instances.
# If 40 runs/node is markedly slower than 1, every timed solver would get less
# effective CPU than its nominal limit: use --jobs 20 in bench.slurm_campaign.
#
# Submit after slurm/build_all.sbatch; send back results/jz/density.csv.
#SBATCH --job-name=sv-density
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/density_%j.out

set -euo pipefail
cd "$WORK/SAVANT_pub"
export TMPDIR=${JOBSCRATCH:-/tmp}
lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket|NUMA'

run() {  # run <set> <stride> <seeds> <jobs>
  .venv/bin/python -m bench.run_savant --set data/cvrplib/$1 --stride $2 --seeds $3 \
      --steps 100000000 --extra "--restarts 1" --jobs $4 --tag density_$1_j$4 \
      --out results/jz/density.csv --binary cw_repro
}
for S in XL X; do
  run $S 10 4 40      # 40 runs at once
  run $S 10 4 20      # same 40 runs, 20 at a time
  run $S 10 1 1       # seed 1 alone: the unloaded reference
done
