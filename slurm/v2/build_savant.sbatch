#!/bin/bash
# Rebuild cw only (October 2026 code: sorted kNN, kick-early default, bnx/d0),
# on a compute node (-march=native), then a 3-second smoke run on X-n101.
# The baselines are untouched: their binaries and results stay valid.
#SBATCH --job-name=sv2-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/v2_build_%j.out

set -euo pipefail
cd "$WORK/SAVANT_pub"
module purge
module load gcc
lscpu | grep -E 'Model name|^CPU\(s\)'
gcc --version | head -1
rm -f cw cw_repro
make cw repro
sha256sum SAVANT/cw.c
./cw --help | grep -E 'kick-early|kick-repair' | head -3
T=$(mktemp -d); cp data/cvrplib/X/X-n101-k25.vrp $T/
./cw --dir $T --round --sa-wall 3 --threads 1 | tail -5
