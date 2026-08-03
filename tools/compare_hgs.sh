#!/bin/sh
# compare_hgs.sh — SAVANT vs HGS-CVRP on the same instances, same machine.
#
# Sweeps a CPU-time budget for HGS and a step budget for SAVANT over the same
# subset of a .cvrpb bundle, so the two can be plotted as time-quality curves
# rather than compared at a single arbitrary operating point.
#
# Runs strictly sequentially: both solvers saturate the machine, so overlapping
# them would corrupt every timing.
#
#   sh tools/compare_hgs.sh [N_INSTANCES] [BUNDLE]
set -e
cd "$(dirname "$0")/.."

N=${1:-1000}
BUNDLE=${2:-data/cvrp_100.cvrpb}
TAG=$(basename "$BUNDLE" .cvrpb)

echo "### HGS-CVRP — CPU-time budget sweep, $N instances of $BUNDLE"
for T in 0.1 0.25 0.5 1.0 2.0; do
    uv run --no-project tools/run_hgs.py \
        --bundle "$BUNDLE" --limit "$N" --time "$T" --jobs 12 --seed 0 \
        --name "HGS_${TAG}_t${T}"
done

echo "### HGS-CVRP — default termination (20000 non-improving iterations)"
uv run --no-project tools/run_hgs.py \
    --bundle "$BUNDLE" --limit "$N" --jobs 12 --seed 0 \
    --name "HGS_${TAG}_default"

echo "### SAVANT — step budget sweep, same instances"
for S in 100000 300000 1000000 3000000; do
    uv run --no-project tools/run.py --name "SAVANT_${TAG}_${S}" \
        --bundle "$BUNDLE" --limit "$N" \
        --sa-steps "$S" --restarts 10 --ops 1,1,1,0 --sa-knn 20 \
        --split end --split-every 1000
done

echo "### done"
echo
echo "For the CVRPLib sets (X, XL, XML100), which are ROUNDED and ship"
echo "reference solutions, see tools/compare_cvrplib.sh instead."
