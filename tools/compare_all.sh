#!/bin/sh
# compare_all.sh — run SAVANT, HGS-CVRP, LKH-3 and AILS-II on the same CVRPLib
# set at the same budget, validate every run, and print the gap to the set's
# optima / best known solutions.
#
#   sh tools/compare_all.sh [SET] [N_INSTANCES] [SECONDS]
#
#   sh tools/compare_all.sh X 24 10      # the quick one: ~4 min
#   sh tools/compare_all.sh X 100 10     # the whole Set X: ~15 min
#   sh tools/compare_all.sh XML100 500 1 # optimality gaps, proven optima
#
# Budgets are matched on CPU seconds per instance. SAVANT takes a step count
# rather than a time, so its budget is calibrated by a short probe run and
# scaled — see below. Everything runs sequentially: all four saturate the
# machine, so overlapping them would corrupt every timing.
#
# The CVRPLib sets are ROUNDED. cw needs --round; the other three drivers
# infer it and print what they chose.
set -e
cd "$(dirname "$0")/.."

SET=${1:-X}
N=${2:-24}
SECS=${3:-10}
TAG="${SET}_n${N}_t${SECS}"

BUNDLE="data/cvrplib/${SET}.cvrpb"
DIR="data/cvrplib/${SET}"
BKS="data/cvrplib/${SET}_bks.csv"
# XL ships pre-challenge solutions; the paper's table has the final BKS.
[ "$SET" = XL ] && [ -f baseline/xl_bks.csv ] && BKS="baseline/xl_bks.csv"

[ -d "$DIR" ] || { echo "$DIR not found — run: sh tools/setup.sh data" >&2; exit 1; }

# HGS holds a dense n x n double matrix: ~2.3 GiB at n = 10000.
JOBS=12
[ "$SET" = XL ] && JOBS=4

SAVANT_OPTS="--round --restarts 10 --ops 1,1,1,0 --sa-knn 20 --split end --split-every 1000"

echo "### set $SET, first $N instances, ~$SECS CPU-s each, $JOBS jobs"

echo
echo "--- HGS-CVRP"
uv run --no-project tools/run_hgs.py --dir "$DIR" --limit "$N" \
    --time "$SECS" --jobs "$JOBS" --name "HGS_$TAG" || true

echo
echo "--- LKH-3"
uv run --no-project tools/run_lkh.py --dir "$DIR" --limit "$N" \
    --time "$SECS" --jobs "$JOBS" --name "LKH_$TAG" || true

echo
echo "--- AILS-II"
if [ -f external/AILS-II/AILSII.jar ]; then
    # -limit is WALL clock, unlike HGS's CPU budget: with $JOBS running at
    # once each gets less real work. Use --jobs 1 for a strict timing claim.
    uv run --no-project tools/run_ails.py --dir "$DIR" --limit "$N" \
        --time "$SECS" --jobs "$JOBS" --name "AILS_$TAG" || true
else
    echo "skipped: external/AILS-II/AILSII.jar not built (needs a JDK)"
fi

echo
echo "--- SAVANT"
# cw's budget is a step count. Probe with 1M steps to measure this set's cost
# per step, then scale to the target CPU budget. The probe is a real run and
# is kept; the scaled one is the comparable point.
PROBE=$(uv run --no-project tools/run.py --name "SAVANT_probe_$TAG" \
    --bundle "$BUNDLE" --limit "$N" $SAVANT_OPTS --sa-steps 1000000 2>&1 \
    | grep "total time" | sed -E 's/.*, ([0-9.]+) s \(cumulative CPU\).*/\1/')
STEPS=$(awk -v p="$PROBE" -v n="$N" -v s="$SECS" \
    'BEGIN{ per=p/n; if(per<=0) per=0.001; printf "%d", 1000000*s/per }')
echo "probe: 1M steps = $PROBE CPU-s over $N instances -> using $STEPS steps"
uv run --no-project tools/run.py --name "SAVANT_$TAG" \
    --bundle "$BUNDLE" --limit "$N" $SAVANT_OPTS --sa-steps "$STEPS"

echo
echo "### validation (recomputed from the coordinates, no code shared with any solver)"
for d in results/*_HGS_$TAG results/*_LKH_$TAG results/*_AILS_$TAG results/*_SAVANT_$TAG; do
    [ -d "$d" ] || continue
    printf '%-44s ' "$(basename "$d")"
    uv run --no-project tools/validate.py "$d" 2>&1 | grep -E "error\(s\)" || echo "FAILED"
done

echo
echo "### gap to $BKS"
for d in results/*_HGS_$TAG results/*_LKH_$TAG results/*_AILS_$TAG results/*_SAVANT_$TAG; do
    [ -d "$d" ] || continue
    printf '%-44s ' "$(basename "$d")"
    uv run --no-project tools/gap_to_bks.py "$d" "$BKS" 2>/dev/null \
        | grep "mean gap" || echo "-"
done

echo
echo "### cpu per instance"
uv run --no-project tools/compare_table.py results/*_$TAG 2>/dev/null || true
