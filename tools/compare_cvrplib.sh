#!/bin/sh
# compare_cvrplib.sh — SAVANT vs HGS on the CVRPLib sets that ship reference
# solutions (X, XL, XML100), measured as a gap to those references.
#
# Unlike the NeuOpt sets these are ROUNDED (TSPLIB EUC_2D), which is why every
# cw invocation below carries --round; run_hgs.py infers it and prints what it
# chose. XML100 ships proven optima, so its gap is an optimality gap; X and XL
# ship best known solutions, which bound it.
#
# Budgets are matched on CPU seconds per instance, not on wall clock: HGS's -t
# is a CPU budget and cw reports cumulative CPU, so that is the only figure
# that makes the two comparable.
#
# HGS needs ~2.3 GiB per process at n = 10000, hence --jobs 4 on XL.
#
#   sh tools/compare_cvrplib.sh [SET] [N_INSTANCES]
set -e
cd "$(dirname "$0")/.."

SET=${1:-X}
N=${2:-0}
LIM=""
[ "$N" -gt 0 ] 2>/dev/null && LIM="--limit $N"

SAVANT="--restarts 10 --ops 1,1,1,0 --sa-knn 20 --split end --split-every 1000"

case "$SET" in
  X)
    JOBS=12
    for T in 1 10; do
        uv run --no-project tools/run_hgs.py --dir data/cvrplib/X $LIM \
            --time $T --jobs $JOBS --name "HGS_X_t$T" || true
    done
    for S in 1000000 10000000 30000000; do
        uv run --no-project tools/run.py --name "SAVANT_X_$S" \
            --bundle data/cvrplib/X.cvrpb $LIM --round --sa-steps $S $SAVANT
    done
    ;;
  XML100)
    JOBS=12
    for T in 0.25 1; do
        uv run --no-project tools/run_hgs.py --dir data/cvrplib/XML100 $LIM \
            --time $T --jobs $JOBS --name "HGS_XML100_t$T" || true
    done
    for S in 1000000 3000000; do
        uv run --no-project tools/run.py --name "SAVANT_XML100_$S" \
            --bundle data/cvrplib/XML100.cvrpb $LIM --round --sa-steps $S $SAVANT
    done
    ;;
  XL)
    # 4 jobs, not 12: HGS peaks at ~2.3 GiB per process at n = 10000.
    # 1 restart, not 10: at this size the budget is better spent on one long
    # trajectory than on ten short ones.
    uv run --no-project tools/run_hgs.py --dir data/cvrplib/XL $LIM \
        --time 20 --jobs 4 --name "HGS_XL_t20" || true
    uv run --no-project tools/run.py --name "SAVANT_XL_130M" \
        --bundle data/cvrplib/XL.cvrpb $LIM --round --sa-steps 130000000 \
        --restarts 1 --ops 1,1,1,0 --sa-knn 20 --split end --split-every 1000
    ;;
  *)
    echo "unknown set: $SET (expected X, XL or XML100)" >&2
    exit 2
    ;;
esac

echo
echo "### gaps to the reference solutions"
for d in results/*_HGS_${SET}_* results/*_SAVANT_${SET}_*; do
    [ -d "$d" ] || continue
    printf '%-42s ' "$(basename "$d")"
    uv run --no-project tools/gap_to_bks.py "$d" \
        "data/cvrplib/${SET}_bks.csv" 2>/dev/null | grep "mean gap" || echo "-"
done
