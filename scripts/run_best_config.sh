#!/bin/sh
# run_best_config.sh — SAVANT's current defaults against the previous ones, on
# every benchmark set in data/, at an equal budget.
#
# The two configurations
# ---------------------
#   cand   the defaults as of 2026-08-07: the candidate the sweep, its
#          combination study and the CVRPLib X confirmation converged on
#          (README, "How these defaults were found")
#   prev   the defaults the solver shipped with until then
#
# They differ in exactly four flags, and passing prev's four to today's binary
# restores the old solver bit for bit:
#
#          --ops 1,1,1,0,1,0.05   vs  --ops 1,1,1,0,0,0     (swap*, route opening)
#          --t-decades 1          vs  --t-decades 2
#          --pick2 2              vs  --pick2 1
#          --kick 100             vs  --kick 0
#
# The change was measured on generated n = 100 instances and on CVRPLib X.
# This directory is the check on everything else, XL (n up to 10,000) included,
# which nothing in the derivation ever looked at.
#
# The budget
# ----------
# By default both configurations get the SAME WALL TIME per instance
# (--sa-time), not the same number of steps. Steps are not a fair unit here:
# `cand` draws swap*, which scans two whole routes, and fires a ruin & recreate
# every 100 steps, so one of its steps costs more than one of prev's. An
# iso-step comparison silently hands it more CPU. --sa-time removes that: cw
# times a short throwaway chain on the real instance and buys as many steps as
# the budget affords, so the two sides are matched on the only resource a user
# actually spends.
#
# What that costs is reproducibility: the step count depends on the machine and
# on what else it is doing. cw records the count it settled on for every
# instance in solutions.txt (#sa-steps), and re-running one instance with
# --sa-steps <its count> replays it exactly; a whole bundle cannot be replayed
# in one command, because the counts differ per instance. Pass a step count as
# the third argument for the reproducible-but-unfair mode.
#
# Usage:
#   sh scripts/run_best_config.sh                    # all sets, both configs, 0.6 s/inst
#   sh scripts/run_best_config.sh "n100 X"           # a subset
#   sh scripts/run_best_config.sh all 50             # smoke test: 50 instances each
#   sh scripts/run_best_config.sh all 0 1.5s         # a larger time budget
#   sh scripts/run_best_config.sh all 0 10000000     # iso-step instead of iso-time
#
# The three CVRPLib sets (XML100, X, XL) are scored with INTEGER distances
# (TSPLIB EUC_2D), so they get --round. The generated sets are float. Without
# --round the gap to the published references is not a gap to anything.
#
# Results land in results/best_config/<timestamp>_<set>_<config>/, each with
# run.py's usual config.json / run.log / results.csv / solutions.txt plus:
#   validation.txt   tools/validate.py  — feasibility and cost recomputation
#   bks_gap.txt      tools/gap_to_bks.py — CVRPLib sets only
#
# Then:  uv run --no-project scripts/summarize.py

set -eu
cd "$(dirname "$0")/.."

# OUT can be overridden to keep a smoke test out of the real results directory:
#   OUT=results/smoke sh scripts/run_best_config.sh n100 20 0.05s
OUT=${OUT:-results/best_config}
ALL_SETS="n20 n50 n100 XML100 X XL"

SETS=${1:-all}
[ "$SETS" = all ] && SETS=$ALL_SETS
LIMIT_N=${2:-0}
case "$LIMIT_N" in
    '' | 0) LIMIT="" ;;
    *)      LIMIT="--limit $LIMIT_N" ;;
esac

# ------------------------------------------------------------------ the budget
# "<x>s" -> a wall-clock budget per instance; a bare integer -> a step count.
# --sa-time needs a --sa-steps to size its calibration chain (cw uses
# min(20000, --sa-steps) steps for it), which is why the two travel together.
BUDGET_ARG=${3:-0.6s}
case "$BUDGET_ARG" in
    *s) SECS=${BUDGET_ARG%s}
        BUDGET="--sa-time $SECS --sa-steps 20000"
        BUDGET_WHAT="$SECS s/instance (iso-time)" ;;
    *)  BUDGET="--sa-steps $BUDGET_ARG"
        BUDGET_WHAT="$BUDGET_ARG steps/instance (iso-step)" ;;
esac

[ -x ./cw ] || { echo "./cw not found — run \`make\` first" >&2; exit 1; }

# ---------------------------------------------------------------- the configs
# Everything both sides share, spelled out at its default value so a run does
# not depend on what cw's built-in defaults happen to be on the day. The only
# options left out are the ones with no "off" spelling: --2opt-knn (a flag),
# and --t0 / --tend, which stay calibrated.
COMMON="--init cw \
  --knn 0 --lambda 1 --mu 0 \
  --or-max 3 --kick-max 10 \
  --t-accept 0.001 --sa-knn 20 \
  --pick 2 --pick-crit lb --pick-eps 0.3 \
  --vrank 1 --reloc-side coin \
  --cw-rand perturb --cw-alpha 0.03 \
  --race 0 --race-at 0.25 --pair 0 \
  --restarts 1 \
  --split off --split-every 0 --split-tour both \
  --empty-p 0 --dlb 0 --reheat 0 --t0-trim 0 \
  --check"

# --restarts stays at 1 on both sides. The sweep liked 8, but at a fixed total
# budget restarts lost on X (0.998 % -> 1.237 % gap to BKS) and the runs
# already in this directory say the same on XL: 16 x 625,000 reached 533,243
# against 529,404 for one run of 10,000,000.
CFG_LIST="cand prev"
FLAGS_cand="--ops 1,1,1,0,1,0.05 --t-decades 1 --pick2 2 --kick 100"
FLAGS_prev="--ops 1,1,1,0,0,0 --t-decades 2 --pick2 1 --kick 0"

# --------------------------------------------------------------- the datasets
# select_set TAG -> BUNDLE, EXTRA (per-set cw flags), BKS (reference CSV, "" if none)
select_set() {
    BUNDLE=''; EXTRA=''; BKS=''
    case "$1" in
        n20)    BUNDLE=data/cvrp_20.cvrpb ;;
        n50)    BUNDLE=data/cvrp_50.cvrpb ;;
        n100)   BUNDLE=data/cvrp_100.cvrpb ;;
        n200)   BUNDLE=data/cvrp_200.cvrpb ;;
        XML100) BUNDLE=data/cvrplib/XML100.cvrpb
                EXTRA=--round
                BKS=data/cvrplib/XML100_bks.csv ;;      # proven optima
        X)      BUNDLE=data/cvrplib/X.cvrpb
                EXTRA=--round
                BKS=data/cvrplib/X_bks.csv ;;
        XL)     BUNDLE=data/cvrplib/XL.cvrpb
                EXTRA=--round
                # the shipped .sol files are pre-challenge; the paper's table
                # has the final BKS, and scoring against the old ones would
                # understate every gap (same choice as tools/compare_all.sh).
                BKS=baseline/xl_bks.csv
                [ -f "$BKS" ] || BKS=data/cvrplib/XL_bks.csv ;;
        *)      echo "unknown set: $1 (expected one of: $ALL_SETS n200)" >&2
                return 1 ;;
    esac
    return 0
}

# ------------------------------------------------------------------- the runs
echo "### budget: $BUDGET_WHAT;  sets: $SETS;  configs: $CFG_LIST"

failed=''

for SET in $SETS; do
    select_set "$SET"
    if [ ! -f "$BUNDLE" ]; then
        echo "!! $SET: $BUNDLE not found — run \`sh tools/setup.sh data\`; skipped" >&2
        failed="$failed $SET(missing)"
        continue
    fi

    for CFG in $CFG_LIST; do
        eval "FLAGS=\$FLAGS_$CFG"
        NAME="${SET}_${CFG}"

        echo
        echo "=============================================================="
        echo "### $SET / $CFG   ($FLAGS)"
        echo "=============================================================="

        # shellcheck disable=SC2086  # the option strings must word-split
        uv run --no-project tools/run.py --out "$OUT" --name "$NAME" \
            --bundle "$BUNDLE" $EXTRA $LIMIT $COMMON $FLAGS $BUDGET

        DIR=$(ls -d "$OUT"/*_"$NAME" 2>/dev/null | tail -1)
        [ -n "$DIR" ] || { echo "!! no run directory for $NAME" >&2; exit 1; }

        # -- validation: independent recomputation of every cost ------------
        echo
        echo "--- validate $NAME"
        set +e
        uv run --no-project tools/validate.py "$DIR" > "$DIR/validation.txt" 2>&1
        vrc=$?
        set -e
        cat "$DIR/validation.txt"
        if [ $vrc -ne 0 ]; then
            echo "!! VALIDATION FAILED: $NAME" >&2
            failed="$failed $NAME(validate)"
        fi

        # -- gap to the published references, CVRPLib sets only -------------
        if [ -n "$BKS" ] && [ -f "$BKS" ]; then
            echo
            echo "--- gap to $BKS"
            set +e
            uv run --no-project tools/gap_to_bks.py "$DIR" "$BKS" \
                --csv "$DIR/bks_per_instance.csv" > "$DIR/bks_gap.txt" 2>&1
            grc=$?
            set -e
            cat "$DIR/bks_gap.txt"
            [ $grc -eq 0 ] || failed="$failed $NAME(bks)"
        fi
    done
done

echo
echo "=============================================================="
uv run --no-project scripts/summarize.py --out "$OUT" --csv || true

if [ -n "$failed" ]; then
    echo
    echo "!! problems in:$failed" >&2
    exit 1
fi
