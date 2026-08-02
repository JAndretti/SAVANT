#!/usr/bin/env bash
# run_sweep.sh — parameter sweep for ./cw
#
# Every execution writes three files into sweep/results/<study>/:
#     <tag>.log    raw cw output (resolved config header + summary)
#     <tag>.csv    per-instance results (--csv)
#     <tag>.meta   study, tag, exact command line, exit code, wall time
#
# The .meta file records the full command line rather than encoding parameters
# in the tag: analyze_sweep.py re-parses it into an option dict, so adding an
# option here needs no change on the analysis side.
#
# All runs of a study share the same (n, m, seed), hence the *same instances*
# (instance k is generated from seed+k, cw.c:2064). Comparisons are therefore
# paired, which is what analyze_sweep.py exploits.
#
# Usage:
#     sweep/run_sweep.sh                 # everything
#     sweep/run_sweep.sh init ops        # only those studies
#     M=200 sweep/run_sweep.sh           # fewer instances (quick check)
#     RESUME=1 sweep/run_sweep.sh        # skip runs already on disk
#     sweep/run_sweep.sh --list          # list the studies

set -u
shopt -s inherit_errexit 2>/dev/null || true

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-$ROOT/cw}
OUT=${OUT:-$ROOT/sweep/results}

# ---------------------------------------------------------------- defaults
M=${M:-1000}            # instances per run
SEED=${SEED:-42}        # instance seed (same instances across every run)
N=${N:-100}             # default dimension
STEPS=${STEPS:-100000}  # default SA budget
RESUME=${RESUME:-0}     # 1 = skip a run whose .meta already exists

STUDIES=(init ops knn timing restarts split pick temp construct tuned)

n_run=0; n_skip=0; n_fail=0
FAILED=()
T_START=$(date +%s)

usage() {
    echo "usage: $0 [--list] [study ...]"
    echo "studies: ${STUDIES[*]}"
    echo "env: M=$M SEED=$SEED N=$N STEPS=$STEPS RESUME=$RESUME OUT=$OUT"
}

# run <study> <tag> <cw args...>
run() {
    local study=$1 tag=$2; shift 2
    local dir="$OUT/$study"
    mkdir -p "$dir"
    local base="$dir/$tag"

    if [[ $RESUME == 1 && -f "$base.meta" ]] && grep -q '^exit=0$' "$base.meta" 2>/dev/null; then
        n_skip=$((n_skip + 1)); return 0
    fi

    local cmd=("$BIN" "$@" --csv "$base.csv")
    local t0 t1 rc
    t0=$(date +%s.%N)
    "${cmd[@]}" >"$base.log" 2>&1
    rc=$?
    t1=$(date +%s.%N)

    {
        echo "study=$study"
        echo "tag=$tag"
        echo "cmd=${cmd[*]}"
        echo "exit=$rc"
        echo "wall_s=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b-a}')"
    } >"$base.meta"

    n_run=$((n_run + 1))
    if [[ $rc -ne 0 ]]; then
        n_fail=$((n_fail + 1)); FAILED+=("$study/$tag")
        printf '  [FAIL rc=%d] %s/%s\n' "$rc" "$study" "$tag"
    else
        printf '  [%3d] %-12s %-34s %6.2fs\n' "$n_run" "$study" "$tag" \
               "$(awk -v a="$t0" -v b="$t1" 'BEGIN{print b-a}')"
    fi
    return 0
}

# common instance-source arguments for a given n
src() { echo "--random -n $1 -m $M --seed $SEED"; }

banner() { printf '\n=== %s: %s\n' "$1" "$2"; }

# ============================================================== S1  init
# Does a random initial solution catch up with Clarke & Wright, and how fast?
# Same SA budget on both sides, budget swept over three decades.
study_init() {
    banner init "random vs C&W construction across the SA budget"
    local steps n init
    for n in 20 50 100 200; do
        for init in cw random; do
            for steps in 1000 2000 5000 10000 20000 50000 100000 200000 500000 1000000; do
                run init "n${n}_${init}_s${steps}" \
                    $(src "$n") --sa-steps "$steps" --init "$init"
            done
        done
    done
}

# ============================================================== S2  ops
# Every non-empty subset of {relocate, swap, 2-opt, or-opt}, plus a few
# unbalanced weightings, plus the or-opt segment length (--or-max, the extra
# knob that only matters once or-opt is switched on).
study_ops() {
    banner ops "operator subsets, weightings and --or-max"
    local r s t o L
    for r in 0 1; do for s in 0 1; do for t in 0 1; do for o in 0 1; do
        [[ "$r$s$t$o" == 0000 ]] && continue
        run ops "sub_$r$s$t$o" $(src "$N") --sa-steps "$STEPS" --ops "$r,$s,$t,$o"
    done; done; done; done

    # unbalanced weights: is the uniform 1,1,1,1 mix the right one?
    for w in "2,1,1,1" "1,2,1,1" "1,1,2,1" "1,1,1,2" "4,1,1,1" "1,1,1,4" "3,3,1,1"; do
        run ops "w_${w//,/-}" $(src "$N") --sa-steps "$STEPS" --ops "$w"
    done

    # --or-max is only meaningful with or-opt active (weight > 0). Valid range
    # is 2..8 (cw.c:2042); segments are drawn uniformly in [2, or_max].
    for L in 2 3 4 5 6 7 8; do
        run ops "ormax_$L" $(src "$N") --sa-steps "$STEPS" --ops 1,1,1,1 --or-max "$L"
    done
}

# ============================================================== S3  sa-knn
# Candidate-neighbourhood size, across dimension. K is clamped to n-1
# (cw.c:1399), so at n=20 the values 30 and 50 collapse onto 19 -- the analysis
# reports the *effective* K.
study_knn() {
    banner knn "--sa-knn across dimension"
    local n K
    for n in 20 50 100 200; do
        for K in 0 5 10 20 30 50; do
            run knn "n${n}_K${K}" $(src "$n") --sa-steps "$STEPS" --sa-knn "$K"
        done
    done
}

# ============================================================== S4  timing
# Compute cost vs dimension. Each n is run three times:
#   --no-sa            construction alone
#   --sa-steps S       construction + S annealing steps
#   --sa-steps 10S     construction + 10S annealing steps
# The marginal cost of annealing is (t(10S) - t(S)) / 9: both runs pay the same
# construction, so it cancels exactly. Two reasons not to subtract the --no-sa
# run instead, both visible at n=1000 where construction is 95 % of the time:
# run-to-run spread on the construction (+-3 %) is then larger than the whole
# annealing cost, and the two runs sit in different memory-pressure regimes.
# A 2S delta is not enough either -- 10S puts the signal well clear of the noise.
study_timing() {
    banner timing "cost of construction vs annealing across n, and thread scaling"
    local n T
    for n in 20 50 100 200 500 1000; do
        run timing "n${n}_nosa"  $(src "$n") --no-sa
        run timing "n${n}_sa"    $(src "$n") --sa-steps "$STEPS"
        run timing "n${n}_sa10"  $(src "$n") --sa-steps $((STEPS * 10))
    done
    for T in 1 2 4 8 12 16 24; do
        run timing "threads_$T" $(src "$N") --sa-steps "$STEPS" --threads "$T"
    done
}

# ============================================================== S5  restarts
# (a) raw gain at fixed per-restart budget -- R restarts cost R times more.
# (b) iso-budget: R * steps held constant. This is the only fair question,
#     "given a fixed number of SA steps, how should they be split?".
study_restarts() {
    banner restarts "restart count, at fixed and at equal total budget"
    local R S
    for R in 1 2 4 8 16 32; do
        run restarts "fixed_R${R}"  $(src "$N") --sa-steps 10000 --restarts "$R"
    done

    local BUDGET=320000
    for R in 1 2 4 8 16 32; do
        S=$((BUDGET / R))
        run restarts "iso_R${R}"    $(src "$N") --sa-steps "$S" --restarts "$R"
    done

    # does a random start profit more from restarts than C&W does?
    for R in 1 4 16; do
        S=$((BUDGET / R))
        run restarts "isorand_R${R}" $(src "$N") --sa-steps "$S" --restarts "$R" --init random
    done

    # how the restart diversity is produced
    for MODE in off perturb param both; do
        run restarts "cwrand_$MODE" $(src "$N") --sa-steps 40000 --restarts 8 --cw-rand "$MODE"
    done
    for A in 0.01 0.03 0.1 0.3 0.9; do   # cw.c:2041 requires 0 <= alpha < 1
        run restarts "alpha_$A"     $(src "$N") --sa-steps 40000 --restarts 8 \
            --cw-rand perturb --cw-alpha "$A"
    done
}

# ============================================================== S6  split
# --split (where Split is applied) and --split-every (periodic Split during
# annealing) are independent code paths (cw.c:1440 vs 1640/1658), so the full
# grid including "off + every" is meaningful.
study_split() {
    banner split "--split mode x --split-every, and --split-tour"
    local MODE EV n
    for MODE in off cw end both; do
        for EV in 0 100 1000 10000; do
            run split "m${MODE}_e${EV}" $(src "$N") --sa-steps "$STEPS" \
                --split "$MODE" --split-every "$EV"
        done
    done
    for TOUR in routes sweep both; do
        run split "tour_$TOUR" $(src "$N") --sa-steps "$STEPS" \
            --split both --split-every 1000 --split-tour "$TOUR"
    done
    # more routes at larger n means more for Split to repartition
    for n in 200 500; do
        for MODE in off both; do
            for EV in 0 1000; do
                run split "n${n}_m${MODE}_e${EV}" $(src "$n") --sa-steps "$STEPS" \
                    --split "$MODE" --split-every "$EV"
            done
        done
    done
}

# ============================================================== S7  pick
# Vertex-selection rule. T=1 is uniform and ignores the criterion, so it is run
# once. The last block crosses --pick with --sa-knn on purpose: at K=0 the code
# silently forces uniform selection (cw.c:1400) while the header still prints
# the requested rule -- the sweep should show a flat row there.
study_pick() {
    banner pick "--pick tournament size x --pick-crit, and the K=0 fallback"
    local T C E K
    run pick "T1_uniform" $(src "$N") --sa-steps "$STEPS" --pick 1
    for T in 0 2 3 4 8 16 32; do
        for C in lb rem remnorm raw; do
            run pick "T${T}_${C}" $(src "$N") --sa-steps "$STEPS" \
                --pick "$T" --pick-crit "$C"
        done
    done
    for E in 0.03 0.1 0.3 1.0 3.0; do
        run pick "eps_${E}" $(src "$N") --sa-steps "$STEPS" \
            --pick 0 --pick-crit rem --pick-eps "$E"
    done
    for T in 1 2 8; do
        for K in 0 10 20; do
            run pick "inter_T${T}_K${K}" $(src "$N") --sa-steps "$STEPS" \
                --pick "$T" --sa-knn "$K"
        done
    done
}

# ============================================================== S8  temp
# Annealing schedule: initial acceptance target and the number of decades
# spanned by T.
study_temp() {
    banner temp "--t-accept x --t-decades"
    local A D
    for A in 0.0001 0.001 0.01 0.1 0.5; do
        for D in 1 2 3 4; do
            run temp "a${A}_d${D}" $(src "$N") --sa-steps "$STEPS" \
                --t-accept "$A" --t-decades "$D"
        done
    done
}

# ============================================================== S9  construct
# Construction quality on its own (--no-sa) and after annealing. The interesting
# question is whether a better construction survives the SA, or whether the SA
# washes the difference out.
study_construct() {
    banner construct "savings parameters, construction kNN and 2-opt"
    local L MU K
    for L in 0.6 0.8 1.0 1.2 1.4; do
        for MU in 0 0.2 0.5 1.0; do
            run construct "nosa_l${L}_m${MU}" $(src "$N") --no-sa --lambda "$L" --mu "$MU"
            run construct "sa_l${L}_m${MU}"   $(src "$N") --sa-steps "$STEPS" \
                --lambda "$L" --mu "$MU"
        done
    done
    # savings-list truncation, at a size where it actually bites
    for K in 0 5 10 20 50; do
        run construct "nosa_n200_knn${K}" $(src 200) --no-sa --knn "$K"
        run construct "sa_n200_knn${K}"   $(src 200) --sa-steps "$STEPS" --knn "$K"
    done
    run construct "nosa_n200_exact" $(src 200) --no-sa --exact
    run construct "sa_n200_exact"   $(src 200) --sa-steps "$STEPS" --exact
    # intra-route 2-opt after construction
    run construct "nosa_2opt_off" $(src "$N") --no-sa
    run construct "nosa_2opt_on"  $(src "$N") --no-sa --2opt
    run construct "sa_2opt_off"   $(src "$N") --sa-steps "$STEPS"
    run construct "sa_2opt_on"    $(src "$N") --sa-steps "$STEPS" --2opt
}

# ============================================================== S10 tuned
# Candidate combinations against the stock defaults, at equal SA budget. These
# are hypotheses, not the outcome of the sweep: analyze_sweep.py reports the
# per-study winners so the combination can be rebuilt from the actual results.
study_tuned() {
    banner tuned "candidate combinations vs defaults, at equal budget"
    local n
    for n in 50 100 200; do
        run tuned "n${n}_default"  $(src "$n") --sa-steps "$STEPS"
        run tuned "n${n}_oropt"    $(src "$n") --sa-steps "$STEPS" --ops 1,1,1,1
        run tuned "n${n}_critrem"  $(src "$n") --sa-steps "$STEPS" --pick-crit rem
        run tuned "n${n}_split"    $(src "$n") --sa-steps "$STEPS" --split both --split-every 1000
        run tuned "n${n}_all"      $(src "$n") --sa-steps "$STEPS" --ops 1,1,1,1 \
            --pick-crit rem --split both --split-every 1000
        run tuned "n${n}_all_r8"   $(src "$n") --sa-steps $((STEPS / 8)) --restarts 8 \
            --ops 1,1,1,1 --pick-crit rem --split both --split-every 1000
    done
}

# ==========================================================================
main() {
    if [[ ${1:-} == --list || ${1:-} == -h || ${1:-} == --help ]]; then usage; return 0; fi
    [[ -x $BIN ]] || { echo "$BIN not found or not executable: run \`make\` first" >&2; return 1; }

    local wanted=("$@")
    [[ ${#wanted[@]} -eq 0 ]] && wanted=("${STUDIES[@]}")

    mkdir -p "$OUT"
    printf 'binary   : %s\n' "$BIN"
    printf 'output   : %s\n' "$OUT"
    printf 'instances: m=%s seed=%s (identical across every run)\n' "$M" "$SEED"
    printf 'defaults : n=%s sa-steps=%s\n' "$N" "$STEPS"
    printf 'studies  : %s\n' "${wanted[*]}"

    "$BIN" --help >"$OUT/cw_help.txt" 2>&1
    (cd "$ROOT" && git rev-parse HEAD 2>/dev/null) >"$OUT/git_head.txt" 2>/dev/null || true
    md5sum "$BIN" >"$OUT/binary.md5" 2>/dev/null || true

    local s
    for s in "${wanted[@]}"; do
        if ! declare -F "study_$s" >/dev/null; then
            echo "unknown study: $s (known: ${STUDIES[*]})" >&2; n_fail=$((n_fail+1)); continue
        fi
        "study_$s"
    done

    local elapsed=$(( $(date +%s) - T_START ))
    printf '\n--------------------------------------------------\n'
    printf 'runs: %d  skipped: %d  failed: %d  elapsed: %dm%02ds\n' \
           "$n_run" "$n_skip" "$n_fail" $((elapsed / 60)) $((elapsed % 60))
    if [[ ${#FAILED[@]} -gt 0 ]]; then
        printf 'failed runs:\n'; printf '  %s\n' "${FAILED[@]}"
    fi
    printf 'analyse with: uv run sweep/analyze_sweep.py\n'
    return 0
}

main "$@"
