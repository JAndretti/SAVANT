#!/usr/bin/env bash
# run_sweep.sh — parameter sweep for ./cw
#
# Every execution writes three files into sweep/results/<tier>/s<seed>/<study>/:
#     <tag>.log    raw cw output (resolved config header + summary)
#     <tag>.csv    per-instance results (--csv)
#     <tag>.meta   study, tag, exact command line, exit code, wall time
#
# The .meta file records the full command line rather than encoding parameters
# in the tag: analyze_sweep.py re-parses it into an option dict, so adding an
# option here needs no change on the analysis side.
#
# TWO THINGS THE LAYOUT ENCODES
#
# *Seeds.* Every configuration is run at each seed in $SEEDS (5 by default).
# `--seed` drives both the instance set (instance k is generated from seed+k,
# cw.c:2574) and the annealing RNG (cw.c:2679) -- cw has no separate solver
# seed -- so a seed is a fresh instance set solved with fresh randomness.
# Within one seed the comparison is exactly paired; analyze_sweep.py pools the
# per-instance deltas across seeds, so a study's conclusion rests on 5 x $M
# paired instances and cannot be an artefact of one instance draw.
#
# *Tiers.* A knob tuned at 10^5 annealing steps need not still be the right
# setting at 10^7, which is where this solver is actually run. Each tier fixes
# its own ($M, $STEPS); the tags are identical across tiers, so the analysis can
# put the same comparison at the two budgets side by side and show which
# recommendations move. Step counts that used to be absolute are now multiples
# of $STEPS, so every ladder rescales with the tier.
#
#     tier   instances   sa-steps   what it is
#     lo         1000        10^5   the historical sweep, kept for the contrast
#     hi          200        10^7   the operating point, + 3x10^7 on `tuned`
#
# $M is smaller at `hi` because the budget per run is 100x larger; pooled over
# the 5 seeds the `hi` tier still carries 1000 paired instances per comparison,
# the same as the headline pairing in the top-level README.
#
# Usage:
#     sweep/run_sweep.sh                 # everything: both tiers, all seeds
#     sweep/run_sweep.sh init ops        # only those studies
#     TIERS=hi sweep/run_sweep.sh        # only the operating-point tier
#     SEEDS="42 43" sweep/run_sweep.sh   # fewer seeds (quick check)
#     M=200 sweep/run_sweep.sh           # override the tier's instance count
#     RESUME=1 sweep/run_sweep.sh        # skip runs already on disk
#     sweep/run_sweep.sh --list          # list the studies
#     sweep/run_sweep.sh --plan          # count the runs, run nothing

set -u
shopt -s inherit_errexit 2>/dev/null || true

# awk's printf honours the locale, so under e.g. fr_FR the wall times land in
# the .meta files as "0,055" and the analysis cannot read them back. Everything
# written here is data, not user-facing text, so pin the numeric locale.
export LC_ALL=C

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-$ROOT/cw}
OUT=${OUT:-$ROOT/sweep/results}

# ---------------------------------------------------------------- defaults
SEEDS=${SEEDS:-"42 43 44 45 46"}   # >=5: the study is replicated on each
TIERS=${TIERS:-"lo hi"}            # which budget tiers to run
N=${N:-100}                        # default dimension
RESUME=${RESUME:-0}                # 1 = skip a run whose .meta already exists
PLAN=0

# Per-tier (instances, sa-steps). M/STEPS in the environment override both.
tier_M()     { case $1 in lo) echo 1000;; hi) echo 200;; esac; }
tier_STEPS() { case $1 in lo) echo 100000;; hi) echo 10000000;; esac; }

STUDIES=(init ops newops knn timing restarts split pick select race temp
         construct tuned)

n_run=0; n_skip=0; n_fail=0; n_plan=0
FAILED=()
T_START=$(date +%s)

usage() {
    echo "usage: $0 [--list] [--plan] [study ...]"
    echo "studies: ${STUDIES[*]}"
    echo "env: SEEDS='$SEEDS' TIERS='$TIERS' N=$N RESUME=$RESUME OUT=$OUT"
    echo "     M/STEPS override the per-tier defaults (lo: 1000/1e5, hi: 200/1e7)"
}

# mul <numerator> <denominator> -- a multiple of $STEPS, floored at 1.
# Every absolute step count in the studies below goes through this, which is
# what makes the whole sweep rescale when the tier changes.
mul() {
    local v=$(( STEPS * $1 / $2 ))
    [[ $v -lt 1 ]] && v=1
    echo "$v"
}

# run <study> <tag> <cw args...>
run() {
    local study=$1 tag=$2; shift 2
    if [[ $PLAN == 1 ]]; then n_plan=$((n_plan + 1)); return 0; fi
    local dir="$OUT/$TIER/s$SEED/$study"
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
        echo "tier=$TIER"
        echo "seed=$SEED"
        echo "cmd=${cmd[*]}"
        echo "exit=$rc"
        echo "wall_s=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b-a}')"
    } >"$base.meta"

    n_run=$((n_run + 1))
    if [[ $rc -ne 0 ]]; then
        n_fail=$((n_fail + 1)); FAILED+=("$TIER/s$SEED/$study/$tag")
        printf '  [FAIL rc=%d] %s/%s\n' "$rc" "$study" "$tag"
    else
        printf '  [%4d] %-2s s%-3s %-10s %-32s %7.2fs\n' \
               "$n_run" "$TIER" "$SEED" "$study" "$tag" \
               "$(awk -v a="$t0" -v b="$t1" 'BEGIN{print b-a}')"
    fi
    return 0
}

# common instance-source arguments for a given n
src() { echo "--random -n $1 -m $M --seed $SEED"; }

banner() { printf '\n=== %s [%s s%s]: %s\n' "$1" "$TIER" "$SEED" "$2"; }

# ============================================================== S1  init
# Does a random initial solution catch up with Clarke & Wright, and how fast?
# Same SA budget on both sides, budget swept over three decades below $STEPS.
# At `lo` the rungs are the historical 10^3..10^6; at `hi` they are 10^5..10^7,
# so in both tiers the ladder ends at the tier's own operating point.
study_init() {
    banner init "random vs C&W construction across the SA budget"
    local rungs n init r steps
    if [[ $TIER == lo ]]; then
        rungs=("1 100" "1 50" "1 20" "1 10" "1 5" "1 2" "1 1" "2 1" "5 1" "10 1")
    else
        rungs=("1 100" "1 50" "1 20" "1 10" "1 5" "1 2" "1 1")
    fi
    for n in 20 50 100 200; do
        for init in cw random; do
            for r in "${rungs[@]}"; do
                set -- $r
                steps=$(mul "$1" "$2")
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

# ============================================================== S2b newops
# The two operators added in positions 5 and 6 of --ops: swap* (Vidal 2022) and
# route opening. Both default to weight 0, so the baseline here is the stock
# --ops 1,1,1,0,0,0.
#
# swap* costs O(L1+L2) per draw instead of O(1), so an equal-STEP win is not a
# win. The `bud_` block is a budget ladder: the same four configurations swept
# over six step counts, so the analysis can read cost against *measured wall
# time* and interpolate an iso-time delta rather than trusting equal steps.
study_newops() {
    banner newops "swap* and route opening: weights, and the iso-time question"
    local X E cfg mult steps n
    for X in 0.25 0.5 1 2 4; do
        run newops "sstar_$X" $(src "$N") --sa-steps "$STEPS" --ops "1,1,1,0,$X,0"
    done
    for E in 0.01 0.02 0.05 0.1 0.3 1; do
        run newops "open_$E"  $(src "$N") --sa-steps "$STEPS" --ops "1,1,1,0,0,$E"
    done

    # combinations, including the "does swap* subsume swap" question. For a
    # given pair the in-place exchange is one of the positions swap* sweeps, so
    # delta(swap*) <= delta(swap) always -- but swap is ~10x cheaper per draw,
    # so whether to keep it is a weighting question, not a dominance one.
    for w in "1,1,1,0,1,0.05" "1,0,1,0,1,0.05" "1,0,1,0,1,0" \
             "1,1,1,1,1,0.05" "1,0,1,1,1,0.05" "0,0,1,0,1,0.05"; do
        run newops "mix_${w//,/-}" $(src "$N") --sa-steps "$STEPS" --ops "$w"
    done

    # Budget ladder for the iso-time comparison. The rungs are multiples of
    # $STEPS rather than absolute counts, so the x1 rung is always the same run
    # as the blocks above -- which is what makes it a valid baseline for them.
    for cfg in "def:1,1,1,0,0,0" "sstar:1,1,1,0,1,0" \
               "sstaropen:1,1,1,0,1,0.05" "dropswap:1,0,1,0,1,0.05"; do
        for mult in "x0125 1 8" "x025 1 4" "x05 1 2" \
                    "x1 1 1" "x2 2 1" "x4 4 1"; do
            set -- $mult                       # $1=tag  $2=numerator  $3=denom
            steps=$(mul "$2" "$3")
            run newops "bud_${cfg%%:*}_$1" \
                $(src "$N") --sa-steps "$steps" --ops "${cfg#*:}"
        done
    done

    # does the gain survive across dimension?
    for n in 20 50 100 200 500; do
        run newops "n${n}_off" $(src "$n") --sa-steps "$STEPS"
        run newops "n${n}_on"  $(src "$n") --sa-steps "$STEPS" --ops 1,1,1,0,1,0.05
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
#
# At `hi` the 10S rung would be 10^8 steps, which dominates the whole sweep for
# a result that is about the cost model rather than about quality; the timing
# study therefore stays on the `lo` step counts in both tiers (only $M moves).
study_timing() {
    banner timing "cost of construction vs annealing across n, and thread scaling"
    local n T S=100000
    for n in 20 50 100 200 500 1000; do
        run timing "n${n}_nosa"  $(src "$n") --no-sa
        run timing "n${n}_sa"    $(src "$n") --sa-steps "$S"
        run timing "n${n}_sa10"  $(src "$n") --sa-steps $((S * 10))
    done
    for T in 1 2 4 8 12 16 24; do
        run timing "threads_$T" $(src "$N") --sa-steps "$S" --threads "$T"
    done
}

# ============================================================== S5  restarts
# (a) raw gain at fixed per-restart budget -- R restarts cost R times more.
# (b) iso-budget: R * steps held constant. This is the only fair question,
#     "given a fixed number of SA steps, how should they be split?".
study_restarts() {
    banner restarts "restart count, at fixed and at equal total budget"
    local R S FIXED BUDGET A MODE
    FIXED=$(mul 1 10)          # per-restart budget for the (a) block
    BUDGET=$(mul 32 10)        # total budget held constant in the (b) block
    for R in 1 2 4 8 16 32; do
        run restarts "fixed_R${R}"  $(src "$N") --sa-steps "$FIXED" --restarts "$R"
    done

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
    S=$(mul 4 10)
    for MODE in off perturb param both; do
        run restarts "cwrand_$MODE" $(src "$N") --sa-steps "$S" --restarts 8 --cw-rand "$MODE"
    done
    for A in 0.01 0.03 0.1 0.3 0.9; do   # cw.c:2041 requires 0 <= alpha < 1
        run restarts "alpha_$A"     $(src "$N") --sa-steps "$S" --restarts 8 \
            --cw-rand perturb --cw-alpha "$A"
    done
}

# ============================================================== S6  split
# --split (where Split is applied) and --split-every (periodic Split during
# annealing) are independent code paths (cw.c:1440 vs 1640/1658), so the full
# grid including "off + every" is meaningful.
#
# --split-every is a period in steps, so its grid is scaled with the tier too:
# "every 1000 steps out of 10^5" and "every 1000 out of 10^7" are not the same
# experiment, and holding the *number of Splits* fixed is the comparable choice.
study_split() {
    banner split "--split mode x --split-every, and --split-tour"
    local MODE EV n TOUR E100 E1k E10k
    E100=$(mul 1 1000); E1k=$(mul 1 100); E10k=$(mul 1 10)
    for MODE in off cw end both; do
        for EV in 0 "$E100" "$E1k" "$E10k"; do
            run split "m${MODE}_e${EV}" $(src "$N") --sa-steps "$STEPS" \
                --split "$MODE" --split-every "$EV"
        done
    done
    for TOUR in routes sweep both; do
        run split "tour_$TOUR" $(src "$N") --sa-steps "$STEPS" \
            --split both --split-every "$E1k" --split-tour "$TOUR"
    done
    # more routes at larger n means more for Split to repartition
    for n in 200 500; do
        for MODE in off both; do
            for EV in 0 "$E1k"; do
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

# ============================================================== S7b select
# Biases on the SECOND vertex v, and the relocate insertion side. All three
# default to the neutral setting, so the baseline is the plain default run.
#
# --vrank is claimed to be *coupled* with --sa-knn: the longer list gives the
# reach, the rank bias restores the concentration, and either alone is supposed
# to hurt. That claim is only testable on the full grid, hence the cross.
study_select() {
    banner select "--vrank x --sa-knn, --pick2, --reloc-side"
    local V K T S n
    for V in 1 2 3 4; do
        for K in 10 20 30 50; do
            run select "vrank${V}_K${K}" $(src "$N") --sa-steps "$STEPS" \
                --vrank "$V" --sa-knn "$K"
        done
    done
    for T in 1 2 3 4 8; do
        run select "pick2_$T" $(src "$N") --sa-steps "$STEPS" --pick2 "$T"
    done
    # the insertion side, alone and with swap* active -- an informed side may
    # matter more or less depending on what else is reshaping the routes
    for S in coin long; do
        run select "side_${S}"       $(src "$N") --sa-steps "$STEPS" --reloc-side "$S"
        run select "side_${S}_sstar" $(src "$N") --sa-steps "$STEPS" \
            --reloc-side "$S" --ops 1,1,1,0,1,0.05
    done
    # and across dimension, since the kNN list length is n-dependent
    for n in 50 200 500; do
        run select "n${n}_plain"  $(src "$n") --sa-steps "$STEPS"
        run select "n${n}_vrank2" $(src "$n") --sa-steps "$STEPS" --vrank 2 --sa-knn 30
    done
}

# ============================================================== S7c race
# Budget redistribution between restarts, and the two-chain interleaving.
#
# Racing only means anything at equal TOTAL budget, so every run here holds
# restarts * sa-steps constant. --pair is a pure engineering change: without
# --race the trajectories are identical bit for bit, so the cost column must be
# flat and only the wall time may move. With --race it also changes the
# schedule, because racing makes the budget depend on how starts are grouped.
study_race() {
    banner race "--race margin x --race-at, and --pair interleaving"
    local BUDGET R S MARGIN AT n
    BUDGET=$(mul 4 1)
    R=10; S=$((BUDGET / R))
    for MARGIN in off 0.0005 0.002 0.01 0.05 0.2; do
        run race "margin_${MARGIN}" $(src "$N") --sa-steps "$S" --restarts "$R" \
            --race "$MARGIN"
    done
    for AT in 0.1 0.25 0.5 0.75; do
        run race "at_${AT}" $(src "$N") --sa-steps "$S" --restarts "$R" \
            --race 0.002 --race-at "$AT"
    done
    # racing needs starts to race: how does the gain scale with their number?
    for R in 2 4 8 16 32; do
        S=$((BUDGET / R))
        run race "R${R}_off"  $(src "$N") --sa-steps "$S" --restarts "$R" --race off
        run race "R${R}_on"   $(src "$N") --sa-steps "$S" --restarts "$R" --race 0.002
    done
    # --pair: cost must not move (no --race), wall time should, from n large
    # enough that a chain's state spills out of L1/L2
    for n in 100 500 1000 2000; do
        S=$((BUDGET / 8))
        run race "pair0_n${n}" $(src "$n") --sa-steps "$S" --restarts 8 --pair 0
        run race "pair1_n${n}" $(src "$n") --sa-steps "$S" --restarts 8 --pair 1
    done
}

# ============================================================== S8  temp
# Annealing schedule: initial acceptance target and the number of decades
# spanned by T. This is the knob most likely to move with the budget: the
# geometric ratio is (Tend/T0)^(1/(steps-1)), so the same number of decades is
# traversed 100x more slowly at the `hi` tier.
study_temp() {
    banner temp "--t-accept x --t-decades"
    local A D
    for A in 0.0001 0.001 0.01 0.1 0.5; do
        for D in 1 2 3 4 5 6; do
            run temp "a${A}_d${D}" $(src "$N") --sa-steps "$STEPS" \
                --t-accept "$A" --t-decades "$D"
        done
    done
}

# ============================================================== S9  construct
# Construction quality on its own (--no-sa) and after annealing. The interesting
# question is whether a better construction survives the SA, or whether the SA
# washes the difference out -- and that is exactly the question whose answer
# should depend on the budget, so it is worth having in both tiers.
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
#
# This is the "headline" study, so at the `hi` tier it also gets a 3x rung
# (3x10^7 steps) on the two configurations that matter -- enough to say whether
# the default-vs-tuned ordering is still moving at the top of the range.
study_tuned() {
    banner tuned "candidate combinations vs defaults, at equal budget"
    local n S8 S3
    S8=$((STEPS / 8))
    for n in 50 100 200; do
        run tuned "n${n}_default"  $(src "$n") --sa-steps "$STEPS"
        run tuned "n${n}_oropt"    $(src "$n") --sa-steps "$STEPS" --ops 1,1,1,1
        run tuned "n${n}_critrem"  $(src "$n") --sa-steps "$STEPS" --pick-crit rem
        run tuned "n${n}_split"    $(src "$n") --sa-steps "$STEPS" --split both \
            --split-every "$(mul 1 100)"
        run tuned "n${n}_all"      $(src "$n") --sa-steps "$STEPS" --ops 1,1,1,1 \
            --pick-crit rem --split both --split-every "$(mul 1 100)"
        run tuned "n${n}_all_r8"   $(src "$n") --sa-steps "$S8" --restarts 8 \
            --ops 1,1,1,1 --pick-crit rem --split both --split-every "$(mul 1 100)"

        # the new operators, alone and folded into the combination above
        run tuned "n${n}_newops"   $(src "$n") --sa-steps "$STEPS" --ops 1,1,1,0,1,0.05
        run tuned "n${n}_newall"   $(src "$n") --sa-steps "$STEPS" --ops 1,1,1,0,1,0.05 \
            --reloc-side long --split both --split-every "$(mul 1 100)"
        # multi-restart, where racing has something to redistribute
        run tuned "n${n}_r8"       $(src "$n") --sa-steps "$S8" --restarts 8
        run tuned "n${n}_newall_r8" $(src "$n") --sa-steps "$S8" --restarts 8 \
            --ops 1,1,1,0,1,0.05 --reloc-side long --race 0.002 \
            --split both --split-every "$(mul 1 100)"
    done

    # the 3x rung: is the default-vs-tuned ordering still moving at the top?
    if [[ $TIER == hi ]]; then
        S3=$(mul 3 1)
        for n in 100 200; do
            run tuned "n${n}_x3_default" $(src "$n") --sa-steps "$S3"
            run tuned "n${n}_x3_newall"  $(src "$n") --sa-steps "$S3" \
                --ops 1,1,1,0,1,0.05 --reloc-side long --split both \
                --split-every "$(mul 1 100)"
        done
    fi
}

# ==========================================================================
main() {
    local args=()
    for a in "$@"; do
        case $a in
            --list|-h|--help) usage; return 0 ;;
            --plan) PLAN=1 ;;
            *) args+=("$a") ;;
        esac
    done
    [[ -x $BIN ]] || { echo "$BIN not found or not executable: run \`make\` first" >&2; return 1; }

    local wanted=("${args[@]:-}")
    [[ ${#args[@]} -eq 0 ]] && wanted=("${STUDIES[@]}")

    mkdir -p "$OUT"
    printf 'binary   : %s\n' "$BIN"
    printf 'output   : %s\n' "$OUT"
    printf 'seeds    : %s   (each is a fresh instance set AND fresh solver RNG)\n' "$SEEDS"
    printf 'tiers    : %s\n' "$TIERS"
    printf 'studies  : %s\n' "${wanted[*]}"

    if [[ $PLAN == 0 ]]; then
        "$BIN" --help >"$OUT/cw_help.txt" 2>&1
        (cd "$ROOT" && git rev-parse HEAD 2>/dev/null) >"$OUT/git_head.txt" 2>/dev/null || true
        md5sum "$BIN" >"$OUT/binary.md5" 2>/dev/null || true
    fi

    local s
    for TIER in $TIERS; do
        M=${M_OVERRIDE:-$(tier_M "$TIER")}
        STEPS=${STEPS_OVERRIDE:-$(tier_STEPS "$TIER")}
        [[ -n $M && -n $STEPS ]] || { echo "unknown tier: $TIER (known: lo hi)" >&2; return 1; }
        printf '\n########## tier %s: m=%s sa-steps=%s\n' "$TIER" "$M" "$STEPS"
        for SEED in $SEEDS; do
            for s in "${wanted[@]}"; do
                if ! declare -F "study_$s" >/dev/null; then
                    echo "unknown study: $s (known: ${STUDIES[*]})" >&2
                    n_fail=$((n_fail+1)); continue
                fi
                "study_$s"
            done
        done
    done

    if [[ $PLAN == 1 ]]; then
        printf '\nplanned runs: %d\n' "$n_plan"
        return 0
    fi

    local elapsed=$(( $(date +%s) - T_START ))
    printf '\n--------------------------------------------------\n'
    printf 'runs: %d  skipped: %d  failed: %d  elapsed: %dh%02dm%02ds\n' \
           "$n_run" "$n_skip" "$n_fail" \
           $((elapsed / 3600)) $((elapsed % 3600 / 60)) $((elapsed % 60))
    if [[ ${#FAILED[@]} -gt 0 ]]; then
        printf 'failed runs:\n'; printf '  %s\n' "${FAILED[@]}"
    fi
    printf 'analyse with: uv run sweep/analyze_sweep.py\n'
    return 0
}

# M / STEPS in the environment override the per-tier defaults; captured before
# main() so the per-tier assignment does not clobber them.
M_OVERRIDE=${M:-}
STEPS_OVERRIDE=${STEPS:-}
main "$@"
