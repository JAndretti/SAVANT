#!/bin/bash
# Generate every sbatch file of the comparison, with the plan agreed on 2026-09-18:
# X with 10 seeds (7 solvers), XL with 5 seeds (6 solvers, no OR-Tools), and the
# uniform sets with 5 seeds (4 solvers: FILO, FILO2 and LKH-3 round arc costs).
#
#   bash scripts/jz_make_campaigns.sh [JOBS_PER_NODE]
#
# JOBS_PER_NODE comes from the density test (slurm/density_test.sbatch): 40 if the
# contention is negligible, otherwise 20 -- which doubles the billing, since an
# exclusive node bills its 40 cores whatever it runs.
#
# Shard counts are chosen to keep the nodes full: a shard bills 40 cores for its
# whole duration, so few long shards cost far less than many short ones (20 shards
# on XL would bill ~6600 core-hours per solver instead of ~1400).
set -euo pipefail
cd "$(dirname "$0")/.."

J=${1:-40}
PY=${PY:-uv run python}
SAV='--pick 1 --pick2 1 --restarts 4 --kick 50 --kick-max 20'
common="--time-hgs 1 --tag tmax --cores 40 --jobs $J --max-hours 24"

gen() {  # gen <solver> <instance dir> <seeds> <shards> [extra args]
  local solver=$1 dir=$2 seeds=$3 shards=$4; shift 4
  local extra=()
  [ "$solver" = savant ] && extra=(--extra "$SAV")
  $PY -m bench.slurm_campaign --solver "$solver" --set "$dir" --seeds "$seeds" \
      --shards "$shards" $common "${extra[@]}" "$@"
}

for s in savant hgs filo filo2 ails lkh ortools; do gen $s data/cvrplib/X 10 2; done
for s in savant hgs filo filo2 ails lkh;         do gen $s data/cvrplib/XL 5 3; done
for d in data/nds_vrp/n100 data/nds_vrp/n500 data/nds_vrp/n1000 data/neuopt/vrp_100; do
  for s in savant hgs ails ortools; do gen $s "$d" 5 1 --limit 100; done
done
for s in savant hgs ails ortools; do gen $s data/nds_vrp/n2000 5 2 --limit 100; done
