#!/bin/bash
# Build the archive to copy to Jean Zay: code, solver sources (no build directories),
# the CVRPLIB test sets and a subset of the uniform sets. Everything else (tuning
# instances, XML100 outside its 100-instance test set, our results, the upstream results/instances shipped with FILO
# and FILO2, the local .venv and binaries) stays here: the parameter study is done
# and the cluster only runs the comparison.
#
#   bash scripts/jz_bundle.sh [OUT.tar.zst] [N_UNIFORM]
#
# N_UNIFORM (default 100) is the number of instances kept per uniform size.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${1:-/tmp/savant_jz.tar.zst}
NUNI=${2:-100}
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
R=$STAGE/SAVANT_pub
mkdir -p "$R"

# --- our code -------------------------------------------------------------
cp -r SAVANT bench slurm scripts Makefile pyproject.toml uv.lock .python-version roadmap.md "$R"/
find "$R" -name '__pycache__' -prune -exec rm -rf {} +
mkdir -p "$R/results/jz" "$R/slurm/logs"

# --- solver sources (patched; build directories are rebuilt on a compute node) ---
mkdir -p "$R/external_code"
for d in HGS-CVRP-main cobra-master filo-master filo2-main LKH-3.0.14 AILS-CVRP-main jdk21 patches; do
  rsync -a --exclude 'build' --exclude 'build_tl' --exclude 'results' --exclude 'instances' \
        --exclude 'imgs' --exclude 'images' --exclude 'docs' --exclude '*.o' \
        "external_code/$d" "$R/external_code/"
done
test -x "$R/external_code/jdk21/bin/java"
# NDS (GPU): code, CVRP eval configs and the four CVRP checkpoints only -- not its own
# data (324 MB), the VRPTW/PCVRP models, .git, or a locally compiled C++ module (it is
# rebuilt on an A100 node by slurm/nds_smoke.sbatch). Its uv environment is synced on
# the login node from external_code/nds_env.
mkdir -p "$R/external_code/NDS/models"
rsync -a --exclude '*.so' --exclude '__pycache__' external_code/NDS/src external_code/NDS/eval.py \
      external_code/NDS/LICENSE external_code/NDS/README.md "$R/external_code/NDS/"
mkdir -p "$R/external_code/NDS/configs/eval"
cp external_code/NDS/configs/eval/cvrp_*.yaml "$R/external_code/NDS/configs/eval/"
rsync -a external_code/NDS/models/cvrp_{100,500,1000,2000} "$R/external_code/NDS/models/"
mkdir -p "$R/external_code/nds_env"
cp external_code/nds_env/pyproject.toml external_code/nds_env/uv.lock "$R/external_code/nds_env/"

# --- test instances -------------------------------------------------------
mkdir -p "$R/data/cvrplib"
for f in X XL XML100_test; do
  rsync -a "data/cvrplib/$f" "$R/data/cvrplib/"
  [ -f "data/cvrplib/${f}_bks.csv" ] && cp "data/cvrplib/${f}_bks.csv" "$R/data/cvrplib/"
  [ -f "data/cvrplib/${f}_invalid_ref.txt" ] && cp "data/cvrplib/${f}_invalid_ref.txt" "$R/data/cvrplib/"
done
cp data/cvrplib/XML100_bks.csv "$R/data/cvrplib/"   # optima of XML100 (reference of XML100_test)
# uniform sets: only the first N of each size (they hold 10,000 files each)
for d in data/nds_vrp/n100 data/nds_vrp/n500 data/nds_vrp/n1000 data/nds_vrp/n2000 data/neuopt/vrp_100; do
  mkdir -p "$R/$d"
  mapfile -t files < <(printf '%s\n' "$d"/*.vrp | sort)   # no `head`: it would SIGPIPE under pipefail
  cp "${files[@]:0:$NUNI}" "$R/$d/"
done

tar --zstd -cf "$OUT" -C "$STAGE" SAVANT_pub
printf 'wrote %s (%s, %s files)\n' "$OUT" "$(du -h "$OUT" | cut -f1)" "$(find "$R" -type f | wc -l)"
sha256sum "$OUT"
