#!/bin/bash
# Build and smoke-test NDS on ONE A100 before the NDS campaign (tmax_nds_*.sbatch).
#
# Before submitting (login node, it has internet):
#   uv sync --project external_code/nds_env      (torch cu128, numpy<2.4, pybind11<3, ...)
#
# 1. cppimport compiles NDS's C++ operators (external_code/NDS/src/cpp/cvrp/*.so) on the
#    first import. Doing it here, once, keeps the campaign's array tasks from racing to
#    compile the same module.
# 2. Harness check (not a comparison): 2 XML100 instances x 2 seeds and 1 instance of
#    n = 2000 (largest model, GPU memory) at 10 s. Expect "all solutions valid" and a
#    `trace_ok: true` for every line of results/jz/smoke/nds_smoke.trace.jsonl.
#SBATCH --job-name=sv-nds-smoke
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/nds_smoke_%j.out

set -euo pipefail
cd "$WORK/SAVANT_pub"
module purge
module load gcc   # ADAPT if needed: cppimport compiles C++ with the default g++
export TMPDIR=${JOBSCRATCH:-/tmp}
export OMP_NUM_THREADS=1
lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket'
nvidia-smi -L

PY=external_code/nds_env/.venv/bin/python
( cd external_code/NDS && ../../$PY -W ignore::DeprecationWarning -c "
import sys, types; sys.modules['wandb'] = types.ModuleType('wandb')
from src.cpp.cvrp import NDSOps; print('NDS C++ operators built:', NDSOps.__file__)" )
$PY -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

mkdir -p results/jz/smoke
rm -f results/jz/smoke/nds_smoke.csv results/jz/smoke/nds_smoke.trace.jsonl
.venv/bin/python -m bench.run_baselines --solver nds --set data/cvrplib/XML100_test \
    --limit 2 --seeds 2 --time 10 --jobs 1 --tag smoke --out results/jz/smoke/nds_smoke.csv
.venv/bin/python -m bench.run_baselines --solver nds --set data/nds_vrp/n2000 \
    --limit 1 --seeds 1 --time 10 --jobs 1 --tag smoke --out results/jz/smoke/nds_smoke.csv
cut -d, -f4,6,9,11,13,15,16 results/jz/smoke/nds_smoke.csv
grep -o '"trace_ok": [a-z]*' results/jz/smoke/nds_smoke.trace.jsonl | sort | uniq -c
