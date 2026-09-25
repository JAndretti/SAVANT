#!/bin/bash
# Build every solver ON A COMPUTE NODE of the partition used for the campaign:
# cw and the C++ baselines are compiled with -march=native, so a binary built on
# a login node of a different CPU generation could be slower or crash.
#
# Before submitting (login node, it has internet):
#   - copy the repository to $WORK/SAVANT_pub (external_code/ included: sources,
#     cobra, jdk21, xml_generator), without the build*/ directories;
#   - install uv in $WORK and run `uv sync` there (UV_CACHE_DIR=$WORK/.cache/uv);
#   - submit with `sbatch -A <project>@cpu`, and adapt the module names (`module avail gcc cmake`).
#SBATCH --job-name=sv-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/build_%j.out

set -euo pipefail
cd "$WORK/SAVANT_pub"
module purge
module load gcc cmake   # ADAPT: a GCC >= 11 with C++17/20 support, CMake >= 3.16
lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket'

E=$PWD/external_code
make cw repro
# FILO, FILO2 and LKH-3 sources carry a small output-only patch that prints
# 'TRACE <seconds> <cost>' on each new best (grep "SAVANT benchmark patch").

( cd $E/HGS-CVRP-main && rm -rf build && mkdir build && cd build \
  && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8 bin )

( cd $E/cobra-master && rm -rf build && mkdir build && cd build \
  && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$E/prefix && make -j8 install )
( cd $E/filo-master && rm -rf build_tl && mkdir build_tl && cd build_tl \
  && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$E/prefix -DTIMEBASED_TERMINATION=ON \
  && make -j8 )
# FILO2's time-limited build only compiles with verbose output on (upstream bug)
( cd $E/filo2-main && rm -rf build_tl && mkdir build_tl && cd build_tl \
  && cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_TIMELIMIT=ON -DENABLE_VERBOSE=ON && make -j8 )

( cd $E/LKH-3.0.14 && make clean && make -j8 )

# AILS-II: already a portable jar (patched to print its best solution)
test -f $E/AILS-CVRP-main/AILSII.jar

# smoke test: one pinned run of every solver, solutions re-validated
.venv/bin/python -m bench.run_savant --set data/cvrplib/X --stride 50 --time 3 --tag smoke \
    --out results/jz/smoke.csv --jobs 2
for s in hgs filo filo2 ails lkh ortools; do
  .venv/bin/python -m bench.run_baselines --solver $s --set data/cvrplib/X --stride 50 --time 3 \
      --tag smoke --out results/jz/smoke.csv --jobs 2
done

# convergence traces: every run must have one that ends on its checked final cost
.venv/bin/python - <<'EOF'
import collections, json
ok, bad = collections.Counter(), []
for line in open("results/jz/smoke.trace.jsonl"):
    r = json.loads(line)
    ok[r["solver"], r["trace_ok"]] += 1
    if not r["trace_ok"]:
        bad.append((r["solver"], r["instance"], r["seed"], r["trace_msg"], r["trace"][-3:]))
for (solver, good), k in sorted(ok.items()):
    print(f"trace {solver:8s} {'ok ' if good else 'BAD'} {k}")
for b in bad:
    print("  ", *b)
EOF
