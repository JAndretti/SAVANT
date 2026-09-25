# Running the comparison on Jean Zay (IDRIS)

Everything below is run by hand on the cluster; the local machine only builds the
archive. Partition `cpu_p1` (default): 2 × Intel Xeon Gold 6248
(40 physical cores), 192 GB per node, jobs ≤ 100 h. Compute nodes have **no internet**.

---


> **Accounts.** The job files carry no `#SBATCH --account` line: give your project at
> submission, `sbatch -A <project>@cpu <file>` for the CPU jobs and
> `sbatch -A <project>@a100 <file>` for the GPU jobs (NDS), including in the loops below.

## 1. Build the archive (local machine)

```bash
bash scripts/jz_bundle.sh /tmp/savant_jz.tar.zst 100
```

213 MB, ~2100 files: our code, the solver sources with our patches (no build
directories), the X and XL sets, and 100 instances of each uniform size. Left behind:
the tuning instances and XML100 (the parameter study is finished), our results, the
upstream results and instance collections of FILO/FILO2, the local `.venv` and binaries.

`100` is the number of uniform instances kept per size; raise it only if the CPU-hour
allocation allows (a single run at n = 2000 already costs 4800 s).

## 2. Copy it

```bash
# the remote shell does NOT expand $WORK in an rsync target: use the absolute path
rsync -avP /tmp/savant_jz.tar.zst JeanZay:<absolute path of your $WORK>/
ssh JeanZay
idr_quota_user                     # space and inodes ($WORK, not HOME: HOME is full)
cd $WORK && tar --zstd -xf savant_jz.tar.zst && cd SAVANT_pub
```

`$WORK` holds the code, the instances and the results; `$JOBSCRATCH` (per job) is used
for the solvers' temporary files, which the sbatch scripts already set through `TMPDIR`.

## 3. Python environment (login node: it has internet, compute nodes do not)

HOME is at its 3 GB quota, so uv, its cache and its Python must live on `$WORK`:

```bash
export UV_INSTALL_DIR=$WORK/bin UV_CACHE_DIR=$WORK/.cache/uv UV_PYTHON_INSTALL_DIR=$WORK/.uv/python
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$WORK/bin:$PATH                        # add these four lines to ~/.bashrc
uv sync                                            # creates $WORK/SAVANT_pub/.venv
.venv/bin/python -c "import numpy, pandas, scipy, ortools; print('deps ok')"
```

The sbatch scripts call `.venv/bin/python` directly, so nothing else is needed at run
time. If `ortools` cannot be installed, everything else still works: run the campaigns
for the other solvers and drop OR-Tools from the comparison.

## 4. Build the solvers on a compute node

```bash
sbatch -A <project>@cpu slurm/build_all.sbatch
squeue -u $USER
tail -f slurm/logs/build_<jobid>.out
```

It compiles cw, HGS-CVRP, cobra + FILO, FILO2, LKH-3 with `-march=native` on the CPU
that will run the campaign, then runs one 3-second run of every solver and prints a
trace summary per solver. **Check before going further**: no compilation error, the
smoke runs are valid, and the trace lines say `ok` (LKH-3 may fail at 3 s, where its
solution is genuinely infeasible — that is the solver, not the harness).
Adapt `module load gcc cmake` to the module names available (`module avail gcc cmake`).

## 5. Choose how many runs per node (blocking, item B2)

```bash
sbatch -A <project>@cpu slurm/density_test.sbatch
```

Same fixed step budget at 40, 20 and 1 concurrent runs on one node, on X and XL: since
the work is identical, the run time measures the contention. Then:

```bash
.venv/bin/python - <<'EOF'
import pandas as pd
d = pd.read_csv("results/jz/density.csv")
d["set"] = d.tag.str.split("_").str[1]; d["jobs"] = d.tag.str.split("_j").str[1].astype(int)
t = d.pivot_table(index=["set", "instance", "seed"], columns="jobs", values="time_s")
t = t.dropna()
print((t.div(t[1], axis=0)).groupby(level=0).mean().round(3))   # slowdown vs one run alone
EOF
```

A slowdown below ~5 % at 40 runs → keep 40 per node. Otherwise regenerate the campaigns
with `--jobs 20` (twice the node-hours). **Report the measured figure**: it goes in the
paper's protocol.

## 6. Generate the campaign files (plan of 2026-09-18)

```bash
bash scripts/jz_make_campaigns.sh 40      # or 20, from the density test
```

writes the 33 sbatch files of the agreed plan:

| Campaign | Solvers | Seeds | Nodes × wall | Time limits |
|---|---|---|---|---|
| X (100 inst.) | 7 | 10 | 2 × 4.5 h | 1,924 core-h |
| XL (100 inst.) | 6 (no OR-Tools) | 5 | 3 × 15 h | 8,325 core-h |
| NDS n = 100, 500, 1000, 2000 + NeuOpt n = 100 (100 inst. each) | 4 | 5 | 1–2 × 1.3–11.5 h | 4,930 core-h |

**15,180 core-hours in total**, (check what your allocation allows with `idracct`). Re-run the script with `20` if the density test demands it: the billing then
doubles, since an exclusive node bills its 40 cores whatever it runs.

**Why so few shards.** A shard holds a whole node for its whole life, so a campaign split
into many short shards bills mostly idle cores: XL over 20 shards would cost ~6,600
core-hours per solver instead of ~1,400. The counts above keep every node full; the price
is a longer wall-clock time per job (up to 15 h on XL, well under the 100 h limit).

## 7. Submit, in this order

```bash
for f in slurm/tmax_*_X.sbatch; do sbatch -A <project>@cpu $f; done       # X first: 1,924 core-h
squeue -u $USER
```

**Stop there and check X before going on.** The gaps of HGS-CVRP and FILO on X should be
close to their published values; if they are not, something is wrong with the protocol and
the 8,325 core-hours of XL would be wasted.

```bash
for s in savant hgs filo filo2 ails lkh ortools; do
  .venv/bin/python -m bench.merge_csv results/jz/X/tmax_${s}_X_shard*.csv \
      -o results/jz/X/tmax_${s}_X.csv
done
grep -c INVALID slurm/logs/tmax_*_X_*.out      # 0 everywhere is what we want
```

Then XL, then the uniform sets:

```bash
for f in slurm/tmax_*_XL.sbatch; do sbatch -A <project>@cpu $f; done
for f in slurm/tmax_*_n100.sbatch slurm/tmax_*_n500.sbatch slurm/tmax_*_n1000.sbatch \
         slurm/tmax_*_n2000.sbatch slurm/tmax_*_vrp_100.sbatch; do sbatch -A <project>@cpu $f; done
```

## 8. Watching a campaign

```bash
squeue -u $USER                       # PD pending, R running
tail -f slurm/logs/tmax_savant_X_0.out
idr_compuse                           # hours consumed so far
```

A job that hits its Slurm time limit only loses the runs still in flight: the CSV is
resumable, so re-submitting the same file finishes the rest.

## 9. Bring the results back

```bash
# on the login node
tar -czf results_jz.tar.gz results/jz slurm/logs
# locally
rsync -avP JeanZay:<absolute path of your $WORK>/SAVANT_pub/results_jz.tar.gz .
```

The analysis, tables and convergence figures are then produced on the local machine.

---

### Watch out for

- **`--hint=nomultithread` and `--exclusive`** are already in the generated files: one
  run per physical core, no hyperthreading, no other job on the node.
- **Time limits**: `--time` is estimated from the budgets plus 15 %. A shard that hits
  the wall clock loses its unfinished runs only — rerunning the same sbatch resumes,
  since the CSV is resumable.
- **`$JOBSCRATCH` is wiped** when a job ends; results are written directly to `$WORK`.
- **Never run a campaign on a login node.**
- **Keep the build log**: it records the CPU model and the module versions, which the
  paper reports.

## 10. Uniform sets, XML100 and NDS (plan of 2026-09-22/23)

SAVANT and HGS-CVRP on the four uniform sets and XML100_test, on cpu_p1 as before; NDS on
the uniform sets only, on the A100 partition, one run per GPU. **Do not
submit** the old `tmax_{ails,ortools}_*` or `tmax_*_vrp_100` files (dropped from the plan).

The cluster copy predates NDS and XML100_test. Either rebuild and re-extract the archive
(`scripts/jz_bundle.sh` now ships them), or copy only what is new:

```bash
bash scripts/jz_bundle.sh /tmp/savant_jz.tar.zst 100      # then on Jean Zay, in $WORK:
tar --zstd -xf savant_jz.tar.zst SAVANT_pub/external_code/NDS SAVANT_pub/external_code/nds_env \
    SAVANT_pub/data/cvrplib/XML100_test SAVANT_pub/data/cvrplib/XML100_bks.csv \
    SAVANT_pub/bench SAVANT_pub/slurm SAVANT_pub/scripts
```

On the login node (internet), then the A100 smoke test, **then check its log**:

```bash
uv sync --project external_code/nds_env        # torch cu128 + numpy<2.4, pybind11<3 (see roadmap)
sbatch -A <project>@a100 slurm/nds_smoke.sbatch                  # 30 min max, qos_gpu_a100-dev
# log: "NDS C++ operators built", torch ... True NVIDIA A100..., "all solutions valid",
#      every line trace_ok: true
```

Then the campaigns. NDS: **1 seed x 100 instances per uniform set, not run on XML100**
(to fit the GPU allocation). Every NDS file is an array of one-GPU tasks under 18 h
(A100 jobs are limited to 20 h); A100 nodes are shared and billed per GPU:

| Set | NDS shards | NDS GPU-h (budget) | SAVANT/HGS files (5 seeds; XML100 10) |
|---|---|---|---|
| n100 | 1 | 7 | `tmax_{savant,hgs}_n100` |
| n500 | 3 | 33 | `tmax_{savant,hgs}_n500` |
| n1000 | 5 | 67 | `tmax_{savant,hgs}_n1000` |
| n2000 | 10 | 133 | `tmax_{savant,hgs}_n2000` |
| XML100_test | -- | -- | `tmax_{savant,hgs}_XML100_test` |

**~240 GPU-hours** for NDS and ~2,530 core-hours for SAVANT + HGS-CVRP.

```bash
for d in n100 n500 n1000 n2000; do sbatch -A <project>@a100 slurm/tmax_nds_${d}.sbatch; done
for s in savant hgs; do
  for d in n100 n500 n1000 n2000 XML100_test; do sbatch -A <project>@cpu slurm/tmax_${s}_${d}.sbatch; done
done
```

Regenerate the NDS files with `bench.slurm_campaign --solver nds --gpus 1 ...`
(docstring); `--gpus` sets `-C a100`, `--gres=gpu:N`, 8 cores per GPU and one run per GPU.
