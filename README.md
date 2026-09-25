# SAVANT — savings, simulated annealing and truncated neighbourhoods for the CVRP

Code, benchmark harness and experimental protocol of

> J. Andretti, Y. Strozecki, J. Cabessa. *SAVANT: savings, simulated annealing and
> truncated neighbourhoods for the capacitated vehicle routing problem.* Submitted to
> Computers & Operations Research, 2026.

SAVANT builds an initial solution with the parallel savings heuristic (Clarke & Wright)
and improves it by simulated annealing over six neighbourhoods restricted to truncated
candidate lists, including a sampled SWAP\* and a periodic ruin-and-recreate move, with a
few restarts separated by an optimal Split. It is a single C file with no dependency.

## Repository layout

| Path | Content |
|---|---|
| `SAVANT/cw.c` | the solver |
| `Makefile` | builds `cw` (fast) and `cw_repro` (bit-reproducible) |
| `bench/` | benchmark harness: runners for every solver, independent solution checker, parameter study, analyses, tables and figures of the paper |
| `slurm/` | the Slurm jobs of the comparison on the Jean Zay supercomputer, and `JEANZAY.md`, the full procedure |
| `scripts/` | generation of the campaign files and of the archive copied to the cluster |
| `external_code/patches/` | our output-only patches to the baselines (they are not distributed here, see below) |
| `external_code/nds_env/` | the Python environment used to run NDS |
| `data/tuning/` | the tuning instances we generated (uniform `gen*`, X-like `xlike`) and the names of the 378 XML100 tuning instances (`xml100.txt`) |
| `data/cvrplib/` | the best known solutions used as reference (snapshot of 15 September 2026), the reference solutions we found invalid, and the names of the 100 XML100 test instances |

## Build and run SAVANT

```bash
make            # cw: gcc -O3 -march=native with OpenMP (one instance per thread)
make repro      # cw_repro: no FMA contraction, bit-identical results across builds
make auto       # picks what the machine supports (macOS: Homebrew libomp, or serial)
./cw --help
```

Quick start on the X instances of CVRPLIB: download them, then solve every instance for
60 seconds with the configuration of the paper (it is the compiled default) and integer
(TSPLIB) distances:

```bash
# the 100 X instances, as a .7z archive of .vrp and reference .sol files -> data/cvrplib/X/
curl -L -o X.7z https://galgos.inf.puc-rio.br/cvrplib/en/download/instance-set/17
uvx py7zr x X.7z data/cvrplib          # or: 7z x X.7z -odata/cvrplib

./cw --dir data/cvrplib/X --round --sa-wall 60 --sol X.sol
```

Instances are solved in parallel, one per core, so the set takes about 100 x 60 s divided by
the number of cores. To use the time limit of the paper instead (2.4 x n seconds, growing
with the instance) and get every solution re-checked and its gap to the best known solution:
`uv sync`, then
`uv run python -m bench.run_savant --set data/cvrplib/X --time-hgs 1 --out results/X.csv`.

**Always give a budget.** Without `--sa-wall` or `--sa-steps`, the annealing runs only 1,000
steps per restart, far too few to be useful. `--sa-wall` is a wall-clock budget per instance
(the paper uses 2.4 × n seconds, i.e. 240 s for 100 customers); `--sa-steps` fixes the number
of annealing steps instead (per restart), which is what the parameter study uses. `--round`
is for the CVRPLIB sets (X, XL, XML100); leave it out for real-valued instances. `--sol`
writes the routes, `--seed` changes the random seed (default 42), and several instances in a
directory are solved in parallel, one per core (`--threads 1` to solve them one at a time).

No other option is needed: the configuration selected in the paper is compiled in as the
default --- savings construction, simulated annealing with candidate lists of the 20 nearest
customers (`--sa-knn 20`), uniform selection of the vertices a move acts upon
(`--pick 1 --pick2 1`), 4 restarts separated by an optimal Split (`--restarts 4`), and a
ruin-and-recreate move every 50 steps removing up to 20 customers (`--kick 50 --kick-max 20`).

## Reproducing the experiments

Python is managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync                      # numpy, pandas, scipy, ortools
uv sync --extra viz          # + matplotlib, for the figures only
```

### Instances

Not redistributed here; place them as follows.

- **X, XL, XML100** from [CVRPLIB](https://galgos.inf.puc-rio.br/cvrplib/) (`.vrp` and
  reference `.sol` files) into `data/cvrplib/X/`, `data/cvrplib/XL/`, `data/cvrplib/XML100/`.
  `uv run python -m bench.check_bks X XL XML100` re-checks every reference solution.
  The two XML100 splits used in the paper are fixed by the name lists in this repository:
  ```bash
  mkdir -p data/tuning/xml100 data/cvrplib/XML100_test
  while read f; do cp data/cvrplib/XML100/$f.vrp data/tuning/xml100/; done < data/tuning/xml100.txt
  while read f; do cp data/cvrplib/XML100/$f.vrp data/cvrplib/XML100_test/; done < data/cvrplib/XML100_test.txt
  ```
- **Uniform test sets** of NDS (Hottung et al. 2025): copy the four `*_test_*.pkl` files of
  `data/cvrp/` in [ahottung/NDS](https://github.com/ahottung/NDS) into `data/nds/`, then
  `uv run python -m bench.convert_nds` writes them as `.vrp` files to `data/nds_vrp/`. The first
  100 instances of each size are used.

### Baselines

The comparison runs the public source code of each solver, changed only where output was
needed. Download each one into `external_code/` under the directory name below, apply our
patch from `external_code/` with `patch -p1 < patches/<file>`, and build it as in
`slurm/build_all.sbatch`.

| Solver | Source (downloaded September 2026) | Directory | Our patch |
|---|---|---|---|
| HGS-CVRP | [vidalt/HGS-CVRP](https://github.com/vidalt/HGS-CVRP), `main` | `HGS-CVRP-main` | `hgs-trace-precision.diff` (full precision in the convergence file) |
| FILO | [acco93/filo](https://github.com/acco93/filo), `master`, with [acco93/cobra](https://github.com/acco93/cobra) | `filo-master`, `cobra-master` | `filo-trace.diff` (prints each new best) |
| FILO2 | [acco93/filo2](https://github.com/acco93/filo2), `main` | `filo2-main` | `filo2-trace.diff` (prints each new best) |
| AILS-II | [vinymax10/AILS-CVRP](https://github.com/vinymax10/AILS-CVRP), `main` (e631331) | `AILS-CVRP-main` | `ails-best-solution.diff` (prints its best solution) |
| LKH-3 | [LKH-3.0.14](http://webhotel4.ruc.dk/~keld/research/LKH-3/) | `LKH-3.0.14` | `lkh3-trace.diff` (prints each new best) |
| OR-Tools | PyPI `ortools`, installed by `uv sync` | — | none (`bench/ortools_cvrp.py`) |
| NDS | [ahottung/NDS](https://github.com/ahottung/NDS), commit `ca37420`, pretrained CVRP models | `NDS` | none (`bench/nds_cvrp.py`) |

AILS-II needs a Java 21 runtime (we used a portable Eclipse Adoptium JDK in
`external_code/jdk21/`); after patching, rebuild its jar with
`javac -d classes $(find src -name '*.java') && jar cfe AILSII.jar SearchMethod.AILSII -C classes .`.
NDS runs on a GPU in its own environment: `uv sync --project external_code/nds_env`.

Each solver keeps its own licence; in particular LKH-3 may not be redistributed, which is
why none of them is included here.

### Runs

Every runner schedules one single-threaded run per physical core, pinned with `taskset`,
re-validates every solution independently of the solver (coverage, capacities, cost
recomputed from the coordinates) and records the convergence trace of timed runs:

```bash
# SAVANT and a baseline on X, T_max = 2.4 n seconds (Vidal 2022), 10 seeds
uv run python -m bench.run_savant    --set data/cvrplib/X --time-hgs 1 --seeds 10 --tag tmax --out results/X/savant.csv
uv run python -m bench.run_baselines --solver hgs --set data/cvrplib/X --time-hgs 1 --seeds 10 --tag tmax --out results/X/hgs.csv

# analysis: gaps, paired Wilcoxon tests with Holm's correction, convergence checkpoints
uv run python -m bench.analyze_compare --set results/X
```

The comparison of the paper ran on the Jean Zay supercomputer (IDRIS): CPU solvers on
nodes of 2 x Intel Xeon Gold 6248, 40 concurrent runs per node, NDS on NVIDIA A100 GPUs,
one run per GPU. `slurm/JEANZAY.md` gives the full procedure and `bench/slurm_campaign.py`
generates the job files. The parameter study (`bench/sweep.py`, `bench/time_match.py`,
`bench/interact.py`, `bench/ablate.py` and their `analyze_*` counterparts) ran on a
single workstation. Each module documents its own usage in its docstring.

## Results

The raw results of the paper (one line per run with its re-evaluated cost, and the
convergence trace of every timed run) are archived at: **TODO: Zenodo DOI**.

## Licence

MIT, for the files of this repository (see `LICENSE`). The baselines are not part of it.
