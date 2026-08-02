# `sweep/` — parameter study for `cw`

Two scripts. One runs every configuration, the other turns the output into
figures and a report.

```sh
make                        # build ./cw first
sweep/run_sweep.sh          # ~2 min, 346 runs, writes sweep/results/
uv run sweep/analyze_sweep.py   # writes sweep/figures/, report.tex, report.pdf
```

Read **`report.pdf`** for the results. It explains what each option controls,
embeds the figures from `figures/`, gives the full paired statistics, and ends
with the command for the best configuration found.

`report.tex` is generated — do not edit it by hand, edit `analyze_sweep.py`. It
is compiled automatically when `pdflatex` is on the path; `--no-pdf` emits the
`.tex` only.

## Running a subset

```sh
sweep/run_sweep.sh --list             # the study names
sweep/run_sweep.sh pick knn           # only those two
RESUME=1 sweep/run_sweep.sh           # skip runs already on disk
uv run sweep/analyze_sweep.py --no-verify   # skip the confirmation re-runs
M=100 STEPS=10000 sweep/run_sweep.sh  # quick pass (~15 s), for checking changes
SEED=7 OUT=sweep/results_seed7 sweep/run_sweep.sh   # independent replication
```

| variable | default | meaning |
|---|---|---|
| `M` | 1000 | instances per run |
| `SEED` | 42 | instance seed |
| `N` | 100 | dimension for the studies that fix one |
| `STEPS` | 100000 | SA budget for the studies that fix one |
| `RESUME` | 0 | 1 = skip runs whose `.meta` records `exit=0` |
| `OUT` | `sweep/results` | output root |
| `BIN` | `./cw` | binary to run |

`sweep/results/` is already covered by the repo's `.gitignore` (`results/`
matches at any depth); it is ~22 MB per sweep and is fully regenerable.

## Layout

Each run writes three files to `results/<study>/<tag>.*`:

- `.log` — cw's stdout: the resolved-config header and the summary block
- `.csv` — `--csv`, one row per instance
- `.meta` — study, tag, **the exact command line**, exit code, shell wall time

Parameters are recorded as the command line rather than encoded in the tag:
`analyze_sweep.py` re-parses it into an option dict against a defaults table, so
adding an option to a study needs no change on the analysis side.

## The studies

| name | question |
|---|---|
| `init` | `--init random` vs `cw`, budget 10³→10⁶, at n = 20/50/100/200 |
| `ops` | all 15 non-empty operator subsets, unbalanced weights, `--or-max` |
| `knn` | `--sa-knn` ∈ {0,5,10,20,30,50} × n ∈ {20,50,100,200} |
| `timing` | construction vs annealing cost across n; thread scaling |
| `restarts` | `--restarts` at fixed and at **equal total budget**; `--cw-rand`, `--cw-alpha` |
| `split` | `--split` × `--split-every` grid, `--split-tour`, and the same at larger n |
| `pick` | `--pick` × `--pick-crit`, `--pick-eps`, and the `--pick` × `--sa-knn` interaction |
| `temp` | `--t-accept` × `--t-decades` |
| `construct` | `--lambda` × `--mu` with and without SA; `--knn`/`--exact`; `--2opt` |
| `tuned` | candidate combinations against the stock defaults, at equal budget |

## Why the numbers are trustworthy

**Comparisons are paired.** Instance *k* is generated from `seed + k`
(`cw.c:2064`), so every run of a study sees byte-identical instances. The
analysis compares `cost_i(A)` with `cost_i(B)` on the same instance and reports
the mean of the 1000 differences with a 95 % CI, plus a distribution-free sign
test. Comparing two means with their independent standard deviations would be
far looser — the between-instance spread (σ ≈ 1.9 at n = 100) dwarfs the effects
being measured (~0.1 %).

**Equal-budget comparisons where the knob buys work.** `--restarts` multiplies
the work by R, so the study also holds `R × sa-steps` constant; that is the only
comparison that answers "should I spend my budget this way?".

**Timing uses CPU time, not the CSV.** The `time_ms` column is wall time measured
inside the 24-thread parallel region, so it absorbs memory contention — at
n = 1000 it overstates the cost by ~2.5× and is not additive. The annealing cost
is taken as `(t(10S) − t(S)) / 9` from the log's cumulative CPU time, which makes
construction cancel exactly. A 2× budget delta was tried first and is *not*
enough at n = 1000, where the run-to-run spread on construction (±3 %, ±0.6 s)
exceeds the entire annealing cost.

## Caveats

- The overview table picks each knob's best setting **because** it scored
  lowest, so its Δ is optimistically biased (winner's curse). Rows whose CI
  straddles 0 are knobs that did nothing. The report handles this itself: the
  recommended configuration is re-run on a seed the sweep never saw, and those
  rows are the only bias-free numbers in it.
- One instance distribution only (Kool/NeuOpt: coordinates U[0,1]², demands
  U{1..9}, capacity from `default_capacity`). Nothing here is evidence about
  clustered or real-world instances — run `tools/fetch_neuopt.py` and point the
  sweep at `--bundle` for that.
- `--sa-knn K` is clamped to `n-1` (`cw.c:1399`), so at n = 20 the values 30 and
  50 are the same run as 19. The analysis prints the effective K.
- At `--sa-knn 0` the vertex-selection rule is silently forced to uniform
  (`cw.c:1400`) while the header still prints the requested `--pick`. The
  `pick` study crosses the two knobs specifically to expose this.
