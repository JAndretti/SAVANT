# `sweep/` — parameter study for `cw`

Two scripts. One runs every configuration, the other turns the output into
figures and a report.

```sh
make                        # build ./cw first
sweep/run_sweep.sh          # 4680 runs, ~3 h 20 m on 24 cores, ~200 MB
uv run sweep/analyze_sweep.py   # writes sweep/figures/, report.tex, report.pdf
```

4680 runs is 468 configurations × **5 seeds** × **2 budget tiers**. The two
multipliers are the point of the design, and each answers a way the old
single-run sweep could have been wrong.

Read **`report.pdf`** for the results. It explains what each option controls,
embeds the figures from `figures/`, gives the full paired statistics, and ends
with the command for the best configuration found.

`report.tex` is generated — do not edit it by hand, edit `analyze_sweep.py`. It
is compiled automatically when `pdflatex` is on the path; `--no-pdf` emits the
`.tex` only.

### Seeds

`cw` has a single `--seed`, and it drives **both** the instance set (instance
`k` comes from `seed + k`, `cw.c:2574`) and the annealing RNG (`cw.c:2679`).
There is no way to re-randomise the solver while holding the instances fixed, so
a seed is a complete replication: fresh instances, fresh randomness. Each
configuration is run at all five, and `analyze_sweep.py` concatenates the
per-instance vectors in seed order before taking any statistic — the comparison
stays exactly paired, and every Δ in the report is the mean of `5 × m` paired
differences. Nothing rests on one instance draw.

### Tiers

A knob tuned at 10⁵ annealing steps need not still be the right setting at 10⁷,
which is where this solver is actually run. Every grid is therefore run twice:

| tier | instances/seed | `--sa-steps` | what it is |
|---|---:|---:|---|
| `lo` | 1000 | 10⁵ | the historical sweep, kept for the contrast |
| `hi` | 200 | 10⁷ | the operating point, plus a 3×10⁷ rung on `tuned` |

`m` is smaller at `hi` because each run costs 100× more; pooled over the five
seeds `hi` still carries 1000 paired instances per comparison, the same as the
headline pairing in the top-level README. Step counts that used to be absolute
are now multiples of `$STEPS`, so every ladder rescales with the tier — the
`init` budget ladder, `restarts`' iso-budget, `race`'s total, `--split-every`
(a *period*, so the comparable choice is to hold the number of Splits fixed) and
`tuned`'s restart split.

The report's body is the `hi` tier; the section **"Does any of this survive a
100× larger budget?"** puts the two side by side and names the knobs whose
recommendation does not transfer. That section is the only thing here a
single-budget sweep cannot produce.

Two studies deliberately do **not** rescale. `timing` keeps the 10⁵/10⁶ pair in
both tiers, because it measures the cost model rather than quality and a 10⁸-step
rung would dominate the whole sweep. `init`'s ladder stops at the tier's own
budget rather than continuing 10× past it, for the same reason.

## Running a subset

```sh
sweep/run_sweep.sh --list             # the study names
sweep/run_sweep.sh --plan             # count the runs, run nothing
sweep/run_sweep.sh pick knn           # only those two
TIERS=hi sweep/run_sweep.sh           # only the operating-point tier
SEEDS="42 43" sweep/run_sweep.sh      # fewer seeds (quick check)
RESUME=1 sweep/run_sweep.sh           # skip runs already on disk
uv run sweep/analyze_sweep.py --tier lo     # report the 10^5 tier instead
uv run sweep/analyze_sweep.py --no-verify   # skip the confirmation re-runs
M=100 STEPS=10000 TIERS=lo SEEDS=42 sweep/run_sweep.sh   # ~15 s smoke pass
```

| variable | default | meaning |
|---|---|---|
| `SEEDS` | `42 43 44 45 46` | one full replication per seed |
| `TIERS` | `lo hi` | which budget tiers to run |
| `M` | per tier (1000 / 200) | instances per run; overrides the tier |
| `STEPS` | per tier (10⁵ / 10⁷) | SA budget; overrides the tier |
| `N` | 100 | dimension for the studies that fix one |
| `RESUME` | 0 | 1 = skip runs whose `.meta` records `exit=0` |
| `OUT` | `sweep/results` | output root |
| `BIN` | `./cw` | binary to run |

`sweep/results/` is already covered by the repo's `.gitignore` (`results/`
matches at any depth); it is ~200 MB per sweep and is fully regenerable.

## Layout

Each run writes three files to `results/<tier>/s<seed>/<study>/<tag>.*`:

- `.log` — cw's stdout: the resolved-config header and the summary block
- `.csv` — `--csv`, one row per instance
- `.meta` — study, tag, **the exact command line**, exit code, shell wall time

Parameters are recorded as the command line rather than encoded in the tag:
`analyze_sweep.py` re-parses it into an option dict against a defaults table, so
adding an option to a study needs no change on the analysis side.

## The studies

| name | question |
|---|---|
| `init` | `--init random` vs `cw`, budget ladder ending at the tier's own, at n = 20/50/100/200 |
| `ops` | all 15 non-empty subsets of the four original operators, unbalanced weights, `--or-max` |
| `newops` | `swap*` and route-opening weights, mixtures, and a **budget ladder for the iso-time reading** |
| `knn` | `--sa-knn` ∈ {0,5,10,20,30,50} × n ∈ {20,50,100,200} |
| `timing` | construction vs annealing cost across n; thread scaling |
| `restarts` | `--restarts` at fixed and at **equal total budget**; `--cw-rand`, `--cw-alpha` |
| `split` | `--split` × `--split-every` grid, `--split-tour`, and the same at larger n |
| `pick` | `--pick` × `--pick-crit`, `--pick-eps`, and the `--pick` × `--sa-knn` interaction |
| `select` | `--vrank` × `--sa-knn` (the coupling claim), `--pick2`, `--reloc-side` |
| `race` | `--race` × `--race-at` at **equal total budget**, scaling with restart count, and `--pair` |
| `temp` | `--t-accept` × `--t-decades` (6 decade settings — the range was widened because the cooling ratio is budget-dependent) |
| `construct` | `--lambda` × `--mu` with and without SA; `--knn`/`--exact`; `--2opt` |
| `tuned` | candidate combinations against the stock defaults, at equal budget |

Two of these need a word on how they are set up.

**`newops` measures against the clock, not the step count.** `swap*` is the only
non-elementary operator in the solver — `O(L₁+L₂)` per draw where everything else
is `O(1)` — so an equal-step win is not a win. The study therefore sweeps four
configurations over a 32× range of budgets (`bud_<cfg>_x0125` … `_x4`, multiples
of `$STEPS`) and the report interpolates both curves in log(wall time). The `x1`
rung is by construction the same run as the weight blocks, which is what makes it
a valid baseline for them; the analysis checks that and refuses to plot if the
budgets ever disagree.

**`race` and `pair` only mean anything at equal total budget.** Every run in the
`race` study holds `restarts × sa-steps` constant, because racing redistributes a
budget rather than adding to one. `--pair` is a pure engineering change: without
`--race` the two interleaved chains follow identical trajectories, so the report
prints an *identical* column that must read `yes` on every row — if it does not,
the interleaving has broken something.

## Why the numbers are trustworthy

**Comparisons are paired, over five independent replications.** Instance *k* is
generated from `seed + k` (`cw.c:2574`), so every run of a study sees
byte-identical instances. The analysis compares `cost_i(A)` with `cost_i(B)` on
the same instance and reports the mean of the differences with a 95 % CI, plus a
distribution-free sign test. Comparing two means with their independent standard
deviations would be far looser — the between-instance spread (σ ≈ 1.9 at
n = 100) dwarfs the effects being measured (~0.1 %). The five seeds are pooled
into that same paired mean, so each Δ rests on 5 000 paired instances at `lo` and
1 000 at `hi`, drawn from five separate instance sets.

**Nothing is tuned at a budget it will not be used at.** The `hi` tier measures
every knob at 10⁷ steps, and the budget-dependence section reports which
recommendations differ from the 10⁵ ones. A knob that only wins at one of the two
is reported as such rather than averaged.

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
  recommended configuration is re-run on five seeds the sweep never saw, and
  those rows are the only bias-free numbers in it.
- One instance distribution only (Kool/NeuOpt: coordinates U[0,1]², demands
  U{1..9}, capacity from `default_capacity`). Nothing here is evidence about
  clustered or real-world instances — run `tools/fetch_neuopt.py` and point the
  sweep at `--bundle` for that.
- `--sa-knn K` is clamped to `n-1` (`cw.c:1675`), so at n = 20 the values 30 and
  50 are the same run as 19. The analysis prints the effective K.
- At `--sa-knn 0` the vertex-selection rule is silently forced to uniform
  (`cw.c:1676`) while the header still prints the requested `--pick`. The
  `pick` study crosses the two knobs specifically to expose this.
- **Two budgets is not a curve.** 10⁵ and 10⁷ are enough to say whether a
  recommendation transfers, not to model how it varies. If you run at 10⁸ or
  beyond, re-measure rather than extrapolate — `TIERS=hi STEPS=100000000
  sweep/run_sweep.sh` overrides the tier's budget.
- **The two tiers are not equally powered.** `m` is 1000 per seed at `lo` and
  200 at `hi`, so at equal seed count the `hi` CIs are √5 wider. Comparing a
  `lo` Δ with a `hi` Δ is comparing two estimates of different precision; the
  budget-dependence table reports both intervals for that reason.
