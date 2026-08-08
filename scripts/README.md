# `scripts/` — the defaults, on every dataset, at matched time

The defaults changed on 2026-08-07. They were derived from a sweep on one
instance family (NeuOpt, $n=100$, uniform points in the unit square), a
combination study on the same family, and a confirmation on CVRPLib X —
README's "How these defaults were found" has the derivation. This directory is
the check on everything that derivation never looked at: every benchmark set in
`data/`, XL (up to $n=10000$) included, and the reference solvers at the same
budget.

| file | role |
|---|---|
| `run_best_config.sh` | runs both configurations on every set, validates each run |
| `summarize.py` | one table over all of it, plus the paired comparison |
| `run_matched_time.py` | HGS-CVRP, LKH-3 and AILS-II at the budget SAVANT spent |
| `summarize_matched.py` | the four solvers side by side, paired, per set |
| `timing_profile.py` | what SAVANT's per-instance budget is made of (single-threaded) |

Everything is driven through the `tools/run_*.py` drivers, so every run is a
self-contained directory with `config.json`, `run.log`, `results.csv` and
`solutions.txt` — the layout `tools/validate.py` and `tools/gap_to_bks.py`
already understand.

---

## The two configurations

| tag | | |
|---|---|---|
| `cand` | `--ops 1,1,1,0,1,0.05 --t-decades 1 --pick2 2 --kick 100` | the defaults as of 2026-08-07 |
| `prev` | `--ops 1,1,1,0,0,0 --t-decades 2 --pick2 1 --kick 0` | the defaults before that |

Four flags, and `prev`'s four restore the old solver bit for bit on today's
binary — which is what makes this a clean two-armed comparison rather than a
comparison of two binaries. Everything else is written out in full (defaults
included) so a run does not depend on what `cw`'s built-in defaults happen to
be on the day:

```
--init cw --knn 0 --lambda 1 --mu 0
--or-max 3 --kick-max 10
--t-accept 0.001 --sa-knn 20
--pick 2 --pick-crit lb --pick-eps 0.3
--vrank 1 --reloc-side coin
--cw-rand perturb --cw-alpha 0.03
--race 0 --race-at 0.25 --pair 0
--restarts 1
--split off --split-every 0 --split-tour both
--empty-p 0 --dlb 0 --reheat 0 --t0-trim 0
--check
```

Two notes on that list:

* **`--restarts 1` on both sides.** The sweep's own recommendation was 8, and
  an earlier version of this directory compared `16 x 625,000` against
  `1 x 10,000,000` to test it. That question is settled and the runs are still
  in `results/best_config` (`summarize.py --pair split16 single` re-prints
  them): at a fixed total budget the split *lost* on both real sets — 63,983
  against 63,802 on X, 533,243 against 529,404 on XL — and separately, on X at
  a fixed total budget, restarts moved the gap to BKS from 0.998 % to 1.237 %.
  The sweep had measured restarts at $n=100$ only.
* **`--check` is added.** The incremental-cost drift is computed whether or not
  it is asked for (`cw.c:2170`); `--check` only prints it, so it is free, and
  `run.py` then records it in `config.json`.

## The budget: `--sa-time`, not `--sa-steps`

Both configurations get the **same wall time per instance**, 0.6 s by default,
rather than the same number of steps.

That is not a detail. A step of `cand` is not a step of `prev`: `cand` draws
`swap*`, which scans two whole routes rather than a constant number of edges,
and fires a ruin & recreate every 100 steps. Measured, at 0.6 s per instance:

| | n20 | n50 | n100 | XML100 | X | XL |
|---|---|---|---|---|---|---|
| `cand` steps | 8.82 M | 8.53 M | 8.32 M | 7.49 M | 7.35 M | 5.91 M |
| `prev` steps | 15.56 M | 15.48 M | 15.33 M | 14.77 M | 14.02 M | 11.84 M |
| ratio | 1.77× | 1.82× | 1.84× | 1.97× | 1.91× | 2.00× |

An iso-step comparison therefore hands `cand` roughly **twice the CPU**, and
rather more of it the larger the instances get. The sweep is budgeted in steps
throughout, so this directory is where the comparison is first made in the unit
a user actually spends.

`--sa-time S` works by timing a short throwaway chain on the real instance and
buying as many steps as the rest of the budget affords. Two consequences:

* **the budget is per instance, not per set.** At a fixed step count SAVANT's
  cost per instance varies 2.3x on X and 10.6x on XL — driven by mean route
  length, not by $n$ (`timing_profile.py`). Under `--sa-time` that spread moves
  from the time to the step count, which is exactly what makes the flat scalar
  budget handed to HGS and LKH-3 defensible instead of merely convenient;
* **the calibration is itself timed, so it has noise.** Measured on `n20`,
  where a 20,000-step chain takes about a millisecond: the achieved time is
  0.93×–1.04× of its median between the 5th and 95th percentiles, and the mean
  lands within 1 % of what was asked — but the extremes run 0.33×–1.28×. At
  $n = 100$ and above the chain is long enough that even the extremes are
  tight. `run_matched_time.py` prints both ranges, and the warning fires on the
  5–95 % one, because the min and max of 10,000 instances describe one badly
  timed instance rather than the set;
* **a time-budgeted run is not reproducible as a whole.** The step count
  depends on the machine and on what else is running. `cw` records the count it
  settled on for every instance in `solutions.txt` (`#sa-steps`), and
  re-running *one* instance with `--sa-steps <its count>` replays it exactly,
  bit for bit; a whole bundle cannot be replayed in one command, because the
  counts differ per instance. Pass a step count as the third argument for the
  reproducible-but-unfair mode.

## The datasets

| tag | bundle | instances | $n$ | distances | reference |
|---|---|---|---|---|---|
| `n20` | `data/cvrp_20.cvrpb` | 10,000 | 20 | float | HGS / LKH-3 via `baseline/baseline.csv` |
| `n50` | `data/cvrp_50.cvrpb` | 10,000 | 50 | float | idem |
| `n100` | `data/cvrp_100.cvrpb` | 10,000 | 100 | float | idem |
| `XML100` | `data/cvrplib/XML100.cvrpb` | 10,000 | 100 | **integer** | proven optima |
| `X` | `data/cvrplib/X.cvrpb` | 100 | 100–1000 | **integer** | best known |
| `XL` | `data/cvrplib/XL.cvrpb` | 100 | 1047–10000 | **integer** | best known (`baseline/xl_bks.csv`) |

`n200` (`data/cvrp_200.cvrpb`) is wired up in the script but not in the default
set list; add it by name if you want it.

The three CVRPLib sets are scored with **integer distances** (TSPLIB `EUC_2D`),
so they get `--round`. The generated sets are float. Without `--round` the gap
to the published references is not a gap to anything.

## Running it

```bash
sh scripts/run_best_config.sh                # every set, both configs (~35 min, 24 threads)
sh scripts/run_best_config.sh "n100 X"       # a subset
sh scripts/run_best_config.sh all 50         # smoke test: first 50 instances of each
sh scripts/run_best_config.sh all 0 1.5s     # a larger time budget
sh scripts/run_best_config.sh all 0 10000000 # iso-step instead of iso-time
uv run --no-project scripts/summarize.py --csv    # re-print the tables without re-running
```

`--limit` applies to `cw` and to nothing else, so a limited run is still
validated and still scored — but only against the first *L* rows of the
reference CSV, which is what `gap_to_bks.py` matches on. `OUT=<dir>` in the
environment keeps a smoke test out of `results/best_config`.

## Validation

Every run is checked twice, by two things that share no code with each other.

1. **`tools/validate.py <run>`** re-reads the instances from the `.cvrpb`
   bundle and recomputes every cost in Python from the coordinates: coverage
   (each customer served exactly once), capacity per route, cost consistency
   to 1e-9 relative, and the recomputed mean against the mean `cw` reported.
   The `#round` flag in `solutions.txt` is honoured, so the integer sets are
   recomputed as integers. Written to `<run>/validation.txt`; a non-zero exit
   is reported at the end of the driver and turns its exit code non-zero.
2. **`--check`**, i.e. `cw`'s own incremental-cost drift: the difference
   between the cost tracked move by move and the cost recomputed from scratch
   at the end. It catches a wrong delta in an operator, which validate.py
   cannot see — a solution built from wrong deltas is still feasible and its
   *reported* cost is still whatever the final recomputation says. The value
   appears as `drift` in the summary; anything above ~1e-9 is a bug.

For the CVRPLib sets, `tools/gap_to_bks.py` additionally scores the run against
the shipped references (`<run>/bks_gap.txt`, per-instance in
`<run>/bks_per_instance.csv`). For XML100 those are *proven optima*, so the
number is an optimality gap; for X and XL they are best known solutions, which
bound it from above.

Validation is feasibility and cost consistency, never quality: a feasible but
terrible solution passes cleanly.

## Reading the output

`summarize.py` prints two tables. The first is a row per run, with the budget it
was given — `0.6s ~4.1M` means 0.6 s per instance, which bought 4.1 M steps on
average; `10.00M` means a fixed step count. Rows with different budgets are not
comparable, which is why the column is there. The second table is the paired
`cand` vs `prev` comparison, in the same terms `sweep/report.tex` uses:

* instance *k* of one run is instance *k* of the other — same bundle, same
  order — so the statistic is the mean of the per-instance **differences**, not
  the difference of two means. The between-instance spread is orders of
  magnitude larger than the effect being measured, and an unpaired comparison
  would drown in it;
* `delta %` is that paired mean as a percentage of `prev`'s mean cost, with a
  95 % interval, and `*` marks a row where the interval excludes zero *and* the
  sign test rejects at 5 %;
* negative means the new defaults win.

Unlike the sweep, these numbers carry no selection bias on `X`, `XL` or `XML100`:
nothing there was chosen because it came out lowest. `n100` is not in that
position — it *is* the family the defaults were tuned on, so read it as a
consistency check rather than as evidence. `X` and `XL` are the real test:
they are the only sets whose instances are neither uniform nor $n \approx 100$,
and XL is the one nothing in the derivation ever saw.

## What it currently says

2026-08-07, 24 threads, 0.6 s per instance, every instance of every set:

| set | m | `cand` | `prev` | delta % | 95 % CI | win/loss | gap to ref |
|---|---|---|---|---|---|---|---|
| n20 | 10,000 | 6.12996 | 6.13155 | **−0.026** | [−0.030, −0.022] | 306/0 | — |
| n50 | 10,000 | 10.37404 | 10.39425 | **−0.194** | [−0.203, −0.186] | 4681/813 | — |
| n100 | 10,000 | 15.62165 | 15.68914 | **−0.430** | [−0.442, −0.418] | 7831/1820 | — |
| XML100 | 10,000 | 17053.751 | 17110.937 | **−0.334** | [−0.352, −0.317] | 6682/1976 | 0.390 % vs 0.713 % |
| X | 100 | 63830.85 | 64235.58 | **−0.630** | [−0.808, −0.452] | 85/13 | 1.053 % vs 1.599 % |
| XL | 100 | 529524.94 | 533606.91 | **−0.765** | [−1.051, −0.479] | 89/11 | 2.851 % vs 4.090 % |

Every row is significant at 5 % on both the interval and the sign test, so the
change transfers everywhere — at equal time, not merely at equal steps. Two
things worth reading off it:

* **the effect grows with instance size**, from −0.026 % at $n=20$ to −0.765 %
  on XL. XL is the set the derivation never saw, and the one where the kick —
  O(k(K+L)) every 100 steps — was the plausible failure mode at $n = 10000$. It
  is instead where the new defaults gain the most.
* **the old report configuration is not beaten on X.** The `single` rows in the
  first table are the sweep's per-knob recommendation without the restarts
  (`--ops 1,0,1,0,1,0.05`, no kick, 10 M steps): 63,802 on X against `cand`'s
  63,831, at 523 against 578 ms/instance. On 100 instances that difference is
  well inside the ±0.18 % the paired interval carries, and the two budget modes
  are not directly comparable, so read it as a tie rather than as a result. On
  XL the same configuration ties on cost (529,404 against 529,525) while
  spending 39 % more CPU (951 against 686 ms).

---

# The reference solvers, at SAVANT's own budget

`run_matched_time.py` takes the budget SAVANT spent above and hands it to
HGS-CVRP, LKH-3 and AILS-II on the same instances, with the same validator.

```bash
uv run --no-project scripts/run_matched_time.py                    # everything
uv run --no-project scripts/run_matched_time.py --dry-run          # budgets + commands only
uv run --no-project scripts/run_matched_time.py --sets X XL --solvers hgs lkh
uv run --no-project scripts/run_matched_time.py --limit 50         # quick
uv run --no-project scripts/summarize_matched.py --csv             # re-print the tables
```

Runs land in `results/matched_time/<stamp>_<SOLVER>_<set>/`, each with
`validation.txt` and, on the CVRPLib sets, `bks_gap.txt`.

## Which budget

Per set, the reference is SAVANT's best run **at the budget it was last run
at**: among the runs sharing the newest one's `--sa-time`/`--sa-steps` setting,
the one with the lower mean cost, ties to the cheaper run. Mixing budgets here
would hand out a number that belongs to a run nobody is comparing against;
`--savant-config cand` forces the choice instead.

The budget itself is that run's

```
cumulative CPU seconds / instances
```

— what SAVANT actually spent, not what it was asked for. CPU rather than wall
clock is the only figure that makes the four commensurable: SAVANT spreads one
instance-parallel OpenMP loop over every core, the other three are
single-threaded processes run *J* at a time. This is the convention
`tools/compare_all.sh` already uses.

## Four things that are not clean, and are reported rather than hidden

* **AILS-II cannot be matched at these budgets.** Its `-limit` is wall clock
  (it polls `currentTimeMillis`) and there is no CPU option, so two overheads
  land on top of the budget and neither shrinks when the budget does. Measured
  at n = 100 against a 0.634 s budget: **1.06 s wall** per instance, because
  JVM start, class loading, parsing and teardown sit outside the timed search
  loop; and **1.90 s CPU** against that 1.06 s wall, because the JVM runs JIT
  and GC threads alongside the search — so even at `--ails-jobs 1` it is not a
  single-threaded process. Total overrun 3–4×, and 43× on XL.
  `--ails-jobs` defaults to 1 anyway (anything else understates the budget
  further, since J JVMs share the wall clock), and `summarize_matched.py`
  warns on every row above 1.25×. Read those rows as *AILS-II given several
  times SAVANT's CPU* — which bounds it from the favourable side, but is not a
  matched comparison. HGS's `-t` (`clock()`, Genetic.cpp) and LKH-3's
  `TIME_LIMIT` (`getrusage`, GetTime.c) are both CPU budgets and land within a
  few percent at n ≈ 100.

* **The budget is still one scalar per set**, because all three drivers take
  one. Under `--sa-time` that is now nearly exact rather than a compromise:
  SAVANT spends the same time on every instance by construction, so the per-set
  mean describes the set instead of averaging over a spread. What is left is
  the calibration's own error, which the printed per-instance spread measures,
  and a systematic overshoot at large *n* — the Clarke & Wright construction is
  paid twice, once inside the calibration chain and once for the run proper,
  but only the first is charged against the budget. Measured against a 0.6 s
  request: **0.96×–0.99× on the five sets up to $n = 1000$, and 1.14× on XL**,
  where the construction at $n = 10000$ is no longer negligible. Read it in the
  `x asked` column; the budget handed to the other solvers is the achieved
  figure, so the overshoot is passed on rather than pocketed.

  The per-instance spread tells the same story from the other side. On XL it is
  0.88×–1.58× of the median between the 5th and 95th percentiles under
  `--sa-time`, against **10.5× overall** for the same set at a fixed step
  count.

  Under `--sa-steps` the same column reads very differently, and that is the
  honest reason this directory moved to `--sa-time`: at a fixed step count the
  per-instance cost varies 2.3× on X and 10.6× on XL, so a flat mean over-funds
  the cheap instances and starves the dear ones. What drove that was *not* size
  — the correlation of SAVANT's CPU with *n* is only +0.24 on X and +0.21 on XL
  — but **mean route length** (+0.80 and +0.93), because `swap*` scans both
  routes while every other operator is O(1). Matching per instance would have
  handed more CPU to the instances where *SAVANT's own operator mix* happens to
  be expensive: a property of the solver, not of the instance's difficulty, and
  one HGS and LKH-3 do not share. Budgeting SAVANT in time removes the problem
  at the source instead.

* **LKH-3 has no float mode.** EUC_2D is rounded by TSPLIB rules, so the
  continuous NeuOpt sets (`n20`, `n50`, `n100`) cannot be run without changing
  the problem. They are skipped and reported as skipped, not silently rounded.

* **A driver may return fewer instances than it was given.** LKH-3 reports
  `Cost = P_C` with `P > 0` when capacity is violated, and `run_lkh.py`
  records that as a failure rather than averaging an infeasible answer into the
  mean — likely at sub-second budgets on the larger instances. The run
  directory is still validated and scored, and the summary shows the count.

## Reading the output

One block per set, one row per solver. Every row is scored on **its own**
intersection with SAVANT and carries SAVANT's mean over exactly those
instances (`SAVANT` columns), so a solver that returned fewer instances is
still read against the right baseline instead of against a mean taken over a
different set. `x budget` is achieved CPU over requested CPU — the column that
says whether the match actually held.

As of 2026-08-07 the HGS-CVRP and LKH-3 rows were re-run against the `cand`
budget and land at 1.00–1.02× on every set except HGS/XL (2.92×: its dense
$n \times n$ matrix does not fit in 0.69 s at $n = 10000$). Two rows are older
and say so in that column: **AILS-II everywhere** (3.8–5×, and never matchable —
see above) and **LKH-3 on XL** (259×, 96 of 100 instances returned with a
capacity penalty). The LKH-3/XL re-run was abandoned after 1 h 53 min, which is
itself the result: candidate-set generation at $n = 10000$ does not fit in a
sub-second budget.

## Cost, and what `--limit` actually selects

**`--limit` defaults to 0 — every instance of every set.** That is 10,000 each
for the NeuOpt sets and XML100, 100 each for X and XL. The script prints a
wall-time floor (`instances × budget / jobs`) before launching anything, so the
cost is visible in advance rather than discovered halfway through:

```
    set         inst       HGS       LKH      AILS
    n20        10000      0:08         -      1:36
    n50        10000      0:08         -      1:41
    n100       10000      0:08         -      1:45
    XML100     10000      0:10      0:10      2:06
    X            100      0:00      0:00      0:00
    XL           100      0:00      0:00      0:01
    total                 0:36      0:11      7:12     -> 8:00 sequential
```

It is a floor, not a prediction: set-up work sits outside every solver's time
flag. For HGS and LKH-3 at *n* ≈ 100 the floor is accurate — **the two of them
over all six sets is about 50 minutes**. AILS-II runs ~3× its floor
(the ~0.45 s of JVM start and teardown per instance), so `--ails-jobs 1` puts
it near **12 hours**. Raising `--ails-jobs` is the lever: its budget is wall
clock, so *J* at once costs *J*-way contention rather than *J*-way less work,
and on 24 cores `--ails-jobs 8` brings it to roughly 1.5 h. `--solvers hgs lkh`
drops it entirely.

If you do pass a `--limit`, it keeps the **first** *L* instances of the set's
own order, which is a valid subsample only when that order carries no
information:

* **NeuOpt sets** (`n20`, `n50`, `n100`) — instance *k* is generated from
  `seed + k`, i.i.d. by construction, so a prefix is a random sample. The only
  cost is power: the paired interval widens as 1/√m, from roughly ±0.008 % at
  m = 10,000 to ±0.05 % at m = 200. That resolves solver-vs-solver gaps
  (typically 0.5–2 %) comfortably; it does not resolve the sub-0.1 % effects
  the sweep was chasing.
* **XML100** — the names encode the generator's configuration
  (`XML100_<depot><customers><demand><routes>_<rep>`), so the sorted order is
  *blocked by configuration*: the first 200 instances cover **8 of the 378
  configurations** in the set, one depot placement and one customer
  distribution. That is a slice, not a sample, and `run_matched_time.py` says
  so before it runs anything.
* **X, XL** — 100 instances each. A limit would take a lexicographic prefix,
  which is not even ordered by *n* (`X-n1001` sorts before `X-n101`).
