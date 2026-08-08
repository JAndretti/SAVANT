# `bench/` — the recommended configuration, on everything, in one table

One command, one configuration, every dataset in `data/`. This is the "what
does it actually do" directory: no tuning per set, no restarts, no per-set
budget — cw's 2026-08-07 defaults with `--sa-steps N`, which sets each
instance's budget from its own dimension.

| file | role |
|---|---|
| `run_bench.py` | SAVANT: one run per set, validated and scored |
| `run_baselines.py` | HGS-CVRP, LKH-3 and AILS-II at SAVANT's own per-instance time |
| `report_bench.py` | the tables → `report.tex` → `report.pdf` |
| `report.pdf` | **the output** |

```bash
uv run bench/run_bench.py                 # SAVANT, ~15 min
uv run bench/run_bench.py --sets X XL
uv run bench/run_bench.py --limit 20      # smoke test
uv run bench/run_baselines.py             # the other three, hours (see below)
uv run bench/report_bench.py              # re-print without re-running
uv run bench/report_bench.py --format md  # the same tables as markdown
uv run bench/report_bench.py --verify     # re-check the XML100 name decoding
```

Which solver on which set: **HGS-CVRP everywhere**; **LKH-3 on XML100, X and
XL** only, since it has no float mode and the generated sets are continuous;
**AILS-II on X and XL**. `--skip lkh:XL` drops a pair.

Runs land in `results/bench/<stamp>_<set>/` with `run.py`'s usual
`config.json` / `run.log` / `results.csv` / `solutions.txt`, plus
`validation.txt` and, on the CVRPLib sets, `bks_gap.txt`.

## What is reported, and why it differs by set

* **n20, n50, n100** — 10,000 generated instances each, continuous distances,
  no reference solutions. Mean, min and max cost. `min`/`max` are over the
  *instances*, not over repeated runs of one instance: they describe the spread
  of the set.
* **XML100** — 10,000 instances of 100 customers with **proven optima**, so the
  gap is a true optimality gap. Reported in aggregate and then broken down by
  the four generator attributes, in the layout of Table 4 of the XL paper
  (`paper/2601.11467v2-2.pdf`).
* **X and XL** — one row per instance: cost, gap to the reference, compute
  time, and the budget the dimension rule gave it. These are the sets where
  instances differ in size, so a per-instance table is the only honest one.

## Two choices worth knowing about

**X and XL run single-threaded.** `results.csv`'s `time_ms` is wall time
measured *inside* cw's instance-parallel region, so with several threads it
absorbs the memory contention between them — the sweep measures that at ~2.5×
at n = 1000, and it is not additive. With `--threads 1`, wall = CPU and the
per-instance column is a true single-core time. It costs about 10 extra minutes
of wall clock on XL and makes the number mean something. The 10,000-instance
sets use every core, because only their cost is reported.

**The CVRPLib sets are read with `--dir`, not `--bundle`.** The bundle format
carries no names, and the XML100 breakdown needs them: the four digits of
`XML100_<depot><customers><demand><route>_<rep>` *are* the attributes.

## The XML100 decoding is measured, not assumed

Getting the four digits wrong would put every row of the breakdown in the wrong
bucket while still looking entirely plausible, so the mapping was derived from
the `.vrp` files rather than read off the paper:

| digit | evidence |
|---|---|
| depot | the depot's coordinates: `2` is exactly (500, 500), `3` is (0, 0), `1` is neither → Central, Eccentric, Random |
| customers | mean nearest-neighbour distance over 5 instances each: `1` = 52.8, `2` = 31.2, `3` = 47.1 → Random, Clustered, Random-Clustered |
| demand | the observed support: `1` = {1}, `2` = [1,10], `3` = [5,10], `4` = [1,100], `5` = [51,100], `6` quadrant-dependent, `7` mostly small with a few large |
| route size | *r* = n / (total demand / capacity) over 8 instances each: 3–4, 5–7, 8–11, 12–15, 16–23, 25–46 → U[3,5] U[5,8] U[8,12] U[12,16] U[16,25] U[25,50] |

`uv run bench/report_bench.py --verify` re-derives the depot, demand and
route-size mappings and exits non-zero if any of them has moved.

## The budgets are requested, not achieved

Each reference solver is asked for SAVANT's own measured single-core CPU per
instance — 34 ms on `n20`, 41 ms on XML100, 154 ms on X, 5.8 s on XL. None of
them is matched at those budgets, and none was ever going to be:

* a process launch is a few milliseconds, which is not nothing against 34 ms;
* HGS-CVRP builds a dense *n* × *n* distance matrix and LKH-3 generates a
  candidate set, neither of which is inside the time flag;
* **AILS-II is budgeted in wall clock** (`-limit` polls `currentTimeMillis`),
  has no CPU option, and runs its JIT and GC threads alongside the search — so
  what it costs in CPU is whatever the JVM decides. On X it spent 2.31 CPU-s
  against a 0.154 s budget, **15× over**.

Every table therefore carries the solver's *achieved* CPU and an `× asked`
column. Read those before the gap columns: a row several times over budget is
not being compared at equal cost, and the tables say which rows those are
rather than burying it.

**Cost.** HGS and LKH-3 on the four 10,000-instance sets take about half a
minute each. The expensive pairs are on XL, where the per-instance budget is
5.8 s and the set-up dwarfs it: LKH-3 needs a candidate set at *n* up to
10,000, and AILS-II needs a JVM plus parsing. Budget a couple of hours for the
whole battery, or skip those two pairs.

## Rows are compared on the same instances

A solver that fails on some instances would otherwise be averaged over an
easier subset and look better for it — LKH-3 returned 59 of the 100 X instances
at this budget. So the summary tables report `returned` separately and compute
every other column on the instances **every** solver returned. The `own subset`
column is the same mean over whatever that solver alone managed: the gap
between the two columns is exactly how much it is flattered by what it failed
on.

Pairing across solvers is by instance name where the drivers agree on names
(the CVRPLib sets, read with `--dir`) and by position otherwise — cw over a
bundle writes `cvrp_20.cvrpb#7` where `run_hgs.py` writes `cvrp_20_0007`.
Either way *n* is checked on every pair, and a disagreement drops the pairing
with a warning instead of averaging two runs in the wrong order.

## What this is not

A tuned comparison. The XL references are the final BKSs of the CVRPLib BKS
Challenge — months of dedicated large-scale solvers, LLM-guided heuristic
design and, in one case, ~117 CPU-years. A single sub-minute run of a
3,000-line single-file solver is not in that regime, and the gap column is
there to show where the budget lands, not to claim parity.

Nor is it a tuning study: every solver here runs at its own defaults, and
SAVANT's budget is the one rule from `timing/`, not something fitted per set.
`scripts/` is where the defaults were checked against their predecessors, and
`sweep/` is where they came from.
