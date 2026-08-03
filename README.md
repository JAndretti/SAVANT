# SAVANT — **SAV**ings, **AN**nealing, **T**ournament

A fast, fully verifiable classical solver for the Capacitated Vehicle Routing Problem,
in C: the Clarke & Wright *savings* heuristic (parallel version, 1964) followed by
simulated annealing with regret-guided move selection. Benchmarked against the test
sets of

> Ma, Y.; Cao, Z.; Chee, Y. M. *Learning to Search Feasible and Infeasible Regions of
> Routing Problems with Flexible Neural k-Opt*, NeurIPS 2023 (arXiv:2310.18264),
> repository `yining043/NeuOpt`, folder `datasets/`.

On the full CVRP-100 set (10 000 instances) it reaches **+0.75 % of the published HGS
objective and +0.21 % of LKH-3 in 5.2 minutes** on a 12-core laptop, with every
solution independently revalidated. Those two references are table entries obtained
on other hardware at an unstated budget; held to the *same CPU budget on the same
machine*, the gap to HGS is **+0.24 %** — see
[against HGS run here](#against-hgs-run-here), which also says plainly that HGS still
wins.

## Why "SAVANT"

The three letters are the three stages of the pipeline, in the order they run:

| | stage | what it does |
|---|---|---|
| **SAV** | **savings** | Clarke & Wright construction — savings truncated to the k nearest neighbours, radix-sorted, routes merged via adjacency lists + union-find so that no route reversal is ever needed |
| **AN** | **annealing** | simulated annealing over doubly linked routes, with the initial temperature calibrated from the instance rather than tuned by hand |
| **T** | **tournament** | the vertex to move is chosen by a size-2 tournament on a *regret* — how far a customer's two current edges exceed the two shortest it could possibly carry |

The name deliberately foregrounds the tournament, because that is where the method
departs from the textbook combination. Everything else — savings construction,
annealing, relocate/swap/2-opt\*/or-opt, kNN candidate lists, the Prins/Vidal Split,
multi-restart — is standard practice, and this README says so wherever it applies.
Two things are not:

* **Regret-guided selection with bounded pressure.** Biasing the *first* vertex of
  each move by `inc[u] − lb[u]` costs two array reads, because `inc[]` is maintained
  incrementally rather than recomputed. Sampling it by tournament rather than
  proportionally caps the selection pressure at ≈ T/n, and that cap turns out to be
  the whole point: exactly proportional sampling through a Fenwick tree is *worse than
  uniform* on CVRP-200, because an isolated customer whose bound is unattainable keeps
  a high regret forever and absorbs the budget. Three further hypotheses about the
  right definition of regret were also refuted by measurement. See
  [guided selection](#guided-selection-of-the-vertex-to-move).
* **Verifiability as a design constraint.** Every cost is recomputed from the
  coordinates by `validate.py`, which shares no line of code with the solver; the
  construction is cross-checked against a naive Python re-implementation, the O(n)
  Split against a naive O(n²) DP, and the pure descent against an exhaustive local
  search. That independence is what caught the `--round` bug the internal validator
  structurally could not see. See [audit and testing](#audit-and-testing).

**Contents**

- [Getting started](#getting-started) · [Results](#results) · [against HGS run here](#against-hgs-run-here)
- [How a solution is represented](#how-a-solution-is-represented)
- [How feasibility is enforced](#how-feasibility-is-enforced)
- [Execution pipeline](#execution-pipeline) — [top level](#1-top-level--one-instance) · [inside `anneal()`](#2-inside-anneal) · [Split](#3-split)
- [Option reference](#option-reference)
- Measurements: [annealing](#simulated-annealing) · [does the construction matter?](#does-the-construction-matter---init-random) · [vertex selection](#guided-selection-of-the-vertex-to-move) · [restarts](#multiple-restarts) · [Split](#optimal-split-vidal-2016)
- [Writing and validating solutions](#writing-and-validating-solutions)
- [Hot-path optimisation](#hot-path-optimisation) · [Audit and testing](#audit-and-testing)
- [Implementation choices](#implementation-choices) · [Known caveats](#known-caveats) · [Files](#files)

---

## Getting started

```bash
make                                    # gcc -O3 -march=native -fopenmp
make macos                              # macOS + Apple clang (needs brew install libomp)
python3 tools/fetch_neuopt.py           # download + convert all 4 sizes

./cw --bundle data/cvrp_100.cvrpb       # 10 000 instances
./cw --random -n 1000 -m 100 --cap 100  # generated instances

# one self-contained directory per run (config + results + solutions)
python3 tools/run.py --name my_run --bundle data/cvrp_100.cvrpb --sa-steps 200000
python3 tools/validate.py results/<run> # independent check + gaps to HGS/LKH-3
python3 tools/analyze.py results/<run>  # statistics + analysis.png (needs uv sync)

# instrumented single run: cost trajectory and per-operator counts
make trace
./cw_trace --bundle data/cvrp_100.cvrpb --index 0 --sa-steps 200000 --every 20
python3 tools/analyze.py --trace results/trace_<timestamp>

# HGS-CVRP as a local reference (see "Against HGS run here")
git clone https://github.com/vidalt/HGS-CVRP.git external/HGS-CVRP
cmake -S external/HGS-CVRP -B external/HGS-CVRP/build -DCMAKE_BUILD_TYPE=Release
make -C external/HGS-CVRP/build bin
python3 tools/run_hgs.py --bundle data/cvrp_100.cvrpb --time 0.25 --name HGS_t0.25
sh tools/compare_hgs.sh 1000                   # the whole sweep, both solvers
```

`data/`, `edge/`, `results/`, `external/` and the `cw` binary are git-ignored — all
are regenerated by the commands above.

### Repository layout

```
├── src/
│   ├── cw.c                the solver (single translation unit)
│   └── trace.c             instrumented driver: cost trajectory + operator counts
├── Makefile              build targets; binaries land at ./cw and ./cw_trace
├── tools/                things you run
│   ├── run.py              drives ./cw into a self-contained run directory
│   ├── validate.py         independent solution checker + gaps to HGS/LKH-3
│   ├── analyze.py          statistics and plots for a run, comparison, or trace
│   ├── fetch_neuopt.py     download + convert the NeuOpt test sets
│   ├── run_hgs.py          drives HGS-CVRP into the same run-directory layout
│   ├── bundle_to_vrp.py    .cvrpb -> TSPLIB .vrp, for external solvers
│   ├── compare_table.py    time-quality table across runs, ranked on CPU/instance
│   ├── paired_gap.py       per-instance paired comparison of two runs
│   └── compare_hgs.sh      the SAVANT-vs-HGS budget sweep, end to end
├── tests/                verification suite
│   ├── check.py            C&W vs a naive Python re-implementation
│   ├── checksplit.py       O(n) Split vs a naive O(n²) DP
│   ├── localopt.py         descent vs an exhaustive local search
│   ├── fuzz.py             random option combinations, every solution revalidated
│   ├── mkedge.py           generates the 15 edge-case instances
│   └── _paths.py           locates the root, the binary and tools/ from __file__
├── baseline/             published baselines + their LaTeX rendering
└── data/ edge/ results/  generated, git-ignored
```

Every script derives the repository root from its own location, so they work from
any working directory: `python3 tools/validate.py results/<run>` behaves the same
whether or not you are inside the repo.

## Results

Mean cost; the **HGS** and **LKH-3** references come from tables 1 and 12 of the paper.
Clarke & Wright alone, on the full NeuOpt sets:

| set       | inst. | C&W      | gap /HGS | C&W + 2-opt | gap /HGS | HGS    | LKH-3  | ms/inst. (1 core) |
|-----------|------:|---------:|---------:|------------:|---------:|-------:|-------:|------------------:|
| CVRP-20   | 10000 | 6.36357  | +3.81 %  | 6.34788     | +3.55 %  | 6.130  | 6.135  | 0.007 |
| CVRP-50   | 10000 | 10.89819 | +5.13 %  | 10.85661    | +4.73 %  | 10.366 | 10.375 | 0.026 |
| CVRP-100  | 10000 | 16.49677 | +6.00 %  | 16.42057    | +5.51 %  | 15.563 | 15.647 | 0.096 |
| CVRP-200  |  1000 | 23.36610 | +7.40 %  | 23.22652    | +6.80 %  | 21.756 | 22.010 | 0.332 |

With annealing on top, full CVRP-100 set (10 000 instances), 12 threads:

| configuration | cost | gap /HGS | gap /LKH-3 | wall time |
|---|---:|---:|---:|---:|
| C&W alone | 16.49677 | +6.00 % | +5.43 % | 0.1 s |
| 200k steps, 1 restart | 16.01284 | +2.89 % | +2.34 % | 5.8 s |
| 1M steps, 10 restarts | 15.67949 | +0.75 % | +0.21 % | 5.2 min |

Every solution produced is checked (each customer served once, capacities
respected); `check.py` additionally compares the binary against an independent naive
Python re-implementation (max relative gap observed: 2.5e-12, due solely to the order
in which the floats are added).

Published baselines for comparison are collected in `baseline/baseline.csv`, with a
LaTeX rendering (table + quality/time scatter) built by `baseline/make_tex.py`.

### Against HGS run here

The HGS and LKH-3 columns above are transcribed from a paper. They were produced on
unknown hardware at an unstated budget, so a gap against them measures the difference
between two experiments, not between two solvers. The reference is therefore also
**compiled and run locally**, on the same instances, on the same machine.

```bash
git clone https://github.com/vidalt/HGS-CVRP.git external/HGS-CVRP
cmake -S external/HGS-CVRP -B external/HGS-CVRP/build -DCMAKE_BUILD_TYPE=Release
make -C external/HGS-CVRP/build bin
sh tools/compare_hgs.sh 1000                  # ~26 min on 12 cores, 17 of them
                                              # in the default-termination point alone
python3 tools/compare_table.py results/*HGS_cvrp_100* results/*SAVANT_cvrp_100*
```

First 1000 CVRP-100 instances, 12-core laptop. Every HGS run is checked by
`tools/validate.py` — the same validator, sharing no code with either solver:

| solver | budget | mean | cpu s/inst. |
|---|---|---:|---:|
| SAVANT | 100k steps × 10 | 15.88040 | 0.0455 |
| HGS    | `-t 0.1`        | 15.76620 | 0.1032 |
| SAVANT | 300k steps × 10 | 15.79145 | 0.1288 |
| HGS    | `-t 0.25`       | 15.67453 | 0.2535 |
| SAVANT | 1M steps × 10   | 15.71259 | 0.4327 |
| HGS    | `-t 0.5`        | 15.63336 | 0.5036 |
| HGS    | `-t 1.0`        | 15.61073 | 1.0037 |
| SAVANT | 3M steps × 10   | 15.66599 | 1.2951 |
| HGS    | `-t 2.0`        | 15.60030 | 2.0038 |
| HGS    | default (`-it 20000`) | 15.59477 | 10.3138 |

Two things follow.

**The published figure reproduces.** Local HGS at its own default termination gives
15.5948 against the published 15.563 — 0.57 standard errors apart, indistinguishable.
`baseline/baseline.csv` is sound and the instance pipeline introduces no drift.

**At matched CPU, HGS wins by roughly 0.3–0.45 % throughout.** Compared per instance
by `tools/paired_gap.py`, which pairs on the instance rather than differencing two
means:

| A | B | gap B vs A | B better on | 
|---|---|---:|---:|
| HGS `-t 0.25` (0.254 cpu-s) | SAVANT 1M × 10 (0.433 cpu-s) | +0.243 % ± 0.015 % | 27.5 % |
| HGS `-t 0.5`  (0.504 cpu-s) | SAVANT 1M × 10 (0.433 cpu-s) | +0.507 % ± 0.015 % | 7.9 % |
| HGS `-t 1.0`  (1.004 cpu-s) | SAVANT 3M × 10 (1.295 cpu-s) | +0.354 % ± 0.011 % | 5.3 % |

The first row gives SAVANT 71 % *more* CPU than HGS and it still loses, on 71 % of
instances. This is a better result than the +0.75 % headline suggests — that number
compares against a much larger budget — but it is a loss, and the deficit does not
close as the budget grows.

Pairing is not decoration. The per-instance cost has a standard deviation of ~1.4, so
the standard error of a mean over 1000 instances is 0.36 % — larger than every gap in
the table above. Only the paired difference (SE 0.015 %) resolves them, and no
comparison against a number from another paper can do better than that 0.36 % floor.
That is the whole argument for running the reference locally.

#### Two traps

**`-round` defaults to 1.** These coordinates live in [0,1]², so integer rounding
collapses every distance to 0 or 1 and HGS reports `Cost 0` — a silent failure that
looks like HGS annihilating everything. Same instance, same seed:

```
-round 0  ->  Cost 14.6052
(default) ->  Cost 0
```

`run_hgs.py` always passes `-round 0` and refuses `--round 1` on continuous
coordinates.

**`-t` is CPU time, not wall clock** (`Genetic.cpp` compares against `clock()`). Each
of 12 concurrent processes still gets its full budget, which makes the measurement
contention-independent — but wall time then exceeds `-t` under load. `run_hgs.py`
records both, and `compare_table.py` ranks on CPU seconds per instance, the only
figure commensurable with `cw`'s own cumulative-CPU line.

A third detail, handled rather than trapped: HGS's TSPLIB parser skips exactly three
lines before reading `KEY : VALUE` tokens, expecting `NAME`/`COMMENT`/`TYPE`. The
`.vrp` files written by `fetch_neuopt.py --tsplib` omit `COMMENT`, so `DIMENSION` is
swallowed and the error points elsewhere. `bundle_to_vrp.py` is a separate writer for
that reason, and it exports from the `.cvrpb` bundle so the external solver gets
byte-identical instances rather than a re-derivation.

#### `tools/run_hgs.py`

Runs HGS over a bundle and writes the run-directory layout `validate.py` already
understands (`solutions.txt`, `results.csv`, `config.json`, `run.log`), so an HGS run
is validated by exactly the command that validates a SAVANT run.

| option | meaning |
|---|---|
| `--bundle FILE` | instances (`.cvrpb`), required |
| `--limit N` | solve only the first N |
| `--time SEC` | HGS `-t`: **CPU**-second budget per instance |
| `--it N` | HGS `-it`: iterations without improvement (HGS's own default is 20000) |
| `--seed S` | HGS `-seed` (default 0) |
| `--veh V` | HGS `-veh`: fleet size (default: HGS's estimate, ⌈1.3·LB⌉+3) |
| `--round 0\|1` | integer distances; **default 0**, and 1 is refused on continuous coordinates |
| `--jobs J` | parallel HGS processes (default: all cores) |
| `--hgs PATH` | binary (default `external/HGS-CVRP/build/hgs`) |
| `--vrp DIR` | exported TSPLIB files (default `data/vrp_<n>/`, created once and reused) |
| `--name`, `--out` | run directory naming, as in `run.py` |
| `--keep-sol` | keep HGS's raw `.sol` files in the run directory |
| `--dry-run` | print what would run |

HGS prints its objective with 6 significant digits, too few to validate at 1e-9.
`solutions.txt` therefore carries the cost **recomputed here in float64** from the
coordinates, and HGS's own figure is cross-checked against it to 1e-5 relative; the
worst disagreement observed over 6000 runs was 3.9e-6. `validate.py` then recomputes
it a third time, independently.

The underlying binary's own interface is `./hgs instance.vrp out.sol [options]`, with
`-t`, `-it`, `-seed`, `-veh`, `-round`, `-log`, and the population parameters
`-nbGranular -mu -lambda -nbElite -nbClose -nbIterPenaltyManagement -targetFeasible
-penaltyIncrease -penaltyDecrease`. `run_hgs.py` exposes the first six; the rest are
left at Vidal's defaults deliberately, since a tuned reference would no longer be the
published algorithm.

#### `tools/compare_table.py` and `tools/paired_gap.py`

```bash
python3 tools/compare_table.py results/<run> ...      # one row per run, sorted by CPU
python3 tools/compare_table.py --csv cmp.csv --ref <substring> results/<run> ...
python3 tools/paired_gap.py results/<run_A> results/<run_B>
```

`compare_table.py` reads each `config.json` and ranks on CPU seconds per instance,
taking the mean from `mean_cost` (HGS runs) or `cost_after_sa` (SAVANT runs) and the
CPU from `cpu_s` or `time_cpu_s`. `paired_gap.py` reports the mean paired difference
with its standard error, a 95 % CI, the win counts, a sign test, and — for contrast —
the unpaired standard error that any comparison against a published figure is stuck
with.

---

## How a solution is represented

There are **three** representations, each used by a different phase. Take this
5-customer instance (`edge/tiny5.cvrpb`, Q = 30) whose solution is the single route
`0 → 4 → 2 → 3 → 1 → 5 → 0`:

```
 node        x        y   demand
    0   0.5000   0.5000        0   <- depot
    1   0.1831   0.0646        1
    2   0.1144   0.5411        5
    3   0.0146   0.4659        9
    4   0.4868   0.6015        2
    5   0.9649   0.0889        5
```

### During Clarke & Wright — adjacency lists + union-find

No route lists at all. Each customer stores **at most two neighbours** (src/cw.c:1887):

```c
int *uf, *deg, *adj;   /* adj is 2*(n+1): adj[2*i], adj[2*i+1] */
double *load;          /* load[root] = load of that route */
```

```
customer:  1      2      3      4      5
adj[2i]:   3      4      2      2      1        deg: 1  2  2  2  1
adj[2i+1]: 5      3      1      -               uf root: all 1 after merging
```

A route is an implicit chain, read by walking `adj` from any endpoint (`deg < 2`).
This is why **no route reversal is ever needed**: merging `A—i` with `j—B` is two
`adj` writes plus a union, O(α(n)), regardless of orientation. With explicit lists
one of them would have to be reversed — which is exactly what `check.py`'s naive
reference does with `A[::-1] + B`.

Routes are materialised only once, at the end (src/cw.c:1908), by walking each chain
from an endpoint.

### During annealing — doubly linked lists with virtual depots

```c
typedef struct { int n, R; int *nxt, *prv, *rid; double *load; } Sol;
#define VD(r) (n + 1 + (r))     /* virtual depot of route r */
```

Each route is a **circular** list closed on its own virtual depot at index `n+1+r`,
whose coordinates are copied from the real depot. For the example (R = 1, so the
virtual depot is index 6):

```
index:   1   2   3   4   5   6(=VD0)
nxt:     5   3   1   2   6   4
prv:     3   4   2   6   1   5
rid:     0   0   0   0   0   0
load[0] = 22

traversal from VD0: 6 → 4 → 2 → 3 → 1 → 5 → back to 6
```

Two design consequences:

- **The virtual depot removes every special case.** Because index 6 holds the depot's
  coordinates, `dxy(w, 5, 6)` is just a distance — no "am I at the end of a route?"
  branch anywhere in the operators.
- **`rid[u]` gives u's route in O(1)**, which is what makes the capacity checks below
  O(1) rather than requiring a route walk.

An empty route is `nxt[VD] = prv[VD] = VD`, a self-loop. (See
[caveat 1](#1-emptied-routes-can-never-be-refilled): representable, but unreachable
by the move set.)

### On output — flat array, then text

`sol_out` packs routes into `[n_words, c.., 0, c.., 0]` with `0` as separator
(src/cw.c:2164), written as the self-describing `--sol` text:

```
#CWSOL 1
#instances 1
#source edge/tiny5.cvrpb
#round 0
#format inst <idx> <name> <n> <Q> <cost> <routes>, then one line per route
inst 0 tiny5.cvrpb#0 5 30 2.4425038530140353 1
4 2 3 1 5
```

`validate.py` parses that back into a plain Python list of lists — a fourth
representation, deliberately sharing nothing with the solver.

---

## How feasibility is enforced

Two constraints: **each customer served exactly once**, and **route load ≤ Q**.

The architectural point: feasibility is an **invariant, never a repair**. The solver
never visits an infeasible state and never uses a penalty term. Any move that would
violate capacity returns `0.0` *before* the acceptance test, so it is not even a
candidate.

This is worth stating explicitly given the benchmark: NeuOpt-GIRE's premise is
*"Learning to Search Feasible and **Infeasible** Regions"* — it deliberately
traverses infeasible space to reach better optima. This solver structurally cannot.
That is a genuine difference in search power, not an implementation detail.

### Capacity — checked at every mutation point

| site | check | line |
|---|---|---|
| C&W merge | `load[ri] + load[rj] > cap + EPS` → skip the saving | src/cw.c:1896 |
| relocate | `load[rv] + dem[u] > cap + EPS` | src/cw.c:1058 |
| swap | both directions: `load[ru] − du + dv` and `load[rv] − dv + du` | src/cw.c:1094 |
| or-opt | `load[rv] + sload > cap + EPS` (segment load) | src/cw.c:1170 |
| 2-opt\* | prefix/tail loads recomputed, both new routes checked | src/cw.c:1425 |
| Split | capacity *is* the recurrence: window `D[j] − D[i] ≤ Q` | src/cw.c:1538 |

Two subtleties:

**Intra-route moves skip the check entirely.** Each is guarded by
`if (ru != rv && ...)`. When both vertices lie on the same route the load is
unchanged by construction, so the check is provably unnecessary — and it sits on the
hot path. Intra-route 2-opt (segment reversal) has no check at all, for the same
reason.

**`load[]` is maintained incrementally**, updated on each accepted move
(src/cw.c:1071), so a check costs two array reads rather than summing a route.

The `EPS = 1e-9` tolerance appears in every comparison (`> cap + EPS`) because
demands may be fractional (`edge/frac.cvrpb` uses Q = 7.5 with demands like 0.196);
an exact `>` would reject legitimate moves on rounding alone.

### Coverage — guaranteed structurally

- **C&W**: every customer starts as its own route; merges only ever *join* two
  routes. `deg[i] < 2` prevents a third edge and the union-find root test prevents
  closing a cycle, so the partition invariant is preserved by construction.
- **Annealing**: all four operators are **permutations** of the linked list. Nothing
  is created or destroyed — relocate unlinks and relinks, swap exchanges two
  positions, or-opt moves a contiguous block, 2-opt reverses or exchanges tails. A
  customer physically cannot appear twice or vanish.
- **Split**: reconstructs from the giant tour `t[1..n]`, asserted to hold exactly n
  entries (`if (L != n) return -1.0`, src/cw.c:1510).

### Verified rather than assumed

Because "guaranteed by construction" is exactly the kind of claim that quietly stops
being true, it is checked three times independently:

1. **`solve_cw` exit check** (src/cw.c:2163): a `seen[]` sweep sets `feasible = 0` on any
   out-of-range index, duplicate, unserved customer or overloaded route. Reported as
   `feasible` in the CSV, and drives exit code 2.
2. **`--validate`** re-reads the written file and recomputes everything (src/cw.c:2216).
3. **`validate.py`**, sharing no code with the solver — the version that carries
   evidential weight. On the full 10 000-instance CVRP-100 run: 0 errors, max cost
   deviation 4.3e-16.

On a genuinely infeasible instance — `edge/infeasible.cvrpb`, one demand of 99
against Q = 30 — no amount of checking helps. The solver detects it, reports
`feasible=0`, exits 2, and Split refuses to run (src/cw.c:1522). That guard exists
because the fuzzer found a segfault there.

---

# Execution pipeline

What happens, in order, and what each option controls. Line references point into
`cw.c`.

## 1. Top level — one instance

`solve_cw` (src/cw.c:1977) is called once per instance, from an OpenMP loop over
instances.

```
choose K (savings list size)          ── --knn / --exact
ws_ensure()                            allocate-or-reuse thread buffers
d0[i] = dist(depot, i)                 for all i

for rs = 0 .. restarts-1:              ── --restarts
    │  ── --init cw (default) ───────────────────────────────
    ├─ 1. build savings list           ── --lambda --mu --cw-rand --cw-alpha
    ├─ 2. radix sort (LSD 4x8 on float32 key)
    ├─ 3. merge routes (union-find, degree <= 2)
    ├─ 4. walk adjacency chains -> explicit routes
    │  ── --init random ─────────────────────────────────────
    ├─ 1'. shuffle customers, cut at the capacity limit (1-4 replaced)
    ├─ 5. optional intra-route 2-opt   ── --2opt
    ├─ 6. build linked-list Sol, cost0 = sol_cost()
    ├─ 7. Split, if --split cw|both
    ├─ 8. anneal()                     ── all SA options
    ├─ 9. Split, if --split end|both
    └─ 10. keep this restart if best so far
restore best restart, verify coverage/capacity, emit
```

### Savings list size (src/cw.c:1986)

| `--knn` | behaviour |
|---|---|
| `0` (default) | exact all-pairs if `n <= 1500`, otherwise `K = 32` |
| `K > 0` | truncate to the K nearest neighbours |
| `--exact` (`K = -1`) | full `n(n-1)/2` list |

Clamped to `1 <= K <= n-1`.

### Two properties of the restart loop (src/cw.c:2036)

**Restart 0 is always deterministic C&W.** `rnd = (rs > 0) ? o->cw_rand : 0`, so
`--restarts 1` reproduces no-restart behaviour bit-for-bit, and a multi-start can
never end up worse than a single start.

**Seeds are derived from the instance seed**, not from a thread-local stream:

```c
C&W seed  = seed * 0xD1342543DE82EF95 + rs * 0x9E3779B97F4A7C15   // src/cw.c:1839
anneal    = (seed ^ 0x5DEECE66D)      + rs * 0xBF58476D1CE4E5B9   // src/cw.c:1942
```

Results are therefore independent of `--threads` and of scheduling order — verified
by MD5 on the solution file at 1, 2, 4 and 12 threads.

### `--2opt` is a construction post-process

Applied per route during chain reconstruction (src/cw.c:1920), first-improvement to a
local optimum, *before* the linked-list representation exists. It is not part of the
annealing neighbourhood — the SA has its own 2-opt operator.

## 2. Inside `anneal()`

The annealing chain machinery starts at src/cw.c:1672 (`sa_config`, `chain_init`,
`chain_step`, `anneal_chains`).

### 2.1 Setup, in order

**Step 1 — clamp K, resolve `--pick`** (src/cw.c:1676)

```c
K = sa_knn;  clamp to [0, n-1]
pick_t = (K > 0) ? o->pick_t : 1;      // the regret needs the kNN lists
```

A second downgrade at src/cw.c:1701: if `K == 0` and `--pick-crit` is `lb` or `remnorm`
(both read `lb[]`), `pick_t` falls back to 1 as well. In short, **`--pick` has no
effect under `--sa-knn 0`.**

**Step 2 — operator thresholds** (src/cw.c:1680)

The four `--ops` weights are normalised into cumulative `uint32` cut-points, so
choosing an operator costs three integer comparisons on one random word rather than
a float draw:

```c
th1 = (w_rel   / S) * 2^32
th2 = th1 + (w_swap / S) * 2^32
th3 = th2 + (w_2opt / S) * 2^32
if (w_or <= 0) th3 = 0xFFFFFFFF;
```

**Step 3 — `xy_build`** (src/cw.c:799). Interleaved `(x,y)` per vertex, with the virtual
depots `n+1 .. 2n+3` filled with the real depot's coordinates. This is what lets
`dxy` avoid any "is this a depot?" test.

**Step 4 — kNN and lower bound**, if `K > 0`: `knn_need` + `lb_build`, then

```c
eps0 = pick_eps * sum(lb[u]) / (2n)    // ~ pick_eps * mean nearest-neighbour distance
```

**Step 5 — `inc_build`** (src/cw.c:967). Fills `inc[u]` (cost of u's two current edges)
for every u, plus `bad[u]`, plus the Fenwick tree when `--pick 0`.

**Step 6 — temperature** (src/cw.c:1725)

```c
if (--t0 given)  T0 = t0,  Tend = tend            // default tend = t0 * 1e-4
else             T0 = calibrate_T0(...),  Tend = T0 * 10^(-t_decades)
alpha = (Tend / T0)^(1/(steps-1))                 // geometric, one step per iteration
```

`calibrate_T0` (src/cw.c:1613) draws up to 2000 moves in **probe mode** — delta computed,
move not applied — and collects the first 300 worsening deltas:

```
T0 = -mean(delta+) / ln(chi0)          chi0 = --t-accept
```

This is Ben-Ameur's rule (2004). Read it as: **χ₀ is literally the probability of
accepting an average-badness worsening move on step 0** — substituting back gives
`exp(−mean(Δ⁺)/T₀) = χ₀`. Two consequences:

* Calibration samples the neighbourhood of the **C&W starting solution only**, so T0
  reflects the initial landscape, not the whole trajectory.
* If no worsening move is found in 2000 draws, `T0 = 0` and **annealing is skipped
  entirely** (src/cw.c:1728). This is the frozen-solution guard for degenerate instances
  — all points coincident, `n = 1`, and similar.

Calibration consumes the same RNG stream the main loop continues from.

### 2.2 The main loop (src/cw.c:1738)

```c
for (it = 0; it < steps; it++) {
    cur += sa_draw(...);                              // one move attempt

    if (split_every && (it+1) % split_every == 0) {
        cur = split_apply(...);
        inc_build(...);                               // O(n) — see caveat 2
    }
    if (cur < best - 1e-12) {
        best = cur;
        snapshot into b_nxt / b_prv / b_rid / b_load;
    }
    T *= alpha;
}
restore best snapshot; inc_build(); return best;
```

Cost is tracked **incrementally** (`cur += delta`). `--check` compares that running
total against a from-scratch `sol_cost()` at the end and reports the drift — this is
what validates the delta formulas.

The best solution ever seen is restored at exit, so annealing can never return
something worse than it started with.

### 2.3 One move

Every operator follows the same skeleton:

```
u = pick_u(...)                first vertex  — biased by regret
v = sa_cand(..., u)            second vertex — uniform among u's kNN
compute delta in O(1)
if (probe) return delta                       // calibration: measure, don't apply
if (!sa_accept(delta, T, rng)) return 0
apply the move
inc_upd() on the 4-6 vertices whose neighbourhood changed
```

**`pick_u`** (src/cw.c:987) — three modes:

| `--pick` | mechanism | cost |
|---|---|---|
| `1` | uniform | O(1) |
| `T >= 2` (default 2) | tournament: draw T customers uniformly, keep the worst `bad[u]` | O(T) |
| `0` | Fenwick tree, exactly proportional to `max(regret,0) + eps0` | O(log n) per draw *and* per weight update |

The tournament caps selection pressure at approximately `T/n` no matter how extreme
a vertex's regret is. The Fenwick sampler does not, which is the mechanism proposed
[below](#exactly-proportional-sampling---pick-0) for why it loses.

**`sa_cand`** (src/cw.c:1010) returns a uniform draw from `u`'s kNN list, falling back to
a uniform customer if `K == 0` or the slot is empty.

**`--pick-crit`** selects the regret formula in `bad_of` (src/cw.c:933). All four are
built on the incrementally-maintained `inc[u]` and clamped at 0:

| name | formula | note |
|---|---|---|
| `lb` (default) | `inc[u] - lb[u]` | gap to the two-shortest-edges bound |
| `rem` | `inc[u] - d(prv[u], nxt[u])` | removal gain — what a relocate would recover |
| `remnorm` | `rem / lb[u]` | normalised by local density |
| `raw` | `inc[u]` | raw cost of the two carried edges |

**`sa_accept`** (src/cw.c:852):

```c
if (delta <= 0.0)      return 1;
if (delta > 36.0 * T)  return 0;       // multiplication, avoids the division
return rng_unit(rng) < exp(-delta / T);
```

`exp(-36) ~ 2e-16` is below double resolution, so the early reject is exact and
removes the division from the dominant case.

### 2.4 The six operators

| operator | intra-route | inter-route | delta | apply |
|---|---|---|---|---|
| `relocate` (src/cw.c:1040) | move one customer | same, with capacity check | O(1) | O(1) |
| `swap` (src/cw.c:1083) | exchange two customers | same, both capacities checked | O(1) | O(1) |
| `2-opt` (src/cw.c:1362) | reverse a segment | 2-opt\* (tail exchange) | O(1) | O(L) |
| `or-opt` (src/cw.c:1148) | move a 2..`--or-max` segment, optionally reversed | same | O(1) | O(L), L <= 8 |
| `swap*` (src/cw.c:1216) | — (inter-route only) | exchange two customers, each reinserted at its **best** position in the other's route | O(L₁+L₂) | O(1) |
| `opening` (src/cw.c:1325) | — | isolate one customer in an empty route | O(1) | O(1) |

Default weights are `1,1,1,0,0,0` — **or-opt, swap\* and opening are off by
default**, so the defaults reproduce the four-operator solver exactly. The first
three together are clearly better than any of them alone (pure descent on
CVRP-100: `1,0,0` → 16.451, `0,1,0` → 16.477, `0,0,1` → 16.460, `1,1,1` → 16.370).

**`swap*`** (Vidal, *Hybrid genetic search for the CVRP*, C&OR 2022) is the only
non-elementary operator. Ordinary `swap` forces `u` into `v`'s slot; `swap*`
drops that constraint and reinserts each customer where it fits best in the
other's route. Because the two routes are disjoint the two reinsertions are
independent, so the delta is still **exact** despite the scan. The second vertex
is drawn *by route*, not by proximity: the kNN neighbour `w0` selects the target
**route**, then `v` is the worse (by regret) of two uniform customers of that
route — requiring `v` near `u` is too restrictive, since `v` will not be placed
near `u` anyway.

Note that for a given pair, the in-place exchange is one of the positions
`swap*` sweeps, so Δ(swap\*) ≤ Δ(swap) always. `swap` is nevertheless kept as a
separate operator (it is ~10× cheaper per draw); whether to drop it is a
weighting decision, not a code one — `--ops 1,0,1,0,1,0.05` drops it.

**`opening`** is almost always worsening, and that is the point. The annealer can
empty a route but never repopulate an empty one, so the number of active routes
is a one-way door between two Splits. This move reopens it, at the cost of the
intermediate states the annealer cannot cross. Use a small weight (0.05).

#### Measured effect

2000 instances of `data/cvrp_100.cvrpb`, `--restarts 10 --split end
--split-every 1000`, 12 threads. Everything is compared **at equal wall time**,
which is the only comparison that means anything: `swap*` costs ~20 % more per
step, so an equal-step win is not a win.

| config | steps | mean cost | wall |
|---|---|---|---|
| baseline `1,1,1,0` | 100 000 | 15.8487 | 7.0 s |
| baseline `1,1,1,0` | 140 000 | 15.8178 | 10.1 s |
| baseline `1,1,1,0` | 200 000 | 15.7870 | 14.9 s |
| `1,1,1,0,1,0` (+ swap\*) | 100 000 | 15.7529 | 8.4 s |
| `1,1,1,0,0,0.05` (+ opening) | 100 000 | 15.8422 | 7.2 s |
| `1,1,1,0,1,0.05` + `--reloc-side long --race 0.002` | 100 000 | **15.7156** | 9.2 s |
| `1,0,1,0,1,0.05` + `--reloc-side long --race 0.002` | 100 000 | **15.7057** | 12.0 s |

The baseline needs 14.9 s to reach 15.787; the new operators reach 15.716 in
9.2 s. Almost all of the gain is `swap*` (−0.096 on its own). `opening` alone is
worth little (−0.006) but is nearly free. Dropping `swap` in favour of `swap*`
(`1,0,1,...`) buys a further −0.010 but costs 30 % more time, so at equal time it
is roughly a wash here — unlike in Yann Vaxès' variant, where `swap` is dropped
outright.

One difference worth flagging: `--vrank 2 --sa-knn 30`, which pays off in his
version, **costs** here (15.735 → 15.787 at equal steps). His construction shares
a single K = 30 kNN list between savings and annealing, whereas this one builds
exact savings for n ≤ 1500, so the two are not tuned against the same baseline.
Left off by default; worth revisiting if the savings K is ever unified.

**The 2-opt trick** (src/cw.c:1385) is the notable one. On a cycle, reversing the segment
`nxt[a]..b` or its complement produces the same undirected cycle: the removed edges
are `(a, nxt[a])` and `(b, nxt[b])`, the added edges `(a,b)` and `(nxt[a], nxt[b])`,
either way. So the delta is symmetric in `a` and `b`, needs no knowledge of their
relative order, computes in O(1), and the reversed segment may contain the virtual
depot harmlessly. Only accepted moves (~2 %) pay the O(L) reversal.

**Reuse of `inc[u]` in the deltas.** Because `inc[u] = d(prv[u],u) + d(u,nxt[u])` is
already maintained for the regret, it appears verbatim in relocate (`d(p,u)+d(u,sc)`)
and swap (`d(pu,u)+d(u,su)`, `d(pv,v)+d(v,sv)`). Reading it instead of recomputing
takes relocate from 6 square roots to 4, and swap from 8 to 4. The substitution is
exact to the bit — the neighbours have not changed since the last `inc_upd`.

## 3. Split

### Where it runs

`--split` is a bitmask (src/cw.c:1933, src/cw.c:1951):

| value | bit 0 — after C&W | bit 1 — after annealing |
|---|---|---|
| `off` | | |
| `cw` | yes | |
| `end` | | yes |
| `both` | yes | yes |

`--split-every N` additionally fires inside the SA loop every N steps.

### The algorithm (src/cw.c:1477)

Prins' DP with Vidal's O(n) sliding-window minimum. With `D[i]` the cumulative demand
up to `t[i]` and `C[i]` the path length `t[1]..t[i]`:

```
P[j] = C[j] + d0(t[j]) + min { F(i) : D[j] - D[i] <= Q }
F(i) = P[i] + d0(t[i+1]) - C[i+1]
```

`F` does not depend on `j`, and since `D` is increasing the feasible window is a
sliding interval — a monotone deque gives O(n) overall.

**Infeasibility guard** (src/cw.c:1522). The sliding window relies on the invariant
"predecessor `j-1` is always feasible", which holds because `D[j] - D[j-1]` is one
customer's demand. If a single demand exceeds capacity that invariant breaks, the
deque empties, and `dq[head]` reads out of bounds. The code checks every demand up
front and declines to split (returns `-1.0`) on an infeasible instance. There is a
redundant guard inside the loop at src/cw.c:1539.

### `--split-tour`

| mode | giant tour construction |
|---|---|
| `routes` | concatenate routes in their current order |
| `sweep` | routes sorted by the polar angle of their centroid, each oriented to minimise the join to the previous one |
| `both` (default) | run both, keep the better |

`split_apply` (src/cw.c:1572) compares the two and, if `sweep` came out worse, re-runs
mode 0 to revert — the operation is idempotent, so this restores the better
partition.

---

## Option reference

### Instance source and geometry
```
--dir DIR --bundle FILE --random        (exactly one)
-n N  -m M  --cap C  --seed S           (generation)
--limit L                               first L instances only
--round                                 integer distances, TSPLIB EUC_2D
```

`--round` applies the TSPLIB EUC_2D convention for testing on CVRPLIB instances; by
default distances stay floating point, which is the convention used by NeuOpt and the
other neural solvers.

### Initial solution
```
--init cw|random   cw (default) = savings construction; random = shuffle the
                   customers and cut into routes at the capacity limit
--knn K        savings truncated to K nearest neighbours (0 = auto)
--exact        full savings list
--lambda L     s_ij = d0i + d0j - L*dij + mu*|d0i - d0j|
--mu M
--2opt         intra-route 2-opt after construction (both init modes)
```

Under `--init random` the savings options have no effect and no savings list is
built. Each restart draws its own permutation, so `--restarts` remains a genuine
multi-start.

### Restarts
```
--restarts R   R C&W constructions, annealed independently, best kept
--cw-rand M    off | perturb (default) | param | both
--cw-alpha A   perturb amplitude (default 0.03)
--race M       racing margin between starts, or off (default off)
--race-at F    fraction of the budget at which the checkpoint sits (default 0.25)
--pair P       0 = off (default) | 1 = on | -1 = auto (n >= 400)
```

**Racing (`--race`)** redistributes the budget instead of spending it equally.
At `--race-at` of its budget, a start that is worse than the best *finished*
start was **at the same point of its own trajectory** (same temperature — the
comparison at equal fraction is what makes it meaningful) by a relative margin
`M` is abandoned. Its unspent steps lengthen the schedules of the following
starts, and whatever is left over at the end polishes the best solution found.
Total step budget is unchanged; only its allocation is.

**Interleaving (`--pair 1`)** advances two starts alternately in the same loop,
so the memory latency of one overlaps with the computation of the other. It is
pure engineering: each chain owns its solution, its incremental buffers and its
random stream, and only transient scratch is shared, so the trajectories are
unchanged bit for bit. The one exception is when `--race` is also on: racing
makes the budget depend on how starts are *grouped*, so `--pair 1 --race M`
follows a different (equally valid) schedule from `--pair 0 --race M`.
Incompatible with `--pick 0`, whose Fenwick tree is shared state.

### SA schedule
```
--sa-steps N   number of steps (default 1000)
--no-sa        disable annealing
--t-accept X   target initial acceptance rate chi0 (default 0.001)
--t-decades D  decades spanned by T (default 2)
--t0 T         fix T0 manually, disables calibration
--tend T       final temperature (default T0 * 1e-4)
```

### SA neighbourhood and selection
```
--ops r,s,t,o[,x,e]  operator weights: relocate, swap, 2-opt, or-opt,
                     swap*, opening        (default 1,1,1,0,0,0)
--or-max L     or-opt max segment length, 2..8 (default 3)
--sa-knn K     candidate neighbourhood size (default 20, 0 = uniform)
--pick T       0 = Fenwick proportional, 1 = uniform, T>=2 = tournament (default 2)
--pick-crit C  lb (default) | rem | remnorm | raw
--pick-eps E   Fenwick smoothing, only used by --pick 0 (default 0.3)
--vrank T      rank bias on the second vertex: the kNN index drawn is the
               min of T uniform draws (default 1 = uniform, no bias)
--pick2 T      tournament of size T on the second vertex, on the regret
               (default 1 = none)
--reloc-side S relocate insertion side: coin (default) | long
```

`--vrank` costs nothing: the kNN lists are already sorted by distance, so the
minimum of T uniform indices is a bias towards the nearer neighbours with no
memory access at all. It is meant to be **coupled with a larger `--sa-knn`**
(the longer list gives the reach, the rank bias restores the concentration);
either one alone tends to hurt.

`--reloc-side long` breaks the longer of the two edges adjacent to `v` instead
of flipping a coin, which maximises the `-d(v,q)` term of the insertion cost.
Two reads, no randomness; the diversity of insertion positions is still carried
by the draw of `v` among the kNN.

### Split
```
--split M        off (default) | cw | end | both
--split-tour T   routes | sweep | both (default)
--split-every N  also apply every N annealing steps
```

### Output and diagnostics
```
--sol F          write all solutions (self-describing text)
--dump-bundle F  write the instances actually used, as .cvrpb
--validate F     re-read a solution file and check it; solves nothing
--csv F          per-instance CSV
--per-instance   print each instance's cost
--check          report incremental-cost drift
--threads T  -q
```

---

## Simulated annealing

Measurements behind the defaults above.

### Temperature: automatic calibration

Sweeping (χ₀, decades) on CVRP-100 (300 instances):

| χ₀ \ D | 2 | 3 | 4 | | χ₀ \ D | 2 | 3 | 4 |
|---|---|---|---|---|---|---|---|---|
| **10 000 steps** | | | | | **100 000 steps** | | | |
| 0.001 | **16.415** | 16.405 | 16.407 | | 0.001 | **16.203** | 16.248 | 16.265 |
| 0.01  | 16.453 | 16.441 | 16.426 | | 0.01  | 16.232 | 16.238 | 16.267 |
| 0.05  | 16.520 | 16.490 | 16.468 | | 0.05  | 16.279 | 16.318 | 16.337 |
| 0.20  | 16.575 | 16.566 | 16.550 | | 0.20  | 16.323 | 16.379 | 16.406 |

The optimal schedule is therefore **almost greedy**: the Clarke & Wright solutions are
already close to local optima, the neighbourhood is small, and the budget is better
spent descending than exploring. Defaults chosen: χ₀ = 0.001, D = 2 (verified stable
on CVRP-20/100/200).

At very small budgets the effect is even sharper — at 1000 steps on CVRP-100, pure
descent (`--t0 1e-9 --tend 1e-9`, 16.223) beats the calibrated schedule (16.253),
which in turn beats a hot start (χ₀ = 0.5, 16.284 ≈ no annealing at all). With
n = 100 and K = 20 the neighbourhood holds ~2000 candidate moves, so 1000 steps does
not sample it even once and there is no time to recover from a detour.

### Results with the default settings

1000 steps, three neighbourhoods, calibrated T₀ — full sets:

| set | before annealing | after annealing | gain | ms/inst. |
|-----|-----------------:|----------------:|-----:|---------:|
| CVRP-20  | 6.36357  | 6.29754  | −1.04 % | 0.118 |
| CVRP-50  | 10.89819 | 10.85132 | −0.43 % | 0.194 |
| CVRP-100 | 16.49677 | 16.46045 | −0.22 % | 0.396 |
| CVRP-200 | 23.36610 | 23.33583 | −0.13 % | 0.956 |

With `--sa-steps 2000n`, 1000 instances per set:

| set | steps | before | after | gain | gap /HGS | ms/inst. |
|-----|------:|-------:|------:|-----:|---------:|---------:|
| CVRP-20  |  40 000 | 6.40660  | 6.20740  | −3.11 % | +1.3 % | 2.2  |
| CVRP-50  | 100 000 | 10.99155 | 10.66435 | −2.98 % | +2.9 % | 5.5  |
| CVRP-100 | 200 000 | 16.52124 | 16.07931 | −2.67 % | +3.3 % | 11.1 |
| CVRP-200 | 400 000 | 23.36610 | 22.70366 | −2.84 % | +4.4 % | 23.0 |

The gain is uniform across sizes (2.7 to 3.1 %), a sign that the calibrated schedule
adapts well to the scale of the instances.

### What has been verified

* Pure descent (χ₀ → 0) reaches **exactly** the local optimum computed by an
  independent exhaustive local search in Python (`localopt.py`, relocate + swap +
  2-opt + 2-opt\*) on the instances tested. That local optimum is only about 1 %
  better than Clarke & Wright, and on 3 of 6 CVRP-20 instances C&W is already a local
  optimum: the C&W solutions are robust to these neighbourhoods.
* `--check` reports the drift between the incrementally tracked cost and the cost
  recomputed from scratch. Over tens of millions of moves, the maximum drift observed
  is 2e-13: the delta formulas of the four moves are exact.
* Every solution is revalidated; clean ASan/UBSan build on every path.

### Does the construction matter? (`--init random`)

`--init random` replaces Clarke & Wright with a *feasible* random start: shuffle
the customers, cut the sequence whenever the next one would overflow the
vehicle. Feasible by construction, because the annealing operators reject every
capacity violation and so could never repair an infeasible start. First-fit on a
random permutation also lands close to C&W's route count (10.7 against 10.6), so
the two starts differ in quality rather than in shape.

CVRP-100, 1000 instances, `--split end`:

| steps | C&W start | random start | gap | wall (C&W) |
|---:|---:|---:|---:|---:|
| — (start) | 16.52124 | 57.87344 | ×3.5 | |
| 20 000 | **16.27537** | 17.07014 | +0.795 (4.9 %) | 0.09 s |
| 200 000 | **16.04741** | 16.33386 | +0.286 (1.8 %) | 0.63 s |
| 1 000 000 | **15.88306** | 16.07648 | +0.193 (1.2 %) | 3.07 s |

The random start begins **3.5× worse** and the annealer recovers almost all of
it. What remains is a real but shrinking edge: 4.9 % at 20k steps, 1.2 % at 1M,
still narrowing. On a paired per-instance comparison at 200k steps (40 instances,
`./cw_trace`) the difference is −0.2298 with se 0.0586, |t| = 3.92, C&W better on
29 of 40 — so the residual gap is not noise either.

Read together: **Clarke & Wright buys time far more than quality.** It reaches in
0.1 ms a solution the annealer needs thousands of steps to match, and keeps a
small quality premium that the budget keeps eroding.

The per-operator trace explains the mechanism. From a random start every operator
descends (relocate −26.3, swap −12.0, 2-opt −9.1 summed delta on one instance);
from the C&W start only relocate descends (−1.02) while swap (+0.42) and 2-opt
(+0.08) are net uphill. C&W has already consumed the easy descent — which is the
measured form of the claim, made elsewhere in this README, that its solutions sit
close to a local optimum.

## Guided selection of the vertex to move

The second vertex of a move is already drawn from the k nearest neighbours of the
first. `--pick T` additionally biases the draw of the **first** vertex towards those
that are badly placed. For each customer u:

    lb[u]  = sum of the two smallest distances that can be incident to u
             (among the depot and its k nearest neighbours)
    inc[u] = cost of the two edges u currently carries
    regret(u) = inc[u] − lb[u]

`lb[u]` is a lower bound valid for any solution (the same argument as in the 1-tree
bound), so the regret measures exactly by how much u's current neighbourhood exceeds
the best conceivable one. The draw is a **tournament of size T**: T customers are
drawn uniformly and the worst is kept. T = 1 gives back uniform sampling; the
probability of drawing the customer of rank r is ≈ T(1−r/n)^(T−1)/n.

The performance point: evaluating `inc[u]` at each draw would cost two square roots
per candidate, i.e. +60 % time per step. `inc[]` is therefore **maintained
incrementally** — only 4 to 6 vertices change neighbourhood per *accepted* move, and
the acceptance rate is on the order of 2 %. With two indices additionally drawn from
a single 64-bit random word (Lemire's method), the overhead falls to 8 % for T = 2.

| | CVRP-50 (2000 inst.) | CVRP-100 (1000 inst.) | CVRP-200 (1000 inst.) |
|---|---|---|---|
| T = 1 | 10.68082 — 1.17 ms | 16.14698 — 5.62 ms | 22.93246 — 6.11 ms |
| **T = 2** | **10.66380** — 1.26 ms | **16.12443** — 6.07 ms | **22.89536** — 6.59 ms |
| T = 3 | 10.66384 — 1.35 ms | 16.12664 — 6.26 ms | 22.90392 — 6.81 ms |
| T = 4 | | 16.13787 — 6.34 ms | |

The gain is 0.14 to 0.16 % at equal step count, and about 0.10 % at equal time
(T = 1 with 115 000 steps gives 16.14299 against 16.12443 for T = 2 with 100 000
steps). Modest but systematic across the three sizes, and never unfavourable at
T = 2, which is the default. Beyond T = 4 the selection becomes too deterministic and
concentrates on the same vertices: at T = 16 we lose 1 % relative to uniform.

### Exactly proportional sampling: `--pick 0`

To check whether it is the *shape* of the bias or its *strength* that limits the gain,
mode `--pick 0` replaces the tournament with sampling exactly proportional to
`max(regret(u), 0) + ε`, via a Fenwick tree. The sampler was validated statistically
(4 M draws, n = 97: maximum frequency/weight deviation of 1.8 %, consistent with
sampling noise).

CVRP-100, 1000 instances, 100 000 steps:

| selection | cost | ms/inst. |
|---|---:|---:|
| uniform (T = 1) | 16.14762 | 5.50 |
| **tournament T = 2** | **16.12480** | 6.00 |
| Fenwick, ε = 0.02 | 16.14229 | 9.04 |
| Fenwick, ε = 0.1 | 16.13780 | 9.13 |
| Fenwick, ε = 0.3 | 16.13166 | 9.17 |
| Fenwick, ε = 1.0 | 16.13359 | 9.38 |
| Fenwick, ε = 3.0 | 16.13480 | 9.69 |

CVRP-200, 1000 instances, 100 000 steps: uniform 22.93209 (6.13 ms), tournament T=2
22.89660 (6.68 ms), Fenwick ε=0.3 **22.93620** (10.61 ms) — that is, *worse than
uniform*, for 73 % more time. At equal time on CVRP-100, the tournament crushes it:
16.07557 (T=2, 153 000 steps) against 16.13166 (Fenwick, 100 000 steps).

Proportional sampling is therefore a failure, on two counts. The cost first: the
descent through the tree is a 7-iteration loop with chained dependencies, i.e. +65 %
time per step where the tournament costs +8 %. But above all the quality: the best ε
is the one that **flattens** the distribution the most, and the ranking
ε = 0.3 > ε = 0.1 > ε = 0.02 points the same way as the degradation of the tournament
beyond T = 4. The explanation probably lies here: the concentration of proportional
sampling is unbounded, so a vertex with a high regret is drawn again and again. But
`lb[u]` is a bound that may be *unattainable* — a customer isolated in a sparse area
keeps a high regret whatever one does, and absorbs the budget indefinitely. The
tournament caps the selection pressure at ≈ T/n whatever the magnitude of the regret.

### Which definition of regret? (`--pick-crit`)

`rem` is the only *attainable* criterion: it is 0 as soon as u is aligned between its
neighbours and 2·d(0,u) if it is alone on its route. It was the natural candidate.
The measurements say otherwise (1000 instances, 100 000 steps):

| | uniform | `lb` | `rem` | `raw` |
|---|---:|---:|---:|---:|
| CVRP-100, T = 2 | 16.14637 | **16.12485** | 16.14199 | 16.12728 |
| CVRP-100, T = 3 | 16.14637 | **16.12017** | 16.14594 | 16.12112 |
| CVRP-200, T = 2 | 22.93339 | **22.89257** | 22.91214 | 22.90278 |
| CVRP-200, T = 3 | 22.93339 | **22.90397** | 22.92051 | 22.90544 |

`rem` does barely better than uniform, and neither does `remnorm` (16.14116 and
22.91633 at T = 2): normalising by the local density was therefore not the
explanation. `raw`, on the other hand, does almost as well as `lb` — in other words
**what predicts that a vertex deserves to be moved is simply that it carries long
edges**, the lb[u] correction adding only a small supplement.

The likely interpretation: `rem` is small as soon as d(p,q) is large, that is,
precisely when u sits in the middle of a long jump — yet those are the areas where
there is most to gain. And `rem` is large for a customer zigzagging locally between
two close neighbours, an inexpensive case that 2-opt settles on its own. The removal
gain measures what is recovered by *taking u out*, not what is gained by reworking
the region it sits in — and it is that second quantity that matters.

Three successive hypotheses (unattainable bound, normalisation, attainable gain) were
refuted by measurement; the default criterion stays `lb`, chosen empirically.

## Multiple restarts

Diversification (`--cw-rand`) acts on the order in which pairs of customers are
considered:

* `perturb` (default) — each saving is multiplied by `1 + α·U(−1,1)` before the sort,
  with α = `--cw-alpha`. This is exactly a perturbation of the traversal order of the
  savings list, hence of the order in which customers are merged.
* `param` — draws λ ∈ [0.75, 1.25] and μ ∈ [0, 0.3] (Yellow savings), which changes
  the shape of the routes rather than the order.
* `both` — both; `off` — none (restarts then differ only by the annealing seed).

The kNN neighbourhood is computed once, not per restart.

### Tuning α

5 restarts × 10 000 steps, CVRP-100 (200 instances):

| α | final cost | mean C&W over the restarts |
|---|---:|---:|
| 0.01 | 16.135 | 16.506 |
| 0.02 | 16.119 | 16.577 |
| **0.03** | **16.111** | 16.657 |
| 0.05 | 16.145 | 16.870 |
| 0.12 | 16.244 | 17.802 |

The classic trade-off: too little perturbation does not diversify, too much degrades
the starting points faster than the annealing can recover. α = 0.03 by default. The
modes compared at that setting: `off` 16.220, `param` 16.148, `perturb` 16.111,
`both` 16.103 — randomising C&W therefore brings appreciably more than simply
repeating the annealing with different seeds.

Note that with `--restarts R > 1` the reported *pre-annealing* mean rises (e.g.
16.49677 → 16.65920 on CVRP-100), because it averages over deliberately perturbed
constructions. Restart 0 is still exactly the deterministic C&W.

### Budget allocation

At a constant total budget of 100 000 steps, CVRP-100 (200 instances):

| configuration | cost |
|---|---:|
| 1 × 100 000 | 16.106 |
| 2 × 50 000 | 16.081 |
| **5 × 20 000** | **16.059** |
| 10 × 10 000 | 16.070 |
| 20 × 5 000 | 16.084 |

The optimum is flat around 5 to 10 restarts. And on the same subset:

| configuration | cost | ms/inst. |
|---|---:|---:|
| C&W alone | 16.475 | 0.1 |
| 1 restart × 100 000 steps | 16.106 | 5.8 |
| 1 restart × 1 000 000 steps | 15.883 | 58 |
| 10 restarts × 100 000 steps | 15.866 | 58 |
| 10 restarts × 100 000 + split | 15.864 | 58 |

At equal budget (10⁶ steps in total), the multi-restart does slightly better than a
single very long annealing, for an identical cost — and it parallelises trivially.

## Optimal Split (Vidal 2016)

### What it gives: almost nothing

Full sets, default annealing:

| set | without split | `--split end` | `--split both` |
|-----|--------------:|--------------:|---------------:|
| CVRP-20  | 6.307958  | 6.304989  | 6.302953  |
| CVRP-100 | 16.468427 | 16.467656 | 16.467840 |
| CVRP-200 | 23.343785 | 23.343559 | 23.343825 |

On CVRP-100 (300 instances):

| configuration | cost |
|---|---:|
| C&W alone | 16.581046 |
| C&W + split (`routes`) | 16.581046 &nbsp;*(gain exactly zero)* |
| C&W + split (`sweep`) | 16.580657 |
| annealing 20 000 steps | 16.353295 |
| + final split (`both`) | 16.351219 |
| + split every 2000 steps | 16.332566 |
| annealing 200 000 steps | 16.116313 |
| + final split + every 1000 steps | 16.108031 |

Concatenating the routes in their current order, the gain is **exactly zero**, and
that is structural: moving a cut amounts to attaching the end of one route to the
start of another, yet those two routes have no reason to be neighbours. The angular
sort (`sweep`) makes the tour plausible and unlocks a gain… of 0.002 %.

The underlying explanation: Split optimises the **partition** at a **given order**,
yet C&W and then the annealing operators (relocate, swap and above all 2-opt\*)
already move customers between routes — so the partition is already locally optimal
by the time Split arrives. Its real strength, in HGS, comes from applying it to giant
tours produced by crossover, which derive from no pre-existing feasible partition.
Here it costs O(n) and never degrades, but it brings nothing measurable; the periodic
application during annealing is the only setting that stands slightly out of the
noise, and only at small budgets.

### Verifying Split

`tests/checksplit.py` compares the O(n) Split against a naive O(n²) dynamic program (Prins
recurrence) on exactly the same giant tour: exact agreement (max relative gap 5e-12)
on CVRP-50/100/200, both on raw C&W solutions and on annealed ones.

---

## Writing and validating solutions

```bash
./cw --bundle data/cvrp_100.cvrpb --sol sol.txt      # writes every solution
./cw --bundle data/cvrp_100.cvrpb --validate sol.txt # re-reads and checks
python3 tools/validate.py data/cvrp_100.cvrpb sol.txt      # independent validation
python3 tools/validate.py results/<run>                    # a run directory (run.py)
```

`--sol` writes the self-describing text format shown
[above](#on-output--flat-array-then-text). Solutions are buffered and written
serially after the parallel loop, so the file is identical whatever `--threads` is.

`--dump-bundle F` writes the instances actually used in `.cvrpb` format, including
those generated by `--random`. This is what makes the solution file verifiable by a
third party without replaying the generator.

`--validate F` solves nothing: it re-reads the file and, for each instance, checks
coverage and capacities and **recomputes each cost from the coordinates** to compare
it with the reported value. Exit code 3 on any error. `validate.py` performs the same
checks in Python without sharing a single line with the solver — that is the version
that carries evidential weight.

Given a run directory, `validate.py` locates the instances by itself
(`instances.cvrpb` if present, otherwise the `--bundle` recorded in `config.json`),
cross-checks the recomputed mean against the cost the solver reported, prints the
gaps to HGS and LKH-3 from `baseline/baseline.csv`, and reports the solver's wall
time alongside its single-core equivalent:

```
run 20260731-172943_N100_1M_10R
  instances : data/cvrp_100.cvrpb
10000 instance(s) checked, 0 error(s)
  max relative gap reported / recomputed cost: 4.326e-16
  mean cost : 15.679491
  matches the run summary (15.67949)
  gap to HGS [21]   (15.563): +0.75 %
  gap to LKH-3 [20] (15.647): +0.21 %
  solver time : 312.735 s wall, 3716.558 s single-core, speedup 11.9x on 12 threads
                31.2735 ms/instance wall, 371.656 ms/instance single-core
```

The mean-cost cross-check catches a failure mode the per-instance checks structurally
cannot: pairing the *wrong* bundle with a solution file. If both are internally
consistent, every per-instance check passes and the cost merely belongs to a
different experiment.

### Does the validator actually detect anything?

Check on four deliberate corruptions of a valid file:

| corruption | message |
|---|---|
| one customer removed from a route | `inst 0: 99 customers served out of 100` + cost gap 2.7e-06 |
| one customer duplicated | `inst 0: customer 90 served twice` |
| cost falsified from 15.74 → 14.74 | `reported cost 14.7404, recomputed 15.7404 (gap 6.78e-02)` |
| all routes merged | `route 0 overloaded (473 > 50)` + cost gap |

On the intact file, both validators report 0 errors and a maximum relative gap of
2.7e-16 between reported and recomputed cost, that is, floating-point rounding and
nothing else.

---

## Hot-path optimisation

The annealing consumes almost all the time (Clarke & Wright costs 0.1 ms out of the
6 to 60 ms of an instance). Four optimisations, measured one by one with `gprof` then
timed:

**1. Intra-route 2-opt does not need to know the order** — the O(1) symmetric delta
described [above](#24-the-four-operators). 2-opt alone goes from 15.39 to 12.62 ms,
i.e. −18 %.

**2. Reuse of `inc[u]`**, taking relocate from 6 square roots to 4 and swap from 8 to
4. Exact to the bit, confirmed by the incremental-cost drift staying at 1.7e-13.

**3. Less randomness.** The PRNG weighed ~25 % of the per-step cost, with 4 to 5 calls
to xoshiro256\*\*. One 64-bit draw now serves two 32-bit integers (`rng32`), indices
go through a multiply-shift with no rejection loop (`rng_idx`), coin flips draw from a
64-bit reservoir (`rng_bit`), and the operator is chosen by integer comparisons
against precomputed thresholds instead of a float. We go from ~5 to ~2 calls per step.

**4. Details.** Interleaved `(x,y)` coordinates per vertex with materialised virtual
depots; and the acceptance test compares `delta > 36·T` (a multiply) before performing
the division `delta/T`, which therefore no longer happens in the dominant case.

Result, at identical quality (1000 instances, 100 000 steps):

| set | before | after | gain | cost before | cost after |
|-----|-------:|------:|-----:|------------:|-----------:|
| CVRP-20  | 6.708 ms | 5.458 ms | −18.6 % | 6.185245 | 6.185008 |
| CVRP-50  | 6.596 ms | 5.649 ms | −14.4 % | 10.652965 | 10.651224 |
| CVRP-100 | 6.952 ms | 6.060 ms | −12.8 % | 16.124849 | 16.121405 |
| CVRP-200 | 7.879 ms | 6.988 ms | −11.3 % | 22.892574 | 22.895346 |

The cost differences are noise: since the consumption of randomness changed, the
trajectories differ, but the mean quality is identical.

**Interleaving the link fields is useless.** `nxt`, `prv` and `rid` are three separate
arrays, so visiting a vertex touches three cache lines. Grouping them into a 16-byte
record per vertex would bring that down to one — the obvious optimisation, and it
gains nothing:

| | CVRP-100 | CVRP-200 | n=2000 | n=10000 | n=40000 |
|---|---:|---:|---:|---:|---:|
| 3 arrays | 5.234 ms | 5.904 ms | 13.24 ms | 48.76 ms | 202.9 ms |
| records | 5.246 ms | 5.954 ms | 14.33 ms | 49.84 ms | 199.5 ms |
| gain | −0.2 % | −0.8 % | −8.2 % | −2.2 % | +1.7 % |

Two reasons. Padding to 16 bytes inflates the footprint by 33 % relative to three
`int`, which cancels out the "one line instead of three". And above all, at large
sizes the bottleneck is not vertex access during annealing but construction: at
n=65535, `knn_build` costs 126 ms and the radix sort 70 ms, against 100 ms for
100 000 annealing steps.

**Side effect: FMA contraction.** Comparing the two versions, the costs differed
slightly although the sources were semantically identical. The cause is
`-ffp-contract=fast`, GCC's default in C, which fuses `a*b+c` into an FMA —
different rounding — in a way that depends on the shape of the code. A `dxy` computed
with FMA here and without FMA there flips a borderline comparison in the acceptance
test, and the trajectories diverge. With `-ffp-contract=off` the two versions become
bit-identical again, which incidentally served as a proof of correctness for the
refactoring.

Practical consequence: **results are not reproducible from one compiler or set of
flags to another.** The `make repro` target disables contraction; it costs 22 % at
n=100 (5.34 → 6.53 ms) and nothing at n=10000. Use it for comparing algorithm
variants across machines, not for producing results. This is also why `run.py`
records an md5 of the binary in every `config.json`.

**What did not work either.** Profile-guided optimisation (`-fprofile-use`) degrades
slightly (12.03 ms against 11.77). The rest of the time is spread between `mv_2opt`
26 %, `mv_swap` 24 %, `mv_relocate` 18 % and the annealing loop 16 %, mostly chains of
dependent loads (`prv[u]`, then `xy[2·p]`) inherent to the linked list representation.

## Audit and testing

`tests/mkedge.py` builds fifteen edge-case instances the random generator never produces:

| instance | what is degenerate |
|---|---|
| `tiny1/2/3/5` | n = 1, 2, 3, 5 — boundary cases for loops assuming ≥2 customers |
| `same10/800` | every point coincident → all distances 0, every saving filtered out |
| `line10/800` | collinear points → degenerate grid, many exactly-tied savings |
| `clusters` | two tight clusters → whole rows of empty grid cells for the ring scan |
| `alone` | Q = 9 with all demands 9 → every customer on its own route |
| `onebig` | Q = 10⁶ → everything in a single route |
| `frac` | fractional demands against Q = 7.5 — exercises the `EPS` tolerances |
| `huge` | coordinates scaled by 10⁶ |
| `depotdup` | a customer placed exactly on the depot → zero-length edge |
| `infeasible` | one demand of 99 against Q = 30 → **no valid solution exists** |

`tests/fuzz.py` draws random option combinations — 25 options, including every enumeration
and extreme values of each parameter — runs them on these instances plus the NeuOpt
sets and random instances, and for each run checks the exit code, the absence of
output on stderr, the incremental-cost drift, then **revalidates every solution
written with `validate.py`**.

Outcome: 250 trials with the optimised binary and 90 under ASan/UBSan, zero failures.

### What the fuzzing found

**Out-of-bounds read in Split, on an infeasible instance** — the guard now at
src/cw.c:1522. A segfault reproducible on any run combining `--split` with an infeasible
instance. It was the textbook example of the implicit assumption one does not think to
test: the reasoning was correct *on a feasible instance*.

**`--round` silently ignored.** The option was applied only to the TSPLIB files read
by `--dir`; on a `.cvrpb` bundle or on random instances, the solver kept using
floating-point distances while still accepting the option. The bug was invisible
internally — the built-in validator used the same flag as the solver — and only
appeared against `validate.py`, which did recompute in integers. This is the argument
for a validator that shares no code. `--round` now applies to every source, the
solution file carries a `#round` line, and both validators respect it.

**Division by zero when there is no instance at all** (`-m 0`, `--limit 0`, empty
bundle): the program printed `-nan` instead of an error message.

### What held up

The degenerate instances all pass: all distances zero, collinear points, depot
coincident with a customer, n = 1. The infeasible instance is correctly reported
(exit code 2, `feasible=0` in the CSV) without being confused with an error. The
command-line guards reject `--restarts 0`, `--or-max 1`, `--ops 0,0,0,0`,
`tend > t0`, and `--t-accept` outside ]0,1[.

Finally, the default costs moved in the last digit after these fixes (CVRP-20:
6.29754 → 6.29810) although no change touches the default path. On checking, this is
again FMA contraction: compiled with `make repro`, the binaries from before and after
the audit give exactly 16.250595 on the same set.

## Implementation choices

* **Savings list truncated to the K nearest neighbours.** Exact (all pairs) up to
  n = 1500, otherwise K = 32 by default. Neighbours are obtained via a uniform grid
  scanned in rings, stopping as soon as the radius guarantees no remaining point can
  enter the heap — O(n) in practice instead of O(n²). At n = 5000: 13 ms instead of
  695 ms, for only +0.37 % cost. (Truncation can even *improve* the result: at
  n = 2000, K = 48 gives 78.100 against 78.132 exact — it filters out long-edge merges
  that C&W would accept.)
* **LSD radix sort, 4×8 bits** on the float key. The savings kept are > 0, so the bit
  patterns of the `float` are ordered like the values; the sort is stable, so the
  result is deterministic.
* **Merging via adjacency lists + union-find**, as described
  [above](#during-clarke--wright--adjacency-lists--union-find).
* **OpenMP parallelism at the instance level**, buffers allocated once per thread and
  reused: zero `malloc` in the hot loop.
* **Parameterised savings** (Yellow): `s_ij = d0i + d0j − λ·dij + μ·|d0i − d0j|`, via
  `--lambda` / `--mu` (λ = 1, μ = 0 by default = classic C&W).
* **Intra-route 2-opt** optional (`--2opt`), first improving move, to a local optimum.
  Negligible cost, gain of 0.3 to 0.6 percentage points.

Limit: n ≤ 65535 (indices i, j are packed into 16 bits each to fit in an 8-byte
record). Throughput measured at K = 32: 65 535 customers in 21 ms on one core.

---

## Known caveats

Points where the code and its stated intent do not quite line up. None of them
affects the validity of the results above.

### 1. Emptied routes can never be refilled

The comment at src/cw.c:764 states that an empty route "stays available for a later
reinsertion". The move set does not deliver this.

`sa_cand` only ever returns a **customer** — the kNN lists contain customers only. In
relocate and or-opt the insertion point is `v` or `prv[v]`; `prv` of a customer is
either another customer or the virtual depot of that customer's route, which is
necessarily non-empty. So `rid[v]` is never an empty route. Swap preserves route
sizes. 2-opt\* maps two routes to two routes and *can* empty one (when `ia == -1` and
`ib == LB-1`, src/cw.c:1431).

Net effect: **the route count is monotone non-increasing during annealing.** Only
Split can raise it again (src/cw.c:1567). For distance-minimising CVRP this is rarely
costly — C&W's route count is generally at or above the optimum — but the SA cannot
explore any solution using more vehicles than C&W produced.

### 2. `--split-every` breaks the O(1)-per-step design

Each trigger runs `split_apply` **and** `inc_build`, both O(n) (src/cw.c:1744). At
`--split-every 1` the loop costs O(n) per step. The finding that periodic Split
"stands slightly out of the noise" is measured in *steps*; at equal *time* it would
look worse still.

### 3. `--ops r,s,t,0` does not strictly disable or-opt

With `w_or`, `w_sstar` and `w_open` all <= 0, src/cw.c:1687 sets `th3 = 0xFFFFFFFF`, and `sa_draw` tests `z < th3`.
The single value `z == 0xFFFFFFFF` still falls through to or-opt — roughly once per
4 × 10⁹ draws. Harmless, since the operator is correct and `or_max` defaults to a
valid 3, but "disabled" is off by one.

### 4. Tie-breaking differs between `cw.c` and `check.py`

Two independent sources of divergence that the reported 2.5e-12 agreement attributes
solely to float addition order:

* `cw.c` truncates each saving to **float32** before sorting (src/cw.c:1859), so two
  savings differing by less than ~1e-7 relative become exact ties in C but not in
  `check.py`'s float64 sort.
* `cw.c` iterates the stable radix sort **backwards**
  (`for (size_t t = m; t-- > 0; )`, src/cw.c:1892), consuming ties in reverse insertion
  order; `check.py` sorts by `-s` with Python's stable sort, consuming ties in forward
  `(i,j)` order.

On random uniform instances ties are effectively measure-zero, so the test passes. On
degenerate geometry — `line10` / `line800`, where many savings are genuinely equal —
the two implementations may legitimately diverge, and nothing in the suite
distinguishes that from a bug. `fuzz.py` runs the edge instances but only through
`validate.py`, which checks feasibility and cost-consistency, never agreement with the
reference implementation.

### 5. "Split can never degrade" is stated slightly too broadly

It holds for the `routes` concatenation — the current partition is itself a feasible
cut of its own concatenation — but not for `sweep`, which builds a different giant
tour whose optimal split may exceed the current cost. `split_apply` handles this by
comparing and reverting (src/cw.c:1584), so the *net* behaviour is safe; the invariant as
written just does not cover the `sweep` path.

---

## Files

| file               | role                                                        |
|--------------------|-------------------------------------------------------------|
| `src/cw.c`         | the solver                                                   |
| `src/trace.c`      | instrumented single-instance driver (cost trajectory, per-operator counts) |
| `Makefile`         | `make`, `make macos` (Apple clang + libomp), `make serial` (no OpenMP), `make repro` (bit-reproducible), `make debug` (ASan/UBSan) |
| `tools/run.py`     | runs `./cw` in a self-contained run directory (config + results) |
| `tools/fetch_neuopt.py` | download + conversion `.pkl` → `.cvrpb` (and `.vrp`)          |
| `tools/validate.py` | standalone validation of a solution file, plus gaps to HGS/LKH-3 |
| `tools/analyze.py` | statistics and plots for a run, a comparison, or a trace      |
| `tools/run_hgs.py` | runs HGS-CVRP into the same run-directory layout, so `validate.py` checks it too |
| `tools/bundle_to_vrp.py` | `.cvrpb` → TSPLIB `.vrp`, for external solvers                |
| `tools/compare_table.py` | time-quality table across runs, ranked on CPU per instance |
| `tools/paired_gap.py` | paired per-instance comparison of two runs (SE, CI, sign test) |
| `tools/compare_hgs.sh` | the SAVANT-vs-HGS budget sweep, end to end                   |
| `tests/check.py`   | cross-check of C&W against a naive Python implementation      |
| `tests/checksplit.py` | validation of the O(n) Split against a naive O(n²) DP         |
| `tests/localopt.py` | exhaustive reference local optimum, to validate the descent   |
| `tests/mkedge.py`  | generation of edge-case instances                             |
| `tests/fuzz.py`    | fuzzing of the option space with validation of every solution |
| `baseline/`        | published baselines (`baseline.csv`) and their LaTeX rendering |
# SAVANT
