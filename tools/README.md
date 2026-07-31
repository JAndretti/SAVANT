# `tools/` — things you run

Day-to-day entry points. All three are pure standard library, so
`python3 tools/<script>.py` works without installing anything.

Each resolves the repository root from its own location (`ROOT = dirname(HERE)`),
so they behave identically from any working directory.

| file | role |
|---|---|
| `run.py` | drives `./cw` into a self-contained run directory |
| `validate.py` | independent solution checker, plus gaps to HGS / LKH-3 |
| `fetch_neuopt.py` | downloads and converts the NeuOpt CVRP test sets |
| `analyze.py` | statistics and plots for one run, or a comparison across runs |

`analyze.py` needs `matplotlib` and `numpy` (`uv sync`); the other three are pure
standard library.

---

## `fetch_neuopt.py` — get the benchmark data

```bash
python3 tools/fetch_neuopt.py                      # all 4 sizes
python3 tools/fetch_neuopt.py --sizes 100 200
python3 tools/fetch_neuopt.py --sizes 200 --tsplib --max 50
```

Downloads `cvrp_{20,50,100,200}.pkl` from the `yining043/NeuOpt` repository and
converts them to `data/cvrp_<n>.cvrpb`, a little-endian binary bundle the solver
reads instantly:

```
'CVRPBIN1' | u32 count | u32 0
per instance:  u32 n | f64 Q | f64 x[n+1] | f64 y[n+1] | f64 dem[n+1]
```

Index 0 is the depot. Coordinates stay floating point in [0,1] — the solver does
not round them unless `--round` is given — so costs are directly comparable with
the published values. `--tsplib` additionally writes one `.vrp` file per
instance. The `.pkl` files are deleted after conversion unless `--keep-pkl`.

Sizes and capacities: n = 20 → Q = 30, n = 50 → Q = 40, n = 100 → Q = 50
(10 000 instances each), n = 200 → Q = 70 (1 000 instances).

---

## `run.py` — one directory per experiment

```bash
python3 tools/run.py --name my_run --bundle data/cvrp_100.cvrpb \
        --sa-steps 200000 --restarts 5 --split end --check
```

Every argument it does not recognise is passed straight through to `./cw`, so
there is no second option syntax to learn. It produces:

```
results/<timestamp>[_<name>]/
├── config.json      parameters, resolved defaults, binary fingerprint, parsed summary
├── run.log          raw cw output
├── results.csv      per-instance CSV (--csv)
├── solutions.txt    every solution (--sol)
└── instances.cvrpb  only when the source is --random (--dump-bundle)
```

`config.json` is the useful part. Besides the command line it records:

* **`resolved`** — `cw`'s own header, i.e. the *effective* configuration including
  every default you did not set. Six months later the command line alone will not
  tell you what the defaults were.
* **`binary`** — md5, size, mtime, and whether it is linked against OpenMP.
  Results depend on compilation flags (see [`src/README.md`](../src/README.md)),
  so two runs with the same command line are only comparable with the same md5.
* **`result`** — the summary parsed into typed JSON (`cost_after_sa`,
  `time_wall_s`, `drift_max`, …), so runs can be compared with a glob and a
  `json.load` instead of re-parsing text.

Guards: `--csv`, `--sol` and `--dump-bundle` are injected and rejected if you
pass them by hand (they would break self-containment); `-q` is rejected because
it suppresses the summary that gets recorded. `--dry-run` prints the directory
and command without running anything.

`--random` runs automatically get `--dump-bundle`, since generated instances
would otherwise be unreproducible from the directory alone.

---

## `validate.py` — the independent checker

```bash
python3 tools/validate.py results/<run>                     # run directory
python3 tools/validate.py data/cvrp_100.cvrpb solutions.txt # explicit files
python3 tools/validate.py results/<run> --tol 1e-12 -v
```

**Shares no code with the solver.** Instances are re-read from the `.cvrpb`
bundle and every distance is recomputed in Python with `math.hypot`. That
independence is not decoration: it is what caught the `--round` bug, where the
solver ignored the flag on `.cvrpb` inputs and the *built-in* validator agreed
with itself because it used the same distance function.

Per instance it checks: the instance exists in the bundle; the declared `n` and
`Q` match it; every customer index is in range; no customer is served twice;
every customer in 1..n is served; no route exceeds capacity; and the cost
recomputed from the coordinates matches the reported one within `--tol`
(default 1e-9). The `#round` flag in the solution file is honoured, so integer
distances are recomputed as integers.

Given a run directory it also locates the instances by itself (`instances.cvrpb`
if present, otherwise the `--bundle` recorded in `config.json`), cross-checks the
recomputed mean against the cost the solver reported — which catches pairing a
valid solution file with the *wrong* bundle — and reports the gaps to HGS and
LKH-3 from `baseline/baseline.csv` plus the solver's wall and single-core times.

```
run 20260731-172943_N100_1M_10R
  instances : data/cvrp_100.cvrpb
10000 instance(s) checked, 0 error(s)
  coverage : 1000000/1000000 customers served exactly once   OK
  capacity : 105660 route(s), fullest 50/50 (100.0 % of Q)   OK
  max relative gap reported / recomputed cost: 4.326e-16
  mean cost : 15.679491
  matches the run summary (15.67949)
  gap to HGS [21]   (15.563): +0.75 %
  gap to LKH-3 [20] (15.647): +0.21 %
  solver time : 312.735 s wall, 3716.558 s single-core, speedup 11.9x on 12 threads
```

Errors go to stderr, the summary to stdout, and the exit code is 1 if anything
failed — so `&&` chaining works in scripts.

It checks **feasibility and cost consistency, never quality**: a feasible but
terrible solution passes cleanly. The gaps to HGS/LKH-3 are informational. The
construction logic itself is cross-checked by [`tests/`](../tests/README.md).


---

## `analyze.py` — statistics and plots

```bash
python3 tools/analyze.py results/<run>                    # -> <run>/analysis.png
python3 tools/analyze.py results/<run> --no-plot          # statistics only
python3 tools/analyze.py results/<a> results/<b> ...      # -> comparison.png
```

**One run** prints a statistics block (cost distributions before/after, the
improvement distribution, routes, capacity utilisation, timings, the derived
schedule and operator budget) and writes a six-panel figure:

| panel | shows |
|---|---|
| Cost distribution | C&W vs annealed, with both means marked |
| Improvement per instance | where the gain actually lands, and its spread |
| Temperature schedule | **derived**, log-y so the geometric decay is a straight line |
| Operator draws | **derived** from the `--ops` weights |
| Routes per instance | the structural spread of the solutions |
| Improvement vs starting cost | does annealing rescue bad constructions? (it does not — r ≈ −0.07) |

**Several runs** switches to comparison mode: it detects the single `cw` option
that differs across them and plots mean cost against it, next to a quality/wall
time scatter. If more than one option differs it falls back to a bar chart by
run name.

### Derived, not measured

Two quantities are computed rather than observed, and are exact because the
solver is deterministic in both:

* **The temperature schedule.** `T(it) = T0 · α^it` with
  `α = (Tend/T0)^(1/(steps−1))` and `Tend = T0 · 10^−decades`. `cw` reports the
  mean calibrated T0, and the schedule does not depend on the search, so this is
  the temperature the solver actually used.
* **Operator draws.** The operator is picked by comparing one uniform 32-bit
  word against thresholds derived from the normalised `--ops` weights, so
  `draws_i = steps · w_i / Σw` to within sampling noise (over 10⁵ steps, well
  under 0.1 %).

### Not available from a plain `cw` run

The cost trajectory inside a run, the accepted moves per operator, and the
acceptance rate over time are all computed by the solver but never emitted:
`anneal()` keeps `cur` in a local, and all four operators increment one shared
counter `w->acc`.

These are supplied instead by **`./cw_trace`** (see
[`src/README.md`](../src/README.md)), a separate driver that `#include`s `cw.c`
to reuse its static functions without modifying it. Plot its output with:

```bash
make trace
./cw_trace --bundle data/cvrp_100.cvrpb --index 0 --sa-steps 200000 --every 20
python3 tools/analyze.py --trace results/trace_<timestamp>   # -> analysis.png there
```

Each trace lands in its own `results/trace_<timestamp>[_<name>]/`, the same
convention `run.py` uses, so nothing is written to the repository root.

`--trace` gives six panels a plain run cannot: the cost trajectory (current and
best-so-far), the measured temperature, the acceptance rate over time, accepted
moves per operator, the net cost contribution of each operator, and the route
count over time. Temperature deliberately occupies its own panel rather than
being overlaid on the cost — a second y-scale would make the two curves'
crossings and relative slopes meaningless.
