# `tests/` — verification suite

Independent re-implementations that cross-check the solver, plus the fuzzer.
Nothing here is a unit test in the usual sense: each script recomputes something
the C is supposed to produce, by a different method, and compares.

| file | verifies | against |
|---|---|---|
| `check.py` | Clarke & Wright construction | a naive Python re-implementation |
| `checksplit.py` | the O(n) Split | a naive O(n²) DP on the same giant tour |
| `localopt.py` | the pure descent | an exhaustive local search |
| `fuzz.py` | everything, under random option combinations | `tools/validate.py` |
| `mkedge.py` | — | generates 15 edge-case instances |
| `_paths.py` | — | shared path resolution (not a test) |

All of them need the solver built (`make` from the root). They run from any
working directory.

---

## `_paths.py`

Not a test. The scripts live in `tests/` but drive `./cw` at the repository root
and import `validate.py` from `tools/`. Importing `_paths` resolves `ROOT`,
`BIN`, `DATA`, `EDGE` and `VALIDATE` from `__file__`, and puts `tools/` on
`sys.path` as a side effect — which is what makes `from validate import
read_bundle` work from here.

Without it these scripts only worked when you were standing in the repository
root, because they hardcoded `"./cw"` and `"edge/*.cvrpb"`.

---

## `mkedge.py` — adversarial instances

```bash
python3 tests/mkedge.py            # writes <root>/edge/*.cvrpb
python3 tests/mkedge.py somewhere  # or a directory of your choice
```

Fifteen instances the random generator statistically never produces. Seeded
(`random.seed(1234)`), so re-running reproduces them byte-for-byte.

| instance | what is degenerate |
|---|---|
| `tiny1/2/3/5` | n = 1, 2, 3, 5 — boundary cases for loops assuming ≥ 2 customers |
| `same10/800` | every point coincident → all distances 0, every saving filtered out |
| `line10/800` | collinear points → degenerate grid, many exactly-tied savings |
| `clusters` | two tight clusters → whole rows of empty grid cells for the ring scan |
| `alone` | Q = 9 with all demands 9 → every customer alone on its route |
| `onebig` | Q = 10⁶ → everything in one route |
| `frac` | fractional demands against Q = 7.5 — exercises the `EPS` tolerances |
| `huge` | coordinates scaled by 10⁶ |
| `depotdup` | a customer placed exactly on the depot → zero-length edge |
| `infeasible` | one demand of 99 against Q = 30 → **no valid solution exists** |

`infeasible` is the important one: it is what surfaced the out-of-bounds read in
Split, where the monotone deque empties because a single customer does not fit in
a vehicle.

---

## `check.py` — construction cross-check

```bash
python3 tests/check.py data/cvrp_20.cvrpb 100
```

Re-implements Clarke & Wright naively — explicit route lists, a full sort of the
savings, route reversal on merge — and compares the cost against the C binary run
with `--exact`. Also revalidates the solutions written by `--sol`.

Expected: `OK` on every instance, max relative gap on the order of 1e-12, which
is float addition order alone.

> **Caveat.** Agreement is not guaranteed under *ties*. The C truncates each
> saving to float32 before sorting and iterates the stable radix sort backwards;
> `check.py` sorts float64 values forwards. On random instances ties are
> effectively measure-zero, but on `edge/line800.cvrpb` — 800 collinear points,
> identical demands — many savings are genuinely equal and the two may
> legitimately diverge. Nothing in the suite currently distinguishes that from a
> bug; see the "Known caveats" section of the root README.

---

## `checksplit.py` — Split cross-check

```bash
python3 tests/checksplit.py data/cvrp_50.cvrpb 20
python3 tests/checksplit.py data/cvrp_100.cvrpb 10 --sa-steps 20000
```

Runs the solver twice — once with `--split off` to capture the routes, once with
`--split end --split-tour routes` — then rebuilds the same giant tour in Python
and repartitions it with the naive O(n²) Prins DP. Any extra arguments are
forwarded to `./cw`, so the Split can be checked on annealed solutions and not
just raw C&W ones.

Expected: exact agreement, max relative gap ~1e-12.

---

## `localopt.py` — descent cross-check

```bash
python3 tests/localopt.py data/cvrp_20.cvrpb 5
```

Computes a true local optimum in Python by exhaustive local search (relocate +
swap + intra 2-opt + 2-opt\*, applied to a fixpoint) and compares it with the C
annealer driven as a pure descent (`--t0 1e-9 --tend 1e-9`).

Expected: the two agree exactly. This is the check behind the claim that pure
descent reaches the local optimum of these neighbourhoods rather than merely
approaching it.

**Slow by construction** — it recomputes the full solution cost inside the move
loops, so it is O(n⁴)-ish. Use a handful of small instances.

---

## `fuzz.py` — the option-space fuzzer

```bash
python3 tests/fuzz.py 200          # 200 trials with ./cw
python3 tests/fuzz.py 60 --asan    # against ./cw_dbg (make debug), much slower
```

Draws random combinations of ~25 options — every enumeration, extreme values of
each numeric parameter — and runs them over the edge instances, the NeuOpt sets
and freshly generated random instances. For each trial it checks:

* the exit code (0, or 2 when the instance really is infeasible)
* nothing written to stderr
* the incremental-cost drift stays below 1e-6 (`--check`)
* **every solution written is revalidated with `tools/validate.py`**

On a genuinely infeasible instance only capacity overloads are tolerated; that
filter matches on the word `overloaded`, so it is coupled to `validate.py`'s
wording — change one and you must change the other.

Requires `edge/` to exist (`python3 tests/mkedge.py`) and references
`data/cvrp_200*`, so fetch size 200 or expect those trials to fail with a clear
"No such file or directory".

---

## Running the lot

```bash
make                                            # or: make macos
python3 tests/mkedge.py
python3 tests/check.py       data/cvrp_20.cvrpb  20
python3 tests/checksplit.py  data/cvrp_50.cvrpb  10
python3 tests/localopt.py    data/cvrp_20.cvrpb   3
python3 tests/fuzz.py 100
```

There is no runner script and no test framework: each is a standalone program
that prints its own verdict and exits non-zero on failure.
