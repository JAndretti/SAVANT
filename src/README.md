# `src/` — the solver

| file | lines | role |
|---|---:|---|
| `cw.c` | 2266 | the entire solver, one translation unit |
| `trace.c` | ~340 | instrumented single-instance driver; **does not modify `cw.c`** |

Built from the repository root:

```bash
make            # gcc + OpenMP        -> ./cw
make macos      # Apple clang + libomp -> ./cw
make serial     # no OpenMP
make repro      # -ffp-contract=off, bit-reproducible
make debug      # -O0 -g -fsanitize=address,undefined -> ./cw_dbg
```

There are no headers and no other source files. Everything is `static` apart
from `main`, which lets the compiler inline across the whole program — the hot
loop depends on it.

## Map of `cw.c`

| lines | section | notes |
|---:|---|---|
| 1–43 | header comment, includes | OpenMP shim so the file builds without `-fopenmp` |
| 44–73 | utils | `die`, `xmalloc`, `xrealloc`, `now_sec` |
| 74–147 | RNG | splitmix64 seeding, xoshiro256\*\*, plus `rng32` / `rng_idx` / `rng_bit`, the reduced-randomness path |
| 148–165 | `Opts` | every command-line option in one struct |
| 166–220 | `Inst`, random generation | index 0 = depot; Kool/NeuOpt distribution |
| 221–389 | readers | `.cvrpb` bundle, TSPLIB `.vrp`, directory scan |
| 390–521 | `WS` workspace | per-thread buffers, allocated once and reused |
| 522–560 | union-find, LSD radix sort | stable sort on the float32 saving key |
| 561–706 | k nearest neighbours | uniform grid scanned in rings, brute force below n = 512 |
| 707–732 | intra-route 2-opt | the `--2opt` construction post-process |
| 733–977 | annealing scaffolding | `Sol`, virtual depots, `sa_accept`, `lb_build`, Fenwick tree, `inc[]`, `pick_u` |
| 978–1210 | the four operators | `mv_relocate`, `mv_swap`, `mv_oropt`, `mv_2opt` |
| 1211–1356 | optimal Split | Prins recurrence with Vidal's O(n) monotone deque |
| 1357–1460 | annealing driver | `sa_draw`, `calibrate_T0`, `anneal` |
| 1461–1716 | `solve_cw` | the per-instance pipeline; the construction branches on `--init` at line 1526 |
| 1717–1822 | output and validation | `.cvrpb` export, built-in `--validate` |
| 1823–2266 | `usage()` and `main` | CLI parsing, OpenMP loop, statistics, CSV / solution writing |

## Things worth knowing before editing

**Only `main` is non-static.** Adding a second `.c` file would require exposing
symbols and would cost the cross-function inlining the hot loop relies on.

**Results depend on the compilation flags.** `-ffp-contract=fast` (GCC's default
in C) fuses `a*b+c` into an FMA, which changes rounding and can flip a borderline
comparison in the acceptance test, diverging the whole trajectory. Use
`make repro` when comparing algorithm variants across machines. This is also why
`tools/run.py` records an md5 of the binary in every run's `config.json`.

**`MAXN` is 65535** because savings pack `i` and `j` into 16 bits each inside an
8-byte record (`Sav`). Raising the limit means widening that record.

**`--init random` skips the savings path entirely** (src/cw.c:1855): no savings
list is built, no sort, no merge. `--knn`, `--exact`, `--lambda`, `--mu`,
`--cw-rand` and `--cw-alpha` therefore have no effect on that path, and the
savings buffer is not even allocated. Each restart draws its own permutation, so
`--restarts` still gives a genuine multi-start.

**The radix sort is only correct for positive keys.** Savings are filtered to
`> 0` before insertion, which is what makes the IEEE-754 bit patterns of the
`float` keys monotone in the value.

A full walkthrough of the execution order, every option, and the known caveats is
in the root [`README.md`](../README.md).

---

## `trace.c` — instrumentation without touching the solver

```bash
make trace                                    # -> ./cw_trace
./cw_trace --bundle data/cvrp_100.cvrpb --index 0 --sa-steps 200000 --every 20
python3 tools/analyze.py --trace results/trace_<timestamp>
```

Records the two things `cw` computes but never reports: the **cost trajectory
inside one run**, and **per-operator draw / acceptance counts**. Like `run.py`,
each trace gets its own directory — `results/trace_<timestamp>[_<name>]/`,
overridable with `--out DIR` — holding

```
trace.csv      one row per sampled step: step, T, cur, best, op, accepted, delta, routes
trace.json     configuration + the per-operator summary
analysis.png   added by tools/analyze.py --trace
```

Scope by construction: one instance, one run, no multi-restart.

**`--init cw|random`** selects the starting solution. `random` shuffles the
customers and cuts the sequence whenever the next one would overflow the
vehicle — a *feasible* random start, not a uniformly random one, so the
comparison is about solution quality rather than about repairing infeasibility
(which the annealer cannot do: every operator rejects a capacity violation).
First-fit on a random permutation also lands near C&W's route count, so the two
starts are structurally comparable. This is the ablation that isolates what the
construction actually buys.

**How it avoids duplicating the solver.** `trace.c` does
`#define main cw_main_unused` and then `#include "cw.c"`. Because every function
in `cw.c` is `static`, including the translation unit makes all of them visible;
the `#define` moves `cw.c`'s own `main` out of the way so this file can supply
its own. The savings construction, kNN, workspace, RNG, operators and Split are
therefore *the solver's*, not copies that could drift out of sync. Only the
annealing loop is re-written, to log.

**Why the trace is trustworthy.** The traced loop draws the operator with the
same threshold comparison `sa_draw` makes, and runs `calibrate_T0` first, so it
consumes the RNG stream in exactly the order `anneal` does. With the same seed
and options the final cost matches a plain `cw` run **exactly** — verified at
15.836157 → 15.511045 on instance 0 of CVRP-100 with seed 42, identical in both.
That equality is the check that the trace describes the real search rather than
a lookalike; if a future edit to `cw.c` breaks it, the costs stop matching.

Acceptance is read from the solver's own counter (`w->acc` increments only on
acceptance) rather than inferred from `delta == 0`, which would misclassify the
rare zero-cost accepted move.

**Cost.** Writing every step is slow — 200 000 steps took 25 ms traced against
1 ms untraced, dominated by `fprintf`. Use `--every N` (the trajectory is
smooth; every 20th step loses nothing visible) or `--no-csv` for the operator
summary alone.
