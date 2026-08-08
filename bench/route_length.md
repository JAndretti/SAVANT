# Why XL-n9571-k55 costs four times XL-n10001-k1570

## The observation

In the bench run (`--sa-steps N`, single-threaded), two XL instances of almost
the same size came out four times apart:

| instance | n | routes | mean route length | `--sa-steps` | time | ns per step |
|---|---:|---:|---:|---:|---:|---:|
| XL-n9571-k55 | 9,570 | 55 | **174.0** | 189,944,103 | **48.13 s** | 253.4 |
| XL-n10001-k1570 | 10,000 | 1,591 | **6.3** | 201,732,774 | **12.16 s** | 60.3 |

The larger instance is also the one that got *more* steps — 6 % more — and it
still finished four times sooner. So the difference is not in how much work was
asked for. It is entirely in the cost of one step: **4.2× per draw**, against
1.06× in the number of draws.

The hypothesis was route length: 55 routes over 9,570 customers is 174
customers per route, against 6.3 for the other, and `swap*` and the kick both
touch whole routes. That is right, and this is the measurement.

## The answer, in one line

**`swap*` costs 3.13 ns per unit of mean route length, and every other operator
is flat.** On XL-n9571-k55 that single operator is about three quarters of the
per-step cost; on XL-n10001-k1570 it is about one third.

It is not worth reweighting, though: §7 measures the swap\* weight against
solution quality at equal wall time and finds no difference, in either regime.
The operator costs 3–5× more where routes are long and is worth 3–5× more
there. Nothing in the shipped defaults needs to change.

## 1. It is route length, not size — 100 instances

Per-instance nanoseconds per annealing step across the whole XL set:

```
corr(n,            ns/step) = +0.168
corr(route length, ns/step) = +0.949

ns/step = 48.7 + 1.131 · L      R² = 0.901      (L = n / routes)
ns/step = 68.3 + 3.20e-3 · n    R² = 0.028      <- size alone explains nothing
```

Route length spans 2.9 to 185.2 on XL and explains 90 % of the variance in the
cost of a step. Size explains 3 %.

This is the experiment the generated family cannot run. `--random`'s capacity
ladder tops out at 200 with demands on [1, 10], so mean route length is pinned
near 40 for every *n* above 1,000 — which is exactly why `timing/report.md`'s
`b(n) = 26.5 · n^0.13` looked like a gentle function of size. It was never a
function of size. XL varies route length by 64× at nearly constant *n* and the
association falls out immediately.

## 2. Which operator — each one run alone

Four XL instances within 5 % of each other in *n*, route length spanning 28×.
Each operator run **alone** for 2,000,000 draws, single-threaded, so the number
in each cell is that operator's own cost per draw:

| configuration | n9571 | n9363 | n9160 | n10001 |
|---|---:|---:|---:|---:|
| *mean route length* | *170.9* | *44.8* | *24.0* | *6.1* |
| relocate | 48.1 | 48.2 | 43.6 | 40.5 |
| swap | 48.7 | 53.7 | 47.7 | 53.7 |
| 2-opt | 70.6 | 63.3 | 69.9 | 64.2 |
| or-opt | 49.1 | 50.5 | 45.4 | 41.1 |
| **swap\*** | **569.0** | **421.4** | **118.2** | **102.6** |
| opening | 27.4 | 27.3 | 26.6 | 29.2 |
| kick, every step | 909.3 | 934.9 | 885.0 | 681.1 |
| default mix, no kick | 191.1 | 155.2 | 86.2 | 74.4 |
| default mix | 187.0 | 164.9 | 93.9 | 86.0 |

ns per draw. Everything is flat across a 28× change in route length except
`swap*`, which moves 5.5×.

**The kick is not the problem**, despite being the most expensive operator per
firing. It fires every 100 steps, so it contributes ~9 ns per step — which is
what the last two rows differ by. Its cost is also nearly flat in route length
(slope 0.78 ns/L, R² = 0.25): it is dear because it removes up to 10 customers
and reinserts each against O(K+L) candidates, not because routes are long.

**2-opt is not the problem either**, which is worth saying because its segment
reversal *is* O(L). The reversal only happens on acceptance, and at a 3–5 %
acceptance rate that averages out to nothing: 64–71 ns at every route length.

## 3. The slope, on 16 instances

Four instances can attribute but cannot fit: they differ in geometry and in *n*
as well as in route length. So the two ends of the ranking were re-run alone
across 16 instances spread over the whole range:

| operator | m | intercept | ns per unit of L | R² |
|---|---:|---:|---:|---:|
| **swap\*** | 16 | 51.1 | **3.130** | **0.949** |
| relocate | 16 | 25.6 | 0.198 | 0.431 |

## 4. Why 3.13, from the code

`mv_swapstar` (`src/cw.c:1527`) walks whole routes **three times per draw**:

1. it flattens the target route into a buffer — O(L₂) — because the operator
   picks a *route* and then a customer inside it, not a vertex pair;
2. it scans the source route for the best insertion of *v* — O(L₁);
3. it scans the flattened target route for the best insertion of *u* — O(L₂).

On acceptance it walks both once more to recompute the loads as a fresh sum
(deliberately: the incremental update lets ULPs drift into a wrong capacity
test, and the operator is already O(L₁+L₂), so the recomputation is free).

With L₁ ≈ L₂ ≈ L that is **3L pointer-chases per rejected draw** and 5L on
acceptance, so about 3.1L at a 3–5 % acceptance rate. At roughly one nanosecond
per random access into an L2-resident array, the prediction is ≈ 3.1 ns per
unit of L. The measurement is **3.130**. The other three operators do O(1) work
per draw and come out flat, as they should.

## 5. Does the mix add up?

`swap*` takes one draw in 4.05 under the default weights (1, 1, 1, 0, 1, 0.05),
i.e. 24.7 %:

| | XL-n9571-k55 (L = 171) | XL-n10001-k1570 (L = 6.1) |
|---|---:|---:|
| swap\* alone | 569.0 ns | 102.6 ns |
| × its 24.7 % share | 140.5 ns | 25.3 ns |
| measured mix, no kick | 191.1 ns | 74.4 ns |
| **swap\* as a share of the step** | **74 %** | **34 %** |

Predicting the whole mix from the parts lands within 5–7 % on the two
long-route instances and 15–24 % low on the two short-route ones. The
under-prediction is expected and is not a discrepancy to explain away: an
operator run alone drives the solution somewhere a mixed run never goes, so the
route structure it measures its own cost on is not the structure it would see
in the mix. The attribution survives that; the exact arithmetic does not.

Reading the same thing off the fitted slopes: `swap*` contributes
3.130 × 0.247 = 0.773 of the 1.131 ns/L measured across all of XL, i.e. **68 %
of the slope**, with everything else together accounting for another 6 %.

## 6. What this means in practice

**`--sa-steps N` buys a predictable number of steps, not a predictable time.**
That is the trade it was designed to make, but on XL the two come apart by a
factor of seven: the same rule gives every instance the budget its size asks
for, and then the wall clock varies 7× because a step is not a step. On the
generated family the two are nearly the same thing, which is why this did not
show up in `timing/`. If you need predictable *time*, `--sa-time` is the flag;
if you need predictable *work*, `--sa-steps N` is.

**The XL set's total cost is concentrated.** The ten dearest instances are 38 %
of the set's 577 seconds, and they are the long-route ones. Route length on XL
is heavily skewed — the quartiles are 2.9 / 6.4 / 13.9 / 28.2 / 185.2 — so a
handful of instances dominate any XL timing figure. A mean over the set says
very little.

**`timing/report.md`'s `b(n)` does not transfer here**, and now the reason is
concrete rather than a caveat. Its `n^0.13` is a fit to a family whose route
length happens to grow with *n* up to 1,000 and then stops. The underlying
variable is L, and on a family where L is free to vary, the exponent in *n* is
meaningless. The report already says the constants are family-specific; this is
what that costs.

## 7. Is `swap*` worth it where it is dearest? No difference either way

Costing 74 % of every step is not by itself an argument against an operator ---
`sweep/report.pdf` finds that dropping `swap*` is the single most damaging
removal of the six. But that was measured at n = 100, where route length is
about 10 and `swap*` costs what everything else costs. At L = 174 it is buying
its improvement at five times the price, so the question is whether the time
would be better spent on more cheap draws.

That is only answerable at **equal wall time**, where a lighter mix is allowed
to convert its saving into more draws. `--sa-time 10`, twelve XL instances at
the two extremes of route length, three weights, runs adjacent in time so a
busy moment moves all three together:

| instance | L | `w=1` | `w=0.25` | | `w=0` | |
|---|---:|---:|---:|---:|---:|---:|
| XL-n1701-k562 | 2.9 | 523,761 | 524,495 | +0.14 | 524,504 | +0.14 |
| XL-n6168-k1922 | 3.1 | 1,547,191 | 1,546,674 | −0.03 | 1,546,299 | −0.06 |
| XL-n2028-k617 | 3.2 | 548,584 | 548,184 | −0.07 | 548,563 | −0.00 |
| XL-n9784-k2774 | 3.5 | 4,108,222 | 4,109,629 | +0.03 | 4,110,953 | +0.07 |
| XL-n3334-k934 | 3.5 | 1,458,355 | 1,459,417 | +0.07 | 1,457,988 | −0.03 |
| XL-n3888-k1010 | 3.7 | 1,953,040 | 1,958,667 | +0.29 | 1,953,825 | +0.04 |
| **mean, short routes** | | | | **+0.07** ± 0.14 | | **+0.03** ± 0.08 |
| XL-n3804-k29 | 131.1 | 53,293 | 53,175 | −0.22 | 53,185 | −0.20 |
| XL-n1981-k13 | 141.4 | 33,277 | 33,223 | −0.16 | 33,148 | −0.39 |
| XL-n1654-k11 | 150.3 | 36,642 | 36,655 | +0.04 | 36,626 | −0.04 |
| XL-n2634-k17 | 154.9 | 32,010 | 32,170 | +0.50 | 32,304 | +0.92 |
| XL-n9571-k55 | 174.0 | 109,036 | 108,334 | −0.64 | 109,080 | +0.04 |
| XL-n7037-k38 | 185.2 | 71,827 | 72,045 | +0.30 | 71,720 | −0.15 |
| **mean, long routes** | | | | **−0.03** ± 0.43 | | **+0.03** ± 0.48 |

Negative means the lighter mix wins. Intervals are 95 % on the mean of six
paired differences.

**Nothing. In either regime.** Every interval contains zero comfortably, and on
the long-route instances the per-instance differences scatter from −0.64 % to
+0.50 % with no pattern --- that spread is what one seed of a 10-second run
looks like, not a signal.

This corrects a conclusion I drew from a single instance before running the
rest. XL-n9571-k55 alone showed `w=0.25` ahead by 0.45 %, and it is in the
table above at −0.64 %; five more instances put the mean at −0.03 %. One
instance was noise.

The real finding is the null one, and it is worth more than the alternative
would have been: **`swap*`'s value scales with its cost.** It costs 3–5× more
per draw where routes are long, and it is worth 3–5× more there too, closely
enough that reweighting it buys nothing measurable. There is no cheap win here,
and the answer to "should the default weight depend on route length?" is no ---
which is the more useful thing to know, since it means the shipped `--ops` does
not need a special case for large instances.

What would actually help is making the operator cheaper rather than rarer:
sampling a bounded number of insertion positions instead of scanning the whole
route would turn its three O(L) passes into O(1) at some loss of move quality.
That is a change to `mv_swapstar`, not to a weight, and it is not tested here.

## 8. Reproducing this

```bash
uv run bench/probe_route_length.py              # everything above, ~15 min
uv run bench/probe_route_length.py --wide 24    # more support for the slope
uv run bench/probe_route_length.py --report-only
uv run bench/probe_route_length.py --weights 0  # skip the equal-time part
```

Cells are cached under `results/bench/probe/`, so a re-run only fills what is
missing. Every run is `--threads 1` and one at a time: `results.csv`'s
`time_ms` is measured inside cw's parallel region and would otherwise absorb
the contention between threads.
