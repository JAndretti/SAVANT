"""Analyse a comparison campaign: solution quality per solver, paired tests and
convergence at fractions of T_max.

    uv run python -m bench.analyze_compare --set results/jz/X
    uv run python -m bench.analyze_compare --set results/jz/n2000 --ref-from-runs

Reads every CSV of the directory (shards included: rows are de-duplicated on
solver/tag/instance/seed exactly as bench.merge_csv does) and the convergence
traces next to them, and writes three files in the same directory:

  reference.csv     the cost each gap is measured against, one row per instance;
  per_instance.csv  one row per (solver, instance): gaps over the seeds, size, time;
  summary.csv       one row per solver: gaps, failures, time, test against SAVANT;
  checkpoints.csv   mean gap of the incumbent at each fraction of T_max.

Conventions, all of which the paper must state:

  * The reference is the best known solution for X and XL, and the best feasible
    cost found in the campaign for the uniform sets, which have no BKS. That
    second reference moves when runs are added (flaw #10), so it is written to
    reference.csv on the first run and reused afterwards unless --refresh-ref is
    given: every table of the paper then quotes the same denominator.
  * Runs are paired by instance, never by seed. Seeds are not comparable across
    solvers (AILS-II cannot be seeded at all, flaw #26), so each solver is first
    reduced to one mean gap per instance and the Wilcoxon signed-rank test runs
    on those pairs.
  * Infeasible runs are excluded from the gaps and counted separately; an
    instance on which a solver never returned a feasible solution is dropped
    from its pairs, which flatters that solver and must be read with `fails`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from bench import trace as tr
from bench.common import ROOT, holm, load_bks

# display order; the first one is the solver everything else is tested against
SOLVERS = ["savant", "hgs", "filo", "filo2", "ails", "lkh", "ortools"]
GENERATED = {"reference.csv", "summary.csv", "checkpoints.csv", "per_instance.csv"}


def load_runs(d: Path, tag: str) -> pd.DataFrame:
    files = [f for f in sorted(d.glob("*.csv")) if f.name not in GENERATED]
    if not files:
        raise SystemExit(f"no result CSV in {d}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df.tag == tag]
    if df.empty:
        raise SystemExit(f"no rows with tag={tag!r} in {d}")
    n = len(df)
    df = df.drop_duplicates(subset=["solver", "tag", "instance", "seed"], keep="last")
    print(f"{len(files)} files, {n} rows -> {len(df)} unique runs, "
          f"{df.instance.nunique()} instances, solvers: {', '.join(sorted(df.solver.unique()))}")
    return df


def load_traces(d: Path, tag: str) -> dict:
    recs: dict = {}
    for f in sorted(d.glob("*.trace.jsonl")):
        recs.update(tr.read_sidecar(f))
    return {k: r for k, r in recs.items() if k[1] == tag}


def reference(df: pd.DataFrame, path: Path, from_runs: bool, refresh: bool) -> pd.Series:
    """Cost the gaps are measured against, frozen in reference.csv once computed."""
    if path.exists() and not refresh:
        ref = pd.read_csv(path).set_index("instance").ref_cost
        missing = set(df.instance) - set(ref.index)
        if missing:
            raise SystemExit(f"{path} predates {len(missing)} instances "
                             f"(e.g. {sorted(missing)[:3]}); rerun with --refresh-ref")
        print(f"reference: {path.name} ({len(ref)} instances, unchanged)")
        return ref

    ok = df[df.feasible == 1]
    best = ok.groupby("instance").cost.min()
    if from_runs:
        ref, src = best, "best feasible cost found in this campaign"
    else:
        bks = load_bks()
        missing = [i for i in best.index if i not in bks]
        if missing:
            raise SystemExit(f"no BKS for {len(missing)} instances (e.g. {missing[:3]}); "
                             "use --ref-from-runs for sets without a BKS")
        ref = pd.Series({i: bks[i] for i in best.index}, name="ref_cost")
        src = "best known solution (CVRPLIB)"
        beat = best[best < ref.loc[best.index] - 1e-6]
        if len(beat):
            print(f"  NOTE: the campaign improves on the BKS of {len(beat)} instances "
                  f"({', '.join(beat.index[:5])}{'...' if len(beat) > 5 else ''}); "
                  "gaps for those are negative and the BKS is kept as the reference")
    ref = ref.rename("ref_cost").rename_axis("instance")
    ref.to_frame().to_csv(path)
    print(f"reference: {src}, {len(ref)} instances -> {path.name}")
    return ref


def per_instance(df: pd.DataFrame, ref: pd.Series) -> pd.DataFrame:
    """One row per (solver, instance): mean and best gap over the seeds."""
    d = df.copy()
    d["ref_cost"] = d.instance.map(ref)
    d["gap"] = 100 * (d.cost - d.ref_cost) / d.ref_cost
    ok = d[d.feasible == 1]
    g = ok.groupby(["solver", "instance"]).agg(
        gap_mean=("gap", "mean"), gap_best=("gap", "min"), gap_worst=("gap", "max"),
        n_ok=("gap", "size"), time_s=("time_s", "mean"), n=("n", "first"))
    fails = d[d.feasible != 1].groupby(["solver", "instance"]).size().rename("n_fail")
    return g.join(fails).fillna({"n_fail": 0}).reset_index()


def summary(pi: pd.DataFrame, runs: pd.DataFrame, base: str) -> pd.DataFrame:
    """Per-solver aggregate, plus a Wilcoxon test against `base` on instance pairs."""
    if base not in set(pi.solver):
        raise SystemExit(f"{base} is not in this campaign; --base names the reference solver")
    b = pi[pi.solver == base].set_index("instance")
    rows = []
    for s in [x for x in SOLVERS if x in set(pi.solver)] + \
             sorted(set(pi.solver) - set(SOLVERS)):
        x = pi[pi.solver == s].set_index("instance")
        allruns = runs[runs.solver == s]
        j = b.index.intersection(x.index)
        diff = x.gap_mean[j] - b.gap_mean[j]          # > 0: worse than the base
        row = {"solver": s, "instances": len(x), "runs": len(allruns),
               "fails": int((allruns.feasible != 1).sum()),
               "no_feasible": len(b.index.difference(x.index)),
               "gap_mean": x.gap_mean.mean(), "gap_best": x.gap_best.mean(),
               "gap_worst": x.gap_worst.mean(),
               "at_ref": int((x.gap_best <= 1e-9).sum()), "time_s": allruns.time_s.mean(),
               "pairs": len(j), "delta": diff.mean(),
               "wins": int((diff < 0).sum()), "losses": int((diff > 0).sum())}
        row["p"] = (wilcoxon(diff).pvalue
                    if s != base and len(diff) and (diff != 0).any() else np.nan)
        rows.append(row)
    out = pd.DataFrame(rows)
    m = out.p.notna()
    out.loc[m, "p_holm"] = holm(out.p[m].to_numpy())     # one family per campaign
    return out


def checkpoints(recs: dict, ref: pd.Series, base: str) -> pd.DataFrame:
    """Mean gap of the incumbent at each fraction of T_max, per solver.

    Averaged over instances (the seeds of an instance are averaged first) so
    that a checkpoint column is comparable with the final-gap table. `no_sol`
    counts the runs with no solution yet at that point: they are left out of
    the mean, which flatters a slow starter."""
    rows = []
    for (solver, _tag, inst, seed), r in recs.items():
        if not r["trace_ok"] or inst not in ref.index or not r.get("budget_s"):
            continue
        c = ref[inst]
        for f in tr.CHECKPOINTS:
            v = tr.incumbent_at(r["trace"], f * r["budget_s"])
            rows.append({"solver": solver, "instance": inst, "seed": seed, "frac": f,
                         "gap": np.nan if v is None else 100 * (v - c) / c})
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    per_inst = d.groupby(["solver", "frac", "instance"]).gap.mean()      # NaN seeds skipped
    gap = per_inst.groupby(["solver", "frac"]).mean().unstack("frac")
    none = d.gap.isna().groupby([d.solver, d.frac]).sum().unstack("frac")
    out = pd.concat({"gap": gap, "no_sol": none.reindex(gap.index).fillna(0).astype(int)},
                    names=["stat"]).swaplevel().sort_index()
    order = [s for s in SOLVERS if s in gap.index] + sorted(set(gap.index) - set(SOLVERS))
    skipped = sum(1 for r in recs.values() if not r["trace_ok"])
    if skipped:
        print(f"  {skipped} unusable traces skipped")
    return out.loc[order].reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, type=Path, help="directory of campaign CSVs")
    ap.add_argument("--tag", default="tmax")
    ap.add_argument("--base", default="savant", help="solver the others are tested against")
    ap.add_argument("--ref-from-runs", action="store_true",
                    help="no BKS (uniform sets): measure gaps against the best cost found")
    ap.add_argument("--refresh-ref", action="store_true", help="recompute a frozen reference.csv")
    a = ap.parse_args()

    runs = load_runs(a.set, a.tag)
    ref = reference(runs, a.set / "reference.csv", a.ref_from_runs, a.refresh_ref)
    pi = per_instance(runs, ref)
    pi.to_csv(a.set / "per_instance.csv", index=False)
    s = summary(pi, runs, a.base)
    s.to_csv(a.set / "summary.csv", index=False)
    cols = ["solver", "instances", "runs", "fails", "no_feasible", "gap_mean", "gap_best",
            "at_ref", "time_s", "pairs", "delta", "wins", "losses", "p_holm"]
    print("\n=== quality ===")
    print(s[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}", na_rep="--"))

    recs = load_traces(a.set, a.tag)
    print(f"\n=== convergence ({len(recs)} traces) ===")
    cp = checkpoints(recs, ref, a.base)
    if cp.empty:
        print("no usable trace")
    else:
        cp.to_csv(a.set / "checkpoints.csv", index=False)
        print(cp.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    by_size = pi.assign(bucket=pd.cut(pi.n, [0, 200, 400, 600, 1000, 2000, 5000, 10 ** 9]))
    wide = by_size.groupby(["solver", "bucket"], observed=True).gap_mean.mean().unstack("bucket")
    if wide.shape[1] > 1:
        print("\n=== mean gap by size ===")
        print(wide.loc[[s for s in SOLVERS if s in wide.index]]
              .to_string(float_format=lambda v: f"{v:.3f}", na_rep="--"))
    print(f"\nwrote {a.set}/per_instance.csv, {a.set}/summary.csv"
          + (f", {a.set}/checkpoints.csv" if not cp.empty else ""))


if __name__ == "__main__":
    main()
