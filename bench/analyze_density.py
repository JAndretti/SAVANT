"""How much do concurrent runs on one node slow each other down? (roadmap B2)

    uv run python -m bench.analyze_density --csv results/jz/density.csv

`slurm/density_test.sbatch` runs the same (instance, seed) pairs at 40, 20 and 1
concurrent single-threaded runs per node, under a FIXED step budget. The work is
then identical whatever the load, so the ratio of run times measures contention
directly -- and that premise is checked here, not assumed: if the step counts
differ across densities the comparison is void and the script says so.

The reference is the unloaded run (one run alone on the node), which exists for
seed 1 only, so the pairing is per instance at seed 1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bench.common import ROOT


def load(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[d.tag.str.startswith("density_")].copy()
    if d.empty:
        raise SystemExit(f"no density_* rows in {path}")
    part = d.tag.str.split("_", expand=True)          # density_<set>_j<jobs>
    d["set"] = part[1]
    d["jobs"] = part[2].str.lstrip("j").astype(int)
    return d


def check_equal_work(d: pd.DataFrame) -> None:
    """The whole experiment rests on every density doing the same work."""
    if (d.feasible != 1).any():
        print(f"WARNING: {int((d.feasible != 1).sum())} infeasible runs")
    steps = d.groupby(["set", "instance", "seed"]).steps.nunique()
    bad = steps[steps > 1]
    if len(bad):
        raise SystemExit(f"step counts differ across densities for {len(bad)} "
                         f"(instance, seed) pairs, e.g.\n{bad.head()}\n"
                         "the run times are not comparable")
    print(f"work identical across densities: {d.steps.nunique()} distinct step count(s), "
          f"{len(d)} runs, {d.instance.nunique()} instances")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=ROOT / "results" / "jz" / "density.csv")
    a = ap.parse_args()
    d = load(a.csv)
    check_equal_work(d)

    t = d.pivot_table(index=["set", "instance", "seed"], columns="jobs", values="time_s")
    ref = t.columns.min()
    paired = t.dropna()
    print(f"\nruns per density: {dict(d.groupby('jobs').size())}; "
          f"{len(paired)} (instance, seed) pairs have all three")

    print("\n=== run time relative to one run alone on the node ===")
    rel = paired.div(paired[ref], axis=0)
    out = rel.groupby(level="set").agg(["mean", "max"])
    print(out.to_string(float_format=lambda v: f"{v:.3f}"))

    print("\n=== slowdown (%) at each density, mean over instances ===")
    for s, g in rel.groupby(level="set"):
        cells = ", ".join(f"{j} runs: {100 * (g[j].mean() - 1):+.1f}%" for j in g.columns)
        print(f"  {s:3s}  {cells}")

    print("\n=== absolute mean run time (s) ===")
    print(paired.groupby(level="set").mean().to_string(float_format=lambda v: f"{v:.1f}"))

    top = rel.columns.max()
    print(f"\nDecision rule (roadmap B2): keep {top} runs per node if the slowdown there is "
          f"small (< ~5%), otherwise regenerate the campaigns with --jobs 20.")
    for s, g in rel.groupby(level="set"):
        print(f"  {s:3s} at {top} runs: {100 * (g[top].mean() - 1):+.1f}% mean, "
              f"{100 * (g[top].max() - 1):+.1f}% worst instance")


if __name__ == "__main__":
    main()
