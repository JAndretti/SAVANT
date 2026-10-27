"""Analyse the P3 campaign on starts and schedule width (results/p3_starts/screen_B*.csv).

Runs come from `bench.sweep --base final --configs p3` (final configuration,
cw defaults R = 4, D = 1). Quality of a run = gap (%) to the best cost known for
its instance: the best feasible cost over all P3 runs at every budget, or the
BKS when better. Each variant is compared to `final` on the same (instance, seed)
pairs (delta < 0 = better; p_holm = Wilcoxon on the pairs,
Holm-corrected within a budget). The by-size table gives raw p-values: it is
there to suggest a rule R(n, budget), not to prove one.

    uv run python -m bench.analyze_starts
writes results/p3_starts/starts_summary.csv and starts_by_size.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from bench.common import ROOT, holm, load_bks

DIR = ROOT / "results" / "p3_starts"


def size_group(n: int) -> str:
    if n <= 100:
        return "n<=100"
    if n <= 500:
        return "n=200-500"
    if n <= 1000:
        return "n=600-1000"
    return "n=2000"


def paired(budget: str, tag: str, x: pd.DataFrame, base: pd.DataFrame) -> dict:
    """x against base on the same (instance, seed) pairs; both indexed by them."""
    j = base.index.intersection(x.index)
    diff = x.gap[j] - base.gap[j]
    p = wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
    return {"budget": budget, "tag": tag, "pairs": len(j), "gap": x.gap[j].mean(), "delta": diff.mean(),
            "wins": int((diff < 0).sum()), "losses": int((diff > 0).sum()), "p": p,
            "time_x": x.time_s[j].mean() / base.time_s[j].mean()}


def main() -> None:
    files = sorted(DIR.glob("screen_B*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    bad = df[~df.feasible.astype(bool)]
    print(f"{len(df)} runs, {len(bad)} infeasible")
    df = df[df.feasible.astype(bool)].copy()
    bks = load_bks()
    ref = df.groupby("instance").cost.min()
    ref = pd.Series({i: min(c, bks.get(i, np.inf)) for i, c in ref.items()})
    df["gap"] = 100 * (df.cost - df.instance.map(ref)) / df.instance.map(ref)
    df["size"] = df.n.map(size_group)

    rows, by_size = [], []
    for budget, part in df.groupby("budget", sort=False):
        idx = {t: g.set_index(["instance", "seed"]) for t, g in part.groupby("tag")}
        base = idx["final"]
        res = [paired(budget, t, x, base) for t, x in idx.items() if t != "final"]
        res = pd.DataFrame(res)
        res["p_holm"] = holm(res.p.to_numpy())
        res.insert(3, "final_gap", base.gap.mean())
        rows.append(res)
        for t, x in idx.items():
            if t == "final":
                continue
            j = base.index.intersection(x.index)
            d = pd.DataFrame({"diff": x.gap[j] - base.gap[j], "size": base["size"][j]})
            for s, g in d.groupby("size"):
                p = wilcoxon(g["diff"]).pvalue if (g["diff"] != 0).any() else 1.0
                by_size.append({"budget": budget, "tag": t, "size": s, "pairs": len(g),
                                "delta": g["diff"].mean(), "wins": int((g["diff"] < 0).sum()),
                                "losses": int((g["diff"] > 0).sum()), "p_raw": p})
    summary = pd.concat(rows, ignore_index=True)
    sizes = pd.DataFrame(by_size)
    summary.to_csv(DIR / "starts_summary.csv", index=False)
    sizes.to_csv(DIR / "starts_by_size.csv", index=False)
    pd.set_option("display.width", 160)
    for budget, s in summary.groupby("budget", sort=False):
        print(f"\n=== {budget} ===  final gap {s.final_gap.iloc[0]:.4f} %, {s.pairs.max()} pairs")
        print(s[["tag", "pairs", "gap", "delta", "wins", "losses", "p_holm", "time_x"]]
              .sort_values("delta").to_string(index=False, float_format="%.4f"))
    print("\n=== delta by size (raw p) ===")
    piv = sizes.pivot_table(index=["budget", "tag"], columns="size", values="delta", sort=False)
    pv = sizes.pivot_table(index=["budget", "tag"], columns="size", values="p_raw", sort=False)
    print((piv.round(3).astype(str) + " (" + pv.round(2).astype(str) + ")").to_string())


if __name__ == "__main__":
    main()
