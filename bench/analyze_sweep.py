"""Analyse the Phase 2 screening (results/phase2/screen_B*.csv).

Quality of a run = gap (%) to the best cost known for its instance: the best
feasible cost over *all* screening runs, or the BKS when one exists and is
better (XML100). Each configuration is compared to `default` on the same
(instance, seed) pairs:
  delta     mean paired gap difference (negative = better than default)
  wins/loss pairs where the configuration is strictly better / worse
  p_holm    two-sided Wilcoxon signed-rank p-value, Holm-corrected within a budget
  time_x    mean run time relative to default (options change the cost of a step)

The bench.sweep.SORTED_KNN rows come from results/p1_sorted_knn/ instead
(re-run with sorted kNN lists, against a sorted default; column `lists`).

    uv run python -m bench.analyze_sweep
writes results/phase2/screen_summary.csv and screen_by_size.csv, and
results/p1_sorted_knn/sort_check.csv (sorting alone, stock configuration)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from bench.common import ROOT, load_bks
from bench.sweep import CONFIGS, SORTED_KNN

OUT = ROOT / "results" / "phase2"
SORTED_DIR = ROOT / "results" / "p1_sorted_knn"


def size_group(n: int) -> str:
    if n <= 100:
        return "n<=100"
    if n <= 500:
        return "n=200-500"
    if n <= 1000:
        return "n=600-1000"
    return "n=2000"


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def paired(budget: str, tag: str, x: pd.DataFrame, base: pd.DataFrame) -> dict:
    """x against base on the same (instance, seed) pairs; both indexed by them."""
    j = base.index.intersection(x.index)
    diff = x.gap[j] - base.gap[j]
    p = wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
    return {"budget": budget, "tag": tag, "pairs": len(j), "gap": x.gap[j].mean(), "delta": diff.mean(),
            "wins": int((diff < 0).sum()), "losses": int((diff > 0).sum()), "p": p,
            "time_x": x.time_s[j].mean() / base.time_s[j].mean()}


def main() -> None:
    files = sorted(OUT.glob("screen_B*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df.tag.isin(CONFIGS)]    # configurations dropped from the study stay on disk only
    # The SORTED_KNN rows were re-run with sorted kNN lists (2026-10-19), each
    # with its own sorted `default`; they replace the old rows of the same name
    # and are compared with that default, inside the same Holm family.
    resorted = sorted(SORTED_DIR.glob("screen_B*.csv"))
    sk = pd.concat([pd.read_csv(f) for f in resorted], ignore_index=True) if resorted else df.iloc[:0]
    sk = sk[sk.tag.isin(["default", *SORTED_KNN])].assign(lists="sorted")
    df = df[~df.tag.isin(set(sk.tag) - {"default"})].assign(lists="heap")
    df = pd.concat([df, sk], ignore_index=True)
    bad = df[df.feasible != 1]
    if len(bad):
        print(f"WARNING: {len(bad)} infeasible runs excluded:\n{bad.groupby(['budget', 'tag']).size()}")
    df = df[df.feasible == 1].copy()

    bks = load_bks()
    ref = df.groupby("instance").cost.min()
    ref = pd.Series({i: min(c, bks.get(i, np.inf)) for i, c in ref.items()})
    df["gap"] = 100 * (df.cost - df.instance.map(ref)) / df.instance.map(ref)
    df["group"] = df.n.map(size_group)

    rows, checks = [], []
    for budget, d in df.groupby("budget"):
        idx = lambda v: v.set_index(["instance", "seed"])
        base = {k: idx(d[(d.tag == "default") & (d.lists == k)]) for k in ("heap", "sorted")}
        if base["heap"].empty:
            continue
        if not base["sorted"].empty:   # sorting alone: stock config, sorted vs heap-order lists
            checks.append(paired(budget, "default", base["sorted"], base["heap"]))
        part = []
        for (tag, lists), dt in d.groupby(["tag", "lists"]):
            if tag == "default" and lists == "sorted":
                continue
            part.append({**paired(budget, tag, idx(dt), base[lists]), "lists": lists,
                         "extra": dt.extra.iloc[0] if tag != "default" else ""})
        part = pd.DataFrame(part)
        part["p_holm"] = holm(part.p.to_numpy())
        rows.append(part.sort_values("delta"))
    summary = pd.concat(rows, ignore_index=True)
    summary.to_csv(OUT / "screen_summary.csv", index=False)
    if checks:
        pd.DataFrame(checks).to_csv(SORTED_DIR / "sort_check.csv", index=False)
        print("sorting alone (stock, sorted vs heap-order lists):")
        print(pd.DataFrame(checks)[["budget", "pairs", "delta", "wins", "losses", "p", "time_x"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    df = df[~((df.tag == "default") & (df.lists == "sorted"))]

    by_size = df.pivot_table(index="tag", columns=["budget", "group"], values="gap", aggfunc="mean")
    by_size.to_csv(OUT / "screen_by_size.csv")

    pd.set_option("display.width", 200, "display.max_rows", 500)
    for budget, s in summary.groupby("budget"):
        print(f"\n=== {budget} ===")
        print(s[["tag", "pairs", "gap", "delta", "wins", "losses", "p_holm", "time_x"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
