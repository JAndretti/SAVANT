"""Analyse the Phase 2 screening (results/phase2/screen_B*.csv).

Quality of a run = gap (%) to the best cost known for its instance: the best
feasible cost over *all* screening runs, or the BKS when one exists and is
better (XML100). Each configuration is compared to `default` on the same
(instance, seed) pairs:
  delta     mean paired gap difference (negative = better than default)
  wins/loss pairs where the configuration is strictly better / worse
  p_holm    two-sided Wilcoxon signed-rank p-value, Holm-corrected within a budget
  time_x    mean run time relative to default (options change the cost of a step)

    uv run python -m bench.analyze_sweep
writes results/phase2/screen_summary.csv and screen_by_size.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from bench.common import ROOT, load_bks
from bench.sweep import CONFIGS

OUT = ROOT / "results" / "phase2"


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


def main() -> None:
    files = sorted(OUT.glob("screen_B*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df.tag.isin(CONFIGS)]    # configurations dropped from the study stay on disk only
    bad = df[df.feasible != 1]
    if len(bad):
        print(f"WARNING: {len(bad)} infeasible runs excluded:\n{bad.groupby(['budget', 'tag']).size()}")
    df = df[df.feasible == 1].copy()

    bks = load_bks()
    ref = df.groupby("instance").cost.min()
    ref = pd.Series({i: min(c, bks.get(i, np.inf)) for i, c in ref.items()})
    df["gap"] = 100 * (df.cost - df.instance.map(ref)) / df.instance.map(ref)
    df["group"] = df.n.map(size_group)

    rows = []
    for budget, d in df.groupby("budget"):
        base = d[d.tag == "default"].set_index(["instance", "seed"])
        if base.empty:
            continue
        part = []
        for tag, dt in d.groupby("tag"):
            x = dt.set_index(["instance", "seed"])
            j = base.index.intersection(x.index)
            diff = x.gap[j] - base.gap[j]
            p = wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
            part.append({"budget": budget, "tag": tag, "extra": dt.extra.iloc[0] if tag != "default" else "",
                         "pairs": len(j), "gap": x.gap[j].mean(), "delta": diff.mean(),
                         "wins": int((diff < 0).sum()), "losses": int((diff > 0).sum()), "p": p,
                         "time_x": x.time_s[j].mean() / base.time_s[j].mean()})
        part = pd.DataFrame(part)
        part["p_holm"] = holm(part.p.to_numpy())
        rows.append(part.sort_values("delta"))
    summary = pd.concat(rows, ignore_index=True)
    summary.to_csv(OUT / "screen_summary.csv", index=False)

    by_size = df.pivot_table(index="tag", columns=["budget", "group"], values="gap", aggfunc="mean")
    by_size.to_csv(OUT / "screen_by_size.csv")

    pd.set_option("display.width", 200, "display.max_rows", 500)
    for budget, s in summary.groupby("budget"):
        print(f"\n=== {budget} ===")
        print(s[["tag", "pairs", "gap", "delta", "wins", "losses", "p_holm", "time_x"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
