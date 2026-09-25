"""Analyse the ablation of the retained configuration (results/phase2/ablate_f*.csv).

Each cell is compared with `final` on the same (instance, seed) pairs, measured
as the gap (%) to the best cost known for the instance. A positive difference
means the change degrades the solution, i.e. the removed component is worth
that much. Holm correction within a budget.

    uv run python -m bench.analyze_ablate
writes results/phase2/ablate_summary.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from bench.common import ROOT, load_bks

OUT = ROOT / "results" / "phase2"


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def main() -> None:
    files = [f for f in sorted(OUT.glob("ablate_f*.csv")) if "summary" not in f.name]
    if not files:
        raise SystemExit("no ablation results yet")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    others = sorted(OUT.glob("screen_B*.csv")) + sorted(OUT.glob("timematch_f*.csv")) \
        + [f for f in OUT.glob("interact_*.csv") if "summary" not in f.name]
    allr = pd.concat([df] + [pd.read_csv(f) for f in others], ignore_index=True)
    if (df.feasible != 1).any():
        print(f"WARNING: {int((df.feasible != 1).sum())} infeasible runs excluded")
    df = df[df.feasible == 1].copy()

    bks = load_bks()
    ref = allr[allr.feasible == 1].groupby("instance").cost.min()
    ref = pd.Series({i: min(c, bks.get(i, np.inf)) for i, c in ref.items()})
    df["gap"] = 100 * (df.cost - df.instance.map(ref)) / df.instance.map(ref)

    rows = []
    for budget, d in df.groupby("budget"):
        cells = {t: g.set_index(["instance", "seed"]) for t, g in d.groupby("tag")}
        base = cells.get("final")
        if base is None:
            continue
        print(f"\n=== {budget} ===   final gap {base.gap.mean():.4f} %, "
              f"mean time {base.time_s.mean():.1f} s, {len(base)} runs per cell")
        part = []
        for tag, x in cells.items():
            j = base.index.intersection(x.index)
            diff = x.gap[j] - base.gap[j]
            p = wilcoxon(diff).pvalue if len(diff) and (diff != 0).any() else 1.0
            part.append({"budget": budget, "tag": tag, "pairs": len(j), "base_gap": base.gap[j].mean(),
                         "gap": x.gap[j].mean(), "delta": diff.mean(),
                         "wins": int((diff < 0).sum()), "losses": int((diff > 0).sum()), "p": p,
                         "steps_x": x.steps.astype(float)[j].mean() / base.steps.astype(float)[j].mean(),
                         "time_x": x.time_s[j].mean() / base.time_s[j].mean()})
        part = pd.DataFrame(part)
        part["p_holm"] = holm(part.p.to_numpy())
        print(part.sort_values("delta", ascending=False)
              [["tag", "gap", "delta", "wins", "losses", "p_holm", "steps_x"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        rows.append(part)

    pd.concat(rows, ignore_index=True).to_csv(OUT / "ablate_summary.csv", index=False)
    print(f"\nwrote {OUT / 'ablate_summary.csv'}")


if __name__ == "__main__":
    main()
