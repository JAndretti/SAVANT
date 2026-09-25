"""Analyse the time-matched control experiment (results/phase2/timematch_f*.csv).

Same measure as the screening: gap (%) to the best cost known for the instance,
taken over every run of this study (screening + time-matched) and the BKS where
one exists. Configurations are compared with `default` on the same
(instance, seed) pairs, two-sided Wilcoxon signed-rank.

    uv run python -m bench.analyze_timematch
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from bench.common import ROOT, load_bks

OUT = ROOT / "results" / "phase2"


def main() -> None:
    tm_files = sorted(OUT.glob("timematch_f*.csv"))
    if not tm_files:
        raise SystemExit("no time-matched results yet")
    tm = pd.concat([pd.read_csv(f) for f in tm_files], ignore_index=True)
    allr = pd.concat([tm] + [pd.read_csv(f) for f in sorted(OUT.glob("screen_B*.csv"))],
                     ignore_index=True)
    bad = tm[tm.feasible != 1]
    if len(bad):
        print(f"WARNING: {len(bad)} infeasible runs excluded")
    tm = tm[tm.feasible == 1].copy()

    bks = load_bks()
    ref = allr[allr.feasible == 1].groupby("instance").cost.min()
    ref = pd.Series({i: min(c, bks.get(i, np.inf)) for i, c in ref.items()})
    tm["gap"] = 100 * (tm.cost - tm.instance.map(ref)) / tm.instance.map(ref)

    rows = []
    for budget, d in tm.groupby("budget"):
        base = d[d.tag == "default"].set_index(["instance", "seed"])
        print(f"\n=== {budget} ===   default gap {base.gap.mean():.4f} %, "
              f"mean time {base.time_s.mean():.1f} s, mean steps {base.steps.astype(float).mean():.3g}")
        for tag, dt in d.groupby("tag"):
            if tag == "default":
                continue
            x = dt.set_index(["instance", "seed"])
            j = base.index.intersection(x.index)
            diff = x.gap[j] - base.gap[j]
            p = wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
            rows.append({"budget": budget, "tag": tag, "pairs": len(j),
                         "base_time_s": base.time_s[j].mean(), "base_gap": base.gap[j].mean(),
                         "gap": x.gap[j].mean(), "delta": diff.mean(),
                         "wins": int((diff < 0).sum()), "losses": int((diff > 0).sum()), "p": p,
                         "steps_x": x.steps.astype(float)[j].mean() / base.steps.astype(float)[j].mean(),
                         "time_x": x.time_s[j].mean() / base.time_s[j].mean()})
            r = rows[-1]
            print(f"{tag:20s} gap {r['gap']:.4f}  delta {r['delta']:+.4f}  "
                  f"wins {r['wins']}  losses {r['losses']}  p {r['p']:.3g}  "
                  f"steps x{r['steps_x']:.2f}  time x{r['time_x']:.3f}")
    pd.DataFrame(rows).to_csv(OUT / "timematch_summary.csv", index=False)


if __name__ == "__main__":
    main()
