"""Analyse the factorial interaction study (results/phase2/interact_*.csv).

Three views of the same runs, all paired on (instance, seed) and measured as the
gap (%) to the best cost known for the instance:

  cells          every combination against `base` (the default configuration),
                 Holm-corrected within a budget;
  main effects   for each factor, the mean paired difference between the cells
                 where it is on and the matching cells where it is off (8 cell
                 pairs per factor with 4 factors), which is what the screening
                 estimated with a single pair;
  interactions   for each pair of factors, the effect of the first when the
                 second is on minus its effect when the second is off; a value
                 far from zero means the screening cannot be trusted for them.

    uv run python -m bench.analyze_interact
writes results/phase2/interact_summary.csv
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from bench.common import ROOT, load_bks
from bench.interact import FACTORS, cell_name

OUT = ROOT / "results" / "phase2"


def factors_of(tag: str) -> frozenset[str]:
    return frozenset() if tag == "base" else frozenset(tag[2:].split("+"))


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def paired(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.Series, float]:
    """gap(a) - gap(b) on the (instance, seed) pairs both ran."""
    j = a.index.intersection(b.index)
    diff = a.gap[j] - b.gap[j]
    p = wilcoxon(diff).pvalue if len(diff) and (diff != 0).any() else 1.0
    return diff, p


def main() -> None:
    files = sorted(OUT.glob("interact_*.csv"))
    files = [f for f in files if "summary" not in f.name]
    if not files:
        raise SystemExit("no interaction results yet")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    allr = pd.concat([df] + [pd.read_csv(f) for f in
                             sorted(OUT.glob("screen_B*.csv")) + sorted(OUT.glob("timematch_f*.csv"))],
                     ignore_index=True)
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
        base = cells.get("base")
        if base is None:
            continue
        names = sorted({f for t in cells for f in factors_of(t)}, key=list(FACTORS).index)

        print(f"\n=== {budget} ===   base gap {base.gap.mean():.4f} %, "
              f"{len(cells)} cells, {len(base)} runs each")

        part = []
        for tag, x in cells.items():
            diff, p = paired(x, base)
            part.append({"budget": budget, "view": "cell", "name": tag, "delta": diff.mean(),
                         "wins": int((diff < 0).sum()), "losses": int((diff > 0).sum()),
                         "p": p, "gap": x.gap.mean(),
                         "time_x": x.time_s.mean() / base.time_s.mean(),
                         "steps_x": x.steps.astype(float).mean() / base.steps.astype(float).mean()})
        part = pd.DataFrame(part)
        part["p_holm"] = holm(part.p.to_numpy())
        print("-- cells (best first)")
        print(part.sort_values("delta")[["name", "gap", "delta", "wins", "losses", "p_holm", "steps_x"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        rows.append(part)

        eff = {}
        for f in names:
            diffs = []
            for tag, x in cells.items():
                if f in factors_of(tag):
                    off = cells.get(cell_name(tuple(sorted(factors_of(tag) - {f},
                                                           key=list(FACTORS).index))))
                    if off is not None:
                        diffs.append(paired(x, off)[0])
            if diffs:
                s = pd.concat(diffs)
                eff[f] = s
                p = wilcoxon(s).pvalue if (s != 0).any() else 1.0
                rows.append(pd.DataFrame([{"budget": budget, "view": "main", "name": f,
                                           "delta": s.mean(), "wins": int((s < 0).sum()),
                                           "losses": int((s > 0).sum()), "p": p, "p_holm": p}]))
        print("-- main effects (negative = the factor helps)")
        for f, s in sorted(eff.items(), key=lambda kv: kv[1].mean()):
            print(f"   {f:10s} {s.mean():+.4f}  over {len(s)} paired runs")

        print("-- two-factor interactions (|value| small = additive)")
        for f, g in itertools.combinations(names, 2):
            on, off = [], []
            for tag, x in cells.items():
                fs = factors_of(tag)
                if f in fs:
                    base_tag = cell_name(tuple(sorted(fs - {f}, key=list(FACTORS).index)))
                    if base_tag in cells:
                        (on if g in fs else off).append(paired(x, cells[base_tag])[0].mean())
            if on and off:
                inter = float(np.mean(on) - np.mean(off))
                print(f"   {f:10s} x {g:10s} {inter:+.4f}")
                rows.append(pd.DataFrame([{"budget": budget, "view": "interaction",
                                           "name": f"{f}x{g}", "delta": inter}]))

    pd.concat(rows, ignore_index=True).to_csv(OUT / "interact_summary.csv", index=False)
    print(f"\nwrote {OUT / 'interact_summary.csv'}")


if __name__ == "__main__":
    main()
