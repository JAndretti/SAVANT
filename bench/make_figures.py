"""Figure helpers of the manuscript: the page widths, solver labels and styles
that bench.make_paper and bench.plot_xml100_cases use, and
gap_vs_sets (gap_by_size_uniform.pdf). Not a command; matplotlib is optional
(uv sync --extra viz), the cluster does not need it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402  (after the backend choice)

from bench.common import PAPER           # noqa: E402

FIGURES = PAPER / "figures"

# Measured from the cas-dc class of the manuscript (\the\columnwidth and
# \the\textwidth), in TeX points, converted at 72.27 pt/inch. A figure drawn at
# exactly this size is included without scaling, so its 8 pt labels really print
# at 8 pt; anything wider overfills the column.
COLUMN_IN = 238.25444 / 72.27      # 3.2967 in, one column
TEXT_IN = 494.50888 / 72.27        # 6.8426 in, both columns (figure*)

LABEL = {"savant": "SAVANT", "hgs": "HGS-CVRP", "filo": "FILO", "filo2": "FILO2",
         "ails": "AILS-II", "lkh": "LKH-3", "ortools": "OR-Tools", "nds": "NDS"}
# one visual identity per solver, readable in grey-scale print: SAVANT alone is
# solid and thick, the baselines are distinguished by dash pattern and marker
STYLE = {"savant":  dict(color="#000000", ls="-",  marker="o", lw=1.8, zorder=5),
         "hgs":     dict(color="#0072B2", ls="--", marker="s"),
         "filo":    dict(color="#009E73", ls="-.", marker="^"),
         "filo2":   dict(color="#56B4E9", ls=":",  marker="v"),
         "ails":    dict(color="#D55E00", ls="--", marker="D"),
         "lkh":     dict(color="#CC79A7", ls="-.", marker="P"),
         "ortools": dict(color="#666666", ls=":",  marker="X"),
         "nds":     dict(color="#E69F00", ls="-.", marker="*")}
BASE = dict(lw=1.2, ms=3.5, mfc="none", mew=1.0)


def style(solver: str) -> dict:
    return BASE | STYLE.get(solver, dict(ls="-", marker="."))


def gap_vs_sets(dirs: list[Path], out: Path, width: float) -> None:
    """Mean gap against n across campaigns of one size each (the uniform sets)."""
    rows = []
    for d in dirs:
        s = pd.read_csv(d / "summary.csv")
        n = int(pd.read_csv(d / "per_instance.csv").n.iloc[0])
        rows += [(r.solver, n, r.gap_mean) for r in s.itertuples()]
    t = pd.DataFrame(rows, columns=["solver", "n", "gap"]).pivot(index="solver", columns="n", values="gap")
    t = t.loc[[s for s in LABEL if s in t.index]]
    fig, ax = plt.subplots(figsize=(width, width * 0.55))
    x = range(t.shape[1])
    for solver, row in t.iterrows():
        ax.plot(x, row.to_numpy(), label=LABEL.get(solver, solver), **style(solver))
    ax.set_xticks(list(x), [str(c) for c in t.columns], fontsize=7)
    ax.set_xlabel("number of customers")
    ax.set_ylabel("mean gap (%)")
    ax.set_yscale("log")
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    ax.legend(frameon=False, ncols=3, fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, -0.30), handlelength=2.2, columnspacing=1.2,
              handletextpad=0.5, borderaxespad=0.0)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {out}")
