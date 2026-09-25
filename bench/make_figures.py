"""Figures for the manuscript, from the files bench.analyze_compare writes.

    uv sync --extra viz          # matplotlib is optional: the cluster does not need it
    uv run python -m bench.make_figures --set results/jz/X --name X

Writes publication/paper/figures/convergence_<name>.pdf (vector, so it prints at
any resolution) and, when the set spans a range of sizes, gap_by_size_<name>.pdf.

The convergence figure shows the best solution found so far *within a run of T_max*, not what a
shorter run would produce: SAVANT, FILO, FILO2 and AILS-II all schedule their
search around T_max (flaw #23). The caption must say so.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402  (after the backend choice)

from bench.common import ROOT           # noqa: E402

FIGURES = ROOT / "publication" / "paper" / "figures"

# Measured from the cas-dc class of publication/paper (\the\columnwidth and
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


def convergence(cp: pd.DataFrame, out: Path, width: float) -> None:
    gaps = cp[cp.stat == "gap"].set_index("solver").drop(columns="stat")
    gaps.columns = [float(c) for c in gaps.columns]
    fig, ax = plt.subplots(figsize=(width, width * 0.62))
    for solver, row in gaps.iterrows():
        ax.plot(row.index * 100, row.to_numpy(), label=LABEL.get(solver, solver), **style(solver))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"fraction of $T_{\max}$ (\%)" if plt.rcParams["text.usetex"]
                  else "fraction of $T_{max}$ (%)")
    ax.set_ylabel("mean gap to reference (%)")
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    # below the axes: with seven curves converging into the same corner, any
    # in-axes placement sits on top of them
    ax.legend(frameon=False, ncols=3, fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, -0.42), handlelength=2.2, columnspacing=1.2,
              handletextpad=0.5, borderaxespad=0.0)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {out}")


def gap_by_size(pi: pd.DataFrame, out: Path, width: float) -> None:
    """Mean gap against instance size: the effect the pooled table hides."""
    d = pi.copy()
    # same convention as the by-size table of bench.analyze_compare: right-closed,
    # so an instance with n = 1000 belongs to 600-1000 and not to a bucket of its own
    edges = [0, 200, 400, 600, 1000, 2000, 5000, 10 ** 9]
    names = ["<200", "200-400", "400-600", "600-1k", "1k-2k", "2k-5k", ">5k"]
    d["bucket"] = pd.cut(d.n, bins=edges, labels=names, right=True)
    t = d.groupby(["solver", "bucket"], observed=True).gap_mean.mean().unstack("bucket")
    t = t.dropna(axis=1, how="all")
    order = [s for s in LABEL if s in t.index] + sorted(set(t.index) - set(LABEL))
    t = t.loc[order]
    if t.shape[1] < 2:
        print("single size bucket: no gap-by-size figure")
        return
    fig, ax = plt.subplots(figsize=(width, width * 0.55))
    x = range(t.shape[1])
    for solver, row in t.iterrows():
        ax.plot(x, row.to_numpy(), label=LABEL.get(solver, solver), **style(solver))
    ax.set_xticks(list(x), list(t.columns), fontsize=7)
    ax.set_xlabel("number of customers")
    ax.set_ylabel("mean gap (%)")
    ax.set_yscale("log")
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    ax.legend(frameon=False, ncols=3, fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, -0.46), handlelength=2.2, columnspacing=1.2,
              handletextpad=0.5, borderaxespad=0.0)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {out}")


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", type=Path, help="directory analysed by analyze_compare")
    ap.add_argument("--sizes", nargs="+", type=Path,
                    help="several one-size campaigns: draw gap_by_size_<name>.pdf across them")
    ap.add_argument("--name", required=True, help="suffix of the figure files, e.g. X")
    ap.add_argument("--out", type=Path, default=FIGURES)
    ap.add_argument("--double", action="store_true",
                    help="draw at the width of both columns, for a figure* environment")
    ap.add_argument("--width", type=float, default=None,
                    help="inches; the default matches the template exactly "
                         f"({COLUMN_IN:.4f}, or {TEXT_IN:.4f} with --double)")
    a = ap.parse_args()
    if a.width is None:
        a.width = TEXT_IN if a.double else COLUMN_IN
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "axes.linewidth": 0.6,
                         "font.family": "serif", "pdf.fonttype": 42})
    a.out.mkdir(parents=True, exist_ok=True)
    if a.sizes:
        gap_vs_sets(a.sizes, a.out / f"gap_by_size_{a.name}.pdf", a.width)
        return
    if a.set is None:
        ap.error("--set or --sizes is required")

    cp = a.set / "checkpoints.csv"
    if cp.exists():
        convergence(pd.read_csv(cp), a.out / f"convergence_{a.name}.pdf", a.width)
    else:
        print(f"{cp} missing: run bench.analyze_compare first")

    pin = a.set / "per_instance.csv"
    if pin.exists():
        gap_by_size(pd.read_csv(pin), a.out / f"gap_by_size_{a.name}.pdf", a.width)
    else:
        print(f"{pin} missing: run bench.analyze_compare first")


if __name__ == "__main__":
    main()
