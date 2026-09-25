"""Appendix figure: SAVANT's solution against the optimal one on the XML100
instances that carry most of its gap (paper, Section on XML100).

    uv run --extra viz python -m bench.plot_xml100_cases \
        --case XML100_3241_08=results/diag/xml_sol/XML100_3241_08.sol \
        --case XML100_3222_15=results/diag/xml_sol/XML100_3222_15.sol

The campaign stores costs, not routes, so SAVANT's solutions are regenerated
(`cw --round --sa-wall 240 --seed 1 --sol ...`, the configuration of the paper);
the script checks them with the independent checker and prints their gap, to be
compared with the campaign's before the figure is used.

Left: the optimal solution (CVRPLIB), its arcs whose endpoints are not among each
other's 20 nearest customers in red -- arcs SAVANT's candidate lists never propose.
Right: SAVANT's solution, its customer-to-customer arcs absent from the optimal
solution in blue. Depot arcs are always dotted grey (routes differ, so almost all
of them would differ); customer size is proportional to demand.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402

from bench.common import DATA, check_solution, read_cvrplib_sol, read_cwsol, read_vrp  # noqa: E402
from bench.make_figures import FIGURES, TEXT_IN          # noqa: E402

K = 20
RED, BLUE, GREY = "#D55E00", "#0072B2", "#9a9a9a"


def arcs(routes: list[list[int]]) -> set[frozenset[int]]:
    """Undirected arcs, the depot being 0."""
    return {frozenset(e) for r in routes for e in zip([0, *r], [*r, 0])}


def ranks(xy: np.ndarray) -> np.ndarray:
    d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    np.fill_diagonal(d, np.inf)
    return d.argsort(axis=1).argsort(axis=1)              # rank[u, v], 0 = nearest


def panel(ax, inst, routes, highlight: set[frozenset[int]], colour: str, title: str) -> None:
    xy = inst.xy
    for e in arcs(routes):
        a, b = sorted(e)
        hot = e in highlight
        ax.plot(*xy[[a, b]].T, color=colour if hot else GREY,
                lw=1.6 if hot else 0.6, ls=":" if a == 0 and not hot else "-",
                zorder=3 if hot else 1)
    ax.scatter(*xy[1:].T, s=4 + 18 * inst.dem[1:] / inst.dem[1:].max(), color="black",
               lw=0, zorder=2)
    ax.scatter(*xy[0], marker="s", s=28, color="black", zorder=4)
    ax.set_title(title, fontsize=7.5)
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", action="append", required=True, metavar="NAME=SAVANT_SOL")
    ap.add_argument("--out", type=Path, default=FIGURES / "xml100_cases.pdf")
    a = ap.parse_args()
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "pdf.fonttype": 42})

    cases = [c.split("=", 1) for c in a.case]
    fig, axes = plt.subplots(len(cases), 2, figsize=(TEXT_IN, TEXT_IN * 0.47 * len(cases)))
    axes = np.atleast_2d(axes)
    for (name, sol), (ax_opt, ax_sav) in zip(cases, axes):
        inst = read_vrp(DATA / "cvrplib" / "XML100" / f"{name}.vrp")
        opt, _ = read_cvrplib_sol(DATA / "cvrplib" / "XML100" / f"{name}.sol")
        sav = read_cwsol(sol)[name][1]
        opt_cost, ok_o, _ = check_solution(inst, opt, True)
        sav_cost, ok_s, msg = check_solution(inst, sav, True)
        assert ok_o and ok_s, msg
        gap = 100 * (sav_cost - opt_cost) / opt_cost
        rk = ranks(inst.xy[1:])
        outer = {e for e in arcs(opt) if 0 not in e
                 and min(rk[min(e) - 1, max(e) - 1], rk[max(e) - 1, min(e) - 1]) >= K}
        new = {e for e in arcs(sav) - arcs(opt) if 0 not in e}      # customer-customer only
        print(f"{name}: optimum {opt_cost:.0f} ({len(opt)} routes, {len(outer)} arcs beyond rank {K}); "
              f"SAVANT {sav_cost:.0f} ({len(sav)} routes, +{gap:.2f} %, {len(new)} customer arcs not in the optimum)")
        label = name.replace("_", r"\_") if plt.rcParams["text.usetex"] else name
        panel(ax_opt, inst, opt, outer, RED,
              f"{label}, optimal solution: {len(opt)} routes, cost {opt_cost:.0f}")
        panel(ax_sav, inst, sav, new, BLUE,
              f"{label}, SAVANT: {len(sav)} routes, cost {sav_cost:.0f} (+{gap:.2f}%)")
    fig.tight_layout(pad=0.3, w_pad=0.8, h_pad=0.8)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
