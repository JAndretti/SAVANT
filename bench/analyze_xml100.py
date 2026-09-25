"""Where does SAVANT miss the optimum on XML100, and why? (paper, Section on XML100)

    uv run python -m bench.analyze_xml100                  # after bench.analyze_compare on the set
    uv run python -m bench.analyze_xml100 --knn results/diag/xml100_knn.csv

Reads the XML100_test campaign (SAVANT and HGS-CVRP, 10 runs per instance) and,
for every instance, the optimal solution distributed with XML100. An instance is
"missed" by a solver when its mean gap over the runs is above zero, i.e. when at
least one run did not reach the optimum. It reports

  - the instances that carry the gap and whether SAVANT's fleet differs from the optimum's;
  - the misses by the four XML100 attributes (Queiroga et al. 2022: depot position,
    customer position, demand distribution, average route size), counted rather than
    averaged, so that one extreme instance cannot drive a whole attribute level;
  - the arcs of each optimal solution whose endpoints are not within each other's
    K = 20 nearest customers, i.e. arcs SAVANT's candidate lists never propose;
  - with --knn: the control experiment, SAVANT with K = 20 against K = 40 on the
    instances it missed (same machine, same budget, 3 seeds; tags k20 and k40).

Writes <set>/savant_diagnosis.csv and publication/paper/tables/xml100.tex.
"""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, wilcoxon

from bench.common import DATA, ROOT, read_cvrplib_sol, read_vrp

K = 20                                        # SAVANT's --sa-knn default
TABLES = ROOT / "publication" / "paper" / "tables"
SHORT = ("1", "2")                            # route-size codes: 3-5 and 5-8 customers per route


def runs(set_dir: Path) -> pd.DataFrame:
    r = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(set_dir / "tmax_*_shard*.csv")))])
    bks = pd.read_csv(DATA / "cvrplib" / "XML100_bks.csv").set_index("name")
    r["gap"] = 100 * (r.cost - r.instance.map(bks.bks_cost)) / r.instance.map(bks.bks_cost)
    r["k_opt"] = r.instance.map(bks.bks_routes)
    return r


def outside_arcs(name: str) -> tuple[int, int]:
    """Arcs of the optimal solution outside both endpoints' K-nearest lists, and
    the largest neighbour rank (1-based) the optimal solution uses."""
    inst = read_vrp(DATA / "cvrplib" / "XML100" / f"{name}.vrp")
    routes, _ = read_cvrplib_sol(DATA / "cvrplib" / "XML100" / f"{name}.sol")
    xy = inst.xy[1:]
    d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    np.fill_diagonal(d, np.inf)
    rank = d.argsort(axis=1).argsort(axis=1)              # rank[u, v]: 0 = nearest
    ranks = [min(rank[a - 1, b - 1], rank[b - 1, a - 1]) for r in routes for a, b in zip(r, r[1:])]
    return sum(k >= K for k in ranks), max(ranks) + 1


def diagnose(set_dir: Path) -> pd.DataFrame:
    r = runs(set_dir)
    f = r.groupby(["instance", "solver"]).gap.mean().unstack("solver")
    f.columns = [f"{c}_gap" for c in f.columns]
    extra = (r.routes - r.k_opt).groupby([r.instance, r.solver]).mean().unstack("solver")
    f["savant_extra_routes"] = extra["savant"]
    f["k_opt"] = r.groupby("instance").k_opt.first()
    rows = {}
    for name in f.index:
        inst = read_vrp(DATA / "cvrplib" / "XML100" / f"{name}.vrp")
        out, worst = outside_arcs(name)
        rows[name] = dict(lb=math.ceil(inst.dem.sum() / inst.cap),
                          fill=inst.dem.sum() / (f.k_opt[name] * inst.cap),
                          cust_per_route=inst.n / f.k_opt[name],
                          opt_arcs_outside_K20=out, max_rank=worst)
    f = f.join(pd.DataFrame.from_dict(rows, orient="index"))
    f["code"] = f.index.str.split("_").str[1]
    f["savant_miss"] = f.savant_gap > 1e-9
    f["hgs_miss"] = f.hgs_gap > 1e-9
    f.to_csv(set_dir / "savant_diagnosis.csv")
    return f


def knn_control(path: Path, f: pd.DataFrame) -> dict:
    r = pd.read_csv(path)
    bks = pd.read_csv(DATA / "cvrplib" / "XML100_bks.csv").set_index("name")
    r["gap"] = 100 * (r.cost - r.instance.map(bks.bks_cost)) / r.instance.map(bks.bks_cost)
    pi = r.groupby(["tag", "instance"]).gap.mean().unstack("tag")
    out = f.opt_arcs_outside_K20.reindex(pi.index) > 0
    res = {"n": len(pi), "infeasible": int((r.feasible == 0).sum()),
           "k20": pi.k20.mean(), "k40": pi.k40.mean(),
           "steps_ratio": r[r.tag == "k40"].steps.astype(float).mean()
           / r[r.tag == "k20"].steps.astype(float).mean()}
    for key, m in (("out", out), ("in", ~out)):
        d = (pi.k40 - pi.k20)[m]
        nz = d[d != 0]
        res[key] = dict(n=int(m.sum()), k20=pi.k20[m].mean(), k40=pi.k40[m].mean(),
                        better=int((d < 0).sum()), worse=int((d > 0).sum()), equal=int((d == 0).sum()),
                        p=wilcoxon(nz).pvalue if len(nz) > 1 else float("nan"))
    print(f"\n=== K = 20 vs K = 40 on the {res['n']} missed instances ({res['infeasible']} infeasible) ===")
    print(f"mean gap {res['k20']:.3f} -> {res['k40']:.3f} %, steps ratio K40/K20 {res['steps_ratio']:.2f}")
    for key in ("out", "in"):
        g = res[key]
        print(f"  optimum {'uses' if key == 'out' else 'avoids'} arcs outside K=20 ({g['n']}): "
              f"{g['k20']:.3f} -> {g['k40']:.3f}, {g['better']} better / {g['worse']} worse / "
              f"{g['equal']} equal, p = {g['p']:.3g}")
    for name in pi.index[pi.k20 > 1]:
        print(f"  {name}: {pi.k20[name]:.3f} -> {pi.k40[name]:.3f}; routes with K=40:",
              r[(r.instance == name) & (r.tag == "k40")].routes.tolist())
    return res


def table(f: pd.DataFrame) -> None:
    """Misses of each solver by the factors that matter, as counts."""
    short = f.code.str[3].isin(SHORT)
    out = f.opt_arcs_outside_K20 > 0
    groups = [("All instances", pd.Series(True, index=f.index)),
              (r"Routes of $3$--$8$ customers", short),
              (r"Routes of $8$--$50$ customers", ~short),
              (r"Unitary demands", f.code.str[2] == "1"),
              (r"Optimal arc beyond rank $20$", out),
              (r"All optimal arcs within rank $20$", ~out)]
    lines = [r"\begin{table}[t]",
             r"\caption{Instances of the XML100 test set on which a solver misses the optimum in at "
             r"least one of its $10$ runs, by group. The last two rows split the instances by whether "
             r"the optimal solution contains an arc whose endpoints are not among each other's $20$ "
             r"nearest customers, an arc the candidate lists of \savant\ never propose.}\label{tab:xml}",
             r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lrrr@{}}",
             r"\toprule",
             r"Group & Inst. & \savant & HGS \\",
             r"\midrule"]
    for label, m in groups:
        lines.append(f"{label} & {int(m.sum())} & {int(f.savant_miss[m].sum())} & "
                     f"{int(f.hgs_miss[m].sum())} \\\\")
        if label.startswith("All") or label.startswith("Unitary"):
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "xml100.tex").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {TABLES / 'xml100.tex'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", type=Path, default=ROOT / "results" / "jz" / "XML100_test")
    ap.add_argument("--knn", type=Path, help="CSV of the K=20/K=40 control (tags k20, k40)")
    a = ap.parse_args()
    f = diagnose(a.set)
    tot = f.savant_gap.sum()
    top = f.savant_gap.nlargest(2)
    print(f"SAVANT misses {f.savant_miss.sum()} instances, HGS-CVRP {f.hgs_miss.sum()}")
    print(f"mean gap SAVANT {f.savant_gap.mean():.4f} %, HGS-CVRP {f.hgs_gap.mean():.4f} %; "
          f"the two worst instances carry {top.sum() / tot:.0%} of SAVANT's total, "
          f"without them {f.savant_gap.drop(top.index).mean():.4f} %")
    print(f.loc[top.index, ["savant_gap", "hgs_gap", "savant_extra_routes", "k_opt", "lb", "fill",
                            "opt_arcs_outside_K20", "max_rank"]].round(3).to_string())
    print(f"runs with more routes than the optimum: {f.index[f.savant_extra_routes > 0].tolist()}")
    for label, m in (("short routes (codes 1-2)", f.code.str[3].isin(SHORT)),
                     ("optimum uses arcs outside K=20", f.opt_arcs_outside_K20 > 0)):
        t = pd.crosstab(m, f.savant_miss)
        print(f"missed | {label}: {int(f.savant_miss[m].sum())}/{int(m.sum())} vs "
              f"{int(f.savant_miss[~m].sum())}/{int((~m).sum())}, Fisher p = {fisher_exact(t).pvalue:.3g}")
    print("median largest rank used by the optimum: missed",
          f.max_rank[f.savant_miss].median(), "/ solved", f.max_rank[~f.savant_miss].median())
    table(f)
    if a.knn:
        knn_control(a.knn, f)


if __name__ == "__main__":
    main()
