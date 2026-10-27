"""Reanalyse existing runs and generate the manuscript's tables and figures.

    uv run python -m bench.make_paper

Reads the result files (results/v2_params, results/phase2, results/p1_sorted_knn,
results/jz, results/diag) and writes only under bench.common.PAPER: analysis/
(effects, tests, intervals, trace coverage, input checksums), tables/ and figures/.
No solver runs. Instances, not individual seeds or factorial cell contrasts, are
the test units. The gap reference of the parameter study is frozen in
data/tuning/reference_costs.csv: changing it on purpose means a new, identified
version, not a silent update of the gaps.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path


import numpy as np
import pandas as pd
import scipy
from scipy.stats import fisher_exact, wilcoxon

from bench import make_tables as old_tables
from bench.analyze_compare import checkpoints, load_runs, load_traces, per_instance, summary
from bench.common import PAPER, ROOT, holm, load_bks, run_csvs
from bench.interact import FACTORS, cell_name, factors_of
from bench.sweep import CONFIGS, SORTED_KNN

ANALYSIS = PAPER / "analysis"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"
REFERENCE = ROOT / "data/tuning/reference_costs.csv"   # frozen tuning reference
PARAMS = ROOT / "results/v2_params"
HISTORICAL = ROOT / "results/phase2"
SORTED = ROOT / "results/p1_sorted_knn"
SETS = ("X", "XL", "n100", "n500", "n1000", "n2000", "XML100_test")
INPUTS: set[Path] = set()


def read(path: Path) -> pd.DataFrame:
    INPUTS.add(path)
    return pd.read_csv(path)


def concat(paths) -> pd.DataFrame:
    return pd.concat([read(p) for p in sorted(paths)], ignore_index=True)


def test(diff: pd.Series | np.ndarray) -> float:
    d = np.round(np.asarray(diff, dtype=float), 12)
    assert np.isfinite(d).all()
    return float(wilcoxon(d, alternative="two-sided", zero_method="wilcox",
                          method="auto").pvalue) if (d != 0).any() else 1.0


def interval(diff) -> tuple[float, float]:
    """Percentile bootstrap of the mean, resampling complete instance effects."""
    d = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(20261027)
    boot = d[rng.integers(0, len(d), size=(9999, len(d)))].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def adjust(rows: list[dict]) -> pd.DataFrame:
    result = pd.DataFrame(rows)
    result["p_holm"] = holm(result.p.to_numpy())
    return result


def effect(diff: pd.Series, **meta) -> dict:
    assert diff.index.is_unique
    lo, hi = interval(diff)
    return {**meta, "instances": len(diff), "delta": diff.mean(),
            "ci_low": lo, "ci_high": hi, "p": test(diff),
            "better": int((diff < -1e-12).sum()),
            "worse": int((diff > 1e-12).sum())}


def paired(x: pd.DataFrame, base: pd.DataFrame, **meta) -> dict:
    """Match the same instance/seed runs, then average seed differences by instance."""
    keys = ["instance", "seed"]
    a = x.set_index(keys)
    b = base.set_index(keys)
    assert a.index.is_unique and b.index.is_unique
    assert a.index.difference(b.index).empty and b.index.difference(a.index).empty
    d = (a.gap - b.gap).groupby(level="instance").mean()
    row = effect(d, **meta)
    row.update(gap=x.gap.mean(), base_gap=base.gap.mean(),
               time_s=base.time_s.mean(),
               steps_x=x.steps.astype(float).mean() / base.steps.astype(float).mean())
    return row


def with_gap(runs: pd.DataFrame, ref: pd.Series) -> pd.DataFrame:
    assert (runs.feasible == 1).all(), "Parameter runs must all be feasible"
    assert runs.instance.map(ref).notna().all()
    return runs.assign(gap=100 * (runs.cost - runs.instance.map(ref)) / runs.instance.map(ref))


def parameter_reference() -> pd.Series:
    paths = run_csvs(PARAMS) + run_csvs(HISTORICAL) + list(SORTED.glob("screen_B*.csv"))
    all_runs = concat(paths)
    best = all_runs[all_runs.feasible == 1].groupby("instance").cost.min()
    for p in (ROOT / "data/cvrplib").glob("*_bks.csv"):
        INPUTS.add(p)
    bks = load_bks()
    proposed = pd.Series({i: min(c, bks.get(i, np.inf)) for i, c in best.items()},
                         name="ref_cost").rename_axis("instance")
    path = REFERENCE
    if path.exists():
        INPUTS.add(path)
        ref = pd.read_csv(path).set_index("instance").ref_cost
        assert set(proposed.index) == set(ref.index), "Frozen reference coverage changed"
        return ref
    proposed.to_csv(path)
    return proposed


def simple_study(ref: pd.Series, pattern: str, baseline: str) -> pd.DataFrame:
    runs = with_gap(concat(PARAMS.glob(pattern)), ref)
    rows = []
    for budget, d in runs.groupby("budget", sort=False):
        base = d[d.tag == baseline]
        part = [paired(g, base, budget=budget, tag=tag)
                for tag, g in d.groupby("tag") if tag != baseline]
        rows.append(adjust(part))
    return pd.concat(rows, ignore_index=True)


def screening(ref: pd.Series) -> pd.DataFrame:
    runs = concat(HISTORICAL.glob("screen_B*.csv"))
    runs = runs[runs.tag.isin(CONFIGS)].assign(lists="heap")
    sorted_runs = concat(SORTED.glob("screen_B*.csv"))
    sorted_runs = sorted_runs[sorted_runs.tag.isin(["default", *SORTED_KNN])].assign(lists="sorted")
    runs = runs[~runs.tag.isin(SORTED_KNN)]
    runs = with_gap(pd.concat([runs, sorted_runs], ignore_index=True), ref)
    output = []
    for budget, d in runs.groupby("budget"):
        part = []
        for (tag, lists), x in d.groupby(["tag", "lists"]):
            if tag == "default":
                continue
            base = d[(d.tag == "default") & (d.lists == lists)]
            row = paired(x, base, budget=budget, tag=tag, lists=lists)
            row["time_x"] = x.time_s.mean() / base.time_s.mean()
            part.append(row)
        assert len(part) == 51
        output.append(adjust(part))
    return pd.concat(output, ignore_index=True)


def factorial(ref: pd.Series) -> pd.DataFrame:
    runs = with_gap(concat(PARAMS.glob("interact_f*.csv")), ref)
    output = []
    for budget, d in runs.groupby("budget"):
        cells = {t: g.set_index(["instance", "seed"]).gap for t, g in d.groupby("tag")}
        names = sorted({f for t in cells for f in factors_of(t)}, key=list(FACTORS).index)

        def name(fs) -> str:
            return cell_name(tuple(sorted(fs, key=list(FACTORS).index)))

        main = []
        for f in names:
            diffs = [x - cells[name(factors_of(t) - {f})]
                     for t, x in cells.items() if f in factors_of(t)]
            d_inst = pd.concat(diffs, axis=1).mean(axis=1).groupby(level="instance").mean()
            main.append(effect(d_inst, budget=budget, view="main", name=f))
        output.append(adjust(main))
        interactions = []
        for f, g in itertools.combinations(names, 2):
            contrasts = []
            for tag in cells:
                fs = factors_of(tag)
                if f in fs and g in fs:
                    contrasts.append(cells[tag] - cells[name(fs - {f})]
                                     - cells[name(fs - {g})] + cells[name(fs - {f, g})])
            d_inst = pd.concat(contrasts, axis=1).mean(axis=1).groupby(level="instance").mean()
            interactions.append(effect(d_inst, budget=budget, view="interaction", name=f"{f}x{g}"))
        output.append(adjust(interactions))
        base = d[d.tag == "base"]
        output.append(adjust([paired(x, base, budget=budget, view="cell", name=t)
                              for t, x in d.groupby("tag") if t != "base"]))
    return pd.concat(output, ignore_index=True)


def comparisons() -> None:
    old_tables.TABLES = TABLES
    for name in SETS:
        src = ROOT / "results/jz" / name
        dest = ANALYSIS / name
        dest.mkdir(exist_ok=True)
        runs = load_runs(src, "tmax")
        INPUTS.update(p for p in src.glob("*.csv")
                      if p.name not in {"summary.csv", "per_instance.csv", "checkpoints.csv", "savant_diagnosis.csv"})
        INPUTS.update(src.glob("*.trace.jsonl"))
        ref = read(src / "reference.csv").set_index("instance").ref_cost
        ref.to_csv(dest / "reference.csv")
        pi = per_instance(runs, ref)
        pi.to_csv(dest / "per_instance.csv", index=False)
        s = summary(pi, runs, "savant")
        base = pi[pi.solver == "savant"].set_index("instance").gap_mean
        for k, r in s.iterrows():
            if r.solver == "savant":
                continue
            x = pi[pi.solver == r.solver].set_index("instance").gap_mean
            j = base.index.intersection(x.index)
            diff = x[j] - base[j]
            s.loc[k, "p"] = test(diff)
            s.loc[k, ["ci_low", "ci_high"]] = interval(diff)
        m = s.solver != "savant"
        s.loc[m, "p_holm"] = holm(s.loc[m, "p"].to_numpy())
        s.to_csv(dest / "summary.csv", index=False)
        cp = checkpoints(load_traces(src, "tmax"), ref, "savant")
        cp.to_csv(dest / "checkpoints.csv", index=False)
        if name in {"X", "XL"}:
            caption = f"the 100 {name} instances " + (r"\citep{uchoa2017}" if name == "X" else r"\citep{queiroga2026xl}")
            if name == "XL":
                # LKH-3 fails on 177 of 500 XL runs: the main table leaves it out, the
                # appendix gives the complete one; the Holm family includes it in both
                old_tables.comparison(dest, name, caption, exclude=("lkh",),
                                      note=r" The supplementary LKH-3 result is in Appendix~\ref{app:lkh}; "
                                           r"the Holm family still includes it.")
                old_tables.comparison(dest, "XL:complete", caption, file="XL_complete",
                                      title="Complete comparison, including LKH-3, on")
            else:
                old_tables.comparison(dest, name, caption)
            old_tables.checkpoints(dest, name, caption)
            old_tables.per_instance(dest, name, caption)
        # Coverage is part of the reproducibility material, including excluded traces.
        coverage = cp[cp.stat == "no_sol"].copy()
        usable = {solver: sum(v["trace_ok"] for k, v in load_traces(src, "tmax").items()
                              if k[0] == solver) for solver in runs.solver.unique()}
        coverage["usable_runs"] = coverage.solver.map(usable)
        coverage.to_csv(dest / "coverage.csv", index=False)
    groups = [(ANALYSIS / f"n{n}", f"Uniform, $n={n}$") for n in (100, 500, 1000, 2000)]
    groups.append((ANALYSIS / "XML100_test", "XML100 (optimal reference costs)"))
    old_tables.group_comparison(groups, "uniform", r"Uniform sets and XML100 at $T_{\max}=2.4n$ s.")
    old_tables.group_checkpoints(groups, "uniform", r"Within-run convergence on the uniform sets and XML100.")
    body = []
    for name in ("X", "XL"):
        cp = pd.read_csv(ANALYSIS / name / "coverage.csv")
        total = pd.read_csv(ANALYSIS / name / "summary.csv").set_index("solver").runs
        body += [r"\midrule", rf"\multicolumn{{6}}{{@{{}}l}}{{\emph{{{name}}}}} \\"]
        for r in cp.itertuples(index=False, name=None):
            solver = r[0]
            row = cp[cp.solver == solver].iloc[0]
            m = int(row.usable_runs)
            avail = [f"{100*(m-row[str(f)])/m:.1f}" for f in (.01, .05, .2, 1.0)]
            body.append(old_tables.SOLVER_TEX[solver] + f" & {m}/{int(total[solver])} & "
                        + " & ".join(avail) + r" \\")
    table_file("coverage", r"Trace coverage on X and XL. Usable/total gives usable traces over all runs. "
               r"Each remaining column is the percentage of usable traces with a reported solution "
               r"by that fraction of $T_{\max}$. Averages in checkpoint tables condition on availability.",
               "llrrrr", r"Solver & Usable/total & 1\% & 5\% & 20\% & 100\%", body, wide=True)


def ptex(p: float) -> str:
    return r"$<0.001$" if p < 0.001 else f"{p:.3f}"


def dtex(row) -> str:
    delta = 0.0 if abs(row.delta) < .0005 else row.delta
    return f"${delta:+.3f}" + (r"^{\dagger}$" if row.p_holm < 0.05 else "$")


def table_file(name: str, caption: str, cols: str, head: str, body: list[str], wide=False, small=False) -> None:
    env = "table*" if wide else "table"
    lines = [rf"\begin{{{env}}}[t]", rf"\caption{{{caption}}}\label{{tab:{name}}}"]
    if small:
        lines.append(r"\footnotesize\renewcommand{\arraystretch}{0.88}")
    if body and body[0] == r"\midrule":      # the group header follows the head rule directly
        body = body[1:]
    lines += [r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}" + cols + "@{}}",
              r"\toprule", head + r" \\", r"\midrule", *body,
              r"\bottomrule", r"\end{tabular*}", rf"\end{{{env}}}"]
    (TABLES / f"{name}.tex").write_text("\n".join(lines) + "\n")


def parameter_tables(screen: pd.DataFrame, tm: pd.DataFrame, inter: pd.DataFrame, abl: pd.DataFrame) -> None:
    body = []
    budgets = ["B=10000", "B=100000", "B=1e+06"]
    for group, tags in old_tables.GROUPS.items():
        body += [r"\midrule", rf"\multicolumn{{5}}{{@{{}}l}}{{\textbf{{{group}}}}} \\"]
        for tag in tags:
            rows = [screen[(screen.budget == b) & (screen.tag == tag)].iloc[0] for b in budgets]
            label = (r"$T_0$: upper $10\%$ trimmed" if tag == "t0trim_0.1" else old_tables.SCREEN_TEX[tag])
            body.append(label + " & " + " & ".join(dtex(r) for r in rows)
                        + f" & {rows[-1].time_x:.2f}" + r" \\")
    table_file("screening", r"Component screening: 51 variants of the stock configuration, "
               r"181 instances and three seeds, with $Bn$ total steps. Entries are mean changes in gap "
               r"(percentage points); $\dagger$: instance-level Wilcoxon with Holm correction over "
               r"51 comparisons within each budget. Time is relative process time at $B=10^6$. "
               r"The three $\ddagger$ rows use Sorted-S1 and their own sorted stock control. "
               r"All other rows use Screening-S0; see Section~\ref{sec:parameters:screening}.",
               "lrrrr", r"Variant & $10^4$ & $10^5$ & $10^6$ & Time", body, wide=True, small=True)
    labels = {"pick_v_only": "First vertex uniform", "pick_u_only": "Second vertex uniform",
              "pick_uniform_both": "Both uniform"}
    body = []
    for budget, part in tm.groupby("budget", sort=False):
        f = float(budget.split("=")[1].split("*")[0])
        r = part.iloc[0]
        body += [r"\midrule", rf"\multicolumn{{5}}{{@{{}}l}}{{$f={f:g}$, $N={r.instances}$, "
                 rf"$\bar T={r.time_s:.1f}$ s, reference gap ${r.base_gap:.3f}\%$}} \\"]
        for tag in labels:
            x = part[part.tag == tag]
            if not x.empty:
                r = x.iloc[0]
                body.append(f"{labels[tag]} & {dtex(r)} & {r.steps_x:.2f} & "
                            f"{ptex(r.p_holm)} & [{r.ci_low:+.3f}, {r.ci_high:+.3f}]" + r" \\")
    table_file("timematch", r"Tournament selection versus uniform selection under equal running time. "
               r"The reference uses tournaments of size two on both vertices. Negative $\Delta$ favours "
               r"the uniform variant. $N$: instances, with three seeds each; steps: ratio at fixed "
               r"restart count. Tests use seed-averaged instance differences, Holm-corrected within "
               r"each budget. Intervals are unadjusted 95\% percentile bootstrap intervals for the mean "
               r"difference (9,999 instance resamples).", "lrrrr",
               r"Uniform stage & $\Delta$ & Steps & $p_{\rm Holm}$ & 95\% interval", body, wide=True)
    body = []
    labels = {"restarts": r"Restarts: $1\to4$", "kick": r"Kick period: $100\to50$",
              "knn": r"List size: $20\to40$", "kickmax": r"Kick size: $10\to20$"}
    for view in ("main", "interaction"):
        body += [r"\midrule", rf"\multicolumn{{3}}{{@{{}}l}}{{\emph{{{'Main effects' if view == 'main' else 'Two-factor interactions'}}}}} \\"]
        names = list(labels) if view == "main" else list(inter[inter.view == view].name.unique())
        for name in names:
            label = labels[name] if view == "main" else " $\\times$ ".join(old_tables.FACTOR_SYM[n] for n in name.split("x", 1))
            entries = []
            for b in ("T=0.02*n*2.4s", "T=0.1*n*2.4s"):
                r = inter[(inter.view == view) & (inter.name == name) & (inter.budget == b)]
                entries.append("--" if r.empty else dtex(r.iloc[0]))
            body.append(label + " & " + " & ".join(entries) + r" \\")
    table_file("interactions", r"Factorial effects in percentage points, averaged first over the "
               r"other factors and seeds within each instance. The design uses 92 instances at "
               r"$f=0.02$ and 47 at $f=0.1$. $\dagger$: instance-level Wilcoxon with Holm correction "
               r"separately for main effects and interactions within each budget. List size is not "
               r"varied at $f=0.1$ (--).", "lrr", r"Factor & $f=0.02$ & $f=0.1$", body)
    body = []
    for budget, part in abl.groupby("budget", sort=False):
        r = part.iloc[0]
        f = float(budget.split("=")[1].split("*")[0])
        body += [r"\midrule", rf"\multicolumn{{3}}{{@{{}}l}}{{$f={f:g}$, $N={r.instances}$, reference gap ${r.base_gap:.3f}\%$}} \\"]
        for tag, label in old_tables.ABLATE_TEX.items():
            x = part[part.tag == tag]
            if not x.empty:
                r = x.iloc[0]
                body.append(label + " & " + dtex(r) + " & " + ptex(r.p_holm) + r" \\")
    table_file("ablation", r"Ablation of the retained configuration at equal running time. "
               r"Positive $\Delta$ means the change increases the mean gap. Tests average the three "
               r"seeds per instance and apply Holm across the changes tested at each budget; "
               r"$\dagger$: $p_{\rm Holm}<0.05$. Longer budgets retest only the listed changes. "
               r"Unadjusted bootstrap intervals are recorded with the analysis files.", "lrr",
               r"Change & $\Delta$ & $p_{\rm Holm}$", body)


def diagnostics() -> None:
    diag = read(ROOT / "results/jz/XML100_test/savant_diagnosis.csv").set_index("instance")
    diag["code"] = diag.index.str.split("_").str[1]
    masks = {"All instances": pd.Series(True, index=diag.index),
             "Short routes (3--8 customers)": diag.code.str[3].isin(["1", "2"]),
             "Other route-size classes": ~diag.code.str[3].isin(["1", "2"]),
             "Unit demands": diag.code.str[2] == "1",
             "Reference arc beyond rank 20": diag.opt_arcs_outside_K20 > 0,
             "No reference arc beyond rank 20": diag.opt_arcs_outside_K20 == 0}
    body = [f"{label} & {int(m.sum())} & {int(diag.savant_miss[m].sum())} & "
            f"{int(diag.hgs_miss[m].sum())}" + r" \\" for label, m in masks.items()]
    table_file("xml100", r"XML100 instances on which at least one of ten runs misses the optimum. "
               r"The final two groups refer to the supplied optimal solution, not to all possible optima.",
               "lrrr", r"Group & $N$ & \savant & HGS", body)
    bks = load_bks()
    control = read(ROOT / "results/diag/xml100_knn.csv")
    control["gap"] = 100 * (control.cost - control.instance.map(bks)) / control.instance.map(bks)
    a = control.groupby(["instance", "tag"]).gap.mean().unstack("tag")
    rows = []
    for label, mask in (("all", pd.Series(True, index=a.index)),
                        ("outside", diag.opt_arcs_outside_K20.reindex(a.index) > 0),
                        ("inside", diag.opt_arcs_outside_K20.reindex(a.index) == 0)):
        rows.append(effect((a.k40-a.k20)[mask], group=label,
                           k20=a.k20[mask].mean(), k40=a.k40[mask].mean()))
    adjust(rows).to_csv(ANALYSIS / "xml_control.csv", index=False)
    fisher_rows = []
    for label, mask in (("short_routes", masks["Short routes (3--8 customers)"]),
                        ("outside_arc", masks["Reference arc beyond rank 20"])):
        fisher_rows.append({"group": label, "p": fisher_exact(pd.crosstab(mask, diag.savant_miss)).pvalue})
    adjust(fisher_rows).to_csv(ANALYSIS / "xml_fisher.csv", index=False)
    sensitivity = []
    for n in (100, 500, 1000, 2000):
        src = ROOT / f"results/jz/n{n}"
        r = load_runs(src, "tmax")
        ref = read(src / "reference.csv").set_index("instance").ref_cost
        r["gap"] = 100 * (r.cost-r.instance.map(ref))/r.instance.map(ref)
        a = r[r.seed == 1].pivot(index="instance", columns="solver", values="gap")
        sensitivity.append(effect(a.savant-a.nds, n=n))
    adjust(sensitivity).to_csv(ANALYSIS / "nds_one_seed.csv", index=False)


def convergence_figure(cp: pd.DataFrame, path: Path, plots) -> None:
    """Display sampled checkpoints without shading an unobserved lead interval."""
    import matplotlib.pyplot as plt
    gaps = cp[cp.stat == "gap"].set_index("solver").drop(columns="stat")
    fig, ax = plt.subplots(figsize=(plots.COLUMN_IN, plots.COLUMN_IN * .62))
    for solver, row in gaps.iterrows():
        ax.plot(np.array(row.index, dtype=float)*100, row.to_numpy(dtype=float),
                label=plots.LABEL.get(solver, solver), **plots.style(solver))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(25, color=".45", lw=.6, ls=":")
    ax.set_xlabel("fraction of $T_{max}$ (%)")
    ax.set_ylabel("mean gap (%)")
    ax.grid(which="both", lw=.3, alpha=.4)
    ax.legend(frameon=False, ncols=3, fontsize=7, loc="upper center",
              bbox_to_anchor=(.5,-.42), handlelength=2.2, columnspacing=1.2)
    fig.savefig(path, bbox_inches="tight", pad_inches=.02)
    plt.close(fig)


def size_figure(pi: pd.DataFrame, path: Path, plots) -> None:
    import matplotlib.pyplot as plt
    d = pi.copy()
    d["bucket"] = pd.cut(d.n, [0,200,400,600,1000,2000,5000,10**9],
                         labels=["≤200", "201–400", "401–600", "601–1k", "1–2k", "2–5k", ">5k"])
    means = d.groupby(["solver","bucket"], observed=True).gap_mean.mean().unstack("bucket")
    means = means.dropna(axis=1, how="all")
    fig, ax = plt.subplots(figsize=(plots.COLUMN_IN, plots.COLUMN_IN*.55))
    for solver in plots.LABEL:
        if solver in means.index:
            ax.plot(range(means.shape[1]), means.loc[solver].to_numpy(),
                    label=plots.LABEL[solver], **plots.style(solver))
    ax.set_xticks(range(means.shape[1]), list(means.columns), fontsize=6.5)
    ax.set_xlabel("number of customers")
    ax.set_ylabel("mean gap (%)")
    ax.set_yscale("log")
    ax.grid(which="both",lw=.3,alpha=.4)
    ax.legend(frameon=False,ncols=3,fontsize=7,loc="upper center",
              bbox_to_anchor=(.5,-.46),handlelength=2.2,columnspacing=1.2)
    fig.savefig(path,bbox_inches="tight",pad_inches=.02)
    plt.close(fig)



# --- descriptive inventories (no solver, no statistics) -----------------------

SOURCES = {   # solver -> (root, glob patterns, excluded names); physical lines of C/C++/Java sources
    "SAVANT minimal": ("SAVANT", ("savant.c",), ()),
    "SAVANT research": ("SAVANT", ("cw.c",), ()),
    "SAVANT experimental": ("results/v2_params", ("cw_9f43ed0b.c",), ()),
    "HGS-CVRP": ("external_code/HGS-CVRP-main/Program", ("*.cpp", "*.h"), ()),
    "FILO2": ("external_code/filo2-main", ("*.cpp", "*.hpp", "*.h"), ("Renderer.hpp",)),
    "AILS-II": ("external_code/AILS-CVRP-main/src", ("**/*.java",), ()),
}
FILO2_DIRS = ("base", "instance", "localsearch", "movegen", "opt", "solution")


def source_files(name: str) -> list[Path]:
    root, patterns, excluded = SOURCES[name]
    base = ROOT / root
    if name == "FILO2":     # top-level sources and the library directories, not builds or results
        files = [f for pat in patterns for f in base.glob(pat)]
        files += [f for d in FILO2_DIRS for pat in patterns for f in (base / d).rglob(pat)]
    else:
        files = [f for pat in patterns for f in base.glob(pat)]
    return sorted({f for f in files if f.is_file() and f.name not in excluded})


def source_inventory() -> None:
    """Physical source lines of each implementation (analysis/source_inventory.json). The
    baselines are local copies, not redistributed: without them the existing file is kept."""
    path = ANALYSIS / "source_inventory.json"
    if not all(source_files(name) for name in SOURCES):
        print(f"source_inventory: some sources are missing, {path.name} left unchanged")
        return
    solvers = {}
    for name in SOURCES:
        manifest = [{"path": str(f.relative_to(ROOT)), "lines": len(f.read_bytes().splitlines()),
                     "sha256": hashlib.sha256(f.read_bytes()).hexdigest()} for f in source_files(name)]
        solvers[name] = {"files": len(manifest), "physical_lines": sum(m["lines"] for m in manifest),
                         "manifest": manifest}
    path.write_text(json.dumps({
        "convention": "Physical source lines including blank lines and comments; exclude builds, tests, "
                      "datasets, dependencies and FILO2 optional GUI Renderer.hpp; source files and headers "
                      "in listed manifests. These are local copies, including output-only patches.",
        "solvers": solvers}, indent=2) + "\n")


def xml_reference_inspection() -> None:
    """Why the listed XML100 reference solutions fail the checker: customers repeated or
    missing in the stored routes (analysis/xml_reference_inspection.json)."""
    coll = ROOT / "data/cvrplib/XML100"
    listed = (ROOT / "data/cvrplib/XML100_invalid_ref.txt").read_text().split()
    tuning = set((ROOT / "data/tuning/xml100.txt").read_text().split())
    test_set = set((ROOT / "data/cvrplib/XML100_test.txt").read_text().split())
    refs = []
    for inst in sorted(listed):
        sol = coll / f"{inst}.sol"
        n = int(next(l for l in (coll / f"{inst}.vrp").read_text().splitlines()
                     if l.startswith("DIMENSION")).split(":")[1]) - 1
        seen = [int(c) for l in sol.read_text().splitlines() if l.startswith("Route")
                for c in l.split(":", 1)[1].split()]
        counts = {}
        for c in seen:
            counts[c] = counts.get(c, 0) + 1
        refs.append({"instance": inst, "repeated": {str(c): k for c, k in counts.items() if k > 1},
                     "missing": [c for c in range(1, n + 1) if c not in counts],
                     "sha256": hashlib.sha256(sol.read_bytes()).hexdigest()})
    (ANALYSIS / "xml_reference_inspection.json").write_text(json.dumps({
        "inspection": "Text-only inspection of stored solution customer identifiers; no solver or cost recomputation",
        "collection_instances": len(list(coll.glob("*.vrp"))),
        "listed_references": len(listed),
        "with_repeated_customers": sum(bool(r["repeated"]) for r in refs),
        "with_missing_customers": sum(bool(r["missing"]) for r in refs),
        "tuning_instances": len(tuning),
        "excluded_in_tuning": sorted(tuning.intersection(listed)),
        "excluded_in_test": sorted(test_set.intersection(listed)),
        "references": refs}, indent=2) + "\n")


def main() -> None:
    for d in (ANALYSIS, TABLES, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
    ref = parameter_reference()
    screen = screening(ref)
    tm = simple_study(ref, "timematch_f*.csv", "default")
    inter = factorial(ref)
    abl = simple_study(ref, "ablate_f*.csv", "final")
    for name, frame in (("screening", screen), ("timematch", tm), ("factorial", inter), ("ablation", abl)):
        frame.to_csv(ANALYSIS / f"{name}.csv", index=False)
    parameter_tables(screen, tm, inter, abl)
    comparisons()
    diagnostics()
    from bench import make_figures as plots
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "pdf.fonttype": 42})
    for name in ("X", "XL", "n2000"):
        convergence_figure(read(ANALYSIS / name / "checkpoints.csv"), FIGURES / f"convergence_{name}.pdf", plots)
        if name != "n2000":
            size_figure(read(ANALYSIS / name / "per_instance.csv"), FIGURES / f"gap_by_size_{name}.pdf", plots)
    plots.gap_vs_sets([ANALYSIS / f"n{n}" for n in (100, 500, 1000, 2000)], FIGURES / "gap_by_size_uniform.pdf", plots.COLUMN_IN)
    # main-text XL figures without LKH-3 (the complete ones go to the appendix)
    cp, pi = read(ANALYSIS / "XL" / "checkpoints.csv"), read(ANALYSIS / "XL" / "per_instance.csv")
    convergence_figure(cp[cp.solver != "lkh"].copy(), FIGURES / "convergence_XL_main.pdf", plots)
    size_figure(pi[pi.solver != "lkh"].copy(), FIGURES / "gap_by_size_XL_main.pdf", plots)
    source_inventory()
    xml_reference_inspection()
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(INPUTS) if not p.is_relative_to(PAPER)}
    (ANALYSIS / "input_sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (ANALYSIS / "method.json").write_text(json.dumps({
        "unit": "instance; seed and factorial contrasts averaged before inference",
        "test": "two-sided Wilcoxon; wilcox zeros; auto; differences rounded to 12 decimals",
        "multiplicity": "Holm within each budget and view; 4 one-seed NDS tests; 2 XML Fisher tests; 3 XML control tests",
        "interval": "unadjusted percentile bootstrap; 9999 instance resamples; seed 20261027",
        "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "experimental_source_sha256": "9f43ed0b89aa328db958bf3086c4e3151b75402d1aa7c05c1d5b6a06a28b5156",
        "current_cw_sha256": hashlib.sha256((ROOT / "SAVANT/cw.c").read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    print("\nInstance-level time-matched control:\n", tm[["budget", "tag", "delta", "p_holm"]].to_string(index=False))
    print("\nFactorial main effects:\n", inter[inter.view == "main"][["budget", "name", "delta", "p_holm"]].to_string(index=False))
    print("\nAblation:\n", abl[["budget", "tag", "base_gap", "delta", "p_holm"]].to_string(index=False))


if __name__ == "__main__":
    main()
