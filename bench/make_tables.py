"""LaTeX tables for the manuscript, generated from the result files.

    uv run python -m bench.analyze_sweep     # refresh results/phase2/screen_summary.csv
    uv run python -m bench.make_tables       # -> publication/paper/tables/screening.tex
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bench.common import ROOT
from bench.sweep import CONFIGS

TABLES = ROOT / "publication" / "paper" / "tables"
OUT_DIR = ROOT / "results" / "phase2"
BUDGETS = ["B=10000", "B=100000", "B=1e+06"]
BUDGET_TEX = {"B=10000": r"$10^4$", "B=100000": r"$10^5$", "B=1e+06": r"$10^6$"}

# component -> configuration names, in the order of bench.sweep.CONFIGS
GROUPS = {
    "Construction": ["init_random", "lambda_0.6", "lambda_1.4", "mu_0.3", "cw_2opt"],
    "Temperature": ["taccept_1e-4", "taccept_1e-2", "taccept_1e-1", "decades_0.5", "decades_2",
                    "decades_3", "t0trim_0.1"],
    "Operators": ["ops_relocate_only", "ops_no_swap", "ops_no_2opt", "ops_no_swapstar",
                  "ops_swapstar_x2", "ops_oropt", "ops_oropt_max5", "ops_no_open", "ops_open_0.2",
                  "empty_p_0.05"],
    "Move selection": ["pick_uniform", "pick_uniform_both", "pick_3", "pick_4", "pick_fenwick", "crit_rem", "crit_remnorm",
                       "crit_raw", "pick2_1", "pick2_3", "vrank_2_knn30", "saknn_10", "saknn_40",
                       "saknn_uniform", "reloc_long", "dlb_5"],
    "Ruin \\& recreate": ["kick_off", "kick_50", "kick_500", "kickmax_5", "kickmax_20"],
    "Split": ["split_end", "split_both", "split_every"],
    "Restarts ($R=4$)": ["restarts_4", "restarts_4_param", "restarts_4_race"],
}


def tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%")


def cell(row: pd.Series | None) -> str:
    """Delta gap with a mark when Holm-corrected p < 0.05; bold when significantly better."""
    if row is None:
        return "--"
    d = 0.0 if abs(row.delta) < 0.005 else row.delta      # no "-0.00"
    txt = f"${d:+.2f}$"
    if row.p_holm < 0.05:
        # bold only for an improvement that survives rounding
        txt = rf"$\mathbf{{{d:+.2f}}}^\dagger$" if d < 0 else rf"${d:+.2f}^\dagger$"
    return txt


def screening() -> None:
    s = pd.read_csv(ROOT / "results" / "phase2" / "screen_summary.csv")
    idx = {(r.budget, r.tag): r for r in s.itertuples()}
    missing = set(CONFIGS) - {"default"} - {c for g in GROUPS.values() for c in g}
    assert not missing, f"configurations without a group: {missing}"

    base = {b: idx.get((b, "default")) for b in BUDGETS}
    lines = [
        r"\begin{table*}[t]",
        r"\caption{One-factor-at-a-time screening. Mean paired difference of the gap (percentage points) "
        r"with respect to the default configuration, for $B\cdot n$ annealing steps; negative is better. "
        r"$^\dagger$: Holm-corrected Wilcoxon $p<0.05$ ($181$ instances $\times$ $3$ seeds). "
        r"Time: mean run time relative to the default at the largest budget available.}\label{tab:screening}",
        r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}llrrrr@{}}",
        r"\toprule",
        "Component & Variant (options) & " + " & ".join(BUDGET_TEX[b] for b in BUDGETS) + r" & Time \\",
        r"\midrule",
        "& default gap (\\%) & " + " & ".join(f"{base[b].gap:.2f}" if base[b] is not None else "--"
                                              for b in BUDGETS) + r" & 1.00 \\",
    ]
    for group, names in GROUPS.items():
        lines.append(r"\midrule")
        for k, name in enumerate(names):
            rows = [idx.get((b, name)) for b in BUDGETS]
            last = next((r for r in reversed(rows) if r is not None), None)
            opts = tex_escape(CONFIGS[name]).replace("--", "-{}-")
            time_x = f"{last.time_x:.2f}" if last is not None else "--"
            lines.append(f"{group if k == 0 else ''} & \\texttt{{{opts}}} & "
                         + " & ".join(cell(r) for r in rows) + f" & {time_x} \\\\")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "screening.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {TABLES / 'screening.tex'}")


TIMEMATCH_TEX = {"pick_u_only": r"first vertex uniform", "pick_v_only": r"second vertex uniform",
                 "pick_uniform_both": r"both uniform"}


def timematch() -> None:
    """Time-matched control: the same variants under equal wall budgets."""
    s = pd.read_csv(OUT_DIR / "timematch_summary.csv")
    s["t"] = s.base_time_s.round(1)
    order = sorted(s.t.unique())
    lines = [
        r"\begin{table}[t]",
        r"\caption{Time-matched selection experiment. Each configuration receives the same wall-clock "
        r"budget $T=f\cdot n\cdot 2.4$~s; the tournament is disabled on one or both stages. "
        r"$\Delta$: mean paired difference of the gap (percentage points, negative = better than the "
        r"regret tournament); steps: annealing steps run relative to it. "
        r"$^\dagger$: two-sided Wilcoxon $p<0.05$.}\label{tab:timematch}",
        r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lrrrr@{}}",
        r"\toprule",
        r"$T$ (s) & Variant & $\Delta$ & steps & pairs \\",
        r"\midrule",
    ]
    for t in order:
        part = s[s.t == t]
        base = part.base_gap.iloc[0]
        lines.append(rf"\multicolumn{{5}}{{@{{}}l}}{{$T\approx{t:g}$~s, regret tournament: "
                     rf"gap ${base:.3f}\%$}} \\")
        for r in part.itertuples():
            # the dagger stays inside the cell's own math mode: no nested $...$
            d = f"{r.delta:+.3f}" + (r"^\dagger" if r.p < 0.05 else "")
            lines.append(rf" & {TIMEMATCH_TEX.get(r.tag, r.tag)} & ${d}$ & "
                         rf"${r.steps_x:.2f}\times$ & {r.pairs} \\")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "timematch.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {TABLES / 'timematch.tex'}")


FACTOR_TEX = {"restarts": r"$R=4$ restarts", "kick": r"kick period $50$",
              "knn": r"candidate lists $K=40$", "kickmax": r"kick size $k_{\max}=20$"}


def interactions() -> None:
    """Main effects and two-factor interactions of the retained factors, at equal time."""
    path = OUT_DIR / "interact_summary.csv"
    if not path.exists():
        print(f"skipped interactions: {path} not written yet")
        return
    s = pd.read_csv(path)
    budgets = sorted(s.budget.unique(), key=lambda b: float(b.split("=")[1].split("*")[0]))
    head = " & ".join(rf"$f={b.split('=')[1].split('*')[0]}$" for b in budgets)

    def row(view: str, name: str, label: str) -> str:
        cells = []
        for b in budgets:
            r = s[(s.budget == b) & (s.view == view) & (s.name == name)]
            if r.empty:
                cells.append("--")
                continue
            d = r.delta.iloc[0]
            sig = "p_holm" in r and not pd.isna(r.p_holm.iloc[0]) and r.p_holm.iloc[0] < 0.05
            cells.append(f"${d:+.3f}" + (r"^\dagger$" if sig else "$"))
        return f"{label} & " + " & ".join(cells) + r" \\"

    lines = [
        r"\begin{table}[t]",
        r"\caption{Factorial study of the retained factors at equal running time "
        r"($T=f\cdot n\cdot 2.4$~s). Main effect: mean paired difference of the gap "
        r"(percentage points) between the cells where the factor is on and the matching "
        r"cells where it is off; negative is better. An interaction is the change of one "
        r"main effect when the other factor is on; values close to zero mean the factors "
        r"are additive. $^\dagger$: Wilcoxon $p<0.05$.}\label{tab:interactions}",
        r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}l" + "r" * len(budgets) + r"@{}}",
        r"\toprule",
        f"Factor & {head} \\\\",
        r"\midrule",
        r"\multicolumn{" + str(len(budgets) + 1) + r"}{@{}l}{\emph{Main effects}} \\",
    ]
    for f in [f for f in FACTOR_TEX if ((s.view == "main") & (s.name == f)).any()]:
        lines.append(row("main", f, FACTOR_TEX[f]))
    lines.append(r"\multicolumn{" + str(len(budgets) + 1) + r"}{@{}l}{\emph{Interactions}} \\")
    for name in sorted(s[s.view == "interaction"].name.unique()):
        a, b = name.split("x", 1)
        lines.append(row("interaction", name, f"{a} $\\times$ {b}"))
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "interactions.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {TABLES / 'interactions.tex'}")


ABLATE_TEX = {  # in table order
    "stock": r"stock configuration (Table~\ref{tab:screening})",
    "tournament": r"regret tournament instead of uniform selection",
    "no_restarts": r"single start ($R=1$)",
    "kick_period_100": r"ruin-and-recreate every $100$ steps",
    "kick_max_10": r"kick size $k_{\max}=10$",
    "no_kick": r"no ruin-and-recreate",
    "no_swapstar": r"no \swapstar",
    "no_2opt": r"no 2-opt",
    "no_candidate_lists": r"no candidate lists",
    "random_start": r"random initial solution",
}


def ablation() -> None:
    """What each component of the retained configuration is worth, at equal time."""
    path = OUT_DIR / "ablate_summary.csv"
    if not path.exists():
        print(f"skipped ablation: {path} not written yet")
        return
    s = pd.read_csv(path)
    budgets = sorted(s.budget.unique(), key=lambda b: float(b.split("=")[1].split("*")[0]))
    head = " & ".join(rf"$f={b.split('=')[1].split('*')[0]}$" for b in budgets)
    base = s[s.tag == "final"]
    gaps = ", ".join(rf"${base[base.budget == b].base_gap.iloc[0]:.3f}\%$ at $f={b.split('=')[1].split('*')[0]}$"
                     for b in budgets if not base[base.budget == b].empty)

    lines = [
        # full width: the row labels are sentences, and at \columnwidth (238 pt)
        # the alignment overfills by ~69 pt into the neighbouring column
        r"\begin{table*}[t]",
        r"\caption{Ablation of the retained configuration at equal running time "
        r"($T=f\cdot n\cdot 2.4$~s). Each row changes one element of that configuration; "
        rf"a positive value is a degradation, i.e. what the element is worth. Reference gap: {gaps}. "
        r"$^\dagger$: Holm-corrected Wilcoxon $p<0.05$.}\label{tab:ablation}",
        r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}l" + "r" * len(budgets) + r"@{}}",
        r"\toprule",
        f"Change & {head} \\\\",
        r"\midrule",
    ]
    for tag, label in ABLATE_TEX.items():
        cells = []
        for b in budgets:
            r = s[(s.budget == b) & (s.tag == tag)]
            if r.empty:
                cells.append("--")
                continue
            d = r.delta.iloc[0]
            cells.append(f"${d:+.3f}" + (r"^\dagger$" if r.p_holm.iloc[0] < 0.05 else "$"))
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "ablation.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {TABLES / 'ablation.tex'}")


# ---------------------------------------------------------------- comparison
SOLVER_TEX = {"savant": r"\savant", "hgs": "HGS-CVRP", "filo": "FILO", "filo2": "FILO2",
              "ails": "AILS-II", "lkh": "LKH-3", "ortools": "OR-Tools", "nds": "NDS"}


def comparison(set_dir: Path, name: str, caption_set: str) -> None:
    """Per-solver quality on one test set, with the paired test against SAVANT."""
    path = set_dir / "summary.csv"
    if not path.exists():
        print(f"skipped comparison {name}: {path} not written yet")
        return
    s = pd.read_csv(path)
    lines = [
        r"\begin{table*}[t]",
        rf"\caption{{Comparison on {caption_set}. Gaps in \% to the reference cost, averaged over "
        r"instances; \emph{mean} averages the runs of an instance, \emph{best} takes the best of them. "
        r"\emph{Inst.}: instances on which the solver returned at least one feasible solution, "
        r"over which its gaps are averaged. \emph{Fail}: runs whose solution was rejected by the "
        r"independent checker and excluded from the gaps. \emph{Time}: mean wall-clock time of the "
        r"whole process, over every run. "
        r"$\Delta$: mean paired difference against \savant\ on the instances both solve (positive "
        r"= worse than \savant), with the Holm-corrected Wilcoxon $p$-value and the number of "
        r"instances on which the solver beats / loses to \savant."
        rf"}}\label{{tab:cmp:{name}}}",
        r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lrrrrrrrr@{}}",
        r"\toprule",
        r"Solver & Inst. & Mean & Best & Fail & Time (s) & $\Delta$ & W/L & $p$ \\",
        r"\midrule",
    ]
    for r in s.itertuples():
        label = SOLVER_TEX.get(r.solver, tex_escape(r.solver))
        if r.solver == s.solver.iloc[0]:          # the base: no test against itself
            lines.append(f"{label} & {int(r.instances)} & {r.gap_mean:.3f} & {r.gap_best:.3f} & "
                         f"{int(r.fails)} & {r.time_s:.0f} & -- & -- & -- \\\\")
            continue
        p_holm = r.p_holm
        ptex = r"$<$0.001" if p_holm < 0.001 else f"{p_holm:.3f}"
        delta = f"${r.delta:+.3f}" + (r"^\dagger$" if p_holm < 0.05 else "$")
        lines.append(f"{label} & {int(r.instances)} & {r.gap_mean:.3f} & {r.gap_best:.3f} & "
                     f"{int(r.fails)} & {r.time_s:.0f} & {delta} & "
                     f"{int(r.wins)}/{int(r.losses)} & {ptex} \\\\")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / f"comparison_{name}.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {TABLES / f'comparison_{name}.tex'}")


def checkpoints(set_dir: Path, name: str, caption_set: str) -> None:
    """Gap of the best solution found so far at fractions of T_max (Vidal 2022, Fig. 4-5)."""
    path = set_dir / "checkpoints.csv"
    if not path.exists():
        print(f"skipped checkpoints {name}: {path} not written yet")
        return
    d = pd.read_csv(path)
    gaps = d[d.stat == "gap"].drop(columns="stat").set_index("solver")
    fracs = [c for c in gaps.columns]
    head = " & ".join(f"{100 * float(c):g}" for c in fracs)
    lines = [
        r"\begin{table*}[t]",
        rf"\caption{{Convergence on {caption_set}: mean gap (\%) of the best solution found so far at each "
        r"percentage of $T_{\max}$, averaged over instances. These are the values reached "
        r"\emph{within} a run of length $T_{\max}$; solvers that schedule their search around "
        r"$T_{\max}$ would do better if given that shorter budget outright."
        rf"}}\label{{tab:conv:{name}}}",
        r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}l" + "r" * len(fracs) + r"@{}}",
        r"\toprule",
        rf"Solver & \multicolumn{{{len(fracs)}}}{{c}}{{\% of $T_{{\max}}$}} \\",
        rf"\cmidrule(l){{2-{len(fracs) + 1}}}",
        f" & {head} \\\\",
        r"\midrule",
    ]
    for solver, row in gaps.iterrows():
        cells = " & ".join("--" if pd.isna(v) else f"{v:.2f}" for v in row)
        lines.append(f"{SOLVER_TEX.get(solver, tex_escape(solver))} & {cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / f"checkpoints_{name}.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {TABLES / f'checkpoints_{name}.tex'}")


def _ptex(p: float) -> str:
    return r"$<$0.001" if p < 0.001 else f"{p:.3f}"


def group_comparison(sets: list[tuple[Path, str]], name: str, caption: str) -> None:
    """One table for several small campaigns (the uniform sets and XML100): a block
    of rows per set, the same columns as `comparison` minus the always-zero Fail."""
    lines = [
        r"\begin{table*}[t]",
        rf"\caption{{{caption} Gaps in \% to the reference cost, averaged over instances "
        r"($100$ per set); \emph{mean} averages the runs of an instance, \emph{best} takes the best "
        r"of them (\enspace--\enspace for NDS, run once per instance). \emph{Time}: mean wall-clock "
        r"time of the whole process. $\Delta$: mean paired difference against \savant\ (positive = "
        r"worse than \savant), with the number of instances on which the solver beats / loses to "
        r"\savant\ and the Holm-corrected Wilcoxon $p$-value; $^\dagger$: $p<0.05$. Every run of "
        r"every solver returned a feasible solution."
        rf"}}\label{{tab:cmp:{name}}}",
        r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lrrrrrr@{}}",
        r"\toprule",
        r"Solver & Mean & Best & Time (s) & $\Delta$ & W/L & $p$ \\",
    ]
    for set_dir, label in sets:
        s = pd.read_csv(set_dir / "summary.csv")
        assert (s.fails == 0).all(), f"{set_dir}: failed runs, use the full table"
        lines += [r"\midrule", rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{{label}}}}} \\"]
        for r in s.itertuples():
            one_run = r.runs == r.instances
            best = "--" if one_run else f"{r.gap_best:.3f}"
            head = f"{SOLVER_TEX.get(r.solver, tex_escape(r.solver))} & {r.gap_mean:.3f} & {best} & {r.time_s:.0f}"
            if r.solver == s.solver.iloc[0]:
                lines.append(head + r" & -- & -- & -- \\")
                continue
            delta = f"${r.delta:+.3f}" + (r"^\dagger$" if r.p_holm < 0.05 else "$")
            lines.append(head + f" & {delta} & {int(r.wins)}/{int(r.losses)} & {_ptex(r.p_holm)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / f"comparison_{name}.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {TABLES / f'comparison_{name}.tex'}")


def group_checkpoints(sets: list[tuple[Path, str]], name: str, caption: str) -> None:
    """Convergence of several small campaigns in one table, a block of rows per set."""
    blocks = []
    for set_dir, label in sets:
        d = pd.read_csv(set_dir / "checkpoints.csv")
        blocks.append((label, d[d.stat == "gap"].drop(columns="stat").set_index("solver")))
    fracs = list(blocks[0][1].columns)
    head = " & ".join(f"{100 * float(c):g}" for c in fracs)
    lines = [
        r"\begin{table*}[t]",
        rf"\caption{{{caption} Mean gap (\%) of the best solution found so far at each percentage "
        r"of $T_{\max}$, averaged over instances. These are the values reached \emph{within} a run "
        r"of length $T_{\max}$; solvers that schedule their search around $T_{\max}$ would do "
        r"better if given that shorter budget outright."
        rf"}}\label{{tab:conv:{name}}}",
        r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}l" + "r" * len(fracs) + r"@{}}",
        r"\toprule",
        rf"Solver & \multicolumn{{{len(fracs)}}}{{c}}{{\% of $T_{{\max}}$}} \\",
        rf"\cmidrule(l){{2-{len(fracs) + 1}}}",
        f" & {head} \\\\",
    ]
    for label, gaps in blocks:
        lines += [r"\midrule", rf"\multicolumn{{{len(fracs) + 1}}}{{@{{}}l}}{{\emph{{{label}}}}} \\"]
        for solver, row in gaps.iterrows():
            cells = " & ".join("--" if pd.isna(v) else f"{v:.2f}" for v in row)
            lines.append(f"{SOLVER_TEX.get(solver, tex_escape(solver))} & {cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / f"checkpoints_{name}.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {TABLES / f'checkpoints_{name}.tex'}")


def per_instance(set_dir: Path, name: str, caption_set: str, rows_per_page: int = 34) -> None:
    """Appendix: the mean gap of every solver on every instance of a set.

    A hundred instances do not fit on one page, and `longtable` does not work in
    a two-column class, so the rows are split over several full-width floats.
    The reference cost is included so the numbers can be recomputed from the
    per-run costs of the result files.
    """
    path = set_dir / "per_instance.csv"
    ref_path = set_dir / "reference.csv"
    if not path.exists():
        print(f"skipped per-instance {name}: {path} not written yet")
        return
    pi = pd.read_csv(path)
    wide = pi.pivot(index="instance", columns="solver", values="gap_mean")
    order = [s for s in SOLVER_TEX if s in wide.columns] + \
            sorted(set(wide.columns) - set(SOLVER_TEX))
    wide = wide[order]
    meta = pi.groupby("instance").n.first()
    ref = pd.read_csv(ref_path).set_index("instance").ref_cost
    wide = wide.assign(n=meta, ref=ref).sort_values("n")

    head = " & ".join(SOLVER_TEX.get(s, tex_escape(s)) for s in order)
    chunks = [wide.iloc[i:i + rows_per_page] for i in range(0, len(wide), rows_per_page)]
    lines = []
    for k, chunk in enumerate(chunks, 1):
        part = f" ({k} of {len(chunks)})" if len(chunks) > 1 else ""
        lines += [
            r"\begin{table*}[p]",
            rf"\caption{{Mean gap (\%) of each solver on each of {caption_set}{part}. "
            r"$n$ is the number of customers and \emph{Ref.} the reference cost the gaps are "
            r"measured against. The best value of a row is in bold; \enspace--\enspace marks a "
            r"solver that returned no feasible solution for that instance."
            rf"}}\label{{tab:inst:{name}:{k}}}",
            r"\footnotesize",
            r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lrr" + "r" * len(order) + r"@{}}",
            r"\toprule",
            rf"Instance & $n$ & Ref. & {head} \\",
            r"\midrule",
        ]
        for inst, row in chunk.iterrows():
            vals = row[order]
            best = vals.min()
            cells = []
            for v in vals:
                if pd.isna(v):
                    cells.append("--")
                elif abs(v - best) < 1e-12:
                    cells.append(rf"$\mathbf{{{v:.3f}}}$")
                else:
                    cells.append(f"{v:.3f}")
            lines.append(f"{tex_escape(inst)} & {int(row.n)} & {row.ref:.0f} & "
                         + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}", ""]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / f"per_instance_{name}.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {TABLES / f'per_instance_{name}.tex'} ({len(wide)} instances, {len(chunks)} floats)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compare", type=Path, help="directory analysed by bench.analyze_compare")
    ap.add_argument("--name", help="suffix of the .tex files, e.g. X")
    ap.add_argument("--caption", default=None, help="how the set is named in the caption")
    ap.add_argument("--group", nargs="+", metavar="DIR=LABEL",
                    help="several analysed directories in one table (with --name and --caption)")
    a = ap.parse_args()
    if a.group:
        if not (a.name and a.caption):
            ap.error("--group needs --name and --caption")
        sets = [(Path(g.split("=", 1)[0]), g.split("=", 1)[1]) for g in a.group]
        group_comparison(sets, a.name, a.caption)
        group_checkpoints(sets, a.name, a.caption)
    elif a.compare:
        if not a.name:
            ap.error("--compare needs --name")
        cap = a.caption or f"the {a.name} instances"
        comparison(a.compare, a.name, cap)
        checkpoints(a.compare, a.name, cap)
        per_instance(a.compare, a.name, cap)
    else:
        screening()
        timematch()
        interactions()
        ablation()
