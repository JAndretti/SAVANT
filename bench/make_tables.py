"""LaTeX tables of the manuscript: the functions and labels that
bench.make_paper calls, with its own statistics. Not a command.

comparison, checkpoints and per_instance read a directory with summary.csv,
checkpoints.csv and per_instance.csv (as bench.analyze_compare or bench.make_paper write
them); group_comparison and group_checkpoints put several such directories in one
table. They write into TABLES, which bench.make_paper sets.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bench.common import PAPER

TABLES = PAPER / "tables"

# screening: component -> configuration names, in the order of bench.sweep.CONFIGS
GROUPS = {
    "Construction": ["init_random", "lambda_0.6", "lambda_1.4", "mu_0.3", "cw_2opt"],
    "Temperature": ["taccept_1e-4", "taccept_1e-2", "taccept_1e-1", "decades_0.5", "decades_2",
                    "decades_3", "t0trim_0.1"],
    "Operators": ["ops_relocate_only", "ops_no_swap", "ops_no_2opt", "ops_no_swapstar",
                  "ops_swapstar_x2", "ops_oropt", "ops_oropt_max5", "ops_no_open", "ops_open_0.2",
                  "empty_p_0.05"],
    "Candidate lists": ["saknn_uniform", "saknn_10", "saknn_40", "saknn_30", "vrank_2", "vrank_2_knn30"],
    "Move selection": ["pick_uniform", "pick2_1", "pick_uniform_both", "pick_3", "pick_4", "pick2_3",
                       "pick_fenwick", "crit_rem", "crit_remnorm", "crit_raw", "reloc_long", "dlb_5"],
    "Ruin \\& recreate": ["kick_off", "kick_50", "kick_500", "kickmax_5", "kickmax_20"],
    "Split": ["split_end", "split_both", "split_every"],
    "Restarts": ["restarts_4", "restarts_4_param", "restarts_4_race"],
}


SCREEN_TEX = {
    "init_random": r"random initial solution",
    "lambda_0.6": r"$\lambda=0.6$",
    "lambda_1.4": r"$\lambda=1.4$",
    "mu_0.3": r"$\mu=0.3$",
    "cw_2opt": r"2-opt on each savings route",
    "taccept_1e-4": r"$\chi_0=10^{-4}$",
    "taccept_1e-2": r"$\chi_0=10^{-2}$",
    "taccept_1e-1": r"$\chi_0=10^{-1}$",
    "decades_0.5": r"$D=0.5$",
    "decades_2": r"$D=2$",
    "decades_3": r"$D=3$",
    "t0trim_0.1": r"$T_0$ from a $10\%$-trimmed mean",
    "ops_relocate_only": r"relocate only, $w=(1,0,0,0,0,0)$",
    "ops_no_swap": r"no swap",
    "ops_no_2opt": r"no 2-opt/2-opt*",
    "ops_no_swapstar": r"no \swapstar",
    "ops_swapstar_x2": r"\swapstar weight $2$",
    "ops_oropt": r"with or-opt, $L_{\max}=3$",
    "ops_oropt_max5": r"with or-opt, $L_{\max}=5$",
    "ops_no_open": r"no open",
    "ops_open_0.2": r"open weight $0.2$",
    "empty_p_0.05": r"relocate into an empty route w.p.\ $0.05$",
    "saknn_uniform": r"no candidate lists ($v$ uniform in $V$)",
    "saknn_10": r"$K=10$",
    "saknn_40": r"$K=40$",
    "saknn_30": r"$K=30^\ddagger$",
    "vrank_2": r"bias towards nearer ranks ($K=20$)$^\ddagger$",
    "vrank_2_knn30": r"$K=30$, bias towards nearer ranks$^\ddagger$",
    "pick_uniform": r"$T_u=1$ (first vertex uniform)",
    "pick2_1": r"$T_v=1$ (second vertex uniform)",
    "pick_uniform_both": r"$T_u=T_v=1$ (both uniform)",
    "pick_3": r"$T_u=3$",
    "pick_4": r"$T_u=4$",
    "pick2_3": r"$T_v=3$",
    "pick_fenwick": r"$u$ drawn $\propto r(u)$ (Fenwick tree)",
    "crit_rem": r"regret $=$ removal gain",
    "crit_remnorm": r"regret $=$ removal gain$/\ell(u)$",
    "crit_raw": r"regret $=c_{pu}+c_{us}$",
    "reloc_long": r"relocate breaks the longer edge at $v$",
    "dlb_5": r"don't-look bits (threshold $5$)",
    "kick_off": r"no ruin-and-recreate",
    "kick_50": r"$P=50$",
    "kick_500": r"$P=500$",
    "kickmax_5": r"$k_{\max}=5$",
    "kickmax_20": r"$k_{\max}=20$",
    "split_end": r"Split the final solution",
    "split_both": r"Split after construction and at the end",
    "split_every": r"Split every $10^5$ steps and at the end",
    "restarts_4": r"$R=4$",
    "restarts_4_param": r"$R=4$, random $(\lambda,\mu)$ per restart",
    "restarts_4_race": r"$R=4$ with racing",
}


def tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%")


FACTOR_SYM = {"restarts": "$R$", "kick": "$P$", "knn": "$K$", "kickmax": r"$k_{\max}$"}


ABLATE_TEX = {  # in table order
    "stock": r"stock configuration",
    "tournament": r"regret tournament ($T_u=T_v=2$)",
    "no_restarts": r"single start ($R=1$)",
    "kick_period_100": r"ruin \& recreate period $P=100$",
    "kick_max_10": r"ruin \& recreate size $k_{\max}=10$",
    "no_kick": r"no ruin-and-recreate",
    "no_swapstar": r"no \swapstar",
    "no_2opt": r"no 2-opt/2-opt*",
    "no_candidate_lists": r"no candidate lists",
    "random_start": r"random initial solution",
}


# ---------------------------------------------------------------- comparison
SOLVER_TEX = {"savant": r"\savant", "hgs": "HGS-CVRP", "filo": "FILO", "filo2": "FILO2",
              "ails": "AILS-II", "lkh": "LKH-3", "ortools": "OR-Tools", "nds": "NDS"}


def comparison(set_dir: Path, name: str, caption_set: str, *, exclude: tuple[str, ...] = (),
               title: str = "Comparison on", note: str = "", file: str | None = None) -> None:
    """Per-solver quality on one test set, with the paired test against SAVANT.

    exclude drops solvers from the rows only: their tests stay in the Holm family
    of summary.csv. note is appended to the caption; file defaults to name."""
    path = set_dir / "summary.csv"
    if not path.exists():
        print(f"skipped comparison {name}: {path} not written yet")
        return
    s = pd.read_csv(path)
    lines = [
        r"\begin{table*}[t]",
        rf"\caption{{{title} {caption_set}. Gaps in \% to the reference cost, averaged over "
        r"instances; \emph{mean} averages the runs of an instance, \emph{best} takes the best of them. "
        r"\emph{Inst.}: instances on which the solver returned at least one feasible solution, "
        r"over which its gaps are averaged. \emph{Fail}: runs whose solution was rejected by the "
        r"independent checker and excluded from the gaps. \emph{Time}: mean wall-clock time of the "
        r"whole process, over every run. "
        r"$\Delta$: mean paired difference against \savant\ on the instances both solve (positive "
        r"= worse than \savant), with a two-sided Wilcoxon test on paired instance means, Holm-corrected within each collection. \emph{\savant\ better / "
        r"worse}: percentage of these instances on which the mean gap of \savant\ is lower / higher "
        r"than that of the solver; the two need not sum to $100$ when there are ties."
        rf"{note}}}\label{{tab:cmp:{name}}}",
        r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lrrrrrrrr@{}}",
        r"\toprule",
        r"Solver & Inst. & Mean & Best & Fail & Time (s) & $\Delta$ & \savant\ better / worse (\%) & $p$ \\",
        r"\midrule",
    ]
    for r in s.itertuples():
        if r.solver in exclude:
            continue
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
                     f"{_wl(r)} & {ptex} \\\\")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / f"comparison_{file or name}.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


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
        r"\emph{within} a run of length $T_{\max}$; they do not predict results when schedules are "
        r"recalibrated to shorter allowances."
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


def _wl(r) -> str:
    """Instances on which SAVANT is better / worse, in % of the instances both solve.
    `wins` counts the instances on which the *other* solver is better."""
    return f"{100 * r.losses / r.pairs:.0f} / {100 * r.wins / r.pairs:.0f}"


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
        r"worse than \savant), with a two-sided Wilcoxon test on paired instance means, Holm-corrected within each collection; $^\dagger$: $p<0.05$. "
        r"\emph{\savant\ better / worse}: percentage of the instances on which the mean gap of "
        r"\savant\ is lower / higher than that of the solver (the rest are ties). Every run of "
        r"every solver returned a feasible solution."
        rf"}}\label{{tab:cmp:{name}}}",
        r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lrrrrrr@{}}",
        r"\toprule",
        r"Solver & Mean & Best & Time (s) & $\Delta$ & \savant\ better / worse (\%) & $p$ \\",
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
            lines.append(head + f" & {delta} & {_wl(r)} & {_ptex(r.p_holm)} \\\\")
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
        r"of length $T_{\max}$; they do not predict results when schedules are recalibrated to "
        r"shorter allowances."
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
