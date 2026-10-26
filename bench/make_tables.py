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
    "Candidate lists": ["saknn_uniform", "saknn_10", "saknn_40", "saknn_30", "vrank_2", "vrank_2_knn30"],
    "Move selection": ["pick_uniform", "pick2_1", "pick_uniform_both", "pick_3", "pick_4", "pick2_3",
                       "pick_fenwick", "crit_rem", "crit_remnorm", "crit_raw", "reloc_long", "dlb_5"],
    "Ruin \\& recreate": ["kick_off", "kick_50", "kick_500", "kickmax_5", "kickmax_20"],
    "Split": ["split_end", "split_both", "split_every"],
    "Restarts": ["restarts_4", "restarts_4_param", "restarts_4_race"],
}

# what the reference ("stock") configuration does for each component, printed
# under the component's name so that every row reads as a change from it
GROUP_LEGEND = {
    "Construction": r"stock: parallel savings, $\lambda=1$, $\mu=0$ (Eq.~\ref{eq:saving})",
    "Temperature": r"stock: $\chi_0=10^{-3}$, $D=1$ decade, $T_0$ from the plain mean of $\Delta^+$",
    "Operators": r"weights $w$ of relocate, swap, 2-opt, or-opt, \swapstar, open; "
                 r"stock: $(1,1,1,0,1,0.05)$",
    "Candidate lists": r"$v$ drawn among the $K$ nearest customers of $u$; stock: $K=20$, "
                       r"uniform rank",
    "Move selection": r"tournament sizes $T_u,T_v$ on the regret $r$ (Eq.~\ref{eq:regret}); "
                      r"stock: $T_u=T_v=2$",
    "Ruin \\& recreate": r"one move every $P$ steps, removing up to $k_{\max}$ customers; "
                         r"stock: $P=100$, $k_{\max}=10$",
    "Split": r"stock: no Split",
    "Restarts": r"$R$ chains sharing the budget; stock: $R=1$",
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
    "ops_no_2opt": r"no 2-opt",
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

    missing = {c for g in GROUPS.values() for c in g} - set(SCREEN_TEX)
    assert not missing, f"configurations without a label: {missing}"

    base = {b: idx.get((b, "default")) for b in BUDGETS}
    # the rows marked \ddagger were re-run once the kNN lists were sorted
    chk = pd.read_csv(ROOT / "results" / "p1_sorted_knn" / "sort_check.csv").set_index("budget")
    sort_d = ", ".join(f"${0.0 if abs(chk.delta[b]) < 0.005 else chk.delta[b]:+.2f}$" for b in BUDGETS)
    sort_p = min(chk.p)
    lines = [
        r"\begin{table*}[t]",
        r"\caption{One-factor-at-a-time screening at a fixed number of annealing steps. Each row "
        r"changes one parameter of the stock configuration (Table~\ref{tab:params}), whose value "
        r"is given in italics after the name of each component. Columns $10^4$, $10^5$, $10^6$: "
        r"runs of $B\cdot n$ steps. Cells: mean paired difference of the gap with the stock "
        r"configuration, in percentage points, over $181$ tuning instances $\times$ $3$ seeds; "
        r"negative is better, bold marks a significant improvement, $^\dagger$: Holm-corrected "
        r"Wilcoxon $p<0.05$. Time: run time relative to the stock configuration at the largest "
        r"budget, i.e.\ the relative cost of a step. "
        r"$^\ddagger$: candidate lists sorted by distance, against the stock configuration "
        r"run likewise; sorting alone changes its gap by "
        + sort_d + rf" ($p\geq{sort_p:.2f}$)."
        r"}\label{tab:screening}",
        r"\footnotesize\renewcommand{\arraystretch}{0.88}",   # 50 variants and 8 component headers: one page at normal size is too short
        r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lrrrr@{}}",
        r"\toprule",
        "Variant & " + " & ".join(BUDGET_TEX[b] for b in BUDGETS) + r" & Time \\",
        r"\midrule",
        r"stock configuration: gap (\%) & "
        + " & ".join(f"{base[b].gap:.2f}" if base[b] is not None else "--" for b in BUDGETS)
        + r" & 1.00 \\",
    ]
    for group, names in GROUPS.items():
        lines += [r"\midrule",
                  rf"\multicolumn{{5}}{{@{{}}l}}{{\textbf{{{group}}} --- \emph{{{GROUP_LEGEND[group]}}}}} \\"]
        for name in names:
            rows = [idx.get((b, name)) for b in BUDGETS]
            last = next((r for r in reversed(rows) if r is not None), None)
            time_x = f"{last.time_x:.2f}" if last is not None else "--"
            lines.append(rf"\quad {SCREEN_TEX[name]} & "
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
        r"\caption{Regret tournament against uniform selection at equal running time. Rows are "
        r"grouped by budget $T=f\cdot T_{\max}$, from the shortest to the longest; the header "
        r"of each group gives $T$ averaged over the instances and the gap of the regret "
        r"tournament ($T_u=T_v=2$), the reference of the group. Each row replaces the "
        r"tournament by uniform sampling on the first stage, the second, or both. "
        r"$\Delta$: mean paired difference of the gap (percentage points); negative means "
        r"uniform selection is better. Steps: annealing steps performed in the same time, "
        r"relative to the tournament. Pairs: (instance, seed) pairs compared. "
        r"$^\dagger$: two-sided Wilcoxon $p<0.05$.}\label{tab:timematch}",
        r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lrrrr@{}}",
        r"\toprule",
        r" & Variant & $\Delta$ & Steps & Pairs \\",
        r"\midrule",
    ]
    for t in order:
        part = s[s.t == t]
        base = part.base_gap.iloc[0]
        f = part.budget.iloc[0].split("=")[1].split("*")[0]
        lines.append(rf"\multicolumn{{5}}{{@{{}}l}}{{$f={f}$ ($T\approx{t:g}$~s), regret tournament: "
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


FACTOR_TEX = {"restarts": r"restarts: $R=4$ (vs.\ $1$)",
              "kick": r"kick period: $P=50$ (vs.\ $100$)",
              "knn": r"candidate lists: $K=40$ (vs.\ $20$)",
              "kickmax": r"kick size: $k_{\max}=20$ (vs.\ $10$)"}
FACTOR_SYM = {"restarts": "$R$", "kick": "$P$", "knn": "$K$", "kickmax": r"$k_{\max}$"}


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
        r"\caption{Full factorial study of the factors retained by the screening, at equal "
        r"running time $T=f\cdot T_{\max}$. Each factor has two levels, its stock value (off) "
        r"and the value suggested by the screening (on), and every combination of levels is "
        r"run. Main effect: mean paired difference of the gap (percentage points) between the "
        r"runs where the factor is on and the matching runs where it is off; negative is "
        r"better. Interaction $A\times B$: how much the main effect of $A$ changes when $B$ is "
        r"on; a value close to zero means that the two factors add up, a negative value that "
        r"they reinforce each other. The candidate-list size was dropped at $f=0.1$ "
        r"(\enspace--\enspace). $^\dagger$: Wilcoxon $p<0.05$.}\label{tab:interactions}",
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
        lines.append(row("interaction", name, f"{FACTOR_SYM[a]} $\\times$ {FACTOR_SYM[b]}"))
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "interactions.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {TABLES / 'interactions.tex'}")


ABLATE_TEX = {  # in table order
    "stock": r"stock configuration",
    "tournament": r"regret tournament ($T_u=T_v=2$)",
    "no_restarts": r"single start ($R=1$)",
    "kick_period_100": r"ruin \& recreate period $P=100$",
    "kick_max_10": r"ruin \& recreate size $k_{\max}=10$",
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
    s["f"] = s.budget.map(lambda b: float(b.split("=")[1].split("*")[0]))
    s = s[s.tag != "final"]
    budgets = sorted(s.f.unique())
    full, longer = budgets[0], budgets[1:]

    def row(r) -> str:
        d = f"${r.delta:+.3f}" + (r"^\dagger$" if r.p_holm < 0.05 else "$")
        decided = r.wins + r.losses        # pairs that are not tied
        return (f"{ABLATE_TEX[r.tag]} & {d} & "
                f"{100 * r.losses / decided:.0f} / {100 * r.wins / decided:.0f} \\\\")

    def ordered(part: pd.DataFrame) -> list:
        return sorted(part.itertuples(), key=lambda r: list(ABLATE_TEX).index(r.tag))

    lines = [
        r"\begin{table}[t]",
        r"\caption{Ablation of the retained configuration (uniform selection, $R=4$, $P=50$, "
        r"$k_{\max}=20$) at equal running time $T=f\cdot T_{\max}$. Each row changes one element "
        r"of it. $\Delta$: mean paired difference of the gap, in percentage points; positive means "
        r"the change degrades the solution, i.e.\ the element is worth $\Delta$. $^\dagger$: "
        r"Holm-corrected Wilcoxon $p<0.05$. Ret./chg.\ (\%): share of the non-tied (instance, seed) "
        r"pairs on which the retained configuration / the changed one is better. "
        r"(a)~Every element, at the budget at which the ablation was run. "
        r"(b)~Elements re-tested at longer budgets, closer to that of the comparison ($f=1$): those "
        r"whose effect may depend on the budget, and at $f=1$ the stock configuration of "
        r"Table~\ref{tab:screening}. The elements that are clearly significant in (a) were not "
        r"re-run, which is why (b) has fewer rows.}\label{tab:ablation}",
        r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lrr@{}}",
        r"\toprule",
        r"Change & $\Delta$ & Ret./chg.\ (\%) \\",
    ]
    for k, f in enumerate(budgets):
        part = s[s.f == f]
        n_pairs, ref = int(part.pairs.iloc[0]), part.base_gap.iloc[0]
        tag = "(a)" if f == full else ("(b)" if f == longer[0] else "")
        lines += [r"\midrule",
                  rf"\multicolumn{{3}}{{@{{}}l}}{{\emph{{{(tag + ' ') if tag else ''}$f={f:g}$: reference gap "
                  rf"${ref:.3f}\%$, ${n_pairs}$ pairs}}}} \\"]
        lines += [row(r) for r in ordered(part)]
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
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
        r"= worse than \savant), with the Holm-corrected Wilcoxon $p$-value. \emph{\savant\ better / "
        r"worse}: percentage of these instances on which the mean gap of \savant\ is lower / higher "
        r"than that of the solver; the two need not sum to $100$ when there are ties."
        rf"}}\label{{tab:cmp:{name}}}",
        r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lrrrrrrrr@{}}",
        r"\toprule",
        r"Solver & Inst. & Mean & Best & Fail & Time (s) & $\Delta$ & \savant\ better / worse (\%) & $p$ \\",
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
                     f"{_wl(r)} & {ptex} \\\\")
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
        r"worse than \savant), with the Holm-corrected Wilcoxon $p$-value; $^\dagger$: $p<0.05$. "
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
