#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze.py — statistics and plots for one run, or a comparison across runs.

Works entirely from what `cw` already writes: `results.csv` (one row per
instance) and `config.json` (resolved configuration + parsed summary). Nothing
here requires instrumenting the solver.

Two quantities are *derived analytically* rather than measured, because the
solver is deterministic in both:

  * the temperature schedule — geometric from T0 to T0*10^-decades over the
    step budget, so T(it) is exact once T0 is known (cw reports the mean
    calibrated T0);
  * the number of draws per operator — the operator is picked by comparing one
    uniform 32-bit word against thresholds derived from the --ops weights, so
    draws_i = steps * w_i / sum(w) to within sampling noise.

What is *not* available without modifying cw: the cost trajectory inside a run,
and the number of *accepted* moves per operator (cw keeps a single aggregate
acceptance counter). See the README of this directory.

Usage:
    python3 tools/analyze.py results/<run>                # one run
    python3 tools/analyze.py results/<run-a> results/<run-b> ...   # compare
    python3 tools/analyze.py results/<run> --out fig.png
"""

import argparse
import csv
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- appearance
# Categorical slots are assigned in a fixed order and never cycled; the same
# operator keeps the same hue across every panel.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d8d7d2"

OPERATORS = ("relocate", "swap", "2-opt", "or-opt")


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "font.size": 9,
        "lines.linewidth": 2.0,
    })


def grid(ax, axis="y"):
    """Recessive grid, behind the marks."""
    ax.grid(True, axis=axis, alpha=0.7)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------- loading

def load_run(path):
    """Read one run directory into a dict. Missing pieces degrade, not crash."""
    csv_path = os.path.join(path, "results.csv")
    if not os.path.exists(csv_path):
        raise SystemExit(f"{path}: no results.csv — is this a run directory?")
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path}: results.csv is empty")

    cfg = {}
    try:
        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        pass

    num = lambda k: np.array([float(r[k]) for r in rows])
    return {
        "path": path,
        "name": os.path.basename(os.path.normpath(path)),
        "cfg": cfg,
        "n_inst": len(rows),
        "n": num("n"),
        "cap": num("capacity"),
        "cw": num("cost_cw"),
        "sa": num("cost_annealed"),
        "routes": num("routes"),
        "max_load": num("max_load"),
        "feasible": num("feasible"),
        "time_ms": num("time_ms"),
    }


def cw_arg(cfg, flag, default=None, cast=float):
    """Read a value that follows `flag` in the recorded command line."""
    args = (cfg or {}).get("cw_args") or []
    if flag in args:
        i = args.index(flag)
        if i + 1 < len(args):
            try:
                return cast(args[i + 1])
            except ValueError:
                return default
    return default


def resolved(cfg, pattern, cast=float, default=None):
    """Pull a number out of cw's own resolved-configuration header."""
    for line in (cfg or {}).get("resolved") or []:
        m = re.search(pattern, line)
        if m:
            try:
                return cast(m.group(1))
            except ValueError:
                return default
    return default


def schedule(run):
    """(steps, T0, Tend, T array) for one restart, or None if SA was disabled.

    The schedule is geometric and independent of the search, so this is the
    exact temperature the solver used — not an approximation.
    """
    cfg = run["cfg"]
    steps = cw_arg(cfg, "--sa-steps", None, int)
    if steps is None:
        steps = resolved(cfg, r"annealing\s*:\s*(\d+) steps", int, 1000)
    if not steps or steps <= 0:
        return None

    t0 = (cfg.get("result") or {}).get("t0_mean")
    if t0:                                     # calibrated mode
        decades = cw_arg(cfg, "--t-decades", None)
        if decades is None:
            decades = resolved(cfg, r"([\d.]+) decades", float, 2.0)
        tend = t0 * 10.0 ** (-decades)
    else:                                      # --t0 / --tend given by hand
        t0 = cw_arg(cfg, "--t0")
        tend = cw_arg(cfg, "--tend", (t0 or 0) * 1e-4)
        if not t0:
            return None
    if tend <= 0 or t0 <= 0:
        return None

    alpha = (tend / t0) ** (1.0 / (steps - 1)) if steps > 1 else 1.0
    it = np.unique(np.geomspace(1, steps, min(steps, 400)).astype(int)) - 1
    return steps, t0, tend, it, t0 * alpha ** it


def op_weights(run):
    """Normalised --ops weights, defaulting to the solver's own default."""
    raw = cw_arg(run["cfg"], "--ops", "1,1,1,0", str)
    parts = re.split(r"[,:/]", raw)
    w = []
    for i in range(4):
        try:
            w.append(float(parts[i]))
        except (IndexError, ValueError):
            w.append(0.0)
    tot = sum(w)
    return [x / tot for x in w] if tot > 0 else [0.0] * 4


# ------------------------------------------------------------------- reports

def stats(run):
    """Print the numbers behind the plots."""
    cfg, res = run["cfg"], (run["cfg"].get("result") or {})
    cw, sa = run["cw"], run["sa"]
    gain = cw - sa
    pct = 100.0 * gain / np.where(cw > 0, cw, 1.0)
    util = run["max_load"] / np.where(run["cap"] > 0, run["cap"], 1.0)

    print(f"run {run['name']}")
    for line in (cfg.get("resolved") or [])[:12]:
        print(f"  | {line}")
    print()
    print(f"  instances            : {run['n_inst']}  "
          f"(n={int(run['n'][0])}, Q={run['cap'][0]:g}, "
          f"{int((run['feasible'] == 0).sum())} infeasible)")
    for label, v in (("cost C&W", cw), ("cost annealed", sa)):
        print(f"  {label:21}: mean {v.mean():.5f}  sd {v.std(ddof=1):.5f}  "
              f"median {np.median(v):.5f}  min {v.min():.5f}  max {v.max():.5f}")
    print(f"  improvement          : mean {pct.mean():+.2f} %  "
          f"median {np.median(pct):+.2f} %  best {pct.max():+.2f} %  "
          f"worst {pct.min():+.2f} %")
    print(f"                         {int((gain <= 1e-12).sum())} instance(s) "
          f"not improved at all")
    print(f"  routes               : mean {run['routes'].mean():.3f}  "
          f"min {int(run['routes'].min())}  max {int(run['routes'].max())}")
    print(f"  capacity utilisation : mean {100 * util.mean():.1f} %  "
          f"{100 * (util > 0.999).mean():.1f} % of instances have a full route")
    print(f"  time                 : {run['time_ms'].mean():.4f} ms/instance "
          f"(single-core), total wall {res.get('time_wall_s', float('nan')):.3f} s")

    sch = schedule(run)
    if sch:
        steps, t0, tend, _, _ = sch
        restarts = cw_arg(run["cfg"], "--restarts", 1, int)
        print(f"  annealing            : {steps} steps x {restarts} restart(s), "
              f"T {t0:.4g} -> {tend:.4g}")
        w = op_weights(run)
        total = steps * restarts
        parts = ", ".join(f"{OPERATORS[i]} {int(round(w[i] * total)):,}"
                          for i in range(4) if w[i] > 0)
        print(f"  operator draws       : {parts}   (analytic, "
              f"{total:,} draws/instance)")
    if res.get("sa_accept_pct") is not None:
        print(f"  acceptance rate      : {res['sa_accept_pct']} % "
              f"(all operators pooled — cw keeps no per-operator counter)")
    if res.get("drift_max") is not None:
        print(f"  incremental drift    : {res['drift_max']:.3e}")
    print()


# --------------------------------------------------------------- single run

def figure_single(run, out):
    style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
    fig.suptitle(f"SAVANT — {run['name']}", fontsize=13, color=INK,
                 x=0.012, ha="left", y=0.985)
    cw, sa = run["cw"], run["sa"]

    # 1. cost distribution, before vs after — two series, so a legend is present
    ax = axes[0][0]
    lo, hi = min(cw.min(), sa.min()), max(cw.max(), sa.max())
    bins = np.linspace(lo, hi, 45)
    # filled + outlined rather than two alpha fills: overlapping translucent
    # fills blend into a third colour that belongs to neither series
    ax.hist(cw, bins=bins, color=SERIES[0], alpha=0.35, label="Clarke & Wright")
    ax.hist(sa, bins=bins, color=SERIES[1], histtype="step", lw=2,
            label="after annealing")
    ax.axvline(cw.mean(), color=SERIES[0], lw=1.4, ls="--")
    ax.axvline(sa.mean(), color=SERIES[1], lw=1.4, ls="--")
    ax.set_title("Cost distribution")
    ax.set_xlabel("tour cost")
    ax.set_ylabel("instances")
    ax.legend(loc="upper right")
    grid(ax)

    # 2. per-instance improvement
    ax = axes[0][1]
    pct = 100.0 * (cw - sa) / np.where(cw > 0, cw, 1.0)
    ax.hist(pct, bins=45, color=SERIES[0])
    ax.axvline(pct.mean(), color=INK, lw=1.4, ls="--")
    ax.annotate(f"mean {pct.mean():.2f} %", xy=(pct.mean(), 0),
                xytext=(6, 8), textcoords="offset points",
                xycoords=("data", "axes fraction"), color=INK, fontsize=8.5)
    ax.set_title("Improvement per instance")
    ax.set_xlabel("cost reduction (%)")
    ax.set_ylabel("instances")
    grid(ax)

    # 3. temperature schedule — analytic, exact
    ax = axes[0][2]
    sch = schedule(run)
    if sch:
        steps, t0, tend, it, T = sch
        ax.plot(it + 1, T, color=SERIES[0])
        ax.set_yscale("log")          # geometric decay -> a straight line
        ax.set_title("Temperature schedule (derived)")
        ax.set_xlabel("step")
        ax.set_ylabel("T")
        ax.annotate(f"T0 = {t0:.4g}", xy=(1, t0), xytext=(8, -10),
                    textcoords="offset points", color=INK_2, fontsize=8.5)
        ax.annotate(f"Tend = {tend:.3g}", xy=(steps, tend), xytext=(-6, 10),
                    textcoords="offset points", ha="right",
                    color=INK_2, fontsize=8.5)
        grid(ax, axis="y")
    else:
        ax.text(0.5, 0.5, "annealing disabled", ha="center", va="center",
                color=INK_2, transform=ax.transAxes)
        ax.set_axis_off()

    # 4. operator budget — draws are analytic; acceptances are not available
    ax = axes[1][0]
    if sch:
        steps = sch[0] * cw_arg(run["cfg"], "--restarts", 1, int)
        w = op_weights(run)
        draws = [x * steps for x in w]
        y = np.arange(4)[::-1]
        ax.barh(y, draws, height=0.62,
                color=[SERIES[i] for i in range(4)])
        for yi, d, wi in zip(y, draws, w):
            # direct labels: also the relief required by the contrast warning
            ax.annotate(f"{d:,.0f}   ({100 * wi:.0f} %)",
                        xy=(d, yi), xytext=(6, 0), textcoords="offset points",
                        va="center", color=INK, fontsize=8.5)
        ax.set_yticks(y, OPERATORS)
        ax.set_xlim(0, max(draws) * 1.42 if max(draws) else 1)
        ax.set_title("Operator draws per instance (derived)")
        ax.set_xlabel("draws")
        grid(ax, axis="x")
    else:
        ax.set_axis_off()

    # 5. route count. Capacity utilisation lives in the printed stats instead:
    # with integer demands it is almost always exactly 100 %, so a histogram of
    # it is a single spike and says nothing.
    ax = axes[1][1]
    r_min, r_max = int(run["routes"].min()), int(run["routes"].max())
    ax.hist(run["routes"], bins=np.arange(r_min - 0.5, r_max + 1.5),
            color=SERIES[2])
    ax.set_xticks(range(r_min, r_max + 1))
    util = 100.0 * run["max_load"] / np.where(run["cap"] > 0, run["cap"], 1.0)
    ax.set_title("Routes per instance")
    ax.set_xlabel(f"routes  (fullest route averages {util.mean():.1f} % of Q)")
    ax.set_ylabel("instances")
    grid(ax)

    # 6. does annealing help more on worse starts?
    ax = axes[1][2]
    ax.scatter(cw, pct, s=9, color=SERIES[0], alpha=0.45,
               edgecolors="none")
    if len(cw) > 2:
        b, a = np.polyfit(cw, pct, 1)
        xs = np.linspace(cw.min(), cw.max(), 2)
        ax.plot(xs, a + b * xs, color=INK, lw=1.4, ls="--")
        r = np.corrcoef(cw, pct)[0, 1]
        ax.annotate(f"r = {r:+.2f}", xy=(0.97, 0.05), xycoords="axes fraction",
                    ha="right", color=INK_2, fontsize=8.5)
    ax.set_title("Improvement vs starting cost")
    ax.set_xlabel("Clarke & Wright cost")
    ax.set_ylabel("cost reduction (%)")
    grid(ax, axis="both")

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out}")


# ------------------------------------------------------------ several runs

def sweep_axis(runs):
    """Find the single cw option that varies across runs, to use as the x axis.

    Returns (label, values) when exactly one numeric option differs, otherwise
    (None, None) and the caller falls back to categorical run names.
    """
    keys = set()
    for r in runs:
        args = (r["cfg"] or {}).get("cw_args") or []
        keys |= {a for a in args if a.startswith("--")}
    varying = []
    for k in sorted(keys):
        vals = [cw_arg(r["cfg"], k) for r in runs]
        if any(v is None for v in vals):
            continue
        if len(set(vals)) > 1:
            varying.append((k, vals))
    if len(varying) == 1:
        return varying[0]
    return None, None


def figure_compare(runs, out):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("SAVANT — run comparison", fontsize=13, color=INK,
                 x=0.012, ha="left", y=0.98)

    means = np.array([r["sa"].mean() for r in runs])
    walls = np.array([(r["cfg"].get("result") or {}).get("time_wall_s", np.nan)
                      for r in runs])
    label, xs = sweep_axis(runs)

    # 1. cost against whatever was swept
    ax = axes[0]
    if xs is not None:
        order = np.argsort(xs)
        x = np.array(xs)[order]
        ax.plot(x, means[order], color=SERIES[0], marker="o", markersize=6)
        ax.set_xlabel(label)
        if x.max() / max(x.min(), 1e-12) > 50:
            ax.set_xscale("log")
        ax.set_title(f"Mean cost vs {label}")
    else:
        y = np.arange(len(runs))[::-1]
        ax.barh(y, means, height=0.6, color=SERIES[0])
        ax.set_yticks(y, [r["name"][-22:] for r in runs])
        ax.set_xlim(means.min() * 0.985, means.max() * 1.004)
        for yi, m in zip(y, means):
            ax.annotate(f"{m:.5f}", xy=(m, yi), xytext=(-6, 0),
                        textcoords="offset points", ha="right", va="center",
                        color="#ffffff", fontsize=8.5)
        ax.set_title("Mean cost after annealing")
    ax.set_ylabel("mean cost")
    grid(ax, axis="both" if xs is not None else "x")

    # 2. quality against wall time — the honest way to compare configurations
    ax = axes[1]
    ax.scatter(walls, means, s=52, color=SERIES[0], zorder=3,
               edgecolors=SURFACE, linewidths=1.5)
    # label by the swept value when there is one — a truncated run id is noise
    tags = ([f"{label} {v:g}" for v in xs] if xs is not None
            else [r["name"].split("_", 1)[-1] for r in runs])
    for tag, w, m in zip(tags, walls, means):
        right = w > (np.nanmin(walls) + np.nanmax(walls)) / 2
        ax.annotate(tag, xy=(w, m),
                    xytext=(-8 if right else 8, 5),
                    textcoords="offset points",
                    ha="right" if right else "left",
                    color=INK_2, fontsize=8)
    if np.nanmax(walls) / max(np.nanmin(walls), 1e-9) > 50:
        ax.set_xscale("log")
    ax.margins(x=0.12, y=0.12)
    ax.set_title("Quality vs wall time")
    ax.set_xlabel("wall time (s)")
    ax.set_ylabel("mean cost")
    grid(ax, axis="both")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"-> {out}")



# ---------------------------------------------------------- traced single run

def trace_base(target):
    """Accept a trace directory (the normal case) or a bare file prefix."""
    if os.path.isdir(target):
        return os.path.join(target, "trace")
    for ext in (".csv", ".json"):
        if target.endswith(ext):
            return target[: -len(ext)]
    return target


def load_trace(target):
    """Read the trace.csv / trace.json pair written by ./cw_trace."""
    base = trace_base(target)
    with open(base + ".json", encoding="utf-8") as f:
        meta = json.load(f)
    rows = None
    if os.path.exists(base + ".csv"):
        with open(base + ".csv", newline="", encoding="utf-8") as f:
            rd = list(csv.DictReader(f))
        rows = {k: np.array([float(r[k]) for r in rd])
                for k in ("step", "T", "cur", "best", "op", "accepted", "delta",
                          "routes")}
    return meta, rows


def figure_trace(target, out):
    """The panels that only an instrumented run can produce."""
    meta, tr = load_trace(target)
    ops = meta["operators"]
    style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
    inst = meta["instance"]
    fig.suptitle(f"SAVANT trace — {inst['name']}  (n={inst['n']}, "
                 f"{meta['steps']:,} steps)", fontsize=13, color=INK,
                 x=0.012, ha="left", y=0.985)

    # 1. the cost trajectory — the whole point of tracing
    ax = axes[0][0]
    if tr is not None:
        ax.plot(tr["step"], tr["cur"], color=SERIES[0], lw=0.7, alpha=0.55,
                label="current")
        ax.plot(tr["step"], tr["best"], color=SERIES[1], lw=1.8, label="best")
        ax.axhline(meta["cost_cw"], color=INK_2, lw=1.0, ls=":")
        ax.annotate("Clarke & Wright", xy=(0, meta["cost_cw"]), xytext=(6, 4),
                    textcoords="offset points", color=INK_2, fontsize=8.5)
        ax.legend(loc="upper right")
    ax.set_title("Cost trajectory")
    ax.set_xlabel("step")
    ax.set_ylabel("tour cost")
    grid(ax)

    # 2. temperature. Deliberately its own panel: overlaying it on the cost
    # would need a second y-scale, which makes the two curves' crossings and
    # relative slopes meaningless.
    ax = axes[0][1]
    if tr is not None:
        ax.plot(tr["step"], tr["T"], color=SERIES[0])
        ax.set_yscale("log")
    ax.set_title("Temperature")
    ax.set_xlabel("step")
    ax.set_ylabel("T")
    grid(ax)

    # 3. acceptance over time, from the sampled rows
    ax = axes[0][2]
    if tr is not None and len(tr["step"]) > 20:
        k = max(5, len(tr["accepted"]) // 60)
        kern = np.ones(k) / k
        roll = np.convolve(tr["accepted"], kern, mode="valid")
        ax.plot(tr["step"][k - 1:], 100 * roll, color=SERIES[0])
        ax.axhline(100 * meta["accepted"] / meta["draws"], color=INK_2,
                   lw=1.0, ls=":")
        ax.annotate(f"run mean {100 * meta['accepted'] / meta['draws']:.2f} %",
                    xy=(0.97, 0.92), xycoords="axes fraction", ha="right",
                    color=INK_2, fontsize=8.5)
    ax.set_title("Acceptance rate over time")
    ax.set_xlabel("step")
    ax.set_ylabel("accepted (%)")
    grid(ax)

    live = [o for o in ops if o["draws"]]
    y = np.arange(len(live))[::-1]
    colors = [SERIES[[o["name"] for o in ops].index(o["name"])] for o in live]

    # 4. accepted moves per operator (draws + rate as direct labels, since the
    #    two counts differ by two orders of magnitude and must not share an axis)
    ax = axes[1][0]
    acc = [o["accepted"] for o in live]
    ax.barh(y, acc, height=0.62, color=colors)
    for yi, o in zip(y, live):
        ax.annotate(f"{o['accepted']:,} of {o['draws']:,}"
                    f"   ({100 * o['accept_rate']:.2f} %)",
                    xy=(o["accepted"], yi), xytext=(6, 0),
                    textcoords="offset points", va="center",
                    color=INK, fontsize=8.5)
    ax.set_yticks(y, [o["name"] for o in live])
    ax.set_xlim(0, max(acc) * 1.75 if acc else 1)
    ax.set_title("Accepted moves per operator")
    ax.set_xlabel("accepted")
    grid(ax, axis="x")

    # 5. who actually moved the cost. Negative = net improvement.
    ax = axes[1][1]
    tot = [o["sum_delta"] for o in live]
    ax.barh(y, tot, height=0.62, color=colors)
    ax.axvline(0, color=INK_2, lw=1.0)
    for yi, v in zip(y, tot):
        ax.annotate(f"{v:+.3f}", xy=(v, yi),
                    xytext=(6 if v >= 0 else -6, 0),
                    textcoords="offset points", va="center",
                    ha="left" if v >= 0 else "right", color=INK, fontsize=8.5)
    ax.set_yticks(y, [o["name"] for o in live])
    ax.margins(x=0.28)
    ax.set_title("Net cost contribution  (negative = improvement)")
    ax.set_xlabel("summed delta of accepted moves")
    grid(ax, axis="x")

    # 6. route count over time — the search can lose routes but never gain them
    ax = axes[1][2]
    if tr is not None:
        ax.plot(tr["step"], tr["routes"], color=SERIES[2], lw=1.6)
        ax.set_ylim(tr["routes"].min() - 0.6, tr["routes"].max() + 0.6)
        ax.set_yticks(range(int(tr["routes"].min()), int(tr["routes"].max()) + 1))
    ax.set_title("Routes in use")
    ax.set_xlabel("step")
    ax.set_ylabel("routes")
    grid(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"-> {out}")


def stats_trace(target):
    meta, _ = load_trace(target)
    inst = meta["instance"]
    print(f"trace {inst['name']}  n={inst['n']}  Q={inst['capacity']:g}  "
          f"seed {meta['seed']}")
    print(f"  cost      : C&W {meta['cost_cw']:.6f} -> {meta['cost_final']:.6f}  "
          f"({100 * (meta['cost_final'] - meta['cost_cw']) / meta['cost_cw']:+.2f} %)")
    print(f"  schedule  : {meta['steps']:,} steps, T {meta['t0']:.5g} -> "
          f"{meta['tend']:.5g}")
    print(f"  drift     : {meta['drift']:.3e}")
    print(f"  {'operator':10} {'draws':>10} {'accepted':>10} {'rate':>8} "
          f"{'sum delta':>12} {'new best':>9}")
    for o in meta["operators"]:
        if not o["draws"]:
            continue
        print(f"  {o['name']:10} {o['draws']:>10,} {o['accepted']:>10,} "
              f"{100 * o['accept_rate']:>7.2f}% {o['sum_delta']:>12.5f} "
              f"{o['new_best']:>9,}")
    print(f"  {'total':10} {meta['draws']:>10,} {meta['accepted']:>10,} "
          f"{100 * meta['accepted'] / meta['draws']:>7.2f}%")
    print()


def main():
    ap = argparse.ArgumentParser(
        description="Statistics and plots for SAVANT runs",
        epilog="One run directory: full analysis. Several: comparison.",
    )
    ap.add_argument("runs", nargs="*", help="run directories (results/<id>/)")
    ap.add_argument("--trace", metavar="DIR",
                    help="a ./cw_trace output directory (results/trace_.../)")
    ap.add_argument("--out", help="output PNG "
                                  "(default: analysis.png inside the run)")
    ap.add_argument("--no-plot", action="store_true", help="statistics only")
    args = ap.parse_args()

    if args.trace:
        stats_trace(args.trace)
        if not args.no_plot:
            default = (os.path.join(args.trace, "analysis.png")
                       if os.path.isdir(args.trace)
                       else trace_base(args.trace) + ".png")
            figure_trace(args.trace, args.out or default)
        if not args.runs:
            return 0

    if not args.runs:
        ap.error("give at least one run directory, or --trace DIR")
    runs = [load_run(p) for p in args.runs]
    for r in runs:
        stats(r)

    if args.no_plot:
        return 0
    if len(runs) == 1:
        out = args.out or os.path.join(runs[0]["path"], "analysis.png")
        figure_single(runs[0], out)
    else:
        out = args.out or "comparison.png"
        figure_compare(runs, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
