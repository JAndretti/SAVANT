#!/usr/bin/env python3
"""
parse_xl_bks.py — extract the XL reference numbers from the LaTeX source of

    Queiroga, Martinelli, Subramanian, Uchoa,
    "The XL Instances and the CVRPLib Best Known Solution Challenge",
    arXiv:2601.11467v2

Two tables are read from `paper/*.tex`:

  Table 1 (`tab:xl_description`)  initial and final BKS of each XL instance
  Table 2 (`tab:xl-results`)      best and mean over 60 runs of 2 hours for
                                  each of 8 reference solvers, HGS-CVRP among
                                  them

and written to `baseline/xl_bks.csv` and `baseline/xl_solvers.csv`.

This is not redundant with the `.sol` files `fetch_cvrplib.py` downloads. The
CVRPLib archive ships the **initial** BKSs, those that stood before the 30-day
BKS Challenge; the paper reports the **final** ones, and 99 of the 100 improved
during the challenge. Measuring a solver against the archive therefore flatters
it, by an amount that varies per instance. Table 2 matters for a different
reason: it gives HGS-CVRP at 2 hours per run, which is the right thing to put
next to a 20-CPU-second local run before drawing any conclusion from it.

The LaTeX source is parsed rather than the PDF: rows are `&`-separated, so no
column heuristics are needed and \\textbf{...} / \\underline{...} markup is
stripped rather than guessed at.

Usage:
    python3 tools/parse_xl_bks.py                    # finds paper/*.tex
    python3 tools/parse_xl_bks.py --check            # vs the shipped .sol
"""

import argparse
import csv
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Table 2, in the order the header declares them.
SOLVERS = ["AILS-II", "FILO", "FILO2", "KGLS-XXL", "HGS-CVRP", "SISRs",
           "LKH-3", "OR-Tools"]


def clean(cell):
    """Strip LaTeX markup from one cell and return its text."""
    cell = re.sub(r"\\(?:textbf|underline|emph|textit)\s*\{([^{}]*)\}",
                  r"\1", cell)
    cell = cell.replace("\\", "").replace("$", "").strip()
    return cell


def number(cell):
    """A cell as a float, or None for the '--' placeholders."""
    c = clean(cell).replace(",", "")
    if not c or c in ("--", "-", "–"):
        return None
    try:
        return float(c)
    except ValueError:
        return None


def table_rows(tex, label):
    """Every `a & b & ... \\\\` row of the longtable carrying `label`."""
    start = tex.find(label)
    if start < 0:
        raise SystemExit(f"{label}: not found in the .tex")
    end = tex.find("\\end{longtable}", start)
    if end < 0:
        end = len(tex)
    body = tex[start:end]
    out = []
    for line in body.splitlines():
        line = line.strip()
        if not line.endswith("\\\\") or line.startswith("\\"):
            continue
        if "multicolumn" in line or "multirow" in line or "cmidrule" in line:
            continue
        out.append([c for c in line[:-2].split("&")])
    return out


def parse_table1(tex):
    rows = []
    for cells in table_rows(tex, "tab:xl_description"):
        if len(cells) < 13:
            continue
        name = clean(cells[1])
        if not name.startswith("XL-n"):
            continue
        rows.append({
            "rank": int(clean(cells[0])),
            "name": name,
            "n": int(name.split("-")[1][1:]) - 1,
            "k_min": int(name.split("-")[2][1:]),
            "depot": clean(cells[2]),
            "customers": clean(cells[3]),
            "demand": clean(cells[4]),
            "capacity": int(number(cells[5])),
            "route_size": number(cells[6]),
            "initial_bks": int(number(cells[7])),
            "initial_method": clean(cells[8]),
            "final_bks": int(number(cells[10])),
            "final_team": clean(cells[11]).replace("--", ""),
            "improvement_pct": number(cells[12]),
        })
    return rows


def parse_table2(tex):
    rows = []
    for cells in table_rows(tex, "tab:xl-results"):
        if len(cells) < 17:
            continue
        name = clean(cells[0])
        if not name.startswith("XL-n"):
            continue
        r = {"name": name}
        for i, soc in enumerate(SOLVERS):
            r[f"{soc}_best"] = number(cells[1 + 2 * i])
            r[f"{soc}_mean"] = number(cells[2 + 2 * i])
        rows.append(r)
    return rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Extract the XL paper's reference "
                                             "numbers into baseline/")
    ap.add_argument("tex", nargs="?", default=None,
                    help="the paper's .tex (default: the one in paper/)")
    ap.add_argument("--out", default=os.path.join(ROOT, "baseline"))
    ap.add_argument("--check", action="store_true",
                    help="cross-check against data/cvrplib/XL_bks.csv, the "
                         "costs read from the distributed .sol files")
    args = ap.parse_args()

    tex_path = args.tex
    if tex_path is None:
        cands = glob.glob(os.path.join(ROOT, "paper", "*.tex"))
        if not cands:
            raise SystemExit("no .tex in paper/ — pass one explicitly")
        tex_path = max(cands, key=os.path.getsize)

    with open(tex_path, encoding="utf-8", errors="replace") as f:
        tex = f.read()

    t1 = parse_table1(tex)
    t2 = parse_table2(tex)
    if len(t1) != 100:
        print(f"warning: table 1 gave {len(t1)} rows, expected 100",
              file=sys.stderr)
    if len(t2) != 100:
        print(f"warning: table 2 gave {len(t2)} rows, expected 100",
              file=sys.stderr)

    p1 = os.path.join(args.out, "xl_bks.csv")
    p2 = os.path.join(args.out, "xl_solvers.csv")
    write_csv(p1, t1)
    write_csv(p2, t2)

    print(f"source: {os.path.relpath(tex_path, ROOT)}")
    print(f"  {len(t1)} instances -> {os.path.relpath(p1, ROOT)}")
    print(f"  {len(t2)} instances -> {os.path.relpath(p2, ROOT)}")

    imp = [r["improvement_pct"] for r in t1]
    print()
    print(f"  improved during the challenge : "
          f"{sum(1 for v in imp if v > 0)}/{len(imp)}")
    print(f"  improvement mean / max        : {sum(imp) / len(imp):.3f} % / "
          f"{max(imp):.3f} %")

    hgs = [r for r in t2 if r["HGS-CVRP_best"] is not None]
    if hgs:
        by_name = {r["name"]: r for r in t1}
        g = [100.0 * (r["HGS-CVRP_best"] - by_name[r["name"]]["final_bks"])
             / by_name[r["name"]]["final_bks"] for r in hgs if r["name"] in by_name]
        print(f"  HGS-CVRP at 2 h x 60 runs     : best-of-60 is "
              f"{sum(g) / len(g):+.3f} % from the final BKS, on {len(g)} instances")

    if args.check:
        shipped = os.path.join(ROOT, "data", "cvrplib", "XL_bks.csv")
        if not os.path.exists(shipped):
            raise SystemExit(f"{shipped} not found — run fetch_cvrplib.py first")
        by_name = {r["name"]: r for r in t1}
        eq_init = eq_final = other = 0
        worst = (0.0, "")
        with open(shipped, newline="") as f:
            for s in csv.DictReader(f):
                p = by_name.get(s["name"])
                if not p or not s["bks_cost"]:
                    continue
                c = float(s["bks_cost"])
                if abs(c - p["initial_bks"]) < 0.5:
                    eq_init += 1
                elif abs(c - p["final_bks"]) < 0.5:
                    eq_final += 1
                else:
                    other += 1
                d = 100.0 * (c - p["final_bks"]) / p["final_bks"]
                if d > worst[0]:
                    worst = (d, s["name"])
        print()
        print("  the .sol files distributed with the instances match:")
        print(f"    the INITIAL (pre-challenge)  BKS : {eq_init}")
        print(f"    the FINAL   (post-challenge) BKS : {eq_final}")
        print(f"    neither                          : {other}")
        if eq_init > eq_final:
            print("  -> the archive ships PRE-challenge solutions. A gap "
                  "measured against\n     data/cvrplib/XL_bks.csv is "
                  f"understated; worst case {worst[0]:.3f} % "
                  f"({worst[1]}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
