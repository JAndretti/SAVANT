"""Convergence traces: the incumbent cost of a run against time.

Every timed runner returns the (elapsed seconds, cost) pairs its solver reports
when the best solution improves. Only the final solution is re-checked from the
coordinates; intermediate costs are the solver's own figures.

Each solver measures time from its own origin, with its own clock:

  savant   start of the instance's solve, after reading it          wall
  hgs      Params construction, after reading the instance          CPU (clock)
  filo     start of main, before reading the instance               wall
  filo2    start of main, before reading the instance               wall
  ails     start of the search, after JVM start-up and reading      wall
  lkh      start of main, before reading the problem                CPU (getrusage)
  ortools  start of the script, before reading the instance         wall

The runs are single-threaded and pinned, so CPU and wall time coincide up to
I/O; the whole-process wall time is recorded separately in the CSV (time_s).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# fractions of T_max at which Vidal (2022) reports the gaps
CHECKPOINTS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00)

Trace = list[tuple[float, float]]


def parse_prefixed(text: str, prefix: str = "TRACE") -> Trace:
    """Lines '<prefix> <seconds> <cost>' (patched FILO, FILO2, LKH-3; OR-Tools wrapper)."""
    out = []
    for line in text.splitlines():
        if line.startswith(prefix + " "):
            _, t, c = line.split()[:3]
            out.append((float(t), float(c)))
    return out


def parse_savant(sol_text: str) -> Trace:
    """'#trace <idx> <seconds> <cost>' lines of a cw solution file (one instance)."""
    return [(float(t), float(c)) for _, _, t, c in
            (l.split() for l in sol_text.splitlines() if l.startswith("#trace "))]


def parse_hgs(pg_csv: str) -> Trace:
    """HGS-CVRP's <solution>.PG.csv: 'instance;seed;cost;seconds' per new best."""
    out = []
    for line in pg_csv.splitlines():
        if line.strip():
            _, _, c, t = line.rsplit(";", 3)
            out.append((float(t), float(c)))
    return out


_AILS = re.compile(r"^solution quality: (\S+) .* time: (\S+)\s*$")


def parse_ails(stdout: str) -> Trace:
    """AILS-II prints 'solution quality: <f> gap: .. time: <s>' on each new best."""
    return [(float(m[2]), float(m[1])) for m in map(_AILS.match, stdout.splitlines()) if m]


def check(trace: Trace, cost: float, rounded: bool) -> tuple[bool, str]:
    """A usable trace has increasing times and ends on the checked final cost."""
    if not trace:
        return False, "empty trace"
    if any(b[0] < a[0] for a, b in zip(trace, trace[1:])):
        return False, "times decrease"
    tol = (1e-6 if rounded else 1e-3) * max(1.0, abs(cost))
    last = trace[-1][1]
    if abs(last - cost) > tol:
        return False, f"last trace cost {last} != final cost {cost}"
    ups = sum(b[1] > a[1] + tol for a, b in zip(trace, trace[1:]))
    return True, f"{ups} increases" if ups else ""


def incumbent_at(trace: Trace, t: float) -> float | None:
    """Cost the solver held at time t (last entry at or before t); None before the first."""
    best = None
    for s, c in trace:
        if s > t:
            break
        best = c
    return best


def sidecar(csv_path: Path) -> Path:
    return csv_path.with_suffix(".trace.jsonl")


def read_sidecar(path: Path) -> dict[tuple[str, str, str, int], dict]:
    """Records keyed by (solver, tag, instance, seed); a later duplicate wins."""
    out = {}
    with path.open() as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[(r["solver"], r["tag"], r["instance"], int(r["seed"]))] = r
    return out
