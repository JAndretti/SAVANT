"""Run SAVANT (cw) on a set of .vrp instances.

Budget (exactly one):
  --steps K        fixed annealing steps (parameter study); --steps N = cw's size rule
  --time S         wall seconds per instance (cw --sa-wall: clock-enforced)
  --time-hgs F     T_max = F * n * 240/100 s  (Vidal 2022 protocol)

Examples:
  uv run python -m bench.run_savant --set data/cvrplib/X --steps 1000000 --tag default \
      --out results/savant_X.csv --binary cw_repro
  uv run python -m bench.run_savant --set data/cvrplib/X --time-hgs 1 --seeds 10 \
      --tag final --extra "--ops 1,1,1,1,1,0.05" --out results/savant_X_tmax.csv
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

from bench import trace as tr
from bench.common import ROOT, is_rounded, read_cwsol
from bench.pool import (RunOutput, Runner, Task, add_common_args, add_time_args, execute,
                        make_tasks, seconds_for)


# cw's compiled defaults until 2026-09-17, i.e. the reference configuration of the
# parameter study (bench.sweep, time_match, interact). The defaults are now the
# retained configuration (--pick 1 --pick2 1 --restarts 4 --kick 50 --kick-max 20);
# prepending STOCK reproduces the old behaviour bit for bit.
STOCK = "--pick 2 --pick2 2 --restarts 1 --kick 100 --kick-max 10"


def restarts_of(options: str) -> int:
    """Restarts a cw command line asks for: the last occurrence wins, default 4."""
    found = re.findall(r"--restarts\s+(\d+)", options)
    return int(found[-1]) if found else 4


def resolve_binary(name: str) -> str:
    """Absolute path of the cw binary to run.

    The runs are launched through `taskset`, which execs its command by PATH
    lookup, and PATH does not contain the working directory: a bare `--binary
    cw` therefore fails with "No such file or directory" even when ./cw exists.
    Bare names are resolved against the repository root, where the Makefile puts
    cw and cw_repro, and anything not found is refused here rather than by each
    of the 40 jobs after the pool has started."""
    p = Path(name)
    if p.is_absolute():
        found = p if p.exists() else None
    elif (ROOT / p).exists():          # cw, cw_repro: the binaries make builds
        found = ROOT / p
    elif p.exists():                   # a path relative to the working directory
        found = p
    else:
        on_path = shutil.which(name)
        found = Path(on_path) if on_path else None
    if found is None:
        raise SystemExit(f"--binary {name}: not found at {ROOT / p}, "
                         f"under {Path.cwd()}, or on PATH (is it built? `make cw repro`)")
    if not os.access(found, os.X_OK):
        raise SystemExit(f"--binary {name}: {found} is not executable")
    return str(found.resolve())


def savant_runner(binary: str, extra: list[str]) -> Runner:
    """Wall-clock tasks run under --sa-wall with a convergence trace, the others
    under --sa-steps task.steps."""
    def runner(t: Task, cpu: int, tmp: Path) -> RunOutput:
        (tmp / "in").mkdir()
        (tmp / "in" / t.vrp.name).symlink_to(t.vrp.resolve())
        if t.budget_s is not None:
            budget = ["--sa-wall", f"{t.budget_s:.3f}", "--trace"]
        else:
            budget = ["--sa-steps", str(t.steps)]
        cmd = ["taskset", "-c", str(cpu), binary, "--dir", str(tmp / "in"),
               "--threads", "1", "--seed", str(t.seed), "-q", "--sol", str(tmp / "r.sol"),
               *(["--round"] if is_rounded(t.vrp) else []), *budget, *extra]
        p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "1"})
        if p.returncode:
            raise RuntimeError(f"{shlex.join(cmd)}\n{p.stderr}")
        text = (tmp / "r.sol").read_text()
        steps = next((l.split()[1] for l in text.splitlines() if l.startswith("#sa-steps")),
                     t.steps or "")
        reported, routes = read_cwsol(tmp / "r.sol")[t.vrp.stem]
        trace = tr.parse_savant(text) if t.budget_s is not None else None
        return RunOutput(routes, reported, steps, trace)
    return runner


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    b = ap.add_mutually_exclusive_group(required=True)
    b.add_argument("--steps")
    add_time_args(b)
    ap.add_argument("--extra", default="", help="extra cw options, one quoted string")
    ap.add_argument("--binary", default=str(ROOT / "cw"),
                    help="cw (fast, for timed runs) or cw_repro (-ffp-contract=off: "
                         "bit-reproducible across code edits, for step-budget studies)")
    a = ap.parse_args()

    if a.steps is not None:
        tasks = make_tasks(a, lambda v: None, lambda v: f"steps={a.steps}")
        for t in tasks:
            t.steps = a.steps
    else:
        tasks = make_tasks(a, lambda v: seconds_for(a, v), lambda v: f"time={seconds_for(a, v):.3f}")
    execute(tasks, savant_runner(resolve_binary(str(a.binary)), shlex.split(a.extra)), solver="savant",
            tag=a.tag, extra=a.extra, out=a.out, jobs=a.jobs)


if __name__ == "__main__":
    main()
