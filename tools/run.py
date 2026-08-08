#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run.py — run ./cw with each execution isolated in its own directory.

Rather than encoding the parameters in file names, every execution creates a
self-contained directory:

    results/<timestamp>[_<name>]/
        config.json      parameters, environment, binary fingerprint, summary
        run.log          raw cw output (header + summary)
        results.csv      --csv
        solutions.txt    --sol
        instances.cvrpb  --dump-bundle, only when the source is --random

Every argument is passed through to ./cw unchanged. `--csv`, `--sol` and
`--dump-bundle` are injected by this script: passing them by hand is an error
(the directory would no longer be self-contained).

The binary fingerprint is recorded because results depend on the compilation
flags: FMA contraction alone is enough to make two binaries built from the
same source diverge (see README, "Hot-path optimisation").

Usage:
    python3 tools/run.py --bundle data/cvrp_100.cvrpb --sa-steps 1000000 --restarts 10
    python3 tools/run.py --name alpha_sweep --random -n 200 --sa-steps 50000
    python3 tools/run.py --dry-run --bundle data/cvrp_20.cvrpb
"""

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))       # tools/
ROOT = os.path.dirname(HERE)                            # repository root
BIN = os.path.join(ROOT, "cw")
RESULTS = os.path.join(ROOT, "results")

INJECTED = ("--csv", "--sol", "--dump-bundle")

# (key, pattern); every line is optional depending on the options passed
SUMMARY_PATTERNS = (
    ("instances", r"instances\s*:\s*(\d+)"),
    ("infeasible", r"instances\s*:\s*\d+\s*\((\d+) infeasible\)"),
    ("cost_before_sa", r"mean cost before annealing\s*:\s*([\d.]+)"),
    ("cost_after_sa", r"mean cost after annealing\s*:\s*([\d.]+)"),
    ("std_dev", r"std deviation\s*:\s*([\d.]+)"),
    ("split_gain", r"mean Split gain\s*:\s*([\d.eE+-]+)"),
    ("sa_accept_pct", r"annealing acceptance rate\s*:\s*([\d.]+) %"),
    ("t0_mean", r"mean calibrated T0\s*:\s*([\d.eE+-]+)\)"),
    ("drift_max", r"max incremental-cost drift\s*:\s*([\d.eE+-]+)"),
    # --sa-time: the step count it bought is the only thing that makes such a
    # run replayable, so it belongs in config.json rather than only in the log
    ("sa_time_s", r"steps bought by --sa-time ([\d.eE+-]+) s"),
    ("steps_mean", r"steps bought by --sa-time [\d.eE+-]+ s\s*:\s*(\d+) mean"),
    ("steps_min", r"steps bought by --sa-time .*?\(min (\d+),"),
    ("steps_max", r"steps bought by --sa-time .*?\(min \d+, max (\d+)"),
    ("cost_median", r"median / min / max\s*:\s*([\d.]+)"),
    ("cost_min", r"median / min / max\s*:\s*[\d.]+ / ([\d.]+)"),
    ("cost_max", r"median / min / max\s*:\s*[\d.]+ / [\d.]+ / ([\d.]+)"),
    ("routes_mean", r"routes \(mean\)\s*:\s*([\d.]+)"),
    ("time_wall_s", r"total time\s*:\s*([\d.]+) s\s*\(wall\)"),
    ("time_cpu_s", r"total time\s*:.*?([\d.]+) s \(cumulative CPU\)"),
    ("throughput_per_s", r"throughput\s*:\s*([\d.]+)"),
)


def parse_summary(text):
    """Extract the metrics from cw's summary. Missing lines are ignored."""
    out = {}
    for key, pat in SUMMARY_PATTERNS:
        m = re.search(pat, text)
        if m:
            v = m.group(1)
            out[key] = int(v) if re.fullmatch(r"\d+", v) else float(v)
    return out


def binary_fingerprint(path):
    """md5 + size + date of the binary: two different binaries = two runs."""
    try:
        with open(path, "rb") as f:
            digest = hashlib.md5(f.read()).hexdigest()
        st = os.stat(path)
        return {
            "path": os.path.relpath(path, ROOT),
            "md5": digest,
            "size": st.st_size,
            "mtime": _dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            "openmp": _has_openmp(path),
        }
    except OSError as e:
        return {"path": path, "error": str(e)}


def _has_openmp(path):
    """True if the binary is linked against libomp (macOS) or libgomp (Linux)."""
    for cmd in (["otool", "-L", path], ["ldd", path]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        return bool(re.search(r"lib(omp|gomp)", out))
    return None


def tee(cmd):
    """Run cmd, echoing its output as it comes, and return (rc, out, err).

    subprocess.run(capture_output=True) would hold the whole child output in
    memory and show nothing until it exits -- minutes of silence on a long
    --per-instance run over 10 000 instances. The two pipes are drained by
    threads, which is what keeps a child writing a lot on stderr from
    deadlocking on a full pipe buffer.
    """
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, bufsize=1)
    chunks = {"out": [], "err": []}

    def pump(src, key, dst):
        for line in src:
            chunks[key].append(line)
            dst.write(line)
            dst.flush()
        src.close()

    threads = [threading.Thread(target=pump, args=(p.stdout, "out", sys.stdout)),
               threading.Thread(target=pump, args=(p.stderr, "err", sys.stderr))]
    for t in threads:
        t.start()
    rc = p.wait()
    for t in threads:
        t.join()
    return rc, "".join(chunks["out"]), "".join(chunks["err"])


def main():
    ap = argparse.ArgumentParser(
        description="Run ./cw inside a dedicated run directory",
        epilog="Any unrecognised argument is passed through to ./cw.",
    )
    ap.add_argument("--name", help="readable suffix appended to the directory name")
    ap.add_argument("--out", default=RESULTS, help="root for runs (default: results/)")
    ap.add_argument("--bin", default=BIN, help="binary to run (default: ./cw)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the command and directory without running anything")
    args, passthrough = ap.parse_known_args()

    if not passthrough:
        ap.error("no argument for cw (e.g. --bundle data/cvrp_100.cvrpb)")
    clash = [o for o in INJECTED if o in passthrough]
    if clash:
        ap.error(f"{', '.join(clash)}: injected by run.py, do not pass by hand")
    if "-q" in passthrough:
        ap.error("-q would suppress the summary run.py records in config.json")
    if not os.path.exists(args.bin):
        ap.error(f"{args.bin} not found: run `make` (or `make macos`) first")

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", args.name).strip("-") if args.name else ""
    run_dir = os.path.join(args.out, f"{stamp}_{slug}" if slug else stamp)
    if os.path.exists(run_dir):                    # two runs in the same second
        run_dir += "_" + hashlib.md5(" ".join(passthrough).encode()).hexdigest()[:6]

    cmd = [args.bin] + passthrough + [
        "--csv", os.path.join(run_dir, "results.csv"),
        "--sol", os.path.join(run_dir, "solutions.txt"),
    ]
    if "--random" in passthrough:                  # otherwise the run is unverifiable
        cmd += ["--dump-bundle", os.path.join(run_dir, "instances.cvrpb")]

    if args.dry_run:
        print(f"directory: {run_dir}")
        print("command  :", " ".join(cmd))
        return 0

    os.makedirs(run_dir, exist_ok=True)
    started = _dt.datetime.now()
    try:
        rc, out, err = tee(cmd)
    except OSError as e:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise SystemExit(f"failed to launch: {e}")
    ended = _dt.datetime.now()

    log = out + (("\n--- stderr ---\n" + err) if err else "")
    with open(os.path.join(run_dir, "run.log"), "w", encoding="utf-8") as f:
        f.write(log)

    config = {
        "run": {
            "id": os.path.basename(run_dir),
            "name": args.name,
            "started": started.isoformat(timespec="seconds"),
            "ended": ended.isoformat(timespec="seconds"),
            "elapsed_s": round((ended - started).total_seconds(), 3),
            "exit_code": rc,
        },
        "command": " ".join(cmd),
        "cw_args": passthrough,
        "resolved": [l[1:].strip() for l in out.splitlines() if l.startswith("#")],
        "binary": binary_fingerprint(args.bin),
        "environment": {
            "host": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
        },
        "result": parse_summary(out),
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\n-> {run_dir}")
    for name in sorted(os.listdir(run_dir)):
        size = os.path.getsize(os.path.join(run_dir, name))
        print(f"   {name:16} {size:>12,} B")
    return rc


if __name__ == "__main__":
    sys.exit(main())
