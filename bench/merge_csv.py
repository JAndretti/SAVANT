"""Merge per-shard result CSVs, dropping duplicate (solver, tag, instance, seed) rows.
The convergence traces next to them (<csv>.trace.jsonl) are merged the same way.

    uv run python -m bench.merge_csv results/jz/X/tmax_hgs_X_shard*.csv -o results/jz/X/tmax_hgs_X.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from bench import trace as tr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("-o", "--out", required=True, type=Path)
    a = ap.parse_args()
    df = pd.concat([pd.read_csv(f) for f in a.files], ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(subset=["solver", "tag", "instance", "seed"], keep="last")
    df.to_csv(a.out, index=False)
    print(f"{len(a.files)} files, {before} rows -> {len(df)} unique rows -> {a.out}")

    traces = {}
    for f in a.files:
        if tr.sidecar(f).exists():
            traces.update(tr.read_sidecar(tr.sidecar(f)))
    if traces:
        with tr.sidecar(a.out).open("w") as fh:
            for rec in traces.values():
                fh.write(json.dumps(rec) + "\n")
        missing = len(df) - len(traces)
        print(f"{len(traces)} traces -> {tr.sidecar(a.out)}"
              + (f" ({missing} runs without a trace)" if missing else ""))


if __name__ == "__main__":
    main()
