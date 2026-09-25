"""Build an XML100 test set disjoint from the tuning set.

    uv run python -m bench.make_xml100_test [--n 100]

XML100 \\citep{queiroga2022} has 10,000 instances; 378 of them (one per attribute
code) are in data/tuning/xml100 and were used to choose SAVANT's configuration,
and 91 more have an invalid reference solution. Both groups are excluded here, so
the resulting set shares no instance with the parameter study.

The sample is stratified over the 4-digit attribute code (depot position, customer
positioning, demand distribution, route size) so that the test set covers the same
range of characteristics as the full collection rather than a random corner of it.
"""

from __future__ import annotations

import argparse
import shutil

import numpy as np

from bench.common import DATA

SEED = 20260922


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="instances to select")
    ap.add_argument("--out", default="XML100_test")
    a = ap.parse_args()

    src = DATA / "cvrplib" / "XML100"
    out = DATA / "cvrplib" / a.out
    tuning = {f.stem for f in (DATA / "tuning" / "xml100").glob("*.vrp")}
    invalid = set((DATA / "cvrplib" / "XML100_invalid_ref.txt").read_text().split())
    pool = [f for f in sorted(src.glob("*.vrp"))
            if f.stem not in tuning and f.stem not in invalid]
    print(f"{len(list(src.glob('*.vrp')))} XML100 instances, "
          f"{len(tuning)} used for tuning, {len(invalid)} with an invalid reference "
          f"-> {len(pool)} available")

    by_code: dict[str, list] = {}
    for f in pool:
        by_code.setdefault(f.stem.split("_")[1], []).append(f)
    codes = sorted(by_code)
    rng = np.random.default_rng(SEED)
    chosen: list = []
    # one per code in a random order of codes, then round again until n is reached
    while len(chosen) < a.n:
        for c in rng.permutation(codes):
            bucket = by_code[c]
            if bucket:
                chosen.append(bucket.pop(rng.integers(len(bucket))))
                if len(chosen) == a.n:
                    break

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in chosen:
        shutil.copy2(f, out / f.name)
    got = sorted(f.stem for f in chosen)
    assert not (set(got) & tuning), "selection overlaps the tuning set"
    assert not (set(got) & invalid), "selection includes an invalid reference"
    assert len(set(got)) == a.n, "duplicate selected"
    print(f"wrote {a.n} instances to {out} over {len({g.split('_')[1] for g in got})} attribute codes")
    print(f"  e.g. {', '.join(got[:4])}")


if __name__ == "__main__":
    main()
