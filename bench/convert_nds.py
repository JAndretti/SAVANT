"""Convert the NDS test pickles (depot, coords, demands, capacity) into .vrp files.

    uv run python -m bench.convert_nds
writes data/nds_vrp/n{N}/nds{N}_{k:05d}.vrp (real-valued EUC_2D, depot = node 1).
"""

import pickle

import numpy as np

from bench.common import DATA, write_vrp

FILES = {100: "vrp100_test_seed1234", 500: "vrp500_test_luo",
         1000: "vrp1000_test_seed1234", 2000: "vrp2000_test_seed1234"}


def main() -> None:
    for n, stem in FILES.items():
        with (DATA / "nds" / f"{stem}.pkl").open("rb") as fh:
            items = pickle.load(fh)
        out = DATA / "nds_vrp" / f"n{n}"
        out.mkdir(parents=True, exist_ok=True)
        for k, (depot, loc, dem, cap) in enumerate(items):
            loc, dem = np.asarray(loc, dtype=np.float64), np.asarray(dem)
            assert loc.shape == (n, 2) and dem.shape == (n,)
            xy = np.vstack([np.asarray(depot, dtype=np.float64)[None], loc])
            write_vrp(out / f"nds{n}_{k:05d}.vrp", f"nds{n}_{k:05d}", xy,
                      np.concatenate([[0], dem]), float(cap),
                      comment="NDS test set (Hottung et al. 2025), converted by bench/convert_nds.py")
        print(f"n={n}: {len(items)} instances -> {out}")


if __name__ == "__main__":
    main()
