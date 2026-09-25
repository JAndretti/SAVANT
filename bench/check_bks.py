"""Sanity check of the checker: every CVRPLIB reference .sol must be feasible and
match the BKS table.  uv run python -m bench.check_bks [X XL XML100]"""

import sys

from bench.common import DATA, check_solution, load_bks, read_cvrplib_sol, read_vrp


def main(sets: list[str]) -> None:
    bks = load_bks()
    bad = 0
    for s in sets:
        sols = sorted((DATA / "cvrplib" / s).glob("*.sol"))
        invalid = []
        for sol in sols:
            inst = read_vrp(sol.with_suffix(".vrp"))
            routes, stated = read_cvrplib_sol(sol)
            cost, ok, msg = check_solution(inst, routes, rounded=True)
            ref = bks.get(inst.name)
            if not ok or stated != cost or (ref is not None and ref != cost):
                bad += 1
                invalid.append(inst.name)
                print(f"MISMATCH {inst.name}: checked={cost} stated={stated} bks={ref} {msg}")
        # instances whose reference solution is not a valid CVRP solution: their
        # BKS is not trustworthy, so they are excluded from tuning and gap tables
        (DATA / "cvrplib" / f"{s}_invalid_ref.txt").write_text("".join(f"{x}\n" for x in invalid))
        print(f"{s}: {len(sols)} reference solutions checked, {len(invalid)} invalid")
    print("all ok" if not bad else f"{bad} mismatches")


if __name__ == "__main__":
    main(sys.argv[1:] or ["X", "XL", "XML100"])
