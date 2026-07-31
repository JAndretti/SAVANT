#!/usr/bin/env python3
"""Exhaustive local optimum (relocate + swap + intra 2-opt + 2-opt*) computed
naively in Python, to compare against the C annealer's descent."""
import math, subprocess, sys
from check import read_bundle, cw_naive

def cost(routes, d):
    return sum(d(0, r[0]) + sum(d(r[i], r[i+1]) for i in range(len(r)-1)) + d(r[-1], 0)
               for r in routes if r)

def local_opt(routes, ds, cap, d):
    routes = [list(r) for r in routes] + [[]]
    改 = True
    while 改:
        改 = False
        load = [sum(ds[c] for c in r) for r in routes]
        # relocate
        for a, ra in enumerate(routes):
            for i in range(len(ra)):
                u = ra[i]
                for b, rb in enumerate(routes):
                    if a != b and load[b] + ds[u] > cap + 1e-9: continue
                    for j in range(len(rb) + 1):
                        if a == b and j in (i, i+1): continue
                        new = [list(x) for x in routes]
                        new[a].pop(i)
                        tgt = new[b] if a != b else new[a]
                        k = j if (a != b or j < i) else j - 1
                        tgt.insert(k, u)
                        if cost(new, d) < cost(routes, d) - 1e-10:
                            routes, 改 = new, True
                            load = [sum(ds[c] for c in r) for r in routes]
                            break
                    if 改: break
                if 改: break
            if 改: break
        if 改: continue
        # swap
        for a, ra in enumerate(routes):
            for i, u in enumerate(ra):
                for b, rb in enumerate(routes):
                    for j, v in enumerate(rb):
                        if (a, i) >= (b, j): continue
                        if a != b and (load[a]-ds[u]+ds[v] > cap+1e-9 or
                                       load[b]-ds[v]+ds[u] > cap+1e-9): continue
                        new = [list(x) for x in routes]
                        new[a][i], new[b][j] = v, u
                        if cost(new, d) < cost(routes, d) - 1e-10:
                            routes, 改 = new, True; break
                    if 改: break
                if 改: break
            if 改: break
        if 改: continue
        # 2-opt intra
        for a, ra in enumerate(routes):
            for i in range(len(ra)):
                for j in range(i+1, len(ra)):
                    new = [list(x) for x in routes]
                    new[a][i:j+1] = new[a][i:j+1][::-1]
                    if cost(new, d) < cost(routes, d) - 1e-10:
                        routes, 改 = new, True; break
                if 改: break
            if 改: break
        if 改: continue
        # 2-opt* (tail exchange)
        for a in range(len(routes)):
            for b in range(a+1, len(routes)):
                for i in range(len(routes[a])+1):
                    for j in range(len(routes[b])+1):
                        A, B = routes[a], routes[b]
                        nA, nB = A[:i] + B[j:], B[:j] + A[i:]
                        if sum(ds[c] for c in nA) > cap+1e-9: continue
                        if sum(ds[c] for c in nB) > cap+1e-9: continue
                        new = [list(x) for x in routes]; new[a], new[b] = nA, nB
                        if cost(new, d) < cost(routes, d) - 1e-10:
                            routes, 改 = new, True; break
                    if 改: break
                if 改: break
            if 改: break
    return cost(routes, d), [r for r in routes if r]

def cw_routes(n, cap, xs, ys, ds):
    """replays C&W, keeping the routes"""
    d = lambda a, b: math.hypot(xs[a]-xs[b], ys[a]-ys[b])
    sav = sorted(((d(0,i)+d(0,j)-d(i,j), i, j)
                  for i in range(1, n+1) for j in range(i+1, n+1)
                  if d(0,i)+d(0,j)-d(i,j) > 0), key=lambda t: -t[0])
    route = {i: [i] for i in range(1, n+1)}
    where = {i: i for i in range(1, n+1)}
    load = {i: ds[i] for i in range(1, n+1)}
    for s, i, j in sav:
        ri, rj = where[i], where[j]
        if ri == rj or load[ri] + load[rj] > cap + 1e-9: continue
        A, B = route[ri], route[rj]
        if   A[-1] == i and B[0]  == j: new = A + B
        elif A[0]  == i and B[-1] == j: new = B + A
        elif A[-1] == i and B[-1] == j: new = A + B[::-1]
        elif A[0]  == i and B[0]  == j: new = A[::-1] + B
        else: continue
        route[ri] = new; load[ri] += load[rj]
        for c in B: where[c] = ri
        del route[rj]
    return list(route.values()), d

if __name__ == "__main__":
    insts = read_bundle(sys.argv[1], int(sys.argv[2]))
    for k, (n, cap, xs, ys, ds) in enumerate(insts):
        R, d = cw_routes(n, cap, xs, ys, ds)
        c0 = cost(R, d)
        c1, _ = local_opt(R, ds, cap, d)
        out = subprocess.run(["./cw", "--bundle", sys.argv[1], "--limit", str(k+1),
                              "-q", "--sa-steps", "400000", "--t0", "1e-9", "--tend",
                              "1e-9", "--ops", "1,1,1", "--sa-knn", "0",
                              "--csv", "/tmp/d.csv"], capture_output=True, text=True)
        cC = float(open("/tmp/d.csv").read().splitlines()[k+1].split(",")[4])
        print(f"inst {k}: C&W={c0:.5f}  python local optimum={c1:.5f}  C descent={cC:.5f}")
