"""One child per mutation operator from one parent elite (default cell 6-4-3).

For each operator, take the first seed whose child uses that operator and simulates without blowing
up, so the samples are not picked for looks. Writes data/mutants.json and work/mtraj/<op>.npy.

Usage: python mutants.py [cell, e.g. 6-4-3]
"""
import json
import os
import sys

import numpy as np

from common import ATLAS_JSON, DATA, WORK
import atlas as A
import particular as P

OPS = ["piece", "structure", "forces", "species", "steps", "crossover"]


def sim(g):
    s = P.build(g)
    with np.errstate(all="ignore"):
        d = P.simulate(s["pos"], s["vel"], g["steps"], log_every=5, g=s["g"], c=s["c"],
                       species=s["species"], rules=s["rules"])
    return s, d


def record(genome, s, d, edges, **extra):
    m = A.measure(d[::2])
    return {"genome": genome, "desc": s["desc"], "n": len(s["pos"]), "fitness": m["fitness"],
            "features": m["features"], "cell": list(A.cell_of(m["features"], edges)), "half": m["half"],
            "terms": m["terms"], **extra}


def main():
    atlas = json.load(open(ATLAS_JSON))
    edges = [ax["edges"] for ax in atlas["axes"]]
    elites = {tuple(e["cell"]): e for e in atlas["elites"]}
    cell = tuple(map(int, (sys.argv[1] if len(sys.argv) > 1 else "6-4-3").split("-")))
    parent = elites[cell]["genome"]
    near = [c for c in elites if c != cell and max(abs(x - y) for x, y in zip(c, cell)) <= A.CROSS_REACH]
    os.makedirs(os.path.join(WORK, "mtraj"), exist_ok=True)

    s, d = sim(parent)
    np.save(os.path.join(WORK, "mtraj", "parent.npy"), d.astype(np.float32))
    out = {"parent": record(parent, s, d, edges)}
    for op in OPS:
        seed = 0
        while True:
            rng = np.random.default_rng([7, seed])
            seed += 1
            mate = None
            if op == "crossover":
                mate = near[rng.integers(len(near))]
                child, got = A.mutate(parent, rng, elites[mate]["genome"])
            else:
                saved, A.OPS = A.OPS, {op: 1.0}          # force this operator
                try:
                    child, got = A.mutate(parent, rng)
                finally:
                    A.OPS = saved
            if got != op:
                continue
            s, d = sim(child)
            if np.isfinite(d).all() and A.measure(d[::2]) is not None:
                break
        np.save(os.path.join(WORK, "mtraj", f"{op}.npy"), d.astype(np.float32))
        out[op] = record(child, s, d, edges, mate=list(mate) if mate else None, seed=seed - 1)
        print(op, "seed", seed - 1, "->", out[op]["cell"], round(out[op]["fitness"], 3))
    json.dump(out, open(os.path.join(DATA, "mutants.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
