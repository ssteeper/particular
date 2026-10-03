"""Simulate and measure the first 300 random genomes of replicate 0 (atlas.random_genome(0, i)).

These supply the low-fitness examples (no structure, haze, escape) and the single-piece worlds.
Writes data/random_measure.json and caches trajectories in work/rtraj/<i>.npy.
"""
import json
import os
from multiprocessing import Pool

import numpy as np

from common import ATLAS_JSON, DATA, WORK
import atlas as A
import particular as P

COUNT = 300
EDGES = [ax["edges"] for ax in json.load(open(ATLAS_JSON))["axes"]]


def run(i):
    g = A.random_genome(0, i)
    s = P.build(g)
    with np.errstate(all="ignore"):
        d = P.simulate(s["pos"], s["vel"], g["steps"], log_every=5, g=s["g"], c=s["c"],
                       species=s["species"], rules=s["rules"])
    m = A.measure(d[::2]) if np.isfinite(d).all() else None
    if m is None:
        return {"i": i, "broken": True}
    np.save(os.path.join(WORK, "rtraj", f"{i}.npy"), d.astype(np.float32))
    return {"i": i, "fitness": m["fitness"], "features": m["features"], "cell": list(A.cell_of(m["features"], EDGES)),
            "terms": m["terms"], "half": m["half"], "desc": s["desc"], "n": len(s["pos"]),
            "steps": g["steps"], "species": g["species"]}


if __name__ == "__main__":
    os.makedirs(os.path.join(WORK, "rtraj"), exist_ok=True)
    with Pool() as pool:
        res = pool.map(run, range(COUNT), chunksize=2)
    json.dump(res, open(os.path.join(DATA, "random_measure.json"), "w"), indent=0)
    print(sum("fitness" in r for r in res), "of", COUNT, "measured")
