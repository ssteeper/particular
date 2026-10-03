"""Re-run every atlas elite on the CPU and measure it.

The published atlas was simulated on a GPU. The worlds are chaotic, so a CPU run drifts from the GPU
run; this records where each elite lands on CPU. Positions are logged every 5 steps for smooth clips,
and measured on every other frame, which is exactly atlas.py's log_every=10.

Writes data/cpu_measure.json and caches trajectories in work/traj/<cell>.npy.
"""
import json
import os
from multiprocessing import Pool

import numpy as np

from common import ATLAS_JSON, DATA, WORK
import atlas as A
import particular as P

ATLAS = json.load(open(ATLAS_JSON))
EDGES = [ax["edges"] for ax in ATLAS["axes"]]


def run(e):
    g = e["genome"]
    s = P.build(g)
    with np.errstate(all="ignore"):
        d = P.simulate(s["pos"], s["vel"], g["steps"], log_every=5, g=s["g"], c=s["c"],
                       species=s["species"], rules=s["rules"])
    np.save(os.path.join(WORK, "traj", "-".join(map(str, e["cell"])) + ".npy"), d.astype(np.float32))
    out = {"cell": e["cell"], "gpu_fitness": e["fitness"], "gpu_features": e["features"]}
    m = A.measure(d[::2]) if np.isfinite(d).all() else None
    if m:
        out.update(cpu_fitness=m["fitness"], cpu_features=m["features"],
                   cpu_cell=list(A.cell_of(m["features"], EDGES)), terms=m["terms"], half=m["half"])
    return out


if __name__ == "__main__":
    os.makedirs(os.path.join(WORK, "traj"), exist_ok=True)
    with Pool() as pool:
        res = pool.map(run, ATLAS["elites"], chunksize=2)
    json.dump(res, open(os.path.join(DATA, "cpu_measure.json"), "w"), indent=0)
    print(len(res), "elites,", sum(r.get("cpu_cell") == r["cell"] for r in res), "in the same cell on CPU")
