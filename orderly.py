"""How long each atlas world stays orderly, run far past the 1400 steps the atlas scored.

Usage: python orderly.py [--horizon STEPS] [--check K] [--runs R] [--workers N]

Every elite is rebuilt from search/atlas.json and run with numpy, logged every 10 steps as atlas.py
logs it, in chunks of one standard run, until it stops being orderly or reaches --horizon. A world is
orderly at a logged frame if, over the 140 frames (one standard run) ending there:
  - chaos, atlas.measure's mean folded turn of a particle's heading per frame (unweighted here; the
    atlas fades it toward the end of the run), is at most the upper edge of the most orderly chaos bin;
  - at most half the particles are past 1.6 x the elite's published half-width (atlas.py's escape
    distance). Particles flying apart barely turn, so chaos alone would call that orderly.
Windows are checked every 10 frames (100 steps). A world's orderly time is the last step of the last
window that passed before the first that failed: 0 if its first run already fails.

--check re-runs the K longest-lasting worlds R times with start positions perturbed by Gaussian noise of
scale 1e-9 (landscape.py's noise scale), since a CPU run of a chaotic world is one draw.
Writes search/orderly.json.
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

import atlas
from particular import build, simulate

LOG = atlas.LOG_EVERY       # steps per logged frame
WINDOW = 140                # frames: one standard run (1400 steps)
STRIDE = 10                 # frames between window checks
CHUNK = WINDOW * LOG        # steps simulated at a time
ESCAPE = 0.5                # share of particles past atlas.py's escape distance that ends orderliness
EPS = 1e-9


def turns(data):
    """Mean folded heading turn per frame over the particles, as in atlas.measure (frames - 2,)."""
    step = data[1:] - data[:-1]
    heading = np.arctan2(step[..., 1], step[..., 0])
    turn = np.abs((np.diff(heading, axis=0) + np.pi) % (2 * np.pi) - np.pi)
    return np.minimum(turn, np.pi - turn).mean(1)


def run(task):
    """Orderly time of one elite: (cell, run, orderly steps, reason, chaos per window end)."""
    entry, edge, horizon, seed = task
    s = build(entry["genome"])
    pos = np.array(s["pos"], float)
    if seed is not None:
        pos += EPS * np.random.default_rng(seed).standard_normal(pos.shape)
    vel = s["vel"]
    far = 1.6 * entry["half"]
    frames, chaos = [], []
    reason, last = "horizon", 0
    with np.errstate(all="ignore"):
        for start in range(0, horizon, CHUNK):
            log, pos, vel = simulate(pos, vel, CHUNK, log_every=LOG, g=s["g"], c=s["c"],
                                     species=s["species"], rules=s["rules"], state=True)
            frames.append(log)
            data = np.concatenate(frames)
            done = len(data)
            if not np.isfinite(data).all():
                reason = "broke"
                break
            m = turns(data)                      # m[t] is the turn at frame t + 1
            for end in range(max(WINDOW, len(chaos) * STRIDE + WINDOW), done + 1, STRIDE):
                c = float(m[end - WINDOW:end - 2].mean())
                out = float((np.hypot(data[end - 1, :, 0], data[end - 1, :, 1]) > far).mean())
                chaos.append(c)
                if c > edge:
                    reason = "chaos"
                elif out > ESCAPE:
                    reason = "escape"
                else:
                    last = end * LOG
                    continue
                break
            if reason != "horizon":
                break
    return {"cell": entry["cell"], "run": seed, "orderly_steps": last, "reason": reason,
            "steps_run": done * LOG, "chaos": [round(c, 5) for c in chaos]}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--horizon", type=int, default=20 * CHUNK, help="most steps to run each world")
    p.add_argument("--check", type=int, default=5, help="re-run the K longest-lasting worlds perturbed")
    p.add_argument("--runs", type=int, default=8, help="perturbed runs per checked world")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    args = p.parse_args()
    horizon = -(-args.horizon // CHUNK) * CHUNK

    with open(os.path.join("search", "atlas.json"), encoding="utf-8") as f:
        data = json.load(f)
    edge = data["axes"][0]["edges"][0]
    elites = data["elites"]
    with Pool(args.workers) as pool:
        res = pool.map(run, [(e, edge, horizon, None) for e in elites], chunksize=1)
        res.sort(key=lambda r: -r["orderly_steps"])
        top = res[:args.check]
        by_cell = {tuple(e["cell"]): e for e in elites}
        checks = pool.map(run, [(by_cell[tuple(r["cell"])], edge, horizon, k)
                                for r in top for k in range(args.runs)], chunksize=1)

    for r in top:
        mine = [c["orderly_steps"] for c in checks if c["cell"] == r["cell"]]
        r["perturbed_steps"] = mine
        r["perturbed_reasons"] = [c["reason"] for c in checks if c["cell"] == r["cell"]]
    out = {"horizon": horizon, "chaos_edge": edge, "window_steps": WINDOW * LOG, "stride_steps": STRIDE * LOG,
           "escape_share": ESCAPE, "eps": EPS, "elites": res}
    path = os.path.join("search", "orderly.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=0)

    counts = {k: sum(r["reason"] == k for r in res) for k in ("chaos", "escape", "broke", "horizon")}
    print(f"{len(res)} elites, horizon {horizon} steps; ended by {counts}")
    print(f"never orderly over their first {WINDOW * LOG} steps: {sum(r['orderly_steps'] == 0 for r in res)}")
    for r in res[:15]:
        e = by_cell[tuple(r["cell"])]
        extra = f"  perturbed: {r['perturbed_steps']}" if "perturbed_steps" in r else ""
        print(f"{'-'.join(map(str, r['cell'])):6s} {r['orderly_steps']:6d} steps ({r['reason']:7s}) "
              f"n={e['n']:3d} {e['desc']}{extra}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
