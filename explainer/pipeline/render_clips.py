"""Render every clip on the page into clips/<name>.mp4 with a poster clips/<name>.jpg (the last frame).

Hand-built scenes are simulated here; atlas, random and mutant worlds reuse the trajectories cached in
work/ by the measuring scripts. Also re-encodes video/particular.mp4 to clips/reel.mp4 (540 px).
Needs ffmpeg on PATH.
"""
import json
import os
import subprocess
from multiprocessing import Pool

import numpy as np

from common import CLIPS, DATA, REPO, WORK
from clip import write
from make_gallery import species_colors
import particular as P

SCENES = ["spiral", "breath", "eco_orbit"]
RANDOM = [275, 78, 282, 35, 36, 3, 271]
ELITES = ["0-5-0", "7-5-0", "2-0-1", "2-7-1", "4-4-0", "4-4-5"]
MUTANTS = ["parent", "piece", "structure", "forces", "species", "steps", "crossover"]


def run(job):
    kind, key = job
    size = 400
    if kind == "scene":
        s = P.SCENES[key]()
        d = P.simulate(s["pos"], s["vel"], s["steps"], log_every=5, species=s["species"], rules=s["rules"])
        half, rgb, name = s["half"], species_colors(s), f"scene-{key}"
    elif kind == "random":
        r = {r["i"]: r for r in json.load(open(os.path.join(DATA, "random_measure.json"))) if "fitness" in r}
        d, half, rgb, name = np.load(os.path.join(WORK, "rtraj", f"{key}.npy")), r[key]["half"], None, f"random-{key}"
    elif kind == "elite":
        c = {"-".join(map(str, r["cell"])): r for r in json.load(open(os.path.join(DATA, "cpu_measure.json")))}
        d, half, rgb, name = np.load(os.path.join(WORK, "traj", f"{key}.npy")), c[key]["half"], None, f"elite-{key}"
    else:
        m = json.load(open(os.path.join(DATA, "mutants.json")))
        d, half, rgb, name = np.load(os.path.join(WORK, "mtraj", f"{key}.npy")), m[key]["half"], None, f"mut-{key}"
        size = 320
    write(np.asarray(d, float), half, os.path.join(CLIPS, name + ".mp4"), size=size, rgb=rgb, crf=24,
          poster=os.path.join(CLIPS, name + ".jpg"))
    return name


if __name__ == "__main__":
    jobs = ([("scene", k) for k in SCENES] + [("random", k) for k in RANDOM]
            + [("elite", k) for k in ELITES] + [("mut", k) for k in MUTANTS])
    with Pool() as pool:
        for name in pool.imap_unordered(run, jobs):
            print("rendered", name, flush=True)
    reel = os.path.join(CLIPS, "reel.mp4")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", os.path.join(REPO, "video", "particular.mp4"),
                    "-vf", "scale=540:540", "-c:v", "libx264", "-preset", "slow", "-crf", "28",
                    "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", reel], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", "70", "-i", reel, "-frames:v", "1", "-q:v", "4",
                    os.path.join(CLIPS, "reel.jpg")], check=True)
    print("rendered reel")
