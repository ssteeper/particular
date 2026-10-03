"""Figures and data tables for the page.

  figs/ray-<cell>.png, ray-<cell>-ink.png   the branching measurement for the swirl and the starburst
  figs/atlas-sprite.jpg                      every atlas still, 128 px, 20 per row, in cell order
  data/figs.json                             angular harmonic spectra and ray-map numbers
  data/atlas_meta.json                       per-elite fields the explorer shows
  data/summary.json                          metrics and locality averaged over replicates

Needs measure_atlas.py and measure_random.py to have run (it reads their trajectories).
"""
import collections
import json
import os

import numpy as np
from PIL import Image

from common import ATLAS_JSON, DATA, FIGS, REPO, WORK
import atlas as A

TILE, COLS = 128, 20


def prep(d, half):
    d = np.asarray(d, float)[::2]
    alpha = 1 - np.linspace(0, 1, len(d))
    return A._trails(d, alpha, half, A.RES)


def spectrum(p, w, size=A.RES, rings=16, sectors=64):
    """Energy per angular harmonic 0..HARMONICS, the same sums as atlas._structure."""
    c = p - size / 2
    r = np.hypot(c[:, 0], c[:, 1]) / (size / 2)
    ok = r < 1
    ring = (r[ok] * rings).astype(np.int64)
    sector = ((np.arctan2(c[ok, 1], c[ok, 0]) / (2 * np.pi) + 0.5) * sectors).astype(np.int64) % sectors
    polar = np.bincount(ring * sectors + sector, w[ok], rings * sectors).reshape(rings, sectors)
    polar /= np.sqrt((np.arange(rings) + 0.5) / rings)[:, None]
    energy = (np.abs(np.fft.rfft(polar, axis=1)[:, :A.HARMONICS + 1]) ** 2).sum(0)
    energy[1:] *= 2
    return energy


def raymap(d, half, path):
    """Trail image and edge map: ring edges amber, ray edges teal, cells passing the ring test shaded."""
    p, pw = prep(d, half)
    fine = A._splat(p, pw, A.RES)
    ink = 1 - np.exp(-2 * fine)
    b = A._blur(ink)
    gx, gy = np.zeros_like(b), np.zeros_like(b)
    gx[1:-1] = (b[2:] - b[:-2]) / 2
    gy[:, 1:-1] = (b[:, 2:] - b[:, :-2]) / 2
    rings, sectors = 16, 32
    cell, ux, uy = A._polar(len(b), rings, sectors)
    radial, tangential = gx * ux + gy * uy, gy * ux - gx * uy
    n = rings * sectors + 1
    gain = np.bincount(cell, (tangential ** 2 - radial ** 2).ravel(), n)[:-1].reshape(rings, sectors)
    padded = np.pad(gain, ((A.PERSIST, A.PERSIST), (0, 0)), constant_values=-np.inf)
    kept = np.min([padded[i:i + rings] for i in range(2 * A.PERSIST + 1)], axis=0) > 0
    keptpix = np.r_[kept.ravel(), False][cell].reshape(b.shape)
    t2, r2 = tangential ** 2, radial ** 2
    m = max(t2.max(), r2.max()) or 1
    img = 0.18 * ink[..., None] + np.zeros(3)
    img[keptpix] += np.array([0.10, 0.10, 0.16])
    img += np.sqrt(t2 / m)[..., None] * np.array([0.35, 0.95, 0.85])
    img += np.sqrt(r2 / m)[..., None] * np.array([1.0, 0.70, 0.25])
    img = np.clip(img, 0, 1).transpose(1, 0, 2)[::-1]          # pixel (i, j) is x = i, y = j; put y up
    Image.fromarray((img * 255).astype(np.uint8)).resize((384, 384), Image.NEAREST).save(path)
    inkimg = (np.clip(ink, 0, 1).T[::-1] * 255).astype(np.uint8)
    Image.fromarray(inkimg).resize((384, 384), Image.NEAREST).save(path.replace(".png", "-ink.png"))
    coarse = 1 - np.exp(-fine.reshape(A.RES // 4, 4, A.RES // 4, 4).sum((1, 3)) / 2)
    return {"fine": float(A._rays(ink, 32)), "coarse": float(A._rays(coarse, 16)), "kept_share": float(kept.mean())}


def main():
    atlas = json.load(open(ATLAS_JSON))
    rand = {r["i"]: r for r in json.load(open(os.path.join(DATA, "random_measure.json"))) if "fitness" in r}
    cpu = {"-".join(map(str, r["cell"])): r for r in json.load(open(os.path.join(DATA, "cpu_measure.json")))}

    out = {}
    for i in (35, 36, 282):
        p, pw = prep(np.load(os.path.join(WORK, "rtraj", f"{i}.npy")), rand[i]["half"])
        e = spectrum(p, pw)
        S, k = A._structure(p, pw, A.RES)
        out[f"r{i}"] = {"harmonics": (e[1:] / e.sum()).tolist(), "angular_share": float(e[1:].sum() / e.sum()),
                        "S": S, "k": k, "structure": float(1 - np.exp(-S / A.STRUCTURE))}
    for c in ("4-4-0", "4-4-5"):
        d = np.load(os.path.join(WORK, "traj", f"{c}.npy"))
        out[f"ray-{c}"] = {**raymap(d, cpu[c]["half"], os.path.join(FIGS, f"ray-{c}.png")),
                           "features": cpu[c]["cpu_features"]}
    json.dump(out, open(os.path.join(DATA, "figs.json"), "w"), indent=1)

    elites = sorted(atlas["elites"], key=lambda e: e["cell"])
    rows = (len(elites) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * TILE, rows * TILE))
    meta = []
    for i, e in enumerate(elites):
        im = Image.open(os.path.join(REPO, e["image"])).convert("RGB").resize((TILE, TILE), Image.LANCZOS)
        sheet.paste(im, ((i % COLS) * TILE, (i // COLS) * TILE))
        g = e["genome"]
        meta.append({"c": e["cell"], "f": e["fitness"], "x": e["features"], "t": e["terms"], "d": e["desc"],
                     "n": e["n"], "s": e["steps"], "sp": g["species"], "k": g["k"], "op": e.get("op"),
                     "p": e.get("parent"), "r": e.get("run"),
                     "pull": round(g["log2_pull"], 2), "push": round(g["log2_push"], 2)})
    sheet.save(os.path.join(FIGS, "atlas-sprite.jpg"), quality=78)
    json.dump(meta, open(os.path.join(DATA, "atlas_meta.json"), "w"), separators=(",", ":"))

    ev = atlas["evaluation"]
    summ = {t: {m: float(np.mean([r[m] for r in ev[t]])) for m in ("coverage", "reliability", "precision", "global")}
            for t in ("random", "map_elites")}
    summ["per_rep"] = {t: ev[t] for t in ("random", "map_elites")}
    loc = ev["locality"]
    summ["locality"] = {op: {m: float(np.mean([l[op][m] for l in loc])) for m in loc[0][op]} for op in loc[0]}
    summ["elite_ops"] = dict(collections.Counter(e.get("op") for e in atlas["elites"]))
    json.dump(summ, open(os.path.join(DATA, "summary.json"), "w"), indent=1)
    print("figures and tables written;", rows, "sprite rows")


if __name__ == "__main__":
    main()
