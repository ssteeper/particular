"""Search random starting setups for patterns worth keeping.

Usage: python search.py [count] [--start SEED] [--workers N] [--top N]

Each candidate is built from a seed with particular.random_setup, simulated
for STEPS steps, and scored at several lengths (the best length wins). Scores:

  fill        share of the frame the visible trails cover
  symmetry    how closely the trail image matches itself rotated by 360/k degrees
  smoothness  how little the particles change direction from one logged frame
              to the next (tangled, chaotic motion turns sharply and often)
  escape      share of particles that fly far outside the frame (penalty)
  clumping    share of the ink piled into the densest 5% of the frame, which is
              high when trails retrace one thin line (penalty)

Writes search/results.json (every candidate, best first), a still per pick in
search/top/, and search/contact.png showing the picks side by side.
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from make_gallery import draw
from particular import random_setup, simulate

STEPS = 1400
LOG_EVERY = 10
LENGTHS = [40, 60, 80, 100, 120, 140]     # prefixes to score, in logged frames
WEIGHTS = {"fill": 1.5, "symmetry": 1.0, "smoothness": 1.0, "escape": -2.0, "clumping": -1.0}
OUT = "search"


def raster(xy, w, half, size):
    """Weighted 2D histogram of points over [-half, half]^2."""
    img, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=size, range=[[-half, half]] * 2, weights=w)
    return img


def score(data):
    """Score each prefix of a trajectory. Returns the best as a dict."""
    best = None
    for frames in LENGTHS:
        d = data[:frames]
        T, N, _ = d.shape
        alpha = 1 - np.linspace(0, 1, T)                 # same fade as the stills
        w = np.repeat(alpha, N)
        xy = d.reshape(-1, 2)
        r = np.hypot(xy[:, 0], xy[:, 1])
        order = np.argsort(r)
        cum = np.cumsum(w[order]) / w.sum()
        half = 1.1 * r[order][np.searchsorted(cum, 0.95)]  # frame holds 95% of the visible ink
        if not np.isfinite(half) or half <= 0:
            continue

        img = raster(xy, w, half, 48)
        fill = (img > 0.25 * w.sum() / img.size).mean()   # faint stray loops don't count
        dense = np.sort(img.ravel())[::-1]
        clump = dense[:dense.size // 20].sum() / dense.sum()  # ink in the densest 5% of cells
        clump = np.clip((clump - 0.3) / 0.3, 0, 1)

        ref = raster(xy, w, half, 96).ravel()
        sym = 0.0
        for k in range(2, 7):
            rot = raster(_rotate(xy, 2 * np.pi / k), w, half, 96).ravel()
            sym = max(sym, ref @ rot / (np.linalg.norm(ref) * np.linalg.norm(rot) + 1e-12))
        sym = np.clip((sym - 0.5) / 0.5, 0, 1)            # rescale so differences show

        step = d[1:] - d[:-1]                            # (T-1, N, 2)
        heading = np.arctan2(step[..., 1], step[..., 0])
        turn = np.abs((np.diff(heading, axis=0) + np.pi) % (2 * np.pi) - np.pi)
        smooth = 1 - (turn * alpha[1:-1, None]).sum() / (alpha[1:-1].sum() * N) / np.pi
        smooth = np.clip((smooth - 0.85) / 0.15, 0, 1)    # almost all runs land in 0.85..1

        far = np.hypot(d[..., 0], d[..., 1]).max(0) > 1.6 * half
        escape = far.mean()

        parts = {"fill": fill, "symmetry": sym, "smoothness": smooth, "escape": escape, "clumping": clump}
        total = sum(WEIGHTS[k] * v for k, v in parts.items())
        if best is None or total > best["score"]:
            best = {"score": total, "steps": frames * LOG_EVERY, "half": half,
                    **{k: round(float(v), 4) for k, v in parts.items()}}
    return best


def _rotate(xy, ang):
    c, s = np.cos(ang), np.sin(ang)
    return np.c_[c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]]


def evaluate(seed):
    pos, vel, desc = random_setup(seed)
    with np.errstate(all="ignore"):
        data = simulate(pos, vel, STEPS, log_every=LOG_EVERY)
    if not np.isfinite(data).all():
        return {"seed": seed, "score": -np.inf, "desc": desc, "n": len(pos)}
    result = score(data) or {"score": -np.inf}
    return {"seed": seed, "desc": desc, "n": len(pos), **result}


def pick(results, count):
    """Best candidates, at most two per arrangement type so the picks differ."""
    seen, picks = {}, []
    for r in results:
        kind = r["desc"].split(" n=")[0] + "|" + str(r["desc"].count(","))
        if seen.get(kind, 0) >= 2:
            continue
        seen[kind] = seen.get(kind, 0) + 1
        picks.append(r)
        if len(picks) == count:
            break
    return picks


def contact_sheet(picks, path, tile=360, cols=4):
    rows = (len(picks) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile, rows * tile))
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
    d = ImageDraw.Draw(sheet)
    for i, r in enumerate(picks):
        im = Image.open(r["image"]).convert("RGB").resize((tile, tile), Image.LANCZOS)
        x, y = (i % cols) * tile, (i // cols) * tile
        sheet.paste(im, (x, y))
        d.text((x + 8, y + 6), f"#{i + 1} seed {r['seed']}  score {r['score']:.2f}", fill=(170, 170, 170), font=font)
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("count", type=int, nargs="?", default=800)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--top", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(os.path.join(OUT, "top"), exist_ok=True)
    seeds = range(args.start, args.start + args.count)
    results = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(evaluate, seeds, chunksize=4)):
            results.append(r)
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{args.count} scored", flush=True)
    results.sort(key=lambda r: -r["score"])

    picks = pick([r for r in results if np.isfinite(r["score"])], args.top)
    for rank, r in enumerate(picks, 1):
        pos, vel, _ = random_setup(r["seed"])
        data = simulate(pos, vel, r["steps"], log_every=LOG_EVERY)
        r["image"] = os.path.join(OUT, "top", f"{rank:02d}-seed{r['seed']}.png")
        draw(data, r["half"], "hsv", r["image"], size=6, dpi=150)
        print(f"#{rank} seed {r['seed']} score {r['score']:.3f}  {r['desc']}")
    contact_sheet(picks, os.path.join(OUT, "contact.png"))

    for r in results:
        if not np.isfinite(r["score"]):
            r["score"] = None
        elif "half" in r:
            r["score"], r["half"] = round(r["score"], 4), round(float(r["half"]), 2)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
