"""Render a still for each scene in the style of the notebook's plot; ecosystem scenes color trails by species.

Usage: python make_gallery.py [scene ...]   (default: all scenes)
Writes gallery/<scene>.png and caches trajectories in gallery/data/<scene>.npy.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba_array
import numpy as np

from particular import PALETTE, SCENES, simulate

OUT = "gallery"


def trajectory(name):
    os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
    path = os.path.join(OUT, "data", name + ".npy")
    if os.path.exists(path):
        return np.load(path)
    s = SCENES[name]()
    data = simulate(s["pos"], s["vel"], s["steps"], species=s["species"], rules=s["rules"])
    np.save(path, data)
    return data


def species_colors(s):
    """Per-particle RGB (n, 3) from the scene's species palette, or None for a classic scene."""
    if s["species"] is None:
        return None
    return to_rgba_array(s["rules"].get("colors", PALETTE))[s["species"], :3]


def draw(data, half, cmap, path, size=8, dpi=200, rgb=None):
    """Save a still of logged positions data (frames, n, 2) centred on the origin.

    rgb: optional per-particle colors (n, 3), e.g. by species; otherwise color by time with cmap.
    """
    frames = len(data)
    t = np.repeat(np.linspace(0, 1, frames), data.shape[1])
    xy = data.reshape(-1, 2)
    colors = plt.get_cmap(cmap)(t) if rgb is None else np.c_[np.tile(rgb, (frames, 1)), t]
    colors[:, 3] = 1 - t                     # fade out over time, as in the notebook
    order = np.argsort(-t)                   # draw newest first so early trails sit on top

    fig = plt.figure(figsize=[size, size], dpi=dpi, facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1], facecolor="black")
    ax.scatter(xy[order, 0], xy[order, 1], c=colors[order], s=4, linewidths=0)
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(path, facecolor="black")
    plt.close(fig)


def render(name, size=8, dpi=200):
    s = SCENES[name]()
    draw(trajectory(name), s["half"], s["cmap"], os.path.join(OUT, name + ".png"), size, dpi,
         species_colors(s))


if __name__ == "__main__":
    for name in sys.argv[1:] or SCENES:
        render(name)
        print("rendered", name)
