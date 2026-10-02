"""Render a still for each scene in the style of the notebook's plot.

Usage: python make_gallery.py [scene ...]   (default: all scenes)
Writes gallery/<scene>.png and caches trajectories in gallery/data/<scene>.npy.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from particular import SCENES, simulate

OUT = "gallery"


def trajectory(name):
    os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
    path = os.path.join(OUT, "data", name + ".npy")
    if os.path.exists(path):
        return np.load(path)
    pos, vel, steps, _, _ = SCENES[name]()
    data = simulate(pos, vel, steps)
    np.save(path, data)
    return data


def render(name, size=8, dpi=200):
    data = trajectory(name)
    _, _, _, half, cmap = SCENES[name]()
    frames = len(data)
    t = np.repeat(np.linspace(0, 1, frames), data.shape[1])
    xy = data.reshape(-1, 2)
    colors = plt.get_cmap(cmap)(t)
    colors[:, 3] = 1 - t                     # fade out over time, as in the notebook
    order = np.argsort(-t)                   # draw newest first so early trails sit on top

    fig = plt.figure(figsize=[size, size], dpi=dpi, facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1], facecolor="black")
    ax.scatter(xy[order, 0], xy[order, 1], c=colors[order], s=4, linewidths=0)
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(os.path.join(OUT, name + ".png"), facecolor="black")
    plt.close(fig)


if __name__ == "__main__":
    for name in sys.argv[1:] or SCENES:
        render(name)
        print("rendered", name)
