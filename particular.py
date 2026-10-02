"""Vectorized version of the particle simulation in particles.ipynb.

Same physics as the notebook: every pair of particles feels an attractive
1/r^2 force and a repulsive 1/r^3 force, integrated with semi-implicit Euler.
Scenes are named initial conditions that produce different trail patterns.
"""
import numpy as np

G = 5e6
C = 1e7
DT = 1e-5


def simulate(pos, vel, steps, log_every=10, dt=DT, g=G, c=C):
    """Run the simulation and return logged positions, shape (frames, n, 2)."""
    r = np.array(pos, dtype=float)
    v = np.array(vel, dtype=float)
    n = len(r)
    eye = np.eye(n, dtype=bool)
    log = []
    for i in range(steps):
        d = r[:, None, :] - r[None, :, :]          # d[i, j] = r_i - r_j
        dist = np.sqrt((d ** 2).sum(-1))
        dist[eye] = np.inf
        coef = -g / dist ** 3 + c / dist ** 4
        a = (coef[:, :, None] * d).sum(1)
        v += a * dt
        r += v * dt
        if i % log_every == 0:
            log.append(r.copy())
    return np.array(log)


def ring(n, radius, center=(0, 0), spin=100.0, radial=0.0, phase=0.0, vel=(0, 0)):
    """Particles on a circle. spin scales tangential speed by radius (as in the notebook)."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False) + phase
    cx, cy = center
    pos = np.c_[cx + radius * np.cos(th), cy + radius * np.sin(th)]
    tang = np.c_[-np.sin(th), np.cos(th)] * spin * radius
    rad = np.c_[np.cos(th), np.sin(th)] * radial
    return pos, tang + rad + np.array(vel, dtype=float)


def _join(*parts):
    return np.vstack([p for p, _ in parts]), np.vstack([v for _, v in parts])


# Each scene returns (pos, vel, steps, view_half_width, colormap).
def scene_spiral():
    # The notebook's original start condition.
    pos, vel = ring(50, 15.0, spin=100.0)
    return pos, vel, 1000, 20, "hsv"


def scene_counter():
    # Two concentric rings spinning in opposite directions.
    return (*_join(ring(36, 9.0, spin=140.0), ring(60, 18.0, spin=-70.0, phase=0.05)),
            1000, 22, "hsv")


def scene_bloom():
    # One ring with an outward kick that varies around the circle, giving petals.
    n = 120
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = np.c_[12 * np.cos(th), 12 * np.sin(th)]
    kick = 900 + 700 * np.cos(6 * th)
    vel = np.c_[np.cos(th), np.sin(th)] * kick[:, None] + np.c_[-np.sin(th), np.cos(th)] * 400
    return pos, vel, 900, 22, "twilight_shifted"


def scene_collision():
    # Two spinning rings fly past each other off-centre.
    a = ring(40, 6.0, center=(-18, -5), spin=250.0, vel=(1500, 300))
    b = ring(40, 6.0, center=(18, 5), spin=250.0, vel=(-1500, -300))
    return (*_join(a, b), 1200, 26, "hsv")


def scene_nested():
    # Four rings, alternating spin direction.
    parts = [ring(12 + 10 * k, 4.0 + 4.0 * k, spin=(1 if k % 2 else -1) * 160.0, phase=0.3 * k)
             for k in range(4)]
    return (*_join(*parts), 1000, 20, "hsv")


def scene_wave():
    # A straight line of particles given a sinusoidal sideways push.
    n = 90
    x = np.linspace(-20, 20, n)
    pos = np.c_[x, np.zeros(n)]
    vel = np.c_[np.zeros(n), 1500 * np.sin(2 * np.pi * x / 20)]
    return pos, vel, 700, 30, "hsv"


def scene_binary():
    # Two rings orbiting a shared centre.
    a = ring(40, 5.0, center=(-10, 0), spin=200.0, vel=(0, -900))
    b = ring(40, 5.0, center=(10, 0), spin=200.0, vel=(0, 900))
    return (*_join(a, b), 1100, 18, "hsv")


def scene_triad():
    # Three small rings thrown around a common centre.
    parts = []
    for k in range(3):
        ang = 2 * np.pi * k / 3
        c = (14 * np.cos(ang), 14 * np.sin(ang))
        v = (-1100 * np.sin(ang), 1100 * np.cos(ang))
        parts.append(ring(30, 4.0, center=c, spin=-300.0, vel=v))
    return (*_join(*parts), 1200, 22, "hsv")


def scene_star():
    # Five spokes of particles launched with a swirl.
    parts_p, parts_v = [], []
    for k in range(5):
        ang = 2 * np.pi * k / 5
        d = np.linspace(4, 20, 16)
        p = np.c_[d * np.cos(ang), d * np.sin(ang)]
        v = np.c_[-np.sin(ang), np.cos(ang)] * (90 * d)[:, None]
        parts_p.append(p)
        parts_v.append(v)
    return np.vstack(parts_p), np.vstack(parts_v), 1000, 24, "cool"


SCENES = {
    "spiral": scene_spiral,
    "counter": scene_counter,
    "bloom": scene_bloom,
    "collision": scene_collision,
    "nested": scene_nested,
    "wave": scene_wave,
    "binary": scene_binary,
    "triad": scene_triad,
    "star": scene_star,
}
