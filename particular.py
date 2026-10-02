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


# Five scenes in order of increasing complexity.
def scene_breath():
    # 1. One ring pushed outward with a little spin; it rises and falls back.
    pos, vel = ring(60, 6.0, spin=20.0, radial=1300.0)
    return pos, vel, 1000, 13, "hsv"


def scene_square():
    # 2. Particles along the edges of a square, swirling.
    k = 20
    s = np.linspace(-12, 12, k, endpoint=False)
    edges = [np.c_[s, np.full(k, -12)], np.c_[np.full(k, 12), s],
             np.c_[-s, np.full(k, 12)], np.c_[np.full(k, -12), -s]]
    pos = np.vstack(edges)
    vel = np.c_[-pos[:, 1], pos[:, 0]] * 90
    return pos, vel, 1000, 20, "hsv"


def scene_quartet():
    # 3. Four rings at the corners of a square, alternating spin, the whole set orbiting.
    parts = []
    for k in range(4):
        ang = np.pi / 4 + k * np.pi / 2
        c = (13 * np.cos(ang), 13 * np.sin(ang))
        v = (-900 * np.sin(ang), 900 * np.cos(ang))
        parts.append(ring(28, 4.0, center=c, spin=(1 if k % 2 else -1) * 280.0, vel=v))
    return (*_join(*parts), 1200, 24, "hsv")


def scene_galaxy():
    # 4. Three logarithmic spiral arms around a counter-rotating core.
    arms_p, arms_v = [], []
    for k in range(3):
        u = np.linspace(0, 1, 50)
        rad = 5 * np.exp(1.4 * u)
        th = 2 * np.pi * k / 3 + 2.2 * u
        p = np.c_[rad * np.cos(th), rad * np.sin(th)]
        arms_p.append(p)
        arms_v.append(np.c_[-np.sin(th), np.cos(th)] * (200 + 20 * rad)[:, None])
    core = ring(30, 2.5, spin=-500.0)
    pos = np.vstack(arms_p + [core[0]])
    vel = np.vstack(arms_v + [core[1]])
    return pos, vel, 600, 24, "hsv"


def scene_constellation():
    # 5. A spinning central ring, six spinning rings orbiting it, and a loose outer halo.
    parts = [ring(36, 5.0, spin=-260.0)]
    for k in range(6):
        ang = 2 * np.pi * k / 6
        c = (16 * np.cos(ang), 16 * np.sin(ang))
        v = (-1000 * np.sin(ang), 1000 * np.cos(ang))
        parts.append(ring(22, 3.0, center=c, spin=(1 if k % 2 else -1) * 380.0, vel=v, phase=ang))
    parts.append(ring(90, 26.0, spin=-30.0, radial=-200.0))
    return (*_join(*parts), 1100, 30, "hsv")


# Random setups for search.py. Every setup is built from a seed so it can be rebuilt.
MIN_GAP = 0.8      # closest allowed starting spacing; tighter packing explodes on step one


def _rot(xy, ang):
    c, s = np.cos(ang), np.sin(ang)
    return np.c_[c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]]


def random_setup(seed):
    """Return (pos, vel, description) for a random arrangement with k-fold symmetry."""
    rng = np.random.default_rng(seed)
    while True:                                          # retry until there are enough particles
        pos, vel, desc = _random_pieces(rng)
        if len(pos) >= 40:
            break
    # drop particles that start on top of an earlier one
    gap = np.sqrt(((pos[:, None] - pos[None]) ** 2).sum(-1))
    keep = ~np.any(np.tril(gap < 0.3, -1), axis=1)
    pos, vel = pos[keep], vel[keep]
    if len(pos) > 260:
        idx = np.sort(rng.choice(len(pos), 260, replace=False))
        pos, vel = pos[idx], vel[idx]
    pos -= pos.mean(0)
    vel -= vel.mean(0)                                   # no net drift out of frame
    return pos, vel, desc


def _random_pieces(rng):
    k = int(rng.choice([1, 2, 3, 4, 5, 6], p=[0.1, 0.18, 0.2, 0.2, 0.16, 0.16]))
    pieces, desc = [], []
    for _ in range(int(rng.integers(1, 4))):
        kind = rng.choice(["ring", "satellites", "spokes"], p=[0.4, 0.4, 0.2])
        if kind == "ring":
            radius = rng.uniform(3, 18)
            n = min(int(rng.integers(20, 90)), int(2 * np.pi * radius / MIN_GAP))
            radial = rng.uniform(-400, 1400) if rng.random() < 0.5 else 0.0
            pieces.append(ring(n, radius, spin=rng.uniform(-250, 250), radial=radial,
                               phase=rng.uniform(0, 2 * np.pi)))
            desc.append(f"ring n={n} r={radius:.1f}")
        elif kind == "satellites":
            r = rng.uniform(2, 6)
            n = min(int(rng.integers(8, 36)), int(2 * np.pi * r / MIN_GAP))
            dist, orbit = rng.uniform(8, 20), rng.uniform(-1500, 1500)
            inward, spin = rng.uniform(-500, 500), rng.uniform(-400, 400)
            alternate = rng.random() < 0.4
            off = rng.uniform(0, 2 * np.pi)
            for j in range(k):
                a = off + 2 * np.pi * j / k
                u, t = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
                pieces.append(ring(n, r, center=dist * u, spin=-spin if alternate and j % 2 else spin,
                                   vel=orbit * t - inward * u, phase=a))
            desc.append(f"{k} satellites n={n} r={r:.1f}")
        else:
            m = int(rng.integers(6, 18))
            d = np.linspace(rng.uniform(1, 6), rng.uniform(10, 22), m)
            bend, swirl, radial = rng.uniform(-0.6, 0.6), rng.uniform(-150, 150), rng.uniform(-300, 800)
            off = rng.uniform(0, 2 * np.pi)
            for j in range(k):
                a = off + 2 * np.pi * j / k + bend * (d - d[0]) / d[-1]
                u, t = np.c_[np.cos(a), np.sin(a)], np.c_[-np.sin(a), np.cos(a)]
                pieces.append((d[:, None] * u, (swirl * d)[:, None] * t + radial * u))
            desc.append(f"{k} spokes m={m}")
    pos, vel = _join(*pieces)
    return pos, vel, ", ".join(desc)


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
    "breath": scene_breath,
    "square": scene_square,
    "quartet": scene_quartet,
    "galaxy": scene_galaxy,
    "constellation": scene_constellation,
}
