"""Vectorized version of the particle simulation in particles.ipynb.

Same physics as the notebook: every pair of particles feels an attractive
1/r^2 force and a repulsive 1/r^3 force, integrated with semi-implicit Euler.
Scenes are named initial conditions that produce different trail patterns.

Ecosystems: each particle belongs to a species, and a set of rules gives
S x S pull and push multipliers, so particle i (species a) is accelerated by
j (species b) with -g*pull[a][b]/r^2 + c*push[a][b]/r^3 along r_i - r_j.
The matrices may be asymmetric (a chases b while b flees a) and pull may be
negative (long-range repulsion). RULES holds the built-in presets.
"""
import numpy as np

G = 5e6
C = 1e7
DT = 1e-5


def simulate(pos, vel, steps, log_every=10, dt=DT, g=G, c=C, species=None, rules=None, state=False):
    """Run the simulation and return logged positions, shape (frames, n, 2).

    species: int array (n,) of values 0..S-1, or None for all species 0.
    rules: {"pull": S x S, "push": S x S} multipliers, or None for all ones.
    state: also return the final positions and velocities, (log, pos, vel), to continue the run;
    with steps a multiple of log_every, chained runs log the same frames as one long run.
    """
    r = np.array(pos, dtype=float)
    v = np.array(vel, dtype=float)
    n = len(r)
    s = np.zeros(n, int) if species is None else np.asarray(species, int)
    rules = rules or RULES["classic"]
    gp = -g * np.asarray(rules["pull"], float)[s[:, None], s]   # gp[i, j] for i pulled by j
    cp = c * np.asarray(rules["push"], float)[s[:, None], s]
    eye = np.eye(n, dtype=bool)
    log = []
    for i in range(steps):
        d = r[:, None, :] - r[None, :, :]          # d[i, j] = r_i - r_j
        dist = np.sqrt((d ** 2).sum(-1))
        dist[eye] = np.inf
        coef = gp / dist ** 3 + cp / dist ** 4
        a = (coef[:, :, None] * d).sum(1)
        v += a * dt
        r += v * dt
        if i % log_every == 0:
            log.append(r.copy())
    return (np.array(log), r, v) if state else np.array(log)


# Rules of a universe: pull[a][b] and push[a][b] scale how species a reacts to species b.
# The settling distance between a and b is 2 * push[a][b] / pull[a][b].
PALETTE = ["#ff5c7a", "#5cd6ff", "#ffd45c", "#9dff7a"]     # species colors when rules give none
RULES = {
    "classic": {"species": 1, "pull": [[1]], "push": [[1]], "colors": ["#b9c3ff"],
                "note": "One species: the notebook's physics."},
    "chase": {"species": 2, "pull": [[1, -0.2], [1, 1]], "push": [[1, 1], [2.5, 1]],
              "colors": ["#7dffb0", "#ff6a4d"],
              "note": "Predators (red) are drawn to prey (green) and hold off 5 units; prey flee them."},
    "clusters": {"species": 3, "pull": [[1.5, 0.3, 0.3], [0.3, 1.5, 0.3], [0.3, 0.3, 1.5]],
                 "push": [[1, 1.5, 1.5], [1.5, 1, 1.5], [1.5, 1.5, 1]],
                 "colors": PALETTE[:3],
                 "note": "Like attracts like; unlike species barely pull and keep 10 units apart."},
    "orbit": {"species": 3, "pull": [[1, 0.2, 0.2], [6, 0.1, 0.1], [6, 0.1, 0.1]],
              "push": [[1, 1, 1], [4, 0.2, 0.2], [4, 0.2, 0.2]],
              "colors": ["#ffc94d", "#5cc8ff", "#c38cff"],
              "note": "A heavy core (gold) pulls the others hard but hardly feels them, so they circle it."},
    "compete": {"species": 3, "pull": [[1, 0.6, -0.2], [-0.2, 1, 0.6], [0.6, -0.2, 1]],
                "push": [[1, 1.5, 1.5], [1.5, 1, 1.5], [1.5, 1.5, 1]],
                "colors": PALETTE[:3],
                "note": "Rock-paper-scissors: each species chases the next and flees the one before."},
}


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


def scene(pos, vel, steps, half, cmap="hsv", species=None, rules=None):
    """A scene: starting state, run length, view half-width, colormap and optional ecosystem.

    species and rules go straight to simulate(); rules is a RULES entry, so its colors
    give each species its trail color. Classic scenes leave both None.
    """
    return {"pos": pos, "vel": vel, "steps": steps, "half": half, "cmap": cmap,
            "species": species, "rules": rules}


def scene_spiral():
    # The notebook's original start condition.
    pos, vel = ring(50, 15.0, spin=100.0)
    return scene(pos, vel, 1000, 20)


def scene_counter():
    # Two concentric rings spinning in opposite directions.
    return scene(*_join(ring(36, 9.0, spin=140.0), ring(60, 18.0, spin=-70.0, phase=0.05)), 1000, 22)


def scene_bloom():
    # One ring with an outward kick that varies around the circle, giving petals.
    n = 120
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = np.c_[12 * np.cos(th), 12 * np.sin(th)]
    kick = 900 + 700 * np.cos(6 * th)
    vel = np.c_[np.cos(th), np.sin(th)] * kick[:, None] + np.c_[-np.sin(th), np.cos(th)] * 400
    return scene(pos, vel, 900, 22, "twilight_shifted")


def scene_collision():
    # Two spinning rings fly past each other off-centre.
    a = ring(40, 6.0, center=(-18, -5), spin=250.0, vel=(1500, 300))
    b = ring(40, 6.0, center=(18, 5), spin=250.0, vel=(-1500, -300))
    return scene(*_join(a, b), 1200, 26)


def scene_nested():
    # Four rings, alternating spin direction.
    parts = [ring(12 + 10 * k, 4.0 + 4.0 * k, spin=(1 if k % 2 else -1) * 160.0, phase=0.3 * k)
             for k in range(4)]
    return scene(*_join(*parts), 1000, 20)


def scene_wave():
    # A straight line of particles given a sinusoidal sideways push.
    n = 90
    x = np.linspace(-20, 20, n)
    pos = np.c_[x, np.zeros(n)]
    vel = np.c_[np.zeros(n), 1500 * np.sin(2 * np.pi * x / 20)]
    return scene(pos, vel, 700, 30)


def scene_binary():
    # Two rings orbiting a shared centre.
    a = ring(40, 5.0, center=(-10, 0), spin=200.0, vel=(0, -900))
    b = ring(40, 5.0, center=(10, 0), spin=200.0, vel=(0, 900))
    return scene(*_join(a, b), 1100, 18)


def scene_triad():
    # Three small rings thrown around a common centre.
    parts = []
    for k in range(3):
        ang = 2 * np.pi * k / 3
        c = (14 * np.cos(ang), 14 * np.sin(ang))
        v = (-1100 * np.sin(ang), 1100 * np.cos(ang))
        parts.append(ring(30, 4.0, center=c, spin=-300.0, vel=v))
    return scene(*_join(*parts), 1200, 22)


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
    return scene(np.vstack(parts_p), np.vstack(parts_v), 1000, 24, "cool")


# Five scenes in order of increasing complexity.
def scene_breath():
    # 1. One ring pushed outward with a little spin; it rises and falls back.
    pos, vel = ring(60, 6.0, spin=20.0, radial=1300.0)
    return scene(pos, vel, 1000, 13)


def scene_square():
    # 2. Particles along the edges of a square, swirling.
    k = 20
    s = np.linspace(-12, 12, k, endpoint=False)
    edges = [np.c_[s, np.full(k, -12)], np.c_[np.full(k, 12), s],
             np.c_[-s, np.full(k, 12)], np.c_[np.full(k, -12), -s]]
    pos = np.vstack(edges)
    vel = np.c_[-pos[:, 1], pos[:, 0]] * 90
    return scene(pos, vel, 1000, 20)


def scene_quartet():
    # 3. Four rings at the corners of a square, alternating spin, the whole set orbiting.
    parts = []
    for k in range(4):
        ang = np.pi / 4 + k * np.pi / 2
        c = (13 * np.cos(ang), 13 * np.sin(ang))
        v = (-900 * np.sin(ang), 900 * np.cos(ang))
        parts.append(ring(28, 4.0, center=c, spin=(1 if k % 2 else -1) * 280.0, vel=v))
    return scene(*_join(*parts), 1200, 24)


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
    return scene(pos, vel, 600, 24)


def scene_constellation():
    # 5. A spinning central ring, six spinning rings orbiting it, and a loose outer halo.
    parts = [ring(36, 5.0, spin=-260.0)]
    for k in range(6):
        ang = 2 * np.pi * k / 6
        c = (16 * np.cos(ang), 16 * np.sin(ang))
        v = (-1000 * np.sin(ang), 1000 * np.cos(ang))
        parts.append(ring(22, 3.0, center=c, spin=(1 if k % 2 else -1) * 380.0, vel=v, phase=ang))
    parts.append(ring(90, 26.0, spin=-30.0, radial=-200.0))
    return scene(*_join(*parts), 1100, 30)


# Ecosystems: the same physics with species and rules from RULES.
def _groups(k, n, r, dist, orbit, spin, lag=0.0):
    """k rings of n on a circle of radius dist, all orbiting the centre; returns pos, vel, copy index."""
    parts = []
    for j in range(k):
        ang = 2 * np.pi * j / k - lag
        u, t = np.array([np.cos(ang), np.sin(ang)]), np.array([-np.sin(ang), np.cos(ang)])
        parts.append(ring(n, r, center=dist * u, spin=spin, vel=orbit * t))
    return (*_join(*parts), np.repeat(np.arange(k), n))


def scene_eco_chase():
    # Four prey herds, each with a predator pack just behind; the packs drive the herds round and out.
    prey_p, prey_v, _ = _groups(4, 12, 2.0, 11, 800, 60)
    pred_p, pred_v, _ = _groups(4, 6, 1.0, 11, 800, 100, lag=0.5)
    return scene(np.vstack([prey_p, pred_p]), np.vstack([prey_v, pred_v]), 1500, 26,
                 species=np.r_[np.zeros(48, int), np.ones(24, int)], rules=RULES["chase"])


def scene_eco_clusters():
    # Six swirling groups, two of each species; like groups merge, unlike ones keep apart.
    pos, vel, copy = _groups(6, 14, 1.8, 9, 800, -150)
    return scene(pos, vel, 1500, 20, species=copy % 3, rules=RULES["clusters"])


def scene_eco_orbit():
    # A heavy core with two rings in Kepler orbit around it, the outer one going the other way.
    core = _join(ring(6, 1.2, spin=150.0), ring(10, 2.6, spin=150.0, phase=0.3))
    gm = G * RULES["orbit"]["pull"][1][0] * 16          # core's pull on a ring particle
    inner = ring(28, 7.0, spin=np.sqrt(gm / 7) / 7)
    outer = ring(44, 12.0, spin=-np.sqrt(gm / 12) / 12, phase=0.07)
    return scene(*_join(core, inner, outer), 1500, 20,
                 species=np.repeat([0, 1, 2], [16, 28, 44]), rules=RULES["orbit"])


def scene_eco_compete():
    # Six groups, species alternating round a circle; each chases the next and flees the last.
    pos, vel, copy = _groups(6, 12, 1.8, 11, 0, 100)
    return scene(pos, vel, 1500, 28, species=copy % 3, rules=RULES["compete"])


# Random setups for search.py and atlas.py. A setup is a genome: a JSON-friendly dict of the
# pieces' parameters, force scales and species rules. build(genome) turns it into particles, and
# random_setup(seed) builds the genome drawn from that seed, so old seeds still rebuild.
MIN_GAP = 0.8      # closest allowed starting spacing; tighter packing explodes on step one
MAX_PARTICLES = 260
MAX_SPECIES = 3
LOG2_RANGE = (-2.0, 2.0)    # force scales and rule entries, as the web sandbox's log2 sliders

# Ranges each piece gene is drawn from. Mutation clamps to them; "phase" and "off" wrap around.
PIECE_GENES = {
    "ring": {"n": (20, 89), "radius": (3, 18), "spin": (-250, 250), "radial": (-400, 1400),
             "phase": (0, 2 * np.pi)},
    "satellites": {"n": (8, 35), "r": (2, 6), "dist": (8, 20), "orbit": (-1500, 1500),
                   "inward": (-500, 500), "spin": (-400, 400), "off": (0, 2 * np.pi)},
    "spokes": {"m": (6, 17), "d0": (1, 6), "d1": (10, 22), "bend": (-0.6, 0.6),
               "swirl": (-150, 150), "radial": (-300, 800), "off": (0, 2 * np.pi)},
}


def random_setup(seed):
    """Return (pos, vel, description) for a random arrangement with k-fold symmetry."""
    setup = build(random_genome(np.random.default_rng(seed)))
    return setup["pos"], setup["vel"], setup["desc"]


def random_genome(rng, wild=False):
    """Draw a genome. Pieces have k-fold symmetry; forces are the defaults and there is one species.

    wild also draws force scales, 1-3 species, their rules and a species per piece. Those draws
    come after everything a plain genome draws, so the same rng state gives the same pieces.
    Genome keys:
      k            symmetry order of satellites and spokes
      pieces       [{"kind": "ring" | "satellites" | "spokes", <genes in PIECE_GENES>,
                     "species": [a, b]}], satellites also "alternate"; a piece alternates
                     species a and b particle by particle (ring) or copy by copy (others)
      thin         rng state that picks which MAX_PARTICLES survive when there are more
      log2_pull, log2_push   global force scales, G * 2**log2_pull and C * 2**log2_push
      species      number of species S
      log2_rules   {"pull": S x S, "push": S x S}; rules[a][b] scales how a reacts to b
    """
    while True:                                          # retry until there are enough particles
        k = int(rng.choice([1, 2, 3, 4, 5, 6], p=[0.1, 0.18, 0.2, 0.2, 0.16, 0.16]))
        pieces = [_random_piece(rng) for _ in range(int(rng.integers(1, 4)))]
        if sum(_piece_size(p, k) for p in pieces) >= 40:
            break
    bits = rng.bit_generator.state
    genome = {"k": k, "pieces": pieces,
              "thin": [str(bits["state"]["state"]), str(bits["state"]["inc"]),
                       int(bits["has_uint32"]), int(bits["uinteger"])],
              "log2_pull": 0.0, "log2_push": 0.0,
              "species": 1, "log2_rules": {"pull": [[0.0]], "push": [[0.0]]}}
    if wild:
        genome["log2_pull"], genome["log2_push"] = (float(x) for x in rng.uniform(*LOG2_RANGE, 2))
        s = int(rng.integers(1, MAX_SPECIES + 1))
        genome["species"] = s
        genome["log2_rules"] = {key: rng.uniform(*LOG2_RANGE, (s, s)).tolist() if s > 1 else [[0.0]]
                                for key in ("pull", "push")}
        for p in pieces:
            a = int(rng.integers(s))
            p["species"] = [a, int(rng.integers(s)) if rng.random() < 0.5 else a]
    return genome


def _random_piece(rng):
    kind = str(rng.choice(["ring", "satellites", "spokes"], p=[0.4, 0.4, 0.2]))
    if kind == "ring":
        radius = rng.uniform(3, 18)
        n = int(rng.integers(20, 90))
        radial = rng.uniform(-400, 1400) if rng.random() < 0.5 else 0.0
        genes = {"n": n, "radius": radius, "spin": rng.uniform(-250, 250), "radial": radial,
                 "phase": rng.uniform(0, 2 * np.pi)}
    elif kind == "satellites":
        genes = {"r": rng.uniform(2, 6), "n": int(rng.integers(8, 36))}
        genes["dist"], genes["orbit"] = rng.uniform(8, 20), rng.uniform(-1500, 1500)
        genes["inward"], genes["spin"] = rng.uniform(-500, 500), rng.uniform(-400, 400)
        genes["alternate"] = bool(rng.random() < 0.4)
        genes["off"] = rng.uniform(0, 2 * np.pi)
    else:
        genes = {"m": int(rng.integers(6, 18))}
        genes["d0"], genes["d1"] = rng.uniform(1, 6), rng.uniform(10, 22)
        genes["bend"], genes["swirl"], genes["radial"] = (rng.uniform(-0.6, 0.6), rng.uniform(-150, 150),
                                                          rng.uniform(-300, 800))
        genes["off"] = rng.uniform(0, 2 * np.pi)
    return {"kind": kind, **{key: float(v) if isinstance(v, float) else v for key, v in genes.items()},
            "species": [0, 0]}


def _piece_size(p, k):
    """Particle count of a piece before overlaps are dropped; rings are capped so spacing >= MIN_GAP."""
    if p["kind"] == "ring":
        return min(p["n"], int(2 * np.pi * p["radius"] / MIN_GAP))
    if p["kind"] == "satellites":
        return min(p["n"], int(2 * np.pi * p["r"] / MIN_GAP)) * k
    return p["m"] * k


def _piece(p, k):
    """(pos, vel, species, description) of one piece."""
    a, b = p["species"]
    if p["kind"] == "ring":
        n = _piece_size(p, k)
        pos, vel = ring(n, p["radius"], spin=p["spin"], radial=p["radial"], phase=p["phase"])
        return pos, vel, np.where(np.arange(n) % 2, b, a), f"ring n={n} r={p['radius']:.1f}"
    parts, species = [], []
    if p["kind"] == "satellites":
        n = _piece_size(p, 1)
        r, dist, spin = p["r"], p["dist"], p["spin"]
        for j in range(k):
            ang = p["off"] + 2 * np.pi * j / k
            u, t = np.array([np.cos(ang), np.sin(ang)]), np.array([-np.sin(ang), np.cos(ang)])
            parts.append(ring(n, r, center=dist * u, spin=-spin if p["alternate"] and j % 2 else spin,
                              vel=p["orbit"] * t - p["inward"] * u, phase=ang))
            species.append(np.full(n, b if j % 2 else a))
        desc = f"{k} satellites n={n} r={r:.1f}"
    else:
        d = np.linspace(p["d0"], p["d1"], p["m"])
        for j in range(k):
            ang = p["off"] + 2 * np.pi * j / k + p["bend"] * (d - d[0]) / d[-1]
            u, t = np.c_[np.cos(ang), np.sin(ang)], np.c_[-np.sin(ang), np.cos(ang)]
            parts.append((d[:, None] * u, (p["swirl"] * d)[:, None] * t + p["radial"] * u))
            species.append(np.full(p["m"], b if j % 2 else a))
        desc = f"{k} spokes m={p['m']}"
    pos, vel = _join(*parts)
    return pos, vel, np.concatenate(species), desc


def build(genome):
    """Turn a genome into a setup; a pure function of the genome.

    Returns {"pos", "vel", "desc", "species", "rules", "g", "c"}, ready for
    simulate(pos, vel, steps, g=g, c=c, species=species, rules=rules).
    species and rules are None when there is one species.
    """
    built = [_piece(p, genome["k"]) for p in genome["pieces"]]
    pos = np.vstack([b[0] for b in built])
    vel = np.vstack([b[1] for b in built])
    species = np.concatenate([b[2] for b in built])
    # drop particles that start on top of an earlier one
    gap = np.sqrt(((pos[:, None] - pos[None]) ** 2).sum(-1))
    keep = ~np.any(np.tril(gap < 0.3, -1), axis=1)
    pos, vel, species = pos[keep], vel[keep], species[keep]
    if len(pos) > MAX_PARTICLES:
        state, inc, has_uint32, uinteger = genome["thin"]
        bits = np.random.PCG64()
        bits.state = {"bit_generator": "PCG64", "state": {"state": int(state), "inc": int(inc)},
                      "has_uint32": has_uint32, "uinteger": uinteger}
        idx = np.sort(np.random.Generator(bits).choice(len(pos), MAX_PARTICLES, replace=False))
        pos, vel, species = pos[idx], vel[idx], species[idx]
    pos -= pos.mean(0)
    vel -= vel.mean(0)                                   # no net drift out of frame
    single = genome["species"] == 1
    rules = None if single else {key: (2.0 ** np.asarray(m, float)).tolist()
                                 for key, m in genome["log2_rules"].items()}
    return {"pos": pos, "vel": vel, "desc": ", ".join(b[3] for b in built),
            "species": None if single else species, "rules": rules,
            "g": G * 2.0 ** genome["log2_pull"], "c": C * 2.0 ** genome["log2_push"]}


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
    "eco_chase": scene_eco_chase,
    "eco_clusters": scene_eco_clusters,
    "eco_orbit": scene_eco_orbit,
    "eco_compete": scene_eco_compete,
}
