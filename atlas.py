"""An atlas of possible worlds: MAP-Elites over random setups (see research/map-elites-notes.md).

Usage: python atlas.py [evals] [--init N] [--replicates R] [--workers N] [--inflight N]
                       [--device cpu|cuda|auto] [--batch N] [--out DIR]

Instead of ranking setups by one score, keep the best setup in every cell of a 3D map of
behaviour, measured on the trajectory:

  chaos      orderly -> chaotic: mean turn of a particle's heading per logged frame, folded so a
             reversal (pi) counts like going straight (radians, faded like the stills); calm
             orbits and back-and-forth rays barely turn, tangles turn sharply and often
  density    sparse -> dense: share of the frame the visible trails cover (search.py's fill)
  branching  circular -> branching: how much of the trail image's edge energy belongs to lines
             pointing at the centre. Ink gradients are split into radial and tangential parts and
             summed per polar cell (16 rings); a cell counts tangential minus radial energy only
             where that stays positive over 5 consecutive rings, so short ticks on a ring don't
             count, only rays. Read at 128 px (32 sectors) and 32 px (16 sectors), the larger kept.
             Rings, discs, rosettes and separate circles score near 0; spokes, starbursts, crosses
             and arms high

Fitness rates how well a world is drawn, from things that are not axes:
  fitness = max(0, 1 - 2 * escape) * (0.4 + 0.6 * structure) * (0.4 + 0.6 * crispness)
            * (frames / 140) ** 0.2
  structure  macro-scale k-fold symmetry. Of the trail image's energy in angular harmonics 1..12
             (per radius ring), the share on multiples of the best k = 2..6 above chance, times the
             angular share of all the energy: S, mapped to 1 - exp(-S / 0.3). Only low harmonics,
             so the n-fold ripple of n discrete particles earns nothing
  crispness  fine detail: the share of the 128 px trail image's variance that a 5x5 blur removes,
             mapped 0.2 -> 0 and 0.9 -> 1; thin separate trails score high, haze and blobs low
  duration   (frames / 140) ** 0.2: longer runs score higher, so the map prefers worlds that keep
             developing over a short initial expansion
  escape     share of particles that ever go past 1.6 x the frame's half-width
search.py's fill and smoothness are axes, and its clumping penalty is left out because it is mostly
sparseness again (correlation -0.88 with fill over random genomes), so it would score density twice.
Genomes come from particular.random_genome(wild=True) plus a run length.

Each replicate evaluates `evals` random genomes (the random-sampling control) and runs MAP-Elites
for the same number of evaluations, seeded with the first --init of those random genomes. Bin
edges are fixed from replicate 0's random genomes. Writes atlas.json (axes, edges, elites with
genomes, metrics), a still per elite in atlas/ and atlas.png, the map as a grid, to --out (search/).

--device cuda simulates on the GPU (gpu_sim.py, needs torch with CUDA): MAP-Elites in batches of
--batch children, random sampling in batches of GPU_SAMPLE_BATCH, while the CPU pool builds setups
and scores trajectories. auto, the default, uses the GPU when torch can see one. A genome simulates
the same way every time on one device, but the devices round differently, so the same seeds give
maps that differ in detail (see research/gpu-notes.md).
"""
import argparse
import copy
import glob
import json
import os
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw

import particular as P
from make_gallery import draw
from search import LOG_EVERY, load_font, raster

OUT = "search"
AXES = [
    {"name": "chaos", "label": "orderly → chaotic", "bins": 8},
    {"name": "density", "label": "sparse → dense", "bins": 8},
    {"name": "branching", "label": "circular → branching", "bins": 6},
]
STEP_CHOICES = list(range(400, 1401, 200))   # run lengths; the run is scored as a whole
SIGMA = 0.06            # piece gene mutation, as a share of the gene's range
SIGMA_LOG2 = 0.25       # force scale and rule mutation, in log2 units
OPS = {"piece": 0.45, "structure": 0.1, "forces": 0.15, "species": 0.15, "steps": 0.15}
CROSSOVER = 0.15        # chance a child takes a piece from an elite at most CROSS_REACH cells away
CROSS_REACH = 2
GPU_STREAMS = 4         # GPU batches simulating at once; a batch's slowest world sets its time
GPU_SAMPLE_BATCH = 256  # GPU batch for random sampling, which has no selection to keep fresh


# ---- measuring a trajectory ----
# measure() draws the trails the way the stills do (faded with age) and reads the picture: radial
# line energy (branching), low angular harmonics (structure) and fine-scale variance (crispness).

RES = 128                   # trail image, pixels across the frame
HARMONICS = 12              # angular harmonics the structure term looks at
STRUCTURE = 0.3             # structure = 1 - exp(-S / STRUCTURE)
CRISP = (0.2, 0.9)          # detail share mapped to crispness 0 and 1
FLOOR = 0.4                 # weight a world keeps with no structure, or no crispness
DURATION = 0.2              # fitness *= (frames / 140) ** DURATION
PERSIST = 2                 # a ray must stay radial over 2 * PERSIST + 1 rings
_POLAR = {}


def _trails(data, alpha, half, size, jump=0.1):
    """Points one pixel apart along every particle's path, in pixel coordinates of a size x size
    frame, weighted fade x spacing: a trail adds about its faded length to each pixel it crosses.
    A jump longer than jump * size (a particle flung across the frame) stays a dot, as in the stills."""
    T, N, _ = data.shape
    u = (data + half) * (size / (2 * half))
    a, d = u[:-1].reshape(-1, 2), (u[1:] - u[:-1]).reshape(-1, 2)
    length = np.hypot(d[:, 0], d[:, 1])
    far = length > jump * size
    count = np.where(far, 1, np.maximum(np.ceil(length), 1)).astype(np.int64)
    seg = np.repeat(np.arange(len(a)), count)
    f = (np.arange(len(seg)) - (np.cumsum(count) - count)[seg]) / count[seg]
    f[far[seg]] = 0.0
    w = (np.repeat(alpha[:-1], N) * np.where(far, 1.0, length / count))[seg]
    return a[seg] + d[seg] * f[:, None], w


def _splat(p, w, size):
    ij = p.astype(np.int64)
    ok = (p >= 0).all(1) & (ij < size).all(1)
    return np.bincount(ij[ok, 0] * size + ij[ok, 1], w[ok], size * size).reshape(size, size)


def _blur(img):
    p = np.pad(img, 1)
    img = (p[:-2] + 2 * p[1:-1] + p[2:])[:, 1:-1] / 4
    p = np.pad(img, 1)
    return (p[:, :-2] + 2 * p[:, 1:-1] + p[:, 2:])[1:-1] / 4


def _polar(size, rings, sectors):
    """Polar cell of every pixel (rings * sectors = outside) and the radial unit vector, cached."""
    key = (size, rings, sectors)
    if key not in _POLAR:
        x = np.arange(size) - (size / 2 - 0.5)
        X, Y = np.meshgrid(x, x, indexing="ij")
        r = np.hypot(X, Y)
        ring = np.minimum((r / (size / 2) * rings).astype(int), rings)
        sector = ((np.arctan2(Y, X) / (2 * np.pi) + 0.5) * sectors).astype(int) % sectors
        cell = np.where(ring < rings, ring * sectors + sector, rings * sectors).ravel()
        r = np.maximum(r, 1e-9)
        _POLAR[key] = (cell, X / r, Y / r)
    return _POLAR[key]


def _rays(ink, sectors, rings=16):
    """Share of the image's edge energy in lines pointing at the centre that persist over rings."""
    b = _blur(ink)
    gx, gy = np.zeros_like(b), np.zeros_like(b)
    gx[1:-1] = (b[2:] - b[:-2]) / 2
    gy[:, 1:-1] = (b[:, 2:] - b[:, :-2]) / 2
    cell, ux, uy = _polar(len(b), rings, sectors)
    radial, tangential = gx * ux + gy * uy, gy * ux - gx * uy
    n = rings * sectors + 1
    gain = np.bincount(cell, (tangential ** 2 - radial ** 2).ravel(), n)[:-1].reshape(rings, sectors)
    total = np.bincount(cell, (tangential ** 2 + radial ** 2).ravel(), n)[:-1].sum()
    if total <= 0:
        return 0.0
    padded = np.pad(gain, ((PERSIST, PERSIST), (0, 0)), constant_values=-np.inf)
    gain = np.min([padded[i:i + rings] for i in range(2 * PERSIST + 1)], axis=0)
    return float(np.maximum(gain, 0).sum() / total)


def _structure(p, w, size, rings=16, sectors=64):
    """Symmetric angular structure S = a * s: a is the share of the image's energy in angular
    harmonics 1..HARMONICS (per radius ring, area-weighted), s how far above chance that angular
    energy sits on multiples of the best k = 2..6. Returns (S, k)."""
    c = p - size / 2
    r = np.hypot(c[:, 0], c[:, 1]) / (size / 2)
    ok = r < 1
    ring = (r[ok] * rings).astype(np.int64)
    sector = ((np.arctan2(c[ok, 1], c[ok, 0]) / (2 * np.pi) + 0.5) * sectors).astype(np.int64) % sectors
    polar = np.bincount(ring * sectors + sector, w[ok], rings * sectors).reshape(rings, sectors)
    polar /= np.sqrt((np.arange(rings) + 0.5) / rings)[:, None]
    energy = (np.abs(np.fft.rfft(polar, axis=1)[:, :HARMONICS + 1]) ** 2).sum(0)
    energy[1:] *= 2
    angular = energy[1:].sum()
    if angular <= 0:
        return 0.0, 1
    m = np.arange(1, HARMONICS + 1)
    best, best_k = 0.0, 1
    for k in range(2, 7):
        chance = (m % k == 0).mean()
        s = (energy[1:][m % k == 0].sum() / angular - chance) / (1 - chance)
        if s > best:
            best, best_k = s, k
    return float(angular / energy.sum() * best), best_k


def measure(data):
    """Behaviour features and fitness of a whole trajectory (frames, n, 2), or None."""
    T, N, _ = data.shape
    alpha = 1 - np.linspace(0, 1, T)                     # same fade as the stills
    w = np.repeat(alpha, N)
    xy = data.reshape(-1, 2)
    r = np.hypot(xy[:, 0], xy[:, 1])
    order = np.argsort(r)
    cum = np.cumsum(w[order]) / w.sum()
    half = 1.1 * r[order][np.searchsorted(cum, 0.95)]    # frame holds 95% of the visible ink
    if not np.isfinite(half) or half <= 0:
        return None

    img = raster(xy, w, half, 48)
    fill = (img > 0.25 * w.sum() / img.size).mean()       # faint stray loops don't count

    step = data[1:] - data[:-1]
    heading = np.arctan2(step[..., 1], step[..., 0])
    turn = np.abs((np.diff(heading, axis=0) + np.pi) % (2 * np.pi) - np.pi)
    turn = np.minimum(turn, np.pi - turn)                 # a reversal along a ray is not a turn
    chaos = (turn * alpha[1:-1, None]).sum() / (alpha[1:-1].sum() * N) + 1e-6

    escape = (np.hypot(data[..., 0], data[..., 1]).max(0) > 1.6 * half).mean()

    p, pw = _trails(data, alpha, half, RES)
    fine = _splat(p, pw, RES)                             # trail length per pixel
    ink = 1 - np.exp(-2 * fine)                           # overlapping trails saturate, as alpha does
    coarse = 1 - np.exp(-fine.reshape(RES // 4, 4, RES // 4, 4).sum((1, 3)) / 2)
    branching = max(_rays(ink, 32), _rays(coarse, 16))

    var = ((ink - ink.mean()) ** 2).sum()
    detail = 1 - ((_blur(_blur(ink)) - ink.mean()) ** 2).sum() / var if var > 0 else 0.0
    crisp = min(max((detail - CRISP[0]) / (CRISP[1] - CRISP[0]), 0.0), 1.0)
    S, k = _structure(p, pw, RES)
    structure = 1 - np.exp(-S / STRUCTURE)
    duration = (T / 140) ** DURATION

    fitness = (max(0.0, 1 - 2 * escape) * (FLOOR + (1 - FLOOR) * structure)
               * (FLOOR + (1 - FLOOR) * crisp) * duration)
    return {"features": [float(chaos), float(fill), float(branching)], "fitness": float(fitness),
            "half": float(half),
            "terms": {"structure": round(float(structure), 4), "order": k, "crispness": round(crisp, 4),
                      "duration": round(duration, 4), "escape": round(float(escape), 4)}}


def setup_of(genome):
    s = P.build(genome)
    return s, {"g": s["g"], "c": s["c"], "species": s["species"], "rules": s["rules"]}


def evaluate(task):
    """task = (genome, meta); returns meta plus the genome's measurements (fitness None if broken)."""
    genome, meta = task
    s, physics = setup_of(genome)
    with np.errstate(all="ignore"):
        data = P.simulate(s["pos"], s["vel"], genome["steps"], log_every=LOG_EVERY, **physics)
    return score((genome, meta, s["desc"], data))


def score(task):
    """task = (genome, meta, desc, trajectory); evaluate's result for a trajectory simulated elsewhere."""
    genome, meta, desc, data = task
    m = measure(data) if np.isfinite(data).all() else None
    return {**meta, "genome": genome, "desc": desc, "n": data.shape[1], **(m or {"fitness": None})}


def simulate_gpu(pool, genomes, device):
    """(setups, trajectories) of genomes: built in the pool, then simulated together on the GPU."""
    import gpu_sim
    setups = pool.map(P.build, genomes, chunksize=4)
    data = gpu_sim.simulate_batch([{**s, "steps": g["steps"]} for s, g in zip(setups, genomes)],
                                  log_every=LOG_EVERY, device=device)
    return setups, data


def random_genome(seed, i):
    rng = np.random.default_rng([seed, i])
    genome = P.random_genome(rng, wild=True)
    genome["steps"] = int(rng.choice(STEP_CHOICES))
    return genome


# ---- variation ----

def _clamp(value, lo, hi, wrap):
    if wrap:
        return float(lo + (value - lo) % (hi - lo))
    return float(min(max(value, lo), hi))


def _mutate_piece(p, rng):
    for gene, (lo, hi) in P.PIECE_GENES[p["kind"]].items():
        v = _clamp(p[gene] + rng.normal(0, SIGMA * (hi - lo)), lo, hi, gene in ("phase", "off"))
        p[gene] = int(round(v)) if isinstance(p[gene], int) else v


def _new_piece(rng, s):
    p = P._random_piece(rng)
    a = int(rng.integers(s))
    p["species"] = [a, int(rng.integers(s)) if rng.random() < 0.5 else a]
    return p


def _mutate_structure(g, rng):
    pieces, choices = g["pieces"], ["resample", "k"]
    if len(pieces) < 4:
        choices.append("add")
    if len(pieces) > 1:
        choices.append("remove")
    if any(p["kind"] == "satellites" for p in pieces):
        choices.append("alternate")
    what = rng.choice(choices)
    if what == "resample":
        pieces[rng.integers(len(pieces))] = _new_piece(rng, g["species"])
    elif what == "k":
        g["k"] = int(np.clip(g["k"] + rng.choice([-1, 1]), 1, 6))
    elif what == "add":
        pieces.append(_new_piece(rng, g["species"]))
    elif what == "remove":
        pieces.pop(rng.integers(len(pieces)))
    else:
        sats = [p for p in pieces if p["kind"] == "satellites"]
        p = sats[rng.integers(len(sats))]
        p["alternate"] = not p["alternate"]


def _mutate_species(g, rng):
    s, rules = g["species"], g["log2_rules"]
    choices = (["add"] if s < P.MAX_SPECIES else []) + (["rule", "assign", "remove"] if s > 1 else [])
    what = rng.choice(choices)
    if what == "rule":                                   # nudge every entry of one matrix
        key = rng.choice(["pull", "push"])
        m = np.asarray(rules[key]) + rng.normal(0, SIGMA_LOG2, (s, s))
        rules[key] = np.clip(m, *P.LOG2_RANGE).tolist()
    elif what == "assign":
        p = g["pieces"][rng.integers(len(g["pieces"]))]
        p["species"][rng.integers(2)] = int(rng.integers(s))
    elif what == "add":                                  # a new species with mild rules, given to one piece slot
        for key in ("pull", "push"):
            m = np.zeros((s + 1, s + 1)) if s == 1 else np.pad(np.asarray(rules[key]), (0, 1))
            m[s, :] = rng.normal(0, 0.5, s + 1)
            m[:, s] = rng.normal(0, 0.5, s + 1)
            rules[key] = np.clip(m, *P.LOG2_RANGE).tolist()
        g["species"] = s + 1
        p = g["pieces"][rng.integers(len(g["pieces"]))]
        p["species"][rng.integers(2)] = s
    else:                                                # drop one species; its pieces join another
        gone = int(rng.integers(s))
        keep = [i for i in range(s) if i != gone]
        for key in ("pull", "push"):
            rules[key] = np.asarray(rules[key])[np.ix_(keep, keep)].tolist() if s > 2 else [[0.0]]
        g["species"] = s - 1
        for p in g["pieces"]:
            p["species"] = [int(rng.integers(s - 1)) if a == gone else a - (a > gone) for a in p["species"]]


def mutate(genome, rng, mate=None):
    """Return (child, operator name). mate, if given, donates one piece (crossover)."""
    for _ in range(20):                                  # retry until the child has enough particles
        g = copy.deepcopy(genome)
        if mate is not None:
            op = "crossover"
            p = copy.deepcopy(mate["pieces"][rng.integers(len(mate["pieces"]))])
            p["species"] = [min(a, g["species"] - 1) for a in p["species"]]
            g["pieces"][rng.integers(len(g["pieces"]))] = p
        else:
            op = str(rng.choice(list(OPS), p=list(OPS.values())))
            if op == "piece":
                _mutate_piece(g["pieces"][rng.integers(len(g["pieces"]))], rng)
            elif op == "structure":
                _mutate_structure(g, rng)
            elif op == "forces":
                for key in ("log2_pull", "log2_push"):
                    g[key] = _clamp(g[key] + rng.normal(0, SIGMA_LOG2), *P.LOG2_RANGE, False)
            elif op == "species":
                _mutate_species(g, rng)
            else:
                g["steps"] = int(np.clip(g["steps"] + rng.choice([-200, 200]), STEP_CHOICES[0], STEP_CHOICES[-1]))
        if sum(P._piece_size(p, g["k"]) for p in g["pieces"]) >= 40:
            return g, op
    return copy.deepcopy(genome), "none"


# ---- the map ----

def cell_of(features, edges):
    return tuple(int(np.searchsorted(e, v)) for v, e in zip(features, edges))


def calibrate(results):
    """Interior bin edges per axis: evenly spaced between the 1st and 99th percentile of the
    random genomes' features (chaos in log space, it spans decades); the end bins are open."""
    f = np.array([r["features"] for r in results if r["fitness"] is not None])
    edges = []
    for i, axis in enumerate(AXES):
        v = np.log(f[:, i]) if axis["name"] == "chaos" else f[:, i]
        lo, hi = np.percentile(v, [1, 99])
        e = np.linspace(lo, hi, axis["bins"] + 1)[1:-1]
        edges.append(np.exp(e) if axis["name"] == "chaos" else e)
    return [e.tolist() for e in edges]


def run_pool(pool, tasks, total, inflight, on_result):
    """Keep `inflight` evaluations running; tasks() makes the next one, on_result inserts results."""
    done = queue.Queue()
    submitted = 0

    def submit():
        nonlocal submitted
        pool.apply_async(evaluate, (tasks(),), callback=done.put,
                         error_callback=lambda e: done.put({"fitness": None, "error": repr(e)}))
        submitted += 1

    for _ in range(min(inflight, total)):
        submit()
    t0 = time.time()
    for i in range(total):
        on_result(done.get())
        if submitted < total:
            submit()
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{total} ({time.time() - t0:.0f}s)", flush=True)


def run_gpu(pool, device, batch, tasks, total, on_result):
    """run_pool's job with the physics on the GPU. tasks() is called `batch` at a time, so
    selection sees the archive as it is when each batch is made. Up to GPU_STREAMS batches
    simulate at once; each is built in the pool before and scored in the pool after, and the next
    batch is made as soon as one leaves the GPU."""
    done = queue.Queue()
    made = 0

    def simulate(work):                     # in a GPU thread
        try:
            setups, data = simulate_gpu(pool, [genome for genome, _ in work], device)
        except BaseException as e:
            done.put(e)
            raise
        done.put(None)                      # the GPU can take another batch
        for (genome, meta), s, d in zip(work, setups, data):
            pool.apply_async(score, ((genome, meta, s["desc"], d),), callback=done.put,
                             error_callback=lambda e, meta=meta: done.put({**meta, "fitness": None, "error": repr(e)}))

    with ThreadPoolExecutor(GPU_STREAMS) as gpu:
        def make():
            nonlocal made
            work = [tasks() for _ in range(min(batch, total - made))]
            made += len(work)
            gpu.submit(simulate, work)

        for _ in range(GPU_STREAMS):
            if made < total:
                make()
        t0 = time.time()
        received = 0
        while received < total:
            item = done.get()
            if isinstance(item, BaseException):
                raise item
            if item is None:
                if made < total:
                    make()
                continue
            on_result(item)
            received += 1
            if received % 500 == 0:
                print(f"  {received}/{total} ({time.time() - t0:.0f}s)", flush=True)


def run_random(run, seed, evals):
    """`evals` random genomes through run(tasks, total, on_result), run_pool's or run_gpu's job."""
    results = []
    index = iter(range(evals))
    run(lambda: (random_genome(seed, (i := next(index))), {"index": i}), evals, results.append)
    return sorted(results, key=lambda r: r["index"])


def bin_archive(results, edges):
    archive = {}
    for r in results:
        if r["fitness"] is None:
            continue
        cell = cell_of(r["features"], edges)
        if cell not in archive or r["fitness"] > archive[cell]["fitness"]:
            archive[cell] = {**r, "cell": list(cell)}
    return archive


def run_map_elites(run, seed, init, evals, edges):
    """MAP-Elites from the random results `init`, for evals - len(init) children.
    Returns the archive and one record per child for the locality check."""
    archive = bin_archive(init, edges)
    rng = np.random.default_rng([seed, 1 << 20])
    children = []

    def next_task():
        cells = list(archive)
        parent = archive[cells[rng.integers(len(cells))]]
        mate = None
        if rng.random() < CROSSOVER:
            near = [c for c in cells if c != tuple(parent["cell"])
                    and max(abs(a - b) for a, b in zip(c, parent["cell"])) <= CROSS_REACH]
            if near:
                mate = archive[near[rng.integers(len(near))]]["genome"]
        child, op = mutate(parent["genome"], rng, mate)
        return child, {"parent": parent["cell"], "parent_features": parent["features"], "op": op}

    def insert(r):
        rec = {"op": r.get("op"), "parent": r.get("parent")}
        if r["fitness"] is not None:
            cell = cell_of(r["features"], edges)
            rec.update(cell=list(cell), parent_features=r["parent_features"], features=r["features"])
            if cell not in archive or r["fitness"] > archive[cell]["fitness"]:
                archive[cell] = {**r, "cell": list(cell)}
                rec["new"] = True
        children.append(rec)

    run(next_task, evals - len(init), insert)
    return archive, children


# ---- evaluation ----

def metrics(archives, best):
    """The paper's four metrics for one archive, normalised by the best fitness known per cell."""
    out = []
    for archive in archives:
        found = [archive[c]["fitness"] / best[c] if c in archive and best[c] > 0 else 0.0 for c in best]
        filled = [v for c, v in zip(best, found) if c in archive]
        out.append({"coverage": len(archive) / len(best), "reliability": float(np.mean(found)),
                    "precision": float(np.mean(filled)),
                    "global": max(a["fitness"] for a in archive.values()) / max(best.values())})
    return out


def locality(children, edges, random_results):
    """Parent -> child feature distance in bin widths, against two unrelated random genomes."""
    width = []
    for i, axis in enumerate(AXES):
        e = np.asarray(edges[i])
        width.append(np.log(e[1] / e[0]) if axis["name"] == "chaos" else e[1] - e[0])

    def dist(a, b):
        a, b = np.asarray(a), np.asarray(b)
        d = (a - b).copy()
        d[:, 0] = np.log(a[:, 0] + 1e-9) - np.log(b[:, 0] + 1e-9)
        return np.sqrt(((d / width) ** 2).sum(1))

    kids = [c for c in children if "cell" in c]
    out = {}
    for op in sorted({c["op"] for c in kids}) + ["all"]:
        group = [c for c in kids if op in ("all", c["op"])]
        child_f = np.array([c["features"] for c in group])
        parent_f = np.array([c["parent_features"] for c in group])
        cells = np.array([c["cell"] for c in group])
        parents = np.array([c["parent"] for c in group])
        cheb = np.abs(cells - parents).max(1)
        out[op] = {"children": len(group), "median_bins": float(np.median(dist(child_f, parent_f))),
                   "same_cell": float((cheb == 0).mean()), "within_1": float((cheb <= 1).mean()),
                   "inserted": float(np.mean([c.get("new", False) for c in group]))}
    f = np.array([r["features"] for r in random_results if r["fitness"] is not None])
    rng = np.random.default_rng(0)
    a, b = f[rng.integers(len(f), size=5000)], f[rng.integers(len(f), size=5000)]
    out["random_pairs"] = {"median_bins": float(np.median(dist(a, b)))}
    ca = np.array([cell_of(x, edges) for x in a])
    cb = np.array([cell_of(x, edges) for x in b])
    cheb = np.abs(ca - cb).max(1)
    out["random_pairs"].update(same_cell=float((cheb == 0).mean()), within_1=float((cheb <= 1).mean()))
    return out


# ---- pictures ----

def render(task):
    """task = (genome, half, path, trajectory); trajectory None simulates it here, with numpy."""
    genome, half, path, data = task
    if data is None:
        s, physics = setup_of(genome)
        data = P.simulate(s["pos"], s["vel"], genome["steps"], log_every=LOG_EVERY, **physics)
    draw(data, half, "hsv", path, size=2, dpi=80)
    return path


def grid_sheet(elites, path, tile=88):
    """The map as a picture: one panel per branching bin, chaos left to right, density bottom to top."""
    nx, ny, nz = (a["bins"] for a in AXES)
    font, small = load_font(18), load_font(12)
    pad, top = 24, 30
    pw, ph = nx * tile + pad, ny * tile + top + pad
    cols = 3
    rows = (nz + cols - 1) // cols
    sheet = Image.new("RGB", (cols * pw + pad, rows * ph + 40), (18, 18, 18))
    d = ImageDraw.Draw(sheet)
    d.text((pad, 8), f"{AXES[0]['label']} (left to right)   ·   {AXES[1]['label']} (bottom to top)   ·   "
                     f"{AXES[2]['label']} (panel by panel)", fill=(200, 200, 200), font=font)
    for e in elites:
        x, y, z = e["cell"]
        px, py = pad + (z % cols) * pw, 40 + (z // cols) * ph
        im = Image.open(e["image"]).convert("RGB").resize((tile, tile), Image.LANCZOS)
        sheet.paste(im, (px + x * tile, py + top + (ny - 1 - y) * tile))
    for z in range(nz):
        px, py = pad + (z % cols) * pw, 40 + (z // cols) * ph
        d.text((px, py + 8), f"branching bin {z + 1}/{nz}", fill=(170, 170, 170), font=small)
        d.rectangle([px - 1, py + top - 1, px + nx * tile, py + top + ny * tile], outline=(60, 60, 60))
    sheet.save(path)


def pick_device(name):
    """--device to "cpu" or "cuda"; auto takes cuda when torch is installed and sees a GPU."""
    if name == "cpu":
        return "cpu"
    try:
        import torch
        found = torch.cuda.is_available()
    except ImportError:
        found = False
    if name == "cuda" and not found:
        raise SystemExit("--device cuda needs torch with CUDA and a GPU (see README)")
    return "cuda" if found else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("evals", type=int, nargs="?", default=3000, help="evaluations per treatment per replicate")
    ap.add_argument("--init", type=int, default=600, help="random genomes MAP-Elites starts from")
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--inflight", type=int, default=0, help="cpu: evaluations queued at once (default 3 x workers)")
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto", help="where to simulate")
    ap.add_argument("--batch", type=int, default=32,
                    help=f"cuda: MAP-Elites genomes per GPU batch, {GPU_STREAMS} in flight; larger is faster, "
                         "smaller lets selection see newer elites")
    ap.add_argument("--out", default=OUT, help="output folder")
    args = ap.parse_args()
    inflight = args.inflight or 3 * args.workers
    device = pick_device(args.device)
    if device == "cuda":
        import torch
        print(f"device: cuda ({torch.cuda.get_device_name()}), MAP-Elites batches of {args.batch}", flush=True)
    else:
        print(f"device: cpu, {args.workers} workers", flush=True)
    t0 = time.time()

    os.makedirs(os.path.join(args.out, "atlas"), exist_ok=True)
    with Pool(args.workers) as pool:
        if device == "cuda":
            def on_gpu(batch):
                return lambda tasks, total, on_result: run_gpu(pool, device, batch, tasks, total, on_result)
            sample, select = on_gpu(GPU_SAMPLE_BATCH), on_gpu(args.batch)
        else:
            sample = select = lambda tasks, total, on_result: run_pool(pool, tasks, total, inflight, on_result)
        randoms, maps, kids = [], [], []
        for rep in range(args.replicates):
            print(f"replicate {rep}: random sampling, {args.evals} evaluations", flush=True)
            randoms.append(run_random(sample, rep, args.evals))
            if rep == 0:
                edges = calibrate(randoms[0])
                print("bin edges:", [[round(v, 3) for v in e] for e in edges], flush=True)
            print(f"replicate {rep}: MAP-Elites, {args.evals} evaluations", flush=True)
            archive, children = run_map_elites(select, rep, randoms[rep][:args.init], args.evals, edges)
            maps.append(archive)
            kids.append(children)
        search_time = time.time() - t0

        baselines = [bin_archive(r, edges) for r in randoms]
        best = {}
        for archive in baselines + maps:
            for c, e in archive.items():
                best[c] = max(best.get(c, 0.0), e["fitness"])
        evaluation = {
            "evaluations_per_run": args.evals, "init": args.init, "replicates": args.replicates,
            "cells": int(np.prod([a["bins"] for a in AXES])), "fillable": len(best),
            "random": metrics(baselines, best), "map_elites": metrics(maps, best),
            "broken": {"random": [sum(r["fitness"] is None for r in rs) for rs in randoms],
                       "map_elites": [sum(c.get("cell") is None for c in ch) for ch in kids]},
            "locality": [locality(ch, edges, randoms[0]) for ch in kids],
            "ops": {op: sum(c["op"] == op for c in kids[0]) for op in sorted({c["op"] for c in kids[0]})},
            "search_seconds": round(search_time), "device": device,
        }

        # the published atlas: best elite per cell over the MAP-Elites replicates
        atlas = {}
        for rep, archive in enumerate(maps):
            for c, e in archive.items():
                if c not in atlas or e["fitness"] > atlas[c]["fitness"]:
                    atlas[c] = {**e, "run": rep}
        elites = sorted(atlas.values(), key=lambda e: e["cell"])
        for e in elites:
            e["image"] = f"{args.out}/atlas/" + "-".join(map(str, e["cell"])) + ".png"
        for old in glob.glob(f"{args.out}/atlas/*.png"):
            os.remove(old)
        print(f"rendering {len(elites)} elites", flush=True)
        genomes = [e["genome"] for e in elites]
        # the stills come from the simulator that scored them; the runs drift apart chaotically
        data = simulate_gpu(pool, genomes, device)[1] if device == "cuda" else [None] * len(elites)
        list(pool.imap_unordered(render, [(g, e["half"], e["image"], d) for g, e, d in zip(genomes, elites, data)],
                                 chunksize=2))
    grid_sheet(elites, os.path.join(args.out, "atlas.png"))

    keep = ("cell", "fitness", "features", "terms", "genome", "desc", "n", "half", "parent", "op", "run", "image")
    out = {"axes": [{**a, "edges": e} for a, e in zip(AXES, edges)],
           "fitness": "max(0, 1 - 2 * escape) * (0.4 + 0.6 * structure) * (0.4 + 0.6 * crispness)"
                      " * (frames / 140) ** 0.2",
           "log_every": LOG_EVERY,
           "evaluation": evaluation,
           "elites": [{k: (round(e[k], 4) if k == "fitness" else e[k]) for k in keep if k in e} for e in elites]}
    for e in out["elites"]:
        e["features"] = [round(v, 4) for v in e["features"]]
        e["half"] = round(e["half"], 2)
        e["steps"] = e["genome"]["steps"]
    with open(os.path.join(args.out, "atlas.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: evaluation[k] for k in ("fillable", "random", "map_elites", "broken")}, indent=1))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
