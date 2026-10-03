"""Probe the fitness landscape behind the atlas (see research/landscape-notes.md).

Usage: python landscape.py analyse [--run DIR] [--atlas FILE] [--out DIR]
       python landscape.py noise [--runs N] [--subset N] [--run DIR] [--atlas FILE] [--out DIR] [--headroom H]
       python landscape.py sweep [--size N] [--atlas FILE] [--out DIR] [--headroom H]
       python landscape.py structure [--per-row N] [--run DIR] [--atlas FILE] [--out DIR] [--headroom H]

analyse reads the published atlas (--atlas, search/atlas.json) and the log of a logged atlas run
(--run, search/landscape/run, made by `python atlas.py 50000 --init 2000 --replicates 3
--device cuda --out search/landscape/run --log`), bins every logged evaluation with the published
atlas's edges, and writes figures and analysis.json to --out (search/landscape/):

  fitness.png      published elite fitness per cell, one panel per branching bin
  limits.png       the fitness factor that costs each elite most (structure, crispness, escape,
                   duration), with the elite's fitness x 100 in the cell
  bias.png         fitness terms against the axes over the random genomes, with the elites on top
  rarity.png       how often random genomes land in each cell (log10)
  gain.png         published elite fitness minus the best random genome in the cell
  headroom.png     spread of the three replicates' best fitness per cell
  late.png         how many replicates were still improving each cell in the last 20% of the run
  rerun_diff.png   published atlas minus this run's merged atlas, per cell
  convergence.png  reliability (mean over cells of best / best known) against evaluations
  traits.png       which genome traits predict fitness, cell-relative fitness and the three axes
                   (gradient boosting, permutation importance), and partial dependence of fitness

noise measures the chaos noise floor. Every published elite is simulated once as built (this
must reproduce its published fitness, features and cell exactly) and --runs (32) times with its
start positions perturbed by iid Gaussian noise of scale 1e-9; --subset (40) elites spread over
the map (see spread_picks) also run 16 times at each of EPS. Writes to --out:

  noise.json        sanity check; per elite fitness mean, std, min, max, published - mean, share of
                    runs leaving the cell and feature std in bin widths; spread against eps; and,
                    if --run has a log, the replicate spread and rerun difference of analyse
                    against the noise floor
  noise.npz         every run: elite index, eps index (-1 unperturbed), run, fitness, features
  noise.png         fitness std over the perturbed runs per cell
  noise_cells.png   share of perturbed runs that leave the elite's cell
  noise_eps.png     fitness std and cell changes against the perturbation scale
  winners_curse.png published fitness against the perturbed mean

sweep re-simulates SWEEPS elites (the FAVOURITES plus the rest from spread_picks) on a --size x
--size (48) grid of (log2_pull, log2_push) over particular.LOG2_RANGE, everything else fixed.
Run noise first: each plane's roughness is compared with its elite's noise floor. Writes:

  sweeps/<cell>.png  one plane: fitness with outlines where the cell changes, the elite marked
                     (star) and nine lettered points; the three axes' bins; the elite's cell
                     region; stills of the lettered points
  sweeps.png         every plane's fitness, small
  sweeps.json        per plane: elite against plane and cell maxima, share near the elite's
                     fitness, cell region, cells reached, roughness against the noise floor,
                     fitness variance explained by log2_push - log2_pull; pooled roughness and
                     cell changes per chaos bin
  sweeps.npz         every grid point's fitness and features

structure asks whether the dense rows' structure cap is real or measured. It re-simulates every
published elite and --per-row (200) random genomes per density row from --run's log (a seeded,
reproducible sample), checks they reproduce their logged fitness, and scores each world's
structure four ways: published (atlas._structure: 64 sectors, harmonics 1..12, k 2..6), all (512
sectors, every harmonic to 256, k 2..STRUCT_K), image (the saturated 128-px trail image on a
48 x 1024 polar grid, all harmonics: only symmetry the picture resolves) and independent (the
image's angular part correlated with itself turned by 2pi/k, k 2..128, times its angular share of
variance). Each is also scored on a null copy with every particle's path turned by its own random
angle. Writes:

  structure.json              a, s and structure per density row for elites, random genomes and
                              their nulls; agreement with the independent check; elite fitness if
                              structure read all harmonics; the examples
  structure_decomposition.png S = a x s by density row, published and all harmonics
  structure_ceiling.png       random 95th percentile and mean and elite mean per variant, with nulls
  structure_agreement.png     each variant against the independent check, by density row
  structure_examples.png      stills and angular spectra of three elites and one random genome

noise, sweep and structure simulate on the GPU (gpu_sim.py, torch with CUDA) and measure in a
process pool. They leave --headroom (0.05) of the machine free: (1 - headroom) x cores workers at
below-normal priority, a sleep of headroom x each GPU batch's time after it, and GPU memory capped
at 1 - 2 x headroom of the card. They need psutil.

Needs scikit-learn and scipy (pip install scikit-learn) besides the atlas's own dependencies.
"""
import argparse
import gzip
import io
import json
import os
import time
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.patheffects import withStroke
from scipy import ndimage
from scipy.stats import spearmanr

import atlas
import particular as P
from make_gallery import draw

TERMS = ["structure", "crispness", "escape", "duration"]
TRAITS = ["k", "pieces", "rings", "satellites", "spokes", "species", "log2_pull", "log2_push", "steps",
          "particles", "rule_pull_mean", "rule_push_mean", "rule_pull_spread", "rule_push_spread", "rule_asym"]
LATE = 0.8          # a replicate still improves a cell if its last improvement came after this share of the run
DARK = "#111"

FAVOURITES = [(7, 1, 1), (5, 1, 5), (5, 3, 0), (3, 6, 0), (5, 7, 2)]   # always swept
SWEEPS = 20         # planes swept: the favourites plus spread_picks
EPS = [1e-12, 1e-9, 1e-6, 1e-3]   # start-position noise scales on the noise subset
NOISE_EPS = 1e-9    # the scale every elite runs at
EPS_RUNS = 16       # runs per scale on the subset (the first of the NOISE_EPS runs are reused)
HEADROOM = 0.05     # share of CPU and GPU the probes leave free
CHUNK = 2048        # setups per GPU batch; gpu_sim launches them in groups of 512
FMAX = 0.95         # top of the fitness colour scale
RANGE3 = 3 / np.sqrt(np.pi)   # expected range of 3 iid normal draws, in std units
ABSDIFF2 = 2 / np.sqrt(np.pi)  # expected |difference| of 2 iid normal draws, in std units
STRUCT_PER_ROW = 200       # random genomes per density row re-measured by structure
STRUCT_SECTORS = 512       # polar sectors of the all-harmonics variant: harmonics 1..256
STRUCT_K = 85              # largest symmetry order tried: every k with at least 3 multiples up to 256
STRUCT_RADII, STRUCT_ANGLES = 48, 1024   # polar resampling of the trail image (image variant, check)
STRUCT_EXAMPLES = [(0, 0, 0), (2, 7, 4), (2, 7, 2)]   # sparse reference, a sunburst, filled fans
STRUCT_VARIANTS = ["published", "all", "image"]
STRUCT_COLORS = {"published": "#e8684c", "all": "#4c9be8", "image": "#9be84c", "independent": "#ffb000"}


# ---- loading ----

def traits(genome, n):
    kinds = [p["kind"] for p in genome["pieces"]]
    s = genome["species"]
    pull, push = (np.asarray(genome["log2_rules"][key], float) for key in ("pull", "push"))
    multi = s > 1
    return [genome["k"], len(kinds), kinds.count("ring"), kinds.count("satellites"), kinds.count("spokes"), s,
            genome["log2_pull"], genome["log2_push"], genome["steps"], n,
            pull.mean() if multi else 0.0, push.mean() if multi else 0.0,
            pull.std() if multi else 0.0, push.std() if multi else 0.0,
            (np.abs(pull - pull.T).mean() + np.abs(push - push.T).mean()) / 2 if multi else 0.0]


def factors(terms):
    """The four fitness factors of terms rows (structure, crispness, escape, duration)."""
    t = np.asarray(terms, float)
    return np.stack([0.4 + 0.6 * t[:, 0], 0.4 + 0.6 * t[:, 1], np.maximum(0, 1 - 2 * t[:, 2]), t[:, 3]], 1)


def load_log(path):
    """Every evaluation of a --log run as arrays; broken evaluations have fitness nan."""
    cols = {k: [] for k in ("rep", "child", "i", "fitness", "features", "terms", "order", "op", "new", "traits")}
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            ok = r.get("fitness") is not None
            t = r.get("terms", {})
            cols["rep"].append(r["rep"])
            cols["child"].append(r["kind"] == "child")
            cols["i"].append(r["i"])
            cols["fitness"].append(r["fitness"] if ok else np.nan)
            cols["features"].append(r["features"] if ok else [np.nan] * 3)
            cols["terms"].append([t[k] for k in TERMS] if ok else [np.nan] * 4)
            cols["order"].append(t.get("order", 0))
            cols["op"].append(r.get("op", ""))
            cols["new"].append(r.get("new", False))
            cols["traits"].append(traits(r["genome"], r.get("n", 0)))
    return {k: np.array(v) for k, v in cols.items()}


def cells_of(features, edges):
    return np.stack([np.searchsorted(e, features[:, a]) for a, e in enumerate(edges)], 1)


def load_atlas(path):
    """An atlas.json, its bin edges per axis and its map shape."""
    A = json.load(open(path))
    return A, [np.array(a["edges"]) for a in A["axes"]], tuple(a["bins"] for a in A["axes"])


def load_run(run, edges):
    """A --log run's evaluations (load_log), which are finite, and their cells under edges."""
    L = load_log(os.path.join(run, "evals.jsonl.gz"))
    ok = np.isfinite(L["fitness"])
    cells = np.zeros((len(ok), 3), int)
    cells[ok] = cells_of(L["features"][ok], edges)
    return L, ok, cells


def bin_coords(features, edges):
    """Features (..., 3) in bin widths: interior bin i spans i..i+1 (chaos in log space, as
    atlas.calibrate spaces its edges), so feature differences read as bins."""
    out = np.empty(np.shape(features))
    for a, e in enumerate(edges):
        v = np.asarray(features)[..., a]
        v, e = (np.log(v), np.log(e)) if a == 0 else (v, e)
        out[..., a] = 1 + (v - e[0]) / (e[1] - e[0])
    return out


def measured(ms, edges):
    """fitness (n,), features (n, 3) and cells (n, 3) of measure results; broken ones nan and -1."""
    fit = np.array([np.nan if m is None else m["fitness"] for m in ms])
    feat = np.array([[np.nan] * 3 if m is None else m["features"] for m in ms])
    cells = np.full((len(ms), 3), -1)
    ok = np.isfinite(fit)
    cells[ok] = cells_of(feat[ok], edges)
    return fit, feat, cells


# ---- drawing ----

def slice_figure(values, title, cmap="viridis", vmin=None, vmax=None, labels=None, colorbar=True, ticks=None,
                 path=None, note=None):
    """One 8 x 8 panel per branching bin, chaos left to right and density bottom to top, like atlas.png.
    values: (8, 8, 6) array, nan for empty cells; labels: optional same-shape strings drawn in the cells."""
    nx, ny, nz = values.shape
    fig, axes = plt.subplots(2, 3, figsize=(13, 9), facecolor=DARK)
    cm = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap
    cm.set_bad("#2a2a2a")
    for z, ax in enumerate(axes.flat):
        im = ax.imshow(values[:, :, z].T, origin="lower", cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"branching bin {z + 1}/{nz}", color="w", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#555")
        if labels is not None:
            for x in range(nx):
                for y in range(ny):
                    if labels[x, y, z]:
                        ax.text(x, y, labels[x, y, z], ha="center", va="center", fontsize=6.5, color="w")
    fig.suptitle(title, color="w", fontsize=13)
    fig.text(0.5, 0.04, "orderly → chaotic (across)   ·   sparse → dense (up)" + (f"\n{note}" if note else ""),
             ha="center", color="#bbb", fontsize=9)
    if colorbar:
        cb = fig.colorbar(im, ax=axes, shrink=0.6, ticks=ticks)
        cb.ax.yaxis.set_tick_params(color="w", labelcolor="w")
        if ticks is not None and isinstance(colorbar, list):
            cb.ax.set_yticklabels(colorbar)
    fig.savefig(path, dpi=110, facecolor=DARK)
    plt.close(fig)


def grid(cells, values, shape):
    out = np.full(shape, np.nan)
    for c, v in zip(cells, values):
        out[tuple(c)] = v
    return out


# ---- analyses ----

def replay(L, rep, init, cells, shape):
    """MAP-Elites replicate `rep` replayed from the log: best fitness per cell, the child index of each
    cell's last improvement (-1 if never improved after init), and reliability checkpoints."""
    best = np.full(shape, np.nan)
    last = np.full(shape, np.nan)
    trace = []
    rnd = np.where((L["rep"] == rep) & ~L["child"] & (L["i"] < init))[0]
    kids = np.where((L["rep"] == rep) & L["child"])[0]
    kids = kids[np.argsort(L["i"][kids])]
    for j, idx in enumerate(np.concatenate([rnd, kids])):
        f = L["fitness"][idx]
        if np.isfinite(f):
            c = tuple(cells[idx])
            if not f <= best[c]:                 # nan-safe: an empty cell takes anything
                best[c] = f
                last[c] = j - len(rnd)
        if j % 500 == 499:
            trace.append((j + 1, best.copy()))
    return best, last, trace


def random_best(L, rep, cells, shape, checkpoints=()):
    best = np.full(shape, np.nan)
    trace = []
    idx = np.where((L["rep"] == rep) & ~L["child"])[0]
    idx = idx[np.argsort(L["i"][idx])]
    for j, k in enumerate(idx):
        f = L["fitness"][k]
        if np.isfinite(f):
            c = tuple(cells[k])
            if not f <= best[c]:
                best[c] = f
        if j % 500 == 499:
            trace.append((j + 1, best.copy()))
    return best, trace


def reliability(best, known):
    ok = np.isfinite(known) & (known > 0)
    return float(np.nansum(np.where(ok, np.nan_to_num(best) / np.where(ok, known, 1), 0)) / ok.sum())


def agreement(B):
    """Of replicate bests B (reps, *shape): replicates that filled each cell, their spread (max - min,
    nan below two) and the merged atlas (best over replicates, nan if none)."""
    filled = np.isfinite(B).sum(0)
    spread = np.where(filled >= 2, np.nanmax(B, 0) - np.nanmin(B, 0), np.nan)
    merged = np.nanmax(np.where(np.isfinite(B), B, -1), 0)
    return filled, spread, np.where(merged < 0, np.nan, merged)


def importance(L, ok, cells, shape, rng):
    """Gradient boosting from genome traits to fitness, cell-relative fitness and the axes, on random genomes."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import partial_dependence, permutation_importance
    X = L["traits"][ok]
    f = L["fitness"][ok]
    c = cells[ok]
    flat = np.ravel_multi_index(c.T, shape)
    mean = np.bincount(flat, f, np.prod(shape)) / np.maximum(np.bincount(flat, minlength=np.prod(shape)), 1)
    feats = L["features"][ok]
    targets = {"fitness": f, "fitness within cell": f - mean[flat], "chaos (log)": np.log10(feats[:, 0]),
               "density": feats[:, 1], "branching": feats[:, 2]}
    order = rng.permutation(len(X))
    train, test = order[: int(0.8 * len(X))], order[int(0.8 * len(X)):][:20000]
    out, model_f = {}, None
    for name, y in targets.items():
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=0).fit(X[train], y[train])
        r2 = m.score(X[test], y[test])
        pi = permutation_importance(m, X[test], y[test], n_repeats=4, random_state=0, n_jobs=-1)
        out[name] = {"r2": round(float(r2), 3),
                     "importance": {t: round(float(v), 4) for t, v in zip(TRAITS, pi.importances_mean)}}
        if name == "fitness":
            model_f = m
    top = sorted(TRAITS, key=lambda t: -out["fitness"]["importance"][t])[:6]
    pd = {}
    for t in top:
        j = TRAITS.index(t)
        # brute: the fast "recursion" method leaves out the model's baseline, so its curves are offsets
        r = partial_dependence(model_f, X[test][:3000], [j], grid_resolution=20, method="brute")
        pd[t] = (r["grid_values"][0].tolist(), r["average"][0].tolist())
    return out, pd


def analyse(args):
    rng = np.random.default_rng(0)
    A, edges, shape = load_atlas(args.atlas)
    R = json.load(open(os.path.join(args.run, "atlas.json")))
    same_edges = all(np.allclose(a["edges"], b["edges"]) for a, b in zip(A["axes"], R["axes"]))
    init, reps = R["evaluation"]["init"], R["evaluation"]["replicates"]
    print("loading log", flush=True)
    L, ok, cells = load_run(args.run, edges)
    rand = ok & ~L["child"]
    summary = {"evaluations": int(len(ok)), "broken": int((~ok).sum()), "same_edges": bool(same_edges)}

    # published elites
    E = A["elites"]
    ec = np.array([e["cell"] for e in E])
    ef = np.array([e["fitness"] for e in E])
    eterms = np.array([[e["terms"][k] for k in TERMS] for e in E])
    efac = factors(eterms)
    deficit = -np.log(np.maximum(efac, 1e-6))
    limiter = deficit.argmax(1)
    slice_figure(grid(ec, ef, shape), "Published atlas: elite fitness", "magma", 0.25, 0.92,
                 labels=grid(ec, ef, shape).round(2).astype(str), colorbar=True, path=f"{args.out}/fitness.png")
    lab = np.full(shape, "", object)
    for c, f in zip(ec, ef):
        lab[tuple(c)] = f"{100 * f:.0f}"
    lim_cm = ListedColormap(["#4c78a8", "#e45756", "#54a24b", "#b279a2"])
    slice_figure(grid(ec, limiter, shape), "What costs each elite most fitness", lim_cm, -0.5, 3.5, labels=lab,
                 colorbar=TERMS, ticks=[0, 1, 2, 3], path=f"{args.out}/limits.png",
                 note="colour = factor with the largest -log(factor); number = fitness x 100")
    summary["limiter_counts"] = {t: int((limiter == i).sum()) for i, t in enumerate(TERMS)}
    summary["factor_mean"] = {t: round(float(efac[:, i].mean()), 3) for i, t in enumerate(TERMS)}
    summary["factor_by_density_row"] = {t: [round(float(efac[ec[:, 1] == y, i].mean()), 3) for y in range(shape[1])]
                                        for i, t in enumerate(TERMS)}
    summary["fitness_by_density_row"] = [round(float(ef[ec[:, 1] == y].mean()), 3) for y in range(shape[1])]
    summary["log_deficit_share"] = {t: round(float(deficit[:, i].sum() / deficit.clip(0).sum()), 3)
                                    for i, t in enumerate(TERMS)}

    # bias: terms against axes over random genomes
    F, T, fit = L["features"][rand], L["terms"][rand], L["fitness"][rand]
    names = ["chaos", "density", "branching"]
    summary["spearman_random"] = {
        a: {t: round(float(spearmanr(F[:, j], T[:, i]).statistic), 3) for i, t in enumerate(TERMS)}
        | {"fitness": round(float(spearmanr(F[:, j], fit).statistic), 3)} for j, a in enumerate(names)}
    allc = cells[ok]
    rows = {}
    for i, t in enumerate(TERMS[:2] + ["fitness"]):
        vals = L["fitness"][ok] if t == "fitness" else L["terms"][ok][:, i]
        rv = fit if t == "fitness" else T[:, i]
        rows[t] = {"random_p50": [], "random_p95": [], "logged_max": []}
        for y in range(shape[1]):
            sel = cells[rand][:, 1] == y
            rows[t]["random_p50"].append(round(float(np.median(rv[sel])), 3))
            rows[t]["random_p95"].append(round(float(np.percentile(rv[sel], 95)), 3))
            rows[t]["logged_max"].append(round(float(vals[allc[:, 1] == y].max()), 3))
    summary["by_density_row"] = rows
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.6), facecolor=DARK)
    panels = [("density", 1, "crispness", T[:, 1], eterms[:, 1]), ("density", 1, "structure", T[:, 0], eterms[:, 0]),
              ("branching", 2, "structure", T[:, 0], eterms[:, 0]), ("density", 1, "fitness", fit, ef)]
    efeat = np.array([e["features"] for e in E])
    for ax, (axis, j, term, y, ey) in zip(axs, panels):
        ax.set_facecolor("black")
        ax.hexbin(F[:, j], y, gridsize=60, bins="log", cmap="Greys_r", mincnt=1)
        xs = np.linspace(np.percentile(F[:, j], 0.5), np.percentile(F[:, j], 99.5), 25)
        mids, p50, p95 = [], [], []
        for a, b in zip(xs[:-1], xs[1:]):
            sel = (F[:, j] >= a) & (F[:, j] < b)
            if sel.sum() > 30:
                mids.append((a + b) / 2)
                p50.append(np.median(y[sel]))
                p95.append(np.percentile(y[sel], 95))
        ax.plot(mids, p50, color="#4c9be8", lw=2, label="random median")
        ax.plot(mids, p95, color="#4c9be8", lw=1, ls="--", label="random 95th pct")
        ax.scatter(efeat[:, j], ey, s=7, color="#ffb000", label="published elites", zorder=3)
        for e in edges[j]:
            ax.axvline(e, color="#333", lw=0.6)
        ax.set_xlabel(axis, color="w")
        ax.set_ylabel(term, color="w")
        ax.tick_params(colors="w")
    axs[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Fitness terms against the map's axes (150k random genomes)", color="w")
    fig.tight_layout()
    fig.savefig(f"{args.out}/bias.png", dpi=110, facecolor=DARK)
    plt.close(fig)

    # does the measured order find the genome's k?
    order, k, kinds_sym = L["order"][rand], L["traits"][rand][:, 0], L["traits"][rand][:, 3] + L["traits"][rand][:, 4]
    sym = kinds_sym > 0                                    # satellites and spokes are k-fold; rings are not
    tab = {int(kk): {int(o): int(((k == kk) & (order == o) & sym).sum()) for o in range(2, 7)} for kk in range(1, 7)}
    summary["genome_k_vs_measured_order"] = tab
    summary["elite_order"] = {int(o): int(sum(e["terms"]["order"] == o for e in E)) for o in range(2, 7)}

    # rarity
    cnt = np.zeros(shape)
    np.add.at(cnt, tuple(cells[rand].T), 1)
    rbest = np.full(shape, np.nan)
    for c, f in zip(cells[rand], fit):
        if not f <= rbest[tuple(c)]:
            rbest[tuple(c)] = f
    pub = grid(ec, ef, shape)
    lc = np.where(cnt > 0, np.log10(np.maximum(cnt, 1)), np.nan)
    lab = np.where(cnt > 0, cnt.astype(int).astype(str), "")
    slice_figure(lc, f"How often random genomes land in each cell (log10 count of {int(rand.sum())})",
                 "cividis", 0, np.nanmax(lc), labels=lab, path=f"{args.out}/rarity.png")
    gain = pub - rbest
    slice_figure(gain, "Published elite fitness minus best random genome in the cell", "RdBu", -0.3, 0.3,
                 labels=np.where(np.isfinite(gain), np.round(100 * np.nan_to_num(gain)).astype(int).astype(str), ""),
                 path=f"{args.out}/gain.png", note="number = difference x 100; red = random did better")
    both = np.isfinite(pub) & (cnt > 0)
    summary["rarity"] = {
        "cells_hit_by_random": int((cnt > 0).sum()),
        "cells_under_10_random": int(((cnt > 0) & (cnt < 10)).sum()),
        "top_10_cells_share": round(float(np.sort(cnt.ravel())[::-1][:10].sum() / cnt.sum()), 3),
        "spearman_count_vs_elite_fitness": round(float(spearmanr(cnt[both], pub[both]).statistic), 3),
        "spearman_count_vs_gain": round(float(spearmanr(cnt[both], gain[both]).statistic), 3),
        "gain_mean": round(float(np.nanmean(gain)), 3),
        "gain_median_by_log10count": {},
    }
    for lo in range(0, 5):
        sel = both & (np.log10(np.maximum(cnt, 1)) >= lo) & (np.log10(np.maximum(cnt, 1)) < lo + 1)
        if sel.any():
            summary["rarity"]["gain_median_by_log10count"][f"1e{lo}-1e{lo + 1}"] = [int(sel.sum()),
                                                                                    round(float(np.median(gain[sel])), 3)]

    # headroom and convergence
    print("replaying replicates", flush=True)
    known = np.full(shape, np.nan)            # best fitness known per cell: every logged evaluation and the atlas
    for c, f in zip(cells[ok], L["fitness"][ok]):
        if not f <= known[tuple(c)]:
            known[tuple(c)] = f
    known = np.fmax(known, pub)
    bests, lasts, traces, rtraces = [], [], [], []
    for rep in range(reps):
        b, last, tr = replay(L, rep, init, cells, shape)
        rb, rtr = random_best(L, rep, cells, shape)
        bests.append(b)
        lasts.append(last)
        traces.append(tr)
        rtraces.append(rtr)
    B = np.stack(bests)
    children = R["evaluation"]["evaluations_per_run"] - init
    late = np.sum([np.nan_to_num(l, nan=-1) >= LATE * children for l in lasts], 0).astype(float)
    filled, spread, merged = agreement(B)
    diff = pub - merged
    late = np.where(np.isfinite(known), late, np.nan)
    slice_figure(spread, "Spread of the three replicates' best fitness per cell", "inferno", 0, 0.3,
                 labels=np.where(np.isfinite(spread), np.round(100 * np.nan_to_num(spread)).astype(int).astype(str), ""),
                 path=f"{args.out}/headroom.png", note="number = (max - min) x 100 over replicates that filled the cell")
    slice_figure(late, "Replicates still improving the cell in the last 20% of the run", "YlOrRd", 0, 3,
                 labels=np.where(np.isfinite(late), np.nan_to_num(late).astype(int).astype(str), ""),
                 path=f"{args.out}/late.png")
    slice_figure(diff, "Published atlas minus this logged rerun (same settings)", "RdBu", -0.2, 0.2,
                 labels=np.where(np.isfinite(diff), np.round(100 * np.nan_to_num(diff)).astype(int).astype(str), ""),
                 path=f"{args.out}/rerun_diff.png", note="number = difference x 100; blue = published is better")
    fs = np.isfinite(spread)
    summary["headroom"] = {
        "cells_known": int(np.isfinite(known).sum()),
        "cells_filled_by_n_replicates": {int(n): int((filled == n).sum()) for n in range(reps + 1)},
        "spread_median": round(float(np.nanmedian(spread)), 3),
        "spread_p90": round(float(np.nanpercentile(spread, 90)), 3),
        "cells_spread_over_0.1": int((spread > 0.1).sum()),
        "late_cells": {int(n): int((late == n).sum()) for n in range(reps + 1)},
        "rerun_vs_published": {"cells_both": int(np.isfinite(diff).sum()),
                               "mean_abs": round(float(np.nanmean(np.abs(diff))), 3),
                               "within_0.02": round(float(np.nanmean(np.abs(diff[np.isfinite(diff)]) <= 0.02)), 3),
                               "published_better": int((diff > 0.02).sum()), "rerun_better": int((diff < -0.02).sum())},
        "rerun_merged_fitness_p50": round(float(np.nanmedian(merged)), 3),
        "spread_by_density_row": [round(float(np.nanmedian(spread[:, y, :][fs[:, y, :]])), 3) for y in range(shape[1])],
    }
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=DARK)
    ax.set_facecolor("black")
    for rep in range(reps):
        x = [n for n, _ in traces[rep]]
        ax.plot(x, [reliability(b, known) for _, b in traces[rep]], color="#ffb000", lw=1.5,
                label="MAP-Elites" if rep == 0 else None)
        x = [n for n, _ in rtraces[rep]]
        ax.plot(x, [reliability(b, known) for _, b in rtraces[rep]], color="#4c9be8", lw=1.5,
                label="random sampling" if rep == 0 else None)
    ax.axvline(init, color="#555", ls=":")
    ax.set_xlabel("evaluations", color="w")
    ax.set_ylabel("reliability (mean best / best known)", color="w")
    ax.tick_params(colors="w")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{args.out}/convergence.png", dpi=110, facecolor=DARK)
    plt.close(fig)
    curve = [reliability(b, known) for _, b in traces[0]]
    summary["convergence"] = {"rep0_reliability_at": {str(n): round(float(curve[n // 500 - 1]), 3)
                                                      for n in (2000, 10000, 25000, 40000, 50000)},
                              "rep0_gain_last_10k": round(float(curve[-1] - curve[80 - 1]), 3)}
    imp = np.zeros(reps)
    for rep in range(reps):
        kid = (L["rep"] == rep) & L["child"]
        imp[rep] = L["new"][kid & (L["i"] >= LATE * children)].sum()
    summary["convergence"]["insertions_last_20pct"] = imp.astype(int).tolist()

    # genome traits
    print("fitting trait models", flush=True)
    imps, pd = importance(L, rand, cells, shape, rng)
    summary["traits"] = imps
    fig, axs = plt.subplots(1, 2, figsize=(17, 6), facecolor=DARK, gridspec_kw={"width_ratios": [1.3, 1]})
    M = np.array([[max(imps[t]["importance"][x], 0) for x in TRAITS] for t in imps])
    M = M / M.max(1, keepdims=True)
    axs[0].imshow(M, cmap="magma", aspect="auto")
    axs[0].set_xticks(range(len(TRAITS)), TRAITS, rotation=60, ha="right", color="w", fontsize=8)
    axs[0].set_yticks(range(len(imps)), [f"{t}  (R² {imps[t]['r2']:.2f})" for t in imps], color="w")
    axs[0].set_title("Permutation importance per target (row max = 1)", color="w")
    ax = axs[1]
    ax.set_facecolor("black")
    for t, (x, y) in pd.items():
        x = np.asarray(x, float)
        ax.plot((x - x.min()) / max(np.ptp(x), 1e-9), y, lw=2, label=t)
    ax.set_xlabel("trait, scaled to its range", color="w")
    ax.set_ylabel("predicted fitness (partial dependence)", color="w")
    ax.tick_params(colors="w")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.out}/traits.png", dpi=110, facecolor=DARK)
    plt.close(fig)
    summary["partial_dependence"] = {t: {"x": [round(v, 3) for v in x], "fitness": [round(v, 3) for v in y]}
                                     for t, (x, y) in pd.items()}

    with open(f"{args.out}/analysis.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


# ---- probes ----

def _lower_priority():
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)


def polite_pool(headroom):
    """A measuring pool that leaves headroom of the machine free: (1 - headroom) x cores workers,
    this process and the workers at below-normal priority, this process's GPU memory capped at
    1 - 2 x headroom of the card. probe adds the GPU idle time."""
    import torch
    if headroom > 0:
        _lower_priority()
        torch.cuda.set_per_process_memory_fraction(1 - 2 * headroom)
    workers = max(1, int(os.cpu_count() * (1 - headroom)))
    print(f"{workers} workers, headroom {headroom}", flush=True)
    return Pool(workers, initializer=_lower_priority if headroom > 0 else None)


def _measure(data):
    """atlas.score's measurement of a trajectory: measure's dict, None if the world blew up."""
    return atlas.measure(data) if np.isfinite(data).all() else None


def probe(pool, setups, headroom, keep=False):
    """measure's result per setup (a gpu_sim.simulate_batch dict with steps), simulated on the GPU in
    batches of CHUNK while the pool measures the previous batch. After each batch the GPU idles for
    headroom x the batch's time. keep also returns the trajectories."""
    import gpu_sim
    out, kept, pending, t0, busy = [], [], None, time.time(), 0.0
    for start in range(0, len(setups), CHUNK):
        t = time.time()
        data = gpu_sim.simulate_batch(setups[start:start + CHUNK], log_every=atlas.LOG_EVERY, device="cuda")
        dt = time.time() - t
        busy += dt
        time.sleep(headroom * dt)
        if pending is not None:
            out += pending.get()
        pending = pool.map_async(_measure, data, chunksize=4)
        kept += data if keep else []
        print(f"  {start + len(data)}/{len(setups)} simulated ({time.time() - t0:.0f} s, GPU calls "
              f"{busy / (time.time() - t0):.0%} of the time)", flush=True)
    out += pending.get() if pending is not None else []
    return (out, kept) if keep else out


def built(genome):
    """genome's gpu_sim setup."""
    return {**P.build(genome), "steps": genome["steps"]}


def spread_picks(E, shape, start, n):
    """Indices of n elites spread over the map and its fitness range: start, then repeatedly the
    elite farthest from every pick so far in (chaos, density, branching, fitness), each scaled to
    0..1 (cell coordinates; fitness by the atlas's range). Farthest-point picks take the map's
    corners and both fitness extremes first, then fill in between."""
    f = np.array([e["fitness"] for e in E])
    X = np.c_[np.array([e["cell"] for e in E]) / (np.array(shape) - 1), (f - f.min()) / np.ptp(f)]
    picks = list(start)
    d = np.linalg.norm(X[:, None] - X[picks][None], axis=2).min(1)
    while len(picks) < n:
        j = int(d.argmax())
        picks.append(j)
        d = np.minimum(d, np.linalg.norm(X - X[j], axis=1))
    return picks[:n]


def sweep_picks(E, shape):
    """The swept elites: the FAVOURITES, then spread_picks up to SWEEPS."""
    cells = [tuple(e["cell"]) for e in E]
    return spread_picks(E, shape, [cells.index(c) for c in FAVOURITES], SWEEPS)


def dark(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor("black")
    ax.set_xlabel(xlabel, color="w")
    ax.set_ylabel(ylabel, color="w")
    ax.set_title(title, color="w", fontsize=10)
    ax.tick_params(colors="w")
    for spine in ax.spines.values():
        spine.set_color("#555")


def r3(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 3)


def by_axis(values, cells, shape):
    """Median of per-elite values per bin of each axis."""
    return {a["name"]: [r3(np.nanmedian(values[cells[:, i] == b])) if (cells[:, i] == b).any() else None
                        for b in range(shape[i])] for i, a in enumerate(atlas.AXES)}


def noise(args):
    A, edges, shape = load_atlas(args.atlas)
    E = A["elites"]
    ec = np.array([e["cell"] for e in E])
    pub = np.array([e["fitness"] for e in E])
    k0 = EPS.index(NOISE_EPS)
    sub = spread_picks(E, shape, sweep_picks(E, shape), args.subset)
    jobs = [(j, -1, 0) for j in range(len(E))]
    jobs += [(j, k0, r) for j in range(len(E)) for r in range(args.runs)]
    jobs += [(j, k, r) for j in sub for k in range(len(EPS)) if k != k0 for r in range(EPS_RUNS)]
    base = [built(e["genome"]) for e in E]
    setups = [base[j] if k < 0 else
              {**base[j], "pos": base[j]["pos"] + np.random.default_rng([j, k, r]).normal(0, EPS[k], base[j]["pos"].shape)}
              for j, k, r in jobs]
    print(f"{len(setups)} runs", flush=True)
    t0 = time.time()
    with polite_pool(args.headroom) as pool:
        ms = probe(pool, setups, args.headroom)
    seconds = time.time() - t0
    jobs = np.array(jobs)
    fit, feat, cells = measured(ms, edges)
    np.savez_compressed(f"{args.out}/noise.npz", jobs=jobs, fitness=fit, features=feat)

    # sanity: the unperturbed runs must reproduce the atlas
    b = jobs[:, 1] < 0
    f0, x0, c0 = fit[b], feat[b], cells[b]
    fmiss = [j for j, e in enumerate(E) if not np.isfinite(f0[j]) or round(float(f0[j]), 4) != e["fitness"]]
    xmiss = [j for j, e in enumerate(E) if [round(float(v), 4) for v in x0[j]] != e["features"]]
    sanity = {"elites": len(E), "fitness_mismatch": [E[j]["cell"] for j in fmiss],
              "features_mismatch": [E[j]["cell"] for j in xmiss],
              "cell_mismatch": [E[j]["cell"] for j in np.where((c0 != ec).any(1))[0]],
              "max_abs_fitness_diff": float(np.nanmax(np.abs(f0 - pub)))}
    print("sanity", sanity, flush=True)

    # every elite at NOISE_EPS; f0, the unperturbed re-run, is the published fitness unrounded
    m = jobs[:, 1] == k0
    F = fit[m].reshape(len(E), args.runs)
    C = cells[m].reshape(len(E), args.runs, 3)
    X = bin_coords(feat[m].reshape(len(E), args.runs, 3), edges)
    mean, std = np.nanmean(F, 1), np.nanstd(F, 1, ddof=1)
    bias = f0 - mean
    leave = (C != ec[:, None]).any(2).mean(1)
    fstd = np.nanstd(X, 1, ddof=1)
    elites = [{"cell": e["cell"], "published": e["fitness"], "mean": r3(mean[j]), "std": r3(std[j]),
               "min": r3(np.nanmin(F[j])), "max": r3(np.nanmax(F[j])), "published_minus_mean": r3(bias[j]),
               "p_cell_change": r3(leave[j]), "broken": int(np.isnan(F[j]).sum()),
               "feature_std_bins": [r3(v) for v in fstd[j]]} for j, e in enumerate(E)]

    # the subset against eps (the first EPS_RUNS of the NOISE_EPS runs are reused): fitness std, mean
    # shift from the unperturbed run, share leaving the cell, feature RMS deviation from the unperturbed run
    S, D, Q = (np.zeros((len(sub), len(EPS))) for _ in range(3))
    XR = np.zeros((len(sub), len(EPS), 3))
    x0b = bin_coords(x0, edges)
    for i, j in enumerate(sub):
        for k in range(len(EPS)):
            sel = (jobs[:, 0] == j) & (jobs[:, 1] == k) & (jobs[:, 2] < EPS_RUNS)
            S[i, k] = np.nanstd(fit[sel], ddof=1)
            D[i, k] = np.nanmean(fit[sel]) - f0[j]
            Q[i, k] = (cells[sel] != ec[j]).any(1).mean()
            XR[i, k] = np.sqrt(np.nanmean((bin_coords(feat[sel], edges) - x0b[j]) ** 2, 0))

    # how close elites sit to a cell edge they could cross, in bins (end bins are open on one side)
    edge = np.stack([np.minimum(np.where(ec[:, a] > 0, x0b[:, a] - ec[:, a], np.inf),
                                np.where(ec[:, a] < shape[a] - 1, ec[:, a] + 1 - x0b[:, a], np.inf))
                     for a in range(3)], 1)
    summary = {
        "seconds": round(seconds), "runs": int(len(jobs)), "headroom": args.headroom, "sanity": sanity,
        "eps": NOISE_EPS, "runs_per_elite": args.runs,
        "std": {"median": r3(np.median(std)), "mean": r3(std.mean()), "p90": r3(np.percentile(std, 90)),
                "max": r3(std.max()), "share_zero": r3((std == 0).mean()), "share_under_0.001": r3((std < 0.001).mean()),
                "share_over_0.01": r3((std > 0.01).mean()), "share_over_0.05": r3((std > 0.05).mean()),
                "by_axis": by_axis(std, ec, shape)},
        "range_median": r3(np.median(np.nanmax(F, 1) - np.nanmin(F, 1))),
        "winners_curse": {"published_minus_mean_median": r3(np.median(bias)), "mean": r3(bias.mean()),
                          "share_positive": r3((bias > 1e-6).mean()), "share_negative": r3((bias < -1e-6).mean()),
                          "p90": r3(np.percentile(bias, 90)), "max": r3(bias.max()),
                          "share_published_above_all_runs": r3((f0 > np.nanmax(F, 1)).mean()),
                          "spearman_vs_std": r3(spearmanr(bias, std).statistic)},
        "p_cell_change": {"median": r3(np.median(leave)), "mean": r3(leave.mean()),
                          "share_over_0.5": r3((leave > 0.5).mean()), "share_zero": r3((leave == 0).mean()),
                          "mean_by_axis": {a["name"]: [r3(leave[ec[:, i] == b].mean()) if (ec[:, i] == b).any() else None
                                                      for b in range(shape[i])] for i, a in enumerate(atlas.AXES)}},
        "feature_std_bins_median": {a["name"]: r3(np.median(fstd[:, i])) for i, a in enumerate(atlas.AXES)},
        "elite_edge_distance_bins": {"nearest_median": r3(np.median(edge.min(1))),
                                     "nearest_p25": r3(np.percentile(edge.min(1), 25)),
                                     "per_axis_median": {a["name"]: r3(np.median(edge[np.isfinite(edge[:, i]), i]))
                                                         for i, a in enumerate(atlas.AXES)}},
        "broken_share": r3(np.isnan(F).mean()),
        "against_eps": {"subset": [E[j]["cell"] for j in sub], "eps": EPS, "runs": EPS_RUNS,
                        "std_median": [r3(v) for v in np.median(S, 0)],
                        "std_p90": [r3(v) for v in np.percentile(S, 90, 0)],
                        "mean_shift_median": [r3(v) for v in np.median(D, 0)],
                        "mean_shift_p10": [r3(v) for v in np.percentile(D, 10, 0)],
                        "share_shift_below_-0.01": [r3(v) for v in (D < -0.01).mean(0)],
                        "p_cell_change_median": [r3(v) for v in np.median(Q, 0)],
                        "feature_rms_bins_median": [[r3(v) for v in row] for row in np.median(XR, 0)]},
    }

    # against phase 2: the logged rerun's replicate spread and published-vs-rerun difference
    if os.path.exists(os.path.join(args.run, "evals.jsonl.gz")):
        print("loading log", flush=True)
        R = json.load(open(os.path.join(args.run, "atlas.json")))
        L, _, lcells = load_run(args.run, edges)
        B = np.stack([replay(L, rep, R["evaluation"]["init"], lcells, shape)[0]
                      for rep in range(R["evaluation"]["replicates"])])
        _, spread, merged = agreement(B)
        sg = grid(ec, std, shape)
        diff = np.abs(grid(ec, pub, shape) - merged)
        s, d = np.isfinite(spread) & np.isfinite(sg), np.isfinite(diff) & np.isfinite(sg)
        summary["phase2"] = {
            "cells_spread": int(s.sum()),
            "replicate_spread_median": r3(np.median(spread[s])),
            "noise_range_of_3_median": r3(np.median(RANGE3 * sg[s])),
            "share_spread_within_noise_range": r3((spread[s] <= RANGE3 * sg[s]).mean()),
            "spearman_spread_vs_noise_std": r3(spearmanr(spread[s], sg[s]).statistic),
            "cells_spread_over_0.1": int((spread[s] > 0.1).sum()),
            "of_which_noise_range_over_0.1": int(((spread[s] > 0.1) & (RANGE3 * sg[s] > 0.1)).sum()),
            "cells_diff": int(d.sum()),
            "rerun_abs_diff_median": r3(np.median(diff[d])),
            "rerun_abs_diff_mean": r3(diff[d].mean()),
            "noise_abs_diff_of_2_median": r3(np.median(ABSDIFF2 * sg[d])),
            "share_diff_within_noise": r3((diff[d] <= ABSDIFF2 * sg[d]).mean()),
            "spearman_abs_diff_vs_noise_std": r3(spearmanr(diff[d], sg[d]).statistic),
        }
    summary["elites"] = elites
    with open(f"{args.out}/noise.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "elites"}, indent=1))

    # figures
    sg = grid(ec, std, shape)
    hot = ListedColormap(plt.get_cmap("inferno")(np.linspace(0, 0.75, 256)))   # white numbers stay readable
    slice_figure(sg, f"Fitness std over {args.runs} runs with start positions perturbed by {NOISE_EPS:g}",
                 hot, 0, 0.05, labels=np.where(np.isfinite(sg), np.round(1000 * np.nan_to_num(sg)).astype(int)
                                               .astype(str), ""),
                 path=f"{args.out}/noise.png", note="number = std x 1000")
    lg = grid(ec, leave, shape)
    slice_figure(lg, f"Share of the {args.runs} perturbed runs ({NOISE_EPS:g}) that leave the elite's cell",
                 hot, 0, 1, labels=np.where(np.isfinite(lg), np.round(100 * np.nan_to_num(lg)).astype(int)
                                            .astype(str), ""),
                 path=f"{args.out}/noise_cells.png", note="number = % of runs")

    fig, axs = plt.subplots(1, 4, figsize=(20, 4.8), facecolor=DARK)
    panels = [(S, "fitness std", "log"), (D, "perturbed mean - unperturbed fitness", "linear"),
              (Q, "share of runs leaving the cell", "linear"),
              (XR.max(2), "largest feature RMS change (bins)", "log")]
    for ax, (Y, label, scale) in zip(axs, panels):
        for y in Y:
            ax.plot(EPS, np.maximum(y, 1e-5) if scale == "log" else y, color="#888", lw=0.6, alpha=0.6)
        med = np.median(Y, 0)
        ax.plot(EPS, np.maximum(med, 1e-5) if scale == "log" else med, color="#ffb000", lw=2.5, marker="o",
                label=f"median of {len(sub)} elites")
        ax.set_xscale("log")
        ax.set_yscale(scale)
        dark(ax, "start position noise eps (units)", label)
    axs[0].legend(fontsize=8)
    fig.suptitle(f"Effect against perturbation scale ({EPS_RUNS} runs per elite and scale; positions are O(1-20) "
                 "units; log panels floored at 1e-5)", color="w")
    fig.tight_layout()
    fig.savefig(f"{args.out}/noise_eps.png", dpi=110, facecolor=DARK)
    plt.close(fig)

    big = EPS[-1]
    fig, axs = plt.subplots(1, 3, figsize=(17, 5), facecolor=DARK)
    ax = axs[0]
    ax.errorbar(f0, mean, yerr=std, fmt="none", ecolor="#555", lw=0.6)
    ax.scatter(f0, mean, s=9, color="#4c9be8", zorder=3, label=f"every elite, eps {NOISE_EPS:g}")
    ax.scatter(f0[sub], f0[sub] + D[:, -1], s=14, color="#ffb000", zorder=4, label=f"subset, eps {big:g}")
    ax.plot([0, 1], [0, 1], color="#bbb", lw=0.8, ls="--")
    dark(ax, "published fitness", "mean over perturbed runs (bar: std)")
    ax.legend(fontsize=8)
    ax = axs[1]
    bins = np.linspace(-0.05, 0.5, 45)
    ax.hist(np.clip(bias, bins[0], bins[-1]), bins=bins, color="#4c9be8", label=f"every elite, eps {NOISE_EPS:g}")
    ax.hist(np.clip(-D[:, -1], bins[0], bins[-1]), bins=bins, color="#ffb000", alpha=0.8,
            label=f"subset, eps {big:g}")
    ax.axvline(0, color="#bbb", lw=0.8)
    ax.set_yscale("log")
    dark(ax, "published - perturbed mean (clipped to the axis)", "elites")
    ax.legend(fontsize=8)
    ax = axs[2]
    ax.scatter(std, bias, s=9, color="#4c9be8")
    ax.axhline(0, color="#bbb", lw=0.8)
    dark(ax, f"fitness std over perturbed runs (eps {NOISE_EPS:g})", "published - perturbed mean",
         f"Spearman {spearmanr(bias, std).statistic:.2f}")
    fig.suptitle("Winner's curse: published fitness against perturbed re-runs", color="w")
    fig.tight_layout()
    fig.savefig(f"{args.out}/winners_curse.png", dpi=110, facecolor=DARK)
    plt.close(fig)


def plane_metrics(F, C, axis, e, floor):
    """Metrics of one pull x push plane: F (n, n) fitness and C (n, n, 3) cells, indexed [push, pull].
    Returns them, the grid points in the elite's cell, and the part of those connected to the elite."""
    cell, f0, g = np.array(e["cell"]), e["fitness"], e["genome"]
    x0, y0 = g["log2_pull"], g["log2_push"]
    xx, yy = np.meshgrid(axis, axis)
    step = axis[1] - axis[0]
    ok = np.isfinite(F)
    same = (C == cell).all(-1)
    near = np.hypot(xx - x0, yy - y0) <= atlas.SIGMA_LOG2
    lab, _ = ndimage.label(same)
    seeds = same & (np.abs(xx - x0) <= step) & (np.abs(yy - y0) <= step)
    region = same & np.isin(lab, lab[seeds])

    def top(mask):
        return r3(F[mask & ok].max()) if (mask & ok).any() else None

    def lagged(lag):
        return np.concatenate([np.abs(F[:, lag:] - F[:, :-lag]).ravel(), np.abs(F[lag:] - F[:-lag]).ravel()])

    def explained(key):
        """Share of the plane's fitness variance explained by the mean over grid points sharing key."""
        k, f = key[ok], F[ok]
        mean = np.bincount(k, f) / np.maximum(np.bincount(k), 1)
        return r3(1 - ((f - mean[k]) ** 2).sum() / ((f - f.mean()) ** 2).sum())

    iy, ix = np.indices(F.shape)
    i = np.unravel_index(np.nanargmax(F), F.shape)
    sigma = floor.get("std")
    rough = float(np.nanmean(lagged(1)))
    change = np.concatenate([(C[:, 1:] != C[:, :-1]).any(-1).ravel(), (C[1:] != C[:-1]).any(-1).ravel()])
    m = {"cell": e["cell"], "fitness": f0, "noise_mean": floor.get("mean"), "noise_std": sigma,
         "log2_pull": round(x0, 3), "log2_push": round(y0, 3),
         "plane_max": r3(F[i]), "plane_max_at": [r3(axis[i[1]]), r3(axis[i[0]])],
         "plane_max_cell": C[i].tolist(),
         "share_above_elite": r3((F[ok] > f0).mean()),
         "share_above_noise_mean": r3((F[ok] > floor["mean"]).mean()) if floor else None,
         "cell_max": top(same), "share_of_cell_above_elite": r3((F[same & ok] > f0).mean()) if same.any() else None,
         "near_max": top(near), "near_cell_max": top(near & same),
         "share_within_0.05": r3((np.abs(F[ok] - f0) <= 0.05).mean()),
         "cell_points": int(same.sum()), "cell_region_points": int(region.sum()),
         "cell_share": round(float(same.mean()), 4), "cell_region_connected": round(float(region.mean()), 4),
         "cells_reached": int(len({tuple(c) for c in C[ok]})),
         "roughness": r3(rough),
         # against the |difference| of two noisy draws; noise.json's stds resolve 0.001, so below that this is a floor
         "roughness_over_noise": r3(rough / (ABSDIFF2 * max(sigma, 0.001))) if sigma is not None else None,
         "variogram": {lag: r3(np.nanmean(lagged(lag))) for lag in (1, 2, 4, 8, 16) if lag < len(axis)},
         "cell_change_adjacent": r3(change.mean()), "broken": r3((~ok).mean()),
         "variance_explained": {"push_minus_pull": explained(iy - ix + len(axis)), "pull": explained(ix),
                                "push": explained(iy), "push_plus_pull": explained(iy + ix)}}
    m["cell_gap"] = r3(m["cell_max"] - f0) if m["cell_max"] is not None else None
    m["optimum"] = {"plane": f0 >= m["plane_max"],
                    "cell": m["cell_max"] is None or f0 >= m["cell_max"],
                    "near_in_cell": m["near_cell_max"] is None or f0 >= m["near_cell_max"]}
    return m, same, region


def lettered(F, axis, e, n=9):
    """n points to show: the elite's own (pull, push), the plane's best grid point, then grid points
    farthest from those so far (in log2 units)."""
    xx, yy = np.meshgrid(axis, axis)
    pts = np.c_[xx.ravel(), yy.ravel()]
    i = np.unravel_index(np.nanargmax(F), F.shape)
    picks = [(e["genome"]["log2_pull"], e["genome"]["log2_push"]), (axis[i[1]], axis[i[0]])]
    d = np.linalg.norm(pts[:, None] - np.array(picks)[None], axis=2).min(1)
    while len(picks) < n:
        j = int(d.argmax())
        picks.append(tuple(pts[j]))
        d = np.minimum(d, np.linalg.norm(pts - pts[j], axis=1))
    return picks


def still(data, half):
    """atlas.render's still of a trajectory, as an image array."""
    buf = io.BytesIO()
    draw(data, half, "hsv", buf, size=2, dpi=100)
    buf.seek(0)
    return plt.imread(buf)


def outlines(C, axis, color="w", lw=0.6):
    """Segments between neighbouring grid points of different cells (C indexed [push, pull])."""
    h = (axis[1] - axis[0]) / 2
    segs = [[((axis[x] + axis[x + 1]) / 2, axis[y] - h), ((axis[x] + axis[x + 1]) / 2, axis[y] + h)]
            for y, x in zip(*np.where((C[:, 1:] != C[:, :-1]).any(-1)))]
    segs += [[(axis[x] - h, (axis[y] + axis[y + 1]) / 2), (axis[x] + h, (axis[y] + axis[y + 1]) / 2)]
             for y, x in zip(*np.where((C[1:] != C[:-1]).any(-1)))]
    return LineCollection(segs, colors=color, linewidths=lw)


def plane(ax, F, C, axis, e, cmap="magma", vmin=0, vmax=FMAX, lines=True, ms=14):
    """A plane's values F with cell outlines and the elite as a star."""
    h = (axis[1] - axis[0]) / 2
    cm = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap
    cm.set_bad("#2a2a2a")
    im = ax.imshow(F, origin="lower", cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest",
                   extent=(axis[0] - h, axis[-1] + h, axis[0] - h, axis[-1] + h))
    if lines:
        ax.add_collection(outlines(C, axis, lw=0.5 if ms < 10 else 0.7))
    ax.plot(e["genome"]["log2_pull"], e["genome"]["log2_push"], marker="*", ms=ms, mfc="none", mec="#00e5ff",
            mew=1.6)
    return im


def plane_figure(path, e, axis, F, C, same, region, marks, stills, m, shape):
    cell = e["cell"]
    fig = plt.figure(figsize=(18, 12.4), facecolor=DARK)
    gs = fig.add_gridspec(3, 18, height_ratios=[4, 4, 2.9], hspace=0.32, wspace=0.8, left=0.04, right=0.97,
                          top=0.9, bottom=0.01)
    ax = fig.add_subplot(gs[:2, :8])
    im = plane(ax, F, C, axis, e, ms=18)
    for i, (letter, (x, y)) in enumerate(zip("ABCDEFGHI", marks)):
        ax.plot(x, y, "o", ms=4, mfc="w", mec="black", mew=0.8)
        dy = 8 if (y < 1.6) != (i == 0 and y > -1.6) else -8     # A below its point, so B can sit next to it
        ax.annotate(letter, (x, y), xytext=(8 if x < 1.6 else -8, dy), textcoords="offset points",
                    color="w", fontsize=12, fontweight="bold", ha="center", va="center",
                    path_effects=[withStroke(linewidth=3, foreground="black")])
    dark(ax, "log2_pull", "log2_push", "fitness; white lines: the cell changes; star: the elite")
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02).ax.yaxis.set_tick_params(color="w", labelcolor="w")
    slots = [gs[0, 8:13], gs[0, 13:18], gs[1, 8:13], gs[1, 13:18]]
    for a, (slot, axis_def) in enumerate(zip(slots, atlas.AXES)):
        ax = fig.add_subplot(slot)
        nb = shape[a]
        bins = np.where(np.isfinite(F), C[..., a], np.nan)
        im = plane(ax, bins, C, axis, e, plt.get_cmap("viridis", nb), -0.5, nb - 0.5, lines=False, ms=12)
        dark(ax, title=f"{axis_def['name']} bin ({axis_def['label']}); elite {cell[a]}")
        cb = fig.colorbar(im, ax=ax, ticks=range(nb), shrink=0.9, pad=0.02)
        cb.ax.yaxis.set_tick_params(color="w", labelcolor="w")
    ax = fig.add_subplot(slots[3])
    plane(ax, np.where(region, 2, same.astype(float)), C, axis, e,
          ListedColormap(["#333", "#7a5cc0", "#ffb000"]), -0.5, 2.5, lines=False, ms=12)
    dark(ax, title=f"cell {cell}: {same.sum()} of {same.size} points (purple and orange),\n"
                   f"{region.sum()} connected to the elite (orange); the plane reaches {m['cells_reached']} cells")
    for i, (letter, img, (fit, c)) in enumerate(zip("ABCDEFGHI", *stills)):
        ax = fig.add_subplot(gs[2, 2 * i:2 * i + 2])
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        same_cell = c is not None and list(c) == list(cell)
        name = "elite" if i == 0 else "best" if i == 1 else ""
        ax.set_title(f"{letter} {name} {fit:.2f} {list(c) if c is not None else 'broken'}",
                     color="w" if same_cell else "#ff9a3c", fontsize=9)
    noise_txt = f" (perturbed {m['noise_mean']:.3f} ± {m['noise_std']:.3f})" if m["noise_std"] is not None else ""
    ratio = m["roughness_over_noise"]
    rough_txt = "" if ratio is None else f" ({'≥ ' if m['noise_std'] < 0.001 else ''}{ratio:.0f}x noise)"
    fig.suptitle(f"Cell {cell}: log2_pull x log2_push, everything else fixed\n"
                 f"elite {m['fitness']:.3f}{noise_txt} · plane max {m['plane_max']:.3f} · best in cell "
                 f"{m['cell_max'] or np.nan:.3f} · roughness (mean |Δ fitness| between neighbours) "
                 f"{m['roughness']:.3f}{rough_txt} · orange still titles: another cell", color="w", fontsize=12)
    fig.savefig(path, dpi=90, facecolor=DARK)
    plt.close(fig)


def sweep(args):
    A, edges, shape = load_atlas(args.atlas)
    E = A["elites"]
    picks = sweep_picks(E, shape)
    n = args.size
    axis = np.linspace(*P.LOG2_RANGE, n)
    npath = os.path.join(args.out, "noise.json")
    floors = {tuple(e["cell"]): e for e in json.load(open(npath))["elites"]} if os.path.exists(npath) else {}
    setups = []
    for j in picks:
        s = built(E[j]["genome"])
        setups += [{**s, "g": P.G * 2.0 ** x, "c": P.C * 2.0 ** y} for y in axis for x in axis]
    print(f"{len(picks)} planes, {len(setups)} runs", flush=True)
    t0 = time.time()
    with polite_pool(args.headroom) as pool:
        fit, feat, cells = measured(probe(pool, setups, args.headroom), edges)
        seconds = time.time() - t0
        F = fit.reshape(len(picks), n, n)
        C = cells.reshape(len(picks), n, n, 3)
        np.savez_compressed(f"{args.out}/sweeps.npz", picks=np.array([E[j]["cell"] for j in picks]), axis=axis,
                            fitness=F, features=feat.reshape(len(picks), n, n, 3))
        marks = [lettered(F[p], axis, E[j]) for p, j in enumerate(picks)]
        shows = [{**built(E[j]["genome"]), **({} if i == 0 else {"g": P.G * 2.0 ** x, "c": P.C * 2.0 ** y})}
                 for j, mk in zip(picks, marks) for i, (x, y) in enumerate(mk)]
        ms, data = probe(pool, shows, args.headroom, keep=True)

    os.makedirs(f"{args.out}/sweeps", exist_ok=True)
    planes = []
    k = 0
    for p, j in enumerate(picks):
        e = E[j]
        m, same, region = plane_metrics(F[p], C[p], axis, e, floors.get(tuple(e["cell"]), {}))
        planes.append(m)
        sm = ms[k:k + len(marks[p])]
        imgs = [still(d, r["half"]) if r else np.zeros((2, 2, 3)) for d, r in zip(data[k:k + len(marks[p])], sm)]
        info = [(r["fitness"], atlas.cell_of(r["features"], edges)) if r else (np.nan, None) for r in sm]
        k += len(marks[p])
        name = "-".join(map(str, e["cell"]))
        plane_figure(f"{args.out}/sweeps/{name}.png", e, axis, F[p], C[p], same, region, marks[p], (imgs, info),
                     m, shape)
        print(f"  {e['cell']}: elite {m['fitness']:.3f} plane max {m['plane_max']:.3f} cells {m['cells_reached']} "
              f"roughness {m['roughness']:.3f} ({m['roughness_over_noise']}x noise)", flush=True)

    rows, cols = -(-len(picks) // 5), 5
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4.3 * rows), facecolor=DARK)
    for ax, p, m in zip(axs.flat, range(len(picks)), planes):
        im = plane(ax, F[p], C[p], axis, E[picks[p]], ms=10)
        fav = " *" if tuple(m["cell"]) in FAVOURITES else ""
        dark(ax, title=f"{m['cell']}{fav}: elite {m['fitness']:.2f}, max {m['plane_max']:.2f}, "
                       f"{m['cells_reached']} cells")
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
    for ax in axs.flat[len(picks):]:
        ax.axis("off")
    fig.suptitle("Fitness over log2_pull (across) x log2_push (up) per elite; white lines: the cell changes; "
                 "star: the elite; * user favourite", color="w", fontsize=13)
    fig.tight_layout(rect=(0, 0, 0.94, 0.97))
    cb = fig.colorbar(im, ax=axs, shrink=0.5, pad=0.01)
    cb.ax.yaxis.set_tick_params(color="w", labelcolor="w")
    fig.savefig(f"{args.out}/sweeps.png", dpi=80, facecolor=DARK)
    plt.close(fig)

    med = lambda key: r3(np.median([m[key] for m in planes if m[key] is not None]))
    # neighbouring grid points over all planes, grouped by the more chaotic one's chaos bin
    pairs = lambda V: [(V[:, :, 1:], V[:, :, :-1]), (V[:, 1:], V[:, :-1])]
    dF = np.concatenate([np.abs(a - b).ravel() for a, b in pairs(F)])
    moved = np.concatenate([(a != b).any(-1).ravel() for a, b in pairs(C)])
    chaos = np.concatenate([np.maximum(a, b).ravel() for a, b in pairs(C[..., 0])])
    ok = np.isfinite(dF)
    summary = {
        "seconds": round(seconds), "runs": len(setups) + len(shows), "grid": n, "range": list(P.LOG2_RANGE),
        "headroom": args.headroom,
        "selection": "the five favourites, then farthest-point picks over (chaos, density, branching, fitness) "
                     "scaled to 0..1 (landscape.spread_picks)",
        "median": {key: med(key) for key in ("roughness", "roughness_over_noise", "noise_std", "cells_reached",
                                             "cell_change_adjacent", "cell_points", "cell_region_points",
                                             "share_within_0.05", "share_above_elite", "share_above_noise_mean",
                                             "cell_gap")},
        "variogram_median": {lag: r3(np.median([m["variogram"][lag] for m in planes])) for lag in planes[0]["variogram"]},
        "variance_explained_median": {k: r3(np.median([m["variance_explained"][k] for m in planes]))
                                      for k in planes[0]["variance_explained"]},
        "elite_optimum": {k: int(sum(m["optimum"][k] for m in planes)) for k in ("plane", "cell", "near_in_cell")},
        "cell_gap_within_0.01": int(sum(m["cell_gap"] is not None and m["cell_gap"] <= 0.01 for m in planes)),
        "by_chaos_bin": {"pairs": [int((ok & (chaos == b)).sum()) for b in range(shape[0])],
                         "roughness": [r3(dF[ok & (chaos == b)].mean()) if (ok & (chaos == b)).any() else None
                                       for b in range(shape[0])],
                         "cell_change": [r3(moved[ok & (chaos == b)].mean()) if (ok & (chaos == b)).any() else None
                                         for b in range(shape[0])]},
        "planes": planes,
    }
    with open(f"{args.out}/sweeps.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "planes"}, indent=1))


# ---- structure: is the dense rows' cap real or measured? ----

def random_records(run):
    """The finite random-genome evaluations of a --log run, in log order."""
    out = []
    with gzip.open(os.path.join(run, "evals.jsonl.gz"), "rt") as f:
        for line in f:
            if '"random"' in line[:120]:
                r = json.loads(line)
                if r["kind"] == "random" and r.get("fitness") is not None:
                    out.append(r)
    return out


def structure_frame(data):
    """The fade and frame half-width atlas.measure uses."""
    T, N, _ = data.shape
    alpha = 1 - np.linspace(0, 1, T)
    w = np.repeat(alpha, N)
    r = np.hypot(data[..., 0], data[..., 1]).ravel()
    order = np.argsort(r)
    return alpha, 1.1 * r[order][np.searchsorted(np.cumsum(w[order]) / w.sum(), 0.95)]


def polar_trails(p, w, sectors, rings=16):
    """atlas._structure's polar histogram of trail points p, weights w (its rings, sectors and ring
    weighting), at any number of sectors."""
    c = p - atlas.RES / 2
    r = np.hypot(c[:, 0], c[:, 1]) / (atlas.RES / 2)
    ok = r < 1
    ring = (r[ok] * rings).astype(np.int64)
    sector = ((np.arctan2(c[ok, 1], c[ok, 0]) / (2 * np.pi) + 0.5) * sectors).astype(np.int64) % sectors
    polar = np.bincount(ring * sectors + sector, w[ok], rings * sectors).reshape(rings, sectors)
    return polar / np.sqrt((np.arange(rings) + 0.5) / rings)[:, None]


def angular_energy(polar, weights=None):
    """Energy per angular harmonic 0..sectors/2 summed over rings (one-sided: harmonics > 0 doubled)."""
    e = np.abs(np.fft.rfft(polar, axis=1)) ** 2
    e[:, 1:-1] *= 2
    return (e if weights is None else e * weights[:, None]).sum(0)


_MULTIPLES = {}


def symmetric_share(E, M, K):
    """(S, a, s) as atlas._structure computes them from energies E over harmonics 0..M and k = 2..K:
    a the angular share of the energy, s how far above chance it sits on the best k's multiples."""
    if (M, K) not in _MULTIPLES:
        on = (np.arange(1, M + 1)[None] % np.arange(2, K + 1)[:, None] == 0).astype(float)
        _MULTIPLES[M, K] = on, on.mean(1)
    on, chance = _MULTIPLES[M, K]
    E = E[:M + 1]
    ang = E[1:].sum()
    if ang <= 0:
        return 0.0, 0.0, 0.0
    s = max(float(((on @ E[1:] / ang - chance) / (1 - chance)).max()), 0.0)
    return ang / E.sum() * s, ang / E.sum(), s


def rotated_correlation(r, polar):
    """The independent check on a polar resampling (radii r) of the saturated trail image: its angular
    part (ring means removed) correlated with itself turned by 2pi/k, area-weighted. Returns
    Q = max over k = 2..128 of C(2pi/k) - max(C(pi/k), 0) (1 for k thin rays, ~0 for noise or for
    variation too smooth to tell the two turns apart) and the angular share of the disc's variance."""
    J = polar - polar.mean(1, keepdims=True)
    total = (r[:, None] * (polar - (r[:, None] * polar).sum() / (r.sum() * polar.shape[1])) ** 2).sum()
    n = polar.shape[1]
    ac = np.fft.irfft((np.abs(np.fft.rfft(J, axis=1)) ** 2 * r[:, None]).sum(0), n=n)
    if ac[0] <= 0 or total <= 0:
        return 0.0, 0.0
    ac = np.append(ac / ac[0], 1.0)
    ks = np.arange(2, 129)

    def C(lag):
        i = np.floor(lag).astype(int)
        return ac[i] * (1 - (lag - i)) + ac[i + 1] * (lag - i)

    return float((C(n / ks) - np.maximum(C(n / (2 * ks)), 0)).max()), float((r[:, None] * J ** 2).sum() / total)


def structure_variants(data):
    """A trajectory's structure under each variant: published (_structure: 64 sectors, harmonics 1..12,
    k 2..6), all (STRUCT_SECTORS sectors, every harmonic, k 2..STRUCT_K), image (the saturated trail
    image resampled on STRUCT_RADII x STRUCT_ANGLES, every harmonic, k 2..STRUCT_K) and independent
    (rotated_correlation's Q x angular share); plus the all-harmonics spectrum over its m = 0 energy."""
    alpha, half = structure_frame(data)
    p, pw = atlas._trails(data, alpha, half, atlas.RES)
    E64 = angular_energy(polar_trails(p, pw, 64))
    Eall = angular_energy(polar_trails(p, pw, STRUCT_SECTORS))
    ink = 1 - np.exp(-2 * atlas._splat(p, pw, atlas.RES))
    c = atlas.RES / 2 - 0.5
    r = np.linspace(1, atlas.RES / 2 - 1, STRUCT_RADII)
    th = np.arange(STRUCT_ANGLES) * 2 * np.pi / STRUCT_ANGLES
    polar = ndimage.map_coordinates(ink, [c + r[:, None] * np.cos(th), c + r[:, None] * np.sin(th)], order=1)
    out = {}
    for name, (E, M, K) in {"published": (E64, atlas.HARMONICS, 6), "all": (Eall, STRUCT_SECTORS // 2, STRUCT_K),
                            "image": (angular_energy(polar, r), STRUCT_ANGLES // 2, STRUCT_K)}.items():
        S, a, s = symmetric_share(E, M, K)
        out[name] = {"S": S, "a": a, "s": s, "structure": 1 - np.exp(-S / atlas.STRUCTURE)}
    Q, ang = rotated_correlation(r, polar)
    out["independent"] = {"Q": Q, "angular": ang, "structure": Q * ang}
    out["spectrum"] = Eall / max(Eall[0], 1e-300)
    return out


def scrambled(data, rng):
    """Each particle's whole path turned by its own random angle: the same trails, no shared symmetry."""
    th = rng.uniform(0, 2 * np.pi, data.shape[1])
    c, s = np.cos(th), np.sin(th)
    return np.stack([c * data[..., 0] - s * data[..., 1], s * data[..., 0] + c * data[..., 1]], -1)


def _structure_pair(task):
    """structure_variants of a trajectory and of its scrambled null (seeded by the world's index)."""
    i, data = task
    return structure_variants(data), structure_variants(scrambled(data, np.random.default_rng(i)))


def stack_variants(ws):
    return {v: {k: np.array([w[v][k] for w in ws]) for k in ws[0][v]} for v in STRUCT_VARIANTS + ["independent"]}


def by_row(rows, v, n, f=np.mean):
    return np.array([f(v[rows == d]) if np.any(rows == d) else np.nan for d in range(n)])


def pct95(v):
    return np.percentile(v, 95)


def dark_legend(ax, **kw):
    ax.legend(facecolor="#222", edgecolor="#444", labelcolor="w", fontsize=8, **kw)


def structure_figures(out, n, W, N, rows, We, erows, examples):
    x = np.arange(1, n + 1)   # density rows labelled 1..n, like slice_figure's bins
    xlabel = "density row (sparse → dense)"

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=DARK)
    for ax, (title, V, rw) in zip(axs, [("published elites", We, erows), ("random genomes", W, rows)]):
        for var, ls in (("published", "-"), ("all", "--")):
            c = STRUCT_COLORS[var]
            ax.plot(x, by_row(rw, V[var]["a"], n), ls, color=c, lw=2, marker="o", ms=4, label=f"a ({var})")
            ax.plot(x, by_row(rw, V[var]["s"], n), ls, color=c, lw=1, marker="s", ms=3, alpha=0.7, label=f"s ({var})")
            ax.plot(x, by_row(rw, V[var]["structure"], n), ls, color="w" if var == "published" else "#aaa", lw=1.5,
                    label=f"structure ({var})")
        dark(ax, xlabel, "mean", f"{title}: S = a × s by density row")
        ax.set_ylim(0, 1.02)
        dark_legend(ax, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{out}/structure_decomposition.png", dpi=110, facecolor=DARK)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6), facecolor=DARK)
    for var in STRUCT_VARIANTS + ["independent"]:
        c = STRUCT_COLORS[var]
        for ax, f in ((axs[0], pct95), (axs[1], np.mean)):
            ax.plot(x, by_row(rows, W[var]["structure"], n, f), color=c, lw=2, marker="o", ms=4, label=var)
            ax.plot(x, by_row(rows, N[var]["structure"], n, f), ":", color=c, lw=1)
        axs[2].plot(x, by_row(erows, We[var]["structure"], n), color=c, lw=2, marker="o", ms=4, label=var)
    for ax, title in zip(axs, ["random genomes: 95th percentile (dotted: symmetry scrambled)",
                               "random genomes: mean (dotted: symmetry scrambled)", "published elites: mean"]):
        dark(ax, xlabel, "structure", title)
        ax.set_ylim(0, 1.02)
        dark_legend(ax, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{out}/structure_ceiling.png", dpi=110, facecolor=DARK)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK, layout="constrained")
    ind = W["independent"]["structure"]
    for ax, var in zip(axs, STRUCT_VARIANTS):
        sc = ax.scatter(ind, W[var]["structure"], c=rows + 1, cmap="viridis", s=6, vmin=1, vmax=n)
        rs = [spearmanr(ind[m], W[var]["structure"][m])[0] for m in (rows < n // 2, rows >= n - 3)]
        dark(ax, "independent: Q × angular share", f"structure ({var})",
             f"{var}: Spearman rows 1-{n // 2} {rs[0]:.2f}, rows {n - 2}-{n} {rs[1]:.2f}")
    cb = fig.colorbar(sc, ax=axs, fraction=0.02)
    cb.set_label("density row", color="w")
    cb.ax.yaxis.set_tick_params(color="w", labelcolor="w")
    fig.savefig(f"{out}/structure_agreement.png", dpi=110, facecolor=DARK)
    plt.close(fig)

    fig, axs = plt.subplots(2, len(examples), figsize=(4.4 * len(examples), 8.4), facecolor=DARK,
                            gridspec_kw={"height_ratios": [1.15, 1]})
    for j, (title, img, w) in enumerate(examples):
        axs[0, j].imshow(img)
        axs[0, j].axis("off")
        axs[0, j].set_title(title, color="w", fontsize=9)
        ax = axs[1, j]
        ax.semilogy(np.arange(1, len(w["spectrum"])), np.maximum(w["spectrum"][1:], 1e-6), color="#4c9be8", lw=0.8)
        ax.axvspan(0.5, atlas.HARMONICS + 0.5, color="#e8684c", alpha=0.25, label="harmonics _structure reads")
        ax.set_ylim(1e-5, 3)
        ax.text(0.98, 0.97, "\n".join(f"{v:>11}: structure {w[v]['structure']:.2f}" +
                                      (f"  a {w[v]['a']:.2f} s {w[v]['s']:.2f}" if v != "independent" else "")
                                      for v in STRUCT_VARIANTS + ["independent"]),
                transform=ax.transAxes, ha="right", va="top", color="w", fontsize=7.5, family="monospace")
        dark(ax, "angular harmonic m", "energy / energy at m = 0" if j == 0 else "")
        if j == 0:
            dark_legend(ax, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{out}/structure_examples.png", dpi=100, facecolor=DARK)
    plt.close(fig)


def structure(args):
    A, edges, shape = load_atlas(args.atlas)
    E = A["elites"]
    n = shape[1]
    erows = np.array([e["cell"][1] for e in E])
    R = random_records(args.run)
    rrows = np.searchsorted(edges[1], np.array([r["features"][1] for r in R]))
    rng = np.random.default_rng(0)
    pick = np.concatenate([rng.choice(np.flatnonzero(rrows == d), args.per_row, replace=False) for d in range(n)])
    rows = rrows[pick]
    recs = E + [R[i] for i in pick]
    print(f"{len(E)} elites and {len(pick)} of {len(R)} random genomes", flush=True)

    with polite_pool(args.headroom) as pool:
        ms, data = probe(pool, [built(r["genome"]) for r in recs], args.headroom, keep=True)
        pairs = pool.map(_structure_pair, enumerate(data), chunksize=8)
    worst = max(abs(m["fitness"] - r["fitness"]) for m, r in zip(ms, recs))
    pub = np.array([p[0]["published"]["structure"] for p in pairs])
    worst_s = float(np.abs(pub - [r["terms"]["structure"] for r in recs]).max())
    print(f"largest |re-measured - logged| fitness {worst:.1e}, structure term {worst_s:.1e}", flush=True)
    ne = len(E)
    We, Ne = stack_variants([p[0] for p in pairs[:ne]]), stack_variants([p[1] for p in pairs[:ne]])
    W, N = stack_variants([p[0] for p in pairs[ne:]]), stack_variants([p[1] for p in pairs[ne:]])

    fit = np.array([r["fitness"] for r in recs])
    refit = fit * (0.4 + 0.6 * np.r_[We["all"]["structure"], W["all"]["structure"]]) / (0.4 + 0.6 * pub)
    fe, ge, fr, gr = fit[:ne], refit[:ne], fit[ne:], refit[ne:]
    fill = np.array([R[i]["features"][1] for i in pick])
    dense = erows >= n - 3
    rowl = lambda rw, v, f=np.mean: [r3(x) for x in by_row(rw, v, n, f)]
    summary = {"random_per_row": args.per_row, "max_abs_fitness_diff": worst, "max_abs_structure_diff": worst_s}
    for key, V, rw in (("elites", We, erows), ("random", W, rows), ("null_elites", Ne, erows), ("null_random", N, rows)):
        summary[key] = {v: {k: rowl(rw, V[v][k]) for k in V[v]} for v in V}
        if key.endswith("random"):
            for v in V:
                summary[key][v]["structure_p95"] = rowl(rw, V[v]["structure"], pct95)
    ind = W["independent"]["structure"]
    summary["spearman_with_independent"] = {
        v: {f"rows_1_{n // 2}": r3(spearmanr(ind[rows < n // 2], W[v]["structure"][rows < n // 2])[0]),
            f"rows_{n - 2}_{n}": r3(spearmanr(ind[rows >= n - 3], W[v]["structure"][rows >= n - 3])[0])}
        for v in STRUCT_VARIANTS}
    summary["spearman_fill_structure_random"] = {v: r3(spearmanr(fill, W[v]["structure"])[0]) for v in W}
    summary["random_share_structure_ge_0.8"] = {v: rowl(rows, (W[v]["structure"] >= 0.8).astype(float))
                                                for v in STRUCT_VARIANTS}
    top = int(np.argmax(ge - fe))
    summary["elites_with_all_harmonics"] = {
        "mean_fitness": {"published": rowl(erows, fe), "all": rowl(erows, ge)},
        "dense_rows": int(dense.sum()),
        "dense_fitness_gain_over": {str(t): int(((ge - fe)[dense] > t).sum()) for t in (0.05, 0.1, 0.2)},
        "dense_structure_gain_over_0.5": int(((We["all"]["structure"] - We["published"]["structure"])[dense] > 0.5).sum()),
        "largest_gain": {"cell": E[top]["cell"], "desc": E[top]["desc"], "fitness": [r3(fe[top]), r3(ge[top])]}}
    summary["random_best_fitness"] = {"published": rowl(rows, fr, np.max), "all": rowl(rows, gr, np.max)}

    cells = [tuple(e["cell"]) for e in E]
    examples, summary["examples"] = [], {}
    for c in STRUCT_EXAMPLES:
        i = cells.index(c)
        examples.append((f"{'-'.join(map(str, c))}  {E[i]['desc'][:34]}\nfill {E[i]['features'][1]:.2f}, "
                         f"fitness {fe[i]:.2f} → {ge[i]:.2f} with all harmonics", still(data[i], ms[i]["half"]),
                         pairs[i][0]))
        summary["examples"]["-".join(map(str, c))] = {
            "desc": E[i]["desc"], "fitness": [r3(fe[i]), r3(ge[i])],
            **{v: {k: r3(x) for k, x in pairs[i][0][v].items()} for v in STRUCT_VARIANTS + ["independent"]}}
    j = int(np.argmax(np.where(rows >= n - 3, W["all"]["structure"] - W["published"]["structure"], -np.inf)))
    desc = P.build(recs[ne + j]["genome"])["desc"]
    examples.append((f"random genome, row {rows[j] + 1}  {desc[:30]}\n"
                     f"fill {fill[j]:.2f}, fitness {fr[j]:.2f} → {gr[j]:.2f} with all harmonics",
                     still(data[ne + j], ms[ne + j]["half"]), pairs[ne + j][0]))
    summary["examples"]["random"] = {"desc": desc, "row": int(rows[j]) + 1, "fitness": [r3(fr[j]), r3(gr[j])],
                                     **{v: {k: r3(x) for k, x in pairs[ne + j][0][v].items()}
                                        for v in STRUCT_VARIANTS + ["independent"]}}

    structure_figures(args.out, n, W, N, rows, We, erows, examples)
    with open(f"{args.out}/structure.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("spearman_with_independent", "spearman_fill_structure_random",
                                              "elites_with_all_harmonics")}, indent=1))


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--atlas", default="search/atlas.json")
    common.add_argument("--out", default="search/landscape")
    run = argparse.ArgumentParser(add_help=False)
    run.add_argument("--run", default="search/landscape/run")
    gpu = argparse.ArgumentParser(add_help=False)
    gpu.add_argument("--headroom", type=float, default=HEADROOM, help="share of CPU and GPU to leave free")
    ap = argparse.ArgumentParser(description="Probe the fitness landscape behind the atlas.")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("analyse", parents=[common, run], help="analyse a logged atlas run")
    p = sub.add_parser("noise", parents=[common, run, gpu], help="the chaos noise floor of the published elites")
    p.add_argument("--runs", type=int, default=32)
    p.add_argument("--subset", type=int, default=40)
    p = sub.add_parser("sweep", parents=[common, gpu], help="pull x push planes through chosen elites")
    p.add_argument("--size", type=int, default=48)
    p = sub.add_parser("structure", parents=[common, run, gpu],
                       help="is the dense rows' structure cap real or measured?")
    p.add_argument("--per-row", type=int, default=STRUCT_PER_ROW)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    {"analyse": analyse, "noise": noise, "sweep": sweep, "structure": structure}[args.command](args)


if __name__ == "__main__":
    main()
