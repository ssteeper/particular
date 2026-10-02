"""Render a short hype video of five atlas worlds, colored by time as in the viewer, with a soundtrack.

Usage: python make_hype.py [output.mp4]   (default: video/hype.mp4)
Needs ffmpeg on PATH. Rebuilds each world from search/atlas.json with particular.build and simulates
it with numpy. Also writes <output stem>_frames.png, a contact sheet of 12 frames from the finished video.

Each WORLDS row is tuned by eye: the view half-width at the segment's start and end (a slow push in or
pull out), the trail length in logged snapshots and how sharply it fades, the dot size in the viewer's
slider units, the run (first and last simulation step shown) and the logging interval.
"""
import json
import os
import subprocess
import sys
import time
import wave

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import make_video
from make_video import SR, note, text_layer
from particular import build, simulate

W = H = make_video.W    # square, like the viewer canvas; text_layer draws at this size
FPS = make_video.FPS    # render_audio reads the motion at this rate
BPM = 160
BAR = 4 * 60 / BPM      # 1.5 s
INTRO = 2 * BAR         # title card; equals make_video.INTRO, so its intro swell lands on the first world
SEG = 4 * BAR           # screen time of each world, cut on the downbeat
XF = 0.25               # crossfade into the next world, ending on its downbeat
OUTRO = 4 * BAR         # grid of all five worlds, then the title
LINGER = 0.6            # the last world keeps moving while it fades under the outro grid
TILE_GAP = make_video.TILE_GAP   # outro tiles appear on make_video's bell spacing
TILE, TILE_PAD, GRID_TOP = 320, 24, 70
BACKDROP = 2            # WORLDS row whose still sits behind the title card
GLOW = 0.3              # bloom strength
PEAK = 0.84             # soundtrack peak, about -1.5 dBFS
FONTS = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")   # DejaVu, as make_video, on any OS
FONT = os.path.join(FONTS, "DejaVuSans.ttf")
BOLD = os.path.join(FONTS, "DejaVuSans-Bold.ttf")
MONO = os.path.join(FONTS, "DejaVuSansMono.ttf")
LUT = plt.get_cmap("hsv")(np.linspace(0, 1, 128))[:, :3]    # the viewer's color-by-time palette

# cell: atlas cell. half: view half-width at the world's first and last frame (linear in between).
# tail: logged snapshots of trail kept; fade: trail alpha = 0.9 * (1 - age / tail) ** fade.
# dot: viewer dot size (radius = dot * 1.6 px per 1000 px). steps: first and last simulation step shown,
# the first > 0 where the opening would otherwise be bare dots. log: steps per logged snapshot.
WORLDS = [
    dict(cell=[7, 1, 1], half=(11.5, 10.0), tail=375, fade=1.5, dot=1.8, steps=(250, 2400), log=2),
    dict(cell=[5, 1, 5], half=(7.6, 6.9), tail=600, fade=1.5, dot=1.8, steps=(400, 3400), log=2),
    dict(cell=[5, 3, 0], half=(14.5, 13.6), tail=300, fade=2.0, dot=1.2, steps=(50, 3000), log=2),
    dict(cell=[3, 6, 0], half=(26.0, 31.5), tail=150, fade=2.0, dot=1.2, steps=(50, 3000), log=4),
    dict(cell=[5, 7, 2], half=(21.0, 56.0), tail=100, fade=1.5, dot=1.2, steps=(0, 650), log=2),
]


# ---------------------------------------------------------------- rasterizing

def discs(x, y, r, size):
    """Antialiased disc coverage at pixel coords (x, y): (flat pixel index, coverage, point index)."""
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    reach = int(np.ceil(r + 0.5))
    pts = np.arange(len(x))
    idxs, covs, which = [], [], []
    for dx in range(-reach, reach + 1):
        for dy in range(-reach, reach + 1):
            px, py = ix + dx, iy + dy
            cov = np.clip(r + 0.5 - np.hypot(px + 0.5 - x, py + 0.5 - y), 0, 1)
            ok = (cov > 0) & (px >= 0) & (px < size) & (py >= 0) & (py < size)
            idxs.append(py[ok] * size + px[ok])
            covs.append(cov[ok])
            which.append(pts[ok])
    return np.concatenate(idxs), np.concatenate(covs), np.concatenate(which)


def composite(idx, cov, layer, alpha, rgb, size):
    """Paint layers of dots over black, oldest layer first, like canvas fills with globalAlpha.

    idx, cov: disc coverage entries; layer: their layer number (higher is drawn later); alpha, rgb: per layer.
    Dots within one layer form a single fill, so their coverage is merged before blending.
    """
    key, inv = np.unique(idx.astype(np.int64) * (layer.max() + 1) + layer, return_inverse=True)
    cov = np.minimum(np.bincount(inv, cov), 1.0)
    pix, lay = key // (layer.max() + 1), key % (layer.max() + 1)
    a = np.minimum(alpha[lay] * cov, 0.999)
    # light reaching the viewer from a layer is dimmed by every later layer over the same pixel
    order = np.lexsort((-lay, pix))
    pix, lay, a = pix[order], lay[order], a[order]
    logt = np.log1p(-a)
    run = np.cumsum(logt) - logt                             # sum over earlier entries in this order
    first = np.r_[True, pix[1:] != pix[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(pix)), 0))
    w = a * np.exp(run - run[start])
    img = np.empty((size * size, 3), np.float32)
    for ch in range(3):
        img[:, ch] = np.bincount(pix, w * rgb[lay, ch], minlength=size * size)
    return img.reshape(size, size, 3)


def bloom(img, glow):
    small = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    small = small.resize((img.shape[1] // 4, img.shape[0] // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(3))
    blur = np.asarray(small.resize((img.shape[1], img.shape[0]), Image.BILINEAR), np.float32) / 255.0
    return np.clip(img + glow * blur, 0, 1)


class World:
    """One atlas world: its trajectory, and frames of it drawn colored by time with a fading tail."""

    def __init__(self, entry, cfg):
        self.cfg = cfg
        self.name = "world " + "-".join(map(str, entry["cell"]))
        self.fitness = entry["fitness"]
        s = build(entry["genome"])
        self.desc = s["desc"]
        first, last = cfg["steps"]
        self.log = cfg["log"]
        data = simulate(s["pos"], s["vel"], last + 1, log_every=self.log, g=s["g"], c=s["c"],
                        species=s["species"], rules=s["rules"])
        self.data = data                       # snapshot j holds the state after step j * log + 1
        self.k0 = first // self.log            # first snapshot shown
        self.n = len(data)

    def snapshot(self, u):
        """Fractional snapshot index at progress u (0..1) through the segment."""
        return self.k0 + u * (self.n - 1 - self.k0)

    def half(self, u):
        h0, h1 = self.cfg["half"]
        return h0 + (h1 - h0) * u

    def frame(self, u, zoom=1.0, size=W):
        """Trails up to progress u: hue follows step / last step, alpha fades with age over the tail.

        zoom > 1 magnifies the view briefly for the beat punch."""
        kf = self.snapshot(u)
        k = int(kf)
        tail, fade = self.cfg["tail"], self.cfg["fade"]
        j0 = max(0, k - tail + 1)
        snaps = np.arange(j0, k + 1)
        age = kf - snaps
        alpha = 0.9 * np.clip(1 - age / tail, 0, 1) ** fade
        hue = snaps / (self.n - 1)
        rgb = LUT[np.round(hue * (len(LUT) - 1)).astype(int)]
        return self._draw(self.data[j0:k + 1], self.half(u) / zoom, alpha, rgb, size, True)

    def still(self, size):
        """The whole run as the viewer's still mode shows it (alpha falls from 1 to 0), at the opening view."""
        t = np.arange(self.n) / (self.n - 1)
        rgb = LUT[np.round(t * (len(LUT) - 1)).astype(int)]
        return self._draw(self.data, self.cfg["half"][0], 1 - t, rgb, size, False)

    def _draw(self, pts, half, alpha, rgb, size, heads):
        scale = size / 2 / half
        r = max(0.6, self.cfg["dot"] * size / 1000 * 1.6)
        n = pts.shape[1]
        x = size / 2 + scale * pts[..., 0].ravel()
        y = size / 2 - scale * pts[..., 1].ravel()
        idx, cov, which = discs(x, y, r, size)
        layer = which // n
        if heads:   # the particles' current positions on top, white as in the viewer
            hi, hc, _ = discs(x[-n:], y[-n:], max(1.0, r), size)
            idx, cov, layer = np.r_[idx, hi], np.r_[cov, hc], np.r_[layer, np.full(len(hi), len(pts))]
            alpha, rgb = np.r_[alpha, 0.9], np.r_[rgb, [[1.0, 1.0, 1.0]]]
        if len(idx) == 0:
            return np.zeros((size, size, 3), np.float32)
        return composite(idx, cov, layer, np.asarray(alpha, float), np.asarray(rgb, float), size)

    def at(self, t):
        """Progress u (0..1) through the world's screen time at video time t; set by timeline()."""
        return float(np.clip((t - self.t0) / (self.t1 - self.t0), 0, 1))

    def _per_frame(self, values):
        """Resample a per-snapshot series at each video frame from the world's downbeat on."""
        u = np.clip((self.start + np.arange(int(SEG * FPS) + 1) / FPS - self.t0) / (self.t1 - self.t0), 0, 1)
        return np.interp(self.snapshot(u), np.arange(self.n), values)

    def mean_radius(self):
        """Mean distance from the centre per video frame, read by make_video.render_audio."""
        d = self.data
        return self._per_frame(np.linalg.norm(d - d.mean(1, keepdims=True), axis=-1).mean(1))

    def speed(self):
        """Mean particle speed per video frame, scaled to 0..1 over the segment; drives the hi-hats."""
        v = np.linalg.norm(np.diff(self.data, axis=0), axis=-1).mean(1)
        v = self._per_frame(np.r_[v[0], v])
        return v / v.max()


# ------------------------------------------------------------------- timeline

def timeline():
    """Load the worlds and set each one's downbeat (start) and screen time t0..t1. Returns (worlds, outro)."""
    with open(os.path.join("search", "atlas.json"), encoding="utf-8") as f:
        elites = {tuple(e["cell"]): e for e in json.load(f)["elites"]}
    worlds = []
    for i, cfg in enumerate(WORLDS):
        w = World(elites[tuple(cfg["cell"])], cfg)
        w.start = INTRO + i * SEG
        w.t0 = w.start - XF
        w.t1 = w.start + SEG + (LINGER if i == len(WORLDS) - 1 else 0)
        worlds.append(w)
    return worlds, INTRO + len(WORLDS) * SEG


def hits(worlds, outro):
    """Drum and bass events (time, kind, world index, velocity). The beat gains a layer with each world."""
    sixteenth = BAR / 16
    ev = []
    for i, w in enumerate(worlds):
        speed = w.speed()
        if i:
            ev.append((w.start, "crash", i, 0.7))
        for bar in range(4):
            b0 = w.start + bar * BAR
            for s in [[0, 8], [0, 8], [0, 8], [0, 6, 8], [0, 6, 8, 11]][min(i, 4)]:
                ev.append((b0 + s * sixteenth, "kick", i, 1.0))
            if i >= 2:
                ev += [(b0 + s * sixteenth, "clap", i, 1.0) for s in (4, 12)]
            if i >= 1:   # hi-hats get louder as the particles speed up
                for s in range(0, 16, 1 if i >= 4 else 2):
                    t = b0 + s * sixteenth
                    v = 0.45 + 0.55 * speed[min(int((t - w.start) * FPS), len(speed) - 1)]
                    offbeat = s % 4 == 2
                    ev.append((t, "open" if i >= 2 and offbeat else "hat", i, v * (1.0 if offbeat else 0.6)))
            if i >= 3:
                ev += [(b0 + s * sixteenth, "bass", i, 1.0) for s in (0, 3, 6, 10, 12)]
            if i == len(worlds) - 1 and bar == 3:   # snare roll into the outro
                ev += [(b0 + s * sixteenth, "snare", i, 0.35 + 0.65 * (s - 8) / 7) for s in range(8, 16)]
    ev += [(outro, "kick", len(worlds), 1.2), (outro, "crash", len(worlds), 1.0)]
    return ev


# ---------------------------------------------------------------------- video

def over(img, mask, rgb, a):
    """Paint a text mask (H, W, 1) over img with color rgb (3,) or (H, W, 3) and opacity a."""
    m = mask * a
    return img * (1 - m) + m * rgb


def rainbow(mask, lo=0.25):
    """The viewer's hue palette across a text mask's ink, left to right, lifted toward white by lo."""
    cols = np.flatnonzero(mask.max(axis=(0, 2)) > 0)
    x = np.clip((np.arange(W) - cols[0]) / max(1, cols[-1] - cols[0]), 0, 1)
    return (lo + (1 - lo) * LUT[np.round(x * (len(LUT) - 1)).astype(int)])[None].astype(np.float32)


def wordmark(y, size, spacing):
    return text_layer([("particular", BOLD, size, y, 1.0, spacing)])


def subtitle(y, size):
    return text_layer([("an atlas of particle worlds", FONT, size, y, 1.0, 3)])


def caption(i, w):
    return text_layer([(f"{i + 1:02d}/{len(WORLDS):02d}  {w.name}", BOLD, 30, -(H - 118), 1.0, 1),
                       (f"{w.desc}  \u00b7  fitness {w.fitness:.2f}", MONO, 20, -(H - 70), 0.65, 0)])


def to_u8(img):
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def tile_layout(count):
    """Top-left corners of the outro grid: rows of three, a short last row centred."""
    cols = 3
    rows = -(-count // cols)
    out = []
    for i in range(count):
        r, c = divmod(i, cols)
        in_row = min(cols, count - r * cols)
        out.append(((W - in_row * TILE - (in_row - 1) * TILE_PAD) // 2 + c * (TILE + TILE_PAD),
                    GRID_TOP + r * (TILE + TILE_PAD)))
    return out, GRID_TOP + rows * TILE + (rows - 1) * TILE_PAD


class Video:
    """Composes the frame at any time t: title card, the worlds cut on the beat, captions, outro grid."""

    def __init__(self, worlds, outro, total):
        self.worlds, self.outro, self.total = worlds, outro, total
        self.kicks = [(t, i) for t, kind, i, _ in hits(worlds, outro) if kind == "kick"]
        self.cuts = [w.start for w in worlds] + [outro]
        self.caps = [caption(i, w) for i, w in enumerate(worlds)]
        # outro tiles: each world's whole run as the viewer's still mode draws it, like an atlas thumbnail,
        # with softened borders so worlds that fill their square don't end in a hard edge
        ramp = np.clip((0.5 - np.abs(np.arange(TILE) + 0.5 - TILE / 2) / TILE) / 0.07, 0, 1)
        edge = (ramp[:, None] * ramp[None, :])[..., None]
        self.tiles = [edge * np.asarray(Image.fromarray(to_u8(w.still(2 * TILE))).resize((TILE, TILE), Image.LANCZOS),
                                        np.float32) / 255.0 for w in worlds]
        self.spots, grid_bottom = tile_layout(len(self.tiles))
        self.backdrop = Image.fromarray(to_u8(worlds[BACKDROP].still(W)))
        self.intro_sub = subtitle(H // 2 + 60, 36)
        self.outro_big = wordmark(grid_bottom + 44, 84, 12)
        self.outro_hue, self.outro_sub = rainbow(self.outro_big), subtitle(grid_bottom + 160, 30)

    def frame(self, t):
        worlds, outro = self.worlds, self.outro
        img = np.zeros((H, W, 3), np.float32)
        ot = t - outro
        fade_out = np.clip((self.total - t) / 1.2, 0, 1)

        # intro: a dim still of one world slowly pushes in behind the title
        if t < INTRO:
            m = (W - W / (1 + 0.15 * t / INTRO)) / 2
            bg = self.backdrop.resize((W, H), Image.BILINEAR, box=(m, m, W - m, H - m))
            img += np.asarray(bg, np.float32) / 255.0 * 0.25 * np.clip(t / 0.8, 0, 1) * np.clip((INTRO - t) / 0.5, 0, 1)

        for i, w in enumerate(worlds):
            if not w.t0 <= t < w.t1:
                continue
            out_len = LINGER if i == len(worlds) - 1 else XF
            env = np.clip((t - w.t0) / XF, 0, 1) * np.clip((w.t1 - t) / out_len, 0, 1)
            punch = np.exp(-(t - w.start) / 0.2) if t >= w.start else 0.0
            img += env * w.frame(w.at(t), zoom=1 + 0.04 * punch)

        # flash on each cut, and a pulse on every kick that grows as the beat builds
        flash = max([np.exp(-(t - s) / 0.12) for s in self.cuts if s <= t] + [0.0])
        pulse = max([0.04 * i * np.exp(-(t - k) / 0.1) for k, i in self.kicks if k <= t < k + 0.5] + [0.0])
        img *= 1 + 0.35 * flash + pulse

        # outro: the five worlds appear one by one on the bells
        if t >= outro:
            for i, ((x0, y0), tile) in enumerate(zip(self.spots, self.tiles)):
                img[y0:y0 + TILE, x0:x0 + TILE] += tile * np.clip((ot - TILE_GAP * i) / 0.5, 0, 1) * fade_out

        img = bloom(img, GLOW)

        if t < INTRO:
            spread = 1 - np.clip(t / 1.2, 0, 1)
            big = wordmark(H // 2 - 110, 120, 16 + 44 * spread ** 3)   # letters close up as it fades in
            out_a = np.clip((INTRO - 0.05 - t) / 0.35, 0, 1)
            img = over(img, big, rainbow(big), np.clip(t / 0.5, 0, 1) * out_a)
            img = over(img, self.intro_sub, 0.85, np.clip((t - 0.8) / 0.5, 0, 1) * out_a)
        for i, w in enumerate(worlds):
            a = np.clip(min((t - w.start - 0.15) / 0.3, (w.start + SEG - XF - 0.2 - t) / 0.3), 0, 1)
            if a > 0:
                img = over(img, self.caps[i], 1.0, a)
        if t >= outro:
            img = over(img, self.outro_big, self.outro_hue, np.clip((ot - 1.4) / 0.6, 0, 1) * fade_out)
            img = over(img, self.outro_sub, 0.8, np.clip((ot - 1.9) / 0.6, 0, 1) * fade_out)
        return to_u8(img)


def render_video(video, path):
    nframes = int(round(video.total * FPS))
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
         "-crf", "20", "-pix_fmt", "yuv420p", path],
        stdin=subprocess.PIPE)
    for f in range(nframes):
        ff.stdin.write(video.frame(f / FPS).tobytes())
        if f % 60 == 0:
            print(f"frame {f}/{nframes}", flush=True)
    ff.stdin.close()
    ff.wait()


# ---------------------------------------------------------------------- audio

def band(x, lo, hi):
    """Keep frequencies lo..hi Hz of x."""
    f = np.fft.rfftfreq(len(x), 1 / SR)
    spec = np.fft.rfft(x)
    spec[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(spec, len(x))


def drum(kind, root, rng):
    """One synthesized hit, stereo (n, 2), peak 1."""
    dur = {"kick": 0.5, "clap": 0.35, "snare": 0.2, "hat": 0.08, "open": 0.35, "crash": 2.0, "bass": 0.4}[kind]
    t = np.arange(int(dur * SR)) / SR
    noise = lambda: rng.standard_normal(len(t))
    if kind == "kick":
        f = 42 + 120 * np.exp(-t / 0.035)
        s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.3) + 0.25 * band(noise(), 1000, 8000) * np.exp(-t / 0.004)
    elif kind in ("clap", "snare"):
        decay = 0.12 if kind == "clap" else 0.07
        s = band(noise(), 900, 6000) * np.exp(-t / decay) + 0.4 * np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.05)
    elif kind in ("hat", "open"):
        s = band(noise(), 7000, 16000) * np.exp(-t / (0.025 if kind == "hat" else 0.16))
    elif kind == "crash":
        env = np.exp(-t / 0.8)
        s = np.c_[band(noise(), 3500, 16000) * env, band(noise(), 3500, 16000) * env]
    else:   # bass, an octave under the world's chord root
        f = note(root - 12)
        s = np.tanh(1.5 * (np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t)))
        s *= np.minimum(t / 0.005, 1) * np.exp(-t / 0.22)
    s = s if s.ndim == 2 else np.c_[s, s]
    return s / np.abs(s).max()


GAIN = {"kick": 0.9, "clap": 0.35, "snare": 0.3, "hat": 0.11, "open": 0.09, "crash": 0.16, "bass": 0.42}
PAN = {"hat": (0.75, 1.0), "open": (1.0, 0.75)}


def render_audio(worlds, outro, total, path):
    """make_video's soundtrack (pads, a voice that follows each world's radius, bells, reverb), plus a beat."""
    tl = [(w, w.start, SEG, w.start + SEG + 0.6) for w in worlds]
    make_video.render_audio(tl, outro + make_video.FADE, total, path)   # its outro bells start at `outro`
    with wave.open(path, "rb") as f:
        pad = np.frombuffer(f.readframes(f.getnframes()), np.int16).reshape(-1, 2) / 32767.0
    n = len(pad)
    t = np.arange(n) / SR
    rng = np.random.default_rng(7)
    roots = [50, 53, 45, 48, 43, 46, 50, 41, 45, 48, 43, 46, 41, 50]   # make_video's chord roots
    beat = np.zeros((n, 2))
    duck = np.zeros(n)
    tonal = {}
    for when, kind, i, v in hits(worlds, outro):
        root = roots[i % len(roots)]
        if kind in ("kick", "bass"):   # tonal hits are reused; noise hits are fresh every time
            s = tonal.get((kind, root))
            if s is None:
                s = tonal[(kind, root)] = drum(kind, root, rng)
        else:
            s = drum(kind, root, rng)
        a = int(when * SR)
        m = min(len(s), n - a)
        pl, pr = PAN.get(kind, (1.0, 1.0))
        beat[a:a + m] += GAIN[kind] * v * s[:m] * [pl, pr]
        if kind == "kick":   # sidechain: the pads dip under every kick
            d = np.exp(-np.arange(m) / (0.12 * SR))
            duck[a:a + m] = np.maximum(duck[a:a + m], d)
    # riser over the intro's last bar, cut off by the first downbeat
    rise = (t >= INTRO - BAR) & (t < INTRO)
    r = ((t[rise] - (INTRO - BAR)) / BAR) ** 2
    sweep = 2 * np.pi * np.cumsum(note(62) * 2 ** (2 * r)) / SR
    beat[rise] += (0.12 * band(rng.standard_normal(rise.sum()), 1500, 9000) * r + 0.05 * r * np.sin(sweep))[:, None]
    mix = pad * (1 - 0.45 * duck)[:, None] + beat
    peak = np.abs(mix).max()
    mix = np.tanh(1.3 * mix / peak) / np.tanh(1.3) * PEAK            # gentle limiter; peak lands at PEAK
    mix *= np.clip((total - t) / 1.0, 0, 1)[:, None]
    with wave.open(path, "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes((mix * 32767).astype(np.int16).tobytes())


# ---------------------------------------------------------------------- output

def contact_sheet(video, total, path, count=12, cols=4, size=360):
    """Frames sampled evenly across the finished video, labelled with their times."""
    rows = -(-count // cols)
    sheet = Image.new("RGB", (cols * size, rows * size))
    font = ImageFont.truetype(MONO, 16)
    for k in range(count):
        at = (k + 0.5) * total / count
        raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{at:.3f}", "-i", video, "-frames:v", "1",
                              "-s", f"{size}x{size}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             capture_output=True, check=True).stdout
        im = Image.frombytes("RGB", (size, size), raw)
        ImageDraw.Draw(im).text((8, 6), f"{at:5.1f}s", fill=(200, 200, 200), font=font)
        r, c = divmod(k, cols)
        sheet.paste(im, (c * size, r * size))
    sheet.save(path)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join("video", "hype.mp4")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    began = time.time()
    worlds, outro = timeline()
    total = outro + OUTRO
    for w in worlds:
        print(f"{w.name}: {w.desc}, fitness {w.fitness:.2f}")
    print(f"{len(worlds)} worlds, {total:.1f} s")
    render_audio(worlds, outro, total, out + ".wav")
    render_video(Video(worlds, outro, total), out + ".silent.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out + ".silent.mp4", "-i", out + ".wav",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out],
                   check=True)
    os.remove(out + ".silent.mp4")
    os.remove(out + ".wav")
    sheet = os.path.splitext(out)[0] + "_frames.png"
    contact_sheet(out, total, sheet)
    print(f"wrote {out} and {sheet} in {time.time() - began:.0f} s")
