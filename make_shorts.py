"""Render vertical shorts, one atlas world each, colored by time as in the viewer, with a soundtrack.

Usage: python make_shorts.py [CELL ...] [--top N] [--all] [--out DIR] [--keep-drift] [--workers N]
  CELL is an atlas cell such as 5-1-5. --top N adds the N fittest elites; --all renders every elite.
  With no cells, renders make_hype.py's five worlds. Writes DIR/<cell>.mp4 (1080 x 1920, about 16 s)
  and DIR/<cell>.jpg, a poster of the finished drawing (default DIR: video/shorts).

Needs ffmpeg on PATH. Each world is rebuilt from search/atlas.json and simulated with numpy. The atlas
was simulated on a GPU and the worlds are chaotic, so a CPU run can drift; each run is measured with
atlas.measure and skipped if it lands outside its published cell, unless --keep-drift is given.
The caption shows the CPU run's own fitness. View, trail and dot settings are set from each elite's
particle count, run length and framing instead of by eye as in make_hype.py.

Timeline at make_hype's 160 BPM: eight bars of the world drawing, two bars of its whole run as the
viewer's still mode draws it, and one bar of the wordmark over that still.
"""
import argparse
import json
import os
import subprocess
import time
import wave
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import atlas
from make_hype import (BAR, BOLD, FONT, GAIN, LUT, MONO, PAN, PEAK, World, band, bloom, drum, over,
                       to_u8)
from make_video import FPS, SR, note

W, H = 1080, 1920       # vertical, 9:16
S, TOP = 1080, 500      # the world's square and its top edge; the header sits above it
EDGE = 90               # soft fade at the square's top and bottom edges, in pixels
DRAW = 8 * BAR          # the world draws
HOLD = 2 * BAR          # its whole run, as a still
END = 1 * BAR           # wordmark over the still
TOTAL = DRAW + HOLD + END
XF = 0.5                # crossfade from the drawing to the still
GLOW = 0.3
LOG = 2                 # simulation steps per logged snapshot; measuring takes every 5th (atlas.LOG_EVERY)
FAVOURITES = ["7-1-1", "5-1-5", "5-3-0", "3-6-0", "5-7-2"]
ROOTS = [50, 53, 45, 48, 43, 46, 41]                    # make_video's chord roots, D F A C G Bb F
PROG = [0, -4, -2, 0]   # chord under each pair of bars while drawing: I, bVI, bVII, I


def settings(entry):
    """make_hype WORLDS row for an elite: view from its framing, trail and dots from its size."""
    n, steps, half = entry["n"], entry["genome"]["steps"], entry["half"]
    snaps = steps // LOG
    tail = int(snaps * np.clip(0.4 * (8 / n) ** 0.25, 0.15, 0.4))
    return dict(cell=entry["cell"], half=(1.15 * half, 1.05 * half), tail=tail,
                fade=1.5 if n <= 24 else 2.0, dot=1.8 if n <= 24 else 1.2,
                steps=(tail * LOG // 2, steps), log=LOG)


def measure(world, steps):
    """atlas.measure on the CPU run, sampled as atlas.py logs it (every LOG_EVERY steps, `steps` steps)."""
    stride = atlas.LOG_EVERY // LOG
    data = world.data[::stride][:steps // atlas.LOG_EVERY]
    with np.errstate(all="ignore"):
        return atlas.measure(data) if np.isfinite(data).all() else None


# ------------------------------------------------------------------- overlays

def text_mask(lines):
    """lines: (text, font_path, size, y, alpha, spacing), centred. Returns an (H, W, 1) mask."""
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    for text, path, size, y, a, spacing in lines:
        font = ImageFont.truetype(path, size)
        widths = [d.textlength(ch, font=font) for ch in text]
        x = (W - sum(widths) - spacing * (len(text) - 1)) / 2
        for ch, cw in zip(text, widths):
            d.text((x, y), ch, fill=int(255 * a), font=font)
            x += cw + spacing
    return np.asarray(img, np.float32)[..., None] / 255.0


def rainbow(mask, lo=0.25):
    """The viewer's hue palette across a text mask's ink, left to right, lifted toward white by lo."""
    cols = np.flatnonzero(mask.max(axis=(0, 2)) > 0)
    x = np.clip((np.arange(W) - cols[0]) / max(1, cols[-1] - cols[0]), 0, 1)
    return (lo + (1 - lo) * LUT[np.round(x * (len(LUT) - 1)).astype(int)])[None].astype(np.float32)


def meters(cell, axes, y0):
    """One row per atlas axis: its two end labels around a row of bins, the world's bin filled.

    Returns (rgb, mask), each (H, W, ...)."""
    img = Image.new("RGB", (W, H))
    font = ImageFont.truetype(MONO, 26)
    d = ImageDraw.Draw(img)
    span, gap, box, row = 330, 6, 26, 48
    x0 = (W - span) // 2
    grey, dim, lit = (150, 150, 150), (90, 90, 90), (235, 235, 235)
    for k, (ax, b) in enumerate(zip(axes, cell)):
        lo, hi = ax["label"].split(" \u2192 ")
        y = y0 + k * row
        d.text((x0 - 18 - d.textlength(lo, font=font), y - 3), lo, font=font, fill=grey)
        d.text((x0 + span + 18, y - 3), hi, font=font, fill=grey)
        bw = (span - gap * (ax["bins"] - 1)) / ax["bins"]
        for i in range(ax["bins"]):
            xa = x0 + i * (bw + gap)
            d.rectangle([xa, y, xa + bw, y + box], fill=lit if i == b else None, outline=lit if i == b else dim,
                        width=2)
    rgb = np.asarray(img, np.float32) / 255.0
    mask = rgb.max(-1, keepdims=True)
    return rgb / np.maximum(mask, 1e-6), mask     # colour at full strength; the mask carries the shading


def layer(mask, rgb):
    """An overlay cropped to the rows its mask covers: (first row, mask, rgb)."""
    rows = np.flatnonzero(mask.max(axis=(1, 2)) > 0)
    y0, y1 = rows[0], rows[-1] + 1
    rgb = rgb[y0:y1] if np.ndim(rgb) == 3 and len(rgb) == H else rgb
    return y0, mask[y0:y1], rgb


def paint(img, lay, a, origin=0):
    """Paint an overlay from layer() in place over img, whose first row is frame row `origin`, with opacity a."""
    if a > 0:
        y0, mask, rgb = lay
        y0 -= origin
        img[y0:y0 + len(mask)] = over(img[y0:y0 + len(mask)], mask, rgb, a)


# ------------------------------------------------------------------- the short

class Short:
    def __init__(self, entry, axes):
        self.entry, self.axes = entry, axes
        self.cfg = settings(entry)
        with np.errstate(all="ignore"):
            self.world = World(entry, self.cfg)
        self.m = measure(self.world, entry["genome"]["steps"])
        edges = [ax["edges"] for ax in axes]
        self.cpu_cell = list(atlas.cell_of(self.m["features"], edges)) if self.m else None
        self.name = "-".join(map(str, entry["cell"]))

    @property
    def drifted(self):
        return self.cpu_cell != self.entry["cell"]

    # ---------------------------------------------------------------- timing

    def snapshot_at(self, t):
        return self.world.snapshot(np.clip(t / DRAW, 0, 1))

    def per_frame(self, values):
        """A per-snapshot series resampled at each video frame of the drawing."""
        k = self.snapshot_at(np.arange(int(DRAW * FPS) + 1) / FPS)
        return np.interp(k, np.arange(self.world.n), values)

    def hits(self):
        """Drum events (time, kind, chord shift, velocity). The beat builds over the eight bars."""
        d = self.world.data
        speed = np.linalg.norm(np.diff(d, axis=0), axis=-1).mean(1)
        speed = self.per_frame(np.r_[speed[0], speed])
        speed = speed / max(speed.max(), 1e-12)
        sixteenth = BAR / 16
        ev = [(0.0, "crash", 0, 0.5)]
        for bar in range(8):
            b0, shift = bar * BAR, PROG[bar // 2]
            if bar >= 2:
                ev += [(b0 + s * sixteenth, "kick", shift, 1.0) for s in ([0, 6, 8, 11] if bar >= 6 else [0, 8])]
            if bar >= 4:
                ev += [(b0 + s * sixteenth, "clap", shift, 1.0) for s in (4, 12)]
                ev += [(b0 + s * sixteenth, "bass", shift, 1.0) for s in (0, 3, 6, 10, 12)]
            for s in range(0, 16, 1 if bar >= 6 else 2):   # hi-hats get louder as the particles speed up
                t = b0 + s * sixteenth
                v = 0.45 + 0.55 * speed[min(int(t * FPS), len(speed) - 1)]
                offbeat = s % 4 == 2
                ev.append((t, "open" if bar >= 4 and offbeat else "hat", shift, v * (1.0 if offbeat else 0.6)))
            if bar == 7:   # snare roll into the still
                ev += [(b0 + s * sixteenth, "snare", shift, 0.35 + 0.65 * (s - 8) / 7) for s in range(8, 16)]
        ev += [(DRAW, "kick", 0, 1.2), (DRAW, "crash", 0, 1.0)]
        return ev

    # ---------------------------------------------------------------- video

    def prepare(self):
        e, w = self.entry, self.world
        ramp = np.clip(np.minimum(np.arange(S) + 0.5, S - 0.5 - np.arange(S)) / EDGE, 0, 1)
        self.edge = (ramp ** 1.5)[:, None, None]
        self.still = w.still(S)
        self.kicks = [t for t, kind, _, _ in self.hits() if kind == "kick"]
        f = self.m["fitness"] if self.m else e["fitness"]
        species = e["genome"].get("species", 1)
        mark = text_mask([("particular", BOLD, 40, 70, 1.0, 6)])
        big = text_mask([("particular", BOLD, 110, TOP + S // 2 - 110, 1.0, 14)])
        meter_rgb, meter = meters(e["cell"], self.axes, 340)
        self.mark = layer(mark, rainbow(mark))
        self.head = layer(text_mask([(f"world {self.name}", BOLD, 64, 140, 1.0, 2),
                                     (e["desc"], MONO, 26, 232, 0.8, 0)]), 1.0)
        self.stats = layer(text_mask([(f"{e['n']} particles \u00b7 {species} species \u00b7 fitness {f:.2f}",
                                       MONO, 26, 274, 0.65, 0)]), 1.0)
        self.meter = layer(meter, meter_rgb)
        self.big = layer(big, rainbow(big))
        self.sub = layer(text_mask([("an atlas of particle worlds", FONT, 36, TOP + S // 2 + 40, 1.0, 3)]), 0.85)
        self._header = {}

    def frame(self, t):
        sq = np.zeros((S, S, 3), np.float32)
        if t < DRAW + XF:
            sq += np.clip((DRAW + XF - t) / XF, 0, 1) * self.world.frame(np.clip(t / DRAW, 0, 1), size=S)
        if t > DRAW:
            dim = 1 - 0.65 * np.clip((t - DRAW - HOLD) / 0.5, 0, 1)
            sq += np.clip((t - DRAW) / XF, 0, 1) * dim * self.still
        flash = np.exp(-(t - DRAW) / 0.12) if t >= DRAW else 0.0
        pulse = max([0.12 * np.exp(-(t - k) / 0.1) for k in self.kicks if k <= t < k + 0.5] + [0.0])
        sq = bloom(sq * (1 + 0.35 * flash + pulse), GLOW) * self.edge

        out = np.clip((TOTAL - t) / 0.4, 0, 1)
        paint(sq, self.big, np.clip((t - DRAW - HOLD) / 0.5, 0, 1) * out, TOP)
        paint(sq, self.sub, np.clip((t - DRAW - HOLD - 0.3) / 0.5, 0, 1) * out, TOP)

        img = np.zeros((H, W, 3), np.uint8)    # everything outside the header and the square stays black
        img[:TOP] = self.header(t)
        img[TOP:TOP + S] = to_u8(sq)
        return img

    def header(self, t):
        """The header rows as uint8; cached, since it only changes while fading in and out."""
        head = np.clip((t - 0.15) / 0.4, 0, 1) * np.clip((TOTAL - 0.3 - t) / 0.6, 0, 1)
        alphas = (head, np.clip((t - 0.4) / 0.4, 0, 1) * head, np.clip((t - 0.6) / 0.4, 0, 1) * head)
        key = tuple(round(float(a), 4) for a in alphas)
        if key not in self._header:
            img = np.zeros((TOP, W, 3), np.float32)
            for lay, a in zip((self.mark, self.head, self.stats, self.meter), (alphas[0],) + alphas):
                paint(img, lay, a)
            self._header[key] = to_u8(img)
        return self._header[key]

    def render_video(self, path):
        self.prepare()
        ff = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
             "-crf", "20", "-pix_fmt", "yuv420p", path],
            stdin=subprocess.PIPE)
        for f in range(int(round(TOTAL * FPS))):
            ff.stdin.write(self.frame(f / FPS).tobytes())
        ff.stdin.close()
        ff.wait()

    def poster(self, path):
        """The frame just before the wordmark: the whole run as a still under the header."""
        Image.fromarray(self.frame(DRAW + HOLD - 0.05)).save(path, quality=92)

    # ---------------------------------------------------------------- audio

    def render_audio(self, path):
        """Pads on a chord progression, a voice that rises as the particles pull together, bells, a beat."""
        cell = self.entry["cell"]
        root = ROOTS[(cell[0] * 64 + cell[1] * 8 + cell[2]) % len(ROOTS)]
        third = 15 if cell[0] >= 4 else 16          # minor for the chaotic half of the map
        n = int(TOTAL * SR)
        t = np.arange(n) / SR
        rng = np.random.default_rng(sum(cell))
        pad = np.zeros((n, 2))

        def env_at(start, end, fade):
            return np.clip(np.minimum((t - start) / fade + 1, (end - t) / fade), 0, 1)

        chords = [(2 * BAR * k, 2 * BAR * (k + 1), s) for k, s in enumerate(PROG)] + [(DRAW, TOTAL + 1, 0)]
        for start, end, shift in chords:
            env = env_at(start, end, 0.15) * np.clip(t / 0.4, 0, 1)
            live = env > 0
            for semis, amp in [(0, 0.2), (7, 0.13), (12, 0.11), (third, 0.07)]:
                for ch, det in ((0, 0.997), (1, 1.003)):
                    pad[live, ch] += amp * env[live] * np.sin(2 * np.pi * note(root + shift + semis) * det * t[live]
                                                              + rng.random() * 6)
        # voice: rises as the particles pull together (make_video's), over the drawing only
        d = self.world.data
        r = self.per_frame(np.linalg.norm(d - d.mean(1, keepdims=True), axis=-1).mean(1))
        r_t = np.interp(t * FPS, np.arange(len(r)), r)
        f = note(root + 24) * np.clip(r[0] / np.maximum(r_t, 1e-3), 0.5, 4) ** 0.5
        v = 0.05 * env_at(0.3, DRAW, 0.3) * np.sin(2 * np.cumsum(np.pi * f / SR))
        pad += np.c_[0.8 * v, 1.2 * v]
        # bells: one as the world starts and one as the still appears, then an arpeggio under the wordmark
        bells = [(0.0, root, 0.12), (DRAW, root, 0.14)]
        bells += [(DRAW + HOLD + 0.12 * i, root + 12 + s, 0.06) for i, s in enumerate((0, 7, 12, 19))]
        for i, (at, midi, amp) in enumerate(bells):
            b = t - at
            on = (b >= 0) & (b < 4)
            for mult, a in [(1, 1.0), (2.01, 0.4), (2.98, 0.2)]:
                s = amp * a * np.exp(-b[on] * 1.6) * np.sin(2 * np.pi * note(midi + 12) * mult * b[on])
                pad[on, i % 2] += s
                pad[on, 1 - i % 2] += 0.6 * s
        # reverb on the pads: convolve with decaying noise
        ir_t = np.arange(int(2.4 * SR)) / SR
        size = 1 << (n + len(ir_t) - 1).bit_length()
        for ch in range(2):
            ir = rng.standard_normal(len(ir_t)) * np.exp(-ir_t * 2.4)
            ir /= np.sqrt((ir ** 2).sum())
            pad[:, ch] = 0.65 * pad[:, ch] + 0.5 * np.fft.irfft(np.fft.rfft(pad[:, ch], size) * np.fft.rfft(ir, size), size)[:n]
        pad /= np.abs(pad).max()

        beat = np.zeros((n, 2))
        duck = np.zeros(n)
        tonal = {}
        for when, kind, shift, vel in self.hits():
            if kind in ("kick", "bass"):   # tonal hits are reused; noise hits are fresh every time
                s = tonal.get((kind, shift))
                if s is None:
                    s = tonal[(kind, shift)] = drum(kind, root + shift, rng)
            else:
                s = drum(kind, root + shift, rng)
            a = int(when * SR)
            m = min(len(s), n - a)
            pl, pr = PAN.get(kind, (1.0, 1.0))
            beat[a:a + m] += GAIN[kind] * vel * s[:m] * [pl, pr]
            if kind == "kick":   # sidechain: the pads dip under every kick
                duck[a:a + m] = np.maximum(duck[a:a + m], np.exp(-np.arange(m) / (0.12 * SR)))
        # riser over the last bar of the drawing, cut off by the still's downbeat
        rise = (t >= DRAW - BAR) & (t < DRAW)
        rr = ((t[rise] - (DRAW - BAR)) / BAR) ** 2
        sweep = 2 * np.pi * np.cumsum(note(root + 12) * 2 ** (2 * rr)) / SR
        beat[rise] += (0.1 * band(rng.standard_normal(rise.sum()), 1500, 9000) * rr + 0.04 * rr * np.sin(sweep))[:, None]

        mix = 0.55 * pad * (1 - 0.45 * duck)[:, None] + beat
        mix = np.tanh(1.3 * mix / np.abs(mix).max()) / np.tanh(1.3) * PEAK   # gentle limiter, peak at PEAK
        mix *= np.clip((TOTAL - t) / 0.8, 0, 1)[:, None]
        with wave.open(path, "wb") as f:
            f.setnchannels(2)
            f.setsampwidth(2)
            f.setframerate(SR)
            f.writeframes((mix * 32767).astype(np.int16).tobytes())


# ---------------------------------------------------------------------- output

def make(task):
    entry, axes, out_dir, keep_drift = task
    began = time.time()
    short = Short(entry, axes)
    name = short.name
    if short.m is None:
        return f"{name}: skipped, the CPU run broke"
    if short.drifted and not keep_drift:
        return f"{name}: skipped, the CPU run lands in {'-'.join(map(str, short.cpu_cell))}"
    out = os.path.join(out_dir, name + ".mp4")
    short.render_audio(out + ".wav")
    short.render_video(out + ".silent.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out + ".silent.mp4", "-i", out + ".wav",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out],
                   check=True)
    os.remove(out + ".silent.mp4")
    os.remove(out + ".wav")
    short.poster(os.path.join(out_dir, name + ".jpg"))
    drift = f" (CPU run in {'-'.join(map(str, short.cpu_cell))})" if short.drifted else ""
    return (f"{name}: {entry['desc']}, fitness {short.m['fitness']:.2f} on CPU, "
            f"{entry['fitness']:.2f} published{drift}, {time.time() - began:.0f} s")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cells", nargs="*", help="atlas cells such as 5-1-5")
    p.add_argument("--top", type=int, default=0, help="also render the N fittest elites")
    p.add_argument("--all", action="store_true", help="render every elite")
    p.add_argument("--out", default=os.path.join("video", "shorts"))
    p.add_argument("--keep-drift", action="store_true", help="render worlds whose CPU run leaves their cell")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    args = p.parse_args()

    with open(os.path.join("search", "atlas.json"), encoding="utf-8") as f:
        data = json.load(f)
    elites = {"-".join(map(str, e["cell"])): e for e in data["elites"]}
    names = list(args.cells) or ([] if args.top or args.all else FAVOURITES)
    ranked = sorted(elites, key=lambda k: -elites[k]["fitness"])
    names += ranked if args.all else ranked[:args.top]
    names = list(dict.fromkeys(names))
    missing = [c for c in names if c not in elites]
    if missing:
        p.error("no elite in cell " + ", ".join(missing))
    os.makedirs(args.out, exist_ok=True)
    tasks = [(elites[c], data["axes"], args.out, args.keep_drift) for c in names]
    print(f"{len(tasks)} shorts, {TOTAL:.1f} s each, {args.workers} workers")
    with Pool(min(args.workers, len(tasks))) as pool:
        for line in pool.imap_unordered(make, tasks):
            print(line, flush=True)


if __name__ == "__main__":
    main()
