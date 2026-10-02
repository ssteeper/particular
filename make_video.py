"""Render a short video that draws each scene's trails, with a generated soundtrack.

Usage: python make_video.py [output.mp4]
Needs ffmpeg on PATH. Reuses trajectories cached by make_gallery.py.
"""
import subprocess
import sys
import wave

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from make_gallery import species_colors, trajectory
from particular import SCENES

W = H = 1080
FPS = 30
SR = 44100
ORDER = ["spiral", "counter", "nested", "bloom", "star", "wave", "binary", "collision", "triad",
         "breath", "square", "quartet", "galaxy", "constellation"]
INTRO = 3.0            # seconds before the first scene starts drawing
HOLD = 1.2             # seconds a finished scene stays on screen
FADE = 0.8             # crossfade between scenes
OUTRO = 7.0            # grid of all scenes plus title
TILE_GAP = 0.18        # seconds between grid tiles appearing
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


# ---------------------------------------------------------------- rasterizing

def splat(acc, x, y, rgb, sigma):
    """Add soft dots at pixel coords (x, y) with per-dot colour rgb (n, 3) into acc."""
    h, w, _ = acc.shape
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    reach = int(np.ceil(2 * sigma))
    idxs, wgts, cols = [], [], []
    for dx in range(-reach, reach + 1):
        for dy in range(-reach, reach + 1):
            px, py = ix + dx, iy + dy
            wgt = np.exp(-((px + 0.5 - x) ** 2 + (py + 0.5 - y) ** 2) / (2 * sigma ** 2))
            ok = (px >= 0) & (px < w) & (py >= 0) & (py < h) & (wgt > 1e-3)
            idxs.append(py[ok] * w + px[ok])
            wgts.append(wgt[ok])
            cols.append(rgb[ok])
    idx, wgt, col = np.concatenate(idxs), np.concatenate(wgts), np.concatenate(cols)
    flat = acc.reshape(-1, 3)
    for ch in range(3):
        flat[:, ch] += np.bincount(idx, weights=col[:, ch] * wgt, minlength=h * w)
    return acc


def tonemap(acc, glow=0.6):
    img = 1 - np.exp(-1.4 * acc)
    small = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((W // 4, H // 4), Image.BILINEAR)
    blur = np.asarray(small.filter(ImageFilter.GaussianBlur(4)).resize((W, H), Image.BILINEAR)) / 255.0
    return np.clip(img + glow * blur, 0, 1)


class Scene:
    """Draws one scene's trails up to a given snapshot with a slowly moving camera."""

    def __init__(self, name):
        self.name = name
        self.data = trajectory(name)
        s = SCENES[name]()
        self.half = s["half"]
        self.n = len(self.data)
        t = np.linspace(0, 1, self.n)
        self.colors = plt.get_cmap(s["cmap"])(t)[:, :3]
        self.rgb = species_colors(s)           # per-particle colors for ecosystem scenes
        self.alpha = 1 - t
        self.spin = np.radians(np.random.default_rng(len(name)).choice([-1, 1]) * 12)

    def frame(self, k, u):
        """k: snapshots drawn so far (float). u: 0..1 progress through the scene's screen time."""
        k = int(np.clip(k, 1, self.n))
        pts = self.data[:k]
        ang = self.spin * (u - 0.5)
        zoom = (W / 2) / self.half * (0.94 + 0.10 * u)
        ca, sa = np.cos(ang), np.sin(ang)
        xy = pts.reshape(-1, 2)
        x = W / 2 + zoom * (ca * xy[:, 0] - sa * xy[:, 1])
        y = H / 2 - zoom * (sa * xy[:, 0] + ca * xy[:, 1])
        per = pts.shape[1]
        if self.rgb is None:
            rgb = np.repeat(self.colors[:k] * self.alpha[:k, None], per, axis=0) * 1.4
        else:
            rgb = (self.alpha[:k, None, None] * self.rgb[None]).reshape(-1, 3) * 1.4
        acc = np.zeros((H, W, 3), np.float32)
        splat(acc, x, y, rgb, 0.9)
        if k < self.n:  # bright heads at the particles' current positions
            head = slice(-per, None)
            splat(acc, x[head], y[head], np.full((per, 3), 1.6), 1.6)
        return acc

    def mean_radius(self):
        c = self.data.mean(1, keepdims=True)
        return np.linalg.norm(self.data - c, axis=-1).mean(1)


def text_layer(lines):
    """lines: list of (text, font_path, size, y, alpha, spacing). Returns float RGB array."""
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    for text, path, size, y, a, spacing in lines:
        if a <= 0:
            continue
        font = ImageFont.truetype(path, size)
        widths = [d.textlength(ch, font=font) for ch in text]
        total = sum(widths) + spacing * (len(text) - 1)
        x = (W - total) / 2 if y >= 0 else 40
        y = abs(y)
        for ch, cw in zip(text, widths):
            d.text((x, y), ch, fill=int(255 * a), font=font)
            x += cw + spacing
    return np.asarray(img, np.float32)[..., None] / 255.0


def caption(i, name, a):
    return text_layer([(f"{i + 1:02d}  {name}", MONO, 22, -(H - 64), 0.55 * a, 2)])


# ------------------------------------------------------------------- timeline

def build_timeline():
    """Return list of (scene, start_time, draw_seconds, end_time)."""
    t = INTRO
    tl = []
    for name in ORDER:
        s = Scene(name)
        draw = s.n / FPS
        end = t + draw + HOLD + FADE
        tl.append((s, t, draw, end))
        t = end - FADE
    return tl, t + FADE


def envelope(t, start, end, fade=FADE):
    return float(np.clip(min((t - start) / fade + 1, (end - t) / fade), 0, 1))


def final_still(scene, size):
    acc = scene.frame(scene.n, 0.5)
    img = tonemap(acc, glow=0.4)
    return np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize((size, size), Image.LANCZOS),
                      np.float32) / 255.0


def render_video(tl, scenes_end, out_silent):
    total = scenes_end + OUTRO
    nframes = int(total * FPS)
    cols = int(np.ceil(np.sqrt(len(tl))))
    rows = int(np.ceil(len(tl) / cols))
    tile = W // cols
    tiles = [final_still(s, tile) for s, *_ in tl]
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
         "-crf", "20", "-pix_fmt", "yuv420p", out_silent],
        stdin=subprocess.PIPE)
    first = tl[0][0]
    for f in range(nframes):
        t = f / FPS
        img = np.zeros((H, W, 3), np.float32)

        # intro: the first scene's starting ring appears under the title
        if t < INTRO:
            a = min(t / 1.2, 1.0)
            acc = first.frame(1, 0.0)
            img += tonemap(acc) * a
            ta = np.clip(min((t - 0.4) / 0.8, (INTRO - 0.2 - t) / 0.6), 0, 1)
            img += text_layer([("particular", FONT, 64, H // 2 - 40, ta, 14)])

        for i, (s, start, draw, end) in enumerate(tl):
            if start - (FADE if i else 0) <= t < end:
                env = envelope(t, start if i else start - FADE, end) if i else min(1.0, (end - t) / FADE)
                k = (t - start) * FPS + 1
                u = (t - start) / (end - start)
                img += tonemap(s.frame(k, u)) * env
                # captions fade out before the crossfade so two never overlap
                cap = np.clip(min((t - start) / 0.4, (end - FADE - t) / 0.4), 0, 1)
                img += caption(i, s.name, cap)

        # outro: tiles appear one by one, then the title
        if t >= scenes_end - FADE:
            ot = t - (scenes_end - FADE)
            grid = np.zeros_like(img)
            for i, tile_img in enumerate(tiles):
                a = np.clip((ot - TILE_GAP * i) / 0.6, 0, 1)
                r, c = divmod(i, cols)
                in_row = min(cols, len(tiles) - r * cols)         # centre a short last row
                x0 = (W - in_row * tile) // 2 + c * tile
                y0 = (H - rows * tile) // 2 + r * tile
                grid[y0:y0 + tile, x0:x0 + tile] += tile_img * a
            fade_out = np.clip((total - t) / 1.5, 0, 1)
            dim = 1 - 0.55 * np.clip((ot - 3.4) / 0.8, 0, 1)
            img += grid * dim * fade_out
            ta = np.clip((ot - 3.6) / 0.8, 0, 1) * fade_out
            img += text_layer([("particular", FONT, 64, H // 2 - 40, ta, 14)])

        ff.stdin.write((np.clip(img, 0, 1) * 255).astype(np.uint8).tobytes())
        if f % 60 == 0:
            print(f"frame {f}/{nframes}", flush=True)
    ff.stdin.close()
    ff.wait()
    return total


# ---------------------------------------------------------------------- audio

def note(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def render_audio(tl, scenes_end, total, path):
    n = int(total * SR)
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    roots = [50, 53, 45, 48, 43, 46, 50, 41, 45, 48, 43, 46, 41, 50]   # D F A C G Bb D F A C G Bb F D
    rng = np.random.default_rng(1)

    def env_at(start, end, fade):
        return np.clip(np.minimum((t - start) / fade + 1, (end - t) / fade), 0, 1)

    for i, (s, start, draw, end) in enumerate(tl):
        root = roots[i % len(roots)]
        env = env_at(start, end, 1.2) ** 1.5
        live = env > 0
        tt = t[live]
        # pad: root, fifth, octave, tenth, each slightly detuned left and right
        for semis, amp in [(0, 0.22), (7, 0.14), (12, 0.12), (15 if i % 2 else 16, 0.07)]:
            f = note(root + semis)
            for ch, det in ((0, 0.997), (1, 1.003)):
                out[live, ch] += amp * env[live] * np.sin(2 * np.pi * f * det * tt + rng.random() * 6)
        # voice that rises as the particles pull together
        r = s.mean_radius()
        r_t = np.interp((tt - start) * FPS, np.arange(len(r)), r)
        f = note(root + 24) * np.clip(r[0] / np.maximum(r_t, 1e-3), 0.5, 4) ** 0.5
        phase = 2 * np.cumsum(np.pi * f / SR)
        drawing = np.clip((start + draw + 0.6 - tt) / 0.6, 0, 1) * np.clip((tt - start) / 0.3, 0, 1)
        v = 0.06 * drawing * env[live] * np.sin(phase)
        out[live, 0] += v * 0.8
        out[live, 1] += v * 1.2
        # bell when the scene begins
        b = t - start
        bell = (b >= 0) & (b < 4)
        for mult, amp in [(4, 0.12), (8.02, 0.05), (11.9, 0.025)]:
            out[bell, i % 2] += amp * np.exp(-b[bell] * 1.6) * np.sin(2 * np.pi * note(root) * mult * b[bell])
            out[bell, 1 - i % 2] += 0.6 * amp * np.exp(-b[bell] * 1.6) * np.sin(2 * np.pi * note(root) * mult * b[bell])

    # outro: one bell per tile, over a D major chord
    o0 = scenes_end - FADE
    for i in range(len(tl)):
        b = t - (o0 + TILE_GAP * i)
        bell = (b >= 0) & (b < 3)
        f = note([62, 64, 66, 69, 71, 74, 76, 78, 81, 83, 86, 88, 90, 93][i % 14])
        out[bell, i % 2] += 0.07 * np.exp(-b[bell] * 2.2) * np.sin(2 * np.pi * f * b[bell])
        out[bell, 1 - i % 2] += 0.04 * np.exp(-b[bell] * 2.2) * np.sin(2 * np.pi * f * b[bell])
    env = np.clip(np.minimum((t - o0) / 1.5, (total - t) / 2.0), 0, 1)
    for semis, amp in [(0, 0.2), (7, 0.12), (12, 0.12), (16, 0.08), (19, 0.05)]:
        for ch, det in ((0, 0.998), (1, 1.002)):
            out[:, ch] += amp * env * np.sin(2 * np.pi * note(38 + semis) * det * t)

    # intro: the first chord swells in
    env = np.clip(t / INTRO, 0, 1) * (t < INTRO + 1.2) * np.clip((INTRO + 1.2 - t) / 1.2, 0, 1)
    for semis, amp in [(0, 0.15), (7, 0.1), (12, 0.08)]:
        out[:, 0] += amp * env * np.sin(2 * np.pi * note(roots[0] + semis) * 0.998 * t)
        out[:, 1] += amp * env * np.sin(2 * np.pi * note(roots[0] + semis) * 1.002 * t)

    # reverb: convolve with decaying noise
    ir_t = np.arange(int(2.8 * SR)) / SR
    wet = np.zeros_like(out)
    for ch in range(2):
        ir = rng.standard_normal(len(ir_t)) * np.exp(-ir_t * 2.2)
        ir /= np.sqrt((ir ** 2).sum())
        m = n + len(ir)
        size = 1 << (m - 1).bit_length()
        wet[:, ch] = np.fft.irfft(np.fft.rfft(out[:, ch], size) * np.fft.rfft(ir, size), size)[:n]
    mix = 0.65 * out + 0.55 * wet
    mix *= np.clip((total - t) / 1.0, 0, 1)[:, None]
    mix = np.tanh(mix / np.abs(mix).max() * 1.2) * 0.85
    pcm = (mix * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "particular.mp4"
    tl, scenes_end = build_timeline()
    total = scenes_end + OUTRO
    print(f"{len(tl)} scenes, {total:.1f} s")
    render_audio(tl, scenes_end, total, out + ".wav")
    render_video(tl, scenes_end, out + ".silent.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out + ".silent.mp4", "-i", out + ".wav",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out],
                   check=True)
    print("wrote", out)
