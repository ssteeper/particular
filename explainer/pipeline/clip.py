"""Render a trajectory (frames, n, 2) as a looping mp4 clip and a poster, in make_video.py's style."""
import subprocess
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import common  # noqa: F401  (puts the repo on sys.path)
from make_video import splat

def tonemap(acc, size, glow=0.5):
    img = 1 - np.exp(-1.4 * acc)
    small = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((size // 4, size // 4), Image.BILINEAR)
    blur = np.asarray(small.filter(ImageFilter.GaussianBlur(3)).resize((size, size), Image.BILINEAR)) / 255.0
    return np.clip(img + glow * blur, 0, 1)

def frames(data, half, size=432, rgb=None, cmap='hsv', hold=40, gain=1.5):
    """Yield uint8 frames: trails accumulate one logged frame at a time, heads bright, then hold."""
    T, N, _ = data.shape
    t = np.linspace(0, 1, T)
    alpha = 1 - t
    col = plt.get_cmap(cmap)(t)[:, :3]
    z = (size / 2) / half
    x = size / 2 + z * data[..., 0]
    y = size / 2 - z * data[..., 1]
    acc = np.zeros((size, size, 3), np.float32)
    sig = 0.75 * size / 432
    for k in range(T):
        c = (np.repeat(col[k:k + 1], N, 0) if rgb is None else rgb) * alpha[k] * gain
        splat(acc, x[k], y[k], c, sig)
        out = acc.copy()
        if k < T - 1:
            splat(out, x[k], y[k], np.full((N, 3), 1.6), 1.4 * size / 432)
        yield (tonemap(out, size) * 255).astype(np.uint8)
    last = (tonemap(acc, size) * 255).astype(np.uint8)
    for _ in range(hold):
        yield last

def write(data, half, path, size=432, rgb=None, fps=30, crf=27, poster=None, stride=1):
    data = data[::stride]
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                           '-s', f'{size}x{size}', '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-preset', 'slow',
                           '-crf', str(crf), '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an', path],
                          stdin=subprocess.PIPE)
    last = None
    for f in frames(data, half, size, rgb):
        ff.stdin.write(f.tobytes()); last = f
    ff.stdin.close(); ff.wait()
    if poster:
        Image.fromarray(last).save(poster, quality=82)
    return last

def still(data, half, size=200, rgb=None):
    acc = None
    for f in frames(data, half, size, rgb, hold=0):
        acc = f
    return acc
