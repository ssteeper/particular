"""Build web/index.html, the interactive page, from web/app.html.

Usage: python make_web.py
Bakes in every scene from particular.SCENES and the picks from the last
search.py run (search/results.json), with thumbnails from gallery/ and search/top/.
"""
import base64
import glob
import io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from particular import SCENES, random_setup
from search import pick


def thumb(path, size=112):
    im = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def lut(name, n=128):
    rgb = (plt.get_cmap(name)(np.linspace(0, 1, n))[:, :3] * 255).round().astype(int)
    return "".join("%02x%02x%02x" % tuple(c) for c in rgb)


def pack(pos, vel):
    return {"p": np.round(pos, 3).ravel().tolist(), "v": np.round(vel, 1).ravel().tolist()}


def main():
    scenes = []
    for name, build in SCENES.items():
        pos, vel, steps, half, cmap = build()
        scenes.append({"name": name, "group": "made", "steps": steps, "half": half, "cmap": cmap,
                       "thumb": thumb(f"gallery/{name}.png"), **pack(pos, vel)})

    results = [r for r in json.load(open("search/results.json")) if r["score"] is not None]
    for r in pick(results, 16):
        pos, vel, desc = random_setup(r["seed"])
        image = glob.glob(f"search/top/*-seed{r['seed']}.png")[0]
        scenes.append({"name": f"seed {r['seed']}", "group": "found", "desc": desc, "score": r["score"],
                       "steps": r["steps"], "half": r["half"], "cmap": "hsv", "thumb": thumb(image),
                       **pack(pos, vel)})

    cmaps = {c: lut(c) for c in {s["cmap"] for s in scenes} | {"hsv"}}
    data = json.dumps({"scenes": scenes, "cmaps": cmaps}, separators=(",", ":"))
    page = open("web/app.html").read().replace("/*DATA*/", "const DATA = " + data + ";")
    with open("web/index.html", "w") as f:
        f.write(page)
    print(f"wrote web/index.html ({len(page) // 1024} KB, {len(scenes)} setups)")


if __name__ == "__main__":
    main()
