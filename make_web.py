"""Build web/index.html, the interactive page, from web/app.html.

Usage: python make_web.py
Bakes in the atlas of possible worlds from atlas.py (search/atlas.json with its stills in
search/atlas/), the built-in RULES presets and the species PALETTE. Without an atlas it stops
and leaves web/index.html as it was: run python atlas.py first.
"""
import base64
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from particular import PALETTE, RULES, build

ATLAS = "search/atlas.json"


def thumb(path, size):
    im = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def lut(name, n=128):
    rgb = (plt.get_cmap(name)(np.linspace(0, 1, n))[:, :3] * 255).round().astype(int)
    return "".join("%02x%02x%02x" % tuple(c) for c in rgb)


def exact(a):
    """Float64 bytes as base64: exact, and shorter than decimals that would round the start.
    The atlas elites are chaotic enough that rounding the start to 6 decimals moves a fifth of them
    to another cell of the map."""
    return base64.b64encode(np.ascontiguousarray(a, "<f8").tobytes()).decode()


def atlas():
    """The atlas for the page: axes and one entry per elite, rebuilt from its genome."""
    if not os.path.exists(ATLAS):
        sys.exit(f"make_web.py: {ATLAS} not found. Run python atlas.py to build the atlas first; "
                 "web/index.html was left unchanged.")
    a = json.load(open(ATLAS))
    if not a.get("elites"):
        sys.exit(f"make_web.py: {ATLAS} holds no worlds. Rerun python atlas.py; web/index.html was left unchanged.")
    elites = []
    for e in a["elites"]:
        g = e["genome"]
        s = build(g)
        elite = {"name": "world " + "-".join(map(str, e["cell"])), "desc": s["desc"],
                 "cell": e["cell"], "features": e["features"], "fitness": e["fitness"],
                 "op": e.get("op"), "parent": e.get("parent"), "steps": e["steps"], "half": e["half"],
                 "thumb": thumb(e["image"], 64), "pull": 2.0 ** g["log2_pull"],
                 "push": 2.0 ** g["log2_push"], "pb": exact(s["pos"]), "vb": exact(s["vel"])}
        if s["species"] is not None:
            elite["species"] = s["species"].tolist()
            elite["rules"] = {"name": None, "note": f"{g['species']} species, with rules evolved by atlas.py.",
                              **s["rules"]}
        elites.append(elite)
    ev = a["evaluation"]
    return {"axes": [{k: ax[k] for k in ("name", "label", "bins", "edges")} for ax in a["axes"]],
            "evaluations": ev["evaluations_per_run"] * ev["replicates"], "elites": elites}


def main():
    worlds = atlas()
    data = json.dumps({"atlas": worlds, "lut": lut("hsv"), "rules": RULES, "palette": PALETTE},
                      separators=(",", ":"))
    page = open("web/app.html", encoding="utf-8").read().replace("/*DATA*/", "const DATA = " + data + ";")
    with open("web/index.html", "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote web/index.html ({len(page) // 1024} KB, {len(worlds['elites'])} atlas worlds)")


if __name__ == "__main__":
    main()
