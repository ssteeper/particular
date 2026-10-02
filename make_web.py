"""Build web/index.html, the interactive page, from web/app.html.

Usage: python make_web.py
Bakes in every scene from particular.SCENES (ecosystem scenes with their species
and rules), the built-in RULES presets, the picks from the last search.py run
(search/results.json) with thumbnails from gallery/ and search/top/, and, if
atlas.py has run, the atlas of possible worlds (search/atlas.json, search/atlas/).
"""
import base64
import glob
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from particular import PALETTE, RULES, SCENES, build, random_setup
from search import pick

ATLAS = "search/atlas.json"


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


def ecosystem(s):
    """Species and rules of a scene for the page; empty for classic scenes."""
    if s["species"] is None:
        return {}
    name = next((k for k, r in RULES.items() if r is s["rules"]), None)
    return {"group": "eco", "species": np.asarray(s["species"]).tolist(), "rules": {"name": name, **s["rules"]}}


def exact(a):
    """Float64 bytes as base64: exact, and shorter than decimals that would round the start.
    The atlas elites are chaotic enough that rounding the start to 6 decimals moves a fifth of them
    to another cell of the map."""
    return base64.b64encode(np.ascontiguousarray(a, "<f8").tobytes()).decode()


def atlas():
    """The atlas for the page: axes and one entry per elite, rebuilt from its genome; None if absent."""
    if not os.path.exists(ATLAS):
        return None
    a = json.load(open(ATLAS))
    elites = []
    for e in a["elites"]:
        g = e["genome"]
        s = build(g)
        elite = {"name": "world " + "-".join(map(str, e["cell"])), "group": "atlas", "desc": s["desc"],
                 "cell": e["cell"], "features": e["features"], "fitness": e["fitness"],
                 "op": e.get("op"), "parent": e.get("parent"), "steps": e["steps"], "half": e["half"],
                 "cmap": "hsv", "thumb": thumb(e["image"], 64), "pull": 2.0 ** g["log2_pull"],
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
    scenes = []
    for name, build in SCENES.items():
        s = build()
        scenes.append({"name": name, "group": "made", "steps": s["steps"], "half": s["half"], "cmap": s["cmap"],
                       "thumb": thumb(f"gallery/{name}.png"), **pack(s["pos"], s["vel"]), **ecosystem(s)})

    results = [r for r in json.load(open("search/results.json")) if r["score"] is not None]
    for r in pick(results, 16):
        pos, vel, desc = random_setup(r["seed"])
        image = glob.glob(f"search/top/*-seed{r['seed']}.png")[0]
        scenes.append({"name": f"seed {r['seed']}", "group": "found", "desc": desc, "score": r["score"],
                       "steps": r["steps"], "half": r["half"], "cmap": "hsv", "thumb": thumb(image),
                       **pack(pos, vel)})

    cmaps = {c: lut(c) for c in {s["cmap"] for s in scenes} | {"hsv"}}
    worlds = atlas()
    data = json.dumps({"scenes": scenes, "cmaps": cmaps, "rules": RULES, "palette": PALETTE, "atlas": worlds},
                      separators=(",", ":"))
    page = open("web/app.html", encoding="utf-8").read().replace("/*DATA*/", "const DATA = " + data + ";")
    with open("web/index.html", "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote web/index.html ({len(page) // 1024} KB, {len(scenes)} setups, "
          f"{len(worlds['elites']) if worlds else 'no'} atlas worlds)")


if __name__ == "__main__":
    main()
