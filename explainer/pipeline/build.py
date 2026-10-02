"""Build explainer/index.html from template.html and data/.

Renders the TeX with render_math.js (run `npm install` in this folder once), then inlines the data the
page's scripts read as `const DATA = {...}`.
"""
import json
import os
import subprocess

from common import ATLAS_JSON, DATA, EXPLAINER, HERE, WORK


def load(name):
    return json.load(open(os.path.join(DATA, name)))


def main():
    atlas = json.load(open(ATLAS_JSON))
    summ, spec, mut = load("summary.json"), load("figs.json"), load("mutants.json")
    data = {
        "atlas": load("atlas_meta.json"),
        "axes": [{k: ax[k] for k in ("name", "label", "bins", "edges")} for ax in atlas["axes"]],
        "mutants": {k: {f: v[f] for f in ("genome", "cell", "fitness", "features", "terms")} for k, v in mut.items()},
        "spectra": {k: spec[k] for k in ("r35", "r36", "r282")},
        "metrics": {t: summ[t] for t in ("random", "map_elites")},
        "per_rep": summ["per_rep"], "locality": summ["locality"], "elite_ops": summ["elite_ops"],
    }
    rendered = os.path.join(WORK, "template.rendered.html")
    subprocess.run(["node", os.path.join(HERE, "render_math.js"), os.path.join(EXPLAINER, "template.html"), rendered],
                   check=True, cwd=HERE)
    page = open(rendered).read().replace("/*DATA*/", "const DATA = " + json.dumps(data, separators=(",", ":")) + ";")
    with open(os.path.join(EXPLAINER, "index.html"), "w") as f:
        f.write(page)
    print("wrote explainer/index.html,", len(page) // 1024, "KB")


if __name__ == "__main__":
    main()
