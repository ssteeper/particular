# particular

Explore and evolve particle worlds: particles attract at long range and repel at short range, producing trails from orderly orbits to branching chaos. `atlas.py` searches this space with MAP-Elites, preserving strong worlds across a behavior map; the web viewer lets you browse and run them, then change their rules.

Project report: [web/report.html](web/report.html) (portable, expandable HTML).
Explainer: [explainer/index.html](explainer/index.html) walks through the physics, the atlas axes, fitness and mutation, with clips of the worlds (see [explainer/README.md](explainer/README.md)).

## Quick start

Install the viewer and simulation dependencies:

```sh
python -m pip install numpy matplotlib pillow
```

The checked-in atlas data and thumbnails are used to build the page; no search or rendering run is needed first:

```sh
python make_web.py
python -m http.server 8000 --bind 127.0.0.1 --directory web
```

Open <http://127.0.0.1:8000/>. The page opens on the fittest world in the atlas.

If `search/atlas.json` is missing or holds no worlds, `make_web.py` exits with an error telling you to run `python atlas.py` and leaves `web/index.html` untouched.

To evolve a new atlas, optionally install PyTorch with CUDA support, then run the search and rebuild the page:

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python atlas.py --device cuda
python make_web.py
```

The PyTorch install is only needed for CUDA; `atlas.py --device cpu` runs without it. `--device auto` (the default) selects CUDA when available, otherwise CPU. CUDA 12.8+ is required for RTX 50xx cards. The atlas command defaults to 3,000 evaluations per treatment and three replicates; the checked-in atlas came from `python atlas.py 50000 --init 2000 --replicates 3 --device cuda` (about 9 minutes on an RTX 5070 Ti, measured 520 s end to end). See `python atlas.py --help` for options. It writes `search/atlas.json` and atlas images.

## How the atlas works

Each candidate world is a genome of particle-layout pieces, force scales, species rules, and run length. MAP-Elites mutates and recombines genomes, keeping the best-fitness world found in each behavior-map cell rather than selecting by one score alone.

The three axes are trajectory-derived proxies: orderly to chaotic measures heading changes; sparse to dense measures trail coverage; circular to branching measures persistent radial edge energy. They are not complete descriptions of a world. Fitness is a separate image/trajectory score combining escape, large-scale symmetry, trail crispness, and duration. The map's finite bins and chosen measurements limit what the search can distinguish; empty cells mean no world was found there, not that none exists.

Worlds can have one to three species, each governed by pull and push rules against every other species. The atlas evolves these rules along with the world, and the viewer's Ecosystem panel can change them: the species count (one to four), species colors, the pull and push matrices, and built-in or saved rule presets. Coloring is user-controlled when loading a world.

## Source map

- `particular.py`: particle simulation, world/genome construction, starting setups, and species rule presets.
- `atlas.py`: MAP-Elites search, behavior measurements, fitness, and atlas output.
- `gpu_sim.py`: optional batched CUDA implementation used by the atlas; imports PyTorch only when used.
- `make_web.py` and `web/app.html`: build the viewer page using the checked-in atlas JSON and thumbnails; output is `web/index.html`.
- `web/report.html`: project research report; `research/` contains the MAP-Elites and GPU notes.

`particles.ipynb` is the original notebook. `search.py`, `make_gallery.py`, and `make_video.py` are legacy tools for the earlier search/gallery/video workflow, not prerequisites for the atlas viewer; `make_hype.py` renders a short hype video of five atlas worlds, colored by time, to `video/hype.mp4`.
