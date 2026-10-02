# particular

Particles that attract each other at long range and repel at short range, plotted as colored trails.

Project report: [web/report.html](web/report.html) (portable, expandable HTML).
`particles.ipynb` is the original notebook.

- `particular.py`: vectorized simulation plus named starting setups (`SCENES`). `breath`, `square`, `quartet`, `galaxy` and `constellation` go from simplest to most complex; the `eco_*` scenes use species (see Ecosystems).
- `make_gallery.py`: renders a still per scene into `gallery/`; ecosystem scenes are colored by species
- `search.py`: builds hundreds of random setups from seeds, simulates them in parallel, scores each one (frame fill, symmetry, smooth motion, escaping particles, clumping) and saves the best to `search/`
- `atlas.py`: maps the possible worlds with MAP-Elites and keeps the best one in every cell of an orderly ↔ chaotic, sparse ↔ dense, circular ↔ branching map (see An atlas of possible worlds)
- `gpu_sim.py`: `simulate_batch(setups)`, the same simulation for many setups at once on an NVIDIA GPU, which `atlas.py` uses when it finds one (see Running the atlas on a GPU); `test_gpu_sim.py` checks it against `simulate`
- `make_web.py`: builds `web/index.html`, an in-browser sandbox with every scene, the rules presets, the search picks and the atlas, from `web/app.html`
- `make_video.py`: renders `video/particular.mp4`, which draws each scene in turn with a soundtrack generated from the simulation

```
pip install numpy matplotlib pillow
python make_gallery.py
python make_video.py video/particular.mp4   # needs ffmpeg
python search.py 800                        # about 6 minutes on 4 cores
python atlas.py                             # 40 s with an RTX 5070 Ti; about 23 minutes on 16 CPU cores
python make_web.py
```

## Ecosystems

Particles can belong to species with their own rules. `simulate(pos, vel, steps, species=..., rules=...)` takes an int array of species (0..S-1) and `{"pull": S x S, "push": S x S}` multipliers: particle i of species a is accelerated by j of species b with `-G*pull[a][b]/r^2 + C*push[a][b]/r^3` along `r_i - r_j`. The matrices can be one-sided (a is drawn to b while b runs from a), and a negative pull repels at long range. Two species settle about `2 * push / pull` apart. Leaving both out gives the original physics exactly.

`RULES` holds the presets, each with species colors and a one-line note:

- `classic`: one species, the notebook's physics
- `chase`: predators are drawn to prey and hold off 5 units; prey flee them
- `clusters`: like attracts like; unlike species barely pull and keep 10 units apart
- `orbit`: a heavy core pulls the others hard but hardly feels them, so they circle it
- `compete`: rock-paper-scissors; each species chases the next and flees the one before

The scenes `eco_chase`, `eco_clusters`, `eco_orbit` and `eco_compete` show each preset at work. Every scene builder returns a dict (`pos`, `vel`, `steps`, `half`, `cmap`, `species`, `rules`); classic scenes have `species` and `rules` set to `None`.

In the web sandbox the Ecosystems tab loads these scenes, and the Ecosystem panel edits the rules in force: 1 to 4 species, their colors, and the pull and push matrices (rows are the species that moves, columns the species it reacts to). Changing the rules restarts the run. Pick a preset, save your own under a name (kept in the browser's local storage), delete saved ones, or export and import them as JSON. Mix species deals the current setup's particles out to the species in turn, the ring tool asks which species the new ring is, and Drawing can color trails by species. Copy setup as JSON includes species and rules.

## An atlas of possible worlds

`atlas.py` doesn't rank setups by one score. It keeps the best setup in every cell of a map of how they behave, using MAP-Elites (Mouret & Clune 2015; notes in `research/map-elites-notes.md`). Each world is a genome: a JSON dict holding the pieces' parameters, the force scales `log2_pull`/`log2_push` (the same log2 multipliers as the sandbox sliders), 1 to 3 species with their `log2_rules` matrices, and a run length. `particular.random_genome(rng, wild=True)` draws one. `particular.build(genome)` turns it into `pos`, `vel`, `species`, `rules`, `g` and `c` for `simulate`, and depends only on the genome. `random_setup(seed)` is `build(random_genome(default_rng(seed)))`, so old seeds rebuild exactly.

The three axes are measured on the trajectory:

- orderly → chaotic: the mean turn of a particle's heading per logged frame, folded so that reversing along a line counts as going straight (binned on a log scale)
- sparse → dense: the share of the frame that the visible trails cover
- circular → branching: how much of the trail image's edge energy belongs to lines pointing at the centre and staying radial over at least five of 16 rings, read at two scales. Rings, discs, rosettes, tick-fringed rings and separate circles score near 0; spokes, starbursts, crosses and arms score high

The map has 8 × 8 × 6 cells. Fitness rates how well a world is drawn, from things that aren't axes: `max(0, 1 - 2 * escape) * (0.4 + 0.6 * structure) * (0.4 + 0.6 * crispness) * (frames / 140) ** 0.2`. Structure is macro-scale k-fold symmetry: of the trail image's energy in angular harmonics 1 to 12, the share on multiples of the best k = 2..6 above what chance would put there, times how much of the image is angular at all. Only low harmonics count, so the ripple of n discrete particles on a ring earns nothing. Crispness is the share of the 128-pixel trail image's variance that a 5×5 blur removes: thin separate trails score high, haze and blobs low. The duration factor favours worlds that keep developing over a short initial burst. Escape is the share of particles that ever fly past 1.6 times the frame's half-width. Fill and smoothness are axes, and clumping is left out because it mostly measures sparseness a second time. Measuring a trajectory takes about 3 ms on one CPU core.

```
python atlas.py [evals] [--init N] [--replicates R] [--workers N]   # defaults: 3000, 600, 3
                [--device cpu|cuda|auto] [--batch N] [--out DIR]    # defaults: auto, 32, search
```

Each replicate runs `evals` random genomes, which is the random-sampling control. It then runs MAP-Elites for the same number of evaluations, starting from the first `--init` of those genomes. Each step picks a random filled cell and mutates its world: piece parameters, structure, forces, species rules or run length, or a crossover that swaps a piece with a nearby cell. The bin edges are fixed from replicate 0's random genomes. The run writes to `--out` (default `search/`):

- `search/atlas.json`: axes and bin edges, the paper's metrics (coverage, reliability, precision, global performance) for both treatments, parent → child locality, and the best elite per cell over the replicates. Each elite has its genome, fitness, features, cell, parent cell, the operator that made it, steps and view half-width.
- a still per elite in `search/atlas/`
- `search/atlas.png`: the map as one picture, with one panel per branching bin

### Running the atlas on a GPU

`gpu_sim.simulate_batch(setups)` runs `simulate` for a list of setups at once on an NVIDIA GPU, in float64. A setup is what `build` returns plus `steps`, and the result is one `(frames, n, 2)` array per setup. It uses one CUDA kernel, compiled at first use by the NVRTC library that ships with torch, so no CUDA toolkit is needed. torch is optional; nothing else imports it. To install it and check the GPU against numpy:

```
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128   # RTX 50xx needs CUDA 12.8+
python -m unittest test_gpu_sim                                                  # skipped without a GPU
```

`atlas.py --device auto`, the default, simulates on the GPU when torch can see one and on the CPU otherwise; `--device cpu` or `--device cuda` picks one. On the GPU the main process makes the genomes and simulates them in batches, four batches at a time, while the CPU pool builds the setups and scores the trajectories. MAP-Elites makes each batch of `--batch` children (default 32) from the archive as it is at that moment. Larger batches run faster, smaller ones let selection see newer elites. The stills are simulated on the same device as the scoring, so they show the run that was scored. `atlas.json` records the device under `evaluation.device`.

On a given device, a genome simulates the same way every time, but the two devices round differently. The worlds are chaotic, so the same seeds give a map that differs in detail between devices, about as much as moving every starting coordinate by one ulp would (`research/gpu-notes.md`). The MAP-Elites half also varies a little from run to run on either device, because results come back in whatever order they finish.

The published atlas comes from `python atlas.py 50000 --init 2000 --replicates 3 --device cuda`: 300,000 simulations in 9 minutes on an RTX 5070 Ti, rendering included.

In the sandbox, the Atlas tab shows the map as a grid of thumbnails. Pick which two axes go across and up; the third becomes a slice control, or All for the fittest world over every slice. Empty cells are worlds the search never reached. Hover over or focus a world to read its description, fitness, features and where it came from. Click it to run it with its own forces and species rules: the Pull and Push sliders and the Ecosystem panel follow, and worlds with several species are colored by species. Arrow keys move across the grid. Loading any other setup restores that setup's own forces and rules (1× and classic when it has none). The atlas starts are stored exactly (as float64), because the worlds are chaotic enough that rounding the start would change many of them.
