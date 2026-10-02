# How Particular works: the explainer

`index.html` is a single page that explains the simulation, the genome, the three behavior axes, the
fitness score, MAP-Elites and the mutation operators, with looping clips of the worlds it talks about.
It ends with an explorer of every cell in the published atlas. Open it from this folder; it loads its
clips from `clips/` and its images from `figs/`, and needs no server.

## What is here

- `index.html`: the built page.
- `template.html`: the page source. TeX is written as `\( ... \)` inline and `$$ ... $$` for display;
  the page's data is inserted at `/*DATA*/`.
- `clips/`: one looping mp4 per world shown, plus a poster (its final frame). `reel.mp4` is
  `video/particular.mp4` re-encoded at 540 px.
- `figs/`: the branching-measurement images and `atlas-sprite.jpg`, every atlas still at 128 px.
- `data/`: the measurements the page is built from.
- `pipeline/`: the scripts that produce all of the above.

## Rebuilding

Needs numpy, matplotlib and pillow, ffmpeg on PATH, and Node for the math. From `explainer/pipeline/`:

```sh
npm install                  # once: MathJax, for rendering TeX to SVG at build time
python measure_atlas.py      # re-run all 379 atlas elites on CPU and measure them
python measure_random.py     # the first 300 random genomes of replicate 0
python mutants.py            # one child per mutation operator from the elite in cell 6-4-3
python figures.py            # ray maps, harmonic spectra, atlas sprite, metric tables
python render_clips.py       # every clip and poster, and the reel
python build.py              # template.html + data/ -> index.html
```

The whole run takes about 3 minutes on 4 cores. The measuring scripts cache trajectories in `work/`
(about 100 MB, not committed). To change only the text or layout, edit `template.html` and run
`build.py`.

## Notes

- The published atlas was simulated on a GPU. The worlds are chaotic and the CPU rounds differently,
  so each clip is a CPU re-run: 371 of the 379 elites land in the same cell, and the numbers quoted
  under the clips are the CPU measurements.
- Clips log positions every 5 steps instead of 10 so the motion is smooth. Measurements use every
  other frame, which is the same as `atlas.py`'s `log_every=10`.
- The mutation examples take the first seed that produces a valid child for each operator, so they
  are not picked for looks.
