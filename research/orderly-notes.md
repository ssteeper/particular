# Which worlds stay orderly longest

The atlas scores each world over one run of at most 1,400 steps, and its chaos axis weights the
start of that run most. `orderly.py` asks how long a world keeps the motion of the orderly end of
the axis when it is run much longer.

    python orderly.py                 # 379 elites to 28,000 steps; about 1.5 min on 4 cores

**Definition.** Positions are logged every 10 steps as in `atlas.py`. Over each window of 140 frames
(one standard run), checked every 100 steps, a world is orderly if
- chaos, `atlas.measure`'s mean folded heading turn per frame (unweighted over the window), is at
  most 0.0081, the upper edge of chaos bin 0, and
- at most half its particles are past 1.6 × its published half-width (`atlas.py`'s escape distance).
  Particles flying apart barely turn, so chaos alone would count that as orderly.

A world's orderly time ends at the last window that passed before the first that failed. Runs are
on the CPU; the atlas was simulated on a GPU (see `explainer/README.md` on drift).

## Results (2026-10-03)

- 323 of 379 elites are never orderly over their first window. Unweighted, most of chaos bin 0
  is not orderly either: the atlas fades the chaos measure toward the end of the run, so a world
  whose motion turns later can still sit in bin 0. 56 pass the first window, 44 of them from bin 0.
- 354 runs end by chaos and 24 by escape; none break (non-finite positions).
- **[0,1,2]** (4 spokes m=10, 40 particles, fitness 0.84) is the only elite still orderly at
  28,000 steps, 20 times its scored run. Run alone to 280,000 steps it is still orderly: its
  largest window chaos per 14,000 steps lies between 0.0042 and 0.0080, and it never passes the
  escape test. The next longest are [0,1,4] (11,800 steps, then chaos), [0,2,0] (10,500, escape),
  [0,2,2] (10,500, chaos) and [0,2,4] (10,300, chaos), all 4-spoke worlds of 40–48 particles.
- Eight runs of each of those five, with start positions perturbed by 1e-9, give exactly the same
  orderly times, so the ranking is not one lucky draw at that noise scale.
- [0,1,2]'s windows come within 0.0001 of the threshold, so its result depends on the
  bin-0 edge; a stricter cut would end it earlier.

`video/shorts/0-1-2-x20.mp4` shows its first 28,000 steps in 12 s (`make_shorts.py 0-1-2 --speed 20`).
