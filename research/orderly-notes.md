# Which worlds stay orderly longest

The atlas scores each world over one run of at most 1,400 steps, and its chaos axis weights the
start of that run most. `orderly.py` asks how long a world keeps the motion of the orderly end of
the axis when it is run much longer.

    python orderly.py                 # 379 elites to 28,000 steps; about 1.5 min on 4 cores

**Definition.** Positions are logged every 10 steps as in `atlas.py`. Over each window of 140 frames
(one standard run), checked every 100 steps, a world is orderly if
- at most 10% of its particles are past 1.6 × its published half-width (`atlas.py`'s escape
  distance) at the window's end, and
- chaos, `atlas.measure`'s mean folded heading turn per frame, unweighted over the window and
  averaged only over the particles still inside the escape distance, is at most 0.0081, the upper
  edge of chaos bin 0.

A world's orderly time ends at the last window that passed before the first that failed. Runs are
on the CPU; the atlas was simulated on a GPU (see `explainer/README.md` on drift).

What this does and does not measure: chaos is how sharply paths turn per 10 steps, so "orderly"
means smooth paths. It does not mean a fixed or repeating pattern. Over thousands of steps an orderly
world's trails can still overlay into a busy image.

## Results (2026-10-03)

- **[0,2,4]** (4 spokes m=11, 44 particles, fitness 0.83) lasts longest: 10,300 steps, 7.4 times
  its scored run, ending by chaos. Through 14,000 steps no particle escapes, the farthest stays
  within 31 units, and its 4-fold symmetry stays exact. Its chaos rises from about 0.005 to 0.0095
  around step 11,000-12,000 and falls back to 0.0073 by 14,000; the orderly time is the first
  crossing.
- Next: [0,0,3] (6 spokes m=7, 9,700 steps), [0,1,3] (6,000), [0,0,0] (5,700), [0,1,1] (5,500).
  Eight runs of each of the top five with start positions perturbed by 1e-9 give exactly the same
  orderly times.
- 326 of 379 elites are never orderly over their first window. Unweighted, most of chaos bin 0
  is not orderly either: the atlas fades the chaos measure toward the end of the run, so a world
  whose motion turns later can still sit in bin 0. 53 pass the first window, 42 of them from bin 0.
- 350 runs end by chaos and 29 by escape; none break (non-finite positions) and none reach the
  28,000-step limit.

`video/shorts/0-2-4-x10.mp4` shows [0,2,4]'s first 14,000 steps in 12 s (`make_shorts.py 0-2-4 --speed 10`).

## Correction: [0,1,2]

The first version of `orderly.py` ended a world only when more than half its particles had
escaped, and averaged chaos over all particles. It named [0,1,2] (4 spokes m=10, 40 particles) the
longest-lasting, orderly past 280,000 steps. That was an artifact. From about step 3,000 its outer
particles leave: 20 of 40 are past the escape distance by step 10,000 and keep coasting outward
(the farthest about 2,850 units out at step 280,000). Exactly half escaping did not trip the
"more than half" test, and the escapees, moving in near-straight lines, pulled the averaged turn
down (0.0064 at the start, 0.0018 at step 5,000). The 20 particles that stayed kept exact 4-fold
symmetry but looping paths. Under the corrected rules [0,1,2] ends by escape at 2,700 steps.
`video/shorts/0-1-2-x20.mp4` (28,000 steps) and `0-1-2-x200.mp4` (280,000 steps) were rendered
before the correction; they show that run, not an orderly world.
