# The fitness landscape behind the atlas

The atlas keeps one elite per cell, which hides how fitness is distributed: how rare good worlds
are, what limits them, and whether the map has converged. These notes cover phases 1–3 of the
landscape study: a logged rerun of the published atlas settings, analyses of that log, and GPU
probes around the published elites.

## Phase 1: the logged rerun

`atlas.py --log` writes every evaluation to `<out>/evals.jsonl.gz` (genome, features, fitness
terms, operator, parent cell, whether it entered the archive), in arrival order.

    python atlas.py 50000 --init 2000 --replicates 3 --device cuda --out search/landscape/run --log
    python landscape.py analyse

- 294,000 evaluations (150k random, 144k MAP-Elites children), 0 broken, 504 s on an RTX 5070 Ti
  (the unlogged published run took 520 s: logging costs nothing measurable). Log: 86 MB.
- Random sampling reproduced the published bin edges exactly, so the random half of the log is
  the same 150k genomes behind the published atlas. MAP-Elites differs in detail (GPU batches
  complete in a different order), so the rerun is an independent replicate set at the same budget.

`landscape.py analyse` bins every logged evaluation with the published edges and writes figures and
`analysis.json` to `search/landscape/`.

## Phase 2 findings

**1. Structure, not crispness, caps the dense half of the map.** Mean elite fitness falls from 0.83
(sparsest row) to 0.52 (densest). The structure factor falls with it (0.95 → 0.64) while crispness
stays flat (0.90 → 0.84). Structure is the main cost for 124 elites, concentrated in the dense rows
and high branching bins; crispness for 245, in the sparse and chaotic cells (`limits.png`). The
ceiling itself drops: the best structure ever logged per density row falls from 0.95 to 0.85 and
the random 95th percentile from 0.89 to 0.37 (`bias.png`). Over random genomes density barely
correlates with fitness (Spearman −0.01); only the top of the distribution is capped. The cap is
mostly measured, not real: the structure term reads only 12 angular harmonics (finding 8).

**2. The reported symmetry order is a tie-break, not a finding.** 330 of 379 elites report order 2.
For a perfectly k-fold image, every divisor of k scores the same in `_structure` (all its energy
lies on multiples of each), and the strict `>` keeps the smallest. Over random genomes, k=4 worlds
report order 2 in 83% of cases and k=6 worlds report 2 or 3 in 93%. Fitness is unaffected; the
`order` label is not a measure of symmetry order. It also means a 2-fold world can score as high
as a 6-fold one.

**3. The atlas has not converged in its dense and chaotic regions.** Reliability (mean of best /
best known per cell) is still rising at 50k evaluations: 0.84 at 25k, 0.88 at 40k, 0.89 at 50k
in replicate 0; random sampling plateaus near 0.65 (`convergence.png`). 273–359 children per
replicate still entered the archive in the last 20% of the run. The three replicates agree within
0.05 in most sparse and orderly cells, but 95 cells differ by more than 0.1, concentrated in the
dense rows of branching bins 3–6 (`headroom.png`). The published atlas and this rerun agree within
0.02 in only 54% of cells (mean |difference| 0.041, each better in about 87 cells). Per-cell
differences of this size are search variance, not measurement noise (finding 6).

**4. Random genomes are mostly chaotic, and orderly cells are rare.** Random sampling reached 366
of the 381 cells ever filled, but 42 of them fewer than 10 times in 150k. The 10 most visited
cells take 26% of all random samples, mostly chaotic ones. Orderly cells at high branching are
the rarest (`rarity.png`). MAP-Elites gains most where random genomes are rarest: median elite
minus best random genome is 0.19 in cells hit fewer than 10 times, 0.11 in cells hit 1,000–10,000
times (`gain.png`).

**5. Global pull and push set behaviour; species rules barely register.** Gradient boosting on
genome traits over the random genomes (R² 0.49 for fitness, 0.53–0.56 for the axes):
- The three axes are dominated by `log2_pull`, then `log2_push` and particle count.
- Fitness depends most on particle count (fewer is better: predicted 0.37 at 47 particles,
  0.22 at 256), pull (weaker is better), and whether there are rings (rings score lower: their
  symmetry sits above the 12 harmonics the structure term reads, finding 8). k = 1 scores lowest.
- Species count and summary statistics of the rule matrices (mean, spread, asymmetry) have near-zero
  importance for fitness and for all three axes. These statistics may miss rule structure that
  matters (for example a chase), so this says that rules matter less than global forces on average,
  not that they never matter.
- About half the variance is unexplained by these coarse traits: piece geometry and chaos.

## Phase 3: GPU probes

    python landscape.py noise
    python landscape.py sweep
    python landscape.py structure

- `noise`: 14,427 runs in 10 s (every elite unperturbed and 32 times with start positions
  perturbed by 1e-9; 40 elites 16 times at each of 1e-12, 1e-6, 1e-3), plus 15 s to replay the log.
  `sweep`: 46,260 runs in 33 s (20 elites, 48 × 48 grid of `log2_pull` × `log2_push`).
  `structure`: 1,979 runs (every elite and 200 logged random genomes per density row, seeded) in
  3 s, 19 s with measuring.
- All leave 5% of the machine free by default (`--headroom 0.05`): 15 of 16 cores, below-normal
  priority, a GPU sleep of 5% of each batch's time, GPU memory capped at 90%.
- Sanity: the unperturbed re-runs reproduce all 379 published elites (fitness, features, cell;
  largest fitness difference 5e-5, the published rounding), and `structure`'s random genomes
  reproduce their logged fitness and structure term to the same 5e-5.

## Phase 3 findings

**6. Fitness is a stable function of the genome; its noise floor is near zero outside the most
chaotic columns. Cells are less stable.** Under 1e-9 start-position noise the median elite's
fitness std is below 0.001 (79% of elites below 0.001, 4.7% above 0.01, max 0.048), and 83% of
elites never leave their cell. Fitness noise concentrates in the chaotic columns: 17 of the 18
elites with std > 0.01 sit in chaos bins 7–8. Cell changes have three causes (`noise.png`,
`noise_cells.png`; 23 elites leave in ≥ 50% of runs):
- Chaos (chaos bins 7–8, mean share leaving 0.13 and 0.25, at most 0.05 in bins 1–6). [7,0,2]
  and [6,5,5] diverge within 2–4 frames even at 1e-15 and scatter over 1–2 bins, so their
  published values are one draw ([7,0,2]: perturbed mean 0.706, published 0.623).
- Edge-sitters. Elites within about 0.1 bins of an edge flip under ordinary small feature noise
  ([1,7,5], [3,5,5], [5,4,1], [5,6,4] and 12 elites in chaos bins 7–8).
- Ring-only worlds. These are an unstable symmetric state: unperturbed, the ring stays perfectly
  symmetric for the whole run, so the published value is the special unbroken run. 1e-9 noise
  grows into clumped rays around frame 60 of 140 (about 30 frames earlier per 1000× more noise)
  and moves every perturbed run to the same, mostly sparser cell, at a fitness cost of 0–0.06
  ([0,5,4], [0,7,3], [2,7,3], [4,7,4], [7,7,2]). 9 of 34 ring-only elites shift by more than
  3 std, against 1% of the others.

For most elites the spread grows with the noise scale instead of saturating. Median std is 0, 0,
0 and 0.005 at 1e-12, 1e-9, 1e-6 and 1e-3;
the feature change is 0, 0, 0.02 and 0.15 bins. Only chaotic elites already spread at 1e-12
([7,1,4]: 0.016 at 1e-12, 0.036 at 1e-9) (`noise_eps.png`). At 1e-3, elites mostly get worse
rather than noisier: the median shift is −0.018, 55% lose more than 0.01 and [7,1,1] loses 0.44.
For the median elite, 88% of those runs leave the cell, since elites sit near cell edges (median
0.05 bins to the nearest crossable edge). There is no winner's curse at this floor: published
minus perturbed mean has median 0.000, mean 0.001 and max 0.056 (`winners_curse.png`).
Phase 2's replicate spread (median 0.047) and published-vs-rerun difference (median 0.017, mean
0.041) are far above this floor. Three noisy draws would spread by a median 0.000, and noise
could explain none of the 95 cells with spread > 0.1. So those differences are search variance
(replicates find different genomes), not measurement noise. Spread still correlates with noise
(Spearman 0.48): the chaotic cells are both noisy and unconverged.

**7. Pull × push planes run in bands along push − pull, are rough at small scales, and each
spans much of the map.** Twenty elites were swept over a 48 × 48 grid of `log2_pull` × `log2_push`
on [−2, 2]² (step 0.085), with everything else fixed (`sweeps.png`, `sweeps/<cell>.png`). They
are the five favourites plus 15 farthest-point picks over cell coordinates and fitness, which
take the map's corners and both fitness extremes first.
- `log2_push − log2_pull` explains a median 67% of a plane's fitness variance (pull alone 25%,
  push alone 8%), so fitness runs in diagonal bands. [Inference: scaling both forces together
  keeps the settling distance 2·push/pull.] Exceptions are flat, high planes: [7,1,1] and
  [5,1,5] (13–14% explained; 42–79% of the plane is within 0.05 of the elite).
- Roughness is real under the discrete dynamics, but it is not all structure (see the closure note
  below; the original "at least 25× the noise floor" came from quiet elites and does not apply in
  explosive regions). The mean |Δfitness| between neighbouring grid points is
  0.036 for the median plane. It grows with distance
  (0.044, 0.059, 0.094 and 0.141 at 2, 4, 8 and 16 steps): broad bands with fine ruggedness on
  top. It also grows with chaos, from 0.007 between neighbours in chaos bin 1 to 0.067 in bin 8.
  Only [7,1,4] (noise std 0.031) is as rough as its own noise. In [7,6,0] the fine ruggedness
  includes timestep-aligned artifacts and sampling moiré.
- One plane reaches 36–151 cells (median 82 of 381). Half of all neighbouring grid points differ
  in cell (21% in the most orderly chaos bin, 75% in the most chaotic). The elite's cell covers a
  median 31 of 2,304 points, 17 of them connected to the elite. For 6 of 20 elites ([7,1,1],
  [0,2,5], [7,5,5], [4,4,3], [3,5,5], [1,7,3]), the elite's point is an island narrower than one
  grid step.
- No elite is its plane's maximum: planes peak 0.002–0.35 higher, in another cell for 17 of 20.
  Within their own cell, though, the elites are near-optimal in pull and push. The best in-cell
  grid point beats the elite by a median 0.003, and by at most 0.01 in 16 of 20 planes; 7
  elites beat every in-cell point. The exceptions are [7,5,5] (+0.068), [0,2,5] (+0.065),
  [5,7,2] (+0.023) and [3,5,5] (+0.020).
- Favourites: [7,1,1] beats every in-cell point (0.691 against 0.657; its cell is 4 isolated
  points). [5,1,5] scores 0.804 against an in-cell best of 0.807. [5,3,0] scores 0.762 against
  0.765, which is also its plane's maximum. [3,6,0] beats its cell (0.710 against 0.703). [5,7,2]
  scores 0.478 against 0.501, the only favourite with clear room to improve along pull and push.

**8. The dense rows' structure cap (finding 1) is mostly measured, not real.** `structure` splits
`_structure`'s S = a × s. All of the fall is in a, the angular share of the energy. Elites, sparsest
to densest row: a 0.77 → 0.19, s 0.99 → 0.94. Random genomes: a 0.25 → 0.06, s 0.63 → 0.58. Dense
worlds keep their symmetry, but `_structure` can't see it. Most of the dense worlds it scores low
are sunbursts in which each particle of a ring draws a ray, so they are N-fold with N = 26–88. All
of their angular energy lies above harmonic 12, the last one `_structure` reads, so a is about
0.01–0.06 for worlds that are perfectly symmetric. 34 of the 38 dense elites (rows 6–8) with
structure below 0.4 are built from rings.

Ruled out:
- Saturation, finding 1's guess: `_structure` bins the raw trail points, and the saturated image
  with the same 12 harmonics scores lower still (densest elite row a 0.13).
- The k range 2..6: +0.04 to the densest random row, because ring counts mostly have small
  divisors (52, 68, 70, 76).
- Ring weighting: unweighted rings give a 0.79 → 0.18.
- Framing: dropping the outer 4 rings moves a by less than 0.01.
- Centring: off by at most 0.003 half-widths.

**An independent check agrees.** It rotates the saturated trail image by 2π/k for k = 2..128,
correlates it with itself (Q), and weights Q by the share of the image's variance that is angular.
For random genomes Q doesn't fall with density (0.50 → 0.56), and the weighted score barely
correlates with fill (Spearman −0.03, against −0.27 for the published structure). Agreement with
the published structure is Spearman 0.68 in rows 1–4 but 0.41 in rows 6–8; with all harmonics it
is 0.85 and 0.86 (`structure_agreement.png`).

**A smaller part of the cap is real.** With every harmonic, a = 1 − coverage exactly (Parseval).
Coverage is each ring's share of energy at m = 0, and it rises with density (elites 0.04 → 0.41), so
filled k-fold patterns have less angular contrast. The 4-spoke fans of [2,7,2] score s = 1.00 but
a = 0.33. Ratio of the densest row's random 95th percentile to the sparsest row's: published 0.46,
all harmonics 0.93, image resolution 0.76, independent check 0.81 (`structure_ceiling.png`). About
two thirds of the drop comes from the measure; one third is real lost contrast.

Examples (`structure_examples.png`):
- [2,7,4], rings of 70 and 26 particles, fill 0.60: structure 0.10 → 0.93 with all harmonics,
  fitness 0.41 → 0.84.
- A random 42-particle ring (row 6): 0.00 → 0.94.
- [2,7,2] (4 spokes, fill 0.56): 0.67 → 0.78, limited by coverage.
- [0,0,0] (sparse reference): 0.91 → 0.95.

With all harmonics, 81 of the 139 dense elites gain more than 0.05 fitness and 34 more than 0.2.
The densest row's mean goes from 0.52 to 0.69. These gains are far above finding 6's noise floor.
They are a lower bound, because the search optimised the published term: the best of just 200
random genomes in that row already reaches 0.80. The favourite [5,7,2] (satellites around a
66-particle ring), finding 7's one favourite with room to improve, goes from 0.48 to 0.66.

Proposals only; fitness is unchanged:
- **A (minimal).** Keep S = a × s and the 0.3 scale, but bin the trails at 512 sectors, read all
  harmonics 1..256 and try k = 2..85. The densest row's random 95th percentile goes from 0.40 to
  0.88. Scrambled worlds (each particle's path turned by its own random angle) score at most 0.07
  on average, below the published term's 0.10.
- **B (stricter).** The same score on the saturated trail image resampled on a 48 × 1024 polar
  grid, so only symmetry the picture resolves counts. Rays that merge into a solid core count as
  uniform. The densest row's 95th percentile is 0.71, and sunbursts score about 0.75.

Rejected:
- The best k per ring: +0.01 for dense elites at twice the chance floor.
- Contrast without coverage, s · a / (1 − a): the scrambled floor of sparse elites rises to 0.57.
- A pixel-resolution limit on the raw points: unsaturated cores dominate, and a 68-ray sunburst
  drops to 0.15.

Caveats:
- Under A, a plain rotating ring of 74 particles scores 0.26 from its 74-fold ripple (published
  0.00, B 0.25).
- The `order` label stays a tie-break (finding 2).
- The best structure per row over the full log was not recomputed.

## Outputs

`search/landscape/`: `analysis.json`, `fitness.png`, `limits.png`, `bias.png`, `rarity.png`,
`gain.png`, `headroom.png`, `late.png`, `rerun_diff.png`, `convergence.png`, `traits.png`
(`analyse`); `noise.json`, `noise.npz`, `noise.png`, `noise_cells.png`, `noise_eps.png`,
`winners_curse.png` (`noise`); `sweeps.json`, `sweeps.npz`, `sweeps.png`, `sweeps/<cell>.png`
(`sweep`); `structure.json`, `structure_decomposition.png`, `structure_ceiling.png`,
`structure_agreement.png`, `structure_examples.png` (`structure`). `search/landscape/run/` (log,
rerun atlas and stills) is regenerable and not committed.

## Closure: integrator and ripple investigation (2026-10-03)

A follow-up looked at whether the fixed-step semi-implicit Euler integrator (DT = 1e-5) distorts the
map. It is closed, with no change: the integrator, DT, force laws, fitness, published atlas, viewer
and video stay as they are. Explosive patterns that arise from the discrete dynamics are accepted as
part of this generative-art project. They are not claimed to be accurate continuous-time physics,
and not every starburst is numerical.

Observations, all under the discrete dynamics and none a statement about physical structure:
- Sweep ripples. In the explosive region of [7,6,0], timestep-aligned close-encounter kicks make
  fine stripes. The 48 × 48 grid aliases them into broad moiré arcs; shifting or refining the grid
  changes the arcs. The speckle mixes chaotic sensitivity with timestep artifacts. Local noise was
  about 0.03 against about 0.055 for neighbouring fitness variation, roughly 2×.
- DT audit of the 379 published elites at DT, DT/4 and DT/16: 13 had a speed at DT more than 10×
  the speed at DT/16. These are strong timestep-sensitivity flags, not a converged-physics proof.
  None of the five favourites tripped the speed flag. 102 elites changed cell, which alone does not
  show an explosion.
- Integrator comparison on 96 worlds: fixed Euler, leapfrog/KDK, Yoshida4, RK4 and adaptive KDK.
  Higher order at fixed DT did not reliably remove extreme ejections. Adaptive KDK looked promising
  but has an expensive tail: 1 of 96 worlds hit the 1024× force-evaluation cap at each of eta 0.2
  and 0.1, and the eta 0.05 reference had 2 capped. The reference is incomplete, so this supports no
  universal-fix or accuracy claim. Short-reference, safe-substep and softened comparisons and the
  convergence analysis were not finished; safe substepping was not tested as a fix, and nothing
  here shows that all stable worlds are unchanged.

Correction to finding 7: "roughness is real structure, not noise" and the "at least 25×" noise
floor ratio rested on the quiet elites' near-zero noise and do not hold as a general claim. In the
explosive region the roughness includes timestep artifacts and chaotic sensitivity (about 2× the
local noise in [7,6,0]). Read the roughness figures as properties of the discrete dynamics.

The scratch outputs of this work (integrator comparison, dt audit and ripple runs, kept in a
temporary directory) were discarded exploratory artifacts and are not part of the repository.
