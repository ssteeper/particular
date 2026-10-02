# Reading notes: MAP-Elites

Mouret & Clune, *Illuminating search spaces by mapping elites*, arXiv:1504.04909v1 (April 2015). The PDF is in this folder.
This is a preliminary draft. The authors say every experiment will be rerun before the final version, so read the numbers as indicative only.

## What the paper says

**Core idea.** Most search returns the single best solution. An *illumination algorithm* instead returns the best solution at **each point of a low-dimensional feature space that the user picks**. The result is a map of where good solutions are and how features trade off against performance. Any illumination algorithm also works as an optimizer, since the best cell is the global best.

**The algorithm (Fig. 2).** Three ingredients:

- a genome `x`, which can be high-dimensional, directly or indirectly encoded
- a fitness `f(x)`
- a feature/behaviour descriptor `b(x)`, N-dimensional, discretised into a grid

The loop:

1. Generate `G` random genomes, evaluate each, and put each one in its cell. When two land in the same cell, the better one stays.
2. Repeat: pick a **uniformly random occupied cell**, mutate and/or cross over its elite, then evaluate the child. The child takes the cell `b(child)` if that cell is empty or the child beats the current occupant.

The archive serves as the population, so there are no other population parameters. Selection pressure is easy to reason about: the only things that change over time are which cells are filled and how good their elites are. Insertion is an O(1) cell lookup. Novelty Search needs a nearest-neighbour query instead.

**Why it beats the alternatives it was compared with:**

- Compared with a plain EA, it doesn't collapse onto one peak. Deceptive problems need stepping stones through low-fitness regions, and the archive keeps those stepping stones.
- Compared with NS+LC and MOLE, there is no global performance competition. A small improvement in a mediocre region is still kept. The archive is not separate from the population, so search doesn't cycle back to regions it already explored.
- **Goal switching:** a good elite in one cell often comes from a parent in a *different* cell. Fig. 4 shows that most parents are nearby in feature space, but often not adjacent, and that full lineages cross large parts of the map.

**Metrics (§5, §9.4).** Every metric is normalised against the best value seen for each cell across all runs of all treatments:

- *Global performance*: the best single solution.
- *Reliability*: the mean over all fillable cells of `found / best_known`, where an empty cell counts as 0. This is the main metric for illumination.
- *Precision*: the same mean, but only over the cells this run filled.
- *Coverage*: filled cells divided by the cells that any run ever filled.

**Results:**

- *Retina neural networks* (512×512 map): MAP-Elites beat a traditional EA, NS+LC and random sampling on all four metrics.
- *Soft robots* (128×128 map, axes % bone × % voxels filled): MAP-Elites had higher reliability and coverage. Global performance was not significantly different. Precision was *lower*, because evaluations were spread over far more cells. The maps showed smooth design changes along each axis and exposed a simulator quirk, a high-performing "one voxel wide" island that a single-objective search would rarely find.
- *Real soft arm* (1D, 64 cells, 420 evaluations): MAP-Elites beat grid search and random sampling in the hard middle and low regions of `x`.

**Lesson from the bone example.** The soft-robot authors could not get the optimizer to use bone. Adding a fitness term for bone would have meant guessing a weight. Making bone percentage a feature *axis* gave them the best robot at every bone level instead. This is the paper's main design lesson: **turn a soft preference into an axis instead of a fitness weight.**

**Variants (§8, §9.2):**

- several elites per cell
- biased cell selection, which did not help in the authors' early tests
- crossover restricted to parents that are close in feature space
- *hierarchical* resolution: start with a coarse grid and subdivide it at fixed evaluation counts
- *batched parallel* evaluation: send a batch of children to workers and insert results as they return

**Stated limitation.** The feature space is fixed in advance, so MAP-Elites cannot create new niches over time.

## How the search works now

`search.py` is **random sampling with a weighted scalar score**:

- `random_setup(seed)` builds a k-fold-symmetric arrangement of rings, satellites and spokes.
- `simulate` runs it for 1400 steps.
- `score` sums `1.5·fill + symmetry + smoothness − 2·escape − clumping` at six prefix lengths and keeps the best length.
- `pick` sorts by score and caps each arrangement type (`desc` prefix + piece count) at two picks, so the top 16 aren't all the same.

Measured on the current `search/results.json` (800 candidates, all finite):

- **The scalar score converges on one kind of setup.** 11 of the top 16 by score are a single ring, and 39 of the top 50 start with a ring. The `pick` cap is a hand-written diversity patch over a single-objective ranking, which is the situation the paper starts from.
- **The weights decide what "good" means, and they are guesses.** Correlation with total score: symmetry 0.61, clumping −0.53, fill 0.52, escape −0.38, smoothness 0.34.
- **Symmetry is saturated.** 25% of candidates sit at exactly 1.0 after the `(sym − 0.5)/0.5` clip, and 5.5% sit at 0. Read as a feature, its top bin can't tell those 25% apart.
- **Random sampling already covers a coarse grid well.** Binning the 800 existing runs into 8×8 grids gives 53/64 cells for symmetry × smoothness, 37/64 for fill × symmetry, and 31/64 for fill × smoothness. At this resolution, expect MAP-Elites to help more with **quality per cell** (reliability/precision) than with coverage. Coverage gains should appear at finer grids and in the corners. The arm experiment shows the same pattern: the gaps were in the hard regions.

## Mapping MAP-Elites onto this project

| Paper concept | Here |
|---|---|
| genome `x` | the parameters `_random_pieces` draws (`k`, plus a list of pieces with each piece's kind and continuous parameters), plus the push/pull force scales (§1b) |
| phenotype | the trajectory `simulate(...)` returns, shape `(frames, n, 2)` |
| fitness `f(x)` | the score terms that are *not* used as axes, at the best prefix length |
| features `b(x)` | 2 behaviour measures taken from the trajectory and binned |
| indirect encoding | yes: a few piece parameters generate 40–260 particles with k-fold symmetry, like the CPPN encoding of the soft robots |
| batched parallel evaluation | the existing `multiprocessing.Pool`. Generate a batch of children, `imap_unordered` them, insert as results arrive |

### 1. The genome must be explicit (prerequisite)

Mutation needs something to perturb, and a seed can't be perturbed: changing one bit of a seed gives an unrelated setup. Split `random_setup` into:

- `random_genome(rng) -> dict`, which does the sampling currently in `_random_pieces`
- `build(genome) -> (pos, vel, desc)`, which is deterministic and keeps the existing gap filtering, the 260 cap and the centring

Then `random_setup(seed) = build(random_genome(default_rng(seed)))`, so seed-based reproduction of the existing results keeps working. The 260-particle subsample inside `random_setup` uses the rng. Either move that draw into `random_genome` or give `build` its own derived rng, so that `build` stays a pure function of the genome.

The archive then stores **genomes, not seeds**. Any code that rebuilds a pick from `r["seed"]` has to read the genome when the result came from MAP-Elites: `search.py` `main`, and `make_web.py` `main`. Elites with non-default forces also need their pull/push values carried into the web page. Today `make_web.py` `pack` exports only positions and velocities, and `loadSetup` in `web/app.html` doesn't touch `ui.gScale`/`ui.cScale`. Both need a per-setup `pull`/`push` field, and `loadSetup` must set the sliders from it.

### 1b. Push/pull force scales in the genome

`search.py` `evaluate` calls `simulate` with the default `G` (pull) and `C` (push), so the current search never changes the forces. The web sandbox already has Pull and Push sliders (`web/app.html`: log2 multipliers from −2 to 2, i.e. 0.25× to 4×, applied as `G0 * gScale` and `C0 * cScale`). `simulate(pos, vel, steps, g=, c=)` already accepts both values, so no physics change is needed.

Add two genome genes, `log2_pull` and `log2_push`, each in [−2, 2] to match the sliders, and pass `g=G * 2**log2_pull, c=C * 2**log2_push` to `simulate`. Using the same parameterisation as the sliders means any elite can be reproduced exactly in the sandbox.

A throwaway probe (`score` from `search.py`; 4 top seeds (475, 37, 6, 261) × 7 pull/push combinations; full 1400 steps):

| pull × | push × | spacing `C·push / (G·pull)` | scores | note |
|---|---|---|---|---|
| 1 | 1 | 2.0 | 2.89, 2.86, 2.69, 2.22 | baseline |
| 4 | 1 | 0.5 | 0.97, 0.92, 0.86, 0.90 | fill drops to about 0.2, escape rises to 0.09–0.15 |
| 1 | 4 | 8.0 | 2.59, 2.81, 2.50, 1.71 | smoother (0.87–0.98) |
| 0.25 | 1 | 8.0 | 2.42, 2.94, 2.51, 2.00 | smoothest (0.95–0.99); seed 37 beats its baseline |
| 1 | 0.25 | 0.5 | 2.85, 2.62, 2.38, 1.63 | less smooth |
| 4 | 4 | 2.0 | 2.63, 2.24, 1.29, 1.41 | chaotic: smoothness 0.49–0.82, escape up to 0.29 |
| 0.25 | 0.25 | 2.0 | 1.24, 2.59, 2.22, 2.32 | very smooth, sometimes sparse |

What the probe shows:

- **None of the 28 runs blew up**, so the existing `DT` holds across the slider range for these seeds. Keep the non-finite guard in `evaluate` anyway, because 28 runs is not a stability proof.
- **Force scales strongly change behaviour.** They shift smoothness and escape more than most per-piece parameters do, so they open regions of the feature map that the current fixed-force search can't reach. One example: calm, high-smoothness patterns at pull 0.25×.
- **Good setups don't transfer between force settings.** The default-force winners lose about 2 points of score at pull 4×. Velocities were tuned for the default forces, so the forces and the setup have to evolve together. Searching setups first and then sweeping forces won't work.
- **Partly degenerate genes.** Scaling pull and push by the same factor `s` with the velocities held fixed is equivalent to default forces with velocities divided by √s and time stretched by √s. So the *ratio* (equilibrium spacing `C·push / (G·pull)`, which the sandbox displays) carries the new information. The overall strength mostly overlaps with the velocity genes and the run length. Keeping both genes is still the simplest choice because it matches the sliders, but mutate them in log space.

### 2. Variation operators

- **Parameter mutation:** add Gaussian noise to the continuous parameters of one piece, scaled to each parameter's range. Clamp to the ranges `_random_pieces` uses so the `MIN_GAP` packing rule still holds.
- **Structural mutation, at low rate:** add, remove or resample one piece; change `k`; flip `alternate`.
- **Push/pull mutation:** add Gaussian noise to `log2_pull` and `log2_push` (σ ≈ 0.25), clamped to [−2, 2]. Use a smaller step than for the piece parameters: the probe shows that force scale moves behaviour a lot, so large steps would break mutation locality (see Risks).
- **Crossover (§8 variant):** swap whole pieces between two elites. The paper suggests limiting this to nearby cells so that incompatible design conventions don't mix.

Mutating the piece *template* keeps the k-fold symmetry by construction. Perturbing individual particles would break it.

### 3. Feature axes, and what stays in fitness

Following the bone lesson, a term that becomes an axis comes **out** of the fitness, because otherwise it is optimised twice. Candidates, all computable from code that already exists in `score`:

- **Symmetry order `k*`**, the argmax over `k=2..6` in the existing symmetry loop, or 1 when the best raw match is low. The value is discrete with 6 levels, and it is a behaviour, not the genome's `k`: chaos can break or change the symmetry the setup started with.
- **Smoothness**, raw rather than clipped, as a calm ↔ chaotic axis.
- **Fill**, compact ↔ space-filling.
- **Best prefix length**, short ↔ long-lived patterns.
- **Equilibrium spacing** (`C·push / (G·pull)`), tight ↔ loose. Unlike the others this is a genome value, not a measured behaviour. Using it as an axis turns that dimension into a direct sweep, giving the best pattern at each force ratio. The paper notes that MAP-Elites isn't needed for a dimension you can step through directly, so prefer leaving it in the genome unless a per-ratio gallery is the goal.

Suggested first map: **`k*` (6) × smoothness (8 bins) = 48 cells**, with fitness `1.5·fill − 2·escape − clumping + raw symmetry at k*`. That gives "the best-looking 5-fold chaotic pattern", "the best calm 3-fold pattern" and so on. A single score never returns those, and `pick`'s per-kind cap only approximates them.

Use **raw, unclipped** values to define the bins. The current clips (`(sym−0.5)/0.5` and `(smooth−0.85)/0.15`) push many runs into the end bins.

### 4. Budget and resolution

The README timing (800 setups in about 6 min on 4 cores) works out to roughly 1.8 core-seconds per evaluation, inferred from the README rather than measured. Scale the grid to the budget the way the paper did: the arm used 64 cells for 420 evaluations, the soft robots 128×128 cells for about 1.4M.

- 48–100 cells with an initial batch of 400 random genomes plus 2–4k children takes about 15–30 min on 4 cores.
- Hierarchical variant: start at `k*` × 4 smoothness bins and split to 8 halfway through.

### 5. Prefix-length interaction

`score` picks the prefix length that maximises the total, so the features also depend on that choice. Either:

- (a) compute features at the fitness-maximising length, which is the simplest change but lets fitness move a run between cells, or
- (b) make the length a genome parameter and score only that prefix, which keeps `b(x)` a property of `x`.

Option (b) is cleaner, and its cost is one more mutated parameter.

## How to tell if it worked

Use the paper's metrics directly. The baseline already exists: bin the 800 entries of `search/results.json` into the same grid, keeping the best per cell. That is the paper's random-sampling control, and it costs no new simulation, though rescoring is needed if the fitness changes.

- Same total number of evaluations for both treatments, and several seeds per treatment. The paper used 10–20 replicates and a Mann-Whitney U test.
- Report **reliability** (primary), **precision**, **coverage** and **global performance**, normalised by the best value found per cell in any run.
- Check what the result looks like: a contact sheet laid out as the grid (rows `k*`, columns smoothness) instead of a ranked list. If neighbouring cells don't change gradually, the features or the mutation step are wrong. The paper used the same visual check for its soft robots (Fig. 6).

## Risks

- **Chaos can break mutation locality.** The approach assumes children land near their parents in feature space (Fig. 4). This is an N-body system with a 1/r³ repulsive core, so a small change in the initial conditions can move a trajectory to a different cell, especially over long runs. Measure the parent → child feature distance early. If children land in effectively random cells, MAP-Elites becomes random sampling with extra steps. Mitigations: smaller mutation steps, or shorter scored prefixes.
- **Fitness is aesthetic.** The weights are proxies for "looks good". MAP-Elites doesn't fix the proxy. It reduces the number of weights that have to be guessed, because the axes replace some of them.
- **Fitness is deterministic, which helps.** `simulate` has no noise, so an elite's score never needs re-evaluation. Noisy-fitness MAP-Elites variants aren't needed.
- **Reproducibility changes.** Results are no longer "seed N". The archive must be saved with the genomes, for example as an `archive.json` next to `results.json`.
- **Genome prior bias.** Rings dominate partly because `_random_pieces` draws them with p=0.4 per piece. MAP-Elites uses the random prior only for the initial batch, after which selection is uniform over cells, so the bias should fade. That is worth checking.

## Suggested order

1. Refactor `random_setup` into `random_genome` + `build` without changing behaviour, and check that the existing seeds still reproduce `results.json`. Add `log2_pull`/`log2_push` to the genome. `random_setup(seed)` keeps them at 0. The MAP-Elites initial batch draws them from its own rng *after* the piece draws, so the seeded piece stream stays the same. Pass the scaled `g`/`c` through `evaluate`.
2. Add `features(data)` next to `score` in `search.py`, returning `k*` and raw smoothness. Bin the existing `results.json` to get the random-sampling baseline map.
3. Add a `--map-elites` mode to `search.py`: an archive dict `cell -> (fitness, genome, result)`, a batched loop over the existing `Pool`, and the mutation operators above.
4. Check parent → child feature distance on a short run (risk 1) before spending a full budget.
5. Compare against the baseline at equal evaluation counts using reliability, precision and coverage. Render the grid contact sheet. If it holds up, let `make_web.py` show the archive as a 2D picker instead of a ranked list.

## What we built / results

`atlas.py` (see the README) follows the plan above, with three changes:

- **The map is 3D, from the user's own axes, not `k*` × smoothness.** The axes are orderly ↔ chaotic (raw mean heading turn per logged frame, binned on a log scale), sparse ↔ dense (raw fill) and circular ↔ branching (the share of the trail image's variance in polar bins that changes with angle rather than radius). The grid is 8 × 8 × 6 = 384 cells. Bin edges are evenly spaced between the 1st and 99th percentiles of replicate 0's random genomes, and the end bins are open.
- **Fitness is `symmetry * max(0, 1 − 2·escape)`.** Fill and smoothness became axes. Clumping was dropped too: over random genomes it correlates −0.88 with fill, so keeping it would score density twice. In a pilot that kept it, half of all runs scored 0.
- **The genome goes beyond the plan.** It has species and rule matrices (1–3 species, log2 entries in [−2, 2]) and the run length (option (b) of §5, 400–1400 steps). Piece genes keep the raw draws, and `build` applies the `MIN_GAP` cap. The 260-particle subsample uses a PCG64 state stored in the genome, so `build` depends only on the genome and seeds 0–1999 rebuild bit for bit.

**First run (old fitness, CPU).** Three replicates. Each one ran 3000 random genomes (the control), then MAP-Elites for 3000 evaluations: the first 600 of those random genomes plus 2400 children. That is 18,000 simulations in 21 minutes on 16 cores, with no non-finite runs. Metrics are normalised by the best fitness per cell over all six runs. 343 cells were filled by at least one run.

| | coverage | reliability | precision | global |
|---|---|---|---|---|
| random sampling | 0.833 (0.825–0.845) | 0.771 (0.767–0.774) | 0.925 (0.915–0.935) | 1.000 |
| MAP-Elites | 0.861 (0.845–0.872) | 0.822 (0.812–0.837) | 0.954 (0.932–0.966) | 1.000 |

Reliability separates fully: every MAP-Elites run beats every random run, which gives an exact one-sided Mann-Whitney p = 0.05 for 3 vs 3. Coverage and precision overlap by one run each. As predicted above, at this resolution the gain is mostly quality per cell, because random sampling already fills 83% of the reachable cells.

**Locality (risk 1) held.** The median parent → child feature distance is 0.68 bin widths, against 4.27 for two unrelated random genomes. 79–80% of children land within one cell of their parent (12% for random pairs), and 31–33% stay in the parent's cell. By operator, as median distance and share within one cell:

| operator | median distance (bins) | within one cell |
|---|---|---|
| steps | 0.46–0.49 | 94–95% |
| forces | 0.51–0.54 | 86–87% |
| piece | 0.56–0.59 | 88–91% |
| species | 0.56–0.73 | 72–81% |
| structure | 1.41–1.65 | 53–63% |
| crossover | 2.03–2.09 | 39–42% (deliberate jumps) |

**Caveat (first run): fitness saturates.** Symmetry sits at about 1 for anything round or cleanly k-fold, and 85% of the published elites score ≥ 0.999. So global performance is 1.0 for every run and tells us nothing, precision is compressed, and inside saturated cells the elite is effectively the first world to reach about 1, not the best. Round worlds are always symmetric, so a "best over all slices" view leans round. A stricter symmetry measure, or a quality term that isn't an axis, would be needed before quality per cell means much. The rerun below replaced this fitness; its own ceiling is described there.

**Other observations.**

- Elites drift to smaller setups (90 particles on average, against 125 for random genomes), so MAP-Elites evaluations ran about twice as fast.
- 181 of the 335 published elites have more than one species.
- The prior's ring bias stays: 46% of elite pieces are rings, against 40% in the prior.
- The worlds are very sensitive to their start. Rounding the start positions and velocities to 6/4 decimals moves 21% of elites to another cell, and even 10/8 decimals moves 3%. The web page therefore ships the starts as exact float64. The page's JavaScript physics matches `simulate` to within 1.2e-13 over 200 steps for multi-species elites with one-sided rules.

**Reading the contact sheet (first run).** sparse → dense reads clearly in every panel. orderly → chaotic reads clearly (crisp rings and petals on one side, hairy tangles on the other), but less so in the sparsest rows, where chaotic worlds look like a collapsed core with a few rays flying off. circular → branching really measures round vs angular: panels go from rings and discs, to radial starbursts, to lobes, pinwheels and arms. k separate satellite rings also count as "branching" even though nothing branches.

### Second run (historical, rejected): new fitness, skeleton branching, GPU

This 840,000-evaluation run was never published as final: its fitness saturated again and its branching axis didn't read (see below), so `measure` was revised and the atlas rerun ("Final run" below). `measure` at the time: branching was the share of the trail skeleton's length that sits in radial branches, and fitness was `max(0, 1 − 2·escape) · (1/4 + 3/4·structure) · (1/4 + 3/4·crispness)`. The physics ran on the GPU (`gpu_sim.py`, `--device cuda`).

**Run.** `python atlas.py 140000 --init 5000 --replicates 3`. Each replicate ran 140,000 random genomes, then MAP-Elites with the first 5000 of them plus 135,000 children, in batches of 32 (`--batch` default; 4 batches in flight). That is 840,000 simulations, 47× the first run. The search took 1022 s, rendering the 384 stills about 9 s, and the whole run 1031 s. No run was non-finite. The grid is the same 8 × 8 × 6. All 384 cells were filled by some run, and all 384 are published.

| | coverage | reliability | precision | global |
|---|---|---|---|---|
| random sampling | 0.961 (0.951–0.969) | 0.846 (0.836–0.854) | 0.880 (0.880–0.882) | 0.999 (0.999–0.999) |
| MAP-Elites | 0.989 (0.979–1.000) | 0.979 (0.965–0.995) | 0.991 (0.986–0.995) | 1.000 (1.000–1.000) |

- MAP-Elites beats random sampling on reliability and precision in every replicate, and on coverage too.
- At this budget random sampling already fills 96% of the cells, so the gain is quality per cell, as before.
- Bigger MAP-Elites batches would not have been faster. Children ran at 1.14 ms each in batches of 32, while random genomes ran at 1.33 ms each in batches of 256. The CPU pool's building and scoring now set the pace, so `--batch` stayed at 32 and selection stays fresh: at most 128 children are in flight. I didn't test whether coverage changes with batch size at this budget.

**Fitness of the 384 published elites.**

- Percentiles 0 / 10 / 25 / 50 / 75 / 90 / 100: 0.573 / 0.965 / 0.983 / 0.989 / 0.991 / 0.994 / 0.994. None scores ≥ 0.999 (85% did in the first run).
- **This fitness has its own ceiling:** 71% of the elites are within 0.01 of the maximum. The causes:
  - crispness is 1.0 for 97% of elites (its mapping clips at 1.25);
  - structure reaches 0.98–0.99, and for 350 of the 384 elites (91%) its best k is 2, the other 34 have k = 4. A ring of an even number of particles puts its angular energy on multiples of 2 by construction, so k = 2 is close to free.
- So the old symmetry saturation is gone, but quality per cell is again decided in the third decimal. Global performance is 1.0 for every run.
- Escape is 0 for every elite.
- The elites drift towards short, small runs: 190 of 384 run 400 steps, and the mean is 72 particles (90 in the first run).
- 347 of 384 elites have more than one species.

**Locality.** Children land further from their parents than in the first run. Over the three replicates, the median parent → child distance is 1.22–1.28 bin widths (0.68 before), against 4.48 for random pairs. 59–61% of children land within one cell of their parent (79–80% before), and 19–21% stay in its cell; for random pairs those are 14% and 2%. 4.4–4.8% of children entered the archive.

| operator | median distance (bins) | within one cell |
|---|---|---|
| forces | 0.75–0.80 | 74–77% |
| steps | 0.88–1.01 | 70–71% |
| species | 1.00–1.19 | 59–63% |
| piece | 1.08–1.19 | 64–68% |
| structure | 2.45–2.83 | 37–39% |
| crossover | 2.48–2.88 | 28–32% |

0.3–1% of children (463–1295 per replicate) are logged as operator `none`. In those cases `mutate` failed 20 times and re-evaluated a copy of the parent.

**Reading the contact sheet.**

- **What improved over the first run:** there are no hazy blown-out cores, nothing lopsided and nothing escaping.
- **But the dense half is one motif.** The upper four or five rows of every panel are near-identical radial starbursts. A ring expands or contracts along straight thin spokes, often around a smaller ring, over short runs (e.g. cells 3-6-0 and 3-6-5).
- **sparse → dense reads clearly.** Sparse rows hold separate small rings, 2- and 4-arm crosses, pinwheels and single lines.
- **orderly → chaotic reads only weakly** in the dense rows, mostly as a brighter, more solid core on the chaotic side.
- **circular → branching doesn't read on the sheet.** Panels 1 to 6 look alike, starbursts in every one, and crosses and spokes appear in bin 1 too.
- **Takeaway:** the elites are cleaner than the first run's but less varied. The fitness favours many crisp radial spokes. The likely fixes are in `measure`:
  - don't let k = 2 come free from even particle counts;
  - stop crispness clipping at 1;
  - check the skeleton branching on starbursts, which score anywhere from bin 1 to bin 6.

**What changed after this run (measure v3).** Branching is now radial line energy: ink gradients are split into radial and tangential parts per polar cell, and a cell counts only where the tangential excess lasts over 5 of 16 rings, read at 128 px and 32 px. The skeleton is gone. Structure uses angular harmonics 1–12 only, so the ripple from discrete particles no longer gives k = 2 for free. Crispness is the share of variance a 5×5 blur removes, so it no longer clips. Fitness gains a `(frames / 140) ** 0.2` duration factor, and its floors are 0.4: `max(0, 1 − 2·escape) · (0.4 + 0.6·structure) · (0.4 + 0.6·crispness) · (frames/140)^0.2`. Chaos folds reversals (a turn of π counts as 0), so particles bouncing along a ray read as orderly. A 50,000-evaluation pilot on cuda gave elite fitness p10/p50/p90 of 0.42/0.72/0.83 (max 0.87), with 2.2% of elites within 0.01 of the max. 70% of elites ran 1400 steps. The median parent → child distance was 0.74 bins, and 76% of children landed within one cell. Measuring takes about 3 ms per call.

### Final run (published): measure v3 on the GPU

**Run.** `python atlas.py 50000 --init 2000 --replicates 3 --device cuda` (`--batch` 32, 4 batches in flight). Each replicate ran 50,000 random genomes, then MAP-Elites with the first 2000 of them plus 48,000 children: 300,000 simulations in all, 17× the first run.

- **Time:** 520 s end to end. The search took 514 s (random sampling 70–75 s per replicate, MAP-Elites 88–116 s), and rendering the 379 stills and the sheet the remaining ~6 s.
- **Runs:** no run was non-finite.
- **Cells:** 380 of the 384 cells were filled by some run. 379 are published (the best MAP-Elites elite per cell; one cell was only reached by random sampling).

| | coverage | reliability | precision | global |
|---|---|---|---|---|
| random sampling | 0.923 (0.918–0.929) | 0.678 (0.673–0.687) | 0.735 (0.731–0.740) | 0.880 (0.863–0.901) |
| MAP-Elites | 0.976 (0.961–0.989) | 0.917 (0.908–0.930) | 0.940 (0.928–0.951) | 0.971 (0.930–1.000) |

- Every MAP-Elites replicate beats every random replicate on all four metrics.
- With the ceiling gone, global performance means something again. Random sampling's best world reaches 0.86–0.90 of the best known, while MAP-Elites reaches 0.93–1.00.
- Reliability rises by 0.24 and precision by 0.21, much more than the first run's gains (+0.05 and +0.03). This is the gain MAP-Elites is for, and it couldn't show while fitness saturated.

**Fitness of the 379 published elites.**

- Percentiles 0 / 10 / 25 / 50 / 75 / 90 / 100: 0.279 / 0.494 / 0.647 / 0.749 / 0.815 / 0.839 / 0.917.
- **No saturation:** 1.1% of elites (4) are within 0.01 of the maximum and 5.3% within 0.05, against 71% within 0.01 in the second run and 85% at ≥ 0.999 in the first. The saturation caveats of both earlier runs are resolved.
- **Terms (min / quartiles / max):**
  - structure: 0.002 / 0.72 / 0.84 / 0.90 / 0.95
  - crispness: 0.34 / 0.65 / 0.79 / 0.83 / 0.94
  - duration: 0.78 / 1 / 1 / 1 / 1
  - escape: 0 for nearly all, max 0.07
- **Best k:** 2 for 330 elites, 3 for 30, 5 for 13, 4 for 5 and 6 for 1. k = 2 still dominates, but now has to be earned at low harmonics.
- **The duration factor pushes runs long:** 303 of 379 elites run 1400 steps, and only 10 run 400 (190 of 384 did in the second run). The mean is 62 particles. 300 elites have more than one species.
- **Pieces in elites:** 291 spokes, 102 rings and 96 satellites, where the second run's elites were mostly rings. The sheet's many pinwheels are spoke pieces.

**Locality is back to good.** Over the three replicates:

- The median parent → child distance is 0.76–0.80 bins, against 4.47 for random pairs.
- 74–75% of children land within one cell (random pairs 14%), and 27–28% stay in the parent's cell (random pairs 1%).
- 7.5–8.6% of children entered the archive.
- 573–732 children per replicate were logged as operator `none` (mutate failed 20 times; the parent was re-evaluated).

| operator | median distance (bins) | within one cell |
|---|---|---|
| steps | 0.33–0.35 | 96–99% |
| forces | 0.47–0.50 | 88–89% |
| species | 0.66–0.80 | 67–77% |
| piece | 0.76–0.80 | 82–84% |
| structure | 2.15–2.51 | 38–45% |
| crossover | 2.30–2.38 | 33–34% |

**Stills match the scored runs exactly.** On cuda, `render` draws trajectories that `gpu_sim` simulated, all 379 elites in one call, while scoring had simulated them in batches of 32.

- `gpu_sim` is bitwise independent of grouping. Re-simulating every elite in reversed batches of 32 gave identical arrays.
- Re-scoring the render-path trajectories reproduced every published fitness and feature exactly, with no cell changes. That includes the 345 elites that run ≥ 1000 steps.

**Reading the contact sheet.** The map is far more varied than the second run's.

- **orderly → chaotic reads clearly** in every panel. Left: clean pinwheels with combed parallel arms, crisp rings and crosses. Right: satellite flowers, small rings orbiting inside a ring, and tangled cores.
- **sparse → dense reads clearly.** The bottom rows hold separate small rings, crosses and single curved strands; the top rows hold full pinwheels, discs and starbursts.
- **circular → branching reads at the extremes.** Bins 5–6 are straight-ray starbursts and crosses, and separate satellite circles now sit low. But the axis is a radial-ray proxy, not tree topology:
  - it counts straight rays pointing at the centre;
  - it undercounts curved arms. Swept pinwheels fill bins 1–3 even though they look armed, e.g. cell 0-3-0, a 4-spoke pinwheel with branching 0.07;
  - real branching (forks, filaments splitting) isn't measured at all.
- **Repetition:** the pinwheel motif repeats across much of the orderly half of bins 1–4, and many cells hold near-identical variants of it.
- **Quality regressions, all at low fitness:**
  - The densest row of bin 6 holds starbursts with blown-out cores, e.g. 2-7-5. Fitness agrees: they score 0.28–0.40, with crispness 0.42–0.51, the lowest on the map.
  - The densest rows of bins 4–5 hold thick solid rings and discs. They score 0.32–0.51, with structure low (0.14–0.15 for 0-7-3 and 0-7-4).
  - Five cells are empty on the sheet: the top-right cell of bin 5 and the top-right four of bin 6, the dense + chaotic + most branching corner. Four were never reached by any run, and one only by random sampling, which isn't published.
  - No lopsided or escaping elites.
