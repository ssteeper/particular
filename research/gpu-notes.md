# GPU acceleration for the atlas: findings

Status: **implemented.** `gpu_sim.py` has `simulate_batch(setups, log_every, device, batch)`, and `test_gpu_sim.py` checks it against numpy `simulate`. `atlas.py --device cpu|cuda|auto` (default auto) uses it. The first sections record the torch-ops benchmark that motivated the work; "The fused kernel" records what was built instead, and "In the atlas" covers how the atlas uses it.

## Machine

- GPU: NVIDIA GeForce RTX 5070 Ti, 16 GB (Blackwell, compute capability 12.0), CUDA 13.3 driver. There is also an integrated AMD GPU, which isn't used.
- CPU: AMD Ryzen 7 7800X3D, 8 cores / 16 threads.
- Python 3.12.10, numpy 2.4.1.
- Installed for the benchmark: `torch 2.11.0+cu128`, via `python -m pip install torch --index-url https://download.pytorch.org/whl/cu128`. Blackwell needs a CUDA 12.8 or later wheel; older wheels fail with "no kernel image".

## Why the GPU fits

- Nearly all the cost is the all-pairs force calculation in `particular.simulate`: up to 260² pairs per step, over 400–1400 steps.
- MAP-Elites already evaluates candidates in batches (`atlas.py` `run_pool`), so a whole batch can be simulated as one `(B, n_max, n_max)` tensor job instead of one CPU process per candidate.
- Candidates have different particle counts, so pad each to the batch's `n_max` and mask the padding out. Park padded particles far away with zero velocity, and zero their force coefficients through the mask.
- Each candidate's species rules gather into `(B, n, n)` pull/push coefficient tensors once, before the step loop, exactly as numpy `simulate` does with `gp`/`cp`.

## Benchmark

Method: the real physics, matching numpy `simulate` (`coef = -g/r^3 + c/r^4`, semi-implicit Euler, `DT = 1e-5`, log every 10 steps), on `random_setup(seed)` for seeds 0..255. Those have 40–260 particles, mean 126. All runs were 1400 steps.

- The GPU step was captured in CUDA graphs and the log kept on the GPU, copied back once at the end.
- Times are per candidate, amortised over the batch.
- The CPU baseline ran while the atlas search was loading all cores, so treat it as rough.
- Species rules weren't benchmarked. They add one gather before the loop and an elementwise multiply per step, so the cost should be small.

| Configuration | ms per candidate | Padding efficiency | Peak GPU memory |
|---|---|---|---|
| CPU numpy, one process, float64 | ~2370 | — | — |
| CPU pool, 16 workers (measured in the atlas run, random genomes) | ~95 | — | — |
| CPU pool, 16 workers (measured in the atlas run, MAP-Elites children, mean n 90) | ~48 | — | — |
| GPU fp64, B=8, n-bucketed | 22.2 | 0.95 | 38 MiB |
| **GPU fp64, B=16, n-bucketed** | **18.8** | 0.90 | 76 MiB |
| GPU fp64, B=32, n-bucketed | 21.9 | 0.83 | 155 MiB |
| GPU fp64, B=64, n-bucketed | 26.5 | 0.70 | 309 MiB |
| GPU fp64, B=64, unsorted | 64.2 | 0.29 | 309 MiB |
| GPU fp64, B=256, unsorted | 69.6 | 0.29 | 1222 MiB |
| GPU fp32, B=16, n-bucketed | 5.2 | 0.90 | 39 MiB |
| **GPU fp32, B=32, n-bucketed** | **4.4** | 0.83 | 77 MiB |
| GPU fp32, B=64, unsorted | 16.1 | 0.29 | 157 MiB |

"n-bucketed" means candidates are sorted by particle count before batching. Without that, about 70% of the work is padding, and large batches (B ≥ 128) get slower for the same reason.

Speed-ups for the physics alone:

| Configuration | vs one CPU process | vs pool, random genomes | vs pool, MAP-Elites children |
|---|---|---|---|
| fp64 B=16 bucketed | ~126× | ~5× | ~2.5× |
| fp32 B=32 bucketed | ~540× | ~22× | ~11× |

## Accuracy

Seeds 475, 37, 6 and 261, compared with numpy `simulate`:

- **GPU fp64:** agrees to about 1e-14 for the first ~400 steps, then diverges O(1) by step 1400. This is chaos amplifying round-off: the GPU sums forces in a different order than numpy. Scores still match numpy within ~0.005, and the best run length chosen was the same in all 4 seeds.
- **GPU fp32:** off by 0.1–0.7 of the frame width by step 800, and fully decorrelated by step 1400. Scores differ by up to 0.14 (seed 37), and the best run length changed in 2 of 4 seeds.
- **The browser has the same problem already.** `web/app.html`'s JS physics sums forces in yet another order, so an atlas elite replayed in the page already drifts from the run that was scored, in fine detail though not in character. Exact cross-platform reproduction of long runs isn't possible in any case; the summary statistics are what's reproducible.

## The fused kernel (`gpu_sim.py`)

The torch-ops version above runs about 20 small kernels per step over a padded `(B, n_max, n_max)` tensor, so it is limited by memory traffic and padding. torch.compile can't fuse them here: it needs Triton, which isn't available on Windows. `gpu_sim.py` uses one hand-written CUDA kernel instead, compiled at first use with the NVRTC that ships in the torch wheel. Torch's own `torch.cuda._compile_kernel` wants `CUDA_HOME` set, so `gpu_sim` calls NVRTC directly. No CUDA toolkit is needed.

- **One block per candidate.** Positions, velocities, species and the S×S force tables sit in shared memory. Each thread sums the force on its particle over j = 0..n−1, the same order as numpy. Then the block steps and logs, with no padding at all. The per-pair cost is `inv = rsqrt(d²)`, `coef = inv³ (gp + cp·inv)` and FMAs.
- **Mixed sizes and run lengths share a launch.** Each block stops at its own `steps`, so no bucketing is needed. Setups are sorted largest first, then grouped `batch` to a launch.
- **Each launch runs about 1e9 pair-steps (~0.1 s)**, then the state goes back to global memory. That keeps every launch far below the Windows driver watchdog (2 s).
- **Deterministic.** There are no atomics, and blocks are independent. Results are bitwise identical however setups are batched.
- CUDA graphs aren't needed, since there are only a few launches per batch. Two variants were tried on 400 random genomes: numpy's exact formula (`sqrt`, two divisions, no FMA) took 6.8 ms per candidate, and the rsqrt + FMA version 2.5 ms. Both are equally close to numpy (see the parity results below).

Throughput: 2000 `atlas.random_genome(0, i)` genomes (mean n 123, mean 905 steps, 16.8 M pair-steps per candidate), with the machine otherwise idle.

| Path | ms per candidate |
|---|---|
| CPU pool, 16 workers, `atlas.evaluate` (300 genomes, simulate + measure) | 108 |
| torch ops + CUDA graphs, B=16, bucketed by steps and n (200 genomes) | 21.7 |
| `gpu_sim`, calls of 64 / 128 / 256 / 512 genomes | 5.2 / 2.9 / 2.1 / 1.4 |
| `gpu_sim`, one call of 2000 genomes | 1.0–1.2 |

The kernel runs at the card's fp64 peak, about 19 fp64 ops per pair. That makes it about 1.9e10 pair-steps/s over 70 SMs, but only 2.8e8 per SM. A single n=260, 1400-step candidate therefore takes 0.34 s on its one SM, and that sets the floor on any call's wall time. Calls need hundreds of candidates to fill the card, or two calls have to overlap on separate streams. The fixed overhead per call is 3–11 ms.

Scoring, on trajectories from the same genomes: `atlas.measure` takes 4.2 ms per call on one core (median 3.5, max 15; it scales with frames × n). In a 16-worker Pool, pickling included, it costs 0.56 ms per candidate. `random_genome` + `build` take 0.8 ms on one core. Scoring is cheaper than the GPU, so the GPU remains the bottleneck as long as building and scoring both run in the pool, overlapped with the GPU.

Parity: GPU vs numpy, against a floor of numpy vs numpy with every starting coordinate moved by one ulp. Tested on the 18 scenes and 30 atlas elites.

- Over 200 steps, the GPU's max |Δpos| is at or below the floor in every case: elites median 5e-14 (floor 9e-14), max 2e-8 (floor 2e-8). The calm scenes agree to 1e-15–1e-12. `wave` and `galaxy` reach 1e-2 by step 200, but numpy is just as far from itself on them.
- Over full runs (48 setups scored), the GPU and the floor look alike. Median |Δfitness| and |Δfeature| are 0.0000 for both. Max |Δfitness| is 0.093 for the GPU and 0.047 for the floor. Max |Δfeature| is 0.003–0.005 for the GPU and 0.002–0.010 for the floor. The cell changed for 1 of 48 in both, and no setup broke on one side only.

## In the atlas

`atlas.py` `run_gpu` does `run_pool`'s job:

- The main process makes genomes `batch` at a time, so MAP-Elites selects parents from the archive as it is when each batch is made.
- Up to `GPU_STREAMS` = 4 batches simulate at once, each from its own thread. `gpu_sim` gives each thread its own CUDA stream, so the batches overlap on the card.
- The pool builds each batch's setups (`P.build`) before it simulates and scores its trajectories (`score`, the second half of `evaluate`) after. A new batch is made as soon as one leaves the GPU.
- Random sampling has no selection to keep fresh, so it uses batches of `GPU_SAMPLE_BATCH` = 256. MAP-Elites uses `--batch` (default 32), so at most 128 children are in flight, against 48 on the CPU path.
- With `--device cuda`, `render` draws trajectories that `gpu_sim` simulated, so each still shows the run that was scored.
- `evaluation.device` in `atlas.json` records the device.

Throughput with several batches in flight: `simulate_batch` alone, 2000 random genomes, ms per genome.

| Batch | 1 stream | 2 | 3 | 4 |
|---|---|---|---|---|
| 64 | 4.9 | 4.8 | 3.5 | 2.8 |
| 128 | 3.1 | 2.8 | 2.1 | 2.1 |
| 256 | 2.0 | 2.0 | 1.5 | 1.4 |
| 512 | 1.5 | 1.3 | 1.3 | 1.2 |

The streams do overlap: 20 copies of the largest world take 0.35 s from 1, 2 or 3 threads at once. A batch takes as long as its slowest world, though, and that world shares its SM with other blocks, so the gain from more streams comes unevenly.

End to end: `atlas.py 2000 --replicates 1` (4000 evaluations, including the stills), with one version of `measure`.

| | Wall time | Random sampling | MAP-Elites | Fillable | Random coverage / reliability | MAP-Elites coverage / reliability / precision / global |
|---|---|---|---|---|---|---|
| `--device cpu`, 16 workers | 261 s | 196 s | ~57 s | 285 | 0.863 / 0.711 | 0.937 / 0.906 / 0.967 / 0.994 |
| `--device cuda`, batch 32 | 9 s | 4 s | ~4 s | 285 | 0.849 / 0.713 | 0.947 / 0.912 / 0.963 / 0.995 |

- The bin edges calibrated on each device's 2000 random genomes match to the third decimal on chaos and branching. On density they differ by up to 0.004 (lowest edge 0.129 on cpu, 0.133 on cuda).
- A cuda MAP-Elites run isn't repeatable exactly, and neither is a cpu one: results are inserted in the order they finish. Two cuda runs filled 267 and 265 cells.
- With `--batch` 32 / 64 / 128 / 256 (4000 evaluations, an earlier `measure`), the search took 13 / 11 / 7 / 6 s, and MAP-Elites coverage didn't change beyond noise.
- **The 840,000-evaluation run** (historical, rejected for its fitness) was `atlas.py 140000 --init 5000 --replicates 3`: 1031 s in all (search 1022 s, rendering 384 stills about 9 s).
  - Throughput was steady. Random sampling (batches of 256) took about 185 s per 140,000 genomes, 1.33 ms each. MAP-Elites children (batches of 32) took about 154 s per 135,000, 1.14 ms each.
  - A 1-replicate pilot at 20,000 evaluations took 59 s end to end. Extrapolated, that predicted about 1200 s for the search; it took 1022 s.
- **The published run**, with measure v3, was `atlas.py 50000 --init 2000 --replicates 3 --device cuda`: 300,000 evaluations in 520 s (search 514 s, rendering 379 stills and the sheet about 6 s).
  - Random sampling took 70–75 s per 50,000 genomes, 1.4–1.5 ms each. MAP-Elites took 88–116 s per 48,000 children, 1.8–2.4 ms each. The children are slower than in the 840k run because measure v3's duration factor pushes elites to 1400-step runs, so the children run longer too.
  - The default run (`atlas.py`, 3000 evaluations × 2 treatments × 3 replicates) took 40 s end to end on cuda.
  - On the CPU (16 workers, measure v3), a 1000 + 1000 evaluation run took 162 s: about 103 ms per random genome and about 62 ms per child. At those rates the default run would take about 23 minutes (extrapolated, not run), so the GPU is about 35× faster end to end.
- **The stills are the scored runs.** With `--device cuda`, `render` draws trajectories that `gpu_sim` simulated, all elites in one call, while scoring had simulated them in batches of 32.
  - `gpu_sim` is bitwise independent of grouping. Re-simulating the 379 published elites in reversed batches of 32 gave identical arrays.
  - Re-scoring the render-path trajectories reproduced every published fitness and feature exactly, including the 345 elites that run ≥ 1000 steps.

## Recommendation

1. **Use fp64.** The fused kernel replaced the padded, bucketed torch-ops design (B ≈ 16). fp32 is ~4× faster still, but it changes which candidates win and which run length gets chosen. If fp32 is ever wanted, search in fp32 and re-simulate in fp64 on the CPU only the candidates that would enter the archive, so the stored elites are faithful.
2. Keep numpy `simulate` as the reference. The parity gate is `python -m unittest test_gpu_sim`, which runs the scenes over 200 steps against the one-ulp floor. Rerun it after any change to the physics in `simulate` or in the kernel.
3. On the GPU, an evaluation takes about 1.1–2.4 ms end to end, depending on run lengths and `measure`'s cost. The CPU pool's building and scoring, not batch size, set the pace: batches of 32 ran as fast as batches of 256. The default `--batch` 32 is fine for big runs too.
