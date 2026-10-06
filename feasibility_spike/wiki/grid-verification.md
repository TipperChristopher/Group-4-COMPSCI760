---
title: Verification of Desmond's seed-0 diversity grid
type: synthesis
updated: 2026-10-06
sources:
  - Group4_DiversityGrid_seed0.zip (343 MB; extracted to COMPSCI 760/_verify/, not in git) (high)
  - results/grid_s0/*.csv, results/track_geometry/ on this branch (high)
  - results/ppo_diagnosis/repro_desmond_grid/*.csv (high; see warning on *_trainpool files)
  - context_for_aolin.md (Desmond's notes, COMPSCI 760 root) (mixed)
  - run journal Attempts 3–9 (mixed)
---

# Verification of Desmond's seed-0 diversity grid

On 2026-10-05 Desmond ran the first real cells of the study's grid: PPO and SAC
× diversity 1/5/20/100, seed 0, 2 M steps, plus baselines. Aolin was asked not
to trust it but to verify it from the raw files. **Verdict: the data and protocol
are sound and every headline number reproduces — one of them bit-for-bit.** The
weakness is statistical: one training seed, one checkpoint, and three distinct
spawn poses per track. This page records what was checked, what it showed, and
one mistake made during the checking.

## What was checked

| check | result |
|---|---|
| Protocol consistency across 8 cells | identical commit `00539c1b`, reward sha `b8fe9de7`, manifest `d70dbc8f`, 0 dirty files, penalty 5, n_envs 1, 2 M steps |
| Evaluation summary recomputed from per-episode CSVs | max deviation from his README **0.48 m** (a rounding) |
| Reward identity | `return == progress_m − 5·collision` to 5e-4 → penalty really is 5 |
| **Re-evaluate his PPO d1 weights on our machine** | **max \|Δ progress_m\| = 0.0000** on 10 episodes — bit-identical |
| PPO internals | σ < 0.1 by 0.51–0.56 M in all 4 cells; post-collapse KL 0.25–0.38, clip 0.51–0.58 |
| Late-decline ("finding 7") | −3.9 / −7.0 / −21.1 / −23.1 % vs his −4/−7/−21/−23 |
| Spawn coverage | 5 eval seeds give exactly **3 distinct spawns** per track (seeds 0≡1, 2≡3); his own `spawn_check` agrees |

## Headline results (his data, recomputed)

**PPO:** 0/860 evaluation episodes completed a lap, and all 860 ended in a crash.
It is flat from the first 200 k steps, so the four PPO cells compare equally broken
policies. The cause is in `ppo-diagnosis.md`.

**SAC — training-log lap rate per 200 k bin (stochastic, % of episodes ≥ 1 lap):**

| div | 0.8–1.0 M | 1.2–1.4 M | 1.6–1.8 M | 1.8–2.0 M |
|---|---|---|---|---|
| 1 | **38** | 31 | 11 | 22 (declining) |
| 5 | 18 | 31 | 50 | 46 |
| 20 | 37 | 41 | 60 | 52 |
| 100 | 23 | 43 | 50 | 54 (still rising) |

d1 is the only cell that degrades. SAC is **not converged at 2 M** for d ≥ 5.

**SAC — generalisation (held-out `test_track_*`, deterministic):**

| div | held-out lap rate | late-training lap rate | transfer ratio |
|---|---|---|---|
| 1 | 2% | 15.9% | **0.13** |
| 5 | 22% | 46.4% | **0.47** |
| 20 | 31% | 54.5% | **0.57** |
| 100 | 38% | 50.2% | **0.76** |

Diversity measurably improves transfer to unseen synthetic tracks. On real
circuits it is **0/460** at every diversity level.

## The noise floor (the reason not to over-read the grid)

Evaluated the last five checkpoints (1.6–2.0 M) of every cell on Spielberg + Silverstone
(`repro_desmond_grid/*_ck*.csv`):

| | within-cell checkpoint SD | between-cell SD (final ckpt) |
|---|---|---|
| PPO | 5.9 m | 11.3 m |
| SAC | 9.1 m | 21.5 m |

The signal-to-noise ratio is only about 2:1. PPO d20 swings 6.5 → 33.7 m across 400 k steps with
nothing changed. **Averaging the last five checkpoints makes both algorithms
monotonic in diversity** (PPO 5.1→25.5→25.7→27.3 m; SAC 10.6→12.6→24.6→46.9 m),
whereas the final checkpoint alone is non-monotonic. The reported "d20 > d100"
inversion is a checkpoint-selection artifact. Adjacent-level Wilcoxon tests are
mostly non-significant; only the d1-vs-d100 extremes are strongly significant.

## The penalty comparison (a lead, confounded)

Our SAC (penalty 40, our `learning_curve.py` code path) against his SAC d1 (penalty 5,
`train.py`), each trained on one track, both evaluated on 5 tracks neither had seen
plus 2 real circuits, with 10 episodes per track:

| model | unseen synthetic finished | real finished |
|---|---|---|
| ours SAC p40 | **29/50 (58%)** | 0/20 |
| his SAC p5 | 6/50 (12%) | 0/20 |

This is consistent on all 5 tracks but **confounded** with the code path (the PPO control
pair differs too) and n=1. It is not evidence for a penalty decision on its own; see
`decisions.md` D6.

### Correction: "SAC cannot lap its own training tracks" (retracted the same day)
That was measured on our local `synthetic_track_*`, which were generated with
seed 0. Desmond's pool is seed 123, and **0/20** tracks match despite identical
names (his track 0 = 164.06 m, ours = 189.04 m). The claim, the "ceiling vs gap"
decomposition built on it, and the first penalty comparison were all void.
Fingerprinting tracks by `progress_m / laps` caught it. The surviving
gap result is the transfer-ratio table above, computed from his files only. Files
evaluated on the wrong maps are kept in `repro_desmond_grid/` (named
`*_trainpool`, `repro_SACd1_oursynth`) as a record only. See `incidents.md`.

## Cost facts (for planning)

From `run_config.json` wall times on Desmond's machine: PPO 2 M = **32–47 min**;
SAC 2 M = **5.4 h**. On Aolin's machine under 4–8 concurrent runs, PPO took
1.0–1.3 h and SAC is pacing ~12 h.

## See also

- `ppo-diagnosis.md` — why every PPO cell is 0 laps
- `sim-to-real-width.md` — why every cell is 0% on real circuits
- `methodology.md` — spawn and checkpoint-selection consequences
- `final-run-plan.md` — what the next grid should change
- `incidents.md` — the track-identity pitfall
