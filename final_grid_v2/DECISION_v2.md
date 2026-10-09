# Final grid v2 (grid B): penalty 5, shared reward, PPO fix — pre-registration

**Status: pre-registered 2026-10-09, before any v2 run was launched.** Nothing below may change after
the first v2 job starts; any deviation is appended at the end with its date and reason.

Grid A (branch `final-grid` @ `7552398`, crash penalty 40, SAC and PPO at SB3 defaults) is **not
changed** by this file and is run and reported exactly as `final_grid/DECISION.md` describes. Grid v2
is an additional, separately pre-registered grid; both grids are reported.

## Why a second grid

In grid A's tuning (penalty 40, varied-width tracks, d20), PPO never learned: the default P0 averages
0.124 mean fractional laps on vw_val (4 seeds) and completes no laps, so PPO's diversity curve cannot
be measured (floor effect). After grid A was decided, validation-only experiments (vw_val, d20, seeds
99/98, Desmond's code `6e5b60e`) showed that **crash penalty 5 plus a diagnosed PPO fix** brings PPO
into SAC's range. Score = best of the 400k/800k/1.2M/1.6M/final checkpoints, as in tuning:

| setting (vw d20) | penalty | seeds | best-checkpoint score | laps at that checkpoint |
|---|---|---|---|---|
| SAC default (S0, grid A) | 40 | 99/98/97/96 | 0.551 mean | — |
| PPO default (P0, grid A) | 40 | 99/98/97/96 | 0.124 mean | 0% (s99, s98) |
| PPO γ 0.999 + n_steps 8192 (P1, tuning) | 40 | 99/98 | 0.028 / 0.126 | 0% / 0% |
| same as P1, penalty 5 | 5 | 99/98 | 0.516 / 0.521 | 18% / 0% |
| same + `--ent-coef 0.01` (the v2 PPO setting) | 5 | 99/98 | 0.498 / 0.672 | 32% / 30% |

- Same code, tracks and seeds as P1; only the penalty differs between rows 3 and 4. The machine that ran
  rows 4–5 reproduces grid A's reference determinism lines exactly (identical trajectory hashes on
  vw_synthetic_track_0/7/12), so the simulator is the same.
- The entropy bonus kept PPO's exploration from collapsing (policy std at 2 M: 0.62 / 0.77 vs
  0.14 / 0.06 without it), matching a 3-seed one-track test (no collapsed seed with it, 1 of 3 without).
- **No test data was used**: no evaluation on vw_test, test, real, narrowA or narrowB.
- SAC γ 0.999 collapsed at penalty 40 (score 0.017 / 0.008 at 800k) but not at penalty 5 (0.18 / 0.29
  at 800k, 0.24 / 0.29 at 1.2M, run `S1P5`). This is **not** used for any SAC setting below.

## Settings (locked)

| | v2 setting |
|---|---|
| shared reward | progress along the centreline, **crash penalty 5** (`--crash-penalty 5`), time cost 0, no lap bonus |
| SAC | SB3 defaults, γ 0.99. **Locked, whatever the S1P5 (γ 0.999) result shows.** |
| PPO | `--gamma 0.999 --n-steps 8192 --ent-coef 0.01`, everything else SB3 default. **Disclosed as PPO-specific tuning.** |
| tracks | `vw_synthetic_track_*`, diversity 1 / 5 / 20 / 100 (the nested prefixes, as grid A) |
| seeds | 0 / 1 / 2 — the same as grid A, so every v2 cell has a paired grid-A cell |
| budget | 2 M steps, 1 environment, checkpoints every 100k (train.py defaults, as grid A) |

Job lines, exactly:

- every SAC line = grid A's SAC line for the same cell **+ `--crash-penalty 5`**, nothing else;
- every PPO line = grid A's PPO line for the same cell **+ `--crash-penalty 5 --gamma 0.999 --n-steps 8192 --ent-coef 0.01`**;
- tags `v2_<ALGO>_d<N>_s<seed>`; `final_grid_v2/check_setup.py` verifies all of this on every machine.

No changes to `train.py`, `sb3_wrapper.py`, `evaluate.py` or `launch_queue.ps1`: v2 differs from grid A
only in job-file flags.

## Evaluation (identical to grid A)

- During training: `launch_queue.ps1 -Checkpoints 200000,400000,600000,800000,1000000,1200000,1400000,1600000,1800000`
  plus the final model, on **vw_val only** — deterministic, one lap, centreline spawn, 5 episodes per
  track, 15,000-step cap (the queue's fixed evaluation).
- **Checkpoint selection (primary):** per run, the best of the 400k / 800k / 1.2M / 1.6M / final
  checkpoints by vw_val mean fractional laps; tie → higher lap % → earlier checkpoint. These five are the
  checkpoints shared with every tuning run. Grid A's documents do not state a test-time selection rule,
  so **this rule is proposed for grid A as well**, so that the paired comparison is defined; Desmond
  confirms before any test evaluation.
- **Secondary (robustness only):** mean vw_val score over the last five evaluated checkpoints
  (1.2 M, 1.4 M, 1.6 M, 1.8 M, final). This differs from the 1.6–2.0 M, 100k-spaced mean used in the
  PPO diagnosis; it is reported, never used to select.
- **Test (after both grids are frozen):** vw_test + the 23 real circuits, 10 episodes per track, once
  per grid, run centrally by Desmond; both grids reported side by side.

## Who runs what

| person | job file | runs | `-MaxThreads` | results folder |
|---|---|---|---|---|
| Grant | `final_grid_v2/jobs_v2_grant.txt` | 6 SAC: seed 0 d1/5/20/100 + seed 1 d1/d5 | 12 | `results/final_grid_v2_grant` |
| Chris | `final_grid_v2/jobs_v2_chris.txt` | 6 SAC: seed 2 d1/5/20/100 + seed 1 d20/d100 | 12 | `results/final_grid_v2_chris` |
| Desmond | `final_grid_v2/jobs_v2_desmond_ppo.txt` | 12 PPO: seeds 0/1/2 × d1/5/20/100 | 12 | `results/final_grid_v2_desmond` |

Each of the 24 cells (SAC/PPO × diversity 1/5/20/100 × seed 0/1/2) appears exactly once.

## Caveats (stated in advance)

1. SAC at penalty 5 on varied-width tracks has not been run before (SAC worked at penalty 5 on the
   uniform-width seed-0 grid, and at penalty 40 on varied width).
2. The v2 PPO setting was tested at d20 with 2 seeds only; d1, d5 and d100 are untested.
3. PPO received more tuning than SAC (grid A tuning: about 9 PPO vs 6 SAC settings, plus the October
   PPO diagnosis), and the v2 PPO setting was chosen after seeing PPO fail under grid A's settings.
4. v2 was decided after grid A's decision; it is an addition, not a replacement, and is labelled as such.

## Deviations

None yet.
