---
title: Results — numbers with sources
type: reference
updated: 2026-10-06
sources:
  - team_repo/feasibility_spike/reward_experiment/results/*.json (high; September-2026 pilots, seed-0 local maps)
  - results/ppo_diagnosis/** on team/crash-penalty-flag (high; 2026-10)
  - results/grid_s0/*.csv, results/track_geometry/ (high; Desmond, 2026-10-05)
---

# Results

Every table cites the JSON file it came from, under
`feasibility_spike/reward_experiment/results/`. **All values are single-seed
pilots** (n=1 seed per training run; 5 eval seeds per eval cell) — point
estimates, not ranked results with confidence intervals. The grid with ≥3
seeds and bootstrap CIs is not yet run (`open-questions.md`).

## Unseen-track evaluation (5 tracks × 5 seeds, one-lap, centreline)

Metric: mean fractional laps (±std where noted); crash/finish counts.

### SAC (1.2M checkpoint) — `eval_sac_1m.json`

| track | laps mean | outcome | lap time |
|---|---|---|---|
| synthetic_track_0 | 1.00 | finish 5/5 | 23.6 s |
| synthetic_track_1 | 0.80 | 2 finish / 3 crash | — |
| synthetic_track_2 | 1.00 | finish 5/5 | 22.4 s |
| Spielberg | 0.106 | crash 5/5 | — |
| Silverstone | 0.057 | crash 5/5 | — |

### PPO (2M) — `eval_ppo_vn_2m.json`

| track | laps mean | outcome |
|---|---|---|
| synthetic_track_0 | 0.134 | crash 5/5 |
| synthetic_track_1 | 0.199 | crash 5/5 |
| synthetic_track_2 | 0.240 | crash 5/5 |
| Spielberg | 0.013 | crash 5/5 |
| Silverstone | 0.014 | crash 5/5 |

### Gap-follower (threshold 20) — `eval_gap_protocol.json`

| track | laps mean | outcome |
|---|---|---|
| synthetic_track_0 | 0.740 (std 0.536) | — |
| synthetic_track_1 | 0.587 | — |
| synthetic_track_2 | 0.044 | — |
| Spielberg | 1.000 | finish 5/5 |
| Silverstone | 0.176 | crash 5/5 |

### Random — `eval_random_protocol.json`

~0.01–0.05 laps on all tracks, crash on all.

**Read:** RL wins in-distribution (SAC 1.0 on synthetic_0/2); baseline wins
out-of-distribution (gap 1.0 on Spielberg where both RL policies crash). The
gap-follower is high-variance on synthetic (std 0.536 on track_0).

## Narrow-track eval (nar7) — `eval_sac_nar7.json`

SAC (1.2M), 5 seeds, one-lap:

| track | spawn clearance | laps mean | outcome |
|---|---|---|---|
| synthetic_nar7_0 | 1.06 m | 0.118 | crash 5/5 |
| synthetic_nar7_1 | 1.05 m | 0.134 | crash 5/5 |

Compare real circuits: Spielberg 0.106, Silverstone 0.057 — nearly identical.
The failure reproduces on synthetic geometry with real-like spawn clearance.

## Learning curves

| run | source JSON | headline |
|---|---|---|
| PPO+VN 1M | `learning_curve_cp40_vn_1000000.json` | best 0.344 @300k → 0.098 @1M; reward osc +37…−19 (std 14.3) |
| PPO+VN 2M (p40) | `learning_curve_cp40_vn_2000000.json` | plateau ~0.1 laps |
| PPO+VN 2M old reward | `learning_curve_cp40_vn_old_2000000.json` | plateau ~0.1 laps |
| PPO+VN 2M (p5) | `learning_curve_cp5_vn_2000000.json` | plateau ~0.1 laps |
| SAC+VN 1M | `learning_curve_cp40_sac_vn_1000000.json` | completes 2 laps @400/600/700k; reward→196.6; lap 40.3→24.7 s |
| SAC+VN 2M (partial) | `learning_curve_cp40_sac_vn_2000000.json` | lap 22.7 s @1.2M; paused 12/20 (superseded: `sac_overfit` finished 2 M, Oct 4) |

## Reward scan — `reward_scan2.json`

Same rollout through both reward formulas (penalty 40):

| behaviour | OLD (V1) | NEW (V2) |
|---|---|---|
| standstill | 100.00 | 0.00 |
| crawl 0.25 m/s | 263 (peak) | ~0 |
| 20 m/s | 189 | pays metres |
| wall crash | — | −22 |

V1 global optimum is the crawl; V2 pays metres. See `reward.md`.

## Baselines (aligned, threshold 20) — `baselines.json`

Consistent with the gap/random eval above. Gap synthetic_0 0.740 (std 0.536),
synthetic_1 0.587, synthetic_2 0.044, Spielberg 1.000 (5/5), Silverstone
0.176 (crash 5/5). Random ~0.01–0.05 all crash.

## Spawn study — `spawn_study_Spielberg.json`, `spawn_study_Silverstone.json`

Env-level: spawn offset **0.79–0.81 m → 0.00 m** switching
`rl_grid_static` → `cl_grid_static` on both real circuits. See
`SPAWN_FIX_RESULTS.md`.

## Track geometry (measured)

See table in `experiments.md` (synthetic max curvature 0.544 vs Spielberg
1.273 / Silverstone 0.938).

## Overfitting-timing probe — `eval_sac_ck{100000,300000,600000,1000000}.json`

SAC checkpoints from the 1M `--keep-checkpoints` run, evaluated zero-shot under
the standard protocol (4 tracks × 5 seeds, one-lap, centreline, correct loader).
Mean laps, finishes in brackets where non-zero:

| checkpoint | synthetic_1 | synthetic_2 | Spielberg | Silverstone |
|---|---|---|---|---|
| 100k | 0.42 | 0.33 | 0.04 | 0.02 |
| 300k | 0.69 | 0.73 (2/5) | 0.09 | 0.17 |
| 600k | 0.91 (2/5) | 0.90 (2/5) | 0.03 | 0.04 |
| 1M | **1.00 (5/5)** | 0.82 | 0.03 | 0.07 |

Unseen-synthetic rises monotonically; real circuits flat-low at every
checkpoint. One training seed; 5 spawn seeds per cell.

## October 2026 results (canonical seed-123 tracks)

All under `results/ppo_diagnosis/` unless stated; d1, 2 M steps, penalty 5. Training-log
metrics are from the stochastic policy on its own track; evaluations are deterministic.

### PPO sweep and fix — `runs_summary.csv` (last 200 k steps, training log)

| run | change from SB3 default | last-200k m | lap rate | max laps |
|---|---|---|---|---|
| V0_baseline | — | 36.4 | 0% | 0.37 |
| V1_ent | ent_coef 0.01 | 41.9 | 0% | |
| V2_kl | target_kl 0.03 | 48.0 | 0% | |
| V3_ent_kl | both | 54.3 | 0% | |
| V4_nsteps | n_steps 8192 | 68.4 | 0% | 0.47 |
| V5_logstd | log_std_init −1 | 37.9 | 0% | |
| V6_updates | n_epochs 4, batch 256 | 53.3 | 0% | |
| V7_lr | lr 1e-4 | 50.4 | 0% | |
| P40_penalty40 | n_steps 8192, penalty 40 | 71.7 | 0% (36% at 868 k) | 0.97 |
| **G999_gamma** | **n_steps 8192, γ 0.999** | **234.8** | **65%** | **2.00** |
| G999L99_full | + λ 0.99 | 156.8 | 59% | 1.49 |
| X1_v4_5M (killed 2.54 M) | V4 at 5 M budget | 66.9 | 0% | |
| R1–R4 (action repeat 10/25) | see `ppo-diagnosis.md` | 54–66 | 0% | |

### γ × n_steps at matched 0.8–1.0 M (`runs/*/monitor_0.monitor.csv`, `progress.csv`)

| | n_steps 2048 | n_steps 8192 |
|---|---|---|
| γ 0.99 | 39.6 m, 0%, σ 0.061 | 62.9 m, 0%, σ 0.261 |
| γ 0.999 | 44.9 m, 0%, σ 0.089 | **217.7 m, 65%**, σ 0.392 |

Replication, same window: seed 0 / 1 / 2 = **65% / 57% / 52%**.

### Held-out evaluation (`eval_sweep_1M/`, `eval_ck800k/`, `eval_ck2M/`, `eval_g999_curve/`)

test_track_0..4 × 5 episodes (= 25, ~3 distinct spawns per track) + Spielberg/Silverstone × 5.

| model @ checkpoint | synth finished | synth m | real finished | real m |
|---|---|---|---|---|
| best sweep variant V2 @ 1 M | 0/25 | 36.8 | 0/10 | 19.2 |
| V4 @ 800 k / 2 M | 0/25 / 0/25 | 21.1 / 31.6 | 0/10 / 0/10 | 10.0 / 33.4 |
| G999 @ 800 k | **10/25** | 98.8 | 0/10 | 43.8 |
| G999 @ 2 M | 0/25 | 67.1 | 0/10 | 24.0 |
| G999 mean of 1.2–2.0 M (5 ckpts) | **15/125 (12%)** | 83.5 | 0/50 | 40.2 |

Full G999 curve (400 k→2 M, finished/25): 0, 0, 10, 0, 6, 5, 4, 0, 0.

### Discounted return of trained policies (`diagnostics/basin_check.py`, 1 episode each, synthetic_track_0)

| policy | distance | disc. @0.99 | disc. @0.999 |
|---|---|---|---|
| PPO G999 | 327.8 m | 6.93 | 101.72 |
| PPO V4 (crashes) | 73.9 m | **7.72** | 52.16 |
| PPO V0 | 19.2 m | 4.99 | 12.75 |
| SAC sac_overfit | 272.5 m | 6.38 | 85.37 |

### Width ablation (`results/grid_s0/*_narrow*.csv`, `eval_width_oursac/`)

See `sim-to-real-width.md` for the table: every SAC goes from 29–58% on test to **0%** on
narrowA (same centrelines, 1.07 m half-width); the gap-follower still laps 36%.

### Desmond's seed-0 grid, verified (`results/grid_s0/`, `repro_desmond_grid/`)

See `grid-verification.md`: PPO 0/860; SAC held-out lap rate 2/22/31/38% (d1/5/20/100);
transfer ratio 0.13/0.47/0.57/0.76; real 0/460.

### Caveat on the September-2026 tables above

The September-2026 pilot tables on this page were measured on **seed-0** local
`synthetic_track_*` maps, not the canonical seed-123 pool. They remain valid as
statements about those maps, but they are not comparable track-for-track with the
October numbers (`incidents.md`).

## Files not central to the current story

`completion_*`, `learning_curve_cp40_ent*`, `..._bonus100_*`, `..._shape0.2_*`,
`..._warmup-*`, `train_compare.json`, `rl_before_after_PPO.json`,
`reward_scan.json` (superseded by `reward_scan2.json`) — earlier ablations,
kept for provenance. `_backup_prelog/` holds pre-logging copies.

## See also

- `experiments.md` — interpretation of these numbers
- `methodology.md` — the protocol that produced them
- `handoff.md` — how to regenerate / extend
- `ppo-diagnosis.md`, `grid-verification.md`, `sim-to-real-width.md` — interpretation of the October tables
