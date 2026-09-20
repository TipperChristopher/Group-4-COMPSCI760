---
title: Results — numbers with sources
type: reference
updated: 2025-09-20
sources:
  - team_repo/feasibility_spike/reward_experiment/results/*.json (high)
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
| SAC+VN 2M (partial) | `learning_curve_cp40_sac_vn_2000000.json` | lap 22.7 s @1.2M; paused 12/20 |

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

## Files not central to the current story

`completion_*`, `learning_curve_cp40_ent*`, `..._bonus100_*`, `..._shape0.2_*`,
`..._warmup-*`, `train_compare.json`, `rl_before_after_PPO.json`,
`reward_scan.json` (superseded by `reward_scan2.json`) — earlier ablations,
kept for provenance. `_backup_prelog/` holds pre-logging copies.

## See also

- `experiments.md` — interpretation of these numbers
- `methodology.md` — the protocol that produced them
- `handoff.md` — how to regenerate / extend
