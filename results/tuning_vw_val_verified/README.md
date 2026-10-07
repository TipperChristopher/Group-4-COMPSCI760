# Tuning round on vw_val: SAC and PPO settings for the final grid

**Purpose.** Choose the training settings for the final grid. 16 runs: 8 settings
× 2 seeds (99, 98), each trained for 2,000,000 steps on the first 20 varied-width
tracks (`vw_synthetic_track_0..19`, half-widths 0.63–1.44 m) with crash penalty 40.
Every checkpoint was scored on **vw_val only** (10 varied-width tracks never
trained on). **No test set was touched** (test, vw_test, real, narrowA and narrowB
were not run).

Code: branch `desmond/heldout-tracks`, commit **`6e5b60e`** (`6e5b60ef27612679fcff74bd5e9ead7301eef289`),
no uncommitted files in any run. Reward code `sb3_wrapper.py` sha256
`9b73ea777f09d17306f0d1d5486edb024ad9c028629d927a38cdbb3c338a8a34` (the same file as the r40 runs). Ran unattended
with `launch_queue.ps1`, 7 Oct 2026, 10:47–18:16.

## The 8 settings

| setting | algorithm | flags (on top of `--diversity 20 --track-prefix vw_synthetic_track_`) | what it tests |
|---|---|---|---|
| S0 | SAC | `(none: SB3 defaults, gamma 0.99)` | baseline |
| S1 | SAC | `--gamma 0.999` | longer value horizon (10 s instead of 1 s) |
| S2 | SAC | `--gamma 0.999 --time-cost 0.01` | S1 + a per-step time cost, so crawling is no longer free |
| S3 | SAC | `--gamma 0.999 --buffer-size 2000000` | S1 + a replay buffer holding the whole run (no overwriting) |
| P0 | PPO | `(none: SB3 defaults, gamma 0.99, n_steps 2048)` | baseline |
| P1 | PPO | `--gamma 0.999 --n-steps 8192` | Aolin's PPO fix (longer horizon, bigger rollouts) |
| P2 | PPO | `--gamma 0.999 --n-steps 8192 --time-cost 0.01` | P1 + the per-step time cost |
| P3 | PPO | `--gamma 0.999 --n-steps 8192 --target-kl 0.03 --ent-coef 0.01` | P1 + entropy bonus and a KL cap (stability) |

Everything not listed is the Stable-Baselines3 2.9.0 default. SAC: buffer 1M,
batch 256, ent_coef auto, net [256, 256]. PPO: n_epochs 10, batch 64, clip 0.2,
gae_lambda 0.95, net pi/vf [64, 64]. One environment, VecNormalize on
observations only, 3,000-step training episodes, 108 LiDAR beams + 5 state values.
Resolved values are in each `run_config.json` (`hyperparameters`).

## Verification before and after the run

- **The per-step time cost reaches the actual reward.** Before launch: driving
  train.py's own environment at zero speed gave exactly −0.0100 per step with
  `--time-cost 0.01` and 0.0000 without, and a collision step paid −40.
- **The same holds in the real training logs.** Measured from every episode as
  (progress − return) / steps: **0.0100–0.0100 per step in the 4
  S2/P2 runs and 0.0000 in the other 12**. Crash penalty measured on
  crash episodes: **40.00–40.00** in all 16 runs.
- Each run's `run_config.json` records exactly its setting's overrides (checked
  for all 16), and runs without flags resolve to the same hyperparameters as
  before the flags were added.
- Queue: all 16 runs exit 0 with status completed; all 80 evaluations have 50
  rows (10 tracks × 5 episodes).

## Selection rule (fixed before the run)

- **Score:** mean fractional laps on vw_val. That is 10 tracks × 5 episodes,
  deterministic, centreline spawn, one-lap termination, 15,000-step cap.
- **Best checkpoint per run**, from 400k, 800k, 1.2M, 1.6M and final: the
  highest score; on a tie, the faster mean lap time; then the earlier checkpoint.
- **Setting score:** the mean of its two seeds' best scores. The seed gap is |s99 − s98|.
- **Winner:** the highest setting score among the candidates, S1–S3 for SAC and
  P1–P3 for PPO. S0 and P0 are baselines and cannot win.
- **Flags:**
  - near-tie: within 0.03 of the winner (none occurred);
  - declines late: either seed's final checkpoint more than 20% below its best;
  - CRAWLING: under 2.5 m/s over a run's last 200 training episodes.

## Results

### Per setting (best-checkpoint score per seed, seed average)

| setting | role | seed 99 | seed 98 | **average** | seed gap | vs baseline | verdict | flags |
|---|---|---|---|---|---|---|---|---|
| S0 | baseline | 0.453 @ 1600k | 0.566 @ final | **0.509** | 0.114 | — |  | CRAWLING ×1 |
| S1 | candidate | 0.277 @ 400k | 0.214 @ 400k | **0.246** | 0.063 | -0.264 |  | declines late, CRAWLING ×2 |
| S2 | candidate | 0.483 @ 1200k | 0.396 @ final | **0.439** | 0.087 | -0.070 | WINNER | declines late |
| S3 | candidate | 0.277 @ 400k | 0.214 @ 400k | **0.246** | 0.063 | -0.264 |  | declines late, CRAWLING ×1 |
| P0 | baseline | 0.175 @ 1600k | 0.109 @ 400k | **0.142** | 0.066 | — |  | declines late |
| P1 | candidate | 0.028 @ final | 0.126 @ final | **0.077** | 0.098 | -0.064 |  | CRAWLING ×1 |
| P2 | candidate | 0.071 @ 1200k | 0.491 @ 1600k | **0.281** | 0.420 | +0.139 | WINNER | declines late |
| P3 | candidate | 0.044 @ 400k | 0.034 @ 400k | **0.039** | 0.010 | -0.103 |  | declines late, CRAWLING ×2 |

**Winners by the rule:** SAC **S2** (0.439) and PPO **P2** (0.281).
**But neither result is clean:**
- **S2 scores below its own baseline:** S0 averages 0.509, against S2's 0.439.
  No SAC candidate beat plain SAC.
- **P2's lead over P0 (+0.139) rests on one seed:** seed 98 scored 0.491 and
  seed 99 only 0.071, a seed gap of 0.42.

### Score at every checkpoint (fractional laps)

| setting | seed | 400k | 800k | 1.2M | 1.6M | final |
|---|---|---|---|---|---|---|
| S0 | 99 | 0.276 | 0.283 | 0.350 | 0.453 | 0.366 |
| S0 | 98 | 0.254 | 0.353 | 0.422 | 0.449 | 0.566 |
| S1 | 99 | 0.277 | 0.017 | 0.010 | 0.009 | 0.009 |
| S1 | 98 | 0.214 | 0.008 | 0.008 | 0.026 | 0.044 |
| S2 | 99 | 0.222 | 0.352 | 0.483 | 0.311 | 0.269 |
| S2 | 98 | 0.205 | 0.308 | 0.362 | 0.351 | 0.396 |
| S3 | 99 | 0.277 | 0.017 | 0.009 | 0.240 | 0.238 |
| S3 | 98 | 0.214 | 0.008 | 0.007 | 0.008 | 0.008 |
| P0 | 99 | 0.092 | 0.076 | 0.138 | 0.175 | 0.099 |
| P0 | 98 | 0.109 | 0.106 | 0.089 | 0.060 | 0.082 |
| P1 | 99 | 0.010 | 0.003 | 0.000 | 0.022 | 0.028 |
| P1 | 98 | 0.079 | 0.122 | 0.065 | 0.084 | 0.126 |
| P2 | 99 | 0.055 | 0.049 | 0.071 | 0.063 | 0.069 |
| P2 | 98 | 0.111 | 0.278 | 0.403 | 0.491 | 0.376 |
| P3 | 99 | 0.044 | 0.009 | 0.014 | 0.039 | 0.039 |
| P3 | 98 | 0.034 | 0.003 | 0.003 | 0.001 | 0.000 |

### Training, last 200 episodes

| run | speed (m/s) | lap % | crash % | hit 3,000 cap % | mean length | |
|---|---|---|---|---|---|---|
| tune_S0_s99 | 2.43 | 0 | 16 | 84 | 2790 | **CRAWLING** |
| tune_S0_s98 | 3.60 | 19 | 39 | 61 | 2581 |  |
| tune_S1_s99 | 2.29 | 0 | 100 | 0 | 73 | **CRAWLING** |
| tune_S1_s98 | 0.38 | 0 | 28 | 72 | 2309 | **CRAWLING** |
| tune_S2_s99 | 7.72 | 8 | 86 | 12 | 1190 |  |
| tune_S2_s98 | 7.34 | 2 | 86 | 14 | 1186 |  |
| tune_S3_s99 | 6.09 | 0 | 74 | 26 | 1178 |  |
| tune_S3_s98 | 2.13 | 0 | 100 | 0 | 63 | **CRAWLING** |
| tune_P0_s99 | 5.48 | 0 | 100 | 0 | 357 |  |
| tune_P0_s98 | 3.94 | 0 | 56 | 44 | 1518 |  |
| tune_P1_s99 | 2.01 | 0 | 40 | 60 | 1942 | **CRAWLING** |
| tune_P1_s98 | 4.11 | 0 | 64 | 36 | 1486 |  |
| tune_P2_s99 | 4.30 | 0 | 64 | 36 | 1202 |  |
| tune_P2_s98 | 5.65 | 18 | 62 | 38 | 1890 |  |
| tune_P3_s99 | 0.72 | 0 | 18 | 82 | 2605 | **CRAWLING** |
| tune_P3_s98 | 0.26 | 0 | 13 | 87 | 2829 | **CRAWLING** |

## Notes

1. **gamma 0.999 alone breaks SAC.** S1 scores 0.21–0.28 at 400k in both
   seeds, then collapses to at most 0.04 from 800k on.
   - By the end, seed 99 drives into the wall within about 73 steps, and seed 98
     crawls at 0.38 m/s.
   - Its entropy coefficient and critic loss blow up between 0.6M and 1.2M (Fig 4).
2. **The time cost prevents that collapse.**
   - S2 trains at 7.3–7.7 m/s (S1: 0.4–2.3), never crawls, and scores 0.44 against S1's 0.25.
   - Its completed laps are mostly fast, 19–23 s (one checkpoint averages 48 s), against S0's 29–135 s. But it crashes
     in 86% of training episodes and completes at most 4% of vw_val laps.
3. **S3 is identical to S1 until the 1M buffer fills.** With the same seed, the
   two produce the same vw_val scores at 400k, 800k and 1.2M; they diverge only
   after 1M steps.
   - After that, seed 99 partly recovered (0.24 at 1.6M and 2M, 6.1 m/s); seed 98
     stayed collapsed.
   - So the bigger buffer neither prevents the gamma-0.999 collapse nor reliably reverses it.
4. **For PPO, the time cost is the only change that produced laps, on one seed only.**
   - P2 seed 98 reached 0.49, with 18–20% of laps completed from 1.2M on, in 23.5–25.8 s.
   - P2 seed 99 reached 0.07.
   - P1 (Aolin's fix without the time cost) did not beat P0: 0.077 against 0.142.
5. **P3 stabilised PPO's optimiser but not its driving.**
   - approx_kl stays around 0.007 and clip_fraction around 0.07.
   - But the entropy bonus pushes the policy std up from 1.0 to 1.5–1.9, and both
     seeds crawl (0.26–0.72 m/s). Its best score is 0.04.
6. **P0 and P1 show the known PPO failure.** By 2M, std is down to 0.02–0.12,
   approx_kl is 0.15–0.56 and clip_fraction 0.19–0.51. P0 seed 98's explained
   variance falls to about −3.5.
7. **Crawling in training doesn't always mean a poor evaluation score.**
   - S0 seed 99 is flagged CRAWLING (2.43 m/s) yet scores 0.45 on vw_val, so the
     deterministic policy behaves better than the noisy training policy.
   - Also crawling: both S1 runs, S3 seed 98, P1 seed 99 and both P3 runs.
8. **Six of the eight settings decline late:** S1, S2, S3, P0, P2 and P3. For
   example, S2 seed 99 falls from 0.483 at 1.2M to 0.269 at 2M. Choosing the
   checkpoint on vw_val matters; final-checkpoint scores alone would rank the
   settings differently.

## Caveats

- **2 seeds per setting.** Seed gaps reach 0.42 (P2), larger than most differences between settings.
- **10 validation tracks** (50 episodes, of which only 3 spawn poses per track are distinct).
- **Diversity 20 only.** These settings may rank differently at d1 or d100.
- **5 checkpoints per run,** spaced 400k apart.
- Fractional laps rewards partial progress, so a slow but safe policy can outscore a fast one that crashes.

## Files

- `1_summary/`
  - `summary.csv`: the queue's own summary
  - `per_run_per_checkpoint.csv`: fractional laps, lap %, progress, crash %,
    timeout % and lap time for every run × checkpoint
  - `per_setting.csv`: the setting table above
  - `training_last200.csv`
- `2_figures/`
  - `fig1`: score vs checkpoint
  - `fig2`: setting scores with both seeds
  - `fig3`: training curves
  - `fig4`: SAC ent_coef and critic loss; PPO std, approx_kl, clip_fraction and explained variance
- `3_evaluation_episodes/`: the 80 raw vw_val CSVs, one row per episode.
  `laps` is the fractional lap, `completed_lap` is authoritative, and the lap time
  is `length × 0.01` s.
- `4_training_logs/<run>/`
  - `per_update_log.csv`: SB3 `progress.csv`
  - `per_episode_log.csv`: Monitor log; **skip the first line, a JSON comment**
  - `run_config.json`
  - `console_log.txt`
- Training rates are stochastic and capped at 3,000 steps, so they are not evaluation rates.

Reproduce: check out `6e5b60e`, then run
`.\launch_queue.ps1 -Jobs tuning_jobs.txt`, and
`.\launch_queue.ps1 -Jobs tuning_jobs.txt -Summary` once it finishes.
