---
title: Reward function — history, incentive flaw, and the V2 redesign
type: concept
updated: 2025-09-20
sources:
  - team_repo/feasibility_spike/reward_experiment/REWARD_ANALYSIS.md (mixed)
  - team_repo/feasibility_spike/reward_experiment/results/reward_scan2.json (high)
  - team_repo/feasibility_spike/reward_experiment/new_reward_wrapper.py (high)
  - Desmond handover (team) — reward V1/interim/V2 history
---

# Reward function

The reward went through three forms. The change from V1 to V2 is a **design
correction of a proven incentive flaw**, not a performance patch — the
distinction matters for how it is presented (`decisions.md`).

## History (from Desmond's handover)

### V1 (Chris) — the original
```
reward = float(reward)            # = timestep, 0.01 per step = 1.0 per second alive
       + speed * 0.1              # ~10x distance term
       - abs(steering) * 0.5      # steering penalty
       - 5   on `terminated`      # applied on ANY terminal, including FINISHING the lap
```
Two problems: it **pays for time alive**, and it **punishes finishing** the
same as crashing (the −5 fires on the terminal step of a completed lap too).

### Interim (main branch, later reverted on Desmond's branch)
Added a low-speed −0.5 penalty and a 1.0 m/s speed floor, plus junk
(`from turtle import speed`, `DEFAULT_TIMESTEPS` bumped to 5M). Reverted on
`desmond/reward-and-episode-fix`. **main still contains this interim junk —
it is not merged.**

### V2 (frozen) — the redesign
```
reward = metres of centreline progress this step
       - 5.0  collision-gated penalty (only on a real crash)
TIME_COST = 0
```
- Progress measured by a **windowed nearest-segment projection** (13.6 µs,
  ~8.9× faster than the buggy `cartesian_to_frenet`).
- Action space normalized to [−1, 1].
- LiDAR downsampled 1080 → 108 beams (SAC replay buffer 8.9 GB → 0.9 GB;
  applied to both algorithms for parity).
- 3000-step truncation in training; one-lap termination in eval.
- **Verified:** on a completed lap, return == distance travelled
  (163.7774 on the synthetic test lap; 343.0324 on Spielberg).

## The incentive flaw — measured, not asserted

We ran the same rollout through both reward formulas
(`reward_scan2.py` → `results/reward_scan2.json`, crash penalty 40):

| behaviour | OLD (V1) return | NEW (V2) return |
|---|---|---|
| standstill | **100.00** (exact match to team's observed value) | 0.00 |
| crawl at 0.25 m/s | **263 (peak)** | ~0 |
| drive at 20 m/s | 189 | pays metres |
| wall crash (any speed) | — | **−22** |

The V1 **global optimum is the crawl**: it maximizes time-alive reward while
minimizing steering penalty and never triggering the terminal −5. So a policy
that learned to crawl looked like a *success* (episode reward rose smoothly)
while doing nothing useful. V2 **pays metres**, so a crash returns a negative
number — the reward now *reports* failure instead of paying for it.

## Why the reward change did NOT rescue PPO

We trained three matched PPO variants at 2M steps to isolate the reward's
effect (`results.md`):

- **PPO on V1 (old reward)** — `learning_curve_cp40_vn_old_2000000.json`
- **PPO on V2, penalty 5** — `learning_curve_cp5_vn_2000000.json`
- **PPO on V2, penalty 40** — `learning_curve_cp40_vn_2000000.json`

All three plateau at ~0.1 laps. **The reward choice does not change PPO's
plateau story.** Stability comes from the algorithm (SAC's replay buffer),
not the reward. So: the reward fix was necessary for the *right incentive*,
but it is not what makes a policy stable.

## Dense-shaping experiment (negative result)

`shaped_reward_wrapper.py` (`--shaping`) added a dense forward-progress
shaping term. It got **gamed** (forward-only penalty → the policy crashed
sideways instead), crashed more (7/10), and still oscillated. This confirms
the fix is the algorithm, not more reward engineering.

## The frozen-penalty decision (open)

Our pilots used crash penalty ~40; the team's canonical V2 uses **5.0**
(matches their baselines, slides, and verification CSVs). The PPO-p5 run
confirms the penalty choice does not change PPO's plateau, so the decision is
low-risk either way — but the team must freeze **one** value before the grid.
See `decisions.md` and `open-questions.md`.

## Caveats

- Single environment: with `n_envs = 1` the PPO batch is one correlated
  trajectory, which is part of why PPO oscillates. Documented in
  `REWARD_ANALYSIS.md`.
- SAC was only ever trained on V2. Whether the reward mattered for SAC is
  **unmeasured** — see `open-questions.md`.

## See also

- `decisions.md` — reward-as-design-decision, penalty freeze
- `experiments.md` — the PPO variant curves in context
- `methodology.md` — how the reward feeds the protocol
