# Reward ablation — results

Validates the reward-function fix (branch `desmond/reward-and-episode-fix`) against the
old reward on `main`. Two experiments, both on the real simulator, `synthetic_track_0`.
Env: Python 3.11, gymnasium 0.29.1, stable-baselines3 2.9.0, torch 2.13 (CPU).

## Experiment 1 — reward-shape scan (`reward_scan.py`)

Drive straight at a constant speed; on the *same* rollout, accumulate all three reward
definitions. Old reward paid `0.01/step` (survival) + `0.1*speed` (= distance×10); new
reward pays centreline progress − a crash penalty (Desmond's real `TrackProgress`).

| speed (m/s) | steps | outcome | dist (m) | OLD_orig | OLD_main | NEW |
|---|---|---|---|---|---|---|
| 0.00 | 3000 | timeout | 0.00 | 30.0 | −1470 | **0.00** |
| 0.25 | 3000 | timeout | 7.45 | 104.5 | −1395 | 7.45 |
| 0.50 | 3000 | timeout | 14.89 | 179.0 | −1321 | 14.89 |
| 1.00 | 1900 | crash | 18.0 | **206.8** | 28.3 | 13.0 |
| 2.00 | 964 | crash | 18.1 | 198.1 | **190.1** | 13.1 |
| 5.00 | 402 | crash | 17.8 | 189.4 | 183.4 | 12.8 |
| 20.0 | 212 | crash | 18.2 | 190.5 | 185.0 | 13.2 |

**Read-out:**
- **OLD reward is maximised by going slow** — `OLD_orig` peaks at ~1 m/s (the crawl), and fast
  speeds crash early and score lower. Standing still already banks 30 (survival). This is the bug.
- **NEW reward makes standing still the worst possible (exactly 0.00)** and *every* motion beats it —
  the "best → worst" flip Desmond reported.
- **A straight-line scan cannot rank *speeds* under the new reward** (all speeds crash into the first
  wall at ~18 m, so they tie ~13). Ranking speed needs the car to *follow* the track → Experiment 2.
  (This is the "shallow basin on a straight line" caveat in Desmond's commit.)

## Experiment 2 — short PPO, old vs new reward (`train_compare.py`, 100k steps each)

Train PPO under each reward, then measure the learned policy's behaviour.

| reward | learned mean speed | max speed |
|---|---|---|
| OLD | **2.76 m/s** (crawls) | 4.17 |
| NEW | **9.13 m/s** (drives) | 14.90 |

**Read-out:** the fix changes learned behaviour from crawling to driving — **3.3× faster mean speed**.

**Honest caveats (not failures):**
- At only 100k steps the policy learns *speed*, not yet *steering along the track*, so centreline
  progress / laps are ≈0 for both — real lap-following needs the full training budget (1–5M).
- The new-reward policy tends to drive fast but circle/oscillate at this tiny budget (consistent with
  the "circling exploit" residual Desmond flagged). Watch for it when evaluating longer runs on wide
  real circuits.

## Reproduce

```bash
cd feasibility_spike/reward_experiment
python make_synth_tracks.py --n 3 --seed 0     # writes synthetic_track_0..2 into the installed maps dir
python reward_scan.py                          # Experiment 1
python train_compare.py --steps 100000         # Experiment 2 (raise --steps for a stronger result)
```

`old_reward_wrapper.py` / `new_reward_wrapper.py` are verbatim copies of `sb3_wrapper.py` from
`main` and from `desmond/reward-and-episode-fix` respectively (reference copies; do not edit).
