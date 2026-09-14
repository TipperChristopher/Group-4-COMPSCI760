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

## Experiment 3 — completion rate & lap time (`train_eval_completion.py`, 1M steps each)

Train PPO under each reward on synthetic_track_0 (1,000,000 steps), then evaluate on
synthetic_track_0 (trained) + _1/_2 (unseen) with a 15000-step cap. Completion = did
cumulative laps reach 1.0; lap_time = sim seconds to the first completed lap.

| reward | track | laps | mean m/s | completed | lap time |
|---|---|---|---|---|---|
| NEW | synthetic_track_0 (trained) | 0.187 | 10.99 | no | n/a |
| NEW | synthetic_track_1 (unseen)  | 0.035 | 10.12 | no | n/a |
| NEW | synthetic_track_2 (unseen)  | 0.057 | 10.57 | no | n/a |
| OLD | synthetic_track_0 (trained) | 0.187 |  9.23 | no | n/a |
| OLD | synthetic_track_1 (unseen)  | 0.002 |  6.40 | no | n/a |
| OLD | synthetic_track_2 (unseen)  | 0.001 |  6.49 | no | n/a |

**COMPLETION RATE: 0% for BOTH (0/3). LAP TIME: n/a (nothing finished a lap).**

Read-out:
- Both policies die at the **same first hard corner** on the trained track (both reach
  exactly 0.187 laps); NEW just reaches it faster (506 steps @ 11 m/s vs 745 @ 9 m/s).
  The reward fix makes the car *drive*; it has not yet learned to *corner*.
- NEW **generalises better on unseen tracks** (0.035-0.057 laps vs OLD's ~0.001) - the
  old policy barely moves off its training track.

**CAVEAT (important):** this harness omits observation normalisation (VecNormalize) that
the team's real `train.py` uses. Desmond's real pipeline reached **0.37 laps at 200k** -
better than this harness's 0.187 at 1M - so obs normalisation materially helps and these
completion numbers **understate** the fixed reward's real capability. This harness fairly
answers "does it drive fast?" (yes, 3.3x) but is under-powered for "does it complete laps?".
For a true completion rate / lap time, run the team's actual `train.py` (with VecNormalize)
at a real budget on Desmond's branch. (Also: eval labels crashes as "reset" because
DummyVecEnv auto-resets and clears the collision flag before it is read; outcomes here are
crashes.)

## Experiment 4 — crash-penalty sweep (does a bigger penalty teach cornering?)

**Motivation.** With the default penalty (5), reaching the first corner banks ~33 progress
(0.187 laps x ~179 m) while crashing costs only 5, so "drive fast, crash" nets +28 -
*crashing pays*. Hypothesis (Desmond Li): a bigger crash penalty forces the agent to slow
down and corner. **Test:** vary ONLY the crash penalty (a reward-design knob), everything
else fixed - 1,000,000 steps, PPO defaults, NEW reward, train on synthetic_track_0, eval on
0/1/2. **n = 1 seed per cell** (point estimates, not a trend claim).

| crash penalty | track_0 laps (mean m/s) | track_1 | track_2 | dominant behaviour | completion |
|---|---|---|---|---|---|
| **5** (default) | 0.187 (10.99) crash | 0.035 (10.1) | 0.057 (10.6) | drive fast -> crash at 1st corner | 0% |
| **40** | 0.001 (0.12) froze | 0.197 (0.42) crept | **0.844 (3.49) crash** | erratic, near-threshold | 0% |
| **120** | 0.001 (0.04) | 0.001 (0.04) | -0.001 (0.04) | **freeze at start** | 0% |
| **400** | 0.001 (0.01) | 0.001 (0.01) | -0.001 (0.01) | **freeze at start** | 0% |

**Read-out - tuning the penalty only SHIFTS the failure mode; it never completes a lap:**
- **Below** the progress-to-corner scale (5 << ~33): crashing is net-positive, so it
  crash-farms the straight and dies at the first corner.
- **Around** it (40 ~ 33): borderline/erratic - froze on one track, crept on another, but
  reached **0.844 laps** on a third (the best progress of ANY run here). Right at the tipping point.
- **Above** it (120, 400 >> 33): standing still (reward 0) beats driving (net-negative), so the
  policy **collapses to not moving** (mean 0.01-0.04 m/s).
- **Completion rate stays 0% at every penalty.**

**Mechanism.** NEW reward = progress - penalty(on crash). If the agent can bank P metres
before crashing, "drive & crash" = P - penalty vs "stand still" = 0. Penalty 5: P(~33) > 5 ->
drive. Penalty >= ~40: P < penalty on most tracks -> freezing is safer. The per-track spread in
reachable-P is exactly why penalty 40 is erratic (froze where P was small, nearly lapped where
P was large). **A flat penalty can't be right for every track/training-stage - that fragility
is the finding.**

**Conclusion (answers the hypothesis).** A bigger crash penalty does NOT teach cornering. It
trades one failure (crash-farming) for another (paralysis), with a narrow, fragile sweet spot
near the progress-per-corner scale where it occasionally near-laps but still completes 0/3.
The penalty fixes the *incentive*; it cannot supply the *skill*.

**Caveats:** n=1 seed per cell (the sweet spot and the 0.844 are single-seed, not firm -
need >=3 seeds to claim). Harness omits VecNormalize, so absolute completion is understated.
`laps` can read slightly negative near the start (TrackProgress sign artifact; harmless).

**Recommendation:** keep the crash penalty **moderate** - clearly above the crash-farming
regime but well below paralysis (order of one straight's progress, ~20-40 here) - and **freeze
one value for both PPO and SAC**. Do NOT rely on the penalty for lap completion; the real levers
are more training + exploration, observation normalisation (VecNormalize), curriculum, and the
diversity sweep itself.

## Reproduce

```bash
cd feasibility_spike/reward_experiment
python make_synth_tracks.py --n 3 --seed 0     # writes synthetic_track_0..2 into the installed maps dir
python reward_scan.py                          # Experiment 1
python train_compare.py --steps 100000         # Experiment 2 (raise --steps for a stronger result)
```

`old_reward_wrapper.py` / `new_reward_wrapper.py` are verbatim copies of `sb3_wrapper.py` from
`main` and from `desmond/reward-and-episode-fix` respectively (reference copies; do not edit).
