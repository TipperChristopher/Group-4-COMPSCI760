---
title: Experiments — PPO vs SAC, generalization, and the coverage finding
type: concept
updated: 2026-10-06
sources:
  - team_repo/feasibility_spike/reward_experiment/results/*.json (high)
  - team_repo/feasibility_spike/reward_experiment/RESULTS.md, REWARD_ANALYSIS.md (mixed)
  - spike/f1tenth_gym_v1/.../random_trackgen.py (high)
---

# Experiments

All numbers with their source files are tabulated in `results.md`; this page
is the narrative and interpretation. **Everything here is single-seed pilot
data unless stated otherwise** — point estimates, not ranked results.

## Contents

- [Stage 1 — capability check (one track)](#stage-1--capability-check-one-track)
- [Learning curves](#learning-curves)
- [Unseen-track evaluation](#unseen-track-evaluation)
- [Why SAC fails on real circuits](#why-sac-fails-on-real-circuits)
- [The nar7 coverage finding](#the-nar7-coverage-finding)
- [Track geometry](#track-geometry)
- [Overfitting-timing probe (answered: no)](#overfitting-timing-probe-answered-no)
- [Later work (2026-10) and corrections to this page](#later-work-2026-10-and-corrections-to-this-page)

## Stage 1 — capability check (one track)

The pilots answer a prerequisite question before the diversity grid: **can
each algorithm learn the task at all on one track, under the frozen
protocol?** Same reward, budget, normalization, spawn for both.

- **SAC** completes laps and keeps improving.
- **PPO** peaks early then forgets — an on-policy instability under this
  protocol, not a setup bias (env setup verified correct, below).

This is a capability result, not the generalization comparison. The grid
(1/5/20/100) is the actual experiment; see `open-questions.md`.

## Learning curves

`learning_curve.py`, VecNormalize, penalty 40, with `ep_rew_mean` logging.
Reproducible bit-identical across re-runs (seed 0).

- **PPO + VN (1M)** — best **0.344 laps at 300k**, then degrades to **0.098
  at 1M**; episode reward oscillates from **+37 to −19** (std 14.3). Source:
  `learning_curve_cp40_vn_1000000.json`.
- **SAC + VN (1M)** — completes 2 laps at 400k / 600k / 700k; reward climbs
  smoothly to **196.6**; lap time **40.3 → 24.7 s**. Source:
  `learning_curve_cp40_sac_vn_1000000.json`.
- **PPO env verified correct** (`_verify_ppo.py`): obs (113,), act (2,) in
  [−1,1], `n_envs=1`, `norm_obs=True`, `norm_reward=False`, standstill
  reward 0, penalty 40, SB3 PPO defaults. The oscillation is real, not an
  artifact.

At 2M steps, three matched PPO variants (old reward, V2/p5, V2/p40) **all
plateau at ~0.1 laps** — the reward choice does not rescue PPO (`reward.md`).
SAC-2M reaches lap time **22.7 s at 1.2M**; the run was paused at 12/20
segments (1.2M) by decision *(later superseded by `sac_overfit`, which finished 2 M; see the update below)* — 1M→2M only optimizes speed, not the
conclusions (`decisions.md`). SAC results use this ~1.2M checkpoint.

## Unseen-track evaluation

5 tracks × 5 seeds, one-lap termination, centreline spawn. Full numbers in
`results.md`. Summary:

- **SAC (1.2M)** completes **unseen synthetic** tracks (track_2: 1.0 laps on
  all 5 seeds, 22.4 s) but **crashes on real circuits** (Spielberg 0.106,
  Silverstone 0.057, crash 5/5). `eval_sac_1m.json`.
- **PPO (2M)** is at roughly **random level on real circuits** (Spielberg
  0.013 vs random ~0.011), crash 5/5 everywhere. `eval_ppo_vn_2m.json`.
- **Gap-follower** completes Spielberg (1.0 lap, ~67 s) with no training.
  `eval_gap_protocol.json`.

Interpretation: RL wins **in-distribution** (synthetic), the reactive
baseline wins **out-of-distribution** (real circuits). That is the
generalization gap the study targets.

## Why SAC fails on real circuits

Verified, and it corrects an earlier mistake of ours (see below).

- **Crash location.** SAC crashes at **~34 m on Spielberg / ~26 m on
  Silverstone** — early, on the opening stretch, **not** at the sharp
  corners it never reaches.
- **Two-phase failure.** It creeps at ~0.6 m/s, then snaps to full-speed +
  hard-left-lock (−0.42 rad) and accelerates **into** the wall (commanded
  14 → 20 m/s as the minimum scan drops 0.60 → 0.26 m).
- **Input distribution shift at spawn.** Real spawns have walls at
  **1.05–1.09 m** vs **1.49–1.53 m** on synthetic (~45 % closer,
  off-distribution); far-field returns ~30 m vs 10–19 m synthetic. First
  action: synthetic 10–17 m/s (confident) vs real 1–2 m/s (creep).

**Corrected error.** An earlier "crash at 2.7–10.7 m on a straight" claim was
a **broken-loader artifact**: `VecNormalize.load(path, venv)` without setting
`training=False` / `norm_obs=True` produced garbage observations → wrong
actions. The correct loader (team's `load_vecnormalize`) gives the two-phase
failure above. See `incidents.md` (loader pitfall).

## The nar7 coverage finding

Question the user posed: is there a synthetic track with **real-like spawn
clearance** to test the input-shift hypothesis in a controlled setting?

- Existing synthetic tracks all have clearance ~1.48–1.53 m — none close to
  real. So we generated narrow tracks additively via `_gen_narrow.py` (a
  patched temp copy of the team's `random_trackgen.py`, `WIDTH` swept). At
  **WIDTH ≈ 7.3** the clearance is **1.04–1.08 m**, matching the real
  circuits (Spielberg 1.09, Silverstone 1.05).
- **First-action probe (SAC 1.2M):**
  - `synthetic_track_0` (1.51 m): drives, 15–17 m/s.
  - `synthetic_nar7_0` (1.06 m): **creeps, 1 m/s** — reproduces the
    real-circuit behaviour.
  - `synthetic_nar7_1/2` (1.05 / 1.08 m): **drive, 15 m/s** — same clearance,
    opposite behaviour.
  - Spielberg / Silverstone: creep.
- **So clearance is necessary-looking but not sufficient.** The beam profile
  distinguishes them: creeping tracks have **forward ~8.1–8.9 m** with side
  walls ~1.2 m; the driving nar7_1/2 have **forward ~5.7 m**. The trigger is
  a **region of scan space** (close sides + moderately-far forward), not one
  number.
- **5-seed eval** (`eval_sac_nar7.json`): nar7_0 **0.118 laps, crash 5/5**;
  nar7_1 **0.134 laps, crash 5/5** — the whole real-circuit failure (creep
  then early crash at ~0.1 laps) is **reproduced on synthetic geometry**.

**Conclusion:** the failure is a **training-coverage problem** — the policy
memorized its training distribution and collapses on scan patterns it never
saw. It is not something magic about real circuits. This gives a
controllable testbed (`synthetic_nar7_0`) and a concrete fix direction:
add narrow-width / varied-geometry tracks to the training pool. This proves
**coverage/memorization**, and is distinct from the (still open)
overfitting-over-time claim — see below and `open-questions.md`.

## Track geometry

Measured curvature over the centreline of each track:

| track | length | mean curv | p90 | max |
|---|---|---|---|---|
| synthetic_0 | 189 m | 0.102 | 0.441 | 0.544 |
| synthetic_1 | 152 m | 0.108 | 0.331 | 0.544 |
| synthetic_2 | 177 m | 0.106 | 0.350 | 0.544 |
| Spielberg | 343 m | 0.050 | 0.153 | **1.273** |
| Silverstone | 458 m | 0.062 | 0.190 | **0.938** |

Real circuits are ~2× longer with more straights (lower mean curvature) but
**much sharper extreme corners**. The synthetic generator **caps corner
sharpness at ~0.544**, traced to `TRACK_TURN_RATE = 0.31` in
`random_trackgen.py` (tunable). This is a **second** ceiling for
synthetic→real transfer, distinct from the spawn input-shift: it only bites
*after* the car survives the spawn, so the spawn shift is the binding
constraint first.

## Overfitting-timing probe (answered: no)

Hypothesis (user's): 1M steps on one track may **overfit** — generalization
peaking early then degrading, while in-distribution lap time keeps improving.
The pilots never kept earlier checkpoints, so we added `--keep-checkpoints` to
`learning_curve.py`, kept a model every 100k, and evaluated 100k / 300k / 600k /
1M zero-shot through the standard protocol and the **correct** loader
(`eval_sac_ck*.json`).

| checkpoint | synth_1 | synth_2 | Spielberg | Silverstone |
|---|---|---|---|---|
| 100k | 0.42 | 0.33 | 0.04 | 0.02 |
| 300k | 0.69 | 0.73 (2/5 fin) | 0.09 | 0.17 |
| 600k | 0.91 (2/5 fin) | 0.90 (2/5 fin) | 0.03 | 0.04 |
| 1M | **1.00 (5/5 fin)** | 0.82 | 0.03 | 0.07 |

**Verdict: no overfitting over time.** Unseen-synthetic generalization rises
**monotonically** with training (0.42 → 1.00; finishes 0 → 5/5), and real-circuit
performance is flat-low (0.02–0.17) at **every** checkpoint including the
earliest. "Train less" is therefore not a fix, and the real-circuit gap exists
from the first checkpoint — consistent with, and complementary to, the nar7
**coverage** conclusion. Caveat: one training seed, 5 spawn seeds per cell.

A 2M `--keep-checkpoints` extension (`saved_models/sac_overfit`) is running to
extend the curve past 1M; the verdict is not expected to change.

*Update 2026-10-06:* `sac_overfit` finished (2 M, Oct 4). Its in-training curve showed
crashes at 1.4 M / 1.7 M / 1.8 M, but each point is **one episode at one spawn**. Measured
with 30 spawns, the 2 M policy completes **22/30** on its own track and the 1.2 M checkpoint
**30/30**. A ~27% per-episode failure rate predicts ~5 crashes in 20 single-episode
checkpoints, and 5 were observed. So the dips are sampling, not a late-training collapse.
On real circuits it is 0/20 at both checkpoints.

## Later work (2026-10) and corrections to this page

The October work is on its own pages: `ppo-diagnosis.md`, `grid-verification.md`,
`sim-to-real-width.md`, `final-run-plan.md`. Corrections to the sections above:

### Correction: track geometry table uses non-canonical maps
The geometry table above (synthetic_0 = 189 m) was measured on locally generated
**seed-0** tracks. The canonical training pool is **seed 123** (synthetic_track_0 = 164.1 m),
and 0/20 match by name (`incidents.md`). For canonical geometry use
`results/track_geometry/summary.json` (`sim-to-real-width.md`).

### Correction: "corner sharpness is a second ceiling" — refuted
The max-curvature comparison (0.544 vs 1.273) compares single extreme points. Over the
whole distribution, real circuits are sharper than anything in training at only **0.95%** of
points, and are straighter at p90 (0.195 vs 0.408). Width is out of distribution at 100% of points
(`sim-to-real-width.md`). The `TRACK_TURN_RATE` fix is withdrawn.

### Correction: nar7 "clearance necessary but not sufficient"
The nar7 tracks varied width *and* centreline together and were scored on 5 seeds
(3 distinct spawns). The controlled narrowA/B ablation (same centrelines, only walls moved,
40 tracks) shows width is **sufficient** to reproduce the real-circuit failure. The
coverage conclusion stands, and the uncovered variable is now named: width.

### Correction: "PPO peaks early then forgets — an on-policy instability"
Accurate as a symptom. The cause is γ 0.99 at 100 Hz (the objective prefers crashing)
together with 2048-step rollouts (exploration collapse). Fixed at d1 by γ 0.999 + n_steps 8192
(`ppo-diagnosis.md`).

## See also

- `results.md` — every number with its JSON source
- `reward.md` — the PPO-variant curves and incentive analysis
- `decisions.md` — budget, penalty, VecNormalize, best-checkpoint
- `open-questions.md` — the grid, CIs, and the overfitting probe
- `ppo-diagnosis.md`, `sim-to-real-width.md`, `grid-verification.md` — the October work that corrects parts of this page
- `open-questions.md` — status of each open item
