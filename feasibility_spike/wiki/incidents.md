---
title: Incidents — the two real bugs and the loader pitfall
type: bug
updated: 2025-09-20
sources:
  - Group4_ProjectUpdate slide 8 "Two Bugs That Voided the Grid" (team)
  - team_repo/train.py, src/track_pool.py (high)
  - spike/f1tenth_gym_v1/.../base_classes.py (RaceCar.scan_simulator) (high)
  - team_repo/feasibility_spike/reward_experiment/REWARD_ANALYSIS.md (mixed)
---

# Incidents

Two of these are **genuine bugs** (they would have silently voided the
diversity grid). The third is a **debugging pitfall** we hit ourselves and
corrected. The reward and spawn changes are **not** in this file — they are
design decisions (`reward.md`, `decisions.md`), not bugs.

## Bug 1 — training updates scaled with diversity N

- **Symptom.** `train.py` created **one environment per track**, so `n_envs`
  equalled the diversity level. SB3 performs one gradient update per
  `vec_env.step()`, so under a fixed total step budget the number of updates
  scaled as `2M / n_envs`: **2,000,000 updates at 1 track but only 20,000 at
  100 tracks.** The high-diversity cells were starved of updates — the grid
  would have measured "fewer updates", not "more diversity".
- **Root cause.** Diversity was implemented as parallel envs instead of one
  env sampling from a pool.
- **Fix.** A **single environment** with a **stratified track-pool sampler**
  on `reset()`, so `n_envs` is always 1. Episodes truncate at **3,000 steps**
  so `reset()` fires and the pool rotates.
- **Verification.** Identical update counts at diversity 1 and 100.
- **Consequence for us.** `n_envs = 1` is now a pinned protocol constant —
  see `methodology.md`. Under a fixed budget, more envs = fewer updates, no
  extra data.

## Bug 2 — shared ray-caster across environments

- **Symptom.** `RaceCar.scan_simulator` is a **class attribute** in
  `f1tenth_gym`, shared **process-wide**. Every environment raycast against
  whichever map loaded **last**, so multi-track runs were effectively
  single-track runs — the diversity variable did nothing.
- **Root cause.** Found by reading the library source: the scan simulator is
  shared state, not per-instance.
- **Fix.** Removed by the single-environment fix (one env, so one map at a
  time, no cross-contamination).
- **Verification.** 20 of 20 tracks now visited, evenly dealt.

Together these two bugs are why the timeline slide says the grid would have
been "voided" — both silently collapse diversity to 1.

## Pitfall — VecNormalize loaded without flags → garbage observations

- **Symptom.** An early analysis claimed SAC "crashes at 2.7–10.7 m on a
  straight." Wrong.
- **Root cause.** `VecNormalize.load(path, venv)` (2-arg signature) does not
  by itself put the wrapper in eval mode. Without `training=False` and
  `norm_obs=True`, observations are un-normalized garbage → the policy
  receives off-distribution input → wrong actions → a fake crash pattern.
- **Fix.** Use the team's `load_vecnormalize` helper (or set the flags
  explicitly). The correct loader shows the real two-phase failure (creep
  ~0.6 m/s, then snap to full speed + hard-left-lock into the wall) at
  ~34 m / ~26 m — see `experiments.md`.
- **Lesson (wire-log the boundary).** When a policy behaves strangely, log
  the literal observation vector at the wrapper boundary **before**
  hypothesising about the policy. One `print(obs[:5])` would have caught the
  un-normalized values immediately.

## See also

- `methodology.md` — why `n_envs = 1` is pinned
- `experiments.md` — the corrected SAC crash mechanics
- `decisions.md` — reward/spawn framed as design, not bugs
