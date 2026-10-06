---
title: Evaluation methodology and protocol
type: reference
updated: 2026-10-06
sources:
  - team_repo/feasibility_spike/reward_experiment/METHODOLOGY_PLAN.md (mixed)
  - team_repo/feasibility_spike/reward_experiment/SPAWN_FIX_RESULTS.md (mixed)
  - spike/f1tenth_gym_v1/.../reset/__init__.py, f110_env.py, reset/utils.py, track/track.py (high)
  - team_repo/evaluate.py, sb3_wrapper.py, src/track_pool.py (high)
  - Desmond handover (team) — canonical protocol
---

# Evaluation methodology and protocol

The protocol is frozen and identical for PPO, SAC, all four diversity
levels, and both baselines. Divergence between train and eval would confound
the diversity comparison, so the spawn, termination, and metric are pinned.

## Train / eval split

*Current splits (2026-10, `tracks/manifest.json`, all checksummed):* `train`
`synthetic_track_0..99` (seed 123), `val` `val_track_0..9` (checkpoint selection),
`test` `test_track_0..19` (reporting), `real` (23 circuits, install with
`tracks/install_real_tracks.py`), `narrowA`/`narrowB` (test centrelines at 1.07 m / 0.75 m
half-width). `evaluate.py --track-set <name>` resolves them. **Track identity must match
the manifest** — see `incidents.md` (seed 0 vs 123).

- **Training pool:** N distinct tracks (N = 1 / 5 / 20 / 100), sampled by a
  stratified track-pool sampler on `reset()` (team's `src/track_pool.py`),
  with a single environment (`n_envs = 1`) — see `incidents.md` for why one
  env is mandatory under a fixed budget.
- **Evaluation set:** held-out synthetic tracks (never in the pool) plus the
  real circuits Spielberg and Silverstone.
- **Zero-shot:** no fine-tuning between train and eval.

## Spawn protocol (pinned)

- **Spawn mode `cl_grid_static`** (centreline grid, static) for both train
  and eval. The gym default is `rl_grid_static`, which for the real circuits
  spawns the car **0.79–0.81 m off the centreline** — a train/eval mismatch
  and an unfair, harder start. With `cl_grid_static` the spawn offset is
  **0.00 m** on Spielberg and Silverstone. Verified at env level; see
  `SPAWN_FIX_RESULTS.md` and `decisions.md`.
- **Reproducible per seed.** `f110_env.py:366-367` calls
  `np.random.seed(seed)`, so a given seed maps to a fixed spawn. Verified
  with `seed_probe.py`: `DummyVecEnv.seed(s)` gives the same spawn as
  `env.reset(seed=s)`, and VecNormalize does not break this.
- **Single-agent note.** `reset/utils.py sample_around_waypoint` applies a
  lateral offset only if `n_agents > 1`, so it is a no-op on our single-agent
  pose.
- **5 evaluation seeds.** Over the ~1 m start line these ~5 seeds resolve to
  **3–4 distinct spawn points** (weighted), not 5 fully independent ones —
  documented so "5 seeds" is not over-claimed.
  *Measured 2026-10-06:* exactly **3** distinct spawns per track in the seed-0 grid
  (seeds 0≡1, 2≡3; Desmond's own `spawn_check` agrees), and 30 seeds give only ~9.
  A marginal policy is badly under-sampled: our SAC d1 scored 0/5 on 5 seeds but
  3/30 on 30. The proposed final protocol uses 10 seeds per track (`final-run-plan.md`).

## Episode termination (pinned)

- **Training:** episodes truncate at a fixed step cap so `reset()` fires and
  the pool rotates (team's V2 uses a 3000-step truncation; earlier pilots
  used other caps — see `reward.md`).
- **Evaluation:** **one-lap termination** — the episode ends when the car
  completes one lap or crashes. This makes the metric comparable across
  track lengths.

## Metric

- **Graded fractional laps**, mean over the evaluation seeds (±std). A
  policy that gets 40 % around the track scores 0.40, not 0. This avoids a
  0 %-completion floor that would hide all the signal on hard tracks.
- We also report crash / finish counts (e.g. "crash 5/5") and, when a lap
  completes, lap time.
- **Best-checkpoint-on-validation** is the frozen selection rule: report the
  checkpoint that generalizes best, not necessarily the final one. (Motivated
  by PPO peaking early then degrading — see `experiments.md`.)

### Correction: the rule was never applied, and one checkpoint is not enough (2026-10-06)
The seed-0 grid reports the **final** checkpoint only. The `val_track_0..9` split
exists for selection but has not been used. Two measurements show why it matters:
- **Noise floor** (`grid-verification.md`): within-cell checkpoint SD is about half the
  between-cell SD. Averaging the last 5 checkpoints makes both algorithms monotone
  in diversity, while the final checkpoint alone does not.
- **Volatility** (`ppo-diagnosis.md`): the fixed PPO's held-out lap completion goes
  0 → 10/25 → 0 → 6/25 across adjacent 200 k checkpoints.
Proposed rule for the final grid: **primary = mean over the last 5 checkpoints**,
secondary = best on `val` (D14).

## Training-log vs evaluation (2026-10-06)

`monitor_0.monitor.csv` lap rates come from the **stochastic** training policy on
**its own training tracks**. Evaluation is **deterministic** on held-out tracks.
They differ a lot (fixed PPO d1: 65% training vs 0–40% held-out; Desmond's SAC d1:
15.9% vs 2%). Never quote a training-log rate as performance without the label.

## Baselines (pinned)

- **Gap-follower** with `safe_threshold = 20` (team's value; we aligned our
  copy from 5 → 20), one-lap termination, centreline spawn, 5 seeds.
- **Random** policy, same harness.

## What is deliberately NOT changed

To keep PPO-vs-SAC fair and the diversity comparison clean, we do **not**
touch: the reward function in `sb3_wrapper.py` (frozen), `n_envs`, the
track-pool sampler, episode truncation, or the gap-follower `safe_threshold`.
Normalization (VecNormalize) is applied identically where used, but note it
is algorithm-specific in effect — see `decisions.md`.

## Verification hooks

Team convention: run `tests/check_reward.py` and `tests/check_experiment.py`
after any change. (Neither is present on our branch, so there is nothing
team-side for us to run; our additive tools carry their own checks.)

## See also

- `decisions.md` — spawn, budget, penalty, normalization decisions
- `reward.md` — the reward the protocol uses
- `results.md` — protocol applied, numbers
- `incidents.md` — the n_envs and ray-caster bugs the protocol depends on
- `final-run-plan.md` — the evaluation protocol proposed for the final grid
- `grid-verification.md` — the spawn and checkpoint measurements behind the 2026-10 notes
- `glossary.md`, `handoff.md`, `overview.md` — terms, commands, context
