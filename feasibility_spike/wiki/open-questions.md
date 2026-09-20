---
title: Open questions and unfinished work
type: open-question
updated: 2025-09-20
sources:
  - team_repo/feasibility_spike/reward_experiment/RESUME.txt (high)
  - results/*.json (high)
---

# Open questions

What is **not** answered yet. Nothing here is stated as a result.

## Q1 — The diversity grid (the actual experiment)

The 2 algorithms × 4 diversity levels (1/5/20/100) × ≥3 seeds at the 2M
budget is **not run**. Everything in `results.md` is single-track,
single-seed pilot data. Until the grid runs, we have no diversity curve and
no PPO-vs-SAC ranking with confidence intervals.

## Q2 — Confidence intervals / rankings

**Every training run is n=1 seed.** Point estimates are samples, not
rankings. Any "A beats B" claim needs ≥3 seeds and bootstrap CIs. Overlapping
CIs or n ≤ 3 mean "not yet supported."

## Q3 — Crash penalty freeze (5 vs 40)

Pilots used ~40; team canonical is 5.0. PPO-p5 vs PPO-p40 both plateau at
~0.1 laps (`results.md`), so the choice does not change PPO's story — but the
team must freeze **one** value before the grid. Adopting 5.0 is the
low-friction option. (`decisions.md` D6.)

## Q4 — Overfitting timing (probe running)

Does 1M steps on one track make generalization **worse over time** while
in-distribution lap time keeps improving? SAC-1M is retraining with a
checkpoint every 100k (`--keep-checkpoints`); next step is to eval each
checkpoint on `synthetic_nar7_0` + Spielberg. **nar7 already proved the
coverage/memorization problem**; this probe tests only the *timing* claim.
If negative, the coverage conclusion still stands. Resumable via `RESUME.txt`.

## Q5 — Did the reward matter for SAC?

SAC was only ever trained on V2. A SAC-on-old-reward run (~7–8 h) would tell
us whether the reward mattered for SAC or only for PPO. **Unmeasured.**

## Q6 — Does adding narrow tracks to the pool fix the creep/crash?

The natural coverage test: retrain SAC with a few narrow tracks in the pool
(e.g. 5 normal + 3 narrow). If the creep/crash on `synthetic_nar7_0`
disappears **without** ever seeing real circuits, it is a coverage problem
(memorization) and the diversity sweep is on the right track. If not, the
problem is deeper. ~3.5 h, resumable.

## Q7 — Corner-sharpness cap

The synthetic generator caps curvature at ~0.544 (`TRACK_TURN_RATE = 0.31`).
Real circuits reach 1.273 (Spielberg) / 0.938 (Silverstone). This is a second
ceiling that only bites **after** the spawn input-shift is fixed. Untested:
does raising `TRACK_TURN_RATE` (or adding sharper tracks) help transfer once
spawn geometry is covered?

## Q8 — Would PPO stabilize with more envs?

Deliberately **not** tested — `n_envs` is a pinned protocol constant (a fixed
budget with more envs means fewer updates). We claim PPO is unstable *under
this protocol*, not that PPO is broken in general.

## Q9 — Merge and freeze before the grid

`desmond/reward-and-episode-fix` must merge into `main` (expect a conflict
with main's interim low-speed-penalty reward + the 5M-steps commit — that
interim junk is not merged). Then freeze one canonical setup: penalty (Q3),
budget (2M), spawn (`cl_grid_static`). Only then run the grid.

## See also

- `experiments.md` — the pilots these questions extend
- `decisions.md` — the frozen choices; D6 is Q3
- `handoff.md` — how to resume the queued runs
