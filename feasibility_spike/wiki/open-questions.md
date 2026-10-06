---
title: Open questions and unfinished work
type: open-question
updated: 2026-10-06
sources:
  - team_repo/feasibility_spike/reward_experiment/RESUME.txt (high)
  - results/*.json (high)
  - results/ppo_diagnosis/**, results/grid_s0/** (high, 2026-10)
  - ppo-diagnosis.md, grid-verification.md, sim-to-real-width.md, final-run-plan.md (synthesis)
---

# Open questions

What is **not** answered yet, and what has since been answered or refuted.
Nothing here is stated as a result unless it is marked ANSWERED and points to a source.

| Q | question | status (2026-10-06) |
|---|---|---|
| Q1 | The diversity grid | **PARTLY RUN** — seed 0, γ 0.99 (Desmond); final grid proposed |
| Q2 | Confidence intervals | **OPEN** — needs ≥3 seeds |
| Q3 | Crash penalty 5 vs 40 | **RECOMMENDATION: 5** — team decision pending (D6) |
| Q4 | Overfitting over time | **ANSWERED (no)** — extended to 2 M; volatility, not decline |
| Q5 | Did the reward matter for SAC? | **OPEN**, low priority |
| Q6 | Does adding narrow tracks fix real circuits? | **REFRAMED** → Q13 (width-randomised pool) |
| Q7 | Corner-sharpness cap | **REFUTED** (`sim-to-real-width.md`) |
| Q8 | Would PPO stabilise with more envs? | **SUPERSEDED** — PPO fixed by γ + n_steps, n_envs stays 1 |
| Q9 | Merge and freeze before the grid | **OPEN** — branch state changed, see below |
| Q10 | Does the PPO fix hold at d5/20/100? | **OPEN** — highest priority, ~1.3 h pilot |
| Q11 | Does γ 0.999 help or hurt SAC? | **RUNNING** (`G999_SAC`, ~12 h) |
| Q12 | Why do γ and n_steps interact? | **OPEN** (hypothesis only) |
| Q13 | Does a width-randomised pool fix real circuits? | **OPEN** — needs a generator width flag |
| Q14 | Final-presentation date and scope | **OPEN** — needed to size the SAC half |

## Q1 — The diversity grid

*Sept 2026:* not run. *2026-10-05:* Desmond ran seed 0 for both algorithms × 1/5/20/100 at
2 M with γ 0.99, plus baselines and narrow sets, and it was verified (`grid-verification.md`).
SAC shows the diversity effect on synthetic tracks; PPO is 0 laps in every cell, so PPO's diversity
curve is unmeasurable under γ 0.99. **The final grid (≥3 seeds, fixed PPO, averaged
checkpoints) is proposed in `final-run-plan.md` and not yet agreed.**

## Q2 — Confidence intervals / rankings

Every training run is still n=1 seed per cell, except the PPO d1 fix (3 seeds,
training log). Adjacent diversity levels are mostly not significantly different at seed 0.
Any "A beats B" needs ≥3 seeds and a hierarchical bootstrap (seeds, then tracks).

## Q3 — Crash penalty freeze (5 vs 40)

Evidence updated in `decisions.md` D6. Recommendation: **5**. Penalty 40 made PPO lap at
γ 0.99, but then it collapsed. γ 0.999 makes penalty 5 matter by a factor of 90. The SAC p40
advantage is confounded with code path.

## Q4 — Overfitting timing — ANSWERED (no)

Checkpoints 100k–1M rose monotonically (Sept 2026). `sac_overfit` finished 2 M: its in-training
"dips" were single-episode samples of a ~27% failure rate (30-spawn eval: 1.2 M 30/30, 2 M 22/30).
For the fixed PPO, held-out performance is **volatile** across checkpoints rather than declining
(`ppo-diagnosis.md`). Desmond's SAC **d1** does decline after 1 M in training (38% → 22%), the
one overfitting-like signal, and it occurs only at diversity 1.

## Q5 — Did the reward matter for SAC?

SAC was only ever trained on V2. Unmeasured, and no longer on the critical path.

## Q6 — Adding narrow tracks → see Q13

The idea is right, and the variable is now confirmed to be width (`sim-to-real-width.md`).
Rather than add a few narrow tracks, randomise width across the whole pool.

## Q7 — Corner-sharpness cap — REFUTED

Real circuits are sharper than training at only 0.95% of points and are straighter at p90.
Width is out of distribution at 100% of points (`sim-to-real-width.md`).

## Q8 — More envs for PPO — SUPERSEDED

PPO's failure was the objective (γ) plus rollout length, not a lack of parallel envs. `n_envs` stays 1
(Bug 1 in `incidents.md`).

## Q9 — Merge and freeze before the grid

Branch state (2026-10-06): `experiments/feasibility-spike` (September-2026 work) →
`desmond/heldout-tracks` (canonical tracks, grid tooling, 23 real circuits, narrow sets) →
`team/crash-penalty-flag` (train.py flags, diagnosis, this wiki update; pushed). None of
these are merged to `main`. Freeze the final configuration (`final-run-plan.md`) on one branch
before launching.

## Q10 — Does γ 0.999 + n_steps 8192 work at higher diversity?

Only d1 has been tested. At d100 each track gets 20 k steps instead of 2 M. Pilot one d100 seed
(~1.3 h) before committing 12 PPO runs.

## Q11 — γ 0.999 for SAC

`G999_SAC` was running at the time of writing (114 k / 2 M at 20:30 on 2026-10-06). The decision rule
is in `final-run-plan.md`.

## Q12 — Mechanism of the γ × n_steps interaction

Hypothesis: the γ 0.999 critic regresses ~1000-step returns, and its explained variance falls
from 0.96 to 0.69, so it needs more data per update. Test: γ 0.999 with n_steps 2048 and more value
epochs or a separate value learning rate. It matters for the write-up, not for the grid.

## Q13 — Width-randomised training pool

Add a half-width parameter to the generator, sample 0.7–1.5 m per track, retrain SAC (and PPO)
at d100, then evaluate on narrowA/B and the 23 real circuits. Prediction if width binds:
a large real-circuit jump. Cost: generator change plus about 6 runs.

## Q14 — Presentation date and scope

Never confirmed in the logs. It decides whether SAC runs the full 4 × 3 grid
(~65–140 CPU-hours, so it must be split across machines) or d1 + d100 only.

## See also

- `final-run-plan.md` — Q1, Q3, Q10, Q11, Q14 resolved into one plan
- `ppo-diagnosis.md` — Q8, Q10, Q12
- `sim-to-real-width.md` — Q6, Q7, Q13
- `decisions.md` — D6 (Q3), D13, D14
- `overview.md`, `experiments.md`, `handoff.md` — context and commands
