---
title: Project overview — Group-4 F1TENTH zero-shot generalization
type: concept
updated: 2026-10-06
sources:
  - team_repo/README.md (high)
  - ppo-diagnosis.md, grid-verification.md, sim-to-real-width.md (synthesis, 2026-10)
  - team_repo/plan.md, PLAN_presentation2.md (mixed)
  - proposal / presentation-1 (team, referenced)
---

# Project overview

## The question

Does an RL driving policy trained on the F1TENTH simulator **learn to drive**
or **memorize its training tracks**? We measure this as **zero-shot
generalization**: train on some set of tracks, then evaluate on tracks the
policy has never seen, with no fine-tuning.

The controlled variable is **training diversity** — the number of distinct
tracks in the training pool: **1 / 5 / 20 / 100**. The hypothesis is that
more diverse training turns track-specific memorization into transferable
driving competence, and that the two algorithms we compare respond to
diversity differently.

## What we compare

- **PPO** (on-policy) vs **SAC** (off-policy), under one identical protocol.
- **Baselines** for reference: a reactive **gap-follower** (hand-written,
  no learning) and a **random** policy.
- Evaluation on **held-out synthetic tracks** and on **real circuits**
  (Spielberg, Silverstone) that are geometrically outside the synthetic
  training distribution.

The comparison is deliberately kept fair: same reward, same step budget,
same normalization, same spawn protocol, same evaluation seeds for both
algorithms and all diversity levels. See `methodology.md`.

## Design principles (constraints we hold ourselves to)

- **Fixed step budget is a controlled variable** — we do not tune training
  steps to help one algorithm. Main-grid budget = **2M steps** (matches the
  proposal); 1M pilots are used for verification because learning saturates
  around 500k. See `decisions.md`.
- **Minimize hyperparameter tuning** so the PPO-vs-SAC comparison is not
  biased by per-algorithm tuning. The crash penalty is the one reward/task
  parameter we allow ourselves to set, and it is frozen identically across
  both algorithms and all cells.
  *Amended 2026-10-06:* the discount factor turned out to define the task
  itself (`ppo-diagnosis.md`), so the proposal is to share γ 0.999 across both
  algorithms and allow PPO one disclosed rollout-length change (`final-run-plan.md`, D13).
- **Additive and clean.** All of our exploration lives under
  `feasibility_spike/`; we never modify the team's `train.py`,
  `evaluate.py`, `sb3_wrapper.py`, or `src/`.
  *Amended 2026-10-06:* with the user's authorisation, `train.py` gained optional
  flags (crash penalty, γ, PPO hyperparameters, action repeat) that all default to
  the existing behaviour; see `decisions.md` D3.

## Where the project is now (2026-10-06)

Final-presentation stage. Since Presentation 2:

- **The diversity grid has been run once** (Desmond, seed 0, 2 M, 8 cells +
  baselines + narrow-width sets) and **independently verified**: every headline number
  reproduces, evaluation is bit-identical across machines (`grid-verification.md`).
- **SAC answers the RQ on synthetic tracks:** held-out lap rate 2/22/31/38% for
  d1/5/20/100, transfer ratio 0.13 → 0.76. One seed, so it is a trend, not yet a ranking.
- **PPO's failure is diagnosed and fixed at d1:** γ 0.99 at 100 Hz makes crashing the
  optimal strategy *and* the default rollout collapses the optimiser. γ 0.999 + n_steps 8192
  laps in 52–65% of training episodes (3 seeds) (`ppo-diagnosis.md`).
- **The real-circuit failure is corridor width**, shown by a controlled ablation; the
  generator has zero width variance (`sim-to-real-width.md`).
- **Next:** the final grid with the fixed PPO, ≥3 seeds and averaged checkpoints
  (`final-run-plan.md`, proposed, not yet agreed).

## Where the project was at Presentation 2 (2026-09-20, historical)

Presentation-2 stage (methodology & results update). Done at that point:

- Evaluation protocol formalized and hardened (`methodology.md`).
- Two real bugs found and fixed; reward and spawn redesigned as deliberate
  design decisions, not bug fixes (`incidents.md`, `reward.md`, `decisions.md`).
- Baselines measured (`results.md`).
- **Single-track, single-seed capability pilots** run for PPO and SAC at
  the 2M budget (SAC at a ~1.2M checkpoint), plus unseen-track evaluation
  (`experiments.md`, `results.md`).
- A finding that the real-circuit failure is **reproducible on narrow
  synthetic tracks** — a training-coverage problem, not something specific
  to real circuits (`experiments.md`, section "nar7").

**Not yet done:** the full 2×4×≥3-seed diversity grid; confidence
intervals; the checkpoint-timing (overfitting) probe is running *(since answered: `open-questions.md` Q4)*. See
`open-questions.md`.

## The headline pilot result

Under the identical protocol, on one training track:

- **SAC** completes laps and keeps improving; it generalizes **within the
  synthetic distribution** but crashes on real circuits.
- **PPO** peaks early then forgets; it is at roughly random level on real
  circuits.
- The **no-learning gap-follower** completes a real circuit (Spielberg)
  that neither RL policy can — the generalization gap the whole study is
  about.

Numbers and citations in `results.md`.

### Correction: the headline pilot result (2026-10-06)
- "SAC generalizes within the synthetic distribution" held for **our penalty-40 SAC**
  (58% on 5 unseen tracks); Desmond's penalty-5 SAC d1 manages 12% on the same kind of test.
  The claim needs the penalty and the diversity level stated (`grid-verification.md`).
- "PPO peaks early then forgets" described the symptom. The cause was the objective
  (γ) together with the optimiser (rollout length), not on-policy forgetting as such (`ppo-diagnosis.md`).
- "The real-circuit failure is training coverage" was right, and the uncovered variable is
  now identified as **width** (`sim-to-real-width.md`).

## Deliverable for Presentation 2

`Group4_ProjectUpdate_FINAL2.pptx` (COMPSCI 760 root). Build system and
slide-by-slide content in `handoff.md`.

## See also

- `methodology.md` — how we evaluate
- `experiments.md` — what we ran and found
- `decisions.md` — why we made the choices we did
- `open-questions.md` — what is still open
- `ppo-diagnosis.md`, `grid-verification.md`, `sim-to-real-width.md` — October findings
- `final-run-plan.md` — the proposed final run
