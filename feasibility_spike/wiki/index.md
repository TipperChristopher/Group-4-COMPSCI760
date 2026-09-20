# Group-4 F1TENTH RL project wiki

Durable, hand-off-ready record of the project. **Start here.**

**Short answer (the project in five lines).** We test whether an RL driving
policy *learns to drive* or *memorizes its training tracks*, measured as
zero-shot generalization vs training diversity (1/5/20/100 tracks), PPO vs
SAC, against a gap-follower baseline. In single-track pilots: SAC completes
laps and generalizes within the synthetic distribution but crashes on real
circuits; PPO is unstable under our protocol; the no-learning baseline beats
both on real circuits. We proved the real-circuit failure is a
**training-coverage** problem (it reproduces on narrow synthetic tracks). The
full diversity grid with confidence intervals is **not yet run**.

## Pages

- [overview.md](overview.md) — project goal, research question, proposal → now.
- [methodology.md](methodology.md) — evaluation protocol: splits, spawn,
  termination, metric, what is pinned and why.
- [reward.md](reward.md) — reward history (V1/interim/V2), the measured
  incentive flaw, why the reward change did not rescue PPO.
- [experiments.md](experiments.md) — PPO vs SAC, learning curves,
  unseen-track eval, why SAC fails on real circuits, the nar7 coverage
  finding, track geometry, the overfitting probe.
- [results.md](results.md) — every number, in tables, each cited to a JSON
  file. Single-seed pilots.
- [decisions.md](decisions.md) — key decisions (D1–D11): choice, rejected
  alternative, rationale.
- [incidents.md](incidents.md) — the two real bugs (n_envs, ray-caster) and
  the VecNormalize loader pitfall.
- [open-questions.md](open-questions.md) — the grid, CIs, penalty freeze,
  overfitting timing, coverage test, and merges — all still open.
- [handoff.md](handoff.md) — file map, git state, resume commands, deck build,
  pre-submission checklist.
- [glossary.md](glossary.md) — domain terms.

## Conventions

See [SCHEMA.md](SCHEMA.md). Every quantitative claim cites its source file;
all current results are single-seed pilots (point estimates, not rankings);
pages carry `updated:` and stale results are re-dated and logged.

## Status snapshot (2025-09-20)

- Done: protocol formalized; two bugs fixed; reward + spawn redesigned;
  baselines measured; PPO/SAC single-track pilots + unseen-track eval; nar7
  coverage finding; Presentation-2 deck (`Group4_ProjectUpdate_FINAL2.pptx`).
- Running: SAC-1M with per-segment checkpoints (overfitting-timing probe).
- Not run: the 2×4×≥3-seed diversity grid with bootstrap CIs.

See [log.md](log.md) for the operation history.
