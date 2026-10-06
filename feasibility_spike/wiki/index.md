# Group-4 F1TENTH RL project wiki

Durable, hand-off-ready record of the project. **Start here.** For a one-file
summary of the October diagnosis work, read `CONTEXT.md` at the repo root.

**Short answer (the project in six lines, as of 2026-10-06).** We test whether
an RL driving policy *learns to drive* or *memorises its training tracks*,
measured as zero-shot generalisation against training diversity (1/5/20/100
tracks), PPO vs SAC. **SAC:** more diversity measurably improves transfer to
unseen synthetic tracks (transfer ratio 0.13 → 0.76, seed 0). **PPO:** never
completed a lap until we found the cause: γ 0.99 at 100 Hz makes crashing
optimal, *and* 2048-step rollouts collapse the optimiser. γ 0.999 + n_steps 8192
fixes it at diversity 1 (52–65% laps, 3 seeds). **Real circuits:** 0% for
everything, because the generator has zero width variance and real tracks are
narrower. The grid with ≥3 seeds and the fixed PPO is **not yet run**.

## Pages

**Start with the October findings**
- [ppo-diagnosis.md](ppo-diagnosis.md) — why PPO never lapped: the full hypothesis
  trail, the discount arithmetic, the measured basin table, the γ×n_steps interaction, and corrections.
- [grid-verification.md](grid-verification.md) — Desmond's seed-0 grid verified from
  raw files: what reproduces, the noise floor, transfer ratios, the track-identity mistake.
- [sim-to-real-width.md](sim-to-real-width.md) — the width ablation: identical centrelines,
  only walls moved, 29–58% → 0%; training width span 4 cm.
- [final-run-plan.md](final-run-plan.md) — **PROPOSED** configuration and environment
  for the final comparison, readiness, cost, pre-flight checklist.

**Foundations**
- [overview.md](overview.md) — project goal, research question, where we are.
- [methodology.md](methodology.md) — evaluation protocol: splits, spawn, termination,
  metric, checkpoint selection.
- [reward.md](reward.md) — reward history (V1/interim/V2), the incentive flaw, and how
  the discount factor interacts with it.
- [experiments.md](experiments.md) — the September-2026 pilots: PPO vs SAC, learning curves,
  unseen-track eval, nar7, geometry, overfitting probe (with later corrections).
- [results.md](results.md) — every number, in tables, each cited to a file.
- [decisions.md](decisions.md) — key decisions D1–D15: choice, rejected alternative, rationale.
- [incidents.md](incidents.md) — the two real bugs, the loader pitfall, and four
  measurement pitfalls from October.
- [open-questions.md](open-questions.md) — what is still open, answered, or refuted.
- [handoff.md](handoff.md) — branches, file map, commands, how to resume.
- [glossary.md](glossary.md) — domain terms (incl. γ, horizon, n_steps vs batch size).

## Conventions

See [SCHEMA.md](SCHEMA.md). Every quantitative claim cites its source file.
Single-seed results are labelled as such. Corrections use a `### Correction:` heading
rather than silent rewrites. Pages carry `updated:`.

## Status snapshot (2026-10-06)

- **Done:** protocol formalised; two bugs fixed; reward and spawn redesigned; 2026
  pilots; Desmond's seed-0 grid (8 cells + baselines + narrow sets) run and
  independently verified; canonical seed-123 pool reproduced locally; `train.py`
  gained 12 optional flags (defaults unchanged); PPO root cause found and replicated
  at d1 (3 seeds); width ablation; all evidence archived in `results/ppo_diagnosis/`.
- **Running (Aolin's PC):** `G999_SAC` (SAC at γ 0.999, ~12 h), `G999_s1/s2`
  (replication to 2 M), `G999_ns2048` (γ without n_steps), `R1_repeat10_alone` (refuted, can be stopped).
- **Not run:** the final grid (≥3 seeds, fixed PPO, averaged checkpoints); the
  width-randomised training pool; 21 of 23 real circuits not installed locally.
- **Needs a team decision:** `final-run-plan.md` (SAC γ, scope versus presentation date).

See [log.md](log.md) for the operation history.
