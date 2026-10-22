# Group-4 F1TENTH RL project wiki

Durable, hand-off-ready record of the project. **Start here.** For a one-file
summary of the October diagnosis work, read `CONTEXT.md` at the repo root.

**Short answer (the project in six lines, as of 2026-10-23).** We asked whether training an RL racing policy on
more tracks (1/5/20/100) improves zero-shot driving on unseen tracks, and whether that benefit differs between PPO and
SAC under a fixed 2M-step budget. **Two pre-registered grids (48 runs, 3 seeds) were tested once** on 20 unseen
synthetic tracks and 23 real circuits and verified from the raw files (`final-results.md`). **More tracks help every
learning agent on synthetic tracks, on every seed** (grid A SAC on real circuits 0.112 → 0.678 fractional laps).
**The hypothesis that SAC benefits more is not supported**: once both learn, there is no consistent difference.
Default PPO never learned; the diagnosed fix (γ 0.999 + n_steps 8192 + entropy, penalty 5) makes it learn, and the crash
penalty turns out to be algorithm-specific. The gap-follower stays more reliable on real circuits (72% vs 47% of laps).
**Next: the final report** — start at `report-handoff.md`.

## Pages

**Start here for the report**
- [report-handoff.md](report-handoff.md) — **start here**: answers by RQ with numbers and sources, methods
  contributions, limitations, do-not-claim list, data/code/figure map.
- [final-results.md](final-results.md) — the final test of both grids (verified arm's-length), seed-level follow-ups,
  crash statistics, width-vs-score, RQ3 reasoning, every later check.
- [final-presentation.md](final-presentation.md) — deck versions and fixes, the talk-script review, the agreed wording,
  team contributions and reference list.

**October findings and the final grids**
- [ppo-diagnosis.md](ppo-diagnosis.md) — why PPO never lapped: hypothesis trail, discount arithmetic, γ×n_steps
  interaction, stability A/B, outcome in the final grid.
- [final-run-plan.md](final-run-plan.md) — the final-grid configuration (EXECUTED as grids A + B) and how grid B was built.
- [tuning-vw-val.md](tuning-vw-val.md) — the teammate tuning rounds verified: time-cost mechanism, SAC γ, outcome.
- [sim-to-real-width.md](sim-to-real-width.md) — the width ablation, the varied-width fix, and what width does not explain.
- [grid-verification.md](grid-verification.md) — Desmond's seed-0 grid verified from raw files; the noise floor.

**Foundations**
- [overview.md](overview.md) — project goal, research question, where we are.
- [methodology.md](methodology.md) — evaluation protocol, including the final protocol as run.
- [reward.md](reward.md) — reward history (V1/interim/V2), the incentive flaw, the discount, the penalty decision.
- [experiments.md](experiments.md) — the September-2026 pilots (with later corrections).
- [results.md](results.md) — pilot and October numbers in tables, each cited to a file; pointers to the final numbers.
- [decisions.md](decisions.md) — key decisions D1–D21: choice, rejected alternative, rationale.
- [incidents.md](incidents.md) — the two real bugs, the loader pitfall, measurement and process pitfalls.
- [open-questions.md](open-questions.md) — status table Q1–Q20: answered, open, future work.
- [handoff.md](handoff.md) — branches, worktrees, files outside git, commands.
- [glossary.md](glossary.md) — domain terms (incl. grid A/B, vw splits, track-level tests).

## Conventions

See [SCHEMA.md](SCHEMA.md). Every quantitative claim cites its source file.
Single-seed results are labelled as such. Corrections use a `### Correction:` heading
rather than silent rewrites. Pages carry `updated:`. Sources: [raw-sources/index.md](raw-sources/index.md).

## Status snapshot (2026-10-23)

- **Done:** both final grids trained (grid A 10-08/09, grid B 10-09/10) and tested once (10-10 18:16–18:57);
  arm's-length verification (0 mismatches); deck review, v2 deck, final-deck fixes, talk-script review (10-10/11);
  presentation (after 10-11).
- **Running:** nothing.
- **Next:** the final report (`report-handoff.md`). The report brief, format and due date are not recorded locally.
- **Not done (future work):** equal-budget reward tuning, more seeds, budget sweep, hybrid with the gap-follower,
  a physical car (`open-questions.md` Q15–Q20).

## Status snapshot (2026-10-06, historical)

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
