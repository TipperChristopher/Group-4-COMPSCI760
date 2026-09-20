# Operation log

Append-only. Semantic entries (what knowledge changed and why), not
file-level diffs — git records the diff.

## [2025-09-20] create | project wiki initialized

- Created the wiki under `feasibility_spike/wiki/` for hand-off, following the
  llm-wiki layout (SCHEMA, index, log, raw-sources registry + topic pages).
- Compiled from verified session context and the result files under
  `reward_experiment/results/`. Every quantitative claim cited to a JSON or
  source path; all current results flagged as single-seed pilots.
- Pages: overview, methodology, reward, experiments, results, decisions,
  incidents, open-questions, handoff, glossary.
- Key facts recorded: two bugs (n_envs update-scaling, shared ray-caster);
  reward V1→V2 incentive-flaw redesign (`reward_scan2.json`); spawn
  `cl_grid_static` fix (0.79–0.81 m → 0.00 m); PPO unstable / SAC completes
  pilots; unseen-track eval (SAC 1.0 in-distribution, crashes real; gap wins
  real); nar7 coverage finding (real-circuit failure reproduces on narrow
  synthetic tracks, `eval_sac_nar7.json`); track geometry (synthetic max
  curvature 0.544 vs Spielberg 1.273).
- Open: diversity grid + CIs not run; crash-penalty freeze (5 vs 40);
  overfitting-timing probe running; SAC-on-old-reward and narrow-in-pool
  tests unmeasured.

## [2025-09-20] note | deck + tooling changes this session

- Fixed the zero-shot generalization chart (grouped-bar width/offset bug that
  made bars overlap across track slots) — `gen_gap_figure.py`.
- Corrected the timeline slide wording "four bugs" → "two bugs + reward/spawn
  redesign" via `fix_team_text()` in the deck builder.
- Added `--keep-checkpoints` to `learning_curve.py`; launched the SAC-1M
  overfitting-timing probe (per-segment checkpoints). Recorded in `RESUME.txt`.
- Added `_gen_narrow.py` (narrow-track generator, patched temp copy) to probe
  the spawn input-shift hypothesis.
