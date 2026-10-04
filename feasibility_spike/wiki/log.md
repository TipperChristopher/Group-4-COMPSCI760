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

## [2025-09-20] note | title-slide names aligned to roles slide
- Title slide now uses Victory / Grant (were Yi Wei / Zihang Zhang) to match
  the roles slide. Mapping team-confirmed: Victory = Yi Wei, Grant = Zihang
  Zhang. Roles slide unchanged; no stray old-name occurrences remain.
- Applied via fix_team_text() in the deck builder (reproducible).

## [2025-09-20] update | overfitting probe answered; branch merged; tests green

- **Q4 (overfitting timing) ANSWERED: no.** Evaluated SAC checkpoints
  100k/300k/600k/1M zero-shot through the standard protocol and the correct
  loader (`eval_sac_ck*.json`). Unseen-synthetic laps rise monotonically
  (0.42 → 0.69 → 0.91 → 1.00; finishes 0 → 5/5); real circuits stay flat-low
  (0.02–0.17) at every checkpoint. "Train less" is not a fix; the real-circuit
  gap is coverage, present from the first checkpoint — complements the nar7
  finding. Updated: `experiments.md`, `results.md`, `open-questions.md` (Q4),
  `index.md` snapshot, `handoff.md`.
- **Repo state:** branch merged with `origin/main` (0 behind, ~37 ahead, still
  purely additive — no team file touched). Main now carries the merged reward
  PR #2. Recorded in `handoff.md`.
- **Team tests green** on the merged branch: `check_reward.py` Stage 1,
  `check_experiment.py` Stage 2 (needed the 20-track pool, generated via
  `make_synth_tracks.py --n 20 --seed 0`).
  - **Correction (added later; the record above is left as it happened):**
    seed 0 was the wrong seed. The canonical training tracks come from seed
    123 (`generate_track_pool.py`'s effective default), and seed 0 produces
    different tracks under the same `synthetic_track_0..19` names, so the
    pool generated here does not match the canonical set. The tests still
    passed because they check mechanics (resets, rotation, determinism), not
    track identity. Any model trained on that pool used non-canonical tracks.
    Regenerate with `--seed 123` and confirm with
    `python tracks/make_heldout_tracks.py --verify`. See `handoff.md`.
- **Corrected our own docs:** `INVESTIGATION.md` and `EXPERIMENT_LOG.md` carried
  the retracted "crash at 2.7–10.7 m" figure (VecNormalize-loader artifact).
  Both now state the verified ~34 m / ~26 m crash locations and point at
  `incidents.md` for the loader pitfall.
- **Added session docs** (referenced from `handoff.md`): `EXPERIMENT_LOG.md`
  (complete what/why/result record for a new session), `INVESTIGATION.md` (the
  two open problems with ranked fixes), `TRAIN_README.md` (frozen train/eval
  commands for teammates).
- Still open: diversity grid + CIs; crash-penalty freeze (5 vs 40); **training
  on a widened generator** (real-like corner sharpness) — never attempted, now
  the top-ranked next experiment; PPO n_envs=8 ablation.
