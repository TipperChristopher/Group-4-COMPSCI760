# Operation log

Append-only. Semantic entries (what knowledge changed and why), not
file-level diffs — git records the diff.

## [2026-09-20] create | project wiki initialized

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

## [2026-09-20] note | deck + tooling changes this session

- Fixed the zero-shot generalization chart (grouped-bar width/offset bug that
  made bars overlap across track slots) — `gen_gap_figure.py`.
- Corrected the timeline slide wording "four bugs" → "two bugs + reward/spawn
  redesign" via `fix_team_text()` in the deck builder.
- Added `--keep-checkpoints` to `learning_curve.py`; launched the SAC-1M
  overfitting-timing probe (per-segment checkpoints). Recorded in `RESUME.txt`.
- Added `_gen_narrow.py` (narrow-track generator, patched temp copy) to probe
  the spawn input-shift hypothesis.

## [2026-09-20] note | title-slide names aligned to roles slide
- Title slide now uses Victory / Grant (were Yi Wei / Zihang Zhang) to match
  the roles slide. Mapping team-confirmed: Victory = Yi Wei, Grant = Zihang
  Zhang. Roles slide unchanged; no stray old-name occurrences remain.
- Applied via fix_team_text() in the deck builder (reproducible).

## [2026-09-20] update | overfitting probe answered; branch merged; tests green

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

## [2026-10-06] ingest | October diagnosis session (Aolin + agent), 2026-10-04 → 06

Source: pi session `01a10425-…` and run journal `2026-10-04-cs760-final-presentation-plan`
(see `raw-sources/index.md`, bucket sessions). Evidence archived in-repo under
`results/ppo_diagnosis/` (193 files) and `diagnostics/`.

- **New pages:** `ppo-diagnosis.md` (why PPO never lapped: full hypothesis trail, the discount
  arithmetic, measured basin table, γ×n_steps interaction, replication, corrections),
  `grid-verification.md` (Desmond's seed-0 grid verified from raw files; bit-identical eval
  reproduction; noise floor; transfer ratio 0.13→0.76; the track-identity mistake),
  `sim-to-real-width.md` (narrowA/B ablation: 29–58% → 0% with centrelines fixed; training
  width span 4 cm), `final-run-plan.md` (PROPOSED final-grid configuration and environment).
- **Key knowledge changes:**
  - PPO's failure = **objective × optimiser**: γ 0.99 at 100 Hz makes crashing optimal
    (13.87 vs 7.00 by arithmetic; 7.72 vs 6.93 on trained policies) AND 2048-step rollouts
    collapse exploration. γ 0.999 + n_steps 8192 → 65% training laps; 3 seeds 52–65% at 1 M.
    Neither change alone laps by 1 M.
  - The real-circuit failure is **width** (controlled ablation), not corner sharpness.
  - Single checkpoints are not measurements (adjacent checkpoints swing 0 ↔ 10/25).
- **Corrections recorded (with `### Correction:` headings):** reward.md ("stability comes
  from the algorithm"), experiments.md (geometry on seed-0 maps; corner-sharpness ceiling
  refuted; nar7 "not sufficient"; PPO "on-policy instability"), overview.md (headline pilot
  claims), methodology.md (checkpoint rule never applied; 3 distinct spawns), ppo-diagnosis.md
  (Oct-4 "worse than standing still" inference used undiscounted returns; "γ is the lever";
  "finishing worth 0.001"; "one step outweighs the penalty"), grid-verification.md
  ("SAC cannot lap its own tracks" — wrong maps).
- **Decisions:** D3 amended (user-authorised `train.py` flags, defaults unchanged);
  D6 evidence updated, recommend 5; D8 status; new D12 (canonical seed-123 tracks),
  D13 (PPO γ 0.999 + n_steps 8192, PROPOSED), D14 (mean of last 5 checkpoints, PROPOSED),
  D15 (claim-granularity checklist).
- **open-questions.md rewritten** as a status table: Q4 answered, Q7 refuted, Q8 superseded,
  Q6 reframed → Q13; new Q10–Q14.
- **incidents.md:** four measurement pitfalls (track identity, training-log quoted as
  performance, two-variable comparison, single-checkpoint trend) + killed-run status note +
  training-not-bit-reproducible note.
- **Housekeeping:** every page's `updated:` and this log's headings said **2025**; the repo's
  first commit is 2026-08-30, so all were typos for 2026 and are corrected. The canonical
  copy of this wiki is now on `team/crash-penalty-flag` (SCHEMA.md).
- **Still open:** final grid not launched; `G999_SAC` running; presentation date unconfirmed.

- **2026-10-07:** verified the teammate's `tuning_vw_val_results.zip` (16 runs, 8 settings x 2 seeds,
  commit `6e5b60e`): configs, reward integrity (time cost 0.0100/step in S2/P2 only), scores (max delta 5e-5),
  spawn defect. New page `tuning-vw-val.md`; open-questions Q10/Q11 updated; final-run-plan SAC gamma rule
  revised (gamma 0.999 breaks SAC at d20; time cost doesn't beat baselines).

## [2026-10-08 – 09] update | stability A/B, penalty-5 tests, grid B pre-registered (back-filled 2026-10-23)

Committed on this branch at the time (`edbf8ec` 10-08 → `35b0f71` 10-09, 11 commits; plus the 10-06/07 commits
`494c9eb`, `9e69990`, `b3595ec`, `dc67eb2` on the PPO fix and the tuning round) without log entries; recorded here.
- Stability A/B (d1, 3 seeds): entropy 0.01 stops late collapse (worst seed 65% vs 17%); time cost 0.0015 worse (`ppo-diagnosis.md`).
- vw d20 penalty 5: PPO best 0.52–0.67 vs 0.03–0.13 at penalty 40; entropy fixes collapse, not checkpoint volatility.
  SAC γ 0.999 at penalty 5 (S1P5) does not collapse by 800k (`open-questions.md` Q11).
- Width breakdown on vw_val: mid widths lappable, narrow < 0.9 m mostly not (`sim-to-real-width.md`).
- Found the team's grid A decision on `origin/final-grid` (defaults, penalty 40) after proposing our own config; built
  and pre-registered grid B on `final-grid-v2` (`985b4bb`, `a22bc32`, `16440e6`) (`final-run-plan.md`).

## [2026-10-10] ingest | final bundle verified; final-results.md created

Source: `G4_full_bundle_2026-10-10.zip`, journal Attempts 33–35.
- Arm's-length recomputation: 0 mismatches (100 raw CSVs, 32 cells, 240 ladder values, 48 configs, 96 test-plan rows);
  selection rule argmax 48/48; chronology and git commits checked.
- New page `final-results.md`: both grids on vw_test + 23 real circuits, five defensible claims, honesty notes.
- Follow-ups: RQ3 "PPO gains more" rests on seed 2 (inconclusive); the synthetic → real drop is lap length, not width;
  training was not clearly still improving enough to claim longer training would beat the gap-follower.

## [2026-10-11] update | deck review, v2 deck, final-deck check, talk-script review

Source: journal Attempts 36–39; this session.
- Insights (speed vs reliability, validation predicts test, memorisation, per-track oracle) and deck v2 recorded in
  `final-results.md`.
- Final deck checked; three fixes (replay buffer 8.7 GB, widths 1.02 / 1.07 m, RQ1 "under that reward"). Talk script:
  12:47 vs a 12:00 stop, several overclaims; Aolin's part rewritten (2:09).
- Correction recorded: the 1.11 / 1.06 m widths used a different measure (same method: 1.07 / 1.02 m).

## [2026-10-23] ingest | sessions through 2026-10-23; report handoff; status refresh across the wiki

Sources: session `01a11527-≈` (10-07 → 10-23) and the post-ingest part of `01a10425-≈` (10-06 → 10-08); journal
Attempts 21–40; the final bundle; `_verify/s1p5/`. Full session list now in `raw-sources/index.md`.
- **New pages:** `report-handoff.md` (start here for the report), `final-presentation.md` (deck lineage, fixes,
  talk script, agreed wording, team rows, references).
- **final-results.md:** frontmatter added; same-method width correction; width-vs-score and length-vs-score
  correlations; crash statistics at d100 (both distance definitions); RQ1 seed-matched 20/24; 7/48 agents rise to real;
  Holm families; 33/48 early checkpoints; default-PPO lap counts; curvature; runtimes; replay-buffer arithmetic;
  gap-follower configuration; RQ3 reasoning (noise vs unfairness, laps vs percent); the PPO-tuning-bias defence.
- **decisions.md:** D6, D13, D14 resolved/adopted/superseded (headings updated, originals kept); new D16 grid A,
  D17 grid B, D18 selection rule, D19 no time cost, D20 width method, D21 claim discipline.
- **open-questions.md:** new status table (Q1, Q3, Q9, Q10, Q13, Q14 done; Q11 answered no, with the S1P5 final state:
  no collapse, 0.243 / 0.300 best through 1.6M, runs stopped ~1.81M); new Q15–Q20.
- **Status refreshed:** overview, handoff (branches, worktrees, files outside git), final-run-plan (EXECUTED; fixed a
  path where `\r` had become a line break), sim-to-real-width (update + final picture + correction), reward,
  methodology (final protocol as run), ppo-diagnosis (outcome in the final grid), glossary (new terms), incidents
  (five process pitfalls), results (pointers), tuning-vw-val (outcome), SCHEMA layout, index.
- **Still open:** report brief and due date not recorded; delivery date and marks of the final presentation not recorded.

## [2026-10-23] lint | 0 blockers; 4 warnings + 1 stale line fixed; 119/119 numbers verified

- **Structural** (`_verify/wiki_ingest/audit.py`): markdown links and anchors resolve; `index.md` lists all 21 pages;
  frontmatter complete (added to `final-results.md`); no orphans. Fixed: 4 one-way See-also links, 1 stale "is running"
  line (annotated), the `` path in `final-run-plan.md`.
- **Ground truth** (`_verify/wiki_ingest/check_numbers.py`): 119/119 numbers on the new and updated pages recomputed from
  the raw bundle files, the pilot CSVs, the S1P5 CSVs, the diagnosis monitors and the run configs.
- **Lifecycle words:** 10 hits, all in dated historical sections or annotated.
- **Remaining WARNING (pre-existing):** 10 older pages cite `mixed` sources with inline tier tags but no
  `## Source reliability` section; added to `report-handoff.md`, `final-results.md`, `final-presentation.md`.
- **Encoding incident during this ingest:** one edit wrote 39 characters as byte fragments in `final-results.md`;
  repaired with `_verify/wiki_ingest/fix_bytes.py`; a full scan of every page is clean.

