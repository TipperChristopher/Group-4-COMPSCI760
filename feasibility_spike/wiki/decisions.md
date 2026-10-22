---
title: Key decisions — choice, alternative, rationale
type: decision
updated: 2026-10-23
sources:
  - final_grid/DECISION.md on origin/final-grid @ 7552398; final_grid_v2/DECISION_v2.md on origin/final-grid-v2 (high, team pre-registrations)
  - team_repo/feasibility_spike/reward_experiment/METHODOLOGY_PLAN.md, RESULTS.md (mixed)
  - results/*.json (high)
  - Desmond handover (team)
---

# Key decisions

Each entry: the choice, the alternative rejected, and why. These are the
decisions a teammate most needs to understand to keep the study coherent.

## D1 — Fixed step budget is a controlled variable

- **Choice.** Main grid = **2M steps** (matches the proposal); 1M pilots for
  verification (learning saturates ~500k).
- **Rejected.** Tuning steps per algorithm/cell to "help" a struggling run.
- **Why.** Steps are a controlled variable. Tuning them would confound the
  PPO-vs-SAC and diversity comparisons. If PPO is unstable at 2M, that is a
  finding, not a bug to paper over.

## D2 — Minimize hyperparameter tuning

- **Choice.** Keep SB3 defaults; the **crash penalty** is the one
  reward/task parameter we allow ourselves to set, frozen identically across
  both algorithms and all cells.
- **Rejected.** Per-algorithm tuning to maximize each one's score.
- **Why.** Per-algorithm tuning biases the comparison. A frozen, shared task
  definition keeps PPO-vs-SAC honest.

## D3 — Additive, clean separation

- **Choice.** All exploration under `feasibility_spike/`; never modify the
  team's `train.py`, `evaluate.py`, `sb3_wrapper.py`, or `src/`. Narrow-track
  generation patches a **temp copy** of `random_trackgen.py`, not the original.
- **Why.** Keeps our work reviewable and non-destructive; the team's canonical
  code stays the reference.
- **Amended 2026-10-06 (user-authorised).** `train.py` on `team/crash-penalty-flag`
  gained 12 optional flags: `--crash-penalty`, `--run-tag`, `--gamma`, `--gae-lambda`,
  `--learning-rate`, `--batch-size`, and PPO-only `--n-steps`, `--n-epochs`, `--ent-coef`,
  `--target-kl`, `--log-std-init`, plus `--action-repeat` (new `action_repeat.py`).
  **All default to `None`/1, so behaviour is unchanged**, and every override is recorded in
  `run_config.json`. The reason: the team could not run a penalty-40 or γ experiment
  without editing the file. Commits `66b9257`, `67cd9d5`.

## D4 — Spawn `cl_grid_static` (a design correction, not a bug)

- **Choice.** Centreline grid spawn for both train and eval.
- **Rejected.** Gym default `rl_grid_static` (spawns 0.79–0.81 m off centre
  on real circuits — a train/eval mismatch).
- **Why.** Consistent, reproducible spawn; removes an unfair harder start on
  eval. Verified 0.79–0.81 m → 0.00 m (`spawn_study_*.json`). Framed as a
  design decision on slides, not a bug.

## D5 — Reward V2 is a design correction of a proven incentive flaw

- **Choice.** V2 = metres of centreline progress − collision-gated penalty.
- **Rejected.** V1 (time-alive + speed − steering − terminal penalty), whose
  **global optimum is the crawl** (`reward_scan2.json`).
- **Why.** V1 paid for stalling and punished finishing; a crawling policy
  looked like success. V2 pays metres, so a crash returns negative. **This
  is an incentive correction, not a performance fix** — PPO plateaus at ~0.1
  laps on both rewards (`reward.md`). Stability comes from the algorithm.

## D6 — Crash penalty freeze (RESOLVED 2026-10-09: 40 in grid A, 5 in grid B — see D16, D17)

> **Resolution (2026-10-23).** No single value was frozen. Grid A kept the team's penalty 40 (D16) and grid B
> used 5 (D17). The final test showed the penalty is algorithm-specific: 40 suits SAC, 5 suits PPO and cripples
> SAC (`final-results.md`). The text below is the record as of 2026-10-06.

- **State.** Pilots used ~40; the team's canonical V2 uses **5.0** (matches
  their baselines, slides, verification CSVs).
- **Evidence.** The PPO-p5 vs PPO-p40 2M runs both plateau at ~0.1 laps, so
  the penalty choice does not change PPO's story (`results.md`).
- **Decision needed.** Team must freeze **one** value before the grid. Adopting
  the team's 5.0 is the low-friction choice. See `open-questions.md`.
- **Evidence update 2026-10-06 — recommendation: freeze 5.**
  - The PPO-only basis above was weak: penalty is not the cause of PPO's failure (the discount is).
  - PPO penalty 40 via `train.py` (n_steps 8192, γ 0.99): 36% laps at 868 k, then **0% at 2 M** — unstable.
  - PPO penalty 5 with γ 0.999: 65% at 2 M, replicated (3 seeds) — stable.
  - SAC p40 (ours) beat SAC p5 (Desmond's) 58% vs 12% on unseen tracks, but that is **confounded**
    with code path and n=1 (`grid-verification.md`).
  - At γ 0.999 a 5-point penalty 5 s away is already worth 3.03, versus 0.033 at γ 0.99 (`reward.md`).
  - 5 matches the seed-0 grid, the baselines and the slides.
  Still requires a team decision; the alternative (a controlled SAC p5-vs-p40 under `train.py`,
  ≥2 seeds) costs ~22 SAC-hours.

## D7 — VecNormalize is algorithm-specific

- **Finding.** Observation normalization gives **no help to PPO** but
  **enables SAC to complete** laps.
- **Choice.** Apply identically where used, and report the asymmetry rather
  than treat normalization as a neutral preprocessing step.

## D8 — Best-checkpoint-on-validation + graded fractional laps

- **Choice.** Select the checkpoint that generalizes best (not the final
  one); score fractional laps (0.40 for 40 % of a lap), not binary
  completion.
- **Why.** PPO peaks early then degrades — the final checkpoint understates
  it. Binary completion floors all hard-track signal to 0.
- **Status 2026-10-06: not applied in the seed-0 grid** (final checkpoint only). Now that
  rates are 0–70%, lap completion carries signal, and it is the proposed primary metric.
  The selection rule is refined in D14.

## D9 — Use the SAC ~1.2M checkpoint results in the slides

- **Choice.** Present SAC at the ~1.2M checkpoint (labelled "1M" loosely in
  notes); pause the 1M→2M tail.
- **Why.** 1M→2M only optimizes lap time, not the conclusions
  (`experiments.md`). Documented as a caveat.

## D10 — Reward and spawn are design decisions; only n_envs and ray-caster are bugs

- **Choice.** On slides, "two bugs" = the n_envs update-scaling bug and the
  shared ray-caster bug (`incidents.md`). Reward and spawn are "design
  decisions / evolution."
- **Why.** Accuracy: the reward and spawn were deliberate task-definition
  choices, not defects. The timeline slide was corrected from "four bugs" to
  "two bugs + reward/spawn redesign."

## D11 — Keep per-segment checkpoints for the grid (lesson)

- **Choice.** Going forward, every training run saves a checkpoint each
  segment (`--keep-checkpoints`).
- **Why.** The pilots overwrote earlier checkpoints, so the
  overfitting-timing question required a full retrain. Cheap disk now saves
  reruns later.

## D12 — Canonical tracks are seed 123; fingerprint before comparing (2026-10-06)

- **Choice.** All training and evaluation use the seed-123 pool recorded in
  `tracks/manifest.json` (`d70dbc8f`). On a new machine, run
  `diagnostics/build_canonical_tracks.py` (or `generate_track_pool.py` + `tracks/make_heldout_tracks.py`)
  and verify against the manifest. `train.py` refuses non-canonical maps.
- **Rejected.** `make_synth_tracks.py --seed 0` (the old handoff instruction), which writes
  different tracks under the same names.
- **Why.** 0/20 tracks matched across machines despite identical names (`incidents.md`).
  Reproduced: 100/100 tracks, manifest byte-identical.

## D13 — PPO uses γ 0.999 + n_steps 8192 (ADOPTED in grid B with ent_coef 0.01, 2026-10-09)

> **Outcome (2026-10-23).** Adopted for grid B only, with `ent_coef 0.01` added (late-collapse fix) and penalty 5
> (D17). SAC did **not** share γ 0.999: it broke SAC at d20 with penalty 40, and at penalty 5 it was no better
> than γ 0.99 (Q11). Grid A kept PPO at defaults (D16). The text below is the 2026-10-06 proposal.

- **Choice (proposed).** For the final grid, PPO uses γ 0.999 and n_steps 8192, with all
  other settings at SB3 defaults. SAC shares γ 0.999 if `G999_SAC` is not worse (`final-run-plan.md`).
- **Rejected.** SB3 defaults (PPO 0 laps everywhere), per-algorithm γ (confounds
  the algorithm with the objective), penalty 40 (unstable), action repeat (refuted).
- **Why.** It is the only PPO configuration tested that completes laps, and it replicates on 3 seeds.
  It is disclosed as the one algorithm-specific change; SAC receives no matching tuning budget.
- **Supersedes** D2's "keep SB3 defaults" for γ and for PPO's rollout length.

## D14 — Score the mean of the last 5 checkpoints (SUPERSEDED by D18 as the primary rule, 2026-10-09)

> **Outcome.** Grid B's pre-registration made best-of-five on vw_val the primary rule and kept the last-five mean
> as a secondary robustness score, "reported, never used to select" (`DECISION_v2.md`). See D18.

- **Choice (proposed).** Primary = mean over checkpoints 1.6–2.0 M. Secondary = best on `val`.
- **Rejected.** Final checkpoint only, and best-on-`val` as the primary rule.
- **Why.** Within-cell checkpoint noise is about half the between-cell signal (`grid-verification.md`), and
  adjacent checkpoints swing 0 ↔ 10/25 (`ppo-diagnosis.md`). Selecting the best of a 10-track val set
  on noise this large mostly selects noise. Averaging made the seed-0 grid monotone.

## D15 — Every claim cites a measurement at the right granularity (process, 2026-10-06)

- **Choice.** Before a claim enters the wiki or the deck, check: (a) one-variable comparison,
  not a two-variable one; (b) held-out deterministic eval, not a training-log rate;
  (c) more than one checkpoint and seed, or say n=1; (d) track identity fingerprinted.
- **Why.** Each item was violated once in October and caught. See the corrections in
  `ppo-diagnosis.md` and `incidents.md`.

## D16 — Grid A: the team's pre-registered tuning-round-3 result (defaults, penalty 40, no time cost) (2026-10-08)

- **Choice.** SAC = S0 and PPO = P0 (SB3 defaults), crash penalty 40, no time cost, varied-width tracks, 24 runs
  (Desmond SAC s0–1, Grant SAC s2, Chris PPO s0–2). Source: `final_grid/DECISION.md`, branch `final-grid` @ `7552398`.
- **The rule (fixed before round 3 ran).** Adopt the time cost only if, for both algorithms, the best time-cost
  candidate that passes the reliability bar (beats the default on 3 of 4 seeds) is within 0.03 of the best
  no-time-cost candidate; otherwise no time cost, and defaults if no tuned setting passes.
- **Rejected.** The time cost for both (no SAC time-cost candidate passed: S4 1/4, S2 0/2, S5 0/4); PPO P2
  (γ 0.999 + n_steps 8192 + time cost), which passed for PPO (0.277 vs 0.124, 3/4 seeds) but the rule needs both
  algorithms. Penalty 5 and the entropy bonus were not candidates in that round.
- **Consequence.** Default PPO never learned (≈0.1 laps, 0% everywhere), so grid A alone could not answer RQ3. That
  motivated D17.

## D17 — Grid B: add a second pre-registered grid rather than change grid A (2026-10-09)

- **Choice.** Keep grid A untouched and add grid B (branch `final-grid-v2`): shared reward with penalty 5; PPO
  `--gamma 0.999 --n-steps 8192 --ent-coef 0.01`; SAC at defaults with γ 0.99 locked; the same cells and seeds as grid A
  (paired). `DECISION_v2.md` was committed first (`985b4bb`), then job files and checker (`a22bc32`) and the reference
  fingerprint (`16440e6`); every job line = grid A's line + exactly the v2 flags (checker-verified).
- **Rejected.** (A) grid A only: PPO's diversity curve would be unmeasurable. (B) Change grid A: discards a
  pre-registered decision and needed SAC-at-penalty-5 evidence that did not exist. Per-algorithm penalties inside a
  grid: the reward must be shared for a within-grid comparison.
- **Why.** Each grid is internally fair under one reward; the two together show that the reward choice favours one
  algorithm (A suits SAC, B suits PPO), which is disclosed rather than hidden.
- **Cost.** PPO received more tuning than SAC; disclosed in both DECISION documents and on the final deck.

## D18 — Checkpoint selection: best of five on vw_val, in both grids (2026-10-09)

- **Choice.** Per run, the best of the 400k / 800k / 1.2M / 1.6M / final checkpoints by vw_val mean fractional laps;
  tie → higher lap % → earlier checkpoint. These five are the checkpoints every tuning run was scored on. Grid A's
  documents stated no test-time rule, so grid B's pre-registration proposed the same rule for grid A (Desmond
  confirmed). Secondary, robustness only: the last-five mean (D14).
- **Rejected.** The final checkpoint only (33/48 runs peaked earlier, 31 of them by > 0.02 laps).
- **Record.** Grid B's selection was committed (`dd87128`, 18:13) before the test started (18:16). Grid A's
  `selection.csv` was written 10-09 19:06 but committed only with the results (`fdf6252`, 10-10 19:00); recomputing
  the rule from the raw vw_val logs reproduces all 48 picks, and model sha256s tie each pick to a file.
- **Caveat.** Near-ties: 7/24 grid B and 11/24 grid A ladders have a best-vs-second margin < 0.02 laps.

## D19 — No time cost in either grid (2026-10-08/09)

- **Choice.** `TIME_COST 0` in both grids.
- **Rejected.** 0.01 per step: SAC S2 scored below the baseline (0.439 vs 0.509), and at γ 0.999 a cost c makes
  crashing beat idling once the penalty is below c/(1−γ) = 10, so it is unsafe with penalty 5 (`tuning-vw-val.md`).
  0.0015 for PPO (below the suicide threshold): the TC0015 arm was worse, and one seed never lapped (`ppo-diagnosis.md`).
- **Why.** The progress reward with a crash penalty is enough; lap time is reported as a secondary metric (deck slide 6).

## D20 — Measure track width one way for every track set (2026-10-11)

- **Choice.** Report half-width with the team's `analysis/track_geometry.py` (the method behind `track_summary.csv`)
  for synthetic and real sets alike.
- **Rejected.** Mixing in the verification JSONs' `median_wall_clearance_m`, which reads 3–6 cm higher.
- **Why.** The mix put 1.06 / 1.11 m on a draft slide; the same-method values are 1.02 / 1.07 m (`final-results.md`).

## D21 — Claim discipline for the final talk and report (process, 2026-10-11; extends D15)

- **Choice.** Report every seed; describe track-level p-values as properties of the three agents in a cell, not of the
  algorithm; name the multiple-testing family (Holm 0.036 for the two grid-B RQ3 tests, 0.054 for seven tests); RQ3 is
  "not supported"; RQ1 always "under that reward"; no "train longer" claim; the agreed wording list in
  `final-presentation.md`.
- **Rejected.** "PPO gains more from diversity in grid B" (rests on seed 2), "real circuits are narrow", "the drop
  lowers every score".
- **Why.** Each was written once in a draft and caught against the raw data (`final-results.md` follow-ups).

## See also

- `methodology.md` — the protocol these decisions pin down
- `final-run-plan.md` — D13/D14 applied to the final grid; grid B construction (D17)
- `final-results.md` — what D16–D18 produced
- `final-presentation.md` — D21's wording list
- `reward.md` — the incentive-flaw evidence behind D5
- `open-questions.md` — status of Q1–Q20 (D6 now resolved)
- `report-handoff.md` — which decisions the report needs to explain
- `overview.md` — design principles these decisions implement
