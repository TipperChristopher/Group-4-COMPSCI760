---
title: Key decisions — choice, alternative, rationale
type: decision
updated: 2026-10-06
sources:
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

## D6 — Crash penalty freeze (PENDING)

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

## D13 — PPO uses γ 0.999 + n_steps 8192 (PROPOSED, 2026-10-06)

- **Choice (proposed).** For the final grid, PPO uses γ 0.999 and n_steps 8192, with all
  other settings at SB3 defaults. SAC shares γ 0.999 if `G999_SAC` is not worse (`final-run-plan.md`).
- **Rejected.** SB3 defaults (PPO 0 laps everywhere), per-algorithm γ (confounds
  the algorithm with the objective), penalty 40 (unstable), action repeat (refuted).
- **Why.** It is the only PPO configuration tested that completes laps, and it replicates on 3 seeds.
  It is disclosed as the one algorithm-specific change; SAC receives no matching tuning budget.
- **Supersedes** D2's "keep SB3 defaults" for γ and for PPO's rollout length.

## D14 — Score the mean of the last 5 checkpoints (PROPOSED, 2026-10-06)

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

## See also

- `methodology.md` — the protocol these decisions pin down
- `final-run-plan.md` — D13/D14 applied to the final grid
- `reward.md` — the incentive-flaw evidence behind D5
- `open-questions.md` — D6 (penalty) and the grid still open
- `overview.md` — design principles these decisions implement
