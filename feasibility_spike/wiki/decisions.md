---
title: Key decisions — choice, alternative, rationale
type: decision
updated: 2025-09-20
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

## See also

- `methodology.md` — the protocol these decisions pin down
- `reward.md` — the incentive-flaw evidence behind D5
- `open-questions.md` — D6 (penalty) and the grid still open
