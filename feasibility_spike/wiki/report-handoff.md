---
title: Report handoff — start here for the final report
type: synthesis
updated: 2026-10-23
sources:
  - final-results.md, final-presentation.md (verified numbers and the agreed wording)
  - ppo-diagnosis.md, sim-to-real-width.md, tuning-vw-val.md, decisions.md, open-questions.md (synthesis)
  - G4_full_bundle_2026-10-10.zip, extracted at COMPSCI 760/_verify/G4_full_bundle_2026-10-10/ (high)
  - run journal ~/.pi/runs/2026-10-04-cs760-final-presentation-plan/journal.md, Attempts 1–40 (mixed)
---

# Report handoff

**State on 2026-10-23.** All experiments are finished. Both pre-registered grids (48 runs) were trained, the
final test was run once (2026-10-10), every number was re-derived from the raw files with zero mismatches, and the
results were presented. The remaining deliverable is the **final report**. No new experiment is needed to write
it: every claim below already has a measured, cited source.

**Not known locally, ask the team first:** the report brief, rubric, length, format and due date. The only
requirement file on Aolin's machine is `presentation_2_requirement.png`. Slide 17 lists Chris as doing "report
assembly". Agree who writes which section before drafting.

## Reading order for a new session

1. This page: claims, numbers, limitations, file map.
2. `final-results.md`: every final-test table and check, with the script that produced it.
3. `final-presentation.md`: the wording the group agreed on, and what was corrected in the slides.
4. `ppo-diagnosis.md`: the methods story (why default PPO never learned, and the fix).
5. `sim-to-real-width.md`: the width problem, the varied-width fix, and what remains.
6. `decisions.md` D16–D21 and `open-questions.md`: design decisions, and the material for limitations and future work.

## The study in one paragraph

**RQ3 (headline):** under a fixed 2M-step budget, does the benefit of track diversity for zero-shot driving differ
between on-policy PPO and off-policy SAC? Supporting questions: RQ1 (which algorithm generalises better?), RQ2 (do 1 → 5 → 20
→ 100 training tracks help?) and RQ4 (are differences bigger than seed noise?). The hypothesis was that SAC benefits
more, because its replay buffer keeps experience from many past tracks.

**Design:** F1TENTH gym (100 Hz physics, 1/10-scale car), 108 LiDAR beams plus 5 ego-state values in, and steering
plus speed out. The reward is metres of centreline progress minus a crash penalty, with 30 s training episodes. Tracks come from a
varied-width generator (half-width 0.60–1.50 m), split three ways: vw_train (100, nested pools d1 ⊂ d5 ⊂ d20 ⊂ d100), vw_val
(10, used for every tuning and checkpoint decision), and the test set (20 unseen vw_test tracks plus 23 real
F1TENTH circuits, used once). There are two grids, each 4 diversity levels × 2 algorithms × 3 seeds, at 2M steps:

- **Grid A:** SB3 defaults, penalty 40.
- **Grid B:** penalty 5; PPO γ 0.999, n_steps 8192, ent_coef 0.01; SAC at defaults.

Checkpoint = best of 400k/800k/1.2M/1.6M/final on vw_val. Metric = fractional laps (share of one lap
completed; 1 = a full lap), plus lap completion %. Baselines: follow-the-gap and random.

## Answers by research question

| RQ | answer (agreed wording) | key numbers | source |
|---|---|---|---|
| RQ3 | **Hypothesis not supported.** SAC gains more only in grid A, where default PPO never learned; once both learn, there is no consistent difference | grid B gain d1→d100 per seed: SAC +0.27/+0.22/+0.33 vs PPO +0.27/+0.14/+0.56 (vw_test, p = 0.33); SAC +0.09/+0.03/+0.10 vs PPO −0.05/−0.02/+0.42 (real; track test p = 0.018, Holm 0.036, rests on seed 2) | `final-results.md` RQ3 section |
| RQ1 | **No general winner.** SAC learns at defaults; PPO needs γ 0.999 + n_steps 8192 + entropy + a small penalty, and then leads **under that reward** | real d100: A-SAC 0.678, B-PPO 0.330, B-SAC 0.197, A-PPO 0.082; B-PPO > B-SAC in 20/24 seed-matched cells, but at d100 on real only seed 2 | `final-results.md` |
| RQ2 | **Diversity helps every learning agent on synthetic tracks, on every seed.** On real circuits it holds for SAC in both grids; grid B PPO is noisy | A-SAC real 0.112 → 0.375 → 0.412 → 0.678 (6×); vw_test 0.159 → 0.752; 1-track SAC survives 66% of late-training episodes on its own track but scores 0.11 (0% laps) on real | `final-results.md` |
| RQ4 | **Seed variance is large enough to flip a conclusion** | SD up to 0.30 at d100 (B-PPO real 0.299, A-SAC real 0.276); A-SAC d100 real laps 11% / 45% / 85% by seed | `final-results.md` |

**Bonus finding (measured, SAC only).** Changing only the crash penalty from 40 to 5 cut grid A SAC's real-circuit
score from 0.68 to 0.20, on every seed (0.73 / 0.92 / 0.38 vs 0.19 / 0.20 / 0.20). Speed roughly doubled
(4.5 / 5.5 → 9.8 / 10.1 m/s, vw_test / real) and the real-circuit crash rate rose from 39% to 99%.

**Baseline context.** The gap-follower scores 0.682 (39.5% laps) on vw_test and 0.839 (71.7%) on real circuits.
The best agent (A-SAC d100) beats it on vw_test (0.752), and its validation-best seed beats it on real circuits
(85% vs 72% of laps). Averaged over seeds, the gap-follower is more reliable on real circuits (72% vs 47%). Where both finish, the agent is
faster on 17/19 real circuits (median lap time 0.72×, p = 2e-5), but the gap-follower is capped at 3–5 m/s. Picking
the better of the two per track would give 83% laps (a hindsight upper bound that motivates a hybrid).

**Sim-to-real.** From vw_test to real at d100, the drop is −10% for A-SAC, −41% for B-PPO and −53% for B-SAC. Real circuits are not narrower
(median half-width 1.07 vs 1.02 m, same method) but are 2.4× longer (403 vs 166 m). Grid B agents crash after about
60 m on both sets. Validation ranks the 48 agents almost as real circuits do (Spearman 0.92; 0.98 for vw_test).

## Methods contributions worth their own section

- **PPO diagnosis** (`ppo-diagnosis.md`). Two failures were measured:
  - **The objective preferred crashing.** At 100 Hz, γ 0.99 looks about 1 s ahead, so "fast and crash" scores 13.87 vs 7.00 for "slower and survive".
  - **Short rollouts collapsed exploration.** With 2048-step rollouts, exploration σ fell from 1.0 to under 0.1 by about 0.5M steps.

  Neither γ 0.999 nor n_steps 8192 alone completes laps (0 / 0 / 0% in the heatmap); together they reach 65%, and 52–65% across 3
  seeds. An entropy bonus stops late collapse (2/5 seeds → 0/5). The penalty is the bigger lever at d20 (best 0.03–0.13 at penalty 40 vs
  0.52–0.67 at penalty 5). "Fails to generalise" was really "never learned".
- **Width ablation and the varied-width pool** (`sim-to-real-width.md`). In the pilot, SAC d100 lapped 92% of
  synthetic test tracks and 0% of real circuits. Moving only the walls in on the same centrelines (narrowA) gave 0%.
  Training on varied widths raised this to 37% of real-circuit laps (68% on narrowA). Narrow tracks remain hard (narrowB 2%).
- **Evaluation protocol** (`methodology.md`). The checkpoint is chosen on validation, and 33/48 runs picked an earlier
  checkpoint (31 of them by > 0.02 laps). The checkpoint noise floor is measured. Each test track has only 5–6 distinct start poses (110 / 137
  distinct poses across vw_test / real).
- **Reproducibility.**
  - Grid B was pre-registered in git before training (`985b4bb`), and a fingerprint check made the three PCs run identical setups.
  - The selection was frozen before the test (`dd87128`) for grid B.
  - Arm's-length recomputation found 0 mismatches over 100 raw files, 32 cells, 240 selection-ladder values, 48 configs and 96 test-plan rows.

## Limitations (as presented, plus the caveats a reviewer will ask about)

- **Unequal tuning.** PPO got more tuning (~9 settings plus the fix, vs SAC's 6), and grid B's penalty was chosen for PPO.
  This only affects the grid-B cross-algorithm comparison, where no winner is claimed.
- **3 seeds.** No seed-level test can reach p < 0.05 (the minimum two-sided p is 0.10). Track-level tests describe the three agents in a cell, not the algorithm.
- **Few start poses.** Only 5–6 distinct start poses per test track, so per-track percentages move in 10–20% steps.
- **Budget and tuning level.** 2M steps; grid A SAC's training return was still rising (validation +0.02 from 1.2M to 2M). PPO was tuned at d20 only.
- **Narrow tracks and the baseline.** Narrow tracks (< 0.9 m) are mostly unsolved, and the gap-follower is more reliable on real circuits.
- **Process.** Grid A's selection was committed after the test, though recomputing the rule reproduces all 48 picks. Evaluation
  ran on several machines (re-scoring on one machine would flip 2/24 near-tie picks; 36 grid-B vw_val evaluations were run
  on Desmond's machine). Training is not bit-reproducible; evaluation is.
- **Simulation only.** Real circuits are maps, not a real car.

## Do not claim

- "PPO beats SAC" (always "under grid B's reward").
- "More tracks help on every seed" without "on synthetic tracks".
- "RQ3 refuted" or "the algorithm makes no difference" (say: no consistent difference with 3 seeds).
- "Longer training would beat the gap-follower."
- "Real circuits are narrower", and also not "width doesn't matter" (it predicts per-track difficulty on vw_test, ρ 0.67–0.78).
- "The drop lowers every score" (7/48 agents rise; the ranking is what's kept).
- That "penalty 5 → faster → crashes" is shown for PPO (it is measured for SAC only).
- That grid A's selection was committed before the test.

## Open questions → future work

See `open-questions.md`. Still open: Q12 (mechanism of the γ × n_steps interaction), Q15 (why the replay-buffer
hypothesis failed; PPO rollouts also span several tracks), Q16 (equal-budget reward tuning), Q17 (more seeds and
start poses), Q18 (budget sweep), Q19 (hybrid agent plus gap-follower), Q20 (why the gap-follower does better on
real circuits: the tighter worst corner is a reading, not a test). Q11 (SAC at γ 0.999) is answered in the negative
and can go in a footnote.

## Data, code and figure map

| what | where |
|---|---|
| Final bundle (repo snapshot `fdf6252`: grids, selection, test CSVs, docs) | `COMPSCI 760/G4_full_bundle_2026-10-10.zip`; extracted to `_verify/G4_full_bundle_2026-10-10/` |
| Selected model checkpoints (48) | `G4_selected_models.zip` (separate team zip, **not on Aolin's machine**) |
| Raw test CSVs | `…/4_final_test/results_final_test/test_raw_{final,v2}_{SAC,PPO}_d{1,5,20,100}_s{0,1,2}_{vw_test,real}.csv`, baselines `test_raw_baseline_{gap,random}_*.csv` |
| Selection | `…/2_grid_A/selection.csv`, `…/3_grid_B/final_test_v2/selection.csv` |
| Track geometry | `…/0_pilot/track_geometry/track_summary.csv` (method: `analysis/track_geometry.py`) |
| Pilot width diagnostic | `…/0_pilot/grid_s0_r40/`, `…/0_pilot/grid_s0_r40_vw/` |
| Verification scripts | `_verify/G4_full_bundle_2026-10-10/_verify_*.py`, `_still_learning.py`; `_verify/final_pptx_review/*.py` |
| Report-ready figures (ours, regenerable) | `_verify/final_pptx_review/figs/fig_main.png` (main results), `fig_rq3.png` (per-seed gains), `fig_simreal.png` (synthetic → real); `make_figs.py` rebuilds them from the bundle |
| PPO diagnosis figures | `_verify/handoff/Group4_PPO_fix_and_final_config_2026-10-09/figures/fig0_ppo_gamma_x_nsteps.png`, `fig1`–`fig5` (`make_figures.py` in that bundle) |
| Team deck figures (width CDF + bars, heatmap, framework diagram, training curves, demo GIFs) | inside `Group 4 Final Results (1).pptx`; unzipped copies in `_verify/final_check/x/ppt/media/` (`image-5-1.png` … `image-13-1.png`) |
| Diagnosis evidence | `team_repo_heldout/results/ppo_diagnosis/`, `results/vw_p5_tests/`, `results/tuning_vw_val_verified/`, `diagnostics/` (branch `team/crash-penalty-flag`) |
| Git | GitHub `TipperChristopher/Group-4-COMPSCI760`: `final-grid` @ `7552398` (grid A), `final-grid-v2` @ `fdf6252` (grid B, final test, results), `team/crash-penalty-flag` (diagnosis + this wiki) |
| Slides and script | `final-presentation.md` (file list) |

## Source reliability

- **Numbers: high.** Every number on this page was recomputed from the raw bundle files on 2026-10-23 by
  `_verify/wiki_ingest/check_numbers.py` (119/119 match), on top of the 2026-10-10 arm's-length verification.
- **Interpretation and wording: mixed.** "Why" readings (e.g. Q15, Q20) and the agreed phrasing come from the run journal
  and sessions and are marked as readings where untested.
- **Team facts not in any file** (report brief, due date, presentation marks): **unknown**; ask the team.

## See also

- `final-results.md`: the numbers
- `final-presentation.md`: the wording and the slide history
- `decisions.md`: why the study is designed as it is (D16–D21 for the final grids)
- `open-questions.md`: limitations and future work
