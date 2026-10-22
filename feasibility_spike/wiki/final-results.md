---
title: Final results — grids A + B on the final test (verified)
type: synthesis
updated: 2026-10-23
sources:
  - G4_full_bundle_2026-10-10.zip (repo snapshot fdf6252; extracted to COMPSCI 760/_verify/G4_full_bundle_2026-10-10/, not in git) (high)
  - _verify/G4_full_bundle_2026-10-10/_verify_{summary,selection,configs,tracks,ladders}.py, _still_learning.py (high, our recomputation)
  - _verify/final_pptx_review/{check_claims,check2,insights,width_same_method,width_vs_score}.py (high)
  - run journal Attempts 33–39 (mixed)
---

# Final results (grids A + B, final test)

Status: **verified arm's-length 2026-10-10** (Aolin). Source: `G4_full_bundle_2026-10-10.zip`
(repo snapshot `fdf6252`, test run 2026-10-10 18:16–18:57 NZ). Verification scripts in
`_verify/G4_full_bundle_2026-10-10/`. Every number below was recomputed from the raw per-episode
CSVs; all matched the bundle's summary files exactly (0 mismatches across 100 per-run files, 32
cells, 240 selection-ladder values, 48 run configs, 96 test-plan rows).

## What was run

| | grid A | grid B |
|---|---|---|
| commit | `7552398` | `16440e6` |
| reward | progress − 40 on crash, no time cost | progress − 5 on crash, no time cost |
| SAC | SB3 defaults (γ .99) | SB3 defaults (γ .99) — locked |
| PPO | defaults (γ .99, n_steps 2048, ent 0) | γ .999, n_steps 8192, ent_coef .01 |
| cells | {d1, d5, d20, d100} × seeds {0,1,2}, 2M steps, 1 env, vw_train | same cells (paired comparison) |
| selection | best of 400k/800k/1.2M/1.6M/final on vw_val (laps; tie → lap% → earlier) | same rule |
| test | vw_test (20 tracks) + 23 real circuits, 10 eps/track, eval-seed 0, cl_grid_static, 15000-step cap | same, run once on one machine |

## Final test results (fractional laps, mean ± SD over 3 seeds; lap completion %)

**vw_test**

| grid | algo | d1 | d5 | d20 | d100 |
|---|---|---|---|---|---|
| A | SAC | 0.159 ± 0.003 (0%) | 0.469 ± 0.195 (24.8%) | 0.470 ± 0.158 (23.5%) | **0.752 ± 0.206 (58.8%)** |
| A | PPO | 0.099 ± 0.033 (0%) | 0.150 ± 0.023 (0%) | 0.169 ± 0.122 (0.3%) | 0.113 ± 0.033 (0%) |
| B | SAC | 0.143 ± 0.073 (0.7%) | 0.281 ± 0.073 (2.3%) | 0.331 ± 0.097 (6.0%) | 0.419 ± 0.033 (8.5%) |
| B | PPO | 0.239 ± 0.076 (1.7%) | 0.403 ± 0.122 (8.5%) | **0.570 ± 0.170 (20.2%)** | 0.560 ± 0.194 (24.2%) |

**real circuits (23)**

| grid | algo | d1 | d5 | d20 | d100 |
|---|---|---|---|---|---|
| A | SAC | 0.112 ± 0.009 (0%) | 0.375 ± 0.225 (20.3%) | 0.412 ± 0.238 (18.1%) | **0.678 ± 0.276 (46.8%)** |
| A | PPO | 0.091 ± 0.006 (0%) | 0.097 ± 0.031 (0.6%) | 0.100 ± 0.046 (0%) | 0.082 ± 0.024 (0%) |
| B | SAC | 0.126 ± 0.046 (0.1%) | 0.118 ± 0.040 (0.4%) | 0.160 ± 0.040 (0.3%) | 0.197 ± 0.008 (0.7%) |
| B | PPO | 0.213 ± 0.035 (4.3%) | 0.304 ± 0.046 (4.9%) | 0.265 ± 0.070 (5.1%) | 0.330 ± 0.299 (14.2%) |

Baselines: gap-follower **0.682 (39.5%) vw_test / 0.839 (71.7%) real**; random 0.037 / 0.019 (0%).

## What the final presentation can claim

1. **Diversity → generalisation holds for SAC under the team's original reward (P40).**
   Monotone on both test sets; 6× improvement d1→d100 on real (0.112 → 0.678). d5 already captures
   most of the vw_test gain; real circuits keep improving through d100. PPO (grid B) shows the
   trend on vw_test (0.239 → 0.570 by d20) but it is noisy on real (d20 dips, d100 SD 0.299).
2. **PPO's failure is configuration, not algorithm.** Grid A PPO (defaults) is ≈0.10 laps / 0% on
   everything. Grid B PPO (γ .999 + n_steps 8192 + ent .01, penalty 5) reaches 0.570 (20.2%) on
   vw_test d20 — above grid B SAC at every cell. Root causes: γ horizon × optimiser interaction
   plus σ collapse (see `ppo-diagnosis.md`).
3. **The crash penalty is an algorithm-specific choice, not a neutral one.** Penalty 40 favours
   SAC; penalty 5 favours PPO and cripples SAC (B-SAC d100 0.419/0.197 vs A-SAC 0.752/0.678).
   A single shared reward cannot be fair to both algorithms — state this when comparing grids.
4. **The sim-to-real ordering flips.** On vw_test the best learned policy (A-SAC d100, 0.752)
   beats the gap-follower (0.682). On real circuits the gap-follower (0.839) beats the best learned
   policy (0.678). Real circuits are longer (6 exceed the 15000-step cap at 3 m/s: Melbourne,
   Sepang, Shanghai, Silverstone, Sochi, Spa) and mostly have raceline CSVs; the vw tracks are
   narrower on average. Per-track: gap laps 15/23 real circuits; A-SAC d100 (seed 0) laps
   Oschersleben/Sakhir/Spielberg/MexicoCity 100%, but 0% on Melbourne/Shanghai/YasMarina/Zandvoort
   — complementary strengths, not nested.
5. **Seed noise is the headline uncertainty.** A-SAC d100 real: seeds 11% / 45% / 85% laps.
   All claims must be seed-averaged with SDs. Per-track percentages have 10–20% granularity
   (only 5–6 distinct spawn poses per track: 110 distinct across 200 vw_test episodes, 137 across
   230 real).

## Honesty notes (all disclosed in the bundle README, all verified here)

- Grid A's `selection.csv` was written 10-09 19:06 (mtime verified) but committed only after the
  test; recomputation + model-sha256 checks were done before the test (bundle README documents this).
- Grid B selection was committed (`dd87128`, 18:13) before the test started (18:16) — verified.
- Cross-machine CPU differences: re-scoring grid B candidates on one machine would flip 2/24
  choices (both near-ties). We measured 7/24 grid B and 11/24 grid A ladders with best-vs-second
  margins < 0.02 laps — consistent with that claim.
- 36 of grid B's vw_val evals (Chris's PPO runs) were filled on Desmond's machine.
- PPO received more tuning than SAC (disclosed in both DECISION documents).
- Model checkpoints are in a separate zip (`G4_selected_models.zip`); sha256s recorded in
  `test_plan.csv` and both selection CSVs.

## See also

- `ppo-diagnosis.md` — the two PPO root causes and the fix
- `tuning-vw-val.md` — tuning rounds 1–3, the penalty-40-vs-5 evidence
- `sim-to-real-width.md` — the width story (narrow < 0.9 m unsolved; real median ≈ 1.05 m — that is the median of per-track *minimum* half-widths; the median half-width is 1.07 m, see the 2026-10-11 correction below)
- `final-presentation.md` — how these results were presented (deck versions, fixes, talk script, Q&A)
- `report-handoff.md` — start here when writing the report
- `final-run-plan.md` — pre-registration and grids A/B construction

## Follow-up (2026-10-10, deck review): two claims need seed-level qualification

Checked while reviewing `Group 4 Final Results.pptx` (scripts: `_verify/final_pptx_review/check_claims.py`, `check2.py`).

**1. Grid B PPO's real-circuit gain from diversity rests on one seed.** The track-level Wilcoxon tests in
the deck reproduce exactly (100 vs 1: SAC A 23/23, SAC B 21/23, PPO B 20/23, PPO A 11/23 p=0.34;
"PPO gains more, grid B" 17/23 p=0.018; vw_test p=0.33). But per seed, d1→d100 on real:

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| B-PPO | 0.18→0.13 | 0.21→0.18 | 0.25→**0.67** |
| B-SAC | 0.10→0.19 | 0.18→0.20 | 0.10→0.20 |
| A-SAC | 0.12→0.73 | 0.10→0.92 | 0.11→0.38 |

Without seed 2, "PPO B 100 vs 1" becomes 8/23 (p=0.30) and "PPO gains more" reverses (SAC gains more on
18/23, p=0.003). Track-level p-values describe these three trained agents, not the algorithm. Claims that
hold on all 3 seeds: diversity helps SAC (both grids, both test sets) and PPO-B on vw_test. Claim 2 above
("above grid B SAC at every cell") also needs this caveat at d100 on real (2 of 3 PPO seeds are below all SAC seeds).
**RQ3 is inconclusive, not "PPO gains slightly more".**

**2. Most of the vw_test→real drop is lap length, not track difficulty.** Real circuits are not narrower
than vw_test (median half-width 1.11 vs 1.06 m; min clearance 1.08 vs 1.03 m) but are 2.4× longer
(median 403 vs 166 m). Grid B agents crash after about the same distance on both sets (B-SAC d100 median
59 vs 62 m; crash rate 92% vw_test, 99% real), so their fractional laps halve. Grid B SAC averages
~10 m/s vs ~5 m/s for grid A SAC. Since penalty is the only config difference between the SAC arms, this
measures (not just hypothesises) "penalty 5 → faster → crashes sooner". Correction to claim 4 above: the
"vw tracks are narrower on average" wording overstates a 5 cm difference; length is the real difference.

## Were the agents still learning at 2M? (2026-10-10, `_still_learning.py` in the bundle-verification folder)

- **Grid A SAC (best agent): probably not at its ceiling, but gains were small by then.** Training return at
  d100 kept rising (117 → 127 per episode from 1.25–1.5M to 1.75–2M; 2 of 3 seeds up). Validation mean peaked at the
  final checkpoint, but only +0.02 above 1.2M (0.735 → 0.752): seed 0 still rising (0.63 → 0.75), seed 1 at the
  validation ceiling (0.97 → 1.00), seed 2 noisy with no trend (0.61 → 0.50).
- **Grid B (both algorithms) plateaued by ~1.2–1.6M** (validation peaks at 1.2M for SAC, 1.6M for PPO; training return flat).
- **Grid A PPO never learned**; more steps would not help (earlier 5M test: plateau unchanged).
- **So "training longer would beat the gap-follower" is NOT a supportable claim** (gap on real: 0.68 vs 0.84; recent
  validation slope ~0.02 per 800k; no run went past 2M). Supportable instead: the best agent already beats the
  gap-follower on synthetic tracks (0.75 vs 0.68), and its validation-best seed beats it on real circuits (85% vs 72%
  laps); a budget sweep is future work.

## Extra insights found during the deck review (2026-10-10, `_verify/final_pptx_review/insights.py`)

- **Speed vs reliability.** On the 19 real circuits where both finish laps, the best agent (A-SAC d100, 3 seeds) laps
  **faster than the gap-follower on 17/19** (median lap time 0.72× the gap-follower's; Wilcoxon p = 2e-5). Caveats: the
  gap-follower is configured at 3–5 m/s (corners 3, straights 5), and only completed laps are timed. On vw_test the
  difference is small (faster on 8/13, median 0.95×).
- **Validation predicts test.** Across the 48 selected checkpoints, Spearman ρ = 0.98 (vw_val vs vw_test), 0.92 (vw_val vs
  real), 0.88 (vw_test vs real). The sim-to-real drop shifts every score down but barely changes the ranking.
- **Memorisation evidence.** A-SAC d1 survives the full 30 s in 66% of its late-training episodes on its own track
  (d100: 73% on its 100 tracks), yet scores 0.16 on vw_test and 0.11 on real.
- **Complementary strengths.** Best agent ≥50% where the gap-follower is <20%: MoscowRaceway, Sepang; the reverse: Shanghai.
  Picking the better of the two per track would give 83% laps (gap 72%, agent 47%). This is a hindsight upper bound,
  but it motivates the hybrid future-work item.
- Deck claims traced: slide 6 "0.55 → 0.49" = S0 vs S4 (time cost only, 0.487) ✓. Slide 7 "15%": the nearest numbers
  are 14.2% (B-PPO d100, real) and 13.6% (all grid-B PPO runs pooled, vw_test); it is an average, not a "best case" (best
  cell 24.2%).

## Deck v2 (2026-10-11)

`Group 4 Final Results v2.pptx` (18 slides) applies the corrections above and adds the RQ3 per-seed slide and the
sim-to-real slide. It is rebuilt reproducibly by `_verify/final_pptx_review/build_deck.py` (figures:
`make_figs.py`, from the verified bundle). The original team deck is not modified. What the team finally
presented (their merged deck, plus three fixes) is recorded in `final-presentation.md`.

### Correction: the widths in the 2026-10-10 follow-up (1.11 vs 1.06 m) used a different measure (2026-10-11)

Follow-up point 2 above compares real 1.11 m with vw_test 1.06 m. Those values came from the verification
JSONs' `median_wall_clearance_m` field, which reads 3–6 cm higher than the team's geometry method on every track
(median difference 0.038 m). Measured with the team's method for **both** sets — `analysis/track_geometry.py`
(distance transform of the occupancy map, bilinear, minus half a pixel, sampled every 0.1 m), which reproduces
`0_pilot/track_geometry/track_summary.csv` exactly on the two locally installed circuits (Spielberg 1.0741 vs
1.074, Silverstone 1.0107 vs 1.011):

| set | median of per-track median half-width | range | median of per-track minima |
|---|---|---|---|
| vw_test (20) | **1.02 m** | 0.63–1.45 | 1.00 m |
| real (23) | **1.07 m** | 0.68–1.27 | 1.055 m |

The conclusion is unchanged (real circuits are not narrower than vw_test; they are 2.4× longer). The final
deck's slide 13 uses 1.02 / 1.07. Script: `_verify/final_pptx_review/width_same_method.py`.

## Width and length *within* each test set (2026-10-11, `_verify/final_pptx_review/width_vs_score.py`)

Spearman ρ over tracks between a track's median half-width (same method as above) and the agent's per-track score
(d100 agents, 3 seeds averaged per track):

| agent | vw_test: width (n=20) | real: width (n=23) | real: lap length (n=23) |
|---|---|---|---|
| A-SAC d100 | +0.77 (p<0.001) | +0.35 (p=0.10) | −0.03 (p=0.89) |
| B-SAC d100 | +0.78 (p<0.001) | +0.33 (p=0.13) | −0.26 (p=0.23) |
| B-PPO d100 | +0.67 (p=0.001) | +0.31 (p=0.15) | −0.27 (p=0.21) |
| gap-follower | +0.09 (p=0.71) | +0.23 (p=0.28) | +0.34 (p=0.11) |

- **Width still matters track by track**: on vw_test, narrower tracks score clearly lower for every learned
  agent; on real circuits the link is weaker and not significant. The slow gap-follower hardly depends on width.
- **Width does not explain the synthetic → real drop**, because real circuits are not narrower.
- **The lap-length explanation rests on crash distance, not on a within-real correlation** (which is weak and
  not significant, as expected when real circuits also differ in width and corners). Say so if asked.

## Crash statistics at d100 (2026-10-11, from the raw test CSVs)

| agent | crash rate vw_test / real | median distance, crashed episodes (m) | median distance, all episodes (m) |
|---|---|---|---|
| A-SAC | 33% / 39% | 41 / 97 | — |
| A-PPO | 88% / 99% | 19 / 30 | — |
| B-SAC | 92% / 99% | 55 / 62 | **59 / 62** (slide 13 uses this definition) |
| B-PPO | 72% / 81% | 67 / 56 | 85 / 67 |

600 episodes (vw_test) and 690 (real) per agent, 3 seeds pooled. Both grid B agents crash after a similar distance
on both sets, so their share of a 2.4× longer lap falls. Grid A SAC crashes far less (39% vs 99% on real circuits) and
loses only 10%. For SAC the penalty is the only setting that differs between grids, so "penalty 5 → faster →
crashes far more" is measured for SAC. For PPO there is no working penalty-40 arm, so the same mechanism is not
shown for PPO.

## Other checks made during the final deck and talk-script review (2026-10-11)

All recomputed from the bundle's raw CSVs, selection files and run configs (journal Attempts 38–39).

- **RQ1 is "PPO is ahead under grid B's reward", not "PPO beats SAC".** Seed-matched (same seed, same d), B-PPO >
  B-SAC in **20/24** cells (vw_test 11/12, real 9/12). At d100 on real only seed 2 is ahead (0.67 vs 0.20; seeds
  0/1: 0.13 vs 0.19, 0.18 vs 0.20). Across grids, grid A SAC (0.68) beats grid B PPO (0.33) on real circuits.
- **The sim-to-real drop keeps the ranking, not every score.** 7 of 48 agents score *higher* on real than on
  vw_test (A-SAC d20 s2 0.55→0.60; A-SAC d100 s0 0.67→0.73; A-PPO d1 s2; B-SAC d1 s2; B-PPO d1 s0, d1 s2,
  d5 s1). Spearman vw_val→real 0.92 (see the insights section).
- **Multiple-testing correction.** The RQ3 track test (grid B, real) is p = 0.0179. Holm over the two grid-B RQ3
  tests (real, vw_test) gives **0.036**; Holm over the six real-circuit track tests also gives 0.036; adding the
  vw_test RQ3 test to those six (seven tests) gives **0.054**. The vw_test test (p = 0.33) is unchanged by Holm.
- **Checkpoint selection mattered.** 33/48 runs picked a checkpoint before the final one (grid A 16, grid B 17);
  in 31 of those 33 the chosen checkpoint beat the final one by more than 0.02 laps on vw_val (median margin 0.065).
- **Default PPO never drove.** Diagnosis run `V0_baseline`: 0 completed laps in 5,332 training episodes. Grid A
  PPO: 15 laps in 41,935 training episodes (14 in d20 s1, 1 in d100 s1); at test, 2 of 24 agent×set pairs
  completed any lap (d5 s0 real 1.7%, d20 s1 vw_test 1.0%).
- **Real circuits are gentler on average but have a tighter worst corner**: curvature p90 0.193 vs 0.405 /m,
  median minimum radius 0.9 vs 1.7 m (`0_pilot/track_geometry/track_summary.csv`). That this explains the
  slow gap-follower's advantage on real circuits is a reading, not a test.
- **Bonus finding holds on every seed**: A-SAC d100 real 0.73 / 0.92 / 0.38 vs B-SAC 0.19 / 0.20 / 0.20.
- **1-track agents**: A-SAC d1 completes 0% of real-circuit laps on all three seeds.
- **Runtimes** (`wall_seconds` in the 48 run configs): SAC 6.5–11 h (A), 5.7–8.3 h (B); PPO 1.5–4 h (A),
  5.4–6.5 h (B).
- **Replay buffer**: all 24 SAC runs use SB3's default `buffer_size` 1,000,000; SB3 stores obs and next_obs as
  float32, so 1080 beams + 5 state values = 8,700 B/transition = **8.70 GB**, and the wrapper's 108 beams (every 10th
  of 1080, `sb3_wrapper.py`) + 5 = 924 B = **0.92 GB**. The deck's earlier "~8.9 GB" was corrected to ~8.7 GB.
- **Gap-follower configuration** (`results/final_test/baseline_config_gap_real.json` on `final-grid-v2`): the same
  108 LiDAR beams as the agent (no map), corners 3 m/s, straights 5 m/s, bubble 16 beams, safe threshold 20 beams;
  source f1tenth_benchmarks FollowTheGap (Sezer & Gokasan 2012).

## RQ3: why it is inconclusive, and what the PPO tuning bias does to it (2026-10-11)

**Mostly noise, not the unfairness.** Per-seed gain from d1 to d100 (fractional laps):

| grid B | SAC seeds | PPO seeds | difference in means |
|---|---|---|---|
| vw_test | +0.27, +0.22, +0.33 | +0.27, +0.14, +0.56 | 0.05 (track test p = 0.33) |
| real | +0.09, +0.03, +0.10 | −0.05, −0.02, +0.42 | 0.05, all from seed 2 |

- Grid A cannot answer RQ3: default PPO never learned, so it gains nothing whatever the algorithm.
- In grid B the difference (≈0.05 laps) is buried in seed spread (up to 0.4 laps within PPO). With 3 seeds per
  side the smallest possible two-sided seed-level p (exact Mann–Whitney) is 0.10.
- **The answer depends on the measure.** Relative gains reverse the vw_test ordering:

  | grid B | gain in laps (SAC / PPO) | gain in percent (SAC / PPO) |
  |---|---|---|
  | vw_test | +0.28 / **+0.32** | **+192%** / +135% |
  | real | +0.07 / **+0.12** | +56% / +54% |

  We report laps (interpretable; percent misbehaves near zero and penalises the higher starter); the flip is part
  of the evidence that no consistent difference exists.
- **The tuning bias is a separate limit.** Penalty 5 was chosen for PPO and shrinks SAC's gain on real circuits
  (+0.07 in grid B vs +0.57 in grid A), so even a clean grid-B result would only hold under that reward. A real
  answer needs more seeds **and** equal-budget reward tuning per algorithm.
- Hence the conclusion wording: hypothesis **not supported** (SAC did not gain more where both learn), not refuted.

**Where the tuning bias sits, and the defence.** PPO got more tuning (~9 settings plus the fix, vs SAC's 6; tuning
rule: a new setting must win on 3 of 4 seeds, validation only) and grid B's penalty suits PPO. Disclosed on the
final deck's slides 6, 7, 8, 14 ("under that reward") and 15 (first limitation). It does not touch the within-
algorithm diversity results (RQ2), the strongest result (grid A SAC at defaults, 0.11 → 0.68), or the bonus
finding; the best agent is untuned SAC. It touches only the cross-algorithm comparison in grid B, where no winner
is claimed. Henderson et al. (2018, *Deep Reinforcement Learning that Matters*) is the standard citation for
hyperparameters and seeds flipping algorithm rankings; it was not in the final deck's reference list.

## Source reliability

- **Tables and checks: high.** Recomputed from the bundle's raw per-episode CSVs, selection files and run configs
  (scripts named in each section); the main numbers were re-checked on 2026-10-23 by `_verify/wiki_ingest/check_numbers.py`
  (119/119 match).
- **Mechanism readings: mixed.** "Penalty 5 — faster — crashes" is measured for SAC only; the curvature reading
  (Q20) and the replay-buffer reading (Q15) are untested.
