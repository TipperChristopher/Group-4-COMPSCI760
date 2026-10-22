---
title: Sim-to-real failure is corridor width (controlled ablation)
type: concept
updated: 2026-10-23
sources:
  - G4_full_bundle_2026-10-10: 0_pilot/grid_s0_r40{,_vw}/SAC_d100_*.csv, 0_pilot/track_geometry/track_summary.csv, 4_final_test/ (high)
  - _verify/final_pptx_review/width_same_method.py, width_vs_score.py (high)
  - results/grid_s0/{SAC_d20,SAC_d100,gap}_{test,narrowA,narrowB,real}.csv (high, Desmond)
  - results/track_geometry/summary.json, track_summary.csv (high, Desmond)
  - tracks/make_narrow_tracks.py (high)
  - results/ppo_diagnosis/eval_width_oursac/oursac{1200k,2000k}_*.csv (high)
  - run journal Attempt 8 (mixed)
---

# Sim-to-real failure is corridor width

Every policy trained in this project (both algorithms, every diversity level, every penalty)
completes **0%** of laps on real circuits. On 2026-10-06 a controlled ablation
showed why: **the training generator produces one constant corridor width, and
real circuits are narrower.** If only the walls of the exact held-out test
centrelines are moved inward, real-circuit-like failure appears with the corners
unchanged. Earlier work (the September-2026 *nar7* probe) pointed in this direction but
could not separate width from shape; this ablation does.

> **Update (2026-10-23).** The statement above was true of every policy trained before 2026-10-06. The fix was then
> applied: Desmond's varied-width pool (half-width 0.60–1.50 m) took the pilot SAC from 0% to 37% of real-circuit
> laps, and in the final test grid A SAC d100 completed 47% (3-seed mean; 85% for the best seed) against the
> gap-follower's 72%. The remaining synthetic → real drop is **not** width (real circuits 1.07 m vs synthetic test
> 1.02 m, same method) but lap length (2.4×). See the last section.

## The ablation

`tracks/make_narrow_tracks.py` (Desmond) builds `narrowA_track_k` and
`narrowB_track_k` from **the exact centreline of `test_track_k`**, so shape,
length and curvature are identical. Only the half-width changes: test **1.473 m**,
narrowA **1.072 m** (≈ real median), narrowB **0.751 m**.

| policy | test (1.47 m) | narrowA (1.07 m) | narrowB (0.75 m) | real circuits |
|---|---|---|---|---|
| gap-follower (no learning) | 160.5 m · **71%** | 117.0 m · **36%** | 118.2 m · 29% | 325.8 m · 70% |
| SAC d20 (Desmond, p5) | 109.5 m · 31% | 38.7 m · **0%** | 5.7 m · 0% | 50.0 m · 0% |
| SAC d100 (Desmond, p5) | 120.9 m · **38%** | 37.9 m · **0%** | 9.7 m · 0% | 38.6 m · 0% |
| SAC `sac_overfit` 1.2 M (ours, p40) | 157.0 m · **29/50** | 19.1 m · **0/50** | 8.0 m · 0/50 | 31.7 m · 0/20 |
| SAC `sac_overfit` 2 M (ours, p40) | 141.8 m · 27/50 | 35.1 m · 0/50 | 11.3 m · 0/50 | 16.9 m · 0/20 |

- **Moving only the walls takes every SAC policy from 29–58% lap completion to 0%.**
- narrowA distance (19–39 m) sits right next to real-circuit distance (17–50 m),
  so width alone reproduces the real failure.
- **Feasibility control:** the hand-written gap-follower still laps 36% of narrowA,
  so the narrow tracks are drivable. What collapses is the learned policy.

## Why — the training distribution has no width variance

From `results/track_geometry/summary.json`:

| statistic | training pool (all 100) | real circuits |
|---|---|---|
| half-width range | **1.430–1.474 m** (a 4 cm span) | median 1.075, min 0.647 |
| share of points narrower than anything in training | — | **100%** |
| curvature p90 | 0.408 | 0.195 (straighter) |
| share of points sharper than anything in training | — | **0.95%** |

**Width diversity is zero at every diversity level**, so a bigger pool from the same
generator cannot teach width. That is why the diversity grid improves held-out synthetic
transfer (`grid-verification.md`) but not real circuits.

### Correction: "corner sharpness is the second ceiling" (Q7, INVESTIGATION.md fix #1)
We previously ranked "widen `TRACK_TURN_RATE` so synthetic corners reach real
curvature (max 1.27)" as the top next experiment. The geometry analysis refutes
it. Real circuits are out of distribution on curvature at only 0.95% of points, and are
straighter on average. The max-curvature comparison in `experiments.md` (synthetic 0.544 vs
Spielberg 1.273) was a statement about the single sharpest point, not about the distribution.
Width is out of distribution at 100% of points.

### Relation to the nar7 finding (2026-09)
nar7 showed that a synthetic track with real-like spawn clearance reproduced the
creep-then-crash, but two nar7 tracks with the *same* clearance drove fine,
so clearance looked "necessary but not sufficient". Those nar7 tracks came from a
different generator width *and* different centrelines, and were evaluated on 5 seeds
(3 distinct spawns). The narrowA/B ablation holds the centreline fixed and evaluates
40 tracks, and it gives a clean answer: width is sufficient to cause the failure.

## What would fix it (untested at the time; tested since — see the last section)

Train on a pool whose half-width is **randomised** (e.g. uniformly 0.7–1.5 m per
track, or varying along a track), then evaluate on narrowA/B and the 23 real circuits.
Prediction if width is the binding constraint: a large jump on real circuits with no
real circuit ever seen in training. The generator needs a width parameter (not written yet).
This was the user's original "mock real-like tracks" idea, now aimed at the confirmed variable.

## Caveats

n = 1 training seed per model. Desmond's narrow evaluations use 5 episodes per track
(≈3 distinct spawns); ours use 10. The two SAC families differ in penalty and code path,
yet show the same pattern, which makes the result robust to those differences.


## Update 2026-10-09: policies trained on the varied-width pool, broken down by width

Training on `vw_synthetic_track_*` (d20, half-widths spread 0.60–1.50 m) and scoring on the unseen
`vw_val` tracks (half-widths 0.66–1.39 m; narrow = t1/t3/t7/t9, mid = t0/t4/t5/t8, wide = t2/t6).
Lap completion, checkpoints 1.6 M + final, 2 seeds (200 episodes per row):

| policy | narrow < 0.9 m | mid 0.9–1.2 m | wide > 1.2 m |
|---|---|---|---|
| SAC γ 0.99, penalty 40 (Desmond S0) | 0% | 20% | 45% |
| PPO γ 0.99, penalty 40 (Desmond P0) | 0% | 0% | 0% |
| PPO γ 0.999 + n_steps 8192, penalty 5 (control) | 2% | 26% | 0% |
| PPO + ent_coef 0.01, penalty 5 (σ fix) | **9%** | **34%** | 25% |

- Varied-width training makes **mid-width** unseen tracks lappable for both algorithms — the range that
  contains the real-circuit median (1.07 m).
- **Narrow tracks (< 0.9 m) remain mostly unsolved** (0–9%), even though narrow widths are inside the
  training range — narrow is genuinely harder, not only out-of-distribution.
- Small n: 4 narrow and 2 wide tracks, 3 distinct spawns each; different penalties across rows.
- **Real circuits have not been evaluated for any varied-width-trained model** (Desmond's tuning
  deliberately held out real/test; only Silverstone and Spielberg are installed on this machine).

## Final picture (2026-10-10/11): what width explains and what it does not

**The width fix, pilot SAC d100 (seed 0, penalty 40)** (`0_pilot/grid_s0_r40/` vs `0_pilot/grid_s0_r40_vw/`):

| training pool | synthetic test (1.47 m) | narrowA (same centrelines, 1.07 m) | narrowB (0.75 m) | real circuits |
|---|---|---|---|---|
| fixed width 1.47 m | 0.959 (92% laps) | 0.168 (0%) | 0.056 (0%) | 0.082 (0%) |
| varied width 0.60–1.50 m | 0.897 (76%) | 0.885 (68%) | 0.372 (2%) | 0.676 (37%) |

**After the fix, real circuits are not narrower** than the varied-width test tracks. Same method for both
(`analysis/track_geometry.py`): median half-width vw_test 1.02 m (0.63–1.45), real 1.07 m (0.68–1.27). They are
2.4× longer (median 403 vs 166 m), gentler on average (curvature p90 0.19 vs 0.40 /m) and have a tighter worst corner
(median minimum radius 0.9 vs 1.7 m).

**Width still predicts per-track difficulty on synthetic tracks** (Spearman ρ 0.67–0.78 for the d100 agents, p ≤ 0.001) but
only weakly on real circuits (ρ ≈ 0.3, n.s.); the gap-follower hardly depends on width.

**What explains the drop instead** (`final-results.md`): grid B agents crash after about 60 m on both test sets, so their
share of a 2.4× longer lap falls (B-SAC −53%, B-PPO −41%); grid A SAC crashes far less (39% vs 99% of real-circuit
runs) and loses only 10%. Narrow tracks (< 0.9 m) remain mostly unsolved.

### Correction: "real circuits are narrower" in the 2026-10-10 deck draft

The first deck draft said the drop happened because real circuits are narrow. After the width fix that is false (see
above); the claim was removed (`final-presentation.md`).

## See also

- `final-results.md` — the final test, crash statistics and width-vs-score table
- `experiments.md` — the earlier nar7 probe and geometry table
- `grid-verification.md` — diversity helps synthetic transfer, not real
- `open-questions.md` — Q6/Q7 status, width-randomised pool
- `final-run-plan.md` — where the width arm fits
- `glossary.md` — half-width, narrowA/B
