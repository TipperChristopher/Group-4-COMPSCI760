---
title: Open questions and unfinished work
type: open-question
updated: 2026-10-23
sources:
  - G4_full_bundle_2026-10-10.zip (final grids + test, high) and final-results.md (synthesis)
  - _verify/s1p5/results/tune_S1P5_s{98,99}_*_vw_val.csv, logs/*.out.txt (high, Q11 final status)
  - team_repo/feasibility_spike/reward_experiment/RESUME.txt (high)
  - results/*.json (high)
  - results/ppo_diagnosis/**, results/grid_s0/** (high, 2026-10)
  - ppo-diagnosis.md, grid-verification.md, sim-to-real-width.md, final-run-plan.md (synthesis)
---

# Open questions

What is **not** answered yet, and what has since been answered or refuted.
Nothing here is stated as a result unless it is marked ANSWERED and points to a source.

## Status (2026-10-23, after the final test)

| Q | question | status |
|---|---|---|
| Q1 | The diversity grid | **ANSWERED** — two grids × 24 runs, 3 seeds, tested once (`final-results.md`) |
| Q2 | Confidence intervals | **PARTLY** — seed SDs + track-level Wilcoxon (Holm); a seed-level test cannot reach p < 0.05 with 3 seeds → Q17 |
| Q3 | Crash penalty 5 vs 40 | **RESOLVED by running both** (grid A 40, grid B 5); the penalty is algorithm-specific |
| Q4 | Overfitting over time | **ANSWERED (no)** |
| Q5 | Did the reward matter for SAC? | **PARTLY** — the crash penalty matters a lot (40 → 5: real 0.68 → 0.20); V1 vs V2 untested |
| Q6 | Narrow tracks | **DONE via Q13** |
| Q7 | Corner-sharpness cap | **REFUTED** |
| Q8 | More envs for PPO | **SUPERSEDED** |
| Q9 | Merge and freeze before the grid | **DONE** — frozen on `final-grid` / `final-grid-v2` (not merged to `main`) |
| Q10 | PPO fix at d5/20/100 | **ANSWERED** — grid B: works on vw_test (0.24 → 0.57 by d20), noisy on real circuits |
| Q11 | γ 0.999 for SAC | **ANSWERED (no)** — breaks SAC at penalty 40; at penalty 5 no collapse but no better than γ 0.99 |
| Q12 | Why γ and n_steps interact | **OPEN** (hypothesis only; state it as such in the report) |
| Q13 | Width-randomised pool | **DONE** — vw pool (0.60–1.50 m): pilot real laps 0% → 37%; narrow < 0.9 m still hard |
| Q14 | Presentation date and scope | **DONE** — full grid run; presented after 2026-10-11 |
| Q15 | Why did the replay-buffer hypothesis fail? | **OPEN** (new) |
| Q16 | Equal-budget reward tuning per algorithm | **OPEN** (future work) |
| Q17 | More seeds and start poses | **OPEN** (future work) |
| Q18 | Budget sweep beyond 2M | **OPEN** (future work) |
| Q19 | Hybrid: agent + gap-follower fallback | **OPEN** (future work) |
| Q20 | Why the gap-follower does better on real circuits | **OPEN** (new; reading only) |

## Status (2026-10-06, historical)

| Q | question | status (2026-10-06) |
|---|---|---|
| Q1 | The diversity grid | **PARTLY RUN** — seed 0, γ 0.99 (Desmond); final grid proposed |
| Q2 | Confidence intervals | **OPEN** — needs ≥3 seeds |
| Q3 | Crash penalty 5 vs 40 | **RECOMMENDATION: 5** — team decision pending (D6) |
| Q4 | Overfitting over time | **ANSWERED (no)** — extended to 2 M; volatility, not decline |
| Q5 | Did the reward matter for SAC? | **OPEN**, low priority |
| Q6 | Does adding narrow tracks fix real circuits? | **REFRAMED** → Q13 (width-randomised pool) |
| Q7 | Corner-sharpness cap | **REFUTED** (`sim-to-real-width.md`) |
| Q8 | Would PPO stabilise with more envs? | **SUPERSEDED** — PPO fixed by γ + n_steps, n_envs stays 1 |
| Q9 | Merge and freeze before the grid | **OPEN** — branch state changed, see below |
| Q10 | Does the PPO fix hold at d5/20/100? | **d20 varied-width: YES at penalty 5** (best 0.52–0.67 vs 0.03–0.13 at penalty 40, 2 seeds); d5/100 still open |
| Q11 | Does γ 0.999 help or hurt SAC? | **ANSWERED for d20 at penalty 40: it breaks SAC**; at penalty 5 **RUNNING** (`tune_S1P5_s99/s98`, launched 2026-10-09) |
| Q12 | Why do γ and n_steps interact? | **OPEN** (hypothesis only) |
| Q13 | Does a width-randomised pool fix real circuits? | **OPEN** — needs a generator width flag |
| Q14 | Final-presentation date and scope | **OPEN** — needed to size the SAC half |

## Q1 — The diversity grid

> **ANSWERED (2026-10-10).** Grid A (defaults, penalty 40) and grid B (penalty 5 + PPO fix), 4 levels × 2
> algorithms × 3 seeds each, tested once on vw_test + 23 real circuits. Diversity helps every learning agent on
> synthetic tracks on every seed; grid A SAC real 0.112 → 0.678 (`final-results.md`). The history below is as written.

*Sept 2026:* not run. *2026-10-05:* Desmond ran seed 0 for both algorithms × 1/5/20/100 at
2 M with γ 0.99, plus baselines and narrow sets, and it was verified (`grid-verification.md`).
SAC shows the diversity effect on synthetic tracks; PPO is 0 laps in every cell, so PPO's diversity
curve is unmeasurable under γ 0.99. **The final grid (≥3 seeds, fixed PPO, averaged
checkpoints) is proposed in `final-run-plan.md` and not yet agreed.**

## Q2 — Confidence intervals / rankings

> **Partly answered (2026-10-10).** The final grid has 3 seeds per cell: results are reported as seed means ±
> SDs, every seed shown, plus track-level Wilcoxon signed-rank tests with Holm correction. Those tests describe the
> three agents in a cell, not the algorithm; with 3 vs 3 seeds the smallest two-sided seed-level p is 0.10. A
> hierarchical analysis (seeds, then tracks) needs more seeds (Q17).

Every training run is still n=1 seed per cell, except the PPO d1 fix (3 seeds,
training log). Adjacent diversity levels are mostly not significantly different at seed 0.
Any "A beats B" needs ≥3 seeds and a hierarchical bootstrap (seeds, then tracks).

## Q3 — Crash penalty freeze (5 vs 40)

> **RESOLVED (2026-10-09/10).** Both were run, one per grid (D16, D17). 40 suits SAC (real d100 0.68 vs 0.20 at 5);
> 5 suits PPO (grid A PPO ≈0.1 at 40). A single shared value cannot be neutral between the algorithms.

Evidence updated in `decisions.md` D6. Recommendation: **5**. Penalty 40 made PPO lap at
γ 0.99, but then it collapsed. γ 0.999 makes penalty 5 matter by a factor of 90. The SAC p40
advantage is confounded with code path.

## Q4 — Overfitting timing — ANSWERED (no)

Checkpoints 100k–1M rose monotonically (Sept 2026). `sac_overfit` finished 2 M: its in-training
"dips" were single-episode samples of a ~27% failure rate (30-spawn eval: 1.2 M 30/30, 2 M 22/30).
For the fixed PPO, held-out performance is **volatile** across checkpoints rather than declining
(`ppo-diagnosis.md`). Desmond's SAC **d1** does decline after 1 M in training (38% → 22%), the
one overfitting-like signal, and it occurs only at diversity 1.

## Q5 — Did the reward matter for SAC?

SAC was only ever trained on V2. Unmeasured, and no longer on the critical path.

> **Partly answered (2026-10-10).** Within V2, the crash penalty matters a great deal for SAC: penalty 5 instead of
> 40 roughly doubles its speed and cuts real-circuit laps from 0.68 to 0.20, on every seed (`final-results.md`).
> V1 vs V2 for SAC remains untested.

## Q6 — Adding narrow tracks → see Q13

The idea is right, and the variable is now confirmed to be width (`sim-to-real-width.md`).
Rather than add a few narrow tracks, randomise width across the whole pool.

## Q7 — Corner-sharpness cap — REFUTED

Real circuits are sharper than training at only 0.95% of points and are straighter at p90.
Width is out of distribution at 100% of points (`sim-to-real-width.md`).

## Q8 — More envs for PPO — SUPERSEDED

PPO's failure was the objective (γ) plus rollout length, not a lack of parallel envs. `n_envs` stays 1
(Bug 1 in `incidents.md`).

## Q9 — Merge and freeze before the grid

> **DONE (2026-10-08/09).** Grid A was frozen on `final-grid` (`7552398`), grid B on `final-grid-v2` (`985b4bb` →
> `16440e6`), each with a setup checker and a reference fingerprint. Nothing is merged to `main`.

Branch state (2026-10-06): `experiments/feasibility-spike` (September-2026 work) →
`desmond/heldout-tracks` (canonical tracks, grid tooling, 23 real circuits, narrow sets) →
`team/crash-penalty-flag` (train.py flags, diagnosis, this wiki update; pushed). None of
these are merged to `main`. Freeze the final configuration (`final-run-plan.md`) on one branch
before launching.

## Q10 — Does γ 0.999 + n_steps 8192 work at higher diversity?

> **ANSWERED (2026-10-10).** In grid B (with ent_coef 0.01, penalty 5) PPO learns at every level: vw_test 0.239 /
> 0.403 / 0.570 / 0.560 (d1/5/20/100), lap completion up to 24%. On real circuits it is noisy: 0.213 / 0.304 / 0.265 /
> 0.330, and the d100 gain rests on one seed. The 2026-10-07 negative d20 point below was at penalty 40.

Only d1 has been tested. **New data point (2026-10-07, teammate's verified tuning round,
d20 varied-width, 2 seeds): the fix does NOT beat PPO defaults there** — P1 0.077 vs P0 0.142
(best-checkpoint fractional laps, `tuning-vw-val.md`). Note the differences from our d1
validation: 20 training tracks (not 1), varied widths 0.63–1.44 m, fractional-lap metric,
and only 2 seeds. So the d1 fix has not transferred yet — do not claim it generalises.
The d100 pilot question is unchanged (each track gets 20 k steps instead of 2 M), but a
single d20 result now exists and is negative.

## Q11 — γ 0.999 for SAC

> **ANSWERED (no), 2026-10-23.** The penalty-5 test (`tune_S1P5_s99/s98`) never collapsed, but it never completed a
> vw_val lap either. Best score through 1.6M: 0.243 / 0.300 (0% laps at 400k, 800k, 1.2M and 1.6M). Both runs
> stopped at ~1.81M / 1.82M without a final checkpoint (logs end there, no "Training complete"), most likely
> stopped to free the machine for the final grids. For comparison, grid B SAC d20 (γ 0.99, penalty 5) best
> vw_val: 0.364 / 0.383 / 0.253, and grid A SAC d20 (γ 0.99, penalty 40): 0.302 / 0.536 / 0.583. Seeds and
> machines differ, so this is indicative. γ 0.99 was locked for SAC in grid B (D17) regardless of this test.

**Answered for d20 (2026-10-07): γ 0.999 alone breaks SAC.** In the teammate's verified tuning
round, S1 (γ 0.999) scored 0.246 against the S0 baseline's 0.509, collapsing after 400 k with a
critic loss spike to ~3789 (`tuning-vw-val.md`). The time-cost variant (S2) rescues it from
collapsing but still scores below the γ 0.99 baseline (0.439). *Correction (2026-10-09):* `G999_SAC` (our d1 run, penalty 5) was **not** still running — it died
at ~450 k steps on 2026-10-06 ~22:05 (its run_config still says `running`). Up to then: 0% laps,
8.6–10 m/s, no crawling — too early to judge (the teammate's S1 also looked fine at 400 k).
SAC γ 0.999 at penalty 5 is therefore untested; the final-run plan recommends SAC γ 0.99.

**Clean test launched 2026-10-09 13:56:** Desmond's S1 exactly (commit `6e5b60e`, vw d20, seeds 99/98,
γ 0.999), only `--crash-penalty 5` changed. Worktree `team_repo_tuning`; vw tracks rebuilt with his
`make_vw_tracks.py` — all 130 manifest entries identical to his committed manifest; smoke test: crash pays
exactly 5. Checkpoints scored with his queue's protocol (`evaluate.py --track-set vw_val --episodes 5
--max-steps 15000`, 400k/800k/1.2M/1.6M/final) by `_verify/s1p5/eval_watcher.py`. S1 collapsed between
400 k and 800 k, so ~800 k (~4 h) is the first decisive point. Reading: no collapse → penalty 40 was the
problem at γ 0.999; collapse → γ 0.999 itself breaks SAC.

**Interim at 800 k (2026-10-09 18:30):** no collapse. Scores 0.185 → 0.182 (s99) and 0.250 → 0.287 (s98)
vs Desmond's S1 at penalty 40: 0.277 → 0.017 and 0.214 → 0.008. Critic loss stays 0.05–0.07 through
400–800 k (his blew up to ~21). Training speed 8.7–8.9 m/s, no crawl. Not yet better than SAC γ 0.99
(his S0 at 800 k: 0.283 / 0.353); 0% laps so far. Decision rule needs the best checkpoint by 2 M.

## Q12 — Mechanism of the γ × n_steps interaction

Hypothesis: the γ 0.999 critic regresses ~1000-step returns, and its explained variance falls
from 0.96 to 0.69, so it needs more data per update. Test: γ 0.999 with n_steps 2048 and more value
epochs or a separate value learning rate. It matters for the write-up, not for the grid.

## Q13 — Width-randomised training pool

> **DONE (2026-10-06, Desmond).** Varied-width splits vw_train / vw_val / vw_test (half-width 0.60–1.50 m, commit
> `eaaf143`). Pilot SAC d100 (seed 0, penalty 40) went from 0% to 37% of real-circuit laps (narrowA 0% → 68%);
> narrowB (median 0.75 m) stayed at 2%. All final runs use the vw pool (`sim-to-real-width.md`).

Add a half-width parameter to the generator, sample 0.7–1.5 m per track, retrain SAC (and PPO)
at d100, then evaluate on narrowA/B and the 23 real circuits. Prediction if width binds:
a large real-circuit jump. Cost: generator change plus about 6 runs.

## Q14 — Presentation date and scope

> **DONE.** The full 4 × 3 grid ran for both algorithms in both grids, split across three PCs; the talk was
> due shortly after 2026-10-11 (`final-presentation.md`).

Never confirmed in the logs. It decides whether SAC runs the full 4 × 3 grid
(~65–140 CPU-hours, so it must be split across machines) or d1 + d100 only.

## Q15 — Why did the replay-buffer hypothesis fail? (new, 2026-10-11)

The hypothesis was that SAC benefits more from diversity because its 1M-transition replay buffer keeps experience from
many past tracks, while PPO learns only from its latest rollout. It was not supported (RQ3). An untested reading: a
new track is drawn every episode, and each 8,192-step PPO rollout spans at least three episodes (an episode is at most
3,000 steps), so PPO also sees several tracks per update. A test would vary rollout length at fixed diversity, or
log the number of distinct tracks per update.

## Q16 — Equal-budget reward tuning per algorithm (future work)

Grid B's penalty was chosen for PPO and never tuned for SAC. A fair RQ3 needs each algorithm's reward tuned with the
same budget and rule, then the diversity grid repeated per algorithm.

## Q17 — More seeds and start poses (future work)

3 seeds per cell cannot support a seed-level test, and each test track has only 5–6 distinct start poses (110 / 137
distinct across vw_test / real). 5–10 seeds, more poses, and a model that treats seeds as the unit.

## Q18 — Budget sweep beyond 2M (future work)

Grid A SAC d100's training return was still rising at 2M, but validation gained only 0.02 from 1.2M to 2M; grid B
plateaued by 1.2–1.6M. A 1M–10M sweep would show whether the gap to the gap-follower closes. Do not claim it would.

## Q19 — Hybrid agent + gap-follower fallback (future work)

Picking the better of the best agent and the gap-follower per real circuit gives 83% laps (agent 47%, gap 72%),
a hindsight upper bound. A switching rule (e.g. fall back near narrow sections) would need its own validation.

## Q20 — Why does the gap-follower do better on real circuits? (new, 2026-10-11)

It scores 0.68 on vw_test but 0.84 on real circuits. Real circuits are gentler on average (curvature p90 0.19 vs 0.40 /m)
but have a tighter worst corner (median minimum radius 0.9 vs 1.7 m). A slow, map-free rule may profit from the gentle
average while a fast learned policy is punished by the worst corner. Reading only; untested.

## See also

- `final-run-plan.md` — Q1, Q3, Q10, Q11, Q14 resolved into one plan
- `ppo-diagnosis.md` — Q8, Q10, Q12
- `sim-to-real-width.md` — Q6, Q7, Q13
- `decisions.md` — D6 (Q3), D13, D14
- `overview.md`, `experiments.md`, `handoff.md` — context and commands
- `final-results.md` — the evidence behind the 2026-10-23 statuses
- `report-handoff.md` — Q12 and Q15–Q20 as the report's limitations and future work
