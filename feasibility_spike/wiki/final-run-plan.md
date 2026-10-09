---
title: Final comparison run — proposed configuration and environment
type: decision
updated: 2026-10-06
status: PROPOSED — not yet agreed by the team; two inputs still pending (G999_SAC result, presentation date)
sources:
  - train.py, sb3_wrapper.py, evaluate.py, tracks/manifest.json on team/crash-penalty-flag (high)
  - ppo-diagnosis.md, grid-verification.md, sim-to-real-width.md (synthesis)
  - results/ppo_diagnosis/runs_summary.csv — wall times (high)
---

# Final comparison run — proposed configuration and environment

The research question has not changed: **does training diversity (1/5/20/100
tracks) turn memorisation into transferable driving, and do PPO and SAC respond
differently?** Desmond's seed-0 grid answered it for SAC on synthetic tracks
(`grid-verification.md`). It could not answer it for PPO, because every PPO cell
was at 0 laps (`ppo-diagnosis.md`). This page proposes the run that can answer it
for both, and records what was chosen against and why.

## Readiness verdict

| component | ready? | evidence |
|---|---|---|
| Environment, reward, spawn, track pool | **yes** | canonical seed-123 pool reproduced byte-for-byte (manifest identical); eval bit-reproducible |
| PPO configuration | **yes at d1, with a stability risk; unverified at d5–100** | γ 0.999 + n_steps 8192 laps on 3/3 seeds by 1 M, but 1/3 collapsed by 2 M (`ppo-diagnosis.md`) |
| SAC configuration | **pending** | `G999_SAC` (does γ 0.999 help SAC?) ~12 h from 2026-10-06 20:00 |

> **2026-10-07 update (teammate tuning round, verified — `tuning-vw-val.md`):** γ 0.999 alone
> breaks SAC at d20 (0.246 vs baseline 0.509, collapse after 400 k), and adding the 0.01 time cost
> rescues the collapse but still scores below baseline (0.439). **Recommendation: SAC stays at γ 0.99.**
> Also at d20: PPO P1 (γ .999 + n_steps 8192) does not beat PPO defaults (0.077 vs 0.142), and the
> ent_coef+target_kl arm (P3) stabilised the optimiser but crawled (0.039). The PPO setting for the
> grid is therefore still unresolved; the d100 pilot remains the cheapest next test.
| Evaluation protocol | **needs two fixes** | single-checkpoint volatility; 3 distinct spawns per track |
| Real-circuit set | **needs install** | 2/23 installed locally; `tracks/install_real_tracks.py` exists |
| Compute | **tight for SAC** | SAC 2 M = 5.4 h (fast machine) to ~12 h (this one, loaded) |

**So: the PPO half can launch after a 1.3 h d100 pilot. The SAC half waits on one result.**

## Environment (frozen — identical to Desmond's seed-0 grid)

| setting | value | where |
|---|---|---|
| simulator | f1tenth_gym v1, `timestep 0.01` → **100 Hz**, no action repeat | `spike/f1tenth_gym_v1` |
| observation | 108-beam LiDAR (downsampled from 1080) + vehicle state; `VecNormalize(norm_obs=True, norm_reward=False)` | `sb3_wrapper.py` |
| action | [−1, 1]² → steering ±0.4189 rad, speed 0–20 m/s | `sb3_wrapper.py:233` |
| reward | `1.0 × metres of progress − 5.0 on collision`, `TIME_COST 0`, **no lap bonus** | `sb3_wrapper.py` |
| training episode | truncate at **3000 steps (30 s)**; `n_envs 1`; stratified track-pool sampler on reset | `train.py` |
| spawn | `cl_grid_static` (train and eval) | `methodology.md` |
| training pool | canonical seed-123 `synthetic_track_0..99`, manifest `d70dbc8f`; diversity 1/5/20/100 = first N; `track_seed 0` | `tracks/manifest.json` |
| budget | **2,000,000 env steps**; checkpoint every 100 k | `train.py` |

**Crash penalty: keep 5** (D6). Penalty 40 is what made PPO lap *at γ 0.99*, but
it was unstable (36% → 0%). At γ 0.999 the penalty is already worth 90× more
(5 × 0.999⁵⁰⁰ = 3.03 versus 0.033), and 5 matches the existing grid, the baselines and the slides.

## Algorithms

| | PPO | SAC |
|---|---|---|
| **γ** | **0.999** | **0.999 if `G999_SAC` is not worse than SAC γ 0.99** (rule below) |
| algorithm-specific change | **`n_steps 8192`** (disclosed; see why below) | none |
| everything else | SB3 2.9 defaults: batch 64, epochs 10, lr 3e-4, λ 0.95, clip 0.2, ent 0, net 64×64 | SB3 defaults: buffer 1 M, batch 256, lr 3e-4, τ 0.005, ent auto, net 256×256 |

```bash
python train.py --algo PPO --diversity {1,5,20,100} --seed {0,1,2} \
    --gamma 0.999 --n-steps 8192 --run-tag final
python train.py --algo SAC --diversity {1,5,20,100} --seed {0,1,2} \
    --gamma 0.999 --run-tag final            # or omit --gamma, per the rule below
```

**Why γ is shared and not per-algorithm.** γ defines *what is being optimised*
(`ppo-diagnosis.md`). If PPO optimises a 10 s horizon and SAC a 1 s horizon, an
algorithm comparison is confounded with an objective comparison.

**Why n_steps 8192 is acceptable for PPO only.** It is PPO's rollout length and
SAC has no equivalent. Without it PPO does not learn at γ 0.999 (`G999_ns2048`: 0%).
It must be **disclosed as the one algorithm-specific setting**, with the 2×2 as the
justification. SAC gets no matching tuning budget, so do not frame the result as
"tuned PPO vs tuned SAC".

**SAC γ decision rule (pre-registered).** Compare `G999_SAC` with Desmond's SAC d1
(γ 0.99) on the training-log lap rate in the 0.8–1.0 M and 1.8–2.0 M windows. If it is not
lower at both, use γ 0.999 for SAC. If it is lower, report that as a finding (the horizon fix
helps PPO and hurts SAC), and run SAC at both γ for d100 only.

## Evaluation (pre-registered before anyone sees a result)

| item | choice | why |
|---|---|---|
| sets | `test` (20 synthetic), `real` (23 circuits), `narrowA`, `narrowB` | the RQ, the sim-to-real result, and the width control |
| policy | deterministic; `--target-laps 1`; `--max-steps 15000`; `cl_grid_static` | as the existing grid |
| episodes | **10 per track** (seeds 0–9) | 5 seeds = 3 distinct spawns; the spawn grid caps at ~9 |
| **checkpoint rule (primary)** | **mean over the last 5 checkpoints (1.6–2.0 M)** | adjacent checkpoints swing 0 ↔ 10/25 (`ppo-diagnosis.md`); averaging made the seed-0 grid monotone |
| checkpoint rule (secondary) | best of the last 10 checkpoints on `val` (10 tracks) | the frozen D8 rule; report it, but it is noisier |
| primary metric | **lap completion rate**; also mean laps and metres | binary hides nothing here because rates are 0–70% |
| aggregation | per track, then across tracks; 95% CI by hierarchical bootstrap (seeds, then tracks) | n = 3 seeds × 20 tracks |
| baselines | gap-follower + random on the same sets (exist in `results/grid_s0`) | — |
| also report | training-log lap rate (stochastic), labelled as such | it runs 2–5× higher than deterministic held-out |

## Scope and cost

| scope | runs | CPU-hours | wall-clock |
|---|---|---|---|
| PPO 4 div × 3 seeds | 12 | ~13–16 h (1.1–1.3 h each under load) | ~4 h at 4 parallel |
| SAC 4 div × 3 seeds | 12 | ~65 h (fast machine) to ~140 h (this one) | **days on one PC — split across teammates** |
| optional width arm: d100 × randomised width × 2 algos × 3 seeds | 6 | ~35–75 h | needs a generator width flag first |

Splitting SAC across machines is safe: the seeds are independent,
`diagnostics/build_canonical_tracks.py` reproduces the pool exactly, and
`run_config.json` records the versions. Training is **not** bit-reproducible
across machines, but that does not matter when every run is its own seed.

## Pre-flight checklist (in order)

1. **Pilot PPO d100, γ 0.999 + n_steps 8192, seed 0 (~1.3 h).** If it laps in training, launch the PPO grid.
2. **Wait for `G999_SAC`** (or stop it at 1 M and compare windows) → apply the SAC γ rule.
3. `python tracks/install_real_tracks.py` + `verify_real_tracks.py` on every machine that evaluates.
4. **Confirm the final-presentation date.** It decides full SAC grid vs SAC d1+d100 only.
5. Commit this page as **agreed**, with any changes, before the first grid run starts.

> **Status update 2026-10-09** (supersedes items 1–2 above; nothing ticked yet):
> - **Item 2 replaced:** `G999_SAC` died at ~450 k. The clean test is running instead: Desmond's S1
>   with penalty 5 (`tune_S1P5_s99/s98`, vw d20) — decides whether SAC can share γ 0.999 at penalty 5.
> - **Item 1 updated:** PPO config is now γ 0.999 + n_steps 8192 + **ent_coef 0.01** (stability A/B,
>   `ppo-diagnosis.md`), validated only on the wide d1 track at penalty 5. The pilot must run on the
>   pool the grid will use (varied-width, d20 — directly comparable to Desmond's P0–P3).
> - **New shared decisions (both algorithms, every cell):** crash penalty 5 vs 40 (Desmond's tuning and
>   batch 2 use 40; the seed-0 grid and every working PPO run used 5); track pool varied-width vs uniform.
> - Items 3–5 unchanged and still open.

> **Team decision found 2026-10-09 (supersedes this PROPOSED page for the main grid):** Desmond's
> `final_grid/DECISION.md` (branch `final-grid`, commit `7552398`) applied a pre-registered tuning-round-3
> rule → **SAC and PPO at SB3 defaults, penalty 40, no time cost**, varied-width tracks; 24 runs split
> (Desmond SAC s0–1, Grant SAC s2, Chris PPO s0–2). Penalty 5 and the entropy bonus were not candidates.
> Under that rule's scoring, PPO defaults average 0.124 (4 seeds, 0% laps); our σ-fix at penalty 5 scores
> 0.498 / 0.672 (2 seeds, 30–32% laps). Options put to the team: A keep as decided; B change the grid
> (needs SAC-at-penalty-5 evidence + disclosure); C keep the grid and add a fixed-PPO arm at penalty 5
> (12 PPO runs, within-PPO claims only). Aolin suggests C.

## Alternatives considered

| alternative | rejected because |
|---|---|
| Keep γ 0.99 for both (the pure "defaults" grid) | already run (seed 0): PPO is 0 laps everywhere, so PPO's diversity curve is unmeasurable |
| Each algorithm at its own best γ | confounds algorithm with objective |
| Penalty 40 | unstable for PPO (36% → 0%); the SAC benefit is confounded with code path |
| Action repeat ×10 | refuted: 0 laps with 5× the experience; costs 90% of decisions |
| 5 M budget | PPO plateau unchanged at 2.5 M; SAC cost doubles |
| Final checkpoint only | demonstrably volatile; caused the "d20 > d100" artifact |
| Train on real circuits | would destroy the zero-shot question |

## See also

- `ppo-diagnosis.md` — why γ 0.999 + n_steps 8192
- `grid-verification.md` — the seed-0 grid this extends
- `sim-to-real-width.md` — why real circuits will stay ~0% unless width is randomised
- `decisions.md` — D6, D8, D12–D15
- `open-questions.md` — Q1, Q3, Q10, Q11, Q14
- `methodology.md` — the protocol this extends
