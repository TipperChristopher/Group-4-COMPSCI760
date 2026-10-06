---
title: Why PPO never completed a lap — the diagnosis trail
type: synthesis
updated: 2026-10-06
sources:
  - results/ppo_diagnosis/runs_summary.csv, runs/*/progress.csv, runs/*/monitor_0.monitor.csv (high)
  - results/ppo_diagnosis/eval_sweep_1M, eval_ck800k, eval_ck2M, eval_g999_curve (high)
  - diagnostics/basin_check.py — output reproduced below (high, n=1 episode per policy)
  - sb3_wrapper.py reward constants; train.py; action_repeat.py (high)
  - CONTEXT.md (repo root) — team-facing summary of the same work (mixed)
  - pi session 2026-10-03→06 + run journal, see raw-sources/index.md (mixed)
---

# Why PPO never completed a lap — the diagnosis trail

PPO completed **zero laps** in every run anyone on the team had trained:
0/860 evaluation episodes and ~24,000 training episodes across Desmond's four
seed-0 grid cells, plus every pilot of ours. On 2026-10-06 we found why.
**Two things were wrong at once, and fixing either one alone does nothing:**

1. **The objective preferred crashing.** At 100 Hz control, the default
   discount γ=0.99 means the agent values roughly 1 second of the future. Under
   that objective, *drive fast and crash* scores **higher** than *drive slower
   and finish* — computed by hand and then measured on trained policies.
2. **The optimiser collapsed.** With the default rollout of 2048 steps,
   PPO's exploration noise collapses to σ≈0.06 by ~0.5 M steps and its updates
   overshoot the trust region (approx-KL ~10× normal).

With **γ=0.999 and `n_steps=8192` together**, PPO laps in **52–65% of training
episodes across three seeds** at 1 M steps (diversity 1). This page records the
whole trail — including the hypotheses we got wrong, because several of them
were plausible and a teammate could easily re-propose them.

## The setting that matters

| quantity | value | source |
|---|---|---|
| control rate | **100 Hz** (`timestep 0.01`, no action repeat) | `f110_env.py`, `sb3_wrapper.py` |
| reward | metres of centreline progress − **5** on collision; `TIME_COST 0` | `sb3_wrapper.py` |
| lap bonus | **none** — `train.py` never sets `target_laps`; finishing is not rewarded | `train.py` |
| training episode cap | 3000 steps = **30 s** | `train.py` |
| γ (SB3 default) | 0.99 → value horizon 1/(1−γ) = 100 steps = **1.0 s** | SB3 |
| GAE window, λ=0.95 | 1/(1−γλ) ≈ 17 steps ≈ 0.17 s ≈ 1.7 m of track | SB3 |
| canonical track length | `synthetic_track_0` = 164.1 m | track files |

## The hypothesis trail

Each row is one hypothesis, the experiment that tested it, and the verdict.
All runs are diversity 1, seed 0, 2 M steps, canonical pool, unless stated
(`results/ppo_diagnosis/runs_summary.csv`).

| # | hypothesis | test | result | verdict |
|---|---|---|---|---|
| 1 | Reward V1 rewards crawling | V2 redesign (Sept) | PPO still ~0.1 laps | necessary fix, not sufficient (`reward.md`) |
| 2 | Crash penalty too small (5) | p5 vs p40 pilots; `P40_penalty40` via `train.py` | P40: 36% laps at 868 k → **0% at 2 M** | **can** produce laps, **unstable** |
| 3 | Exploration collapse / oversized updates | sweep V1–V7 (entropy, target-KL, n_steps, log-std, epochs×batch, lr) | std, KL, clip restored (V6 KL 0.008); distance 36→68 m; **0 laps in all** | necessary, not sufficient |
| 3b | Start with less noise (`log_std_init −1`) | V5 | collapses to σ 0.030 | **refuted** (harmful) |
| 4 | Credit window too short (0.17 s) | action repeat ×10/×25 (R1–R4); λ 0.95→0.99 | R1 with ~5× V4's physics experience still behind V4 (63.6 vs 68.4 m), 0 laps; λ 0.99 *slower* | **refuted as the mechanism** |
| 5 | Not enough training | X1: V4 config at 5 M | 66.9 m at 2.54 M vs V4 68.4 m at 2 M | **refuted** (plateau) |
| 6 | The discounted objective prefers crashing | γ 0.99 → 0.999 (+ n_steps 8192) | **234.8 m, 65% laps, max 2.00 laps** | **confirmed** |
| 7 | Is γ alone enough? | γ 0.999 with n_steps 2048 (`G999_ns2048`) | 44.9 m, **0%** at 1 M (std 0.089) | **no — interaction** |

### The arithmetic behind #6

Two strategies on a 164 m track, crash penalty 5, at 14 m/s progress ≈ 0.14 m/step:

| strategy | discounted return, γ=0.99 | γ=0.999 |
|---|---|---|
| A: fast, **crash** at 70 m | **13.87** | 52.07 |
| B: half speed, **survive** the 30 s | 7.00 | 66.52 |
| C: fast **and** survive | ≈14.0 | ≈139.9 |

At γ=0.99 the crash penalty 500 steps ahead is worth 5×0.99⁵⁰⁰ = **0.033**:
crashing is nearly free, so A beats B, and A and C are within 1%. At γ=0.999
the same penalty is worth 3.03 and C dominates.

### Measured on the trained policies (not just arithmetic)

`diagnostics/basin_check.py` — one deterministic rollout per policy on
`synthetic_track_0`, rewards summed with each discount:

| policy | steps | distance | undiscounted | **disc. @0.99** | disc. @0.999 |
|---|---|---|---|---|---|
| PPO G999 (γ 0.999 + n_steps 8192) | 2991 | 327.8 m (2 laps) | 327.8 | 6.93 | **101.72** |
| PPO V4 (γ 0.99, crashes) | 534 | 73.9 m | 68.9 | **7.72** | 52.16 |
| PPO V0 baseline | 222 | 19.2 m | 14.2 | 4.99 | 12.75 |
| SAC `sac_overfit` (γ 0.99) | 3000 | 272.5 m | 272.5 | 6.38 | 85.37 |

**Under γ=0.99 the crashing policy outscores both lap-completing policies.** The
same two-lap G999 policy is worth 6.93 at γ=0.99 and 101.72 at γ=0.999. Caveat:
one episode per policy.

### The interaction behind #7

Matched window 0.8–1.0 M env steps (training log, stochastic policy; optimiser
diagnostics = median over updates in the window):

| | `n_steps` 2048 (default) | `n_steps` 8192 |
|---|---|---|
| **γ 0.99** | V0 · 39.6 m · **0%** · σ 0.061 · KL 0.27 · clip 0.54 | V4 · 62.9 m · **0%** · σ 0.261 · KL 0.047 · clip 0.28 |
| **γ 0.999** | G999_ns2048 · 44.9 m · **0%** · σ 0.089 · KL 0.12 · clip 0.41 | G999 · 217.7 m · **65%** · σ 0.392 · KL 0.029 · clip 0.21 |

- `n_steps` 8192 repairs the optimiser; with γ 0.99 the objective still prefers crashing → no laps.
- γ 0.999 repairs the objective; with 2048-step rollouts the optimiser still collapses → no laps.
- **Hypothesis for why (unverified):** at γ 0.999 the critic must regress ~1000-step returns.
  Its explained variance drops from 0.96 to 0.69, so it needs more data per update, and larger
  rollouts supply it. A test would be γ 0.999 + n_steps 2048 with a larger `n_epochs` or a separate value learning rate.

### Replication

Matched 0.8–1.0 M window, training-log lap rate: seed 0 **65%**, seed 1 **57%**,
seed 2 **52%** (`G999_gamma`, `G999_s1`, `G999_s2`). Not a lucky seed. This
matters because training is **not bit-reproducible** even at a fixed seed (our
re-run of Desmond's PPO d1 differs from iteration 0), whereas deterministic
evaluation is.

### Correction: "seed 0 held 64–65% from 1 M to 2 M" (written 21:00, wrong)
Seed 0 rose to 82% and then fell back to 64%; it did not hold. Final numbers below.

*Final (all three seeds finished 2 M, 2026-10-06 ~21:30):* training-log lap rate per 200 k window —

| window | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 0.8–1.0 M | 65% | 57% | 52% |
| 1.0–1.2 M | 75% | 64% | 39% |
| 1.2–1.4 M | 76% | 52% | 40% |
| 1.4–1.6 M | **82%** | 64% | 46% |
| 1.6–1.8 M | 79% | 79% | 9% |
| 1.8–2.0 M | 64% | **82%** | **3%** |

- **seed 1: still improving at 2 M.** **seed 0: peaked at 82% (97% in one 100 k window at 1.7 M) then fell to 64%.** **seed 2: collapsed** (0% in the 1.8 M and 1.9 M windows).
- *(Corrected below — the "crashes at 72 m, back in the crash-fast basin" reading was wrong.)*
- Optimiser health does not explain it: seed 2 has the *lowest* KL (0.036) and similar σ (0.23) to the seeds that kept lapping. In all three, σ keeps falling (0.8 → ~0.2) and KL keeps rising (0.014 → 0.04–0.08) — the same drift as the original collapse, just ~3× slower.
- `G999_ns2048` (γ alone) finished 2 M at **0%** — no laps at any point.
- **Net:** the fix produces laps on 3/3 seeds by 1 M; at 2 M it is 2/3 lapping well and 1/3 collapsed. Late-training stability is **not** established. Expect large seed variance in the final grid.

### Correction: seed 2 does not "slide back into crash-fast" — it stalls before the hairpin; and the training log misreads seed 0
The "failures at ~72 m" figure counted every non-lap episode as a crash. Split by outcome, and measured with the policy's noise off (`results/ppo_diagnosis/eval_own_track/`, team `evaluate.py`, 30 episodes ≈ 9 distinct spawns, own training track):

| deterministic, own track | 1.0 M | 1.6 M | 2.0 M |
|---|---|---|---|
| seed 0 | 3/30 laps | 25/30 | **30/30, 0 crashes** |
| seed 1 | 18/30 | 23/30 | 18/30 |
| seed 2 | 0/30 | **25/30** | **0/30** (14 crash, 16 stall) |

- **Seed 0 did not degrade** — its noise-free policy went 3 → 25 → 30/30. The training-log drop (82% → 64%) is the *noisy* training policy crashing more; the policy itself got better. (Why the noise hurts more late on is untested — a plausible guess is a tighter, faster line with less margin.)
- **Seed 1 fluctuates** (60–77%).
- **Seed 2 really did lose it:** 25/30 at 1.6 M → 0/30 at 2 M. The track's sharpest corner is a hairpin at **75.5 m (radius 2.4 m)**. At 2 M the noise-free policy brakes at 65–68 m, commands ~0 m/s at 68–71 m and **sits there for ~23 s** until the episode times out. With noise (training log, last 200 k) it laps 3%, crashes 78%, stalls 18% (stall point median 72.6 m). This is *not* the γ 0.99 crash-fast strategy — stopping earns no further reward — so the earlier basin explanation does not fit. Cause still open.
- **Lesson:** the training-log lap rate is a poor proxy for the policy that gets evaluated, in both directions. Score with deterministic evaluation (as `final-run-plan.md` already requires) and look at more than one checkpoint.
Raw rollout traces: `results/ppo_diagnosis/eval_own_track/corner_trace_output.txt`.

## Why SAC copes at γ=0.99 — and is not immune

SAC faces the same objective. From the basin table, SAC's lap-completing
policy scores *less* than PPO's crashing one under γ 0.99. So SAC did not find a
better optimum; it settled in a different, nearly tied one. Plausible tie-breakers
(measured, but their causal role is not isolated):

| | PPO (γ 0.99 grid) | SAC (γ 0.99 grid) |
|---|---|---|
| gradient updates per 2 M env steps | 312,500 | 2,000,000 (6.4×) |
| data | on-policy, ~99% redundant within a rollout | 1 M replay buffer, uniform sampling |
| final steering σ (d100) | **0.039** | **0.437** (11×) |

SAC still plateaus around 0.7 laps on held-out synthetic tracks and scores 0% on
real circuits (`grid-verification.md`). Whether γ 0.999 lifts SAC too is the job
of `G999_SAC`, which was running at the time of writing.

## How good is the fixed PPO? (evidence quality)

- **Training-log lap rate (stochastic, own track):** 65% at 2 M, n=3 seeds at 1 M.
- **Own track, deterministic (30 episodes):** at 2 M seed 0 30/30, seed 1 18/30, seed 2 0/30; at 1.6 M 25/23/25 of 30 (see the correction below).
- **Held-out synthetic, deterministic:** volatile across checkpoints —
  0/0/10/0/6/5/4/0/0 of 25 for 400 k…2 M; last-5 average **12%**, 83.5 m
  (`eval_g999_curve/`). V4 at 2 M: 0/25, 31.6 m. A d1 policy is *expected* to
  transfer poorly; this is what the diversity grid measures.
- **Real circuits:** 0/10 at every checkpoint (width, see `sim-to-real-width.md`).
- **Tested only at diversity 1.** Whether the fix holds at 5/20/100 is untested.

## Corrections made along the way

Recorded because each one was plausible and each affected a decision.

### Correction: "PPO's return is worse than standing still, so the reward is not the cause" (2026-10-04)
PPO with penalty 40 converged to an *undiscounted* return of −18.8, and
standing still scores 0. That was read as an optimisation failure that "rules
out the reward-design family". But PPO maximises the **discounted** return.
With ~19 m of progress over ~190 steps and the crash at the end, the γ=0.99 value is about
+8.5 − 5.9 = **+2.6 > 0**, so under the objective it was actually trained on,
crashing *beats* standing still. The inference compared the wrong quantity, and it
pushed the investigation toward optimiser fixes (sweep V1–V7) for two days.

### Correction: "γ is the lever; n_steps only amplifies it" (2026-10-06)
That came from comparing G999 against V0, which differs in two variables. The
one-variable comparison (the 2×2 above) shows an interaction. The related chat
claim "exploration rose 9× purely from γ" is also wrong: n_steps restores σ
(0.061→0.261), and γ alone does not (0.089).

### Correction: "finishing a lap is worth 0.001"
There is no lap reward to discount. What γ discounts is the *continuation value*
(future progress lost by crashing).

### Correction: "one step of progress outweighs the crash penalty"
That is only true if the crash is ≥ ~4 s away. At a realistic 2 s, the penalty is
worth ~4.9 steps of progress. The measured basin table is the defensible version.

## Open

- Does γ 0.999 + n_steps 8192 work at diversity 5 / 20 / 100? (A single-seed d100 pilot takes ~1.3 h.)
- Does γ 0.999 help or hurt SAC? (`G999_SAC`.)
- Does γ 0.999 with n_steps 2048 ever take off? (`G999_ns2048`, running.)
- Would the critic fix (more value epochs) substitute for n_steps 8192?

## See also

- `grid-verification.md` — the seed-0 grid these PPO numbers sit beside
- `reward.md` — how the discount interacts with the reward design
- `final-run-plan.md` — the configuration proposed from this diagnosis
- `results.md` — tables with source paths
- `open-questions.md` — Q10–Q12
- `incidents.md` — the measurement pitfalls found along the way
