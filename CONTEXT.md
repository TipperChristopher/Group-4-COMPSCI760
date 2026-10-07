# CONTEXT — PPO failure diagnosis, and the two root causes

**Branch:** `team/crash-penalty-flag` (off `desmond/heldout-tracks` @ `ce00721`)
**Written:** 2026-10-06 (revised the same evening — see §0) · **Status:** two root causes identified and verified; replications in flight
**Long-form record:** the wiki at `feasibility_spike/wiki/` (start at `index.md`); raw evidence in `results/ppo_diagnosis/`.

## 0. Revisions since the first push (commit `ec8b784`)

Three claims in the first version were wrong or overstated. Corrected below, recorded here so nobody quotes the old text:

1. **"γ is the lever; `n_steps` only amplifies it" → WRONG.** At a matched 1 M steps, γ=0.999 with the default
   `n_steps=2048` still has collapsed exploration (std 0.089) and **0% laps** — no better than baseline.
   **Both changes are needed together** (§6.2a). γ changes *which strategy is optimal*; `n_steps=8192` keeps the
   optimiser healthy enough to find it.
2. **"800 k = 40%, 2 M = 0% → the final checkpoint is worse" → overstated.** The full checkpoint curve
   (§6.4) is *volatile*, not declining: 0/0/10/0/6/5/4/0/0 out of 25 across nine checkpoints. The 2 M
   checkpoint happens to sit in a trough. The lesson is "one checkpoint is not a measurement", not "train less".
4. **(added ~21:30) "seed 0 held 64–65% from 1 M to 2 M" → wrong.** It peaked at 82% and fell to 64%; seed 2 collapsed
   to 3% by 2 M; seed 1 is still improving (82%). The fix laps on 3/3 seeds by 1 M but late stability is not established (§6.6).
3. **Width table checkpoint was unlabelled** — it is the **1.2 M** checkpoint of `sac_overfit`. The 2 M final
   gives 27/50 → 0/50 → 0/50 → 0/20: same conclusion (§7).


This document records everything done outside Desmond's grid run, so the rest of the
team can pick it up without re-reading a long chat log. Every claim below is tied to a
file or a command you can re-run. Numbers that are still preliminary are marked.

---

## 1. TL;DR

Three findings, in order of importance:

| # | Finding | Evidence |
|---|---|---|
| **1** | **PPO never completed a lap because the reward, as configured, preferred crashing.** At 100 Hz control with γ=0.99 the agent cannot value anything more than ~1 s away, and the crash penalty discounts to ~0.03. Working the arithmetic, *"drive fast and crash"* scored **higher** (13.87) than *"drive slower and survive"* (7.00). Setting γ=0.999 flips the ordering — and **together with `n_steps=8192`** (neither works alone) PPO laps for the first time: 52–65% of training episodes across 3 seeds at 1 M steps. | §4, §6.2 |
| **2** | **The real-circuit failure is a different problem: corridor width.** The generator emits a *constant* 1.47 m half-width, and 100% of real-circuit points are narrower than that. Narrowing **only the walls** of the exact test-track centrelines takes a policy from **58% lap completion to 0%**. | §7 |
| **3** | **A single checkpoint is not a measurement.** Held-out lap completion of the γ run swings 0 → 10/25 → 0 → 6/25 between *adjacent* 200 k checkpoints; the final 2 M checkpoint scores 0/25. Every grid number so far is one checkpoint of one seed. | §6.4 |

---

## 2. Where things live

| What | Path |
|---|---|
| **This repo (grid + diagnostics)** | `COMPSCI 760/team_repo_heldout/` — branch `team/crash-penalty-flag` |
| **Raw evidence, in git** (logs, configs, final weights, every eval CSV) | `results/ppo_diagnosis/` — see its `README.md` |
| Trained models from the sweep (all checkpoints, gitignored) | `team_repo_heldout/models/<ALGO>_<N>tracks_s<seed>_<tag>/` |
| Per-episode + per-update logs | `…/monitor_0.monitor.csv`, `…/progress.csv` |
| Run configs (provenance) | `…/run_config.json` — commit, reward hash, all args |
| Track manifest + splits | `team_repo_heldout/tracks/manifest.json` |
| **Our earlier spike work** | `team_repo/` branch `experiments/feasibility-spike` |
| Spike wiki (protocol, reward, incidents) | `team_repo/feasibility_spike/wiki/` |
| **Evaluation scripts** | `diagnostics/` (copies of the scratch scripts in `COMPSCI 760/_verify/`) |
| Installed track maps | `COMPSCI 760/spike/f1tenth_gym_v1/maps/` |

---

## 3. Code changes (both branches)

### `train.py` — optional overrides, all defaulting to current behaviour

```text
--crash-penalty F     override CRASH_PENALTY for this run   (D6 was never testable before)
--run-tag TAG         suffix the run dir, so sweeps don't collide
--gamma F             discount factor
--gae-lambda F        GAE lambda (PPO only)
--ent-coef F          entropy bonus (PPO only)
--target-kl F         PPO trust-region cap (PPO only)
--n-steps N           PPO rollout length (PPO only)
--n-epochs N          PPO epochs per rollout (PPO only)
--log-std-init F      initial policy log-std (PPO only)
--batch-size N        minibatch size
--learning-rate F     step size
--action-repeat N     hold each action for N physics steps
--max-episode-steps   unchanged meaning; divided by --action-repeat so the 30 s budget is kept
```

Every override is recorded in `run_config.json` — hyperparameters under `args`, the
reward under `reward_constants` — so an overridden cell is self-describing and can
never be mistaken for a default one. PPO-only flags are rejected with `--algo SAC`.

Commits: `66b9257` (crash-penalty + hyperparameters), `67cd9d5` (gamma / gae-lambda + ActionRepeat).

### `action_repeat.py` — new file, **tested and REFUTED** (kept for the negative result)

Frame-skip: holds each action for N physics steps. The idea was that at 100 Hz,
PPO's advantage window `1/(1-γλ)` is only 16.8 steps = 0.168 s ≈ 1.7 m of track, so
it cannot attribute credit to a corner 30 m ahead.

**It did not work** — see §6.3. γ achieves the same horizon extension *without*
giving up control resolution, whereas repeat costs 90% of the decisions per episode
(300 instead of 3000) and can no longer make fine corrections. γ supersedes it.

### `tracks/manifest.json` — extended with `narrowA` / `narrowB` splits

Produced by Desmond's own `tracks/make_narrow_tracks.py`; not hand-edited.

---

## 4. The discounting arithmetic (the core of finding 1)

Config: 100 Hz control (`timestep=0.01`), reward = metres of centreline progress,
crash penalty 5, episode cap 3000 steps = **30 s**, track 164.1 m.

**There is no lap bonus.** `train.py` never sets `target_laps`, so crossing the line
is a *termination*, not a reward event. The only reward is dense progress minus the
crash penalty.

### Value horizon

```text
V(continuing) = r × γ/(1-γ)          r = 0.14 per step at 14 m/s

γ = 0.99   → 13.86  ≈  99 steps  =  1.0 s of driving
γ = 0.999  → 139.86 ≈ 999 steps  = 10.0 s of driving
```

### Two strategies, discounted

| strategy | γ=0.99 | γ=0.999 |
|---|---|---|
| A: fast (14 m/s), **crash** at 70 m | **13.87** | 52.07 |
| B: slower (7 m/s), **survive** the episode | 7.00 | 66.52 |
| C: fast **and** survive | ~14.0 | ~139.9 |

**At γ=0.99, A beats B by 6.87 — the objective prefers crashing.** The crash penalty
arriving 500 steps later is worth `5 × 0.99^500 = 0.033`. Free. And C ties A
(14.0 vs 13.87), so the agent is nearly indifferent between driving fast and crashing
and driving fast and surviving.

### Measured confirmation (not just arithmetic)

Rolled out each trained policy, `diagnostics/basin_check.py`:

| algo | policy | steps | distance | undiscounted | **discounted @0.99** | discounted @0.999 |
|---|---|---|---|---|---|---|
| PPO | G999 (γ=0.999) | 2991 | **327.8 m (2 laps)** | 327.8 | 6.93 | **101.72** |
| PPO | V4 (crashing) | 534 | 73.9 m | 68.9 | **7.72** | 52.16 |
| PPO | V0 baseline | 222 | 19.2 m | 14.2 | 4.99 | 12.75 |
| SAC | `sac_overfit` | 3000 | 272.5 m | 272.5 | 6.38 | 85.37 |

Two things to take from this:

- **Under γ=0.99, PPO's crashing policy (7.72) scores higher than SAC's driving policy (6.38).**
  So SAC did not find a better solution — it found a *different* one. At γ=0.99 the two
  are close, so which basin you land in is decided by secondary factors (§5).
- **The same policy is worth 6.93 at γ=0.99 and 101.72 at γ=0.999.** The γ=0.99 objective
  was registering about 7% of the value that was actually there.

---

## 5. Why SAC copes and PPO does not (same γ, different basin)

Both algorithms use γ=0.99, so both face the same horizon. **SAC is not immune** — it
plateaus at ~0.70 laps and gets 0% on real circuits. It just gets further.

| | PPO | SAC |
|---|---|---|
| gradient updates over 2 M env steps | 312,500 | 2,000,000 (**6.4×**) |
| data reuse | discarded after 10 epochs | 1 M replay buffer |
| sample correlation | **~99% redundant** within a rollout | decorrelated by uniform sampling |
| exploration (measured policy σ) | **0.039** | **0.437** (11×) |

The correlation figure is measured, not assumed: one 100 Hz step changes the 108-beam
scan by ~1% of its own spread at 2–10 m/s (`diagnostics/obs_redundancy.py`), so a 2048-step
rollout holds far fewer than 2048 *distinct* situations, and PPO's 64-sample minibatch
comes from one contiguous stretch. SAC's 256-sample batch is drawn uniformly from a
million transitions.

Since A and C are within ~1% of each other at γ=0.99, the algorithm that keeps
exploring and takes 6.4× more updates is the one that stumbles into *surviving* and
stays there. PPO's σ collapsed by ~0.5 M steps, so it committed to *crashing* early and
could never leave.

---

## 6. Experiment inventory

All runs: `--algo PPO --diversity 1 --seed <s> --total-timesteps 2000000`, canonical
pool, crash penalty 5 unless stated. "last-200k m" = mean metres per episode over the
final 200 k steps (training, stochastic policy).

### 6.1 The symptom-fix sweep — seven candidates, none produced a lap

| tag | change | last-200k m | lap rate | verdict |
|---|---|---|---|---|
| `V0_baseline` | — (SB3 defaults) | 36.4 | 0% | reference |
| `V4_nsteps` | `n_steps` 2048 → 8192 | **68.4** | 0% | best of the sweep |
| `V3_ent_kl` | `ent_coef=0.01` + `target_kl=0.03` | 54.3 | 0% | |
| `V6_updates` | `n_epochs` 10→4, `batch` 64→256 (320→32 updates) | 53.3 | 0% | |
| `V7_lr` | `lr` 3e-4 → 1e-4 | 50.4 | 0% | |
| `V2_kl` | `target_kl=0.03` | 48.0 | 0% | |
| `V1_ent` | `ent_coef=0.01` | 41.9 | 0% | marginal |
| `V5_logstd` | `log_std_init=-1.0` | 37.9 | 0% | **harmful** |

All helped a little (1.2–1.9×) and none got near a lap. They did fix the *optimisation
metrics* — V6 took approx_kl from 0.33 to 0.008 — which is what made it clear the
problem lay elsewhere.

### 6.2 The decisive change

| tag | change | last-200k m | **lap rate** | max laps |
|---|---|---|---|---|
| **`G999_gamma`** | **`gamma` 0.99 → 0.999** (+ `n_steps=8192`) | **234.8** | **65%** | **2.00** |
| `G999L99_full` | + `gae_lambda` 0.95 → 0.99 | 156.8 | 59% | 1.49 |
| `P40_penalty40` | crash penalty 40, `n_steps=8192` | 71.7 | 0% (was 36% at 868 k) | 0.97 |

- **γ=0.999 + `n_steps=8192` laps on 3/3 seeds by 1 M; by 2 M seed 1 is at 82% and still rising, seed 0 peaked at 82% then fell to 64%, seed 2 collapsed to 3%** (§6.6). PPO had never completed a single lap in ~24,000 training
  episodes across Desmond's four grid cells.

### 6.2a Which of the two changes matters? Both — it is an interaction

Matched window 0.8–1.0 M env steps; distance/lap rate from the training log (stochastic),
optimiser diagnostics = median over updates in the window (`progress.csv`):

| | `n_steps` 2048 (default) | `n_steps` 8192 |
|---|---|---|
| **γ 0.99** (default) | V0: 39.6 m, **0%** · std 0.061, KL 0.27, clip 0.54 | V4: 62.9 m, **0%** · std 0.261, KL 0.047, clip 0.28 |
| **γ 0.999** | G999_ns2048: 44.9 m, **0%** · std 0.089, KL 0.12, clip 0.41 | G999: 217.7 m, **65%** · std 0.392, KL 0.029, clip 0.21 |

- `n_steps` alone fixes the optimiser (std recovers, KL in range) but the objective still prefers crashing → no laps.
- γ alone fixes the objective but the optimiser still collapses (std 0.089) → no laps.
- Both → laps. `G999_ns2048` is still running (32.5 m, 0% at 1.16 M) and is n=1, so "γ alone never works"
  is not established — "γ alone does not work by 1 M" is.
- Hypothesis for *why* (unverified): at γ=0.999 the critic must regress ~1000-step returns; its
  explained variance drops from 0.96 to 0.69, so it needs more data per update. Bigger rollouts supply it.
- The λ result is informative: λ 0.95 → 0.99 widens the GAE credit window 5× (19.6 → 91 steps) and is
  **slower**, not faster (59% vs 65%). So the earlier "0.168 s credit window" hypothesis was not the mechanism —
  the change of *objective* is.
- **γ+λ works too, just slower** (59% vs 65% at 2 M). At 800 k it looked worse (0% vs 40%),
  but that was a checkpoint artifact — see §6.4.
- **Penalty 40 learns then forgets** (36% at 868 k → 0% at 2 M). A bigger penalty *can*
  get it lapping but the run is unstable. Fighting the discount is not the same as fixing it.

### 6.3 Action repeat — REFUTED

| tag | repeat | decisions | ≈ physics steps experienced | last-200k m | lap rate |
|---|---|---|---|---|---|
| `R1_repeat10_alone` | 10 | 1.05 M | ~10 M | 63.6 | 0% |
| `R2_repeat10_v6` | 10 | 1.04 M | ~10 M | 54.0 | 0% |
| `R3_repeat10_v4` | 10 | 0.94 M | ~9 M | 58.4 | 0% |
| `R4_repeat25` | 25 | 0.48 M | ~12 M | 58.8 | 0% |
| *(V4 for reference)* | 1 | 2 M | ~2 M | *68.4* | *0%* |

**R1 has had ~5× the simulated driving and is still behind V4.** All four: zero laps.

### 6.4 Checkpoint selection — the trap

`G999_gamma` at every 200 k checkpoint, deterministic, test_track_0..4 × 5 episodes (≈3 distinct spawns each)
+ Spielberg/Silverstone × 5 (`diagnostics/g999_ckpt_curve.py` → `results/ppo_diagnosis/eval_g999_curve/`):

| checkpoint | held-out synth laps finished | mean laps | mean m | real finished | real m |
|---|---|---|---|---|---|
| 400 k | 0/25 | 0.446 | 80.6 | 0/10 | 42.5 |
| 600 k | 0/25 | 0.290 | 52.8 | 0/10 | 3.8 |
| 800 k | **10/25** | 0.540 | 98.8 | 0/10 | 43.8 |
| 1.0 M | 0/25 | 0.257 | 46.7 | 0/10 | 33.7 |
| 1.2 M | 6/25 | 0.498 | 90.5 | 0/10 | 77.6 |
| 1.4 M | 5/25 | 0.603 | 109.6 | 0/10 | 41.2 |
| 1.6 M | 4/25 | 0.563 | 101.7 | 0/10 | 23.1 |
| 1.8 M | 0/25 | 0.264 | 48.4 | 0/10 | 35.3 |
| 2.0 M (final) | 0/25 | 0.365 | 67.1 | 0/10 | 24.0 |

**Volatile, not declining.** Adjacent checkpoints swing 0 ↔ 10/25. Averaged over the last five
checkpoints (1.2–2.0 M): 15/125 = **12%** held-out lap completion, 83.5 m — vs V4's final 0/25, 31.6 m.
Real circuits: 0/10 at every checkpoint.

This is the same phenomenon Desmond's noise-floor check showed for the γ=0.99 grid (within-cell
checkpoint SD ≈ half the between-cell SD), and the same estimator problem as our SAC (1.2 M
30/30 spawns vs 2 M 22/30). Consequences for any comparison:
- never report one checkpoint of one seed;
- average over the last k checkpoints and/or ≥3 seeds;
- the `val_track_*` split exists for checkpoint selection and **has not been used yet**.

Note also the gap between **training-log** lap rate (65%, stochastic policy, its own training track)
and **held-out deterministic** lap rate (0–40%). On its own training track the deterministic 2 M policy
drives 2 clean laps (§4 table). The held-out gap is expected at diversity 1 — it is the thing the
diversity grid measures.

### 6.5 Budget — more steps does not help

`X1_v4_5M` (V4 config, 5 M budget) at 2.53 M steps: **66.9 m** vs V4's 68.4 m at 2 M.
Same plateau. Extrapolating the late slope (+0.8 m per Mstep) to a lap would need
~120 M more steps ≈ 170 h.

### 6.6 Replication status (preliminary — still running)

| run | seed | steps reached | last-200k m | lap rate |
|---|---|---|---|---|
| `G999_gamma` | 0 | 2.0 M (done) | 234.8 | 65% |
| `G999_s1` | 1 | 1.16 M | 237.6 | 72% |
| `G999_s2` | 2 | 1.17 M | 144.7 | 38% |

At the matched 0.8–1.0 M window: **65% / 57% / 52%** for seeds 0 / 1 / 2 (186–218 m) ⇒ **not a lucky seed**
(training-log metric; held-out evaluation of seeds 1–2 not yet run). Per-window rates swing ±15 points
within a seed, so compare windows, not single numbers.

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

`G999_ns2048` (γ=0.999 with SB3's default `n_steps=2048`): 32.5 m, 0% at 1.16 M — see §6.2a.
`G999_SAC` (does γ help SAC too?) — 114 k steps at archive time; ~47 env steps/s under
current load ⇒ ~12 h to 2 M.

**2026-10-07 — teammate tuning round verified** (`tuning-vw-val.md`, 16 runs @ commit `6e5b60e`):
- γ 0.999 alone BREAKS SAC at d20 (0.246 vs baseline 0.509; collapse after 400 k, critic loss spike ~3789).
- Time cost 0.01/step (break-even speed 1 m/s; value cost = 0.01/(1−γ) → 10× at γ .999) stops the crawling and the collapse, but still scores below SAC baseline (0.439 vs 0.509).
- Our PPO fix (γ .999 + n_steps 8192) does NOT beat PPO defaults at d20: 0.077 vs 0.142 (2 seeds). Transfer not established.
- Our proposed P3 arm (ent_coef 0.01 + target_kl 0.03) stabilised the optimiser (KL 0.007) but inflated σ to ~2.0 and crawled → 0.039.
- Eval uses canonical reward (no time cost); scores recomputed from raw CSVs match to ≤5e-5; 3-distinct-spawn defect confirmed.

---

## 7. The width finding

Desmond's `narrowA_track_k` / `narrowB_track_k` have the **exact centreline of
`test_track_k`** — same shape, length and curvature — with only the walls moved.
Half-width: test **1.473 m** · narrowA **1.072 m** (≈ real median) · narrowB **0.751 m**.

Our best SAC (`sac_overfit`, penalty 40, **1.2 M checkpoint**) on these, deterministic, 10 episodes × 5 tracks
(`diagnostics/width_eval_oursac.py` → `results/ppo_diagnosis/eval_width_oursac/oursac1200k_*.csv`).
The 2 M final checkpoint gives 141.8 m 27/50 → 35.1 m 0/50 → 11.3 m 0/50 → real 16.9 m 0/20 (`oursac2000k_*.csv`):

| corridor half-width | mean distance | mean laps | **lap completion** |
|---|---|---|---|
| **1.473 m** (test) | 157.0 m | 0.869 | **29/50 — 58%** |
| **1.072 m** (narrowA ≈ real) | 19.1 m | 0.106 | **0/50** |
| **0.751 m** (narrowB) | 8.0 m | 0.045 | **0/50** |
| **real circuits** (median ~1.075 m) | 31.7 m | 0.083 | **0/20** |

**A policy that completes 58% of laps on test tracks completes zero when the corridor
narrows to real-circuit width — with identical corners.** narrowA's distance (19.1 m)
sits next to the real circuits' (31.7 m).

Mechanism, from `tracks/` geometry analysis:
- Training pool width across **all 100 tracks: 1.430–1.474 m — a 4 cm span. Width
  diversity is exactly zero at every diversity level.**
- Real circuits are **100% narrower** than anything in training.
- Curvature: real circuits are only **0.95%** out-of-distribution, and *straighter*
  on average (p90 0.195 vs training 0.408).
  → This **refutes** our own earlier ranked fix #1 ("widen `TRACK_TURN_RATE`").
  Sharpness was the wrong axis; width is the right one.

**Implication:** more tracks from the current generator cannot fix this, because the
generator has no width variance. The pool needs randomised width.

---

## 8. Caveats and corrections made along the way

Recorded because several were mistakes that other people should not repeat.

1. **"SAC can't lap its own training tracks"** — retracted. I had evaluated Desmond's
   models on *my* `synthetic_track_*`, which are a **different set** (his are seed 123;
   mine came from `make_synth_tracks.py --seed 0`). Fingerprint by track length:
   **0/20 match.** Names matching proves nothing.
2. **"One step of progress outweighs the crash penalty"** — overstated. True only if the
   crash is ≥4 s away; at a realistic 2 s braking distance the penalty is worth 4.9×
   one step. The measured basin numbers in §4 are the defensible version.
3. **"Finishing the lap is worth 0.001"** — wrong framing. There is no lap reward at all.
   What is discounted is the continuation value.
4. **"γ+λ is worse than γ"** — a checkpoint artifact. At 800 k it was 0% vs 40%; by
   2 M it is 59% vs 65%. Don't read conclusions off one checkpoint.
5. **Single-run variance is large** — two nominally identical SAC runs differed 9×
   (0.88 vs 0.10 laps). Every one of Desmond's grid cells is n=1. Replication is not
   optional.
6. **Training is not bit-reproducible** across machines even at a fixed seed, though
   *evaluation* is (verified to the millimetre against Desmond's CSV).

---

## 9. Reproducing from scratch

```bash
PY="<repo>/spike/venv/Scripts/python.exe"     # the venv this project uses

# 1. Canonical tracks. train.py REFUSES to run on non-canonical maps, by design.
#    Our installed maps were seed 0; his are seed 123. This regenerates his set
#    byte-for-byte (verified 100/100, manifest byte-identical).
"$PY" diagnostics/build_canonical_tracks.py
cd team_repo_heldout && "$PY" tracks/make_heldout_tracks.py      # installs val/test
"$PY" tracks/make_narrow_tracks.py                                # installs narrowA/B

# 2. The decisive experiment
"$PY" train.py --algo PPO --diversity 1 --seed 0 --total-timesteps 2000000 \
    --n-steps 8192 --gamma 0.999 --run-tag MYRUN

# 3. Evaluate a checkpoint on held-out tracks
"$PY" evaluate.py --algo PPO --model-path models/PPO_1tracks_s0_MYRUN \
    --tracks test_track_0 test_track_1 Spielberg --episodes 5 --eval-seed 0 \
    --target-laps 1 --max-steps 15000 --reset-type cl_grid_static --no-plot
```

Sweep launchers: `run_ppo_sweep.sh`, `run_repeat_sweep.sh`.

---

## 10. Open questions and what to do next

**Highest value, in order:**

1. **Stop scoring single checkpoints.** Held-out scores swing 0 ↔ 10/25 between adjacent
   checkpoints (§6.4), and every number in the grid was taken at one checkpoint of one seed.
   Average the last 5 checkpoints and/or select on `val_track_*`. Free — all checkpoints exist.
2. **Re-run the PPO diversity grid at γ=0.999.** The PPO diversity curve has never been
   measurable, because PPO sat at 0 laps for every level. This is what the final
   presentation needs.
3. **Re-run SAC at γ=0.999** (`G999_SAC`, running) — is the horizon a shared limit or
   specific to PPO?
4. **Width-randomised training pool** — the direct test of §7. Generate training tracks
   with half-width sampled over ~0.7–1.5 m, retrain SAC, evaluate on real circuits.
   Prediction: a large jump if width is necessary and sufficient.
5. **Decide the crash penalty (D6)** with the evidence now available: penalty 40 gets
   PPO lapping but **unstable**; penalty 5 with γ=0.999 is stable at 65%. Recommend 5.

**Unresolved:**

- Does γ=0.999 *ever* work with `n_steps=2048`? Not by 1.16 M (§6.2a); `G999_ns2048` still running.
- Does γ=0.999 + `n_steps=8192` still work at diversity 5/20/100? Only diversity 1 has been tested.
- SAC's real-circuit ceiling: after width, is anything else left? Untested.
- All reported training-log lap rates use the **stochastic** policy; the deterministic
  evaluated rate is lower. Quote both, never just the training log.

---

## 11. Key files added by this work

| file | purpose |
|---|---|
| `train.py` | +12 optional flags, all defaulting to existing behaviour |
| `action_repeat.py` | frame-skip wrapper (tested, refuted, kept for the record) |
| `run_ppo_sweep.sh` | the 8-variant symptom sweep |
| `run_repeat_sweep.sh` | the action-repeat batch |
| `diagnostics/build_canonical_tracks.py` | install the seed-123 training pool on a fresh machine |
| `diagnostics/*.py` | every analysis quoted above — index in `diagnostics/README.md` |
| `results/ppo_diagnosis/` | the raw outputs those scripts produced |

The scripts carry absolute paths for the original machine; edit `ROOT`/`REPO` at the top.
