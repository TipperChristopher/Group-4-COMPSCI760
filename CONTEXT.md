# CONTEXT — PPO failure diagnosis, and the two root causes

**Branch:** `team/crash-penalty-flag` (off `desmond/heldout-tracks` @ `ce00721`)
**Written:** 2026-10-07 · **Status:** two root causes identified and verified; replications in flight

This document records everything done outside Desmond's grid run, so the rest of the
team can pick it up without re-reading a long chat log. Every claim below is tied to a
file or a command you can re-run. Numbers that are still preliminary are marked.

---

## 1. TL;DR

Three findings, in order of importance:

| # | Finding | Evidence |
|---|---|---|
| **1** | **PPO never completed a lap because the reward, as configured, preferred crashing.** At 100 Hz control with γ=0.99 the agent cannot value anything more than ~1 s away, and the crash penalty discounts to ~0.03. Working the arithmetic, *"drive fast and crash"* scored **higher** (13.87) than *"drive slower and survive"* (7.00). Setting γ=0.999 flips the ordering and PPO laps for the first time. | §4, §6 |
| **2** | **The real-circuit failure is a different problem: corridor width.** The generator emits a *constant* 1.47 m half-width, and 100% of real-circuit points are narrower than that. Narrowing **only the walls** of the exact test-track centrelines takes a policy from **58% lap completion to 0%**. | §7 |
| **3** | **Checkpoint selection matters more than we assumed.** The final 2 M checkpoint is frequently *worse* than an earlier one — for the γ run, 800 k scores **40%** lap completion on held-out tracks and 2 M scores **0%**. | §6.4 |

---

## 2. Where things live

| What | Path |
|---|---|
| **This repo (grid + diagnostics)** | `COMPSCI 760/team_repo_heldout/` — branch `team/crash-penalty-flag` |
| Trained models from the sweep | `team_repo_heldout/models/<ALGO>_<N>tracks_s<seed>_<tag>/` |
| Per-episode + per-update logs | `…/monitor_0.monitor.csv`, `…/progress.csv` |
| Run configs (provenance) | `…/run_config.json` — commit, reward hash, all args |
| Track manifest + splits | `team_repo_heldout/tracks/manifest.json` |
| **Our earlier spike work** | `team_repo/` branch `experiments/feasibility-spike` |
| Spike wiki (protocol, reward, incidents) | `team_repo/feasibility_spike/wiki/` |
| **Evaluation scripts + raw analysis** | `COMPSCI 760/_verify/` (scratch, outside git) |
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

Rolled out each trained policy, `_verify/basin_check.py`:

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
scan by ~1% of its own spread at 2–10 m/s (`_verify/obs_redundancy.py`), so a 2048-step
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

- **γ=0.999 is stable at 65%.** PPO had never completed a single lap in ~24,000 training
  episodes across Desmond's four grid cells.
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

γ run evaluated at matched checkpoints, held-out synthetic, deterministic
(`_verify/g999_curve.py`, `_verify/lap_break.py`, `_verify/lap_break_final/`):

| checkpoint | lap completion (held-out synthetic) | mean m |
|---|---|---|
| 800 k | **40% (10/25 finished)** | 98.8 |
| 2 M (final) | **0% (0/25)** | 67.1 |

The **final checkpoint is worse than 800 k.** Same pattern as our own SAC (1.2 M
completed 30/30 spawns; 2 M completed 22/30) and as Desmond's grid (d20 > d100 on
the final checkpoint only). The `val_track_*` split exists in the manifest precisely
for checkpoint selection and **has not been used yet** — it should be.

### 6.5 Budget — more steps does not help

`X1_v4_5M` (V4 config, 5 M budget) at 2.53 M steps: **66.9 m** vs V4's 68.4 m at 2 M.
Same plateau. Extrapolating the late slope (+0.8 m per Mstep) to a lap would need
~120 M more steps ≈ 170 h.

### 6.6 Replication status (preliminary — still running)

| run | seed | steps reached | last-200k m | lap rate |
|---|---|---|---|---|
| `G999_gamma` | 0 | 2.0 M (done) | 234.8 | 65% |
| `G999_s1` | 1 | 0.69 M | 114.6 | 18% |
| `G999_s2` | 2 | 0.70 M | 197.6 | 58% |

All three tracking together ⇒ **not a lucky seed**. This matters because training is
*not* bit-reproducible across machines even at a fixed seed (evaluation is — it
reproduced Desmond's numbers to the millimetre).

`G999_ns2048` (γ=0.999 with SB3's default `n_steps=2048`): 42.5 m at 0.70 M — behind
the `n_steps=8192` runs, so `n_steps` **amplifies** γ but does not replace it.
`G999_SAC` (does γ help SAC too?) — just started.

---

## 7. The width finding

Desmond's `narrowA_track_k` / `narrowB_track_k` have the **exact centreline of
`test_track_k`** — same shape, length and curvature — with only the walls moved.
Half-width: test **1.473 m** · narrowA **1.072 m** (≈ real median) · narrowB **0.751 m**.

Our best SAC (`sac_overfit`, penalty 40) on these, deterministic, 10 episodes × 5 tracks
(`_verify/width_eval_oursac.py`):

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
"$PY" _verify/build_canonical_tracks.py
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

1. **Use the `val_track_*` split for checkpoint selection.** The final checkpoint is
   demonstrably not the best (§6.4), and every number in the grid was taken at 2 M.
   Free — all checkpoints already exist.
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

- Does `n_steps=8192` matter once γ is fixed? `G999_ns2048` suggests it amplifies but
  does not replace γ — still running.
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
| `_verify/build_canonical_tracks.py` | install the seed-123 training pool on a fresh machine |
| `_verify/{basin_check,obs_redundancy,did_it_brake2,g999_ckpt_curve,width_eval_oursac,lap_break}.py` | the analyses quoted above |

`_verify/` sits outside git; ask if you want it moved in.
