# Investigation: fixing PPO training and zero-shot generalization

Status: working document for the final-phase experiments. Everything below is
grounded in runs in `results/` (n=1 seeds unless stated — treat mechanisms as
supported, magnitudes as samples). The deck is frozen; this is what we do NEXT.

---

## Problem 1 — PPO cannot hold a policy (oscillates, then degrades)

### What we did / measured
- **PPO+VN 1M:** peaks at **0.344 laps @ 300k**, then degrades to **0.098 @ 1M**.
  Training reward oscillates: +37 → −9 → +14 → −19 (mean 9.7, **std 14.3** — the
  spread is bigger than the mean; 3 sign flips).
- **3M test:** saturates ~500k then wanders — **more steps do not help**.
- **2M (frozen protocol):** same pattern, final 0.13. The 1M→2M tail just re-oscillates.
- **Reward variants (all 2M, all PPO):** original V1, new-p5, new-p40 →
  **all three plateau ≈ 0.1 laps.** Changing the reward does not rescue PPO.
- **VecNormalize ablation:** PPO+VN ≈ PPO (no stabilization) — VN helps SAC, not PPO.
- **Dense shaping test:** forward wall-proximity penalty got gamed (side crashes at
  15.8 m/s) and still oscillated.
- **Entropy probes:** 0.05 collapsed; 0.01 hit best-ever 0.609 but was the least stable.
- **Crash forensics:** crash speed RISES with training (2.3 → 7.0 m/s); the final
  policy crashes while ACCELERATING into the sharpest corner — it learns to go faster,
  never to brake.

### Diagnosis (mechanism, verified)
1. **Catastrophic forgetting.** PPO is on-policy: it learns from its latest rollout
   and discards it. With a single environment, one noisy update can erase the best
   policy it found (0.34 @ 300k is never recovered). SAC's replay buffer is exactly
   the difference — same reward, same budget, SAC completes laps.
2. **n_envs=1 by protocol design.** Fixed for the "updates scale with diversity" bug,
   but single-env on-policy updates are high-variance. Under a FIXED step budget,
   adding envs trades update count for variance (976 updates at 1 env → 122 at 8),
   so parallelism is not free data.
3. **Safety-blind reward.** Progress under a fixed step cap is a speed incentive.
   There is no gradient for braking, so the policy sits on a knife-edge between
   "crawl" and "crash" — the stable middle (cornering) is the hard skill.
4. The original crawl problem was the OLD reward's time-payment; the current problem
   is purely algorithmic (points 1-3).

### Candidate fixes (ranked by expected value / cost)
1. **n_envs = 8 CONSTANT with the track-pool sampler (measured ablation).**
   Decoupled from diversity → does not reintroduce the n_envs bug. Expect lower
   variance per update at 8× fewer updates. *Test:* PPO 2M, n_envs=8, frozen else.
   This is the single most informative PPO experiment left.
2. **Small entropy bonus (0.005-0.01), multi-seed.** 0.01 gave the best-ever laps
   (0.609) in one run; needs ≥3 seeds to tell signal from noise.
3. **Best-checkpoint selection.** Mitigates the *reporting* of the degradation but
   does not fix learning — we already adopt it as the frozen eval rule.
4. **Dense braking reward (all-around clearance × speed).** Our forward-only version
   was gamed; an all-around term rewards margin directly and gives PPO the braking
   gradient it lacks. *Test:* shaping weight sweep, SAC included for comparison.
5. **Frame honestly:** under a fixed budget, on-policy vs replay is itself a
   legitimate *finding* (the literature survey predicted it). "Fixing PPO" may end
   up as: PPO needs more variance-reduction than a fixed-budget single-env protocol
   allows — report it as the comparison result, with SAC as the workhorse.

---

## Problem 2 — zero-shot breaks on real circuits

### What we did / measured
- **Baselines:** gap-follower completes Spielberg (1.0 lap, 67 s) and fails
  Silverstone at the same hairpin every seed; random ≈ 0 everywhere.
- **PPO-2M zero-shot:** ≈ random level on real circuits (0.013 vs 0.011 laps);
  0.13-0.24 on synthetic tracks.
- **SAC-1M zero-shot:** completes UNSEEN synthetic tracks (track_2: 1.0 laps 5/5,
  22.4 s; track_1: 0.80, 2/5) but fails real circuits (0.06-0.11 laps).
- **Action-level verification (the key evidence):** on Spielberg/Silverstone the
  policy CREEPS for ~20 s (0.2-0.7 m/s, vs 5-15 m/s on synthetic) then suddenly
  commands full steering lock + ~17 m/s throttle and rams the wall from a standstill.
  Crash at 2.7-10.7 m — on straights, ~100 m before the first hard corner.
- **Narrow-track reproduction:** the same crash reproduces on a NARROW synthetic
  track with the same spawn scan → it is an INPUT-DISTRIBUTION effect, not
  real-track magic.
- **Geometry audit:** the synthetic generator caps corner sharpness at ~0.54 1/m
  (`TRACK_TURN_RATE = 0.31` in random_trackgen.py); real circuits reach 1.27
  (Spielberg) on laps 2-2.5× longer with more straights.
- **Overfitting check (DONE):** SAC checkpoints 100k/300k/600k/1M evaluated zero-shot.
  Unseen synthetic laps RISE monotonically (synth_1: 0.42 -> 0.69 -> 0.91 -> 1.00;
  finishes 0 -> 0 -> 2 -> 5/5), while real circuits stay flat-low (~0.02-0.17) at EVERY
  checkpoint. **Verdict: not overfitting** \u2014 longer training keeps helping
  in-distribution; the real-circuit gap is input coverage, present from the first
  checkpoint. (n=1 seed; evidence files eval_sac_ck*.json.)

### Diagnosis
1. **Narrow generalization, not memorization.** One-track SAC completes *different*
   synthetic layouts → it did not memorize its track. But its scan→action mapping
   is brittle: unfamiliar scan statistics (narrower walls, real spawns) produce
   degenerate actions (creep, then panic).
2. **A second, latent ceiling:** even a robust policy could not meet real hairpins
   (1.27 1/m) because the generator never produces them (0.54 cap). More tracks
   within the current generator settings do NOT cover that sharpness.
3. The reactive baseline has neither problem (no learned mapping) — which is why it
   currently wins on real circuits.

### Candidate fixes (ranked)
1. **Widen the generator.** Raise `TRACK_TURN_RATE` / allow smaller radii so the
   synthetic family covers real-corner sharpness (target max_k ≈ 1.3) and width
   variety. *Cheap, necessary*: without it the diversity sweep has a built-in
   ceiling. Freeze the new generator before the grid.
2. **Diversity sweep as designed (5/20/100 tracks)** — wider scan coverage should
   fix the spawn-level failure (the dominant observed failure mode).
3. **Domain randomization on observations** (scan noise/jitter/dropout during
   training) — widens the effective input manifold without new tracks. Cheap,
   well-precedented, orthogonal to the diversity axis.
4. **Dense centreline-following term** — keeps the car centred, buying margin
   against wall-position shift (also helps PPO's braking problem).
5. **Checkpoint-based selection if the overfit check says so** — if zero-shot laps
   peak at ~300-600k, adopt the best-generalizing checkpoint instead of the final.
6. **Curriculum over width/curvature** (narrow tracks mixed in) — directly targets
   the reproduced failure; confounds the clean diversity axis, so keep it as a
   diagnostic unless the clean sweep fails.

---

## What we should do next (proposed queue)
1. **Finish the checkpoint overfit diagnostic** (running; eval 100k..2M ckpts zero-shot).
2. **PPO n_envs=8 ablation** (2M, one run, frozen protocol) — the key PPO question.
3. **Generator widening** — measure the new max_k distribution; regenerate a pool
   that includes real-like sharpness and width; re-run one SAC single-track pilot
   and re-eval on real circuits.
4. **Domain-randomization ablation** for SAC (scan jitter), unseen-track eval.
5. **All-around clearance shaping** sweep (PPO and SAC).
6. **Then the grid**, with the frozen widened generator + the winning protocol
   choices, at ≥3 seeds with bootstrap CIs.

## Honesty notes
- All pilots are n=1; rankings above are hypotheses to test, not conclusions.
- Crash penalty (5 vs 40) must be frozen before the grid — unresolved.
- Main branch still carries the OLD reward; merge the reward branch or train from
  this folder (see TRAIN_README.md).
