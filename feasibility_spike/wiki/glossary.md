---
title: Glossary — domain terms
type: reference
updated: 2026-10-23
sources:
  - project usage (mixed)
---

# Glossary

- **Zero-shot generalization** — evaluating a trained policy on tracks it
  never saw, with no fine-tuning. The project's headline metric.
- **Training diversity (N)** — number of distinct tracks in the training
  pool: 1 / 5 / 20 / 100. The controlled variable of the grid.
- **In-distribution / out-of-distribution** — an eval track that is (or is
  not) statistically similar to the training tracks. Synthetic held-out
  tracks are in-distribution; real circuits are out-of-distribution.
- **Coverage vs overfitting** — *coverage*: the policy never saw a pattern in
  training (fails at any checkpoint). *Overfitting*: the policy got worse at
  generalizing as training went on (early checkpoint would do better). The
  nar7 finding proves coverage; the checkpoint probe **ruled out** overfitting
  (generalization rises monotonically — see `results.md`). *(2026-10: the uncovered
  variable is width; checkpoint scores are volatile rather than declining.)*
- **Memorization vs competence** — whether the policy learned the training
  *distribution* (brittle outside it) or a general driving *rule* (transfers).
  The project's framing question.
- **PPO** — Proximal Policy Optimization, on-policy. Collects a batch,
  updates, discards. With one env the batch is one correlated trajectory →
  high variance, oscillation under our protocol.
- **SAC** — Soft Actor-Critic, off-policy. Uses a replay buffer, so it can
  reuse and stabilize on past experience → completes laps in our pilots.
- **On-policy / off-policy** — whether updates use only fresh data
  (on-policy, PPO) or stored past experience (off-policy, SAC).
- **Gap-follower** — reactive hand-written baseline: find the widest LiDAR
  gap and steer into it. No learning. Parameter `safe_threshold = 20`.
- **Fractional laps** — the eval metric: 0.40 = reached 40 % of a lap.
  Graded, not binary completion.
- **VecNormalize** — SB3 wrapper normalizing observations (and optionally
  reward). Here `norm_obs=True`, `norm_reward=False`. Algorithm-specific
  effect: no help for PPO, enables SAC. Must be loaded in eval mode
  (`training=False`) or observations are garbage (`incidents.md`).
- **cl_grid_static / rl_grid_static** — spawn modes. `cl` = centreline grid
  (our pinned choice, 0.00 m offset). `rl` = gym default (0.79–0.81 m off
  centre on real circuits).
- **Spawn clearance / min_scan** — the minimum LiDAR return at spawn ≈ wall
  distance. Real circuits 1.05–1.09 m; synthetic 1.49–1.53 m; narrow (nar7)
  1.04–1.08 m.
- **Creep** — the SAC failure mode at off-distribution spawns: crawling at
  ~1 m/s instead of driving, before an early crash.
- **Curvature (max)** — sharpest corner on a track. Synthetic capped at
  ~0.544 (`TRACK_TURN_RATE = 0.31`); Spielberg 1.273, Silverstone 0.938.
  *(2026-10: the max is misleading. Real circuits exceed training curvature at only 0.95% of
  points; width, not sharpness, is the out-of-distribution axis — `sim-to-real-width.md`.)*
- **Step budget** — total environment steps per training run. Fixed at 2M
  (grid) / 1M (pilots); a controlled variable, never tuned per algorithm.
- **Crash penalty** — the collision-gated reward penalty. Pilots ~40; team
  canonical 5.0. Frozen identically across cells. *(2026-10: recommendation is 5;
  what it is worth depends on γ — see `reward.md`.)*

### Terms added 2026-10-06

- **γ (gamma, discount factor)** — how much a reward one step later counts:
  reward *k* steps ahead is worth γ^k now. SB3 default 0.99.
- **Value horizon** — roughly 1/(1−γ) steps: γ 0.99 → 100 steps = **1 s** at 100 Hz;
  γ 0.999 → 1000 steps = 10 s. At 1 s, a crash more than a second ahead barely registers.
- **GAE λ / credit window** — PPO's advantage estimate looks ~1/(1−γλ) steps ahead
  (≈17–20 steps at λ 0.95). Widening it (λ 0.99) did **not** help, so it was not the bottleneck.
- **Basin** — the strategy a learner settles into. At γ 0.99, "fast and crash" and
  "fast and survive" are nearly tied; PPO landed in the first and SAC in the second.
- **n_steps vs batch_size (PPO)** — `n_steps` = how many env steps are collected
  before each update (rollout length; 2048 by default, **8192 in the fix**). `batch_size` =
  minibatch size for each gradient step (64, **never changed** in the fix). `n_epochs` =
  passes over the rollout (10). Updates per rollout = n_steps / batch_size × n_epochs.
- **Exploration collapse** — PPO's state-independent action std shrinking to ~0.04–0.06
  (on a [−1, 1] action range): the policy stops trying different actions. Measured from
  `train/std` in `progress.csv`.
- **approx_kl / clip_fraction** — how far one PPO update moved the policy, and the share of samples
  where the trust-region clip was active. Healthy ≈ 0.01–0.03 / 0.1–0.2; collapsed PPO 0.25–0.38 / 0.5+.
- **Action repeat (frame skip)** — hold each action for N physics steps. Tested and refuted.
- **Training-log lap rate vs evaluated lap rate** — the first is the stochastic policy on its own
  training tracks (`monitor_0.monitor.csv`); the second is deterministic on held-out tracks.
  They differ by 2–5×. Always label which one.
- **Transfer ratio** — held-out evaluated lap rate ÷ late-training lap rate. SAC seed 0:
  0.13 / 0.47 / 0.57 / 0.76 for d1/5/20/100.
- **Canonical pool / manifest** — seed-123 `synthetic_track_0..99` + val/test/narrow/real
  splits with checksums in `tracks/manifest.json`. Same names ≠ same tracks; verify.
- **narrowA / narrowB** — test-track centrelines with half-width 1.07 m / 0.75 m (test 1.47 m).
- **Half-width** — distance from centreline to wall. Training 1.430–1.474 m everywhere; real median 1.075 m.
- **Noise floor (checkpoint)** — how much a score moves between checkpoints of one run with
  nothing changed. About half the between-cell differences in the seed-0 grid.
- **Distinct spawns** — `cl_grid_static` maps eval seeds onto a few start poses: 5 seeds →
  3 distinct, 30 seeds → ~9.

### Terms added 2026-10-23 (final grids and analysis)

- **Grid A / grid B** — the two pre-registered final grids. A: SB3 defaults, crash penalty 40 (branch `final-grid`).
  B: penalty 5; PPO γ 0.999, n_steps 8192, ent_coef 0.01; SAC defaults (branch `final-grid-v2`). Same cells and seeds.
- **vw_train / vw_val / vw_test** — the varied-width splits (half-width 0.60–1.50 m): 100 training tracks (nested pools),
  10 validation tracks (every tuning and checkpoint decision), 20 unseen test tracks.
- **`laps` vs `completed_lap`** — in the test CSVs, `laps` is fractional progress (a finished lap reads ≈0.999) and
  `completed_lap` is the lap flag. "Fractional laps" means the mean of `laps`; "lap %" means the share of `completed_lap`.
- **Selection ladder** — a run's vw_val scores at 400k / 800k / 1.2M / 1.6M / final; the selection rule picks its maximum.
  **Near-tie:** best minus second-best < 0.02 laps.
- **Seed-matched comparison** — comparing two algorithms at the same seed and diversity level (e.g. "PPO ahead in 20/24").
- **Track-level Wilcoxon signed-rank** — a paired test over tracks of two agents' per-track scores. It treats tracks as the
  sample, so it describes the specific trained agents, not the algorithm (pseudo-replication if read that way).
- **Holm correction** — a step-down multiple-testing correction; the corrected p depends on which tests form the family.
- **Spearman ρ** — rank correlation; used for "validation ranks agents like the test does" and width-vs-score.
- **Per-track oracle** — the better of two policies on each track, chosen in hindsight (an upper bound for a hybrid).
- **Sim-to-real drop** — the change in fractional laps from vw_test to real circuits for the same agent.

## See also

- `overview.md` — how these terms fit the project
- `methodology.md` — protocol terms in context
- `ppo-diagnosis.md`, `sim-to-real-width.md` — where the 2026-10 terms are used
