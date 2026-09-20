---
title: Glossary — domain terms
type: reference
updated: 2025-09-20
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
  nar7 finding proves coverage; overfitting-timing is still being tested.
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
- **Step budget** — total environment steps per training run. Fixed at 2M
  (grid) / 1M (pilots); a controlled variable, never tuned per algorithm.
- **Crash penalty** — the collision-gated reward penalty. Pilots ~40; team
  canonical 5.0. Frozen identically across cells.

## See also

- `overview.md` — how these terms fit the project
- `methodology.md` — protocol terms in context
