# Why PPO is unstable and SAC completes — reward + algorithm analysis

Reviewable note for the "why it's unstable / what we'd change" slide. All claims
are grounded in `results/learning_curve_cp40_vn_1000000.json` (PPO) and
`results/_backup_prelog/sac_vn_7of10_completed.json` (SAC). n=1 seed — treat the
*direction* as solid, the exact numbers as one sample.

## 0. The single most important point: progress reward = speed reward
The reward is `1.0 x metres-advanced-along-centreline` per step, `TIME_COST=0`,
`-40` once on crash. Because the episode has a **fixed step budget** (3000 steps
= 30 s), the only way to bank more reward is to **cover more track in that time**
— i.e. **go faster**. The wrapper says this itself:

> "inside a fixed step budget, the only way to earn more is to cover more track,
> so faster is monotonically better ... the step limit already supplies the time
> pressure, since total progress is what a bounded episode can accumulate."

So "we reward distance, not speed" is not quite right in effect: with a capped
horizon, **distance-per-episode is maximised by speed.** The agent speeding up is
*correct* under this reward. The failure is that it can't do the *controlled*
version (fast on straights, slow in corners).

## 1. Why the speed keeps rising even though it crashes
RL does not react to the *sign* of reward; it reacts to *advantage* (better or
worse than expected). A crash is one `-40`, but the ~1000 steps of fast driving
before it were all positively rewarded. Net episode return is often still
positive, so the lesson is "fast is mostly good, occasionally punished." Measured:
crash-speed **rises over training, 2.3 -> 7.0 m/s** — it learns to go faster, not
to brake. "Stop" is never a good option either: standing still scores exactly 0.

## 2. Why PPO oscillates and degrades (it is NOT mainly "updates too big")
PPO already limits update size: `clip_range=0.2` caps how far the policy moves per
update. The instability comes from three things combined:
- **A knife-edge reward landscape.** Just-slow = survives but crawls (low reward);
  just-fast = more reward but crashes. The stable middle *is* the cornering skill,
  which is hard to find, so there is no smooth valley to settle into.
- **On-policy with no memory.** PPO learns only from its *latest* rollout
  (`n_steps=2048`), then **discards it**. Even small, clipped updates on a stream
  of only-recent data let the policy **drift** — like editing a document while only
  able to see the last paragraph. The good policy (0.344 laps @ 300k) gets
  overwritten -> catastrophic forgetting.
- **High-variance single-env signal.** `n_envs=1` and a crash swings returns by
  40+, so each gradient estimate is noisy and can knock the policy off the edge.

Result (PPO+VN): reward oscillates mean 9.7, **std 14.3** (spread > mean), 3
sign-flips, best 0.344 @ 300k, **degrades to 0.098** by 1M. Final policy crashes
at 11.6 m/s **accelerating into** a corner sharper than 90% of the track.

## 3. Correction about the LiDAR (I overstated a sensing limit earlier)
The LiDAR sees walls metres ahead — **sensing is adequate.** SAC, on the same
observations, learns exactly the "safe speed near a wall, brake for the corner"
behaviour. So the problem is **not** that it can't perceive the corner in time;
it's that PPO can't **retain** the braking policy once found. Information present
!= skill learned-and-kept.

## 4. Why "more training" does not make it safer
- **Safer is not what it optimises.** Nothing rewards caution or margin, so more
  training makes it faster / closer to the cliff (crash-speed 2.3 -> 7.0 m/s),
  not safer.
- **More steps = more chances to fall off** the good policy, not to consolidate
  it. Confirmed separately: 3M steps did not beat 1M (saturates ~500k, then
  wanders).

## 5. Why SAC is different (memory, not self-awareness)
SAC keeps a **replay buffer** — it reuses old good/bad transitions for many
updates, so one noisy batch cannot erase what it learned. Plus a value function
and controlled (entropy) exploration. Result (SAC+VN): reward climbs **smoothly**
-1 -> 35 -> 70 -> 104 -> 122 -> 140 -> **159**, and it **completes 2 laps** at
400k/600k/700k while getting **faster** (lap time 40.3 -> 33.5 s) *without
crashing*. That is the controlled cornering PPO never kept. SAC has the *same*
safety-blind reward — it is a better optimiser of a flawed reward, not immune to it.

## 6. Exactly how PPO is trained (for reference)
`PPO("MlpPolicy", seed=0, ent_coef=0.0, device=cpu)` on a single `DummyVecEnv`
with `VecNormalize(norm_obs=True, norm_reward=False)`. SB3 defaults, none
overridden: `lr=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99,
gae_lambda=0.95, clip_range=0.2, vf_coef=0.5, max_grad_norm=0.5`; MlpPolicy =
2x64 tanh. Train episode cap 3000 steps.

## 7. What the racing-RL literature does (the survey result)
- Our reward is the **canonical progress-only reward** (Brunnbauer et al., ICRA
  2022). Progress-only is **known to plateau below one lap** for model-free RL —
  exactly what we see. **Good news:** progress-along-centreline *is* the standard
  positive **velocity-projection** term (metres advanced = speed projected on the
  track), so our core reward is already the recommended positive shaping.
- **`TIME_COST=0` was correct** — a per-step time cost creates a "suicidal agent"
  that crashes early to stop the bleed.
- What's typically added for full laps (and is missing here): a **dense safety /
  cross-track term** (stay off the walls / near the racing line), and a
  **completion bonus**. The field prefers **positive, dense** shaping (reward the
  right behaviour every step) over **sparse, negative** shaping (punish the crash
  once), because dense signals give a gradient toward the skill.

## 8. Experiment running now: dense positive shaping
`shaped_reward_wrapper.py` adds a per-step term:
`- w * (speed/SPEED_MAX) * max(0, (d_safe - forward_clearance)/d_safe)`, w=0.2,
d_safe=2 m. This makes being **fast near a wall** costly *before* the crash — a
dense pre-crash gradient that says "slow down near walls," which the sparse -40
cannot teach. Run: `learning_curve.py --algo ppo --vecnormalize --shaping 0.2`.

**RESULT (hypothesis NOT supported):**

| | best | final | mean laps | ep_rew std | ep_len | crashed | final crash |
|---|---|---|---|---|---|---|---|
| PPO baseline | 0.344 | 0.098 | 0.184 | 14.3 | 1021 | 5/10 | 11.6 m/s @ sharp corner |
| PPO +shaping | 0.580 | 0.247 | 0.265 | 12.0 | 487 | 7/10 | 15.8 m/s @ NON-sharp |

Shaping reached a higher *peak* and better mean laps, but **crashed more (7/10)**,
episodes got **shorter** (crashing earlier), variance barely changed, and it
**still oscillated and degraded**. The final policy crashes *faster* (15.8 m/s) on
a non-sharp section: the forward-only penalty was **gamed** — the agent keeps the
front clear, floors it, and clips walls **sideways**. Lesson: (1) you cannot
reward-shape away PPO's *forgetting* — the instability is algorithmic, not just
reward-sparsity; (2) shaping is easily gamed (forward-only -> side crashes), so a
future safety term must use **all-around clearance** and be tuned. This reinforces
that the real lever is the **algorithm (SAC's memory)**, not the reward.
(Phase-2 diagnostic; the main PPO-vs-SAC grid keeps the frozen reward. n=1.)

## 9. Caveats
- n=1 seed everywhere — need >=3 for confidence intervals before any of this is a
  reportable *finding* vs a *sample*.
- Shaping weight (0.2) and d_safe (2 m) are first guesses, not tuned.
- **Single-env caveat (matters for Q&A).** `n_envs=1` is *correct* — it is the
  team's fix for the "updates scaled with diversity" bug. But on-policy PPO with a
  single env is inherently higher-variance than PPO with 8-16 parallel envs, so part
  of PPO's oscillation is the frozen single-env constraint, which penalises on-policy
  PPO far more than off-policy SAC (replay buffer, barely affected). Fair claim:
  *"under our frozen single-env protocol PPO is unstable and SAC is not"* — NOT
  *"PPO is fundamentally broken."* If asked "would more parallel envs help PPO?":
  probably yes, but n_envs is fixed by the protocol.
- Setup audited (`_verify_ppo.py`): obs (113,), act (2,) in [-1,1], n_envs=1,
  norm_obs=True/norm_reward=False, standstill reward=0, penalty=40, SB3 defaults.
  The oscillation is a real property of the policy, not a config artifact.
