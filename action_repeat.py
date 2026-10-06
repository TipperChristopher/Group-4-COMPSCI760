"""Hold each policy action for several physics steps (action repeat / frame skip).

WHY

The policy currently issues a new action every physics step: 100 Hz of control
on a car ~0.5 m long that reaches 20 m/s. That has two consequences.

1. Credit assignment. PPO's advantage estimate decays with 1/(1 - gamma*lambda),
   which at gamma 0.99 / lambda 0.95 is ~16.8 steps = 0.168 s = about 1.7 m of
   track at 10 m/s. A decision whose payoff is a corner 30 m ahead lies far
   outside that window, so the gradient for it is effectively zero. Holding each
   action for N steps multiplies the window by N in *simulated* time without
   touching gamma or lambda.

2. Redundancy. Measured on the canonical track, one 100 Hz step changes the
   108-beam scan by ~1% of its own spread at 2-10 m/s (see
   _verify/obs_redundancy.py). So a fixed step budget is spent mostly on near
   duplicate observations, and consecutive samples are strongly correlated for
   the value function and advantage estimate.

PLACEMENT

This wrapper sits immediately above the raw gym env, so the action it repeats is
the *rescaled physical* action ([steering rad, speed m/s]) produced by
F1TenthSB3Wrapper, and the reward/progress accounting above it runs once per
decision over N physics steps. Correct placement therefore requires the caller
to divide the episode step limit by the repeat, so that the simulated time
budget is unchanged: DEFAULT_MAX_EPISODE_STEPS / repeat decisions of N steps
each still covers 3000 physics steps = 30 s. train.py does this.

Reward semantics are preserved: the wrapper sums the N per-step rewards, so the
total is still metres of centreline progress minus any crash penalty, charged
once because collision sets `terminated` and ends the repeat early.
"""

from __future__ import annotations

import gymnasium as gym


class ActionRepeat(gym.Wrapper):
    """Repeat each action for `repeat` physics steps, summing the reward."""

    def __init__(self, env: gym.Env, repeat: int = 1) -> None:
        super().__init__(env)
        self.repeat = int(repeat)
        if self.repeat < 1:
            raise ValueError(f"repeat must be >= 1, got {self.repeat}")

    def step(self, action):
        total_reward = 0.0
        obs = None
        terminated = truncated = False
        info: dict = {}
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            # Stop early on a real terminal state or a truncation so the caller
            # sees the episode end at the right step rather than N steps later.
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info
