"""ADDITIVE dense reward-shaping experiment (does NOT touch the frozen reward).

Motivation: the frozen reward only signals danger ONCE, at the crash (sparse,
after the fact). This adds a DENSE per-step signal BEFORE the crash, scaled by
speed, so the lesson becomes 'do not be fast near a wall' -> brake for corners.
This is the dense/positive-shaping counterpart to the sparse crash penalty.

    reward = frozen_progress_reward(unchanged)
           - shaping_w * (speed / SPEED_MAX) * max(0, (d_safe - forward_clearance)/d_safe)

forward_clearance = min LiDAR range over the front arc (metres, raw obs).
Eval (laps) is unchanged, so shaped vs baseline is directly comparable on laps.
"""
import numpy as np
from new_reward_wrapper import F1TenthSB3Wrapper, DEFAULT_MAX_EPISODE_STEPS, SPEED_MAX


class ShapedWrapper(F1TenthSB3Wrapper):
    def __init__(self, env, max_episode_steps=DEFAULT_MAX_EPISODE_STEPS, shaping_w=0.2, d_safe=2.0):
        super().__init__(env, max_episode_steps=max_episode_steps)
        self.shaping_w = float(shaping_w)
        self.d_safe = float(d_safe)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # flat obs: [0:108]=downsampled LiDAR ranges (m), [108]=linear_vel_x
        forward = float(np.min(obs[44:64]))          # front ~arc clearance (m)
        speed = abs(float(obs[108]))
        prox = max(0.0, (self.d_safe - forward) / self.d_safe)   # 0..1, 1 = at wall
        pen = self.shaping_w * (speed / SPEED_MAX) * prox
        reward -= pen
        info = dict(info)
        info["shaping_pen"] = pen
        info["forward_clearance"] = forward
        return obs, reward, terminated, truncated, info
