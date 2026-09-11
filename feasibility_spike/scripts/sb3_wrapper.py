"""Minimal single-agent SB3 adapter for f1tenth_gym v1.0.0.

The raw env gives:
  obs    = {"agent_0": {"scan": Box(num_beams,), <state features...>}}
  action = Box((num_agents, 2))  # [steer, speed]
  step() = (obs, reward_dict_or_scalar, terminated, truncated, info)

SB3 wants a flat Box observation, a flat Box action, and scalar reward.
This wrapper flattens agent_0's dict into one float32 vector (scan ++ state),
squeezes/expands the action, and unwraps reward to a scalar. MLP-friendly.
No reward engineering — passes the env's own reward straight through.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import f1tenth_gym  # noqa: F401  (registers f1tenth-v0)

# LiDAR scan + vehicle state = the project's stated observation.
STATE_FEATURES = ["pose_x", "pose_y", "delta", "linear_vel_x", "pose_theta"]
DEFAULT_FEATURES = ["scan"] + STATE_FEATURES


class F1TenthSB3(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, map_name="Spielberg", features=DEFAULT_FEATURES, render_mode=None,
                 scan_beams=None):
        super().__init__()
        self._features = features
        self._scan_beams = scan_beams  # if set, subsample scan to this many beams
        self.env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "map": map_name,
                "num_agents": 1,
                "observation_config": {"type": "features", "features": features},
            },
            render_mode=None,
        )
        # Probe one reset to learn the flattened observation size.
        obs, _ = self.env.reset()
        flat = self._flatten(obs)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat.shape[0],), dtype=np.float32
        )
        # action: drop the agent axis -> (2,); reuse the raw bounds
        raw_a = self.env.action_space
        self.action_space = spaces.Box(
            low=np.asarray(raw_a.low).reshape(-1).astype(np.float32),
            high=np.asarray(raw_a.high).reshape(-1).astype(np.float32),
            dtype=np.float32,
        )

    def _flatten(self, obs):
        a = obs["agent_0"]
        parts = []
        if "scan" in a:
            scan = np.asarray(a["scan"], dtype=np.float32).reshape(-1)
            if self._scan_beams and self._scan_beams < scan.shape[0]:
                idx = np.linspace(0, scan.shape[0] - 1, self._scan_beams).astype(int)
                scan = scan[idx]
            parts.append(scan)
        for k in self._features:
            if k == "scan":
                continue  # scan handled above; avoid double-counting
            parts.append(np.asarray(a[k], dtype=np.float32).reshape(-1))
        return np.concatenate(parts).astype(np.float32)

    @staticmethod
    def _scalar(x):
        if isinstance(x, dict):
            return float(sum(float(v) for v in x.values()))
        return float(np.asarray(x).reshape(-1)[0])

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._flatten(obs), info

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(1, 2)
        obs, reward, terminated, truncated, info = self.env.step(a)
        return (
            self._flatten(obs),
            self._scalar(reward),
            bool(np.asarray(terminated).reshape(-1)[0]),
            bool(np.asarray(truncated).reshape(-1)[0]),
            info,
        )

    def close(self):
        self.env.close()
