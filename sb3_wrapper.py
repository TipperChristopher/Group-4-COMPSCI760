from __future__ import annotations

import os
from turtle import speed

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

# Filename used for the running observation statistics that accompany a saved
# model. Evaluation must load these, because a policy trained on normalised
# observations sees garbage if handed raw ones.
VECNORMALIZE_FILENAME = "vecnormalize.pkl"


class F1TenthSB3Wrapper(gym.Wrapper):
    """
    A custom wrapper to make f1tenth_gym compatible with Stable-Baselines3.
    Downsamples LiDAR and flattens the dictionary observation into a 1D vector.
    """
    def __init__(self, env):
        super().__init__(env)
        
        # 108 beams + 5 ego state values = 113-dimensional vector
        self.obs_dim = 108 + 5
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        
        # SB3 expects flat continuous actions: [steering, speed]
        # Action limits: steering ~[-0.4189, 0.4189] rad, speed ~[-5.0, 20.0] m/s
        self.action_space = gym.spaces.Box(
            low=np.array([-0.4189, 1.0], dtype=np.float32), # Minimum speed is now 1.0 m/s
            high=np.array([0.4189, 20.0], dtype=np.float32), 
            dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process_obs(obs), info

    def step(self, action):
        formatted_action = np.expand_dims(action, axis=0)
        
        obs, reward, terminated, truncated, info = self.env.step(formatted_action)
        
        # --- REWARD SHAPING ---
        speed = obs['linear_vels_x'][0]
        steering = action[0]

        # Reward forward velocity, penalize erratic steering
        # Inside step() reward logic:
        custom_reward = float(reward) + (speed * 0.1) - (abs(steering) * 0.5)

        if speed < 1.0:
            custom_reward -= 0.5 # Constant penalty for cowardly driving
        return self._process_obs(obs), custom_reward, terminated, truncated, info

    def _process_obs(self, obs):
        scan = obs['scans'][0]
        
        # Downsample 1080 beams to 108 by taking every 10th beam
        downsampled_scan = scan[::10] 
        
        # Extract 5 ego state values
        ego_state = np.array([
            obs['linear_vels_x'][0],
            obs['linear_vels_y'][0],
            obs['ang_vels_z'][0],
            obs['poses_theta'][0],
            0.0 
        ], dtype=np.float32)
        
        # Concatenate LiDAR and vehicle state into a single 113-dim array
        flat_obs = np.concatenate([downsampled_scan, ego_state])
        return flat_obs.astype(np.float32)


# --------------------------------------------------------------------------
# Observation normalisation
# --------------------------------------------------------------------------
# The 113-d observation mixes LiDAR ranges (0-30 m), body velocities (-5 to
# 20 m/s) and a heading in radians. Those scales differ by an order of
# magnitude, which hurts PPO and SAC differently and would contaminate the
# algorithm comparison. Normalising observations identically for both puts
# them on equal footing.
#
# Rewards are deliberately left unnormalised: the reward is the dependent
# variable of the study, and VecNormalize's running return scaling would make
# the numbers incomparable across cells.


def wrap_vecnormalize(vec_env: VecEnv, training: bool = True) -> VecNormalize:
    """Apply observation-only normalisation. Used identically for PPO and SAC."""
    return VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,
        training=training,
    )


def vecnormalize_path(save_dir: str) -> str:
    """Canonical location of the statistics file for a run directory."""
    return os.path.join(save_dir, VECNORMALIZE_FILENAME)


def save_vecnormalize(vec_env, save_dir: str) -> str | None:
    """Save running observation statistics next to the model.

    Returns the path written, or None if the env is not normalised.
    """
    if not isinstance(vec_env, VecNormalize):
        return None
    os.makedirs(save_dir, exist_ok=True)
    path = vecnormalize_path(save_dir)
    vec_env.save(path)
    return path


def load_vecnormalize(vec_env: VecEnv, path: str) -> VecNormalize:
    """Restore saved statistics onto ``vec_env`` for evaluation.

    Freezes the statistics and disables reward normalisation so that evaluation
    measures the policy, not a moving normaliser.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No normalisation statistics at {path}. The policy was trained on "
            "normalised observations and cannot be evaluated without them."
        )
    normalised = VecNormalize.load(path, vec_env)
    normalised.training = False
    normalised.norm_reward = False
    return normalised