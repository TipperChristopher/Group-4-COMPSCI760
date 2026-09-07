import gymnasium as gym
import numpy as np

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
            low=np.array([-0.4189, -5.0], dtype=np.float32), 
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
        custom_reward = float(reward) + (speed * 0.1) - (abs(steering) * 0.5)
        
        # Terminal crash penalty
        if terminated:
            custom_reward -= 5.0
            
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