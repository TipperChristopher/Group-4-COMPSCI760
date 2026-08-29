import sys
import types
import os
import argparse

# 1. The Bulletproof 'gym' Override (Must be at the very top)
if "gym" not in sys.modules:
    sys.modules["gym"] = types.ModuleType("gym")
sys.modules["gym"].__version__ = "0.0.0"

import gymnasium as gym
import f1tenth_gym 
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from wrappers import F1TenthSB3Wrapper

def make_env(track_name, seed):
    """Utility function to spawn isolated environments for SubprocVecEnv."""
    def _init():
        env = gym.make("f1tenth_gym:f1tenth-v0", config={
            "num_agents": 1,
            "timestep": 0.01,
            "map": track_name
        })
        env = F1TenthSB3Wrapper(env)
        env.action_space.seed(seed)
        return env
    return _init

def main():
    # Setup command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, choices=["PPO", "SAC"], required=True)
    parser.add_argument("--diversity", type=int, choices=[1, 5, 20, 100], required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    # Dynamically select tracks from the synthetic pool
    track_pool = [f"synthetic_track_{i}" for i in range(args.diversity)]
    print(f"Initializing {args.algo} across {args.diversity} tracks (Seed {args.seed})...")

    # Spawn parallel environments
    env_fns = [make_env(t, args.seed + i) for i, t in enumerate(track_pool)]
    vec_env = DummyVecEnv(env_fns)

    # Instantiate the selected algorithm
    if args.algo == "PPO":
        model = PPO("MlpPolicy", vec_env, verbose=1, seed=args.seed)
    else:
        model = SAC("MlpPolicy", vec_env, verbose=1, seed=args.seed)

    # Setup Checkpointing in organized folders
    save_dir = f"./models/{args.algo}_{args.diversity}tracks_s{args.seed}/"
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100000, 
        save_path=save_dir,
        name_prefix=f"{args.algo}_checkpoint"
    )

    print(f"Starting {args.algo} training loop for 2M steps...")
    model.learn(total_timesteps=2000000, callback=checkpoint_callback)
    
    # Save the final model
    final_path = os.path.join(save_dir, "final_model")
    model.save(final_path)
    print(f"Training complete! Model saved to {final_path}.zip")

if __name__ == "__main__":
    main()