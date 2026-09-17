import sys
import types
import os
import math
import argparse
import importlib.util

# 1. The Bulletproof 'gym' Override (Must be at the very top)
if "gym" not in sys.modules:
    sys.modules["gym"] = types.ModuleType("gym")
sys.modules["gym"].__version__ = "0.0.0"

import gymnasium as gym
import f1tenth_gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULT_TOTAL_TIMESTEPS = 5_000_000
# Fixed default so that every cell of the grid, and both algorithms, draw the
# same tracks unless the track seed is overridden explicitly.
DEFAULT_TRACK_SEED = 0


def _load_local_module(module_name, relative_path):
    """Import a project file by path, independent of the current directory."""
    path = os.path.join(_PROJECT_ROOT, relative_path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not find {module_name} at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_wrapper_module = _load_local_module("sb3_wrapper", "sb3_wrapper.py")
F1TenthSB3Wrapper = _wrapper_module.F1TenthSB3Wrapper
wrap_vecnormalize = _wrapper_module.wrap_vecnormalize
save_vecnormalize = _wrapper_module.save_vecnormalize
DEFAULT_MAX_EPISODE_STEPS = _wrapper_module.DEFAULT_MAX_EPISODE_STEPS

_track_pool_module = _load_local_module(
    "track_pool", os.path.join("src", "track_pool.py")
)
TrackPoolWrapper = _track_pool_module.TrackPoolWrapper


def make_env(track_pool, seed, track_seed, stream=0, cache_size=8,
             max_episode_steps=None):
    """Build one environment that cycles through the whole track pool.

    The pool is handled inside a single environment rather than by spawning one
    environment per track. See src/track_pool.py for why: the number of
    Stable-Baselines3 updates is inversely proportional to n_envs, and
    f1tenth_gym shares one ray-casting engine across every environment in a
    process.
    """

    def _init():
        # The map given here is only the one loaded at construction time.
        # TrackPoolWrapper replaces it on every reset.
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "num_agents": 1,
                "timestep": 0.01,
                "map": track_pool[0],
            },
        )
        env = F1TenthSB3Wrapper(
            env,
            max_episode_steps=(
                DEFAULT_MAX_EPISODE_STEPS if max_episode_steps is None
                else max_episode_steps
            ),
        )
        env = TrackPoolWrapper(
            env,
            tracks=track_pool,
            track_seed=track_seed,
            stream=stream,
            cache_size=cache_size,
        )
        # Monitor records episode returns, the track each episode ran on, and
        # how far around it the car actually got.
        env = Monitor(env, info_keywords=("track_name", "laps", "progress_m"))
        env.action_space.seed(seed)
        return env

    return _init


def expected_updates(model, algo, total_timesteps, n_envs):
    """Describe how much learning the run will actually do.

    This is the quantity the original one-env-per-track setup silently varied
    across diversity levels, so it is worth printing before every run.
    """
    if algo == "PPO":
        rollout = model.n_steps * n_envs
        iterations = total_timesteps // rollout
        minibatches = math.ceil(rollout / model.batch_size)
        gradient_steps = iterations * model.n_epochs * minibatches
        return [
            ("rollout buffer", f"{rollout:,} steps (n_steps={model.n_steps})"),
            ("policy iterations", f"{iterations:,}"),
            ("gradient updates", f"{gradient_steps:,}"),
        ]

    train_freq = model.train_freq
    unit = getattr(train_freq.unit, "value", str(train_freq.unit))
    if unit != "step":
        return [
            ("gradient updates", f"depends on episode length (train_freq={train_freq})"),
        ]

    env_steps_per_train = train_freq.frequency * n_envs
    learning_steps = max(0, total_timesteps - model.learning_starts)
    n_trains = learning_steps // env_steps_per_train
    per_train = (
        model.gradient_steps if model.gradient_steps > 0 else train_freq.frequency
    )
    return [
        ("learning starts", f"{model.learning_starts:,} steps"),
        ("train calls", f"{n_trains:,}"),
        ("gradient updates", f"{n_trains * per_train:,}"),
    ]


def print_startup_summary(args, model, track_pool, n_envs):
    """Guard rail: these numbers must match across cells except for diversity."""
    rows = [
        ("algo", args.algo),
        ("diversity (pool size)", f"{args.diversity} tracks"),
        ("n_envs", str(n_envs)),
        ("track seed", str(args.track_seed)),
        ("run seed", str(args.seed)),
        ("total timesteps", f"{args.total_timesteps:,}"),
        ("max episode steps", f"{args.max_episode_steps:,}"),
    ]
    rows += expected_updates(model, args.algo, args.total_timesteps, n_envs)

    width = max(len(k) for k, _ in rows)
    line = "=" * 62
    print(line)
    print("RUN CONFIGURATION")
    print(line)
    for key, value in rows:
        print(f"  {key.ljust(width)} : {value}")
    preview = ", ".join(track_pool[:4])
    if len(track_pool) > 4:
        preview += f", ... (+{len(track_pool) - 4} more)"
    print(f"  {'track pool'.ljust(width)} : {preview}")
    print(line)
    print(
        "Gradient updates and total timesteps must be identical across all\n"
        "diversity levels. Only the pool size above should change."
    )
    print(line)


def main():
    # Setup command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, choices=["PPO", "SAC"], required=True)
    parser.add_argument("--diversity", type=int, choices=[1, 5, 20, 100], required=True)
    parser.add_argument("--seed", type=int, required=True,
                        help="Run seed: network init, action sampling, torch.")
    parser.add_argument("--track-seed", type=int, default=DEFAULT_TRACK_SEED,
                        help="Seed controlling which tracks are selected and in "
                             "what order. Independent of --seed so that PPO and "
                             "SAC train on identical tracks.")
    parser.add_argument("--n-envs", type=int, default=1,
                        help="Number of parallel environments. Kept at 1 so the "
                             "number of learning updates is constant across "
                             "diversity levels. NOT tied to --diversity.")
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--track-cache-size", type=int, default=8,
                        help="Tracks held in memory to keep reset() cheap. "
                             "Roughly 20 MB each.")
    parser.add_argument("--max-episode-steps", type=int,
                        default=DEFAULT_MAX_EPISODE_STEPS,
                        help="Episode step limit, returned as truncated (not "
                             "terminated). f1tenth_gym never truncates on its "
                             "own, so without this a stalled policy runs "
                             "forever, reset() never fires and the track pool "
                             "never advances. Default 3000 is 30 s of sim time, "
                             "about one lap of a synthetic track at 6 m/s.")
    args = parser.parse_args()

    if args.n_envs < 1:
        parser.error("--n-envs must be at least 1")

    # The pool grows with diversity; the environment count does not.
    track_pool = [f"synthetic_track_{i}" for i in range(args.diversity)]

    n_envs = args.n_envs
    env_fns = [
        make_env(
            track_pool,
            seed=args.seed + i,
            track_seed=args.track_seed,
            stream=i,
            cache_size=args.track_cache_size,
            max_episode_steps=args.max_episode_steps,
        )
        for i in range(n_envs)
    ]

    if n_envs == 1:
        vec_env = DummyVecEnv(env_fns)
    else:
        # f1tenth_gym stores its ray-casting engine in a class attribute shared
        # by every environment in a process, so several environments in one
        # process would all raycast against the same map. Separate processes
        # are the only way to run different maps concurrently.
        print(
            f"n_envs={n_envs}: using SubprocVecEnv because f1tenth_gym shares "
            "its scan simulator across environments within a process."
        )
        vec_env = SubprocVecEnv(env_fns)

    # Identical observation normalisation for both algorithms. Rewards are left
    # untouched because reward is the measured outcome of the experiment.
    vec_env = wrap_vecnormalize(vec_env, training=True)

    # Instantiate the selected algorithm
    if args.algo == "PPO":
        model = PPO("MlpPolicy", vec_env, verbose=1, seed=args.seed, device="cpu")
    else:
        model = SAC("MlpPolicy", vec_env, verbose=1, seed=args.seed)

    print_startup_summary(args, model, track_pool, n_envs)

    # Setup Checkpointing in organized folders
    save_dir = f"./models/{args.algo}_{args.diversity}tracks_s{args.seed}/"
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 100000 // n_envs),
        save_path=save_dir,
        name_prefix=f"{args.algo}_checkpoint",
        save_vecnormalize=True,
    )

    print(f"Starting {args.algo} training loop for {args.total_timesteps:,} steps...")
    model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback)

    # Save the final model alongside the normalisation statistics it needs.
    final_path = os.path.join(save_dir, "final_model")
    model.save(final_path)
    stats_path = save_vecnormalize(vec_env, save_dir)
    vec_env.close()

    print(f"Training complete! Model saved to {final_path}.zip")
    if stats_path:
        print(f"Normalisation statistics saved to {stats_path}")


if __name__ == "__main__":
    main()
