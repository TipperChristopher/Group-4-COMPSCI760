"""Zero-shot evaluation of a trained agent on real F1 circuits.

Training happens entirely on procedurally generated synthetic tracks, so every
circuit evaluated here is unseen. There is no "trained" circuit in this plot.

The environment stack must match training exactly, including observation
normalisation. A policy trained on normalised observations and fed raw ones
produces meaningless numbers without raising, so the statistics saved next to
the model are loaded automatically when present.
"""

import sys
import types
import os
import csv
import glob
import math
import argparse
import importlib.util

# 1. The Bulletproof 'gym' Override (Must be at the very top)
if "gym" not in sys.modules:
    sys.modules["gym"] = types.ModuleType("gym")
sys.modules["gym"].__version__ = "0.0.0"

import gymnasium as gym
import f1tenth_gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULT_TRACKS = ["Spielberg", "Monza", "Silverstone"]
ALGOS = {"PPO": PPO, "SAC": SAC}


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
load_vecnormalize = _wrapper_module.load_vecnormalize
vecnormalize_path = _wrapper_module.vecnormalize_path

_track_pool_module = _load_local_module(
    "track_pool", os.path.join("src", "track_pool.py")
)
TrackPoolWrapper = _track_pool_module.TrackPoolWrapper


class CollisionRecorder(gym.Wrapper):
    """Copy the simulator's collision flag into info at episode end.

    Read inside the env's own step, before any vectorised auto-reset clears it.
    An episode can end either by crashing or by completing the lap count, and
    distinguishing the two is the point of the evaluation.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            info = dict(info) if info else {}
            try:
                info["collision"] = bool(self.env.unwrapped.collisions[0])
            except Exception:
                info["collision"] = None
        return obs, reward, terminated, truncated, info


def find_run_dir(args):
    """Locate the run directory holding final_model.zip."""
    if args.model_path:
        path = os.path.abspath(args.model_path)
        return os.path.dirname(path) if os.path.isfile(path) else path

    if args.diversity is not None and args.seed is not None:
        return os.path.join(
            _PROJECT_ROOT, "models",
            f"{args.algo}_{args.diversity}tracks_s{args.seed}",
        )

    # Nothing specified: fall back to the most recent run so that a bare
    # "python evaluate.py" still does something sensible.
    candidates = glob.glob(os.path.join(_PROJECT_ROOT, "models", "*", "final_model.zip"))
    if not candidates:
        raise SystemExit(
            "No trained model found. Pass --model-path, or --algo/--diversity/--seed, "
            "or train one first with train.py."
        )
    newest = max(candidates, key=os.path.getmtime)
    print(f"No run specified, using the most recent: {os.path.dirname(newest)}")
    return os.path.dirname(newest)


def infer_algo(run_dir, requested):
    """Trust an explicit --algo, otherwise read it off the directory name."""
    if requested:
        return requested
    name = os.path.basename(run_dir.rstrip("/\\"))
    for algo in ALGOS:
        if name.upper().startswith(algo):
            return algo
    raise SystemExit(
        f"Could not tell which algorithm produced {run_dir}. Pass --algo explicitly."
    )


def build_env(track_name, render, track_seed):
    """Recreate the training environment stack for a single circuit."""

    def _init():
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={"num_agents": 1, "timestep": 0.01, "map": track_name},
            render_mode="human" if render else None,
        )
        env = F1TenthSB3Wrapper(env)
        env = CollisionRecorder(env)
        # A one-track pool. Reused from training so the map and the spawn
        # sampler are guaranteed consistent on every reset.
        env = TrackPoolWrapper(env, tracks=[track_name], track_seed=track_seed)
        env = Monitor(env)
        return env

    return _init


def run_episode(venv, model, max_steps, deterministic, render):
    """Roll out one episode and return its statistics."""
    obs = venv.reset()
    total_reward = 0.0
    steps = 0
    outcome = "step_cap"
    collision = None

    while steps < max_steps:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = venv.step(action)
        total_reward += float(rewards[0])
        steps += 1

        if render:
            try:
                venv.env_method("render")
            except Exception:
                pass

        if dones[0]:
            info = infos[0]
            collision = info.get("collision")
            if collision:
                outcome = "crash"
            elif info.get("TimeLimit.truncated"):
                outcome = "truncated"
            else:
                outcome = "finished"
            break

    return {
        "return": total_reward,
        "length": steps,
        "outcome": outcome,
        "collision": collision,
    }


def evaluate_track(track, model, args, stats_path):
    """Run every episode for one circuit."""
    print(f"\n--- {track} ---")
    venv = DummyVecEnv([build_env(track, args.render, args.eval_seed)])

    if stats_path:
        venv = load_vecnormalize(venv, stats_path)

    episodes = []
    for i in range(args.episodes):
        seed = args.eval_seed + i
        try:
            venv.seed(seed)
        except Exception:
            pass
        result = run_episode(venv, model, args.max_steps, not args.stochastic, args.render)
        result["episode"] = i
        result["seed"] = seed
        episodes.append(result)
        print(
            f"  episode {i}: return={result['return']:>10,.2f}  "
            f"steps={result['length']:>6,}  outcome={result['outcome']}"
        )

    venv.close()
    return episodes


def write_csv(path, algo, diversity, seed, results):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "algo", "diversity", "run_seed", "track", "episode", "eval_seed",
            "return", "length", "outcome", "collision",
        ])
        for track, episodes in results.items():
            for ep in episodes:
                writer.writerow([
                    algo, diversity, seed, track, ep["episode"], ep["seed"],
                    f"{ep['return']:.4f}", ep["length"], ep["outcome"],
                    "" if ep["collision"] is None else int(ep["collision"]),
                ])
    return path


def make_plot(results, algo, diversity, out_path):
    tracks = list(results)
    means = [float(np.mean([e["return"] for e in results[t]])) for t in tracks]
    stds = [float(np.std([e["return"] for e in results[t]])) for t in tracks]

    # Every real circuit is unseen: training used synthetic tracks only.
    labels = [f"{t}\n(unseen)" for t in tracks]

    plt.figure(figsize=(9, 6))
    bars = plt.bar(labels, means, yerr=stds if any(stds) else None,
                   capsize=6, color="#4c72b0", edgecolor="black")

    pool = f"{diversity} synthetic track{'s' if diversity != 1 else ''}"
    plt.title(f"{algo} Zero-Shot Transfer ({pool})",
              fontsize=15, fontweight="bold", pad=15)
    plt.ylabel("Episodic Return", fontsize=12, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    span = max(means) - min(min(means), 0) or 1.0
    for bar, mean in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + span * 0.02,
                 f"{mean:,.1f}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Graph saved to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation on real F1 circuits."
    )
    parser.add_argument("--algo", type=str, choices=sorted(ALGOS))
    parser.add_argument("--diversity", type=int)
    parser.add_argument("--seed", type=int, help="Run seed used during training.")
    parser.add_argument("--model-path", type=str,
                        help="Run directory or .zip. Overrides the lookup above.")
    parser.add_argument("--tracks", nargs="+", default=DEFAULT_TRACKS)
    parser.add_argument("--episodes", type=int, default=1,
                        help="f1tenth's default reset places the car on a static "
                             "grid pose, so a greedy policy replays an identical "
                             "episode every time. Variance for the study should "
                             "come from multiple training seeds, not from repeated "
                             "evaluation. Raise this only with --stochastic.")
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=10000,
                        help="Per-episode cap. The env does not always terminate.")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample from the policy instead of acting greedily.")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--out", type=str, help="CSV output path.")
    parser.add_argument("--plot-path", type=str, default="generalization_results.png")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    run_dir = find_run_dir(args)
    args.algo = infer_algo(run_dir, args.algo)

    model_file = os.path.join(run_dir, "final_model.zip")
    if args.model_path and os.path.isfile(os.path.abspath(args.model_path)):
        model_file = os.path.abspath(args.model_path)
    if not os.path.exists(model_file):
        raise SystemExit(f"No model at {model_file}")

    # Observation normalisation: the presence of this file records which regime
    # the policy was trained in, so detection is safer than a flag.
    stats_path = vecnormalize_path(run_dir)
    if os.path.exists(stats_path):
        print(f"Loading normalisation statistics from {stats_path}")
    else:
        stats_path = None
        print(
            "\n"
            "WARNING: no vecnormalize.pkl in this run directory.\n"
            "Evaluating on raw observations. This is correct only for models\n"
            "trained before observation normalisation was added. If this model\n"
            "was trained with the current train.py, the results are invalid.\n"
        )

    print(f"Loading {args.algo} from {model_file}")
    model = ALGOS[args.algo].load(model_file)

    results = {}
    for track in args.tracks:
        results[track] = evaluate_track(track, model, args, stats_path)

    print("\n" + "=" * 62)
    print(f"ZERO-SHOT RESULTS  ({args.algo}, {run_dir})")
    print("=" * 62)
    width = max([len(t) for t in results] + [len("circuit")])
    print(f"  {'circuit':<{width}} {'mean return':>14} {'std':>10} {'crashes':>9}")
    for track, episodes in results.items():
        returns = [e["return"] for e in episodes]
        crashes = sum(1 for e in episodes if e["outcome"] == "crash")
        print(f"  {track:<{width}} {np.mean(returns):>14,.2f} {np.std(returns):>10,.2f} "
              f"{crashes:>6}/{len(episodes)}")
    print("=" * 62)
    print("All circuits are unseen. Training used synthetic tracks only.")

    # Guard against reporting a spread that does not exist. A static start pose
    # and a greedy policy replay the same episode, so a std of 0 over several
    # episodes is not evidence of stability, it is the same run counted twice.
    repeated = [t for t, eps in results.items() if len(eps) > 1]
    if repeated and all(np.std([e["return"] for e in results[t]]) == 0 for t in repeated):
        print(
            "\nNOTE: every episode on a circuit returned an identical result.\n"
            "The start pose is deterministic and the policy is greedy, so these\n"
            "are repeats of one rollout. Do not report them as n>1. Use separate\n"
            "training seeds for error bars, or --stochastic to sample the policy."
        )

    diversity = args.diversity if args.diversity is not None else "?"
    out = args.out or os.path.join(run_dir, "zero_shot_results.csv")
    print(f"\nPer-episode results written to {write_csv(out, args.algo, diversity, args.seed, results)}")

    if not args.no_plot:
        make_plot(results, args.algo, diversity, args.plot_path)


if __name__ == "__main__":
    main()