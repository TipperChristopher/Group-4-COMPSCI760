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


def build_env(track_name, render, track_seed, max_episode_steps):
    """Recreate the training environment stack for a single circuit."""

    def _init():
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={"num_agents": 1, "timestep": 0.01, "map": track_name},
            render_mode="human" if render else None,
        )
        # The wrapper supplies the step limit, the collision flag and the
        # lap/progress counters used below.
        env = F1TenthSB3Wrapper(env, max_episode_steps=max_episode_steps)
        # A one-track pool. Reused from training so the map and the spawn
        # sampler are guaranteed consistent on every reset.
        env = TrackPoolWrapper(env, tracks=[track_name], track_seed=track_seed)
        env = Monitor(env)
        return env

    return _init


def run_episode(venv, model, max_steps, deterministic, render):
    """Roll out one episode and return its statistics.

    The primary quantity is laps completed, not return. Return ranks a crawler
    above a fast driver who crashes, because the old reward paid for elapsed
    time; laps measure what the study actually asks about.
    """
    obs = venv.reset()
    total_reward = 0.0
    steps = 0
    outcome = "step_cap"
    collision = None
    laps = 0.0
    progress_m = 0.0

    while steps < max_steps:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = venv.step(action)
        total_reward += float(rewards[0])
        steps += 1

        # Read every step, so the counters are correct whether the episode ends
        # by crashing, by truncation, or by the loop hitting max_steps.
        info = infos[0]
        laps = float(info.get("laps", laps))
        progress_m = float(info.get("progress_m", progress_m))

        if render:
            try:
                venv.env_method("render")
            except Exception:
                pass

        if dones[0]:
            collision = info.get("collision")
            if collision:
                outcome = "crash"
            elif info.get("TimeLimit.truncated"):
                outcome = "timeout"
            else:
                outcome = "finished"
            break

    return {
        "return": total_reward,
        "length": steps,
        "outcome": outcome,
        "collision": collision,
        "laps": laps,
        "progress_m": progress_m,
        "completed_lap": laps >= 1.0,
    }


def evaluate_track(track, model, args, stats_path):
    """Run every episode for one circuit."""
    print(f"\n--- {track} ---")
    venv = DummyVecEnv([
        build_env(track, args.render, args.eval_seed, args.max_steps)
    ])

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
            f"  episode {i}: laps={result['laps']:>6.2f}  "
            f"dist={result['progress_m']:>8.1f} m  "
            f"steps={result['length']:>6,}  outcome={result['outcome']:<8} "
            f"(return {result['return']:,.2f})"
        )

    venv.close()
    return episodes


def write_csv(path, algo, diversity, seed, results):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "algo", "diversity", "run_seed", "track", "episode", "eval_seed",
            "laps", "completed_lap", "progress_m", "return", "length",
            "outcome", "collision",
        ])
        for track, episodes in results.items():
            for ep in episodes:
                writer.writerow([
                    algo, diversity, seed, track, ep["episode"], ep["seed"],
                    f"{ep['laps']:.4f}", int(ep["completed_lap"]),
                    f"{ep['progress_m']:.3f}",
                    f"{ep['return']:.4f}", ep["length"], ep["outcome"],
                    "" if ep["collision"] is None else int(ep["collision"]),
                ])
    return path


def make_plot(results, algo, diversity, out_path):
    """Plot laps completed. Return is deliberately not the headline number."""
    tracks = list(results)
    means = [float(np.mean([e["laps"] for e in results[t]])) for t in tracks]
    stds = [float(np.std([e["laps"] for e in results[t]])) for t in tracks]

    # Every real circuit is unseen: training used synthetic tracks only.
    labels = [f"{t}\n(unseen)" for t in tracks]

    plt.figure(figsize=(9, 6))
    bars = plt.bar(labels, means, yerr=stds if any(stds) else None,
                   capsize=6, color="#4c72b0", edgecolor="black")
    plt.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.5,
                label="one full lap")
    plt.legend(loc="upper right")

    pool = f"{diversity} synthetic track{'s' if diversity != 1 else ''}"
    plt.title(f"{algo} Zero-Shot Transfer ({pool})",
              fontsize=15, fontweight="bold", pad=15)
    plt.ylabel("Laps Completed", fontsize=12, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    span = max(max(means), 1.0) or 1.0
    for bar, mean in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + span * 0.02,
                 f"{mean:.2f}", ha="center", va="bottom",
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
    parser.add_argument("--max-steps", type=int, default=15000,
                        help="Per-episode cap, also passed to the wrapper as its "
                             "truncation limit. Real circuits are long: "
                             "Silverstone is 458 m, needing about 9,200 steps for "
                             "one lap at 5 m/s, so a small cap would make the lap "
                             "completion metric read zero for reasons unrelated "
                             "to the policy.")
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
    print("  PRIMARY METRIC: lap completion. Return is secondary and is shown")
    print("  only for continuity; it rewards elapsed time as much as speed.")
    print()
    print(f"  {'circuit':<{width}} {'lap rate':>9} {'mean laps':>10} "
          f"{'dist (m)':>10} {'crashes':>9} {'return':>12}")
    all_laps, all_completed = [], []
    for track, episodes in results.items():
        laps = [e["laps"] for e in episodes]
        completed = [e["completed_lap"] for e in episodes]
        crashes = sum(1 for e in episodes if e["outcome"] == "crash")
        all_laps += laps
        all_completed += completed
        print(f"  {track:<{width}} "
              f"{f'{sum(completed)}/{len(completed)}':>9} "
              f"{np.mean(laps):>10.2f} "
              f"{np.mean([e['progress_m'] for e in episodes]):>10.1f} "
              f"{crashes:>6}/{len(episodes)} "
              f"{np.mean([e['return'] for e in episodes]):>12,.2f}")
    print("-" * 62)
    rate = 100.0 * sum(all_completed) / len(all_completed) if all_completed else 0.0
    print(f"  OVERALL lap completion rate : {rate:.1f}% "
          f"({sum(all_completed)}/{len(all_completed)} episodes)")
    print(f"  OVERALL mean laps completed : {np.mean(all_laps):.3f}")
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
