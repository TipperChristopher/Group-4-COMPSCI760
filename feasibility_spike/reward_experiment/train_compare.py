"""Train short PPO under the OLD reward and the NEW (progress) reward, then measure
what the learned policy actually does: mean speed, track progress, laps.

This is the 'faster is better' half of the proof that the straight-line scan
cannot show: a trained policy FOLLOWS the track, so under the progress reward it
learns to go fast, while under the old reward it learns to crawl.

    python train_compare.py --steps 100000
"""
import argparse
import importlib.util
import json
import pathlib
import sys
import types
import warnings

warnings.filterwarnings("ignore")
if "gym" not in sys.modules:
    sys.modules["gym"] = types.ModuleType("gym")
sys.modules["gym"].__version__ = "0.0.0"

import numpy as np
import gymnasium as gym
import f1tenth_gym  # noqa: F401
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

HERE = pathlib.Path(__file__).parent
TRACK = "synthetic_track_0"
EVAL_CAP = 3000
SPEED_IDX = 108  # processed obs = [scan(108), lin_x, lin_y, ang_z, theta, 0]; index 108 = lin_x


def load(mod_name, filename):
    spec = importlib.util.spec_from_file_location(mod_name, HERE / filename)
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = m
    spec.loader.exec_module(m)
    return m


def make_env(WrapperCls, new_style):
    def _init():
        env = gym.make("f1tenth_gym:f1tenth-v0",
                       config={"num_agents": 1, "timestep": 0.01, "map": TRACK})
        env = WrapperCls(env, max_episode_steps=EVAL_CAP) if new_style else WrapperCls(env)
        return Monitor(env)
    return _init


def evaluate(model, WrapperCls, new_style):
    """Deterministic rollout; measure mean speed, progress (m), laps, outcome."""
    from new_reward_wrapper import TrackProgress
    venv = DummyVecEnv([make_env(WrapperCls, new_style)])
    core = venv.envs[0].unwrapped
    tp = TrackProgress(); tp.bind(core.track)
    obs = venv.reset()
    tp.reset(float(core.poses_x[0]), float(core.poses_y[0]))
    speeds, outcome = [], "step_cap"
    for _ in range(EVAL_CAP):
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, dones, infos = venv.step(action)
        speeds.append(abs(float(obs[0][SPEED_IDX])))
        tp.update(float(core.poses_x[0]), float(core.poses_y[0]))
        if dones[0]:
            outcome = "crash" if bool(np.asarray(core.collisions).reshape(-1)[0]) else "reset"
            break
    venv.close()
    return dict(mean_speed=float(np.mean(speeds)), max_speed=float(np.max(speeds)),
                progress_m=float(tp.cumulative_s), laps=float(tp.laps), outcome=outcome)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100_000)
    args = ap.parse_args()

    old = load("old_reward_wrapper", "old_reward_wrapper.py")
    new = load("new_reward_wrapper", "new_reward_wrapper.py")

    cases = [("OLD reward", old.F1TenthSB3Wrapper, False),
             ("NEW reward", new.F1TenthSB3Wrapper, True)]
    results = {}
    for label, WrapperCls, new_style in cases:
        print(f"\n=== training PPO under {label} for {args.steps:,} steps ===")
        venv = DummyVecEnv([make_env(WrapperCls, new_style)])
        model = PPO("MlpPolicy", venv, verbose=0, device="cpu", seed=0)
        model.learn(total_timesteps=args.steps)
        venv.close()
        r = evaluate(model, WrapperCls, new_style)
        results[label] = r
        print(f"  -> mean_speed={r['mean_speed']:.2f} m/s  max={r['max_speed']:.2f}  "
              f"progress={r['progress_m']:.1f} m  laps={r['laps']:.3f}  outcome={r['outcome']}")

    print("\n" + "=" * 66)
    print(f"{'reward':>12} {'mean m/s':>9} {'max m/s':>8} {'progress m':>11} {'laps':>7}")
    for label, r in results.items():
        print(f"{label:>12} {r['mean_speed']:>9.2f} {r['max_speed']:>8.2f} "
              f"{r['progress_m']:>11.1f} {r['laps']:>7.3f}")
    print("=" * 66)
    o, n = results["OLD reward"], results["NEW reward"]
    print(f"NEW mean speed is {n['mean_speed'] / max(o['mean_speed'],1e-6):.1f}x the OLD policy's; "
          f"NEW progress is {n['progress_m'] / max(o['progress_m'],1e-6):.1f}x.")

    out = HERE / "results" / "train_compare.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"steps": args.steps, "results": results}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
