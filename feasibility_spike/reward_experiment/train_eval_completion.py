"""Train PPO under one reward, then measure COMPLETION RATE and LAP TIME.

Unlike train_compare.py (which only measured speed), this evaluates with a long
step cap and records, per track, whether the policy completes a lap and how long
that lap takes.

  completion  : did cumulative laps reach 1.0 within the cap?
  lap_time_s  : sim seconds to the first completed lap (steps * timestep)
  laps        : fractional laps reached (progress / track_length)

Trains on synthetic_track_0; evaluates on synthetic_track_0/1/2 (0 = trained,
1 & 2 = unseen), so completion_rate is over 3 tracks.

    python train_eval_completion.py --reward new --steps 1000000
    python train_eval_completion.py --reward old --steps 1000000
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
TIMESTEP = 0.01
EVAL_CAP = 15000          # 150 s: enough for one ~179 m lap above ~1.2 m/s
TRAIN_TRACK = "synthetic_track_0"
EVAL_TRACKS = ["synthetic_track_0", "synthetic_track_1", "synthetic_track_2"]
SPEED_IDX = 108


def load(mod_name, filename):
    spec = importlib.util.spec_from_file_location(mod_name, HERE / filename)
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = m
    spec.loader.exec_module(m)
    return m


def make_env(WrapperCls, new_style, track):
    def _init():
        env = gym.make("f1tenth_gym:f1tenth-v0",
                       config={"num_agents": 1, "timestep": TIMESTEP, "map": track})
        env = WrapperCls(env, max_episode_steps=EVAL_CAP) if new_style else WrapperCls(env)
        return Monitor(env)
    return _init


def eval_track(model, WrapperCls, new_style, track):
    from new_reward_wrapper import TrackProgress
    venv = DummyVecEnv([make_env(WrapperCls, new_style, track)])
    core = venv.envs[0].unwrapped
    tp = TrackProgress(); tp.bind(core.track)
    obs = venv.reset()
    tp.reset(float(core.poses_x[0]), float(core.poses_y[0]))
    speeds, lap_step, outcome = [], None, "step_cap"
    for step in range(1, EVAL_CAP + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, dones, _info = venv.step(action)
        speeds.append(abs(float(obs[0][SPEED_IDX])))
        tp.update(float(core.poses_x[0]), float(core.poses_y[0]))
        if lap_step is None and tp.laps >= 1.0:
            lap_step = step
        if dones[0]:
            outcome = "crash" if bool(np.asarray(core.collisions).reshape(-1)[0]) else "reset"
            break
    venv.close()
    completed = lap_step is not None
    return dict(track=track, completed=completed,
                lap_time_s=(lap_step * TIMESTEP if completed else None),
                laps=float(tp.laps), mean_speed=float(np.mean(speeds)),
                steps=len(speeds), outcome=outcome)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", choices=["old", "new"], required=True)
    ap.add_argument("--steps", type=int, default=1_000_000)
    args = ap.parse_args()

    if args.reward == "new":
        W = load("new_reward_wrapper", "new_reward_wrapper.py"); new_style = True
    else:
        W = load("old_reward_wrapper", "old_reward_wrapper.py"); new_style = False
    WrapperCls = W.F1TenthSB3Wrapper

    print(f"=== PPO under {args.reward.upper()} reward, {args.steps:,} steps on {TRAIN_TRACK} ===")
    venv = DummyVecEnv([make_env(WrapperCls, new_style, TRAIN_TRACK)])
    model = PPO("MlpPolicy", venv, verbose=0, device="cpu", seed=0)
    model.learn(total_timesteps=args.steps)
    venv.close()

    rows = [eval_track(model, WrapperCls, new_style, t) for t in EVAL_TRACKS]
    completion_rate = float(np.mean([r["completed"] for r in rows]))
    times = [r["lap_time_s"] for r in rows if r["completed"]]

    print(f"\n{'track':>20} {'completed':>10} {'lap_time_s':>11} {'laps':>7} {'mean m/s':>9} {'outcome':>8}")
    for r in rows:
        lt = f"{r['lap_time_s']:.2f}" if r["completed"] else "-"
        print(f"{r['track']:>20} {str(r['completed']):>10} {lt:>11} "
              f"{r['laps']:>7.3f} {r['mean_speed']:>9.2f} {r['outcome']:>8}")
    print(f"\nCOMPLETION RATE ({args.reward}) : {completion_rate*100:.0f}%  ({sum(r['completed'] for r in rows)}/3 tracks)")
    if times:
        print(f"MEAN LAP TIME             : {np.mean(times):.2f} s")
    else:
        print("MEAN LAP TIME             : n/a (no lap completed)")

    out = HERE / "results" / f"completion_{args.reward}_{args.steps}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"reward": args.reward, "steps": args.steps,
                               "completion_rate": completion_rate, "tracks": rows}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
