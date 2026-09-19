"""Evaluate a SAVED SB3 policy (final_model.zip + vecnormalize.pkl) on unseen
tracks and compare against the baselines. ADDITIVE — no team code touched.

Fixes the eval episode cap: 15000 steps (150 s), enough for 2 laps on real
circuits (the wrapper default of 3000 would falsely truncate at 30 s).

    python eval_vs_baseline.py --model saved_models/ppo_vn_cp40 --algo PPO \
        --tracks synthetic_track_1 synthetic_track_2 Spielberg Silverstone
"""
import argparse, json, os, sys, types, importlib.util, pathlib, warnings
warnings.filterwarnings("ignore")
if "gym" not in sys.modules:
    sys.modules["gym"] = types.ModuleType("gym"); sys.modules["gym"].__version__ = "0.0.0"
import numpy as np
import gymnasium as gym
import f1tenth_gym  # noqa: F401
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
from new_reward_wrapper import F1TenthSB3Wrapper as F1, load_vecnormalize, TrackProgress
ALGOS = {"PPO": PPO, "SAC": SAC}
TIMESTEP, CAP = 0.01, 15000

class CollisionRecorder(gym.Wrapper):
    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        if term or trunc:
            info = dict(info) if info else {}
            try: info["collision"] = bool(self.env.unwrapped.collisions[0])
            except Exception: info["collision"] = None
        return obs, r, term, trunc, info

def make_env(track):
    def _init():
        env = gym.make("f1tenth_gym:f1tenth-v0",
                       config={"num_agents": 1, "timestep": TIMESTEP, "map": track,
                               "reset_config": {"type": "cl_grid_static"}})
        return CollisionRecorder(F1(env, max_episode_steps=CAP))
    return _init

def run_ep(model, track, seed, stats_path):
    base = DummyVecEnv([make_env(track)])
    core = base.envs[0].unwrapped
    venv = load_vecnormalize(base, stats_path) if stats_path else base
    tp = TrackProgress(); tp.bind(core.track)
    try: venv.seed(seed)
    except Exception: pass
    obs = venv.reset()
    px, py = float(core.poses_x[0]), float(core.poses_y[0]); tp.reset(px, py)
    crashed, steps, outcome, ret = None, 0, "step_cap", 0.0
    for _ in range(CAP):
        action, _ = model.predict(obs, deterministic=True)
        obs, rews, dones, infos = venv.step(action); ret += float(rews[0]); steps += 1
        if dones[0]:
            info = infos[0] if isinstance(infos[0], dict) else {}
            crashed = info.get("collision")
            outcome = "crash" if crashed else ("truncated" if info.get("TimeLimit.truncated") else "finished")
            break
        tp.update(float(core.poses_x[0]), float(core.poses_y[0]))
        if tp.laps >= 1.0:          # team protocol: evaluation terminates at ONE lap
            outcome = "finished"
            break
    laps = float(tp.laps); venv.close()
    return dict(seed=seed, laps=round(laps, 3), crashed=crashed, outcome=outcome,
                steps=steps, ret=round(ret, 2),
                lap_time_s=(round(steps * TIMESTEP / laps, 2) if laps >= 1.0 else None))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--algo", default="PPO", choices=["PPO", "SAC", "gap", "random"])
    ap.add_argument("--tracks", nargs="+", default=["synthetic_track_1", "synthetic_track_2", "Spielberg", "Silverstone"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.model) if args.model else None
    stats_path = None
    if args.algo in ("gap", "random"):
        from baselines import GapFollower, RandomPolicy
        model = GapFollower() if args.algo == "gap" else RandomPolicy(seed=0)
        print(f"baseline policy: {args.algo} (no model, no VecNormalize \u2014 reads raw LiDAR)\n")
    else:
        if run_dir is None:
            raise SystemExit("--model is required for PPO/SAC")
        if not run_dir.is_absolute(): run_dir = (HERE / args.model).resolve()
        model_file = run_dir / "final_model.zip"; stats = run_dir / "vecnormalize.pkl"
        stats_path = str(stats) if stats.exists() else None
        print(f"model={model_file}\nvecnormalize={'yes' if stats_path else 'NO'}\n")
        model = ALGOS[args.algo].load(str(model_file))

    results = {}
    hdr = f"{'track':>18} {'laps_mean':>9} {'laps_std':>8} {'min':>6} {'max':>6} {'crash':>7} {'finish':>7} {'laptime':>8}"
    print(hdr); print("-" * len(hdr))
    for track in args.tracks:
        eps = [run_ep(model, track, s, stats_path) for s in args.seeds]
        results[track] = eps
        laps = [e["laps"] for e in eps]
        nc = sum(1 for e in eps if e["outcome"] == "crash")
        nf = sum(1 for e in eps if e["outcome"] == "finished")
        lts = [e["lap_time_s"] for e in eps if e.get("lap_time_s")]
        lt = f"{np.mean(lts):.1f}s" if lts else "-"
        print(f"{track:>18} {np.mean(laps):>9.3f} {np.std(laps):>8.3f} {min(laps):>6.3f} {max(laps):>6.3f} "
              f"{nc:>4}/{len(eps)} {nf:>5}/{len(eps)} {lt:>8}", flush=True)

    tag = args.tag or f"{args.algo.lower()}_unseen"
    out = HERE / "results" / f"eval_{tag}.json"
    out.write_text(json.dumps(dict(model=str(run_dir), algo=args.algo, tracks=args.tracks,
                                   seeds=args.seeds, results=results), indent=2))
    print(f"\nsaved {out}")

if __name__ == "__main__":
    main()
