"""Zero-shot RL eval on real circuits: BEFORE (rl_grid_static / raceline) vs
AFTER (cl_grid_static / centreline) the spawn fix. ADDITIVE — imports the team's
sb3_wrapper for obs + VecNormalize compatibility; does not modify team code.

    python evaluate_centreline.py --model <run_dir> --algo PPO --tracks Spielberg Silverstone --seeds 0 1 2 3 4
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
ROOT = HERE.parent.parent  # team_repo

def load_local(name, rel):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / rel))
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m

wrap = load_local("sb3_wrapper", "sb3_wrapper.py")
F1 = wrap.F1TenthSB3Wrapper
load_vecnormalize = wrap.load_vecnormalize
from new_reward_wrapper import TrackProgress
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

def make_env(track, rtype):
    def _init():
        env = gym.make("f1tenth_gym:f1tenth-v0",
                       config={"num_agents":1,"timestep":TIMESTEP,"map":track,"reset_config":{"type":rtype}})
        return CollisionRecorder(F1(env))
    return _init

def run(model, track, rtype, seed, stats_path):
    base = DummyVecEnv([make_env(track, rtype)])
    core = base.envs[0].unwrapped
    venv = load_vecnormalize(base, stats_path) if stats_path else base
    cl = core.track.centerline
    tp = TrackProgress(); tp.bind(core.track)
    try: venv.seed(seed)
    except Exception: pass
    obs = venv.reset()
    px, py = float(core.poses_x[0]), float(core.poses_y[0])
    off_cl = float(np.hypot(np.array(cl.xs)-px, np.array(cl.ys)-py).min())
    tp.reset(px, py)
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
    laps = float(tp.laps); venv.close()
    return dict(seed=seed, reset=rtype, off_centreline=round(off_cl,3), laps=round(laps,3),
                crashed=crashed, outcome=outcome, steps=steps, ret=round(ret,2),
                lap_time_s=(round(steps*TIMESTEP/laps,2) if laps>=1.0 else None))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--algo", default="PPO", choices=["PPO","SAC"])
    ap.add_argument("--tracks", nargs="+", default=["Spielberg","Silverstone"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4])
    args = ap.parse_args()
    run_dir = pathlib.Path(args.model)
    if not run_dir.is_absolute(): run_dir = (HERE / args.model).resolve()
    model_file = run_dir / "final_model.zip"
    stats = run_dir / "vecnormalize.pkl"
    stats_path = str(stats) if stats.exists() else None
    print(f"model={model_file}\nvecnormalize={'yes' if stats_path else 'NO'}\n")
    model = ALGOS[args.algo].load(str(model_file))
    results = {}
    hdr = f"{'track':>12} {'reset':>16} {'seed':>4} {'off_cl':>7} {'laps':>7} {'outcome':>9} {'return':>9} {'steps':>7}"
    print(hdr); print("-"*len(hdr))
    for track in args.tracks:
        results[track] = []
        for rtype in ["rl_grid_static","cl_grid_static"]:
            for s in args.seeds:
                r = run(model, track, rtype, s, stats_path)
                results[track].append(r)
                print(f"{track:>12} {rtype:>16} {s:>4} {r['off_centreline']:>7.3f} {r['laps']:>7.3f} "
                      f"{r['outcome']:>9} {r['ret']:>9.2f} {r['steps']:>7}", flush=True)
    print("\n=== before/after summary (mean laps over seeds) ===")
    for track in args.tracks:
        for rtype in ["rl_grid_static","cl_grid_static"]:
            rows = [r for r in results[track] if r["reset"]==rtype]
            laps = [r["laps"] for r in rows]
            nc = sum(1 for r in rows if r["outcome"]=="crash")
            tag = "BEFORE(raceline)" if rtype=="rl_grid_static" else "AFTER(centreline)"
            print(f"  {track:>12} {tag:18} laps mean={np.mean(laps):.3f} std={np.std(laps):.3f} "
                  f"min={min(laps):.3f} max={max(laps):.3f} crash={nc}/{len(rows)}")
    out = HERE / "results" / f"rl_before_after_{args.algo}.json"
    out.write_text(json.dumps(dict(model=str(run_dir), algo=args.algo, results=results), indent=2))
    print(f"\nsaved {out}")

if __name__ == "__main__":
    main()
