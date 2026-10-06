import warnings; warnings.filterwarnings("ignore")
import sys, types, numpy as np, glob, os
sys.modules.setdefault("gym", types.ModuleType("gym")); sys.modules["gym"].__version__ = "0.0.0"
sys.path.insert(0, ".")
import gymnasium as gym, f1tenth_gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_wrapper import F1TenthSB3Wrapper

def rollout(model_dir, track="synthetic_track_0", seed=0):
    def _init():
        e = gym.make("f1tenth_gym:f1tenth-v0",
                     config={"num_agents": 1, "timestep": 0.01, "map": track,
                             "reset_config": {"type": "cl_grid_static"}})
        return F1TenthSB3Wrapper(e, max_episode_steps=3000)
    venv = DummyVecEnv([_init])
    venv = VecNormalize.load(f"{model_dir}/vecnormalize.pkl", venv)
    venv.training = False; venv.norm_reward = False
    model = PPO.load(f"{model_dir}/final_model.zip", device="cpu")
    obs = venv.reset()
    core = venv.venv.envs[0].unwrapped          # the raw f1tenth env
    speeds, steers = [], []
    for t in range(3000):
        a, _ = model.predict(obs, deterministic=True)
        # raw body speed, straight from the simulator (obs is normalised)
        speeds.append(abs(float(np.asarray(core.linear_vels_x).reshape(-1)[0])))
        steers.append(abs(float(a[0][0])))
        obs, r, done, info = venv.step(a)
        if done[0]:
            prog = float(info[0].get("progress_m", float("nan")))
            break
    else:
        prog = float(info[0].get("progress_m", float("nan")))
    venv.close()
    return np.array(speeds), np.array(steers), prog

runs = []
for tag in ["V0_baseline", "V4_nsteps", "G999_gamma", "P40_penalty40", "R2_repeat10_v6"]:
    d = f"models/PPO_1tracks_s0_{tag}"
    if not os.path.exists(f"{d}/final_model.zip"):
        runs.append((tag, None)); continue
    runs.append((tag, d))

print("Speed trajectory over the final 2 seconds before the crash")
print(f"{'model':>16} {'steps':>6} {'dist m':>7} {'| t-2s':>7} {'t-1.5s':>7} {'t-1s':>7} {'t-0.5s':>7} {'final':>6}   verdict")
for tag, d in runs:
    if d is None: print(f"{tag:>16}  not finished yet"); continue
    sp, st, prog = rollout(d)
    def at(sec):
        i = len(sp) - int(sec*100)
        return sp[max(0, i)]
    verdict = "BRAKING" if at(0.1) < at(2.0) - 3 else ("coasting" if at(0.1) < at(2.0) + 1 else "STILL ACCELERATING")
    print(f"{tag:>16} {len(sp):>6} {prog:>7.1f} {at(2.0):>7.2f} {at(1.5):>7.2f} {at(1.0):>7.2f} {at(0.5):>7.2f} {sp[-1]:>6.2f}   {verdict}")
