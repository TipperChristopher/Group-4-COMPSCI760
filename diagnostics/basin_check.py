"""What did each policy ACTUALLY achieve? Roll them out and compute the return.

The question: at gamma=0.99, 'drive fast and crash' scores ~13.9 and 'drive
moderate and survive' scores ~7.9 on paper. So which did each algorithm find,
and does the surviving policy actually score better under the objective it was
trained on? Measure it rather than argue.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, types, numpy as np, os
sys.modules.setdefault("gym", types.ModuleType("gym")); sys.modules["gym"].__version__ = "0.0.0"
REPO = r"E:/OneDrive - The University of Auckland/Desktop/COMPSCI 760/team_repo_heldout"
sys.path.insert(0, REPO)
import gymnasium as gym, f1tenth_gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_wrapper import F1TenthSB3Wrapper

def rollout(algo, model_dir, track="synthetic_track_0"):
    def _init():
        e = gym.make("f1tenth_gym:f1tenth-v0",
                     config={"num_agents": 1, "timestep": 0.01, "map": track,
                             "reset_config": {"type": "cl_grid_static"}})
        return F1TenthSB3Wrapper(e, max_episode_steps=3000)
    venv = DummyVecEnv([_init])
    venv = VecNormalize.load(f"{model_dir}/vecnormalize.pkl", venv)
    venv.training = False; venv.norm_reward = False
    A = PPO if algo == "PPO" else SAC
    model = A.load(f"{model_dir}/final_model.zip", device="cpu")
    obs = venv.reset()
    rewards, prog = [], float("nan")
    for t in range(3000):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, done, info = venv.step(a)
        rewards.append(float(r[0]))
        if done[0]:
            prog = float(info[0].get("progress_m", float("nan"))); break
    venv.close()
    return np.array(rewards), prog

runs = [("PPO", "G999_gamma", f"{REPO}/models/PPO_1tracks_s0_G999_gamma"),
        ("PPO", "V4_nsteps",  f"{REPO}/models/PPO_1tracks_s0_V4_nsteps"),
        ("PPO", "V0_baseline",f"{REPO}/models/PPO_1tracks_s0_V0_baseline"),
        ("SAC", "oursac_overfit", r"E:/OneDrive - The University of Auckland/Desktop/COMPSCI 760/team_repo/feasibility_spike/reward_experiment/saved_models/sac_overfit")]
print(f"{'algo':>4} {'policy':>16} {'steps':>6} {'dist m':>7} {'undisc. return':>14} {'discounted @0.99':>17} {'discounted @0.999':>18}")
for algo, name, d in runs:
    n = "sac_overfit/final_model.zip"
    if not os.path.exists(f"{d}/(x)"):
        pass
    mfile = f"{d}/final_model.zip"
    if not os.path.exists(mfile):
        print(f"{algo:>4} {name:>16}   missing ({d})"); continue
    try:
        r, prog = rollout(algo, d)
    except Exception as ex:
        print(f"{algo:>4} {name:>16}   error {type(ex).__name__}: {ex}"); continue
    k = np.arange(len(r))
    print(f"{algo:>4} {name:>16} {len(r):>6} {prog:>7.1f} {r.sum():>14.1f} "
          f"{(r*(0.99**k)).sum():>17.2f} {(r*(0.999**k)).sum():>18.2f}")
