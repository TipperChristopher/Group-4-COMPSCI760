"""At the hairpin stall: what does each policy's action DISTRIBUTION look like, and what does its critic think?
Raw observation captured from seed-2 @2M (stalled at ~71 m), then fed to each checkpoint with ITS OWN VecNormalize stats."""
import warnings; warnings.filterwarnings("ignore")
import sys, types, numpy as np, os, torch as th
from scipy.stats import norm
sys.modules.setdefault("gym", types.ModuleType("gym")); sys.modules["gym"].__version__ = "0.0.0"
REPO = r"E:/OneDrive - The University of Auckland/Desktop/COMPSCI 760/team_repo_heldout"; sys.path.insert(0, REPO); os.chdir(REPO)
import gymnasium as gym, f1tenth_gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_wrapper import F1TenthSB3Wrapper
def mk():
    e = gym.make("f1tenth_gym:f1tenth-v0", config={"num_agents": 1, "timestep": 0.01, "map": "synthetic_track_0", "reset_config": {"type": "cl_grid_static"}})
    return F1TenthSB3Wrapper(e, max_episode_steps=3000)
def load(tag, step):
    v = VecNormalize.load(f"models/{tag}/PPO_checkpoint_vecnormalize_{step}_steps.pkl", DummyVecEnv([mk])); v.training = False; v.norm_reward = False
    return v, PPO.load(f"models/{tag}/PPO_checkpoint_{step}_steps.zip", device="cpu")
# 1) capture raw obs along seed-2@2M's stalled trajectory
venv, model = load("PPO_1tracks_s2_G999_s2", 2000000)
venv.seed(0); obs = venv.reset(); core = venv.venv.envs[0].unwrapped
prog, caps = 0.0, {}
for t in range(1500):
    a, _ = model.predict(obs, deterministic=True)
    raw = venv.get_original_obs().copy()
    for mark in (50, 60, 66, 70):
        if mark not in caps and prog >= mark: caps[mark] = (raw, float(core.sim.agents[0].state[3]))
    obs, r, d, _ = venv.step(a); prog += float(r[0])
    if t == 1400: caps["stalled"] = (venv.get_original_obs().copy(), float(core.sim.agents[0].state[3]))
    if d[0]: break
venv.close()
print(f"seed2@2M trajectory: progress {prog:.1f} m after {t+1} steps; captured states at {list(caps)}")
# 2) evaluate every checkpoint on those exact raw states
print("\nspeed action a1 -> commanded speed = (clip(a1,-1,1)+1)*10 m/s.  P(move) = P(sampled a1 > -0.9, i.e. >=1 m/s) under the policy's own Gaussian")
print(f"{'model':>16} {'state':>8} {'v_now':>6} | {'mean a1':>7} {'std a1':>6} {'P(move)':>8} | {'mean steer':>10} | {'V(s)':>7}")
for lab, tag, step in [("seed2 @1.6M", "PPO_1tracks_s2_G999_s2", 1600000), ("seed2 @2.0M", "PPO_1tracks_s2_G999_s2", 2000000), ("seed0 @2.0M", "PPO_1tracks_s0_G999_gamma", 2000000)]:
    v, m = load(tag, step)
    for key, (raw, vnow) in caps.items():
        o = th.as_tensor(v.normalize_obs(raw), dtype=th.float32)
        with th.no_grad():
            dist = m.policy.get_distribution(o).distribution; val = float(m.policy.predict_values(o)[0])
        mu, sd = dist.mean[0].numpy(), dist.stddev[0].numpy()
        pmove = 1 - norm.cdf((-0.9 - mu[1]) / sd[1])
        print(f"{lab:>16} {str(key):>8} {vnow:6.2f} | {mu[1]:7.2f} {sd[1]:6.3f} {pmove:8.3f} | {mu[0]:10.2f} | {val:7.1f}")
    v.close()
