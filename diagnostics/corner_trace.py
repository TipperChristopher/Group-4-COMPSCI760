"""What happens at the ~72 m corner? Deterministic rollouts on synthetic_track_0,
per-step speed / steering vs progress, for each seed at 1.0M and 2.0M."""
import warnings; warnings.filterwarnings("ignore")
import sys, types, numpy as np, pandas as pd, os
sys.modules.setdefault("gym", types.ModuleType("gym")); sys.modules["gym"].__version__ = "0.0.0"
REPO = r"E:/OneDrive - The University of Auckland/Desktop/COMPSCI 760/team_repo_heldout"
MAPS = r"E:/OneDrive - The University of Auckland/Desktop/COMPSCI 760/spike/f1tenth_gym_v1/maps"
sys.path.insert(0, REPO); os.chdir(REPO)
import gymnasium as gym, f1tenth_gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_wrapper import F1TenthSB3Wrapper

# centreline curvature vs arc length
c = pd.read_csv(f"{MAPS}/synthetic_track_0/synthetic_track_0_centerline.csv", comment="#", header=None).values[:, :2]
d = np.diff(np.vstack([c, c[:1]]), axis=0); seg = np.hypot(d[:, 0], d[:, 1]); s = np.concatenate([[0], np.cumsum(seg)[:-1]])
th = np.unwrap(np.arctan2(d[:, 1], d[:, 0])); k = np.gradient(th) / np.maximum(seg, 1e-6)
k = np.convolve(k, np.ones(9) / 9, mode="same")
print(f"track length {seg.sum():.1f} m.  Curvature |k| (1/m) by 5 m bin, 40-100 m:")
print("  " + "  ".join(f"{a}-{a+5}:{np.abs(k[(s>=a)&(s<a+5)]).max():.2f}" for a in range(40, 100, 5)))
top = np.argsort(-np.abs(k))
print("  sharpest points (s, |k|, radius):", [(round(s[i],1), round(abs(k[i]),2), round(1/abs(k[i]),1)) for i in top[:200:40]])

def rollout(seedtag, step, spawn):
    run = f"models/{seedtag}"
    def _init():
        e = gym.make("f1tenth_gym:f1tenth-v0", config={"num_agents": 1, "timestep": 0.01, "map": "synthetic_track_0",
                     "reset_config": {"type": "cl_grid_static"}})
        return F1TenthSB3Wrapper(e, max_episode_steps=3000)
    venv = DummyVecEnv([_init])
    venv = VecNormalize.load(f"{run}/PPO_checkpoint_vecnormalize_{step}_steps.pkl", venv)
    venv.training = False; venv.norm_reward = False
    model = PPO.load(f"{run}/PPO_checkpoint_{step}_steps.zip", device="cpu")
    venv.seed(spawn); obs = venv.reset(); core = venv.venv.envs[0].unwrapped
    rows, prog = [], 0.0
    for t in range(3000):
        a, _ = model.predict(obs, deterministic=True)
        v = abs(float(core.sim.agents[0].state[3]))
        rows.append((prog, v, float(a[0][0]), (float(a[0][1]) + 1) * 10))
        obs, r, done, info = venv.step(a)
        crashed = bool(done[0]) and r[0] < -1
        prog += float(r[0]) + (5.0 if crashed else 0.0)
        if done[0]: break
    venv.close()
    return pd.DataFrame(rows, columns=["s", "speed", "steer", "cmd_speed"]), prog, crashed

RUNS = {0: "PPO_1tracks_s0_G999_gamma", 1: "PPO_1tracks_s1_G999_s1", 2: "PPO_1tracks_s2_G999_s2"}
print("\nper spawn: outcome (crash @ m / LAP)   + speed (m/s) and |steer| entering the corner zone")
for seed, tag in RUNS.items():
    for step in [1000000, 2000000]:
        outs, zone = [], []
        for spawn in [0, 2, 4, 6, 8]:
            df, prog, crashed = rollout(tag, step, spawn)
            outs.append(f"crash@{prog:.0f}" if crashed else ("LAP" if prog > 160 else f"{prog:.0f}m"))
            z = df[(df.s >= 55) & (df.s < 75)]
            if len(z): zone.append((z.speed.mean(), z.speed.max(), z.steer.abs().mean(), df[(df.s>=40)&(df.s<55)].speed.mean()))
        zz = np.array(zone).mean(axis=0) if zone else [np.nan]*4
        print(f"  seed {seed} @ {step/1e6:.1f}M: {' '.join(f'{o:>9}' for o in outs)} | v(40-55m) {zz[3]:5.1f}  v(55-75m) mean {zz[0]:5.1f} max {zz[1]:5.1f}  |steer| {zz[2]:.2f}")
