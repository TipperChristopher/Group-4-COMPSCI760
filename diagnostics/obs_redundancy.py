"""How much NEW information does one 100 Hz step actually carry?

The user's hypothesis: at 100 Hz, consecutive observations are nearly identical,
so a fixed step budget buys far less information than it appears to. Measure it
directly: drive the canonical track at a few constant speeds and record how much
the 108-beam scan and the ego-state block change between consecutive steps.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, types, numpy as np
sys.modules.setdefault("gym", types.ModuleType("gym")); sys.modules["gym"].__version__ = "0.0.0"
import gymnasium as gym, f1tenth_gym
from f1tenth_gym.envs.track import Track

TRACK = "synthetic_track_0"
L = float(np.asarray(Track.from_track_name(TRACK).centerline.spline.s)[-1])
print(f"track {TRACK}: {L:.1f} m\n")

env = gym.make("f1tenth_gym:f1tenth-v0",
               config={"num_agents": 1, "timestep": 0.01, "map": TRACK,
                       "reset_config": {"type": "cl_grid_static"}})
core = env.unwrapped
print(f"{'speed':>6} {'m per step':>11} {'|dscan| per step':>17} {'scan sd':>9} {'ratio':>7}   interpretation")
for speed in [2.0, 5.0, 10.0, 20.0]:
    obs, _ = env.reset(seed=0)
    scans, steers = [], []
    prev = np.asarray(obs["scans"][0])
    for _ in range(400):
        act = np.array([[0.0, speed]])
        obs, *_ = env.step(act)
        cur = np.asarray(obs["scans"][0])
        scans.append(np.abs(cur - prev).mean())
        prev = cur
    d = np.array(scans[100:])                    # skip the start transient
    sd = np.asarray(obs["scans"][0]).std()
    print(f"{speed:>6.1f} {speed*0.01:>11.3f} {d.mean():>17.4f} {sd:>9.3f} {d.mean()/sd:>7.4f}"
          f"   {'scan barely changes' if d.mean()/sd<0.02 else 'real change'}")
env.close()
