"""Does the eval spawn seed actually propagate? (settles AC3 + the teammate discrepancy)"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import learning_curve as lc
import gymnasium as gym, f1tenth_gym  # noqa
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
F1 = lc.W.F1TenthSB3Wrapper
TRACK, RTYPE, CAP = "Silverstone", "rl_grid_static", 15000

def make():
    def _init():
        env = gym.make("f1tenth_gym:f1tenth-v0",
                       config={"num_agents":1,"timestep":0.01,"map":TRACK,"reset_config":{"type":RTYPE}})
        return F1(env, max_episode_steps=CAP)
    return _init

def offcl(core):
    cl = core.track.centerline
    px,py = float(core.poses_x[0]), float(core.poses_y[0])
    return float(np.hypot(np.array(cl.xs)-px, np.array(cl.ys)-py).min()), px, py

print("=== A. DummyVecEnv.seed(s) -> reset  (fresh env each; evaluate.py's path) ===")
for s in [0,0,1,2,3,4,3]:
    base = DummyVecEnv([make()]); core = base.envs[0].unwrapped
    base.seed(s); base.reset()
    o,px,py = offcl(core); print(f"  seed={s}  off_cl={o:.4f}  x={px:.3f} y={py:.3f}"); base.close()

print("=== B. env.reset(seed=s) DIRECTLY (ground truth: f110_env calls np.random.seed) ===")
for s in [0,0,1,2,3,4,3]:
    e = make()(); core = e.unwrapped
    e.reset(seed=s)
    o,px,py = offcl(core); print(f"  seed={s}  off_cl={o:.4f}  x={px:.3f} y={py:.3f}"); e.close()

print("=== C. DummyVecEnv+VecNormalize.seed(s) -> reset (RL eval path) ===")
for s in [0,0,3,3]:
    base = DummyVecEnv([make()]); core = base.envs[0].unwrapped
    venv = VecNormalize(base, norm_obs=True, norm_reward=False, training=False)
    venv.seed(s); venv.reset()
    o,px,py = offcl(core); print(f"  seed={s}  off_cl={o:.4f}  x={px:.3f} y={py:.3f}"); venv.close()

print("=== D. drift: ONE env, reset repeatedly WITHOUT reseeding ===")
e = make()(); core = e.unwrapped; e.reset(seed=0)
for i in range(6):
    o,px,py = offcl(core); print(f"  reset#{i} off_cl={o:.4f} x={px:.3f} y={py:.3f}"); e.reset()
e.close()
