"""Q1 evidence: clean env.reset() + env.step() round trip, headless."""
import time, sys
import numpy as np
import gymnasium as gym
import f1tenth_gym  # registers f1tenth-v0

print("gymnasium:", gym.__version__)
print("numpy:", np.__version__)

t0 = time.time()
env = gym.make(
    "f1tenth_gym:f1tenth-v0",
    config={
        "map": "Spielberg",           # downloaded from api.f1tenth.org on first use
        "num_agents": 1,
        "observation_config": {"type": "kinematic_state"},
    },
    render_mode=None,                 # headless
)
print("make() ok in %.2fs" % (time.time() - t0))
print("observation_space:", env.observation_space)
print("action_space:", env.action_space)

obs, info = env.reset()
print("reset() ok. obs type:", type(obs))
if isinstance(obs, dict):
    for k, v in obs.items():
        print("  obs[%r] keys/shape:" % k, list(v.keys()) if isinstance(v, dict) else np.asarray(v).shape)

# one deterministic step (zero action = [steer, speed])
action = np.zeros((1, 2), dtype=np.float32)
step_ret = env.step(action)
print("step() returned a %d-tuple (5 => gymnasium)" % len(step_ret))
obs2, reward, terminated, truncated, info2 = step_ret
print("reward:", reward, "terminated:", terminated, "truncated:", truncated)
env.close()
print("ROUND TRIP OK")
