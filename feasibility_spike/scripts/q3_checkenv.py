"""Q3 evidence: does SB3 accept the (wrapped) env? Print spaces + run check_env."""
import numpy as np
import torch
from stable_baselines3.common.env_checker import check_env
from sb3_wrapper import F1TenthSB3

print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())

env = F1TenthSB3(map_name="Spielberg")
print("wrapped observation_space:", env.observation_space)
print("wrapped action_space:", env.action_space)

obs, info = env.reset(seed=0)
print("reset obs shape:", obs.shape, obs.dtype)
o2, r, term, trunc, info = env.step(env.action_space.sample())
print("step -> obs shape:", o2.shape, "reward:", r, "term:", term, "trunc:", trunc)

print("running stable_baselines3 check_env ...")
check_env(env, warn=True, skip_render_check=True)
print("CHECK_ENV PASSED")
