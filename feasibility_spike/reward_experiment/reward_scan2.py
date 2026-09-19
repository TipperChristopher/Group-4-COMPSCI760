"""Precise reward retest (additive). One rollout per commanded speed, scored
under BOTH reward formulas simultaneously, so the comparison is purely about
reward definition (zero env variance between the two curves).

  OLD (the bug):   0.01 per step (1.0/s alive) + 0.1*|speed| - 0.5*|steering|
  NEW (frozen):    1.0 * metres of centreline progress - CRASH_PENALTY(40) on crash

Task: straight line from the spawn, steering = 0, commanded speed fixed,
10,000-step cap (matches the team's original 100/120/81 measurement cap).
    python reward_scan2.py
"""
import json, pathlib, types, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import gymnasium as gym
import f1tenth_gym  # noqa
import new_reward_wrapper as W
from new_reward_wrapper import F1TenthSB3Wrapper, TrackProgress
W.CRASH_PENALTY = 40.0   # frozen value used by all experiments

HERE = pathlib.Path(__file__).parent
TRACK = "synthetic_track_0"
CAP = 10000
TIMESTEP = 0.01
SPEEDS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]

def rollout(speed):
    env = gym.make("f1tenth_gym:f1tenth-v0",
                   config={"num_agents": 1, "timestep": TIMESTEP, "map": TRACK,
                           "reset_config": {"type": "cl_grid_static"}})
    env = F1TenthSB3Wrapper(env, max_episode_steps=CAP)
    core = env.unwrapped
    tp = TrackProgress(); tp.bind(core.track)
    env.reset(seed=0)
    tp.reset(float(core.poses_x[0]), float(core.poses_y[0]))
    old = new = 0.0
    steps = 0; crashed = False; speed_hist = []
    for _ in range(CAP):
        obs, r, term, trunc, info = env.step(np.array([0.0, speed / 10.0 - 1]))  # a[1]: -1->0, +1->20 m/s
        vx = abs(float(obs[108]))
        speed_hist.append(vx)
        old += (TIMESTEP + 0.1 * vx - 0.5 * abs(0.0))
        new += float(r)          # wrapper reward = progress delta - penalty (on crash step)
        tp.update(float(core.poses_x[0]), float(core.poses_y[0]))
        steps += 1
        if term or trunc:
            crashed = bool(info.get("collision"))
            break
    env.close()
    return dict(speed=speed, steps=steps, crashed=crashed,
                dist_m=round(tp.cumulative_s, 2), v_avg=round(float(np.mean(speed_hist)), 2),
                old_return=round(old, 2), new_return=round(new, 2))

rows = [rollout(s) for s in SPEEDS]
out = HERE / "results" / "reward_scan2.json"
out.write_text(json.dumps(dict(track=TRACK, cap=CAP, crash_penalty=W.CRASH_PENALTY, rows=rows), indent=2))
print(f"{'speed':>6} {'steps':>6} {'crashed':>7} {'dist_m':>7} {'v_avg':>6} {'old_return':>11} {'new_return':>11}")
for r in rows:
    print(f"{r['speed']:>6.2f} {r['steps']:>6} {str(r['crashed']):>7} {r['dist_m']:>7} {r['v_avg']:>6} "
          f"{r['old_return']:>11.2f} {r['new_return']:>11.2f}")
print(f"\nsaved {out}")
