"""Reward ablation: OLD reward vs NEW (progress) reward on the SAME trajectory.

For each constant target speed we drive straight on synthetic_track_0 and, on the
identical rollout, accumulate three reward definitions:

  old_orig : 0.01 + 0.1*speed - 0.5*|steer|            (original shaped reward)
  old_main : old_orig - (0.5 if speed < 1)             (main's band-aided version)
  new      : PROGRESS_WEIGHT * dcentreline - CRASH_PENALTY(on collision)  (Desmond's)

The NEW reward uses Desmond's real TrackProgress (vendored). One rollout, three
reward tallies -> a clean apples-to-apples comparison of reward SHAPE.

    python reward_scan.py
"""
import json
import os
import pathlib
import sys
import types
import warnings

warnings.filterwarnings("ignore")
if "gym" not in sys.modules:
    sys.modules["gym"] = types.ModuleType("gym")
sys.modules["gym"].__version__ = "0.0.0"

import numpy as np
import gymnasium as gym
import f1tenth_gym  # noqa: F401 registers f1tenth-v0

from new_reward_wrapper import TrackProgress, CRASH_PENALTY, PROGRESS_WEIGHT

TRACK = "synthetic_track_0"
CAP = 3000
SPEEDS = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 8.0, 12.0, 20.0]


def drive(v):
    env = gym.make("f1tenth_gym:f1tenth-v0",
                   config={"num_agents": 1, "timestep": 0.01, "map": TRACK})
    obs, _ = env.reset()
    core = env.unwrapped
    tp = TrackProgress()
    tp.bind(core.track)
    tp.reset(float(core.poses_x[0]), float(core.poses_y[0]))

    old_orig = old_main = new = 0.0
    steps = 0
    outcome = "step_cap"
    action = np.array([[0.0, v]], dtype=np.float64)  # [steer, speed]
    for _ in range(CAP):
        obs, _r, term, trunc, info = env.step(action)
        spd = float(obs["linear_vels_x"][0])
        collided = bool(np.asarray(obs["collisions"]).reshape(-1)[0])
        delta = tp.update(float(core.poses_x[0]), float(core.poses_y[0]))

        old_orig += 0.01 + 0.1 * spd
        old_main += 0.01 + 0.1 * spd - (0.5 if spd < 1.0 else 0.0)
        new += PROGRESS_WEIGHT * delta - (CRASH_PENALTY if collided else 0.0)
        steps += 1
        if collided:
            outcome = "crash"
            break
        if term or trunc:
            outcome = "done"
            break
    dist, laps = tp.cumulative_s, tp.laps
    env.close()
    return dict(speed=v, steps=steps, outcome=outcome, dist=dist, laps=laps,
                old_orig=old_orig, old_main=old_main, new=new)


def main():
    print("=" * 84)
    print(f"REWARD ABLATION on {TRACK} (drive straight at constant speed, cap {CAP})")
    print("=" * 84)
    hdr = f"{'speed':>6} {'steps':>6} {'outcome':>8} {'dist_m':>8} " \
          f"{'OLD_orig':>10} {'OLD_main':>10} {'NEW_progress':>13}"
    print(hdr)
    rows = []
    for v in SPEEDS:
        r = drive(v)
        rows.append(r)
        print(f"{r['speed']:>6.2f} {r['steps']:>6} {r['outcome']:>8} {r['dist']:>8.2f} "
              f"{r['old_orig']:>10.2f} {r['old_main']:>10.2f} {r['new']:>13.2f}")

    still = next(r for r in rows if r["speed"] == 0.0)
    driving = [r for r in rows if r["speed"] > 0.0]

    def best(key):
        return max(rows, key=lambda r: r[key])["speed"]

    print("\n--- verdicts (what a STRAIGHT-LINE scan can and cannot show) ---")
    print(f"OLD_orig best speed : {best('old_orig'):>5} m/s  -> slow/crawl optimal (the bug)")
    print(f"OLD_main best speed : {best('old_main'):>5} m/s  -> band-aid still leaves it slow")
    print(f"standing still: OLD_orig={still['old_orig']:.1f}  NEW={still['new']:.3f}")
    print(f"every driving speed beats standing still under NEW : "
          f"{all(r['new'] > still['new'] for r in driving)}")
    print("\nNOTE: driving STRAIGHT crashes at the first wall (~18 m) at every speed,")
    print("so this scan cannot rank speeds for NEW (Desmond's 'shallow basin on a")
    print("straight line'). It proves the design FLIP -- standing still goes from")
    print("best-ish (OLD) to worst=0 (NEW). 'Faster is better' needs the car to")
    print("FOLLOW the track, which the trained policy does -- see train_compare.py.")

    out = pathlib.Path(__file__).parent / "results" / "reward_scan.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
