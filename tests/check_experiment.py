"""STAGE 2: is the experiment valid? Runs against the real simulator.

    python tests/check_experiment.py                 # diversity 5 and 20
    python tests/check_experiment.py --diversity 20  # one level only

Covers the failure modes that would silently void the grid: episodes that never
reset, tracks that never rotate, and non-reproducible pools. Takes a few
minutes. Exits non-zero if any check fails.
"""

import argparse
import os
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import train  # applies the gym shim and loads the project modules
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

failures = []


def check(ok, msg):
    print(("  PASS  " if ok else "  FAIL  ") + msg)
    if not ok:
        failures.append(msg)


class EpisodeLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rows = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.rows.append((
                    info.get("track_name"), int(ep["l"]), float(ep["r"]),
                    float(info.get("laps", 0.0)),
                    "crash" if info.get("collision")
                    else ("timeout" if info.get("TimeLimit.truncated") else "term"),
                ))
        return True


def rotation_check(diversity, steps, cap):
    pool = [f"synthetic_track_{i}" for i in range(diversity)]
    print()
    print("=" * 78)
    print(f"2.1  RESET AND TRACK ROTATION  (--diversity {diversity})")
    print("=" * 78)
    fn = train.make_env(pool, seed=0, track_seed=0, max_episode_steps=cap)
    venv = train.wrap_vecnormalize(DummyVecEnv([fn]), training=True)
    model = PPO("MlpPolicy", venv, seed=0, n_steps=512, verbose=0)
    cb = EpisodeLogger()
    model.learn(total_timesteps=steps, callback=cb)
    venv.close()

    tracks = [r[0] for r in cb.rows]
    counts = Counter(tracks)
    print(f"  episodes completed      : {len(cb.rows)}")
    print(f"  distinct tracks visited : {len(counts)} of {diversity}")
    if counts:
        lo, hi = min(counts.values()), max(counts.values())
        print(f"  per-track episode counts: min {lo}, max {hi}")
        print("  first 12 episodes       : "
              + " ".join(t.replace("synthetic_track_", "#") for t in tracks[:12]))
    check(len(cb.rows) > 0, f"d={diversity}: reset() fires (episodes end at all)")
    check(len(counts) > 1 or diversity == 1,
          f"d={diversity}: the track actually changes between episodes")
    if len(cb.rows) >= diversity:
        check(len(counts) == diversity,
              f"d={diversity}: every track in the pool appears")
        check(max(counts.values()) - min(counts.values()) <= 2,
              f"d={diversity}: pool is dealt evenly (stratified, not random)")
    else:
        print(f"  NOTE: only {len(cb.rows)} episodes for a {diversity}-track pool; "
              "raise --steps to check even dealing")
    ends = {r[4] for r in cb.rows}
    print(f"  episode end reasons     : {sorted(ends)}")
    return cb.rows


def seed_check():
    print()
    print("=" * 78)
    print("2.2  TRACK SEED REPRODUCIBILITY")
    print("=" * 78)
    Sampler = train._track_pool_module.StratifiedTrackSampler
    pool = [f"synthetic_track_{i}" for i in range(20)]

    a = [Sampler(pool, seed=1234).next() for _ in range(1)]
    run1 = Sampler(pool, seed=1234)
    run2 = Sampler(pool, seed=1234)
    diff = Sampler(pool, seed=4321)
    s1 = [run1.next() for _ in range(60)]
    s2 = [run2.next() for _ in range(60)]
    s3 = [diff.next() for _ in range(60)]
    print(f"  first 8 at seed 1234 : "
          + " ".join(t.replace("synthetic_track_", "#") for t in s1[:8]))
    print(f"  first 8 at seed 4321 : "
          + " ".join(t.replace("synthetic_track_", "#") for t in s3[:8]))
    check(s1 == s2, "same --track-seed reproduces the same order (PPO == SAC)")
    check(s1 != s3, "a different --track-seed gives a different order")
    check(all(sorted(s1[i:i + 20]) == sorted(pool) for i in range(0, 60, 20)),
          "each pass through the pool covers every track exactly once")


def determinism_check(cap=400):
    print()
    print("=" * 78)
    print("2.3  EVALUATION DETERMINISM")
    print("=" * 78)
    print("  A greedy policy from a fixed spawn must produce an identical rollout.")

    def once():
        fn = train.make_env(["synthetic_track_0"], seed=0, track_seed=0,
                            max_episode_steps=cap)
        venv = train.wrap_vecnormalize(DummyVecEnv([fn]), training=False)
        model = PPO("MlpPolicy", venv, seed=7, verbose=0)
        obs = venv.reset()
        rew, steps, laps = 0.0, 0, 0.0
        for _ in range(cap):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, dones, infos = venv.step(action)
            rew += float(r[0])
            steps += 1
            laps = float(infos[0].get("laps", laps))
            if dones[0]:
                break
        venv.close()
        return steps, round(rew, 6), round(laps, 6)

    a, b = once(), once()
    print(f"  run 1 : steps={a[0]:>5}  return={a[1]:>9.4f}  laps={a[2]:.4f}")
    print(f"  run 2 : steps={b[0]:>5}  return={b[1]:>9.4f}  laps={b[2]:.4f}")
    check(a == b, "two identical greedy rollouts give bit-identical results")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--diversity", type=int, nargs="+", default=[5, 20])
    p.add_argument("--steps", type=int, default=12000,
                   help="PPO steps per diversity level.")
    p.add_argument("--cap", type=int, default=300,
                   help="Episode step cap, kept short so many episodes fit.")
    args = p.parse_args()

    for d in args.diversity:
        rotation_check(d, args.steps, args.cap)
    seed_check()
    determinism_check()

    print()
    print("=" * 78)
    print("ALL STAGE 2 CHECKS PASSED" if not failures
          else f"{len(failures)} STAGE 2 CHECK(S) FAILED")
    for f in failures:
        print("  - " + f)
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
