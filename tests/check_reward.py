"""STAGE 1: does the reward do what we think? Runs against the real simulator.

    python tests/check_reward.py

Takes about a minute. Exits non-zero if any check fails.
"""

import os
import sys
import types

if "gym" not in sys.modules:
    sys.modules["gym"] = types.ModuleType("gym")
sys.modules["gym"].__version__ = "0.0.0"

import importlib.util

import gymnasium as gym
import numpy as np

import f1tenth_gym  # noqa: F401  registers f1tenth-v0

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "sb3_wrapper", os.path.join(REPO, "sb3_wrapper.py")
)
W = importlib.util.module_from_spec(_spec)
sys.modules["sb3_wrapper"] = W
_spec.loader.exec_module(W)

TRACK = "synthetic_track_0"
failures = []


def check(ok, msg):
    print(("  PASS  " if ok else "  FAIL  ") + msg)
    if not ok:
        failures.append(msg)


def to_norm(speed, steer=0.0):
    """Physical units -> the normalised action the policy emits."""
    a1 = 2.0 * (speed - W.SPEED_MIN) / (W.SPEED_MAX - W.SPEED_MIN) - 1.0
    return np.array([steer / W.STEERING_LIMIT, a1], dtype=np.float32)


def env(cap=10000, track=TRACK):
    return W.F1TenthSB3Wrapper(
        gym.make("f1tenth_gym:f1tenth-v0",
                 config={"num_agents": 1, "timestep": 0.01, "map": track}),
        max_episode_steps=cap,
    )


def rollout(speed, steer=0.0, cap=10000):
    e = env(cap)
    e.reset()
    a = to_norm(speed, steer)
    total, n, laps, dist = 0.0, 0, 0.0, 0.0
    max_step_delta = 0.0
    prev = 0.0
    outcome, last_r = "running", 0.0
    for _ in range(cap):
        _, r, term, trunc, info = e.step(a)
        total += r
        n += 1
        laps, dist = info["laps"], info["progress_m"]
        max_step_delta = max(max_step_delta, abs(dist - prev))
        prev = dist
        last_r = r
        if term or trunc:
            outcome = ("crash" if info.get("collision")
                       else ("timeout" if trunc else "finished"))
            break
    e.close()
    return dict(total=total, steps=n, outcome=outcome, laps=laps,
                dist=dist, max_delta=max_step_delta, last_r=last_r)


print("=" * 78)
print("1.1  CONSTANT-SPEED SWEEP  (straight line, 10k cap, %s)" % TRACK)
print("=" * 78)
print(f"{'speed m/s':>10} {'steps':>7} {'outcome':>9} {'RETURN':>10} "
      f"{'dist m':>9} {'laps':>7}")
sweep = {}
for v in [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]:
    r = rollout(v)
    sweep[v] = r
    print(f"{v:>10.2f} {r['steps']:>7,} {r['outcome']:>9} {r['total']:>10.2f} "
          f"{r['dist']:>9.2f} {r['laps']:>7.3f}")

still = sweep[0.0]["total"]
driving = [sweep[v]["total"] for v in sweep if v > 0]
print()
print("  Old reward on this same track, for comparison:")
print("    0.00 -> 100.00   0.25 -> 120.21   5.00 -> 80.85   20.00 -> 80.91")
check(abs(still) < 1e-9, "standing still scores exactly 0.00")
check(all(d > still for d in driving), "every driving speed beats standing still")
check(sweep[0.0]["outcome"] == "timeout",
      "a stationary episode ends by truncation, not by running forever")

print()
print("=" * 78)
print("1.2  CRASH PENALTY FIRES ON COLLISION ONLY")
print("=" * 78)
crash = rollout(15.0)
timeout = rollout(0.0, cap=200)
print(f"  collision step reward : {crash['last_r']:+.3f}  (expect about -{W.CRASH_PENALTY})")
print(f"  truncation step reward: {timeout['last_r']:+.3f}  (expect ~0, no penalty)")
check(crash["outcome"] == "crash" and crash["last_r"] < -1.0,
      "collision applies the crash penalty")
check(timeout["outcome"] == "timeout" and timeout["last_r"] > -1.0,
      "truncation applies NO crash penalty")
print("  Lap completion takes the same no-collision path as truncation, so it is")
print("  likewise unpenalised. f110_env sets terminated for collision OR two laps;")
print("  the wrapper keys off the collision flag, never off `terminated`.")

print()
print("=" * 78)
print("1.3  LAP WRAPAROUND  (walk the centreline for 1.5 laps)")
print("=" * 78)
e = env()
e.reset()
trk = e.env.unwrapped.track
L = float(trk.centerline.ss[-1])
prog = e.progress
x0, y0 = trk.centerline.spline.calc_position(0.0)
prog.reset(float(x0), float(y0))
deltas = []
for s in np.arange(0.0, L * 1.5, 0.5):
    x, y = trk.centerline.spline.calc_position(float(s % L))
    deltas.append(prog.update(float(x), float(y)))
e.close()
deltas = np.array(deltas)
print(f"  track length          : {L:.1f} m")
print(f"  min single-step delta : {deltas.min():+.4f} m")
print(f"  max single-step delta : {deltas.max():+.4f} m")
print(f"  measured laps         : {prog.laps:.3f}   (expected 1.500)")
check(deltas.min() > -1.0, "no spurious negative delta when crossing the start line")
check(abs(prog.laps - 1.5) < 0.02, "cumulative progress matches 1.5 laps")

print()
print("=" * 78)
print("1.4  PROJECTION DOES NOT TELEPORT")
print("=" * 78)
# The car cannot advance further than v_max * dt in one step. Anything larger
# means the nearest-point search jumped to a different part of the track.
physical_max = W.SPEED_MAX * 0.01
budget = physical_max * 1.5  # margin for spline curvature and solver noise
print(f"  physical bound v_max*dt : {physical_max:.4f} m/step")
print(f"  alarm threshold         : {budget:.4f} m/step")
worst = 0.0
for v in [1.0, 5.0, 12.0, 20.0]:
    r = rollout(v)
    worst = max(worst, r["max_delta"])
    print(f"  speed {v:>5.1f} m/s -> max observed delta {r['max_delta']:.4f} m/step")
check(worst <= budget, "no single step advanced further than physically possible")
# The 1.5-lap walk above is excluded from this bound on purpose: it teleports
# along the centreline in fixed 0.5 m increments, so its delta is set by the
# walk, not by vehicle dynamics. It is a wraparound test, not a speed test.

print()
print("=" * 78)
print("ALL STAGE 1 CHECKS PASSED" if not failures
      else f"{len(failures)} STAGE 1 CHECK(S) FAILED")
for f in failures:
    print("  - " + f)
print("=" * 78)
sys.exit(1 if failures else 0)
