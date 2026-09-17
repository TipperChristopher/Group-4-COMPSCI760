"""Evaluate non-learned baselines through the SAME harness as the RL runs, so
laps / lap-time are directly comparable to the PPO/SAC learning curves.

  * GapFollower  - Follow-the-Gap, a classical reactive LiDAR controller (no training)
  * RandomPolicy - uniform actions, the floor

Reuses learning_curve.evaluate() and F1TenthSB3Wrapper (obs_rms=None: the
baselines read raw LiDAR ranges). Runs on the train track and two unseen tracks,
so we get both in-distribution and (zero-shot) held-out numbers for free.

    python baseline_eval.py
"""
import json, warnings
warnings.filterwarnings("ignore")
import learning_curve as lc          # sets the gym shim + loads the wrapper (no main() run)
from baselines import GapFollower, RandomPolicy

TRACKS = ["synthetic_track_0", "synthetic_track_1", "synthetic_track_2"]
wrap_fn = lambda env, cap: lc.W.F1TenthSB3Wrapper(env, max_episode_steps=cap)


def main():
    policies = [("gap", GapFollower()), ("random", RandomPolicy(seed=0))]
    results = {}
    print(f"{'baseline':>8} {'track':>18} {'laps':>7} {'lap_time_s':>11} {'crashed':>8} {'eval_steps':>11}")
    for name, policy in policies:
        results[name] = []
        for track in TRACKS:
            ev = lc.evaluate(policy, wrap_fn, track, obs_rms=None)
            lt = f"{ev['lap_time_s']:.1f}" if ev["lap_time_s"] is not None else "-"
            print(f"{name:>8} {track:>18} {ev['laps']:>7.3f} {lt:>11} "
                  f"{str(ev['crashed']):>8} {ev['eval_steps']:>11}", flush=True)
            results[name].append(dict(track=track, laps=ev["laps"], lap_time_s=ev["lap_time_s"],
                                      crashed=ev["crashed"], eval_steps=ev["eval_steps"],
                                      max_progress_m=ev["max_progress_m"]))
    (lc.HERE / "results").mkdir(exist_ok=True)
    out = lc.HERE / "results" / "baselines.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
