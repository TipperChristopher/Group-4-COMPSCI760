"""Spawn-protocol study (ADDITIVE — no team code touched).

Question: does switching the EVAL reset from `rl_grid_static` (raceline spawn)
to `cl_grid_static` (centreline spawn) remove the spurious single-episode
instability the teammate found, where one spawn draw flips crash<->finish?

Method: run Follow-the-Gap and Random (no training required) on a REAL circuit
across several fixed spawn seeds, under BOTH reset protocols, through the exact
rollout machinery evaluate.py uses (DummyVecEnv + CollisionRecorder + venv.seed).
Produces the before/after numbers for the 'changes' slide.

    python spawn_study.py --track Spielberg --seeds 0 1 2 3 4
"""
import argparse, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import learning_curve as lc            # sets gym shim + loads new_reward_wrapper as lc.W
import gymnasium as gym
import f1tenth_gym                      # noqa: F401
from stable_baselines3.common.vec_env import DummyVecEnv
from baselines import GapFollower, RandomPolicy
from new_reward_wrapper import TrackProgress

TIMESTEP = 0.01
EVAL_CAP = 15000
F1 = lc.W.F1TenthSB3Wrapper


class CollisionRecorder(gym.Wrapper):
    """Copy the collision flag into info at episode end, before auto-reset clears it.
    (Same trick as team evaluate.py.)"""
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            info = dict(info) if info else {}
            try:
                info["collision"] = bool(self.env.unwrapped.collisions[0])
            except Exception:
                info["collision"] = None
        return obs, reward, terminated, truncated, info


def make_env(track, rtype, cap):
    def _init():
        env = gym.make("f1tenth_gym:f1tenth-v0",
                       config={"num_agents": 1, "timestep": TIMESTEP, "map": track,
                               "reset_config": {"type": rtype}})
        return CollisionRecorder(F1(env, max_episode_steps=cap))
    return _init


def rollout(policy, track, rtype, seed, cap=EVAL_CAP):
    base = DummyVecEnv([make_env(track, rtype, cap)])
    core = base.envs[0].unwrapped
    cl, rl = core.track.centerline, core.track.raceline
    tp = TrackProgress(); tp.bind(core.track)

    base.seed(seed)                       # evaluate.py's exact seeding path
    obs = base.reset()
    px, py = float(core.poses_x[0]), float(core.poses_y[0])
    off_cl = float(np.hypot(np.array(cl.xs) - px, np.array(cl.ys) - py).min())
    off_rl = float(np.hypot(np.array(rl.xs) - px, np.array(rl.ys) - py).min())
    tp.reset(px, py)

    crashed, steps, outcome = None, 0, "step_cap"
    for _ in range(cap):
        action, _ = policy.predict(obs, deterministic=True)
        obs, rews, dones, infos = base.step(action)
        steps += 1
        if dones[0]:
            info = infos[0] if isinstance(infos[0], dict) else {}
            crashed = info.get("collision")
            if crashed:
                outcome = "crash"
            elif info.get("TimeLimit.truncated"):
                outcome = "truncated"
            else:
                outcome = "finished"
            break
        tp.update(float(core.poses_x[0]), float(core.poses_y[0]))
        if tp.laps >= 1.0:            # team protocol: evaluation terminates at ONE lap
            outcome = "finished"
            break
    laps = float(tp.laps)
    base.close()
    return dict(seed=seed, reset=rtype, off_centreline=round(off_cl, 3),
                off_raceline=round(off_rl, 3), laps=round(laps, 3),
                crashed=crashed, steps=steps, outcome=outcome,
                lap_time_s=(round(steps * TIMESTEP / laps, 2) if laps >= 1.0 else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="Spielberg")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--policies", nargs="+", default=["gap", "random"])
    args = ap.parse_args()

    pol = {"gap": GapFollower(), "random": RandomPolicy(seed=0)}
    results = {}
    print(f"track={args.track}  seeds={args.seeds}\n")
    hdr = f"{'policy':>7} {'reset':>16} {'seed':>4} {'off_cl':>7} {'off_rl':>7} {'laps':>7} {'outcome':>9} {'steps':>7}"
    print(hdr); print("-" * len(hdr))
    for pname in args.policies:
        results[pname] = []
        for rtype in ["rl_grid_static", "cl_grid_static"]:
            for s in args.seeds:
                r = rollout(pol[pname], args.track, rtype, s)
                results[pname].append(r)
                print(f"{pname:>7} {rtype:>16} {s:>4} {r['off_centreline']:>7.3f} "
                      f"{r['off_raceline']:>7.3f} {r['laps']:>7.3f} {r['outcome']:>9} {r['steps']:>7}",
                      flush=True)

    # before/after summary per policy
    print("\n=== before/after summary ===")
    for pname in args.policies:
        for rtype in ["rl_grid_static", "cl_grid_static"]:
            rows = [r for r in results[pname] if r["reset"] == rtype]
            laps = [r["laps"] for r in rows]
            n_crash = sum(1 for r in rows if r["outcome"] == "crash")
            n_fin = sum(1 for r in rows if r["outcome"] == "finished")
            tag = "BEFORE (raceline)" if rtype == "rl_grid_static" else "AFTER  (centreline)"
            print(f"  {pname:>7}  {tag:22}  laps mean={np.mean(laps):.3f} std={np.std(laps):.3f} "
                  f"min={min(laps):.3f} max={max(laps):.3f}  crash={n_crash}/{len(rows)} finish={n_fin}/{len(rows)}")

    out = lc.HERE / "results" / f"spawn_study_{args.track}.json"
    out.write_text(json.dumps(dict(track=args.track, seeds=args.seeds, results=results), indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
