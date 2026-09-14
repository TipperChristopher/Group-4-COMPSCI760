"""Learning curve + crash diagnostics for the NEW (progress) reward.

Answers two questions the single-shot eval could not:
  (1) Does cornering emerge gradually or suddenly?  -> laps vs training steps.
  (2) Why does it crash where it does?              -> speed vs track position,
      and the centreline curvature at the crash point vs the corners it cleared.

Trains in segments (so we can evaluate between them without saving/loading
checkpoints), all on synthetic_track_0. Reward knob only: --penalty. Steps and
PPO hyperparameters stay fixed.

    python learning_curve.py --penalty 40 --steps 1000000 --segments 10
"""
import argparse, importlib.util, json, pathlib, sys, types, warnings
warnings.filterwarnings("ignore")
if "gym" not in sys.modules:
    sys.modules["gym"] = types.ModuleType("gym"); sys.modules["gym"].__version__ = "0.0.0"

import numpy as np
import gymnasium as gym
import f1tenth_gym  # noqa: F401
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

HERE = pathlib.Path(__file__).parent
TIMESTEP = 0.01
TRAIN_CAP = 3000       # training-episode truncation (matches the wrapper default)
EVAL_CAP = 15000       # long enough for a full synthetic lap
SPEED_IDX = 108
TRACK = "synthetic_track_0"


def load(name, fn):
    spec = importlib.util.spec_from_file_location(name, HERE / fn)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


def make_env(WrapperCls, track, cap):
    def _init():
        env = gym.make("f1tenth_gym:f1tenth-v0",
                       config={"num_agents": 1, "timestep": TIMESTEP, "map": track})
        return Monitor(WrapperCls(env, max_episode_steps=cap))
    return _init


def curvature_along(tp):
    """Curvature |k| at each centreline sample, aligned with tp.s (1/m)."""
    p, s = tp.points, tp.s
    dx, dy = np.gradient(p[:, 0], s), np.gradient(p[:, 1], s)
    ddx, ddy = np.gradient(dx, s), np.gradient(dy, s)
    denom = (dx * dx + dy * dy) ** 1.5
    denom[denom == 0] = 1e-9
    return np.abs(dx * ddy - dy * ddx) / denom


def evaluate(model, WrapperCls, track, want_trace=False):
    from new_reward_wrapper import TrackProgress
    venv = DummyVecEnv([make_env(WrapperCls, track, EVAL_CAP)])
    core = venv.envs[0].unwrapped
    tp = TrackProgress(); tp.bind(core.track)
    kappa = curvature_along(tp)
    obs = venv.reset(); tp.reset(float(core.poses_x[0]), float(core.poses_y[0]))
    trace, last = [], None
    crashed, crash_laps = False, 0.0
    for _ in range(EVAL_CAP):
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, dones, infos = venv.step(action)
        if dones[0]:
            info = infos[0] if isinstance(infos[0], dict) else {}
            crashed = bool(info.get("collision"))
            crash_laps = float(info.get("laps", tp.laps))
            break
        tp.update(float(core.poses_x[0]), float(core.poses_y[0]))
        spd = abs(float(obs[0][SPEED_IDX]))
        # curvature at the current centreline position
        idx = int(np.argmin(np.abs(tp.s - tp._prev_s)))
        trace.append((float(tp._prev_s), float(tp.cumulative_s), spd, float(kappa[idx])))
        last = trace[-1]
    venv.close()
    out = dict(laps=float(max(crash_laps, tp.laps)),
               max_progress_m=float(tp.cumulative_s),
               crashed=crashed,
               speed_before_crash=(last[2] if last else 0.0),
               curvature_at_crash=(last[3] if last else 0.0),
               track_curvature_mean=float(np.mean(kappa)),
               track_curvature_p90=float(np.percentile(kappa, 90)))
    if want_trace:
        out["trace"] = trace
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--penalty", type=float, default=40.0)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--segments", type=int, default=10)
    args = ap.parse_args()

    W = load("new_reward_wrapper", "new_reward_wrapper.py")
    W.CRASH_PENALTY = args.penalty
    WrapperCls = W.F1TenthSB3Wrapper
    print(f"=== learning curve: NEW reward, crash_penalty={args.penalty}, "
          f"{args.steps:,} steps in {args.segments} segments, {TRACK} ===")

    train_venv = DummyVecEnv([make_env(WrapperCls, TRACK, TRAIN_CAP)])
    model = PPO("MlpPolicy", train_venv, verbose=0, device="cpu", seed=0)

    seg = max(1, args.steps // args.segments)
    curve = []
    for i in range(args.segments):
        model.learn(total_timesteps=seg, reset_num_timesteps=(i == 0))
        done_steps = (i + 1) * seg
        ev = evaluate(model, WrapperCls, TRACK)
        curve.append(dict(steps=done_steps, laps=ev["laps"],
                          max_progress_m=ev["max_progress_m"],
                          crashed=ev["crashed"],
                          speed_before_crash=ev["speed_before_crash"]))
        print(f"  {done_steps:>9,} steps -> laps={ev['laps']:.3f}  "
              f"progress={ev['max_progress_m']:6.1f} m  "
              f"speed@crash={ev['speed_before_crash']:5.2f}  crashed={ev['crashed']}")
    train_venv.close()

    # detailed final rollout on the trained track (with per-step trace)
    final = evaluate(model, WrapperCls, TRACK, want_trace=True)
    tr = np.array(final["trace"]) if final["trace"] else np.zeros((0, 4))
    print("\n--- final policy on %s ---" % TRACK)
    print(f"  reached {final['laps']:.3f} laps ({final['max_progress_m']:.1f} m), crashed={final['crashed']}")
    print(f"  speed just before crash : {final['speed_before_crash']:.2f} m/s")
    print(f"  curvature at crash      : {final['curvature_at_crash']:.4f} 1/m")
    print(f"  track curvature mean/p90: {final['track_curvature_mean']:.4f} / {final['track_curvature_p90']:.4f} 1/m")
    if final["curvature_at_crash"] > final["track_curvature_p90"]:
        print("  => crash corner is SHARPER than 90% of the track (a hard corner).")
    else:
        print("  => crash corner is NOT unusually sharp (likely too fast, not too tight).")

    results = dict(penalty=args.penalty, steps=args.steps, curve=curve, final=final)
    (HERE / "results").mkdir(exist_ok=True)
    out = HERE / "results" / f"learning_curve_cp{int(args.penalty)}_{args.steps}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out}")

    # plots (best-effort)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cs = [c["steps"] for c in curve]; cl = [c["laps"] for c in curve]
        plt.figure(figsize=(7, 4)); plt.plot(cs, cl, "o-")
        plt.xlabel("training steps"); plt.ylabel("laps reached (eval)")
        plt.title(f"Learning curve (crash_penalty={args.penalty})"); plt.grid(alpha=.4)
        plt.tight_layout(); plt.savefig(HERE / "results" / f"learning_curve_cp{int(args.penalty)}.png", dpi=140); plt.close()
        if tr.shape[0] > 0:
            plt.figure(figsize=(8, 4))
            plt.plot(tr[:, 1], tr[:, 2], label="speed (m/s)")
            plt.plot(tr[:, 1], tr[:, 3] * 20, alpha=.6, label="curvature x20 (1/m)")
            plt.axvline(tr[-1, 1], color="r", ls="--", label="crash")
            plt.xlabel("progress along centreline (m)"); plt.ylabel("speed / scaled curvature")
            plt.title("Final policy: speed vs track position"); plt.legend(); plt.grid(alpha=.4)
            plt.tight_layout(); plt.savefig(HERE / "results" / f"speed_profile_cp{int(args.penalty)}.png", dpi=140); plt.close()
        print("saved plots to results/")
    except Exception as e:
        print(f"(plotting skipped: {e})")


if __name__ == "__main__":
    main()
