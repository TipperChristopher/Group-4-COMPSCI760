"""Learning curve + crash diagnostics for the NEW (progress) reward, with three
optional *diagnostic* levers (each a one-variable change from the baseline):

  --ent-coef F          PPO entropy bonus (exploration).      [algo hyperparam -> diagnostic only]
  --completion-bonus F  one-time reward when a full lap done. [reward/task def -> could be kept if frozen]
  --warmup-track NAME    train the first --warmup-frac of the  [training protocol -> diagnostic only,
  --warmup-frac F        budget on an easier track, then TARGET. confounds the diversity axis]

Answers two questions the single-shot eval could not:
  (1) Does cornering emerge gradually or suddenly?  -> laps vs training steps.
  (2) Why does it crash where it does?              -> speed vs track position,
      and the centreline curvature at the crash point vs the corners it cleared.

Trains in segments (so we can evaluate between them). Evaluation during a phase
uses the SAME track it is training on (the f1tenth scan-simulator is a shared
class attribute, so eval must not swap the map out from under training); the
final detailed rollout is always on the TARGET track for apples-to-apples
crash diagnostics. Reward magnitude knob: --penalty. Steps stay fixed.

    python learning_curve.py --penalty 40 --steps 1000000 --segments 10
    python learning_curve.py --ent-coef 0.01
    python learning_curve.py --completion-bonus 100
    python learning_curve.py --warmup-track synthetic_track_7 --warmup-frac 0.5
"""
import argparse, importlib.util, json, pathlib, sys, types, warnings
warnings.filterwarnings("ignore")
if "gym" not in sys.modules:
    sys.modules["gym"] = types.ModuleType("gym"); sys.modules["gym"].__version__ = "0.0.0"

import numpy as np
import gymnasium as gym
import f1tenth_gym  # noqa: F401
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import safe_mean

HERE = pathlib.Path(__file__).parent
TIMESTEP = 0.01
TRAIN_CAP = 3000       # training-episode truncation (matches the wrapper default)
EVAL_CAP = 15000       # long enough for a full synthetic lap
SPEED_IDX = 108
TARGET = "synthetic_track_0"


def load(name, fn):
    spec = importlib.util.spec_from_file_location(name, HERE / fn)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


W = load("new_reward_wrapper", "new_reward_wrapper.py")


class CompletionBonusWrapper(W.F1TenthSB3Wrapper):
    """Add a one-time reward the first time a full lap is completed."""
    def __init__(self, env, max_episode_steps=W.DEFAULT_MAX_EPISODE_STEPS, completion_bonus=0.0):
        super().__init__(env, max_episode_steps=max_episode_steps)
        self._bonus = float(completion_bonus); self._awarded = False

    def reset(self, **kwargs):
        self._awarded = False
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if not self._awarded and info.get("laps", 0.0) >= 1.0:
            reward += self._bonus; self._awarded = True; info["completion_bonus"] = True
        return obs, reward, terminated, truncated, info


def make_env(wrap_fn, track, cap):
    def _init():
        env = gym.make("f1tenth_gym:f1tenth-v0",
                       config={"num_agents": 1, "timestep": TIMESTEP, "map": track})
        return Monitor(wrap_fn(env, cap))
    return _init


def curvature_along(tp):
    p, s = tp.points, tp.s
    dx, dy = np.gradient(p[:, 0], s), np.gradient(p[:, 1], s)
    ddx, ddy = np.gradient(dx, s), np.gradient(dy, s)
    denom = (dx * dx + dy * dy) ** 1.5
    denom[denom == 0] = 1e-9
    return np.abs(dx * ddy - dy * ddx) / denom


def evaluate(model, wrap_fn, track, want_trace=False, obs_rms=None):
    from new_reward_wrapper import TrackProgress
    base = DummyVecEnv([make_env(wrap_fn, track, EVAL_CAP)])
    core = base.envs[0].unwrapped
    if obs_rms is not None:
        venv = VecNormalize(base, norm_obs=True, norm_reward=False, training=False)
        venv.obs_rms = obs_rms          # reuse the frozen training statistics
    else:
        venv = base
    tp = TrackProgress(); tp.bind(core.track)
    kappa = curvature_along(tp)
    obs = venv.reset(); tp.reset(float(core.poses_x[0]), float(core.poses_y[0]))
    trace, last = [], None
    crashed, crash_laps, steps_taken = False, 0.0, 0
    for _ in range(EVAL_CAP):
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, dones, infos = venv.step(action)
        steps_taken += 1
        if dones[0]:
            info = infos[0] if isinstance(infos[0], dict) else {}
            crashed = bool(info.get("collision"))
            crash_laps = float(info.get("laps", tp.laps))
            break
        tp.update(float(core.poses_x[0]), float(core.poses_y[0]))
        raw = venv.get_original_obs() if obs_rms is not None else obs
        spd = abs(float(raw[0][SPEED_IDX]))
        idx = int(np.argmin(np.abs(tp.s - tp._prev_s)))
        trace.append((float(tp._prev_s), float(tp.cumulative_s), spd, float(kappa[idx])))
        last = trace[-1]
    venv.close()
    laps_val = float(max(crash_laps, tp.laps))
    out = dict(laps=laps_val,
               max_progress_m=float(tp.cumulative_s),
               crashed=crashed,
               eval_steps=steps_taken,
               # time to complete one lap, only meaningful when it actually finished a lap
               lap_time_s=(steps_taken * TIMESTEP / laps_val if laps_val >= 1.0 else None),
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
    ap.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--segments", type=int, default=10)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--completion-bonus", type=float, default=0.0)
    ap.add_argument("--warmup-track", type=str, default="")
    ap.add_argument("--warmup-frac", type=float, default=0.5)
    ap.add_argument("--vecnormalize", action="store_true")
    args = ap.parse_args()

    W.CRASH_PENALTY = args.penalty
    if args.completion_bonus > 0:
        cb = args.completion_bonus
        wrap_fn = lambda env, cap: CompletionBonusWrapper(env, max_episode_steps=cap, completion_bonus=cb)
    else:
        wrap_fn = lambda env, cap: W.F1TenthSB3Wrapper(env, max_episode_steps=cap)

    algo_tag = "" if args.algo == "ppo" else f"_{args.algo}"
    norm_tag = "_vn" if args.vecnormalize else ""
    tag = ""
    if args.ent_coef > 0: tag += f"_ent{args.ent_coef}"
    if args.completion_bonus > 0: tag += f"_bonus{int(args.completion_bonus)}"
    if args.warmup_track: tag += f"_warmup-{args.warmup_track}"

    print(f"=== learning curve [{args.algo.upper()}]: NEW reward, crash_penalty={args.penalty}, "
          f"{args.steps:,} steps in {args.segments} segments ===")
    print(f"    ent_coef={args.ent_coef}  completion_bonus={args.completion_bonus}  "
          f"warmup={args.warmup_track or 'none'}({args.warmup_frac})  vecnormalize={args.vecnormalize}  target={TARGET}")

    # ---- build training phases (curriculum = warmup phase + target phase) ----
    seg = max(1, args.steps // args.segments)
    if args.warmup_track:
        n_warm = max(1, round(args.segments * args.warmup_frac))
        phases = [(args.warmup_track, n_warm), (TARGET, args.segments - n_warm)]
    else:
        phases = [(TARGET, args.segments)]

    model, train_venv, curve, done = None, None, [], 0
    (HERE / "results").mkdir(exist_ok=True)
    out = HERE / "results" / f"learning_curve_cp{int(args.penalty)}{algo_tag}{norm_tag}{tag}_{args.steps}.json"

    def save(final_obj):
        # written after every segment so an interruption keeps the partial curve
        out.write_text(json.dumps(dict(
            penalty=args.penalty, algo=args.algo, vecnormalize=args.vecnormalize,
            steps=args.steps, ent_coef=args.ent_coef, completion_bonus=args.completion_bonus,
            warmup_track=args.warmup_track, warmup_frac=args.warmup_frac,
            curve=curve, final=final_obj), indent=2))

    for track, nseg in phases:
        new_venv = DummyVecEnv([make_env(wrap_fn, track, TRAIN_CAP)])
        if args.vecnormalize:
            new_venv = W.wrap_vecnormalize(new_venv, training=True)
        if model is None:
            if args.algo == "sac":
                model = SAC("MlpPolicy", new_venv, verbose=0, device="cpu", seed=0)
            else:
                model = PPO("MlpPolicy", new_venv, verbose=0, device="cpu",
                            seed=0, ent_coef=args.ent_coef)
        else:
            if train_venv is not None:
                train_venv.close()
            model.set_env(new_venv)
        train_venv = new_venv
        for _ in range(nseg):
            model.learn(total_timesteps=seg, reset_num_timesteps=(done == 0))
            done += 1
            rms = train_venv.obs_rms if args.vecnormalize else None
            ev = evaluate(model, wrap_fn, track, obs_rms=rms)  # eval on the CURRENT training track (safe)
            # SB3 training-side metrics over the last ~100 episodes (Monitor-populated)
            ep_rew = safe_mean([e["r"] for e in model.ep_info_buffer]) if len(model.ep_info_buffer) else float("nan")
            ep_len = safe_mean([e["l"] for e in model.ep_info_buffer]) if len(model.ep_info_buffer) else float("nan")
            curve.append(dict(steps=done * seg, eval_track=track, laps=ev["laps"],
                              max_progress_m=ev["max_progress_m"], crashed=ev["crashed"],
                              speed_before_crash=ev["speed_before_crash"],
                              eval_steps=ev["eval_steps"], lap_time_s=ev["lap_time_s"],
                              ep_rew_mean=float(ep_rew), ep_len_mean=float(ep_len)))
            lt = f"{ev['lap_time_s']:.1f}s" if ev["lap_time_s"] is not None else "  -  "
            print(f"  {done*seg:>9,} steps [{track:>15}] -> laps={ev['laps']:.3f}  "
                  f"ep_rew={ep_rew:7.2f}  ep_len={ep_len:6.0f}  laptime={lt}  "
                  f"crashed={ev['crashed']}", flush=True)
            save(None)   # incremental checkpoint of the curve so far
    final_rms = train_venv.obs_rms if args.vecnormalize else None
    train_venv.close()

    final = evaluate(model, wrap_fn, TARGET, want_trace=True, obs_rms=final_rms)
    tr = np.array(final["trace"]) if final["trace"] else np.zeros((0, 4))
    print(f"\n--- final policy on {TARGET} ---")
    print(f"  reached {final['laps']:.3f} laps ({final['max_progress_m']:.1f} m), crashed={final['crashed']}")
    print(f"  speed just before crash : {final['speed_before_crash']:.2f} m/s")
    print(f"  curvature at crash      : {final['curvature_at_crash']:.4f} 1/m")
    print(f"  track curvature mean/p90: {final['track_curvature_mean']:.4f} / {final['track_curvature_p90']:.4f} 1/m")
    if final["curvature_at_crash"] > final["track_curvature_p90"]:
        print("  => crash corner is SHARPER than 90% of the track (a hard corner).")
    else:
        print("  => crash corner is NOT unusually sharp (likely too fast, not too tight).")

    save(final)
    print(f"\nsaved {out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cs = [c["steps"] for c in curve]; cl = [c["laps"] for c in curve]
        plt.figure(figsize=(7, 4)); plt.plot(cs, cl, "o-")
        plt.xlabel("training steps"); plt.ylabel("laps reached (eval)")
        plt.title(f"Learning curve (cp={args.penalty}{tag})"); plt.grid(alpha=.4)
        plt.tight_layout(); plt.savefig(HERE / "results" / f"learning_curve_cp{int(args.penalty)}{algo_tag}{norm_tag}{tag}_{args.steps}.png", dpi=140); plt.close()
        if tr.shape[0] > 0:
            plt.figure(figsize=(8, 4))
            plt.plot(tr[:, 1], tr[:, 2], label="speed (m/s)")
            plt.plot(tr[:, 1], tr[:, 3] * 20, alpha=.6, label="curvature x20 (1/m)")
            plt.axvline(tr[-1, 1], color="r", ls="--", label="crash")
            plt.xlabel("progress along centreline (m)"); plt.ylabel("speed / scaled curvature")
            plt.title(f"Final policy: speed vs track position ({tag or 'baseline'})"); plt.legend(); plt.grid(alpha=.4)
            plt.tight_layout(); plt.savefig(HERE / "results" / f"speed_profile_cp{int(args.penalty)}{algo_tag}{norm_tag}{tag}_{args.steps}.png", dpi=140); plt.close()
        print("saved plots to results/")
    except Exception as e:
        print(f"(plotting skipped: {e})")


if __name__ == "__main__":
    main()
