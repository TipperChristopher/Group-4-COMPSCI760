import sys
import types
import os
import math
import json
import re
import time
import hashlib
import argparse
import platform
import subprocess
import importlib.util

# 1. The Bulletproof 'gym' Override (Must be at the very top)
if "gym" not in sys.modules:
    sys.modules["gym"] = types.ModuleType("gym")
sys.modules["gym"].__version__ = "0.0.0"

import gymnasium as gym
import f1tenth_gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_WRAPPER_FILE = os.path.join(_PROJECT_ROOT, "sb3_wrapper.py")
_TRACK_MANIFEST = os.path.join(_PROJECT_ROOT, "tracks", "manifest.json")

DEFAULT_TOTAL_TIMESTEPS = 2_000_000
# Fixed default so that every cell of the grid, and both algorithms, draw the
# same tracks unless the track seed is overridden explicitly.
DEFAULT_TRACK_SEED = 0
DEFAULT_TRACK_PREFIX = "synthetic_track_"


def run_dir_name(algo, diversity, seed, run_tag=""):
    """models/ subfolder for a cell; the tag keeps reruns apart from the original grid."""
    name = f"{algo}_{diversity}tracks_s{seed}"
    return f"{name}_{run_tag}" if run_tag else name


def _load_local_module(module_name, relative_path):
    """Import a project file by path, independent of the current directory."""
    path = os.path.join(_PROJECT_ROOT, relative_path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not find {module_name} at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_wrapper_module = _load_local_module("sb3_wrapper", "sb3_wrapper.py")
F1TenthSB3Wrapper = _wrapper_module.F1TenthSB3Wrapper
wrap_vecnormalize = _wrapper_module.wrap_vecnormalize
save_vecnormalize = _wrapper_module.save_vecnormalize
DEFAULT_MAX_EPISODE_STEPS = _wrapper_module.DEFAULT_MAX_EPISODE_STEPS

_track_pool_module = _load_local_module(
    "track_pool", os.path.join("src", "track_pool.py")
)
TrackPoolWrapper = _track_pool_module.TrackPoolWrapper


# The reward constants exactly as committed in sb3_wrapper.py, captured once.
_FILE_REWARD = {"CRASH_PENALTY": _wrapper_module.CRASH_PENALTY,
                "TIME_COST": _wrapper_module.TIME_COST}


def apply_reward_overrides(overrides=None):
    """Set the reward constants the wrapper reads, for this process.

    F1TenthSB3Wrapper.step() reads CRASH_PENALTY and TIME_COST as globals of
    the sb3_wrapper module at call time, so assigning them on that module (the
    one the environment class was defined in) changes the reward the
    environment actually computes. Unset keys are restored to the committed
    file values, so a run without overrides is identical to before. The file on
    disk is never changed.
    """
    overrides = overrides or {}
    for key, file_value in _FILE_REWARD.items():
        value = overrides.get(key)
        setattr(_wrapper_module, key, float(file_value if value is None else value))
    return effective_reward()


def effective_reward():
    w = _wrapper_module
    return {"PROGRESS_WEIGHT": w.PROGRESS_WEIGHT, "TIME_COST": w.TIME_COST,
            "CRASH_PENALTY": w.CRASH_PENALTY}


def make_env(track_pool, seed, track_seed, stream=0, cache_size=8,
             max_episode_steps=None, monitor_path=None, reward_overrides=None):
    """Build one environment that cycles through the whole track pool.

    The pool is handled inside a single environment rather than by spawning one
    environment per track. See src/track_pool.py for why: the number of
    Stable-Baselines3 updates is inversely proportional to n_envs, and
    f1tenth_gym shares one ray-casting engine across every environment in a
    process.

    reward_overrides ({"CRASH_PENALTY": x, "TIME_COST": y}, either optional) is
    applied inside the factory, so it also holds in a SubprocVecEnv worker,
    which imports sb3_wrapper afresh.
    """

    def _init():
        apply_reward_overrides(reward_overrides)
        # The map given here is only the one loaded at construction time.
        # TrackPoolWrapper replaces it on every reset.
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "num_agents": 1,
                "timestep": 0.01,
                "map": track_pool[0],
            },
        )
        env = F1TenthSB3Wrapper(
            env,
            max_episode_steps=(
                DEFAULT_MAX_EPISODE_STEPS if max_episode_steps is None
                else max_episode_steps
            ),
        )
        env = TrackPoolWrapper(
            env,
            tracks=track_pool,
            track_seed=track_seed,
            stream=stream,
            cache_size=cache_size,
        )
        # Monitor records episode returns, the track each episode ran on, and
        # how far around it the car actually got. With a path it also writes
        # one row per episode to <path>.monitor.csv, which is the per-episode
        # learning curve; without one the curve only ever reached stdout.
        env = Monitor(env, filename=monitor_path,
                      info_keywords=("track_name", "laps", "progress_m"))
        env.action_space.seed(seed)
        return env

    return _init


def expected_updates(model, algo, total_timesteps, n_envs):
    """Describe how much learning the run will actually do.

    This is the quantity the original one-env-per-track setup silently varied
    across diversity levels, so it is worth printing before every run.
    """
    if algo == "PPO":
        rollout = model.n_steps * n_envs
        iterations = total_timesteps // rollout
        minibatches = math.ceil(rollout / model.batch_size)
        gradient_steps = iterations * model.n_epochs * minibatches
        return [
            ("rollout buffer", f"{rollout:,} steps (n_steps={model.n_steps})"),
            ("policy iterations", f"{iterations:,}"),
            ("gradient updates", f"{gradient_steps:,}"),
        ]

    train_freq = model.train_freq
    unit = getattr(train_freq.unit, "value", str(train_freq.unit))
    if unit != "step":
        return [
            ("gradient updates", f"depends on episode length (train_freq={train_freq})"),
        ]

    env_steps_per_train = train_freq.frequency * n_envs
    learning_steps = max(0, total_timesteps - model.learning_starts)
    n_trains = learning_steps // env_steps_per_train
    per_train = (
        model.gradient_steps if model.gradient_steps > 0 else train_freq.frequency
    )
    return [
        ("learning starts", f"{model.learning_starts:,} steps"),
        ("train calls", f"{n_trains:,}"),
        ("gradient updates", f"{n_trains * per_train:,}"),
    ]


def resolved_hyperparameters(model, algo):
    """Every hyperparameter as the constructed model actually holds it."""
    def num(v):
        return v(1.0) if callable(v) else v

    common = {"gamma": model.gamma, "learning_rate": num(model.learning_rate),
              "batch_size": model.batch_size}
    if algo == "PPO":
        pol = model.policy
        return {**common, "n_steps": model.n_steps, "n_epochs": model.n_epochs,
                "gae_lambda": model.gae_lambda, "ent_coef": model.ent_coef,
                "vf_coef": model.vf_coef, "max_grad_norm": model.max_grad_norm,
                "clip_range": num(model.clip_range), "target_kl": model.target_kl,
                "log_std_init": pol.log_std_init, "net_arch": str(pol.net_arch)}
    ent = model.ent_coef
    if not isinstance(ent, str):
        ent = float(ent)
    tf = model.train_freq
    return {**common, "buffer_size": model.buffer_size, "ent_coef": ent,
            "target_entropy": float(model.target_entropy), "tau": model.tau,
            "learning_starts": model.learning_starts,
            "train_freq": f"{tf.frequency} {getattr(tf.unit, 'value', tf.unit)}",
            "gradient_steps": model.gradient_steps,
            "target_update_interval": model.target_update_interval,
            "net_arch": str(model.policy.net_arch)}


def print_startup_summary(args, model, track_pool, n_envs, hparams=None):
    """Guard rail: these numbers must match across cells except for diversity."""
    w = _wrapper_module
    rows = [
        ("algo", args.algo),
        ("run tag", args.run_tag or "(none)"),
        ("diversity (pool size)", f"{args.diversity} tracks"),
        ("track prefix", args.track_prefix),
        ("reward (effective)", f"PROGRESS_WEIGHT {w.PROGRESS_WEIGHT}, TIME_COST {w.TIME_COST}, "
                               f"CRASH_PENALTY {w.CRASH_PENALTY}"),
        ("n_envs", str(n_envs)),
        ("track seed", str(args.track_seed)),
        ("run seed", str(args.seed)),
        ("total timesteps", f"{args.total_timesteps:,}"),
        ("max episode steps", f"{args.max_episode_steps:,}"),
    ]
    rows += expected_updates(model, args.algo, args.total_timesteps, n_envs)
    for key, value in (hparams or {}).items():
        overridden = " (override)" if key in getattr(args, "_hp_overrides", {}) else ""
        rows.append((f"hp {key}", f"{value}{overridden}"))

    width = max(len(k) for k, _ in rows)
    line = "=" * 62
    print(line)
    print("RUN CONFIGURATION")
    print(line)
    for key, value in rows:
        print(f"  {key.ljust(width)} : {value}")
    preview = ", ".join(track_pool[:4])
    if len(track_pool) > 4:
        preview += f", ... (+{len(track_pool) - 4} more)"
    print(f"  {'track pool'.ljust(width)} : {preview}")
    print(line)
    print(
        "Gradient updates and total timesteps must be identical across all\n"
        "diversity levels. Only the pool size above should change."
    )
    print(line)


def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=_PROJECT_ROOT,
                             capture_output=True, text=True)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def code_provenance():
    """Commit and reward identity, recorded so every cell is provably comparable.

    The reward hash covers the whole of sb3_wrapper.py, so it also pins the
    observation, action scaling and episode termination. Identical hashes
    across cells therefore prove more than an identical reward. The constants
    are recorded in plain text alongside so a reader need not trust the hash.
    """
    porcelain = _git("status", "--porcelain", "--untracked-files=no") or ""
    # Split on whitespace rather than slicing at column 3: _git strips the
    # output, which removes the leading space of the first line's status code.
    paths = [line.split(None, 1)[1] for line in porcelain.splitlines() if line.strip()]
    # The submodule pointer is long-standing local state, not project code.
    dirty = [p for p in paths if p != "f1tenth_gym"]
    w = _wrapper_module
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty_files": dirty,
        "reward_code_file": "sb3_wrapper.py",
        "reward_code_sha256": _sha256(_WRAPPER_FILE),
        "reward_constants": {
            "PROGRESS_WEIGHT": w.PROGRESS_WEIGHT,
            "TIME_COST": w.TIME_COST,
            "CRASH_PENALTY": w.CRASH_PENALTY,
            "STEERING_LIMIT": w.STEERING_LIMIT,
            "SPEED_MIN": w.SPEED_MIN,
            "SPEED_MAX": w.SPEED_MAX,
        },
    }


# Manifest splits a pool may be drawn from. Everything else (val, test, real,
# narrowA, narrowB) is evaluation-only and refused. vw_train is accepted once
# it exists in the manifest.
TRAINING_SPLITS = ("train", "vw_train")


def verify_training_pool(track_pool):
    """Refuse to train on tracks that differ from the canonical manifest.

    Track names are not unique across machines: make_synth_tracks.py --seed 0
    writes different geometry under the same synthetic_track_N names than the
    seed-123 set in tracks/manifest.json. A model trained on such a pool is not
    comparable with any other cell, and nothing downstream would notice.

    Every pool track must be in the manifest, all in ONE split, and that split
    must be a training split. Held-out names are refused whatever prefix
    produced them.
    """
    if not os.path.exists(_TRACK_MANIFEST):
        raise SystemExit(f"No track manifest at {_TRACK_MANIFEST}. "
                         "Create it with: python tracks/make_heldout_tracks.py")
    with open(_TRACK_MANIFEST) as fh:
        manifest = json.load(fh)
    unknown = [n for n in track_pool if n not in manifest["tracks"]]
    if unknown:
        raise SystemExit(f"Training pool contains tracks not in tracks/manifest.json: {unknown[:5]}")
    splits = sorted({manifest["tracks"][n]["split"] for n in track_pool})
    if len(splits) != 1:
        raise SystemExit(f"Training pool mixes manifest splits {splits}; refusing.")
    split = splits[0]
    if split not in TRAINING_SPLITS:
        raise SystemExit(f"Training pool tracks belong to the '{split}' split, which is "
                         f"held out for evaluation. Training splits: {', '.join(TRAINING_SPLITS)}.")
    mh = _load_local_module("make_heldout_tracks",
                            os.path.join("tracks", "make_heldout_tracks.py"))
    maps = mh.maps_dir()
    bad = []
    for n in track_pool:
        files = manifest["tracks"][n]["files"]
        try:
            ok = mh.same(files, mh.checksums(maps, n, list(files)))
        except FileNotFoundError:
            ok = False
        if not ok:
            bad.append(n)
    if bad:
        raise SystemExit(
            f"{len(bad)} of {len(track_pool)} pool tracks do not match "
            f"tracks/manifest.json, e.g. {bad[:5]}. Refusing to train on "
            "non-canonical tracks. Check with: python tracks/make_heldout_tracks.py --verify")
    return {"manifest_sha256": _sha256(_TRACK_MANIFEST),
            "pool_verified_against_manifest": True,
            "track_split": split}


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def main():
    # Setup command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, choices=["PPO", "SAC"], required=True)
    parser.add_argument("--diversity", type=int, choices=[1, 5, 20, 100], required=True)
    parser.add_argument("--seed", type=int, required=True,
                        help="Run seed: network init, action sampling, torch.")
    parser.add_argument("--track-seed", type=int, default=DEFAULT_TRACK_SEED,
                        help="Seed controlling which tracks are selected and in "
                             "what order. Independent of --seed so that PPO and "
                             "SAC train on identical tracks.")
    parser.add_argument("--n-envs", type=int, default=1,
                        help="Number of parallel environments. Kept at 1 so the "
                             "number of learning updates is constant across "
                             "diversity levels. NOT tied to --diversity.")
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--track-cache-size", type=int, default=8,
                        help="Tracks held in memory to keep reset() cheap. "
                             "Roughly 20 MB each.")
    parser.add_argument("--max-episode-steps", type=int,
                        default=DEFAULT_MAX_EPISODE_STEPS,
                        help="Episode step limit, returned as truncated (not "
                             "terminated). f1tenth_gym never truncates on its "
                             "own, so without this a stalled policy runs "
                             "forever, reset() never fires and the track pool "
                             "never advances. Default 3000 is 30 s of sim time, "
                             "about one lap of a synthetic track at 6 m/s.")
    parser.add_argument("--torch-threads", type=int, default=None,
                        help="Cap torch's intra-op threads for this process. "
                             "Set to 1 when running cells in parallel: torch "
                             "defaults to one thread per physical core, so eight "
                             "concurrent runs would oversubscribe the CPU.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow reusing a run directory that already holds "
                             "a run_config.json. Off by default so relaunching a "
                             "grid cannot clobber a run that is in progress or done.")
    parser.add_argument("--run-tag", type=str, default="",
                        help="Appended to the run directory, e.g. r40 gives "
                             "models/SAC_20tracks_s0_r40/. Default: none, the "
                             "original directory name.")
    parser.add_argument("--track-prefix", type=str, default=DEFAULT_TRACK_PREFIX,
                        help="Pool is <prefix>0..N-1. Must name tracks of a "
                             f"training split in tracks/manifest.json "
                             f"({', '.join(TRAINING_SPLITS)}).")
    parser.add_argument("--checkpoint-freq", type=int, default=100000,
                        help="Save a checkpoint every N env steps (default 100000).")

    # --- Optional overrides (ported from Aolin's team/crash-penalty-flag) ----
    # Every flag defaults to None = not set: the SB3 default, or the value in
    # sb3_wrapper.py, applies, so a run without them is identical to before.
    # Resolved values are recorded in run_config.json and printed at startup.
    hp = parser.add_argument_group("hyperparameter overrides (unset = SB3 default)")
    hp.add_argument("--gamma", type=float, default=None, help="Discount factor (SB3 default 0.99).")
    hp.add_argument("--learning-rate", type=float, default=None)
    hp.add_argument("--batch-size", type=int, default=None)
    hp.add_argument("--ent-coef", type=str, default=None,
                    help="Entropy coefficient. PPO: a number (default 0.0). SAC: a number "
                         "fixes it; 'auto' or 'auto_<init>' tunes it. Unset keeps SAC's "
                         "default 'auto'.")
    hp.add_argument("--gae-lambda", type=float, default=None, help="PPO only (default 0.95).")
    hp.add_argument("--n-steps", type=int, default=None, help="PPO only: rollout length (default 2048).")
    hp.add_argument("--n-epochs", type=int, default=None, help="PPO only (default 10).")
    hp.add_argument("--target-kl", type=float, default=None, help="PPO only (default None).")
    hp.add_argument("--log-std-init", type=float, default=None, help="PPO only (default 0.0).")
    hp.add_argument("--buffer-size", type=int, default=None, help="SAC only (default 1,000,000).")
    rw = parser.add_argument_group("reward overrides (unset = value in sb3_wrapper.py)")
    rw.add_argument("--crash-penalty", type=float, default=None,
                    help="Override CRASH_PENALTY for this run (file value 40.0).")
    rw.add_argument("--time-cost", type=float, default=None,
                    help="Override TIME_COST per step for this run (file value 0.0).")
    args = parser.parse_args()

    if args.n_envs < 1:
        parser.error("--n-envs must be at least 1")
    if args.run_tag and not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_tag):
        parser.error("--run-tag may contain only letters, digits, '_' and '-'")
    if args.checkpoint_freq < 1:
        parser.error("--checkpoint-freq must be at least 1")
    only = {"PPO": ("gae_lambda", "n_steps", "n_epochs", "target_kl", "log_std_init"),
            "SAC": ("buffer_size",)}
    for algo, names in only.items():
        stray = [f"--{n.replace('_', '-')}" for n in names
                 if getattr(args, n) is not None and args.algo != algo]
        if stray:
            parser.error(f"{', '.join(stray)} apply to {algo} only (--algo {args.algo} given)")
    if args.ent_coef is not None:
        try:
            args.ent_coef = float(args.ent_coef)
        except ValueError:
            if args.algo != "SAC" or not args.ent_coef.startswith("auto"):
                parser.error("--ent-coef must be a number (or 'auto[_init]' for SAC)")

    hp_names = ("gamma", "learning_rate", "batch_size", "ent_coef", "gae_lambda", "n_steps",
                "n_epochs", "target_kl", "buffer_size")
    model_kwargs = {n: getattr(args, n) for n in hp_names if getattr(args, n) is not None}
    if args.log_std_init is not None:
        model_kwargs["policy_kwargs"] = {"log_std_init": args.log_std_init}
    args._hp_overrides = {**{k: v for k, v in model_kwargs.items() if k != "policy_kwargs"},
                          **({"log_std_init": args.log_std_init}
                             if args.log_std_init is not None else {})}
    reward_overrides = {k: v for k, v in (("CRASH_PENALTY", args.crash_penalty),
                                          ("TIME_COST", args.time_cost)) if v is not None}
    # Applied here too (not only inside the env factory) so that the provenance
    # block, run_config.json and the startup summary all show what the
    # environment will use.
    apply_reward_overrides(reward_overrides)

    if args.torch_threads:
        import torch
        torch.set_num_threads(args.torch_threads)

    # The pool grows with diversity; the environment count does not.
    track_pool = [f"{args.track_prefix}{i}" for i in range(args.diversity)]

    # One directory per cell. Absolute, so the launch directory cannot change it.
    save_dir = os.path.join(_PROJECT_ROOT, "models", run_dir_name(
        args.algo, args.diversity, args.seed, args.run_tag))
    config_path = os.path.join(save_dir, "run_config.json")
    if os.path.exists(config_path) and not args.overwrite:
        raise SystemExit(f"{save_dir} already holds a run. Refusing to overwrite it; "
                         "pass --overwrite to reuse the directory deliberately.")
    # Before creating the directory, so a refused pool leaves nothing behind.
    pool_info = verify_training_pool(track_pool)
    os.makedirs(save_dir, exist_ok=True)
    prov = code_provenance()
    import torch
    import stable_baselines3
    run_config = {
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pid": os.getpid(),
        "host": platform.node(),
        **prov,
        "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "hyperparameter_overrides": args._hp_overrides,
        "reward_overrides": reward_overrides,
        "effective_reward_constants": effective_reward(),
        "reward_constants_in_file": dict(_FILE_REWARD),
        "run_tag": args.run_tag,
        "track_prefix": args.track_prefix,
        "track_pool": track_pool,
        **pool_info,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "env_threads": {k: os.environ.get(k) for k in
                        ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "stable_baselines3": stable_baselines3.__version__},
        "outputs": {"checkpoints_every": args.checkpoint_freq, "monitor_csv": "monitor_*.monitor.csv",
                    "progress_csv": "progress.csv", "final_model": "final_model.zip"},
    }
    _write_json(config_path, run_config)

    line = "=" * 62
    print(line)
    print("RUN PROVENANCE")
    print(line)
    print(f"  commit            : {prov['git_commit']} ({prov['git_branch']})")
    print(f"  uncommitted code  : {prov['git_dirty_files'] or 'none'}")
    print(f"  reward code sha256: {prov['reward_code_sha256']}")
    rc = prov["reward_constants"]
    print(f"  reward constants  : PROGRESS_WEIGHT {rc['PROGRESS_WEIGHT']}  "
          f"TIME_COST {rc['TIME_COST']}  CRASH_PENALTY {rc['CRASH_PENALTY']}"
          + (f"  (overridden: {reward_overrides}; file: {_FILE_REWARD})" if reward_overrides else ""))
    print(f"  run tag           : {args.run_tag or '(none)'}")
    print(f"  track prefix      : {args.track_prefix} (manifest split "
          f"'{pool_info['track_split']}')")
    print(f"  pool verified     : {len(track_pool)} tracks match tracks/manifest.json")
    print(f"  torch threads     : {torch.get_num_threads()}   device: cpu")
    print(f"  run directory     : {save_dir}")
    print(line)
    if prov["git_dirty_files"]:
        print("WARNING: uncommitted changes to tracked code. This run cannot be "
              "reproduced from its commit hash alone.")

    n_envs = args.n_envs
    env_fns = [
        make_env(
            track_pool,
            seed=args.seed + i,
            track_seed=args.track_seed,
            stream=i,
            cache_size=args.track_cache_size,
            max_episode_steps=args.max_episode_steps,
            monitor_path=os.path.join(save_dir, f"monitor_{i}"),
            reward_overrides=reward_overrides,
        )
        for i in range(n_envs)
    ]

    if n_envs == 1:
        vec_env = DummyVecEnv(env_fns)
    else:
        # f1tenth_gym stores its ray-casting engine in a class attribute shared
        # by every environment in a process, so several environments in one
        # process would all raycast against the same map. Separate processes
        # are the only way to run different maps concurrently.
        print(
            f"n_envs={n_envs}: using SubprocVecEnv because f1tenth_gym shares "
            "its scan simulator across environments within a process."
        )
        vec_env = SubprocVecEnv(env_fns)

    # Identical observation normalisation for both algorithms. Rewards are left
    # untouched because reward is the measured outcome of the experiment.
    vec_env = wrap_vecnormalize(vec_env, training=True)

    # Instantiate the selected algorithm
    # Both pinned to CPU explicitly. SAC previously defaulted to device="auto",
    # so installing CUDA torch would have silently moved one algorithm, and
    # only one, onto the GPU.
    # Overrides are passed only when set, so an unflagged run makes exactly the
    # same constructor call as before (and SAC keeps ent_coef="auto").
    algo_cls = PPO if args.algo == "PPO" else SAC
    model = algo_cls("MlpPolicy", vec_env, verbose=1, seed=args.seed, device="cpu",
                     **model_kwargs)

    hparams = resolved_hyperparameters(model, args.algo)
    run_config["hyperparameters"] = hparams
    run_config["effective_reward_constants"] = effective_reward()
    _write_json(config_path, run_config)

    # Training statistics (ep_rew_mean, losses, fps, ...) to progress.csv as
    # well as stdout, so convergence curves survive the terminal.
    model.set_logger(configure(save_dir, ["stdout", "csv"]))

    print_startup_summary(args, model, track_pool, n_envs, hparams)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq // n_envs),
        save_path=save_dir,
        name_prefix=f"{args.algo}_checkpoint",
        save_vecnormalize=True,
    )

    print(f"Starting {args.algo} training loop for {args.total_timesteps:,} steps...")
    t0 = time.time()
    model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback)

    # Save the final model alongside the normalisation statistics it needs.
    final_path = os.path.join(save_dir, "final_model")
    model.save(final_path)
    stats_path = save_vecnormalize(vec_env, save_dir)
    vec_env.close()

    run_config.update({
        "status": "completed",
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "wall_seconds": round(time.time() - t0, 1),
        "num_timesteps": int(model.num_timesteps),
        "steps_per_second": round(model.num_timesteps / max(time.time() - t0, 1e-9), 1),
    })
    _write_json(config_path, run_config)

    print(f"Training complete! Model saved to {final_path}.zip")
    if stats_path:
        print(f"Normalisation statistics saved to {stats_path}")


if __name__ == "__main__":
    main()
