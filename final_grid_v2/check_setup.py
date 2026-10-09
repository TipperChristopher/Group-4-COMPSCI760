"""Setup fingerprint for final grid v2: prove a machine trains exactly like the v2 reference.

    python final_grid_v2/check_setup.py --out final_grid_v2/fingerprint_<name>.txt
    python final_grid_v2/check_setup.py --compare final_grid_v2/fingerprint_<name>.txt

The first form checks this machine and prints a short fingerprint (and writes it
with --out). The second diffs a fingerprint against final_grid_v2/reference_fingerprint.txt
(Desmond's machine) and prints MATCH or every difference. It exits 0 on MATCH.

Every line is "key = value". Keys starting with "info." are allowed to differ
(machine, timings, floating-point training numbers); every other key must match
exactly. What is checked:

  code          code identity (content hash of the tracked files and final_grid/ as
                they are on disk, so it does not depend on commits or timestamps),
                f1tenth_gym submodule at the pinned commit and installed editable
                from ./f1tenth_gym,
                LF-normalised sha256 of sb3_wrapper.py, train.py, tracks/manifest.json
  python        version, venv, key packages, and every package against
                final_grid/requirements-lock.txt
  tracks        every vw_train and vw_val track file against tracks/manifest.json
  settings      the 24 v2 jobs in final_grid_v2/jobs_v2_{grant,chris,desmond_ppo}.txt
                (each cell exactly once, split as planned, every line = grid A's line
                for the same cell + the v2 flags of final_grid_v2/DECISION_v2.md, tags
                v2_<ALGO>_d<N>_s<seed>), and the SAC and PPO hyperparameters and
                reward constants as train.py resolves them (read back from the
                smoke runs' run_config.json)
  determinism   a fixed open-loop action sequence (no neural network) driven on
                3 vw_train tracks from a seeded spawn: steps, end reason, progress,
                return and a hash of the whole trajectory. Run twice per track,
                which must agree within the machine too.
  smoke         2,000-step SAC and 8,192-step PPO runs through train.py with the v2
                settings, asserting that every crash pays exactly -5.0, the effective
                reward is penalty 5 / time cost 0, and PPO resolves gamma 0.999,
                n_steps 8192, ent_coef 0.01. Their run folders are deleted.
                (determinism deliberately uses the file constants, so its lines are
                identical to grid A's reference.)

Run with the repo's own venv: .venv\\Scripts\\python.exe final_grid_v2\\check_setup.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata as md
import importlib.util
import json
import math
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
HERE = REPO / "final_grid_v2"
GRID_A = REPO / "final_grid"
REFERENCE = HERE / "reference_fingerprint.txt"
LOCK = GRID_A / "requirements-lock.txt"   # same environment as grid A
DIVERSITIES = (1, 5, 20, 100)
# Who runs which v2 cells (algorithm, diversity, seed); each queue runs with -MaxThreads 12.
CELLS = {
    "grant": [("SAC", d, 0) for d in DIVERSITIES] + [("SAC", 1, 1), ("SAC", 5, 1)],
    "chris": [("SAC", d, 2) for d in DIVERSITIES] + [("SAC", 20, 1), ("SAC", 100, 1)],
    "desmond_ppo": [("PPO", d, s) for s in (0, 1, 2) for d in DIVERSITIES],
}
JOB_FILES = {who: HERE / f"jobs_v2_{who}.txt" for who in CELLS}
GRID_A_JOB_FILES = [GRID_A / f"final_jobs_{who}.txt" for who in ("desmond", "grant", "chris")]
# v2 = grid A's line for the same cell + exactly these flags (final_grid_v2/DECISION_v2.md).
V2_EXTRA = {"SAC": ["--crash-penalty", "5"],
            "PPO": ["--crash-penalty", "5", "--gamma", "0.999", "--n-steps", "8192", "--ent-coef", "0.01"]}
V2_EXPECT_REWARD = {"PROGRESS_WEIGHT": 1.0, "TIME_COST": 0.0, "CRASH_PENALTY": 5.0}
V2_EXPECT_HP = {"SAC": {"gamma": 0.99}, "PPO": {"gamma": 0.999, "n_steps": 8192, "ent_coef": 0.01}}

PYTHON_VERSION = "3.12.10"
F1TENTH_PIN = "5a301bd0ae1ceaf7dec653e7549c8d099db58a6b"   # upstream f1tenth/f1tenth_gym, branch v1.0.0
KEY_PACKAGES = ("torch", "stable_baselines3", "gymnasium", "numpy", "numba", "scipy")
HASHED_FILES = ("sb3_wrapper.py", "train.py", "tracks/manifest.json")
# Fingerprint files are excluded from the code identity, so committing the
# reference after generating it does not change what it certifies.
FINGERPRINT_RE = re.compile(r"^final_grid(_v2)?/(reference_)?fingerprint[^/]*\.txt$")

DETERMINISM_TRACKS = ("vw_synthetic_track_0", "vw_synthetic_track_7", "vw_synthetic_track_12")
RESET_SEED = 2026
SMOKE_STEPS = {"SAC": 2000, "PPO": 8192}   # PPO: one full 8,192-step rollout and update
# Used only if the job files are missing (set up before they were committed).
PROVISIONAL = {"SAC": ["--algo", "SAC", "--track-prefix", "vw_synthetic_track_"],
               "PPO": ["--algo", "PPO", "--track-prefix", "vw_synthetic_track_"]}


# ----------------------------------------------------------------- helpers

def git(*args, cwd=REPO, strip=True) -> str:
    out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout.strip() if strip else out.stdout


def sha_lf(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha_raw(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_jobs(path: pathlib.Path) -> list[tuple[str, list[str]]]:
    jobs = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tag, rest = line.split("|", 1)
        jobs.append((tag.strip(), rest.split()))
    return jobs


def get_flag(argv: list[str], flag: str):
    return argv[argv.index(flag) + 1] if flag in argv else None


def drop_flags(argv: list[str], flags) -> list[str]:
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
        elif a in flags:
            skip = True
        else:
            out.append(a)
    return out


def set_flag(argv: list[str], flag: str, value) -> list[str]:
    return drop_flags(argv, {flag}) + [flag, str(value)]


# ------------------------------------------------------------------ checks

def working_tree_id() -> str:
    """Git tree id of the code as it is on disk: HEAD plus any edits to tracked
    files plus every non-ignored file under final_grid/ and final_grid_v2/. Built in a throwaway
    index, so the real index is never touched. A clean checkout gives HEAD's
    tree; uncommitted work gives the tree it will have once committed."""
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        env = dict(os.environ, GIT_INDEX_FILE=str(pathlib.Path(t) / "index"))
        for args in (("read-tree", "HEAD"), ("add", "-u"), ("add", "-A", "--", "final_grid", "final_grid_v2"),
                     ("write-tree",)):
            out = subprocess.run(["git", *args], cwd=REPO, env=env, capture_output=True, text=True)
            if out.returncode:
                raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
        return out.stdout.strip()


def check_code(lines: list[tuple[str, str]]) -> None:
    tree = [ln for ln in git("ls-tree", "-r", working_tree_id()).splitlines()
            if not FINGERPRINT_RE.match(ln.split("\t", 1)[1])]
    lines.append(("git.code_tree", hashlib.sha256("\n".join(tree).encode()).hexdigest()[:16]))
    # Not stripped: porcelain lines start with a status column that may be a space.
    dirty = sorted({ln[3:] for ln in git("status", "--porcelain", "--untracked-files=no",
                                         strip=False).splitlines()
                    if ln.strip() and not FINGERPRINT_RE.match(ln[3:])})
    # Information only: code_tree above already certifies the content. launch_queue.ps1
    # refuses to start from a tree with uncommitted changes anyway.
    lines.append(("info.git.tree", "clean" if not dirty else "uncommitted: " + ", ".join(dirty)))
    lines.append(("git.branch", git("rev-parse", "--abbrev-ref", "HEAD")))
    lines.append(("info.git.commit", git("rev-parse", "HEAD")))

    gitlink = git("ls-tree", "HEAD", "f1tenth_gym").split()
    gitlink = gitlink[2] if len(gitlink) > 2 else "missing"
    sub = REPO / "f1tenth_gym"
    try:
        head = git("rev-parse", "HEAD", cwd=sub)
        sub_dirty = git("status", "--porcelain", cwd=sub)
    except RuntimeError:
        head, sub_dirty = "not checked out", ""
    state = ["pinned" if head == F1TENTH_PIN else f"NOT THE PIN {F1TENTH_PIN[:12]}",
             "clean" if not sub_dirty else "MODIFIED",
             "gitlink matches" if gitlink == head else f"gitlink {gitlink[:12]} differs"]
    lines.append(("f1tenth_gym.commit", f"{head[:12]} ({', '.join(state)})"))
    spec = importlib.util.find_spec("f1tenth_gym")
    origin = pathlib.Path(spec.origin).resolve() if spec and spec.origin else None
    if origin is None:
        where = "NOT INSTALLED"
    elif sub.resolve() in origin.parents:
        where = "./f1tenth_gym (editable)"
    else:
        where = f"ELSEWHERE: {origin.parent}"
    lines.append(("f1tenth_gym.installed_from", where))

    for f in HASHED_FILES:
        lines.append((f"sha256.{f}", sha_lf(REPO / f)))
    # train.py records the RAW hash of sb3_wrapper.py in every run_config.json; it
    # equals the line above only with an LF checkout (core.autocrlf false).
    lines.append(("info.sha256_raw.sb3_wrapper.py", sha_raw(REPO / "sb3_wrapper.py")))


def check_python(lines: list[tuple[str, str]]) -> None:
    venv = pathlib.Path(sys.prefix).resolve()
    where = "./.venv" if venv == (REPO / ".venv").resolve() else f"NOT THE REPO VENV: {venv}"
    bits = "64-bit" if sys.maxsize > 2 ** 32 else "32-bit"
    lines.append(("python", f"{platform.python_version()} {bits}, {where}"))
    for p in KEY_PACKAGES:
        try:
            v = md.version(p)
        except md.PackageNotFoundError:
            v = "NOT INSTALLED"
        lines.append((f"pkg.{p}", v))

    want = {}
    for raw in LOCK.read_text().splitlines():
        m = re.fullmatch(r"([A-Za-z0-9_.\-]+)==(\S+)", raw.strip())
        if m:
            want[norm(m.group(1))] = m.group(2)
    have = {norm(d.metadata["Name"]): d.version for d in md.distributions()}
    have = {k: v for k, v in have.items() if k not in ("pip", "f1tenth-gym")}
    problems = [f"{k} {have[k]} (lock {v})" for k, v in sorted(want.items()) if k in have and have[k] != v]
    problems += [f"missing {k}" for k in sorted(set(want) - set(have))]
    problems += [f"extra {k} {have[k]}" for k in sorted(set(have) - set(want))]
    lines.append(("pip.lock", f"MATCH ({len(want)} packages)" if not problems
                  else "DIFFERS: " + "; ".join(problems)))


def check_tracks(lines: list[tuple[str, str]]) -> bool:
    sys.path.insert(0, str(REPO / "tracks"))
    mh = load_module("make_heldout_tracks", REPO / "tracks" / "make_heldout_tracks.py")
    manifest = json.loads((REPO / "tracks" / "manifest.json").read_text())
    maps = mh.maps_dir()
    all_ok = True
    for split in ("vw_train", "vw_val"):
        names = manifest["splits"][split]["names"]
        bad = []
        for n in names:
            files = manifest["tracks"][n]["files"]
            try:
                ok = mh.same(files, mh.checksums(maps, n, list(files)))
            except FileNotFoundError:
                ok = False
            if not ok:
                bad.append(n)
        all_ok &= not bad
        lines.append((f"tracks.{split}", f"{len(names) - len(bad)}/{len(names)} OK" if not bad else
                      f"FAIL {len(names) - len(bad)}/{len(names)} OK, bad: {', '.join(bad[:5])}"
                      + (" ..." if len(bad) > 5 else "")))
    return all_ok


def final_settings(lines: list[tuple[str, str]]) -> dict[str, list[str]]:
    """Check the 24 v2 jobs; return the SAC and PPO arguments minus --diversity/--seed."""
    missing = [p.name for p in JOB_FILES.values() if not p.exists()]
    if missing:
        lines.append(("jobs", f"MISSING {', '.join(missing)}: run git pull"))
        lines.append(("settings.source", "PROVISIONAL: grid A defaults + v2 flags"))
        return {a: PROVISIONAL[a] + V2_EXTRA[a] for a in PROVISIONAL}
    grid_a = {}
    for p in GRID_A_JOB_FILES:
        for _, a in parse_jobs(p):
            grid_a[(get_flag(a, "--algo"), int(get_flag(a, "--diversity")), int(get_flag(a, "--seed")))] = a
    problems, cells, variants = [], [], {"SAC": set(), "PPO": set()}
    for who, want in CELLS.items():
        jobs = parse_jobs(JOB_FILES[who])
        have = sorted((get_flag(a, "--algo"), int(get_flag(a, "--diversity") or -1),
                       int(get_flag(a, "--seed") or -1)) for _, a in jobs)
        if have != sorted(want):
            problems.append(f"{JOB_FILES[who].name}: cells differ from the planned split")
        for tag, a in jobs:
            al, d, sd = get_flag(a, "--algo"), get_flag(a, "--diversity"), get_flag(a, "--seed")
            if tag != f"v2_{al}_d{d}_s{sd}":
                problems.append(f"{JOB_FILES[who].name}: tag {tag} should be v2_{al}_d{d}_s{sd}")
            base = grid_a.get((al, int(d or -1), int(sd or -1)))
            if base is None or a != base + V2_EXTRA.get(al, ["?"]):
                problems.append(f"{tag}: not grid A's line + {' '.join(V2_EXTRA.get(al, []))}")
            if al in variants:
                variants[al].add(" ".join(drop_flags(a, {"--seed", "--diversity"})))
            cells.append((al, d, sd))
    if len(cells) != 24 or len(cells) != len(set(cells)):
        problems.append(f"{len(cells)} jobs, {len(set(cells))} distinct cells (need 24, each once)")
    settings = {}
    for algo, v in variants.items():
        if len(v) != 1:
            problems.append(f"{algo}: {len(v)} different settings across jobs")
        settings[algo] = sorted(v)[0].split() if v else PROVISIONAL[algo] + V2_EXTRA[algo]
    lines.append(("jobs", "OK: 24 jobs (grant 6 SAC, chris 6 SAC, desmond_ppo 12 PPO), each = grid A's "
                          "line for the same cell + the v2 flags, tags v2_<ALGO>_d<N>_s<seed>"
                  if not problems else "PROBLEM: " + "; ".join(problems)))
    lines.append(("settings.source", "final_grid_v2/jobs_v2_{grant,chris,desmond_ppo}.txt"))
    return settings


def determinism(lines: list[tuple[str, str]]) -> None:
    import numpy as np
    train = load_module("train", REPO / "train.py")
    train.apply_reward_overrides(None)
    w = train._wrapper_module
    lines.append(("env.reward_used", f"file constants: PROGRESS_WEIGHT {w.PROGRESS_WEIGHT}, "
                                     f"TIME_COST {w.TIME_COST}, CRASH_PENALTY {w.CRASH_PENALTY}"))

    def action(t: int):
        # Fixed open-loop sequence: a gentle steering weave at 2.5-3.5 m/s. Open loop
        # always meets a wall at the first real bend; this lasts 250-650 steps.
        return np.array([0.10 * math.sin(2 * math.pi * t / 150.0),
                         -0.70 + 0.05 * math.cos(2 * math.pi * t / 500.0)], dtype=np.float32)

    for name in DETERMINISM_TRACKS:
        env = train.make_env([name], seed=0, track_seed=0, max_episode_steps=3000)()
        runs = []
        for _ in range(2):
            obs, _info = env.reset(seed=RESET_SEED)
            core = env.unwrapped
            h = hashlib.sha256(np.asarray(obs, dtype=np.float32).tobytes())
            h.update(np.array([core.poses_x[0], core.poses_y[0], core.poses_theta[0]]).tobytes())
            ret, end, info, t = 0.0, "cap", {}, 0
            for t in range(1, 3001):
                obs, r, term, trunc, info = env.step(action(t - 1))
                ret += r
                h.update(np.asarray(obs, dtype=np.float32).tobytes())
                h.update(np.float64(r).tobytes())
                if term or trunc:
                    end = ("crash" if info.get("collision") else
                           "lap" if info.get("lap_completed") else "cap")
                    break
            runs.append(f"steps {t}, end {end}, progress_m {info.get('progress_m', 0.0):.9f}, "
                        f"return {ret:.9f}, traj {h.hexdigest()[:16]}")
        env.close()
        lines.append((f"env.{name}", runs[0] if runs[0] == runs[1]
                      else f"NONDETERMINISTIC: {runs[0]} | {runs[1]}"))


def v2_smoke_checks(lines: list[tuple[str, str]], algo: str, cfg: dict, run_dir: pathlib.Path) -> None:
    """v2 assertions on a smoke run: effective reward, resolved hyperparameters, real crash payment."""
    import csv
    eff = cfg.get("effective_reward_constants") or {}
    ok = all(isinstance(eff.get(k), (int, float)) and abs(float(eff[k]) - v) < 1e-12
             for k, v in V2_EXPECT_REWARD.items())
    lines.append((f"smoke.{algo}.effective_reward",
                  "OK: CRASH_PENALTY 5.0, TIME_COST 0.0, PROGRESS_WEIGHT 1.0" if ok else f"FAIL: {eff}"))
    hp = cfg.get("hyperparameters") or {}
    bad = [f"{k}={hp.get(k)!r} (want {v})" for k, v in V2_EXPECT_HP[algo].items()
           if not isinstance(hp.get(k), (int, float)) or abs(float(hp[k]) - v) > 1e-12]
    lines.append((f"smoke.{algo}.hyperparameters",
                  "OK: " + ", ".join(f"{k} {hp.get(k)}" for k in V2_EXPECT_HP[algo]) if not bad
                  else "FAIL: " + "; ".join(bad)))
    pays = []
    mon = run_dir / "monitor_0.monitor.csv"
    if mon.exists():
        rows = list(csv.DictReader(mon.read_text(encoding="utf-8").splitlines()[1:]))  # line 1: JSON comment
        gaps = [float(r["progress_m"]) - float(r["r"]) for r in rows]
        pays = [g for g in gaps if g > 2.5]          # with time cost 0, only a crash opens a gap
    exact = sum(abs(p - 5.0) < 1e-6 for p in pays)
    lines.append((f"smoke.{algo}.crash_pays",
                  f"OK: all {len(pays)} crash episodes pay exactly -5.0" if pays and exact == len(pays)
                  else f"FAIL: {len(pays)} crash episodes, payments {sorted({round(p, 6) for p in pays})[:5]}"))


def smoke(lines: list[tuple[str, str]], settings: dict[str, list[str]]) -> None:
    for algo in ("SAC", "PPO"):
        tag = f"smokecheck_{algo}_{os.getpid()}"
        argv = list(settings[algo])
        for flag, value in (("--diversity", 1), ("--seed", 0), ("--total-timesteps", SMOKE_STEPS[algo]),
                            ("--checkpoint-freq", 1000), ("--run-tag", tag),
                            ("--torch-threads", 2 if algo == "SAC" else 1)):
            argv = set_flag(argv, flag, value)
        run_dir = REPO / "models" / f"{algo}_1tracks_s0_{tag}"
        log = REPO / "logs" / f"check_setup_smoke_{algo}.log"
        log.parent.mkdir(exist_ok=True)
        env = dict(os.environ, MPLBACKEND="Agg", PYTHONUNBUFFERED="1",
                   **{k: get_flag(argv, "--torch-threads")
                      for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
        flags = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
        t0 = time.time()
        try:
            with open(log, "w", encoding="utf-8") as fh:
                rc = subprocess.run([sys.executable, "-u", str(REPO / "train.py"), *argv], cwd=REPO,
                                    stdout=fh, stderr=subprocess.STDOUT, env=env,
                                    creationflags=flags, timeout=1800).returncode
            wall = time.time() - t0
            cfg = json.loads((run_dir / "run_config.json").read_text()) if (run_dir / "run_config.json").exists() else {}
            need = ["final_model.zip", "vecnormalize.pkl", "progress.csv", "monitor_0.monitor.csv",
                    f"{algo}_checkpoint_1000_steps.zip", f"{algo}_checkpoint_vecnormalize_1000_steps.pkl"]
            missing = [f for f in need if not (run_dir / f).exists()]
            ok = rc == 0 and cfg.get("status") == "completed" and not missing
            if ok:
                lines.append((f"smoke.{algo}", f"OK ({SMOKE_STEPS[algo]:,}-step run, checkpoint, VecNormalize "
                                               "and final model written; run folder deleted)"))
                lines.append((f"info.smoke.{algo}", f"{cfg['num_timesteps']:,} steps in {wall:.0f} s, "
                                                    f"{cfg['steps_per_second']:.0f} steps/s"))
            else:
                lines.append((f"smoke.{algo}", f"FAIL (exit {rc}, status {cfg.get('status')}, "
                                               f"missing {missing}); see logs/{log.name}"))
            if cfg:
                lines.append((f"settings.{algo}.args", " ".join(drop_flags(
                    settings[algo], {"--diversity", "--seed"}))))
                lines.append((f"settings.{algo}.hyperparameters",
                              json.dumps(cfg.get("hyperparameters"), separators=(",", ":"))))
                lines.append((f"settings.{algo}.reward",
                              json.dumps(cfg.get("effective_reward_constants"), separators=(",", ":"))))
                v2_smoke_checks(lines, algo, cfg, run_dir)
        finally:
            if run_dir.exists():
                shutil.rmtree(run_dir)


def machine(lines: list[tuple[str, str]]) -> None:
    cpu = platform.processor()
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        cpu = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:
        pass
    ram = ""
    try:
        import ctypes

        class MEM(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MEM()
        m.dwLength = ctypes.sizeof(MEM)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        ram = f", RAM {m.ullTotalPhys / 2 ** 30:.0f} GB"
    except Exception:
        pass
    lines.append(("info.machine", f"{cpu}, {os.cpu_count()} logical CPUs{ram}"))
    lines.append(("info.disk_free_gb", f"{shutil.disk_usage(REPO).free / 2 ** 30:.0f} (drive of {REPO.anchor})"))
    try:
        q = subprocess.run(["powercfg", "/query", "SCHEME_CURRENT", "SUB_SLEEP", "STANDBYIDLE"],
                           capture_output=True, text=True).stdout
        ac = re.search(r"AC Power Setting Index:\s*0x([0-9a-fA-F]+)", q)
        secs = int(ac.group(1), 16) if ac else None
        lines.append(("info.sleep_on_ac", "never" if secs == 0 else
                      f"after {secs // 60} min: TURN OFF before training" if secs else "unknown"))
    except Exception:
        pass
    lines.append(("info.generated", dt.datetime.now().strftime("%Y-%m-%d %H:%M")))


# ------------------------------------------------------------------ output

SECTIONS = [("git.", "code"), ("f1tenth_gym.", "code"), ("sha256", "code"), ("info.git", "code"),
            ("info.sha256", "code"), ("python", "python"), ("pkg.", "python"), ("pip.", "python"),
            ("tracks.", "tracks"), ("jobs", "settings"), ("settings.", "settings"),
            ("env.", "determinism"), ("smoke.", "smoke test"), ("info.smoke", "smoke test"),
            ("info.", "machine")]
ORDER = ["code", "python", "tracks", "settings", "determinism", "smoke test", "machine"]


def section_of(key: str) -> str:
    return next(s for p, s in SECTIONS if key.startswith(p))


def render(lines: list[tuple[str, str]]) -> str:
    out = ["# F1TENTH final grid v2: setup fingerprint (final_grid_v2/check_setup.py, format 1)",
           "# Keys starting with 'info.' may differ between machines; every other line must match.",
           "# Check against the reference: python final_grid_v2/check_setup.py --compare <this file>"]
    last = None
    for k, v in sorted(lines, key=lambda kv: ORDER.index(section_of(kv[0]))):   # stable
        sec = section_of(k)
        if sec != last:
            out.append(f"[{sec}]")
            last = sec
        out.append(f"{k} = {v}")
    return "\n".join(out) + "\n"


def read_fp(path: pathlib.Path) -> dict[str, str]:
    kv = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        if " = " in raw and not raw.startswith(("#", "[")):
            k, v = raw.split(" = ", 1)
            kv[k.strip()] = v.strip()
    return kv


def compare(theirs: pathlib.Path, reference: pathlib.Path) -> int:
    ref, oth = read_fp(reference), read_fp(theirs)
    keys = list(ref) + [k for k in oth if k not in ref]
    hard = [k for k in keys if not k.startswith("info.") and ref.get(k) != oth.get(k)]
    soft = [k for k in keys if k.startswith("info.") and ref.get(k) != oth.get(k)]
    print(f"reference: {reference}\ncompared:  {theirs}\n")
    if ref.get("settings.source", "").startswith("PROVISIONAL"):
        print("NOTE: the reference was made without the final job files (provisional settings).\n")
    if not hard:
        print(f"MATCH ({sum(not k.startswith('info.') for k in keys)} checked lines identical)")
    else:
        print(f"DIFFERENT: {len(hard)} line(s) must match but do not\n")
        for k in hard:
            print(f"  {k}\n    reference: {ref.get(k, '(missing)')}\n    theirs:    {oth.get(k, '(missing)')}")
    if soft:
        print("\nAllowed to differ (for information):")
        for k in soft:
            print(f"  {k}: reference {ref.get(k, '-')} | theirs {oth.get(k, '-')}")
    return 0 if not hard else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=pathlib.Path, help="Also write the fingerprint to this file.")
    ap.add_argument("--compare", type=pathlib.Path, metavar="FINGERPRINT",
                    help="Diff this fingerprint against --reference instead of checking the machine.")
    ap.add_argument("--reference", type=pathlib.Path, default=REFERENCE)
    ap.add_argument("--skip-smoke", action="store_true",
                    help="Skip the two smoke runs (quick check; such a fingerprint never MATCHes).")
    a = ap.parse_args()
    if a.compare:
        return compare(a.compare, a.reference)

    os.chdir(REPO)
    lines: list[tuple[str, str]] = []
    steps = [("code", lambda: check_code(lines)), ("python", lambda: check_python(lines)),
             ("tracks", lambda: check_tracks(lines))]
    for label, fn in steps:
        print(f"checking {label} ...", file=sys.stderr, flush=True)
        fn()
    settings = final_settings(lines)
    print("checking environment determinism ...", file=sys.stderr, flush=True)
    determinism(lines)
    if a.skip_smoke:
        lines += [("smoke.SAC", "SKIPPED"), ("smoke.PPO", "SKIPPED")]
    else:
        print("running 2,000-step SAC and 8,192-step PPO smoke tests (a few minutes) ...",
              file=sys.stderr, flush=True)
        smoke(lines, settings)
    machine(lines)
    text = render(lines)
    print(text, end="")
    if a.out:
        a.out.write_text(text, encoding="utf-8", newline="\n")
        print(f"\nwritten {a.out}", file=sys.stderr)
    bad = [k for k, v in lines if not k.startswith("info.")
           and re.search(r"\b(FAIL|DIFFERS|NOT|NONDETERMINISTIC|PROBLEM|MODIFIED|ELSEWHERE|MISSING|"
                         r"SKIPPED|differs)\b", v)]
    if bad:
        print(f"\nPROBLEMS on this machine: {', '.join(bad)}", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
