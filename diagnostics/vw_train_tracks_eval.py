"""Deterministic eval on each run's OWN 20 training tracks (vw_synthetic_track_0..19), same flags as the vw_val protocol."""
import os, subprocess, pathlib, time
ROOT = pathlib.Path(r"E:/OneDrive - The University of Auckland/Desktop/COMPSCI 760")
PY = ROOT / "spike/venv/Scripts/python.exe"; WT = ROOT / "team_repo_tuning"; OUT = ROOT / "_verify/train_tracks_eval"
TRACKS = [f"vw_synthetic_track_{i}" for i in range(20)]
JOBS = [("PPO", 99, "tune_P1P5_s99", 2000000), ("PPO", 98, "tune_P1P5_s98", 2000000),
        ("PPO", 99, "tune_P1P5ent_s99", 2000000), ("PPO", 98, "tune_P1P5ent_s98", 2000000),
        ("SAC", 99, "tune_S1P5_s99", 1200000), ("SAC", 98, "tune_S1P5_s98", 1200000)]
env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
procs = []
for algo, seed, tag, ck in JOBS:
    out = OUT / f"{tag}_{ck // 1000}k_train20.csv"
    argv = [str(PY), "-u", "evaluate.py", "--algo", algo, "--diversity", "20", "--seed", str(seed), "--run-tag", tag,
            "--tracks", *TRACKS, "--episodes", "5", "--max-steps", "15000", "--no-plot", "--checkpoint", str(ck), "--out", str(out)]
    procs.append((tag, out, subprocess.Popen(argv, cwd=str(WT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)))
for tag, out, p in procs:
    _, err = p.communicate()
    print(time.strftime("%H:%M"), tag, "ok" if out.exists() else "FAILED " + err[-300:], flush=True)
