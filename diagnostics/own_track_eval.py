"""Deterministic lap completion on the TRAINING track (synthetic_track_0), 30 spawns,
for each gamma-0.999 PPO seed at 1.0M / 1.6M / 2.0M. Team evaluate.py, team protocol."""
import shutil, subprocess, pathlib
ROOT = pathlib.Path(r"E:/OneDrive - The University of Auckland/Desktop/COMPSCI 760")
PY = ROOT / "spike/venv/Scripts/python.exe"; REPO = ROOT / "team_repo_heldout"
OUT = ROOT / "_verify/own_track"; OUT.mkdir(parents=True, exist_ok=True)
RUNS = {0: "PPO_1tracks_s0_G999_gamma", 1: "PPO_1tracks_s1_G999_s1", 2: "PPO_1tracks_s2_G999_s2"}
for step in [1000000, 1600000, 2000000]:
    for seed, tag in RUNS.items():
        csv = OUT / f"s{seed}_ck{step}.csv"
        if csv.exists(): continue
        run = REPO / "models" / tag; tmp = ROOT / f"_verify/_tmp_own_{seed}"
        shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir()
        shutil.copy(run / f"PPO_checkpoint_{step}_steps.zip", tmp / "final_model.zip")
        shutil.copy(run / f"PPO_checkpoint_vecnormalize_{step}_steps.pkl", tmp / "vecnormalize.pkl")
        r = subprocess.run([str(PY), "-u", "evaluate.py", "--algo", "PPO", "--model-path", str(tmp),
            "--tracks", "synthetic_track_0", "--episodes", "30", "--eval-seed", "0", "--target-laps", "1",
            "--max-steps", "15000", "--reset-type", "cl_grid_static", "--no-plot", "--out", str(csv)],
            cwd=str(REPO), capture_output=True, text=True)
        print(f"seed {seed} @ {step:,}: {'ok' if csv.exists() else 'FAILED ' + r.stderr[-300:]}", flush=True)
print("done")
