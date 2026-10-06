"""Which checkpoint of the gamma=0.999 run is actually best?

The 800k checkpoint scored 40% lap completion on held-out tracks; the final 2M
checkpoint scores 0%. That is the same checkpoint-selection trap found in
Desmond's grid (and in our own SAC, where 1.2M beat 2M). Measure the whole curve.
"""
import os, shutil, subprocess, pathlib
ROOT = pathlib.Path(r"E:/OneDrive - The University of Auckland/Desktop/COMPSCI 760")
PY = ROOT / "spike/venv/Scripts/python.exe"; REPO = ROOT / "team_repo_heldout"
OUT = ROOT / "_verify/g999_curve"; OUT.mkdir(parents=True, exist_ok=True)
TMP = ROOT / "_verify/_tmp_curve"
RUN = REPO / "models/PPO_1tracks_s0_G999_gamma"
TRACKS = ["test_track_0", "test_track_1", "test_track_2", "test_track_3", "test_track_4",
          "Spielberg", "Silverstone"]
for step in [400000, 600000, 800000, 1000000, 1200000, 1400000, 1600000, 1800000, 2000000]:
    csv = OUT / f"ck{step}.csv"
    if csv.exists(): continue
    m = RUN / f"PPO_checkpoint_{step}_steps.zip"; v = RUN / f"PPO_checkpoint_vecnormalize_{step}_steps.pkl"
    if not (m.exists() and v.exists()):
        print(f"  {step:>9,} missing"); continue
    shutil.rmtree(TMP, ignore_errors=True); TMP.mkdir(parents=True)
    shutil.copy(m, TMP / "final_model.zip"); shutil.copy(v, TMP / "vecnormalize.pkl")
    subprocess.run([str(PY), "-u", "evaluate.py", "--algo", "PPO", "--model-path", str(TMP),
                    "--tracks", *TRACKS, "--episodes", "5", "--eval-seed", "0",
                    "--target-laps", "1", "--max-steps", "15000",
                    "--reset-type", "cl_grid_static", "--no-plot", "--out", str(csv)],
                   cwd=str(REPO), capture_output=True, text=True)
    print(f"  {step:>9,} {'ok' if csv.exists() else 'FAILED'}", flush=True)
print("done")
