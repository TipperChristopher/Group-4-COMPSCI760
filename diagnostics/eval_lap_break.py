"""Definitive check: does the gamma=0.999 PPO really complete laps?

Training-log `laps` comes from the wrapper's progress tracker. Verify it with
the standard evaluation protocol on tracks the models never trained on.
Matched 800k checkpoint so all four are compared at equal budget.
"""
import os, shutil, subprocess, pathlib
ROOT = pathlib.Path(r"E:/OneDrive - The University of Auckland/Desktop/COMPSCI 760")
PY = ROOT / "spike/venv/Scripts/python.exe"; REPO = ROOT / "team_repo_heldout"
OUT = ROOT / "_verify/lap_break_final"; OUT.mkdir(parents=True, exist_ok=True)
TMP = ROOT / "_verify/_tmp_lb"
STEP = 2000000
TRACKS = ["test_track_0", "test_track_1", "test_track_2", "test_track_3", "test_track_4",
          "Spielberg", "Silverstone"]
for tag in ["G999_gamma", "G999L99_full", "P40_penalty40", "V4_nsteps"]:
    run = REPO / "models" / f"PPO_1tracks_s0_{tag}"
    csv = OUT / f"{tag}_ck{STEP//1000}k.csv"
    if csv.exists(): print(f"  {tag:>15} cached"); continue
    m = run / f"PPO_checkpoint_{STEP}_steps.zip"
    v = run / f"PPO_checkpoint_vecnormalize_{STEP}_steps.pkl"
    if not (m.exists() and v.exists()):
        print(f"  {tag:>15} missing ckpt {STEP}"); continue
    shutil.rmtree(TMP, ignore_errors=True); TMP.mkdir(parents=True)
    shutil.copy(m, TMP / "final_model.zip"); shutil.copy(v, TMP / "vecnormalize.pkl")
    subprocess.run([str(PY), "-u", "evaluate.py", "--algo", "PPO", "--model-path", str(TMP),
                    "--tracks", *TRACKS, "--episodes", "5", "--eval-seed", "0",
                    "--target-laps", "1", "--max-steps", "15000",
                    "--reset-type", "cl_grid_static", "--no-plot", "--out", str(csv)],
                   cwd=str(REPO), capture_output=True, text=True)
    print(f"  {tag:>15} {'ok' if csv.exists() else 'FAILED'}", flush=True)
print("done")
