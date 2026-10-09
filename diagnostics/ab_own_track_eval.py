"""D14 protocol: deterministic lap completion on the TRAINING track (synthetic_track_0), 30 episodes,
at the last 5 checkpoints (1.6-2.0M), for baseline G999 (3 seeds) and the A/B arms ENT001 / TC0015 (3 seeds each).
Team evaluate.py, same flags as own_track_eval.py. Reuses existing baseline CSVs from _verify/own_track/ where present."""
import shutil, subprocess, pathlib
ROOT = pathlib.Path(r"E:/OneDrive - The University of Auckland/Desktop/COMPSCI 760")
PY = ROOT / "spike/venv/Scripts/python.exe"; REPO = ROOT / "team_repo_heldout"
OUT = ROOT / "_verify/ab_own_track"; OUT.mkdir(parents=True, exist_ok=True)
OLD = ROOT / "_verify/own_track"
RUNS = {}
for s, tag in {0: "PPO_1tracks_s0_G999_gamma", 1: "PPO_1tracks_s1_G999_s1", 2: "PPO_1tracks_s2_G999_s2"}.items():
    RUNS[f"base_s{s}"] = tag
for arm in ("ENT001", "TC0015"):
    for s in (0, 1, 2):
        RUNS[f"{arm}_s{s}"] = f"PPO_1tracks_s{s}_{arm}"
for step in [2000000, 1600000, 1800000, 1700000, 1900000]:
    for name, tag in RUNS.items():
        csv = OUT / f"{name}_ck{step}.csv"
        if csv.exists(): continue
        if name.startswith("base_"):
            old = OLD / f"s{name[-1]}_ck{step}.csv"
            if old.exists():
                shutil.copy(old, csv); print(f"{name} @ {step:,}: reused", flush=True); continue
        run = REPO / "models" / tag; tmp = ROOT / "_verify/_tmp_ab"
        shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir()
        shutil.copy(run / f"PPO_checkpoint_{step}_steps.zip", tmp / "final_model.zip")
        shutil.copy(run / f"PPO_checkpoint_vecnormalize_{step}_steps.pkl", tmp / "vecnormalize.pkl")
        r = subprocess.run([str(PY), "-u", "evaluate.py", "--algo", "PPO", "--model-path", str(tmp),
            "--tracks", "synthetic_track_0", "--episodes", "30", "--eval-seed", "0", "--target-laps", "1",
            "--max-steps", "15000", "--reset-type", "cl_grid_static", "--no-plot", "--out", str(csv)],
            cwd=str(REPO), capture_output=True, text=True)
        print(f"{name} @ {step:,}: {'ok' if csv.exists() else 'FAILED ' + r.stderr[-300:]}", flush=True)
print("done")
