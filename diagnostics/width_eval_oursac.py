"""The width ablation on a policy that actually drives.

Desmond's narrowA/narrowB keep the EXACT centreline of test_track_k and only move
the walls (half-width 1.473 -> 1.072 -> 0.751 m). His penalty-5 SAC cells drop to
0% laps on narrowA. This asks the same question of OUR penalty-40 SAC, which
completes 73% of spawns on its own training track and 58% on unseen synthetic.

If width is the cause of the real-circuit failure, a working policy should fall
off the same cliff as the walls narrow -- and its distance on narrowA should
approach its (near-zero) distance on real circuits, whose median half-width is
1.075 m.
"""
import os, shutil, subprocess, pathlib
ROOT = pathlib.Path(r"E:/OneDrive - The University of Auckland/Desktop/COMPSCI 760")
PY = ROOT / "spike/venv/Scripts/python.exe"
REPO = ROOT / "team_repo_heldout"
SRC = ROOT / "team_repo/feasibility_spike/reward_experiment/saved_models/sac_overfit"
OUT = ROOT / "_verify/width_eval"; OUT.mkdir(parents=True, exist_ok=True)
TMP = ROOT / "_verify/_tmp_width"

GROUPS = {"A-test-1.47m": ["test_track_%d" % i for i in range(5)],
          "B-narrowA-1.07m": ["narrowA_track_%d" % i for i in range(5)],
          "C-narrowB-0.75m": ["narrowB_track_%d" % i for i in range(5)],
          "D-real": ["Spielberg", "Silverstone"]}

# our SAC at its most reliable checkpoint (1.2M completed 30/30 spawns) and the final
CKPTS = {1200000: ("ckpt_1200000.zip", "vecnormalize_1200000.pkl"),
         2000000: ("final_model.zip", "vecnormalize.pkl")}

for step, (mz, vz) in CKPTS.items():
    for gname, tracks in GROUPS.items():
        csv = OUT / f"oursac{step//1000}k_{gname}.csv"
        if csv.exists():
            print(f"  {step//1000}k {gname:>18} cached"); continue
        shutil.rmtree(TMP, ignore_errors=True); TMP.mkdir(parents=True)
        shutil.copy(SRC / mz, TMP / "final_model.zip")
        shutil.copy(SRC / vz, TMP / "vecnormalize.pkl")
        subprocess.run([str(PY), "-u", "evaluate.py", "--algo", "SAC",
                        "--model-path", str(TMP), "--tracks", *tracks,
                        "--episodes", "10", "--eval-seed", "0", "--target-laps", "1",
                        "--max-steps", "15000", "--reset-type", "cl_grid_static",
                        "--no-plot", "--out", str(csv)],
                       cwd=str(REPO), capture_output=True, text=True)
        print(f"  {step//1000}k {gname:>18} {'ok' if csv.exists() else 'FAILED'}", flush=True)
print("\ndone")
