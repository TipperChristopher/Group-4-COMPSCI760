"""Evaluate every PPO sweep variant at a MATCHED 1.0M-step checkpoint.

Training-log metrics (std / approx_kl / clip_fraction) say whether the
optimisation is healthy.  They are not performance.  This runs the real
evaluation protocol on tracks the models never trained on, so the claims about
"did it help" rest on measured behaviour rather than on internals.

  held-out synthetic : test_track_0..4   (canonical seed-123 val/test split)
  real circuits      : Spielberg, Silverstone

Matched budget matters: comparing each variant at its own latest checkpoint
would confound "better" with "trained longer".
"""
import os
import shutil
import subprocess
import sys
import pathlib

ROOT = pathlib.Path(r"E:/OneDrive - The University of Auckland/Desktop/COMPSCI 760")
REPO = ROOT / "team_repo_heldout"
PY = ROOT / "spike/venv/Scripts/python.exe"
OUT = ROOT / "_verify/ppo_sweep_eval"
TMP = ROOT / "_verify/_tmp_sweep_eval"
STEP = 1000000
TRACKS = ["test_track_0", "test_track_1", "test_track_2", "test_track_3",
          "test_track_4", "Spielberg", "Silverstone"]
EPISODES = 5

OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    runs = sorted(p for p in (REPO / "models").glob("PPO_1tracks_s0_*") if p.is_dir())
    print(f"{len(runs)} variants, checkpoint {STEP:,}, {len(TRACKS)} tracks, "
          f"{EPISODES} episodes each\n")

    for run in runs:
        tag = run.name.replace("PPO_1tracks_s0_", "")
        csv = OUT / f"{tag}.csv"
        if csv.exists():
            print(f"  {tag:>13} cached")
            continue
        models = run / f"PPO_checkpoint_{STEP}_steps.zip"
        stats = run / f"PPO_checkpoint_vecnormalize_{STEP}_steps.pkl"
        if not models.exists() or not stats.exists():
            print(f"  {tag:>13} MISSING checkpoint or stats -> skipped")
            continue

        shutil.rmtree(TMP, ignore_errors=True)
        TMP.mkdir(parents=True)
        shutil.copy(models, TMP / "final_model.zip")
        shutil.copy(stats, TMP / "vecnormalize.pkl")

        r = subprocess.run(
            [str(PY), "-u", "evaluate.py", "--algo", "PPO", "--model-path", str(TMP),
             "--tracks", *TRACKS, "--episodes", str(EPISODES), "--eval-seed", "0",
             "--target-laps", "1", "--max-steps", "15000",
             "--reset-type", "cl_grid_static", "--no-plot", "--out", str(csv)],
            cwd=str(REPO), capture_output=True, text=True)
        print(f"  {tag:>13} {'ok' if csv.exists() else 'FAILED'}"
              + ("" if csv.exists() else f"  {r.stderr[-200:]}"), flush=True)

    print("\nDone. Compare with: python _verify/ppo_sweep_eval_report.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
