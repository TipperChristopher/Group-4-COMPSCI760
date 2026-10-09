"""Mirror of Desmond's launch_queue.ps1 evaluation for the S1-with-penalty-5 runs:
evaluate.py --track-set vw_val --episodes 5 --max-steps 15000 --no-plot [--checkpoint N], 1 thread,
at 400k / 800k / 1.2M / 1.6M and final. Output names match his: <tag>_<label>_vw_val.csv."""
import os, subprocess, time, json, pathlib
ROOT = pathlib.Path(r"E:/OneDrive - The University of Auckland/Desktop/COMPSCI 760")
PY = ROOT / "spike/venv/Scripts/python.exe"; WT = ROOT / "team_repo_tuning"; OUT = ROOT / "_verify/s1p5/results"
TAGS = {99: "tune_S1P5_s99", 98: "tune_S1P5_s98"}
CKS = [(400000, "400k"), (800000, "800k"), (1200000, "1200k"), (1600000, "1600k"), (None, "final")]
env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
t0 = time.time()
while time.time() - t0 < 40 * 3600:
    pending = 0
    for seed, tag in TAGS.items():
        rd = WT / "models" / f"SAC_20tracks_s{seed}_{tag}"
        for steps, label in CKS:
            out = OUT / f"{tag}_{label}_vw_val.csv"
            if out.exists(): continue
            pending += 1
            if steps:
                z, v = rd / f"SAC_checkpoint_{steps}_steps.zip", rd / f"SAC_checkpoint_vecnormalize_{steps}_steps.pkl"
                ready = z.exists() and v.exists() and time.time() - z.stat().st_mtime > 60
            else:
                rc = rd / "run_config.json"
                ready = (rd / "final_model.zip").exists() and rc.exists() and json.load(open(rc)).get("status") == "completed"
            if not ready: continue
            argv = [str(PY), "-u", "evaluate.py", "--algo", "SAC", "--diversity", "20", "--seed", str(seed), "--run-tag", tag,
                    "--track-set", "vw_val", "--episodes", "5", "--max-steps", "15000", "--no-plot", "--out", str(out)]
            if steps: argv += ["--checkpoint", str(steps)]
            r = subprocess.run(argv, cwd=str(WT), env=env, capture_output=True, text=True)
            print(time.strftime("%m-%d %H:%M"), tag, label, "ok" if out.exists() else "FAILED " + r.stderr[-300:], flush=True)
    if pending == 0: print("all evaluations done", flush=True); break
    time.sleep(300)
