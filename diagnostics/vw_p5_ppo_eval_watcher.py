"""PPO sigma-fix test on vw d20, penalty 5. Mirrors Desmond's launch_queue evaluation
(evaluate.py --track-set vw_val --episodes 5 --max-steps 15000 --no-plot --checkpoint N, 1 thread)
at his checkpoints (400k/800k/1.2M/1.6M/2.0M) PLUS 1.7/1.8/1.9M so the D14 last-5 mean can be computed."""
import os, subprocess, time, pathlib
ROOT = pathlib.Path(r"E:/OneDrive - The University of Auckland/Desktop/COMPSCI 760")
PY = ROOT / "spike/venv/Scripts/python.exe"; WT = ROOT / "team_repo_tuning"; OUT = ROOT / "_verify/ppo_vw/results"
TAGS = [(99, "tune_P1P5_s99"), (98, "tune_P1P5_s98"), (99, "tune_P1P5ent_s99"), (98, "tune_P1P5ent_s98")]
CKS = [400000, 800000, 1200000, 1600000, 1700000, 1800000, 1900000, 2000000]
env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
t0 = time.time()
while time.time() - t0 < 12 * 3600:
    pending = 0
    for seed, tag in TAGS:
        rd = WT / "models" / f"PPO_20tracks_s{seed}_{tag}"
        for steps in CKS:
            out = OUT / f"{tag}_{steps // 1000}k_vw_val.csv"
            if out.exists(): continue
            pending += 1
            z, v = rd / f"PPO_checkpoint_{steps}_steps.zip", rd / f"PPO_checkpoint_vecnormalize_{steps}_steps.pkl"
            if not (z.exists() and v.exists() and time.time() - z.stat().st_mtime > 60): continue
            argv = [str(PY), "-u", "evaluate.py", "--algo", "PPO", "--diversity", "20", "--seed", str(seed), "--run-tag", tag,
                    "--track-set", "vw_val", "--episodes", "5", "--max-steps", "15000", "--no-plot", "--out", str(out),
                    "--checkpoint", str(steps)]
            r = subprocess.run(argv, cwd=str(WT), env=env, capture_output=True, text=True)
            print(time.strftime("%m-%d %H:%M"), tag, steps, "ok" if out.exists() else "FAILED " + r.stderr[-300:], flush=True)
    if pending == 0: print("all evaluations done", flush=True); break
    time.sleep(240)
