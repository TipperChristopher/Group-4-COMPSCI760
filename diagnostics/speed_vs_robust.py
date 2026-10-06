"""Does getting FASTER make it less robust to the spawn? Same 30 spawns, 3 checkpoints."""
import os, shutil, subprocess
import pandas as pd
ROOT=r"E:/OneDrive - The University of Auckland/Desktop/COMPSCI 760"
SRC=f"{ROOT}/team_repo/feasibility_spike/reward_experiment/saved_models/sac_overfit"
PY=f"{ROOT}/spike/venv/Scripts/python.exe"; OUT=f"{ROOT}/_verify/repro"; TMP=f"{ROOT}/_verify/_tmpspd"
LAPT={1000000:24.7,1200000:22.7,1500000:21.7,1800000:None,2000000:20.8}
for ck in [1000000,1200000,1500000,1800000,2000000]:
    csv=f"{OUT}/spd_ck{ck}.csv"
    if not os.path.exists(csv):
        shutil.rmtree(TMP,ignore_errors=True); os.makedirs(TMP)
        shutil.copy(f"{SRC}/ckpt_{ck}.zip",f"{TMP}/final_model.zip")
        shutil.copy(f"{SRC}/vecnormalize_{ck}.pkl",f"{TMP}/vecnormalize.pkl")
        subprocess.run([PY,"-u","evaluate.py","--algo","SAC","--model-path",TMP,
            "--tracks","synthetic_track_0","--episodes","30","--eval-seed","0","--target-laps","1",
            "--max-steps","15000","--reset-type","cl_grid_static","--no-plot","--out",csv],
            cwd=f"{ROOT}/team_repo",capture_output=True,text=True)
    if not os.path.exists(csv): print("FAIL",ck); continue
    d=pd.read_csv(csv)
    fin=d[d.outcome=="finished"]
    # distinct spawn poses -> distinct progress values among finishers is ~1; use step count for speed
    lt=fin.length.mean()*0.01 if len(fin) else float("nan")
    npose=d.groupby(d.progress_m.round(2)).ngroups
    print(f"  ckpt {ck:>9,}  in-train laptime {str(LAPT[ck]):>5}s  |  finished {len(fin):>2}/30 "
          f"({len(fin)/30*100:3.0f}%)  mean lap time of finishers {lt:5.1f}s  distinct outcomes {npose}")
