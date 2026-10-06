"""Can each SAC cell lap tracks from its OWN training pool?
synthetic_track_0..4 are: d1 -> only track_0 is training (1-4 held out)
                          d5/d20/d100 -> all five are TRAINING tracks."""
import os, shutil, subprocess
import pandas as pd
ROOT=r"E:/OneDrive - The University of Auckland/Desktop/COMPSCI 760"
B=f"{ROOT}/_verify/Group4_DiversityGrid_seed0/5_trained_models"
PY=f"{ROOT}/spike/venv/Scripts/python.exe"; OUT=f"{ROOT}/_verify/repro"
TRACKS=["synthetic_track_0","synthetic_track_1","synthetic_track_2","synthetic_track_3","synthetic_track_4"]
res=[]
for div in [1,5,20,100]:
    csv=f"{OUT}/SACd{div}_trainpool.csv"
    if not os.path.exists(csv):
        subprocess.run([PY,"-u","evaluate.py","--algo","SAC","--model-path",f"{B}/SAC_{div}tracks_s0",
            "--tracks",*TRACKS,"--episodes","5","--eval-seed","0","--target-laps","1",
            "--max-steps","15000","--reset-type","cl_grid_static","--no-plot","--out",csv],
            cwd=f"{ROOT}/team_repo",capture_output=True,text=True)
    if not os.path.exists(csv): print("FAIL d",div); continue
    d=pd.read_csv(csv)
    pt=d.groupby("track").agg(laps=("laps","mean"),m=("progress_m","mean"),fin=("completed_lap","sum"))
    print(f"\n### SAC d{div}  (tracks 0-4 are {'TRAINING' if div>=5 else 'training=only track_0'})")
    print(pt.round(2).to_string())
    print(f"  -> mean laps {d.laps.mean():.2f}   episodes finished {int(d.completed_lap.sum())}/25   tracks lapped {int((pt.fin>0).sum())}/5")
    res.append(dict(div=div,mean_laps=d.laps.mean(),fin=int(d.completed_lap.sum()),tracks=int((pt.fin>0).sum())))
print("\n================ SUMMARY: own-training-pool performance ================")
print(pd.DataFrame(res).to_string(index=False))
