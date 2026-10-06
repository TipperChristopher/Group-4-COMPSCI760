"""Fair comparison: OUR penalty-40 SAC vs DESMOND's penalty-5 SAC, both 2M / 1 training track.
Plus a PPO control at the SAME penalty (5) to separate 'penalty' from 'training code path'.

Tracks chosen so they are UNSEEN for every model:
  synthetic_track_1..5 -> our models trained only on OUR synthetic_track_0;
                          Desmond's maps are entirely different (verified: 0/20 match).
  Spielberg/Silverstone -> real, maps verified IDENTICAL, unseen by all.
"""
import os, subprocess
import pandas as pd
ROOT=r"E:/OneDrive - The University of Auckland/Desktop/COMPSCI 760"
PY=f"{ROOT}/spike/venv/Scripts/python.exe"; OUT=f"{ROOT}/_verify/repro"
OURS=f"{ROOT}/team_repo/feasibility_spike/reward_experiment/saved_models"
HIS=f"{ROOT}/_verify/Group4_DiversityGrid_seed0/5_trained_models"
TRACKS=["synthetic_track_1","synthetic_track_2","synthetic_track_3","synthetic_track_4",
        "synthetic_track_5","Spielberg","Silverstone"]
RUNS=[("OURS_SAC_p40","SAC",f"{OURS}/sac_overfit"),
      ("HIS_SAC_p5","SAC", f"{HIS}/SAC_1tracks_s0"),
      ("OURS_PPO_p5","PPO",f"{OURS}/ppo_new_p5"),
      ("HIS_PPO_p5","PPO", f"{HIS}/PPO_1tracks_s0")]
rows=[]
for tag,algo,path in RUNS:
    csv=f"{OUT}/cmp_{tag}.csv"
    if not os.path.exists(csv):
        r=subprocess.run([PY,"-u","evaluate.py","--algo",algo,"--model-path",path,
            "--tracks",*TRACKS,"--episodes","10","--eval-seed","0","--target-laps","1",
            "--max-steps","15000","--reset-type","cl_grid_static","--no-plot","--out",csv],
            cwd=f"{ROOT}/team_repo",capture_output=True,text=True)
        if not os.path.exists(csv):
            print("FAIL",tag,r.stdout[-500:],r.stderr[-500:]); continue
    d=pd.read_csv(csv); d["tag"]=tag; rows.append(d)
    syn=d[d.track.str.startswith("synthetic")]; real=d[~d.track.str.startswith("synthetic")]
    print(f"{tag:14s} | SYNTH laps {syn.laps.mean():5.2f}  finished {int(syn.completed_lap.sum()):>2}/{len(syn):<2} "
          f"| REAL laps {real.laps.mean():5.2f} dist {real.progress_m.mean():6.1f}m finished {int(real.completed_lap.sum())}/{len(real)}",flush=True)
pd.concat(rows).to_csv(f"{OUT}/_penalty_compare_all.csv",index=False)
print("\n--- per-track mean laps ---")
a=pd.concat(rows)
print(a.pivot_table(index="track",columns="tag",values="laps",aggfunc="mean").round(2).to_string())
