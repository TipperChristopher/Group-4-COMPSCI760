"""What does the spawn actually change, and does it predict failure?"""
import warnings; warnings.filterwarnings("ignore")
import sys, types, os
if "gym" not in sys.modules:
    sys.modules["gym"]=types.ModuleType("gym"); sys.modules["gym"].__version__="0.0.0"
import numpy as np, pandas as pd, gymnasium as gym, f1tenth_gym
ROOT=r"E:/OneDrive - The University of Auckland/Desktop/COMPSCI 760"
res=pd.read_csv(f"{ROOT}/_verify/repro/OURsac_own30.csv")
env=gym.make("f1tenth_gym:f1tenth-v0",config={"num_agents":1,"timestep":0.01,
    "map":"synthetic_track_0","reset_config":{"type":"cl_grid_static"}})
core=env.unwrapped
cl=np.asarray(core.track.centerline.spline.points,dtype=float)
rows=[]
for s in sorted(res.eval_seed.unique()):
    obs,info=env.reset(seed=int(s))
    x,y,th=float(core.poses_x[0]),float(core.poses_y[0]),float(core.poses_theta[0])
    d=np.hypot(cl[:,0]-x,cl[:,1]-y); idx=int(np.argmin(d))
    scan=np.asarray(obs["scans"][0]); 
    r=res[res.eval_seed==s].iloc[0]
    rows.append(dict(seed=int(s),idx=idx,x=round(x,3),y=round(y,3),theta=round(th,4),
        min_lidar=round(float(scan.min()),3),
        fwd_lidar=round(float(scan[len(scan)//2]),2),
        outcome=r.outcome,progress_m=r.progress_m,length=r.length))
env.close()
d=pd.DataFrame(rows)
print("=== spawn pose per eval seed (synthetic_track_0, cl_grid_static) ===")
print(d.to_string(index=False))
print()
print(f"distinct spawn indices: {sorted(d.idx.unique())}  ({d.idx.nunique()} distinct over {len(d)} seeds)")
print(f"distinct (x,y) poses  : {d.groupby(['x','y']).ngroups}")
print()
print("=== outcome by spawn index ===")
g=d.groupby("idx").agg(n=("seed","size"),finished=("outcome",lambda s:(s=="finished").sum()),
                       mean_progress=("progress_m","mean"))
g["rate"]=(g.finished/g.n*100).round(0)
print(g.to_string())
print()
print("=== where do the failures end up? (crash progress_m) ===")
cr=d[d.outcome!="finished"]
print(sorted(cr.progress_m.round(1).tolist()))
print(f"track length ~{np.asarray(core.track.centerline.spline.s)[-1] if False else 189.0:.1f} m")
