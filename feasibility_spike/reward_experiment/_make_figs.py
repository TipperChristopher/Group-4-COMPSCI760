import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

NAVY="#0E1A2B"; PANEL="#0E1A2B"; LIGHT="#C3CAD6"; GREEN="#2ECB86"; BLUE="#3B82F6"; RED="#FF6B6B"; GREY="#2A3B54"
plt.rcParams.update({
    "figure.facecolor":NAVY, "axes.facecolor":PANEL, "savefig.facecolor":NAVY,
    "axes.edgecolor":"#3A4A63", "axes.labelcolor":LIGHT, "axes.titlecolor":"#FFFFFF",
    "xtick.color":LIGHT, "ytick.color":LIGHT, "text.color":LIGHT, "grid.color":GREY,
    "font.size":12, "axes.titleweight":"bold",
})

ppo=json.load(open("results/learning_curve_cp40_vn_1000000.json"))["curve"]
sac=json.load(open("results/learning_curve_cp40_sac_vn_1000000.json"))["curve"]
ps=[c["steps"]/1e3 for c in ppo]; pl=[c["laps"] for c in ppo]; pr=[c["ep_rew_mean"] for c in ppo]
ss=[c["steps"]/1e3 for c in sac]; sl=[c["laps"] for c in sac]; sr=[c["ep_rew_mean"] for c in sac]

# FIG 1: PPO vs SAC (laps + reward)
fig,ax=plt.subplots(1,2,figsize=(12,4.6))
ax[0].plot(ps,pl,"o-",color=RED,lw=2.4,label="PPO")
ax[0].plot(ss,sl,"s-",color=GREEN,lw=2.4,label="SAC")
ax[0].axhline(1.0,ls="--",c="#8892a3",alpha=.7); ax[0].text(60,1.05,"1 lap",color="#8892a3")
ax[0].axhline(2.0,ls=":",c=GREEN,alpha=.7); ax[0].text(60,2.05,"lap cap (2)",color=GREEN)
ax[0].set_xlabel("training steps (k)"); ax[0].set_ylabel("laps reached (deterministic eval)")
ax[0].set_title("SAC completes 2 laps; PPO plateaus then degrades",fontsize=12)
ax[0].legend(facecolor=NAVY,edgecolor="#3A4A63",labelcolor=LIGHT); ax[0].grid(alpha=.25)
ax[1].plot(ps,pr,"o-",color=RED,lw=2.4,label="PPO")
ax[1].plot(ss,sr,"s-",color=GREEN,lw=2.4,label="SAC")
ax[1].axhline(0,c="#8892a3",lw=1)
ax[1].set_xlabel("training steps (k)"); ax[1].set_ylabel("ep_rew_mean (training)")
ax[1].set_title("SAC reward climbs smoothly; PPO oscillates (+37..-19)",fontsize=12)
ax[1].legend(facecolor=NAVY,edgecolor="#3A4A63",labelcolor=LIGHT); ax[1].grid(alpha=.25)
plt.tight_layout(); plt.savefig("slides_assets/fig_ppo_vs_sac.png",dpi=150); plt.close()

# FIG 2: spawn fix before/after
def offcl(d,reset):
    rows=[r for r in d["results"]["gap"] if r["reset"]==reset]
    return float(np.mean([r["off_centreline"] for r in rows]))
sp=json.load(open("results/spawn_study_Spielberg.json")); si=json.load(open("results/spawn_study_Silverstone.json"))
labels=["Spielberg","Silverstone"]
before=[offcl(sp,"rl_grid_static"),offcl(si,"rl_grid_static")]
after=[offcl(sp,"cl_grid_static"),offcl(si,"cl_grid_static")]
x=np.arange(2); w=0.35
fig,ax=plt.subplots(figsize=(7.6,4.6))
b1=ax.bar(x-w/2,before,w,label="before: raceline spawn (near kerb)",color=RED)
b2=ax.bar(x+w/2,after,w,label="after: centreline spawn (matches training)",color=GREEN)
for b in list(b1)+list(b2): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.015,f"{b.get_height():.2f}",ha="center",fontsize=10,color=LIGHT)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("spawn distance off centreline (m)")
ax.set_title("Spawn fix: eval now starts on the centreline",fontsize=12)
ax.legend(facecolor=NAVY,edgecolor="#3A4A63",labelcolor=LIGHT,fontsize=9); ax.grid(axis="y",alpha=.25); ax.set_ylim(0,1.0)
plt.tight_layout(); plt.savefig("slides_assets/fig_spawn_fix.png",dpi=150); plt.close()

# FIG 3: SAC lap-time improving while completing
lt=[(c["steps"]/1e3,c["lap_time_s"]) for c in sac if c.get("lap_time_s")]
xs=[a for a,_ in lt]; ys=[b for _,b in lt]
fig,ax=plt.subplots(figsize=(7.6,4.3))
ax.plot(xs,ys,"s-",color=GREEN,lw=2.4)
for a,b in lt: ax.text(a,b+0.4,f"{b:.1f}s",ha="center",fontsize=10,color=LIGHT)
ax.set_xlabel("training steps (k)"); ax.set_ylabel("lap time (s) when completing")
ax.set_title("SAC learns CONTROLLED speed: completes AND gets faster (40.3 -> 24.7s)",fontsize=11)
ax.grid(alpha=.25); ax.invert_yaxis()
plt.tight_layout(); plt.savefig("slides_assets/fig_sac_laptime.png",dpi=150); plt.close()
print("figures restyled (dark theme)")
