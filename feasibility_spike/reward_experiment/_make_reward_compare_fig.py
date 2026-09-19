import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
NAVY="#0E1A2B"; LIGHT="#C3CAD6"; GREEN="#10B981"; BLUE="#3B82F6"; RED="#FF6B6B"; ORANGE="#F59E0B"; GREY="#2A3B54"; WHITE="#F8FAFC"
plt.rcParams.update({"figure.facecolor":NAVY,"axes.facecolor":NAVY,"savefig.facecolor":NAVY,
    "axes.edgecolor":"#3A4A63","axes.labelcolor":LIGHT,"axes.titlecolor":WHITE,
    "xtick.color":LIGHT,"ytick.color":LIGHT,"text.color":LIGHT,"grid.color":GREY,
    "font.size":10.5,"axes.titleweight":"bold"})

# panel 1: formula scan (same rollout, both reward formulas)
scan=json.load(open("results/reward_scan2.json"))["rows"]
sp=[r["speed"] for r in scan]; old=[r["old_return"] for r in scan]; new=[r["new_return"] for r in scan]

# panel 2: three trained agents (laps vs steps, 0-2M)
def curve(f,label):
    c=json.load(open("results/"+f))["curve"]
    return [x["steps"]/1e3 for x in c], [x["laps"] for x in c], label

fig,ax=plt.subplots(1,2,figsize=(13.5,4.8))
ax[0].plot(sp,old,"o-",color=RED,lw=2.2,label="old (1/s alive + 10x dist - steer tax)")
ax[0].plot(sp,new,"s-",color=GREEN,lw=2.2,label="new (metres - crash)")
ax[0].annotate("standstill: 100 -> 0",xy=(0,100),xytext=(3,110),color=WHITE,fontsize=9,arrowprops=dict(arrowstyle="->",color=LIGHT))
ax[0].set_xlabel("commanded speed (m/s)"); ax[0].set_ylabel("episode return (same rollout)")
ax[0].set_title("Reward function: crawl was the old optimum",fontsize=11)
ax[0].legend(facecolor=NAVY,edgecolor="#3A4A63",labelcolor=LIGHT,fontsize=8.5); ax[0].grid(alpha=.25)

c1=curve("learning_curve_cp40_vn_old_2000000.json","OLD reward (V1)")
c2=curve("learning_curve_cp5_vn_2000000.json","NEW, penalty 5")
c3=curve("learning_curve_cp40_vn_2000000.json","NEW, penalty 40")
ax[1].plot(c1[0],c1[1],"o-",color=RED,lw=2,label=c1[2])
ax[1].plot(c2[0],c2[1],"^-",color=ORANGE,lw=2,label=c2[2])
ax[1].plot(c3[0],c3[1],"s-",color=BLUE,lw=2,label=c3[2])
ax[1].axhline(1.0,ls="--",c="#8892a3",alpha=.6)
ax[1].set_xlabel("training steps (k)"); ax[1].set_ylabel("laps reached (deterministic eval)")
ax[1].set_title("Trained agents: all three PPO variants plateau ~0.1 laps (reward choice does not rescue PPO)",fontsize=11)
ax[1].legend(facecolor=NAVY,edgecolor="#3A4A63",labelcolor=LIGHT,fontsize=8.5); ax[1].grid(alpha=.25)
plt.tight_layout(); plt.savefig("slides_assets/fig_reward_compare.png",dpi=150); plt.close()
print("saved fig_reward_compare.png")

# refresh fig_ppo_vs_sac with PPO-2M + SAC-1M
sac=json.load(open("results/learning_curve_cp40_sac_vn_1000000.json"))["curve"]
fig,ax=plt.subplots(1,2,figsize=(12,4.6))
ax[0].plot(c3[0],c3[1],"s-",color=BLUE,lw=2.4,label="PPO (2M)")
ax[0].plot([x["steps"]/1e3 for x in sac],[x["laps"] for x in sac],"o-",color=GREEN,lw=2.4,label="SAC (1M; 2M running)")
ax[0].axhline(1.0,ls="--",c="#8892a3",alpha=.7); ax[0].axhline(2.0,ls=":",c=GREEN,alpha=.7)
ax[0].set_xlabel("training steps (k)"); ax[0].set_ylabel("laps reached (deterministic eval)")
ax[0].set_title("SAC completes 2 laps; PPO plateaus then degrades",fontsize=11.5)
ax[0].legend(facecolor=NAVY,edgecolor="#3A4A63",labelcolor=LIGHT); ax[0].grid(alpha=.25)
ppo40=json.load(open("results/learning_curve_cp40_vn_2000000.json"))["curve"]
ax[1].plot([x["steps"]/1e3 for x in ppo40],[x["ep_rew_mean"] for x in ppo40],"s-",color=BLUE,lw=2.4,label="PPO (2M)")
ax[1].plot([x["steps"]/1e3 for x in sac],[x["ep_rew_mean"] for x in sac],"o-",color=GREEN,lw=2.4,label="SAC (1M)")
ax[1].axhline(0,c="#8892a3",lw=1)
ax[1].set_xlabel("training steps (k)"); ax[1].set_ylabel("ep_rew_mean (training)")
ax[1].set_title("SAC reward climbs smoothly; PPO oscillates",fontsize=11.5)
ax[1].legend(facecolor=NAVY,edgecolor="#3A4A63",labelcolor=LIGHT); ax[1].grid(alpha=.25)
plt.tight_layout(); plt.savefig("slides_assets/fig_ppo_vs_sac.png",dpi=150); plt.close()
print("saved fig_ppo_vs_sac.png (PPO 2M + SAC 1M)")
