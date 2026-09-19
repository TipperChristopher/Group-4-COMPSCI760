import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
NAVY="#0E1A2B"; LIGHT="#C3CAD6"; GREEN="#10B981"; BLUE="#3B82F6"; RED="#FF6B6B"; GREY="#2A3B54"; WHITE="#F8FAFC"
plt.rcParams.update({"figure.facecolor":NAVY,"axes.facecolor":NAVY,"savefig.facecolor":NAVY,
    "axes.edgecolor":"#3A4A63","axes.labelcolor":LIGHT,"axes.titlecolor":WHITE,
    "xtick.color":LIGHT,"ytick.color":LIGHT,"text.color":LIGHT,"grid.color":GREY,
    "font.size":11,"axes.titleweight":"bold"})
d=json.load(open("results/reward_scan2.json")); rows=d["rows"]
sp=[r["speed"] for r in rows]; old=[r["old_return"] for r in rows]; new=[r["new_return"] for r in rows]
fig,ax=plt.subplots(1,2,figsize=(13,4.6))
ax[0].plot(sp,old,"o-",color=RED,lw=2.4,label="old reward")
ax[0].annotate("standstill = 100.00",xy=(0,100),xytext=(2.8,135),arrowprops=dict(arrowstyle="->",color=LIGHT),color=WHITE,fontsize=10)
ax[0].annotate("crawl 0.25 m/s = 263 (peak)",xy=(0.25,263),xytext=(3.5,250),arrowprops=dict(arrowstyle="->",color=LIGHT),color=WHITE,fontsize=10)
ax[0].annotate("20 m/s = 189",xy=(20,188.96),xytext=(13,225),arrowprops=dict(arrowstyle="->",color=LIGHT),color=WHITE,fontsize=10)
ax[0].set_xlabel("commanded speed (m/s)"); ax[0].set_ylabel("episode return (10k-step cap)")
ax[0].set_title("OLD (1/s alive + 0.1x speed - 0.5x|steer|): slower = higher return",fontsize=10.5)
ax[0].grid(alpha=.25)
ax[1].plot(sp,new,"s-",color=GREEN,lw=2.4,label="new reward")
ax[1].annotate("standstill = 0.00",xy=(0,0),xytext=(2.8,6),arrowprops=dict(arrowstyle="->",color=LIGHT),color=WHITE,fontsize=10)
ax[1].annotate("every crashing speed: 18 m - 40 = -22 (speed-invariant)",xy=(10,-22),xytext=(2.2,-12),arrowprops=dict(arrowstyle="->",color=LIGHT),color=WHITE,fontsize=10)
ax[1].set_xlabel("commanded speed (m/s)"); ax[1].set_ylabel("episode return (10k-step cap)")
ax[1].set_title("NEW (1.0x metres - 40 crash): time pays nothing, metres pay",fontsize=10.5)
ax[1].grid(alpha=.25)
plt.tight_layout(); plt.savefig("slides_assets/fig_reward_scan.png",dpi=150); plt.close()
print("saved slides_assets/fig_reward_scan.png")
