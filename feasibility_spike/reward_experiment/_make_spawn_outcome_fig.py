"""Spawn fix: geometry AND before/after outcomes in one figure.
Panel 1: spawn offset off-centreline (fix took effect).
Panel 2: actual laps + crash counts before vs after (did it fix crashing?)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE_ABS = __import__("pathlib").Path(__file__).parent
NAVY="#0E1A2B"; LIGHT="#C3CAD6"; GREEN="#10B981"; BLUE="#3B82F6"; RED="#FF6B6B"; GREY="#2A3B54"
plt.rcParams.update({
    "figure.facecolor":NAVY, "axes.facecolor":NAVY, "savefig.facecolor":NAVY,
    "axes.edgecolor":"#3A4A63", "axes.labelcolor":LIGHT, "axes.titlecolor":"#FFFFFF",
    "xtick.color":LIGHT, "ytick.color":LIGHT, "text.color":LIGHT, "grid.color":GREY,
    "font.size":11, "axes.titleweight":"bold"})

def offcl(d, reset):
    rows=[r for r in d["results"]["gap"] if r["reset"]==reset]
    return float(np.mean([r["off_centreline"] for r in rows]))

sp=json.load(open(HERE_ABS/"results"/"spawn_study_Spielberg.json"))
si=json.load(open(HERE_ABS/"results"/"spawn_study_Silverstone.json"))
fig, axs = plt.subplots(1, 2, figsize=(13, 4.7))

# --- panel 1: geometry ---
labels=["Spielberg","Silverstone"]
before=[offcl(sp,"rl_grid_static"), offcl(si,"rl_grid_static")]
after =[offcl(sp,"cl_grid_static"),  offcl(si,"cl_grid_static")]
x=np.arange(2); w=0.35
b1=axs[0].bar(x-w/2,before,w,label="before: raceline spawn",color=RED)
b2=axs[0].bar(x+w/2,after,w,label="after: centreline spawn",color=GREEN)
for b in list(b1)+list(b2):
    axs[0].text(b.get_x()+b.get_width()/2, b.get_height()+0.015, f"{b.get_height():.2f}", ha="center", fontsize=10)
axs[0].set_xticks(x); axs[0].set_xticklabels(labels)
axs[0].set_ylabel("off-centreline (m)"); axs[0].set_ylim(0,1.0)
axs[0].set_title("Fix took effect: spawn moves to the centreline", fontsize=11)
axs[0].legend(facecolor=NAVY, edgecolor="#3A4A63", labelcolor=LIGHT, fontsize=9); axs[0].grid(axis="y", alpha=.25)

# --- panel 2: outcomes (laps + crash counts) ---
# Silverstone BEFORE uses the TEAM's harness CSV (desmond/reward-and-episode-fix):
# rl_grid_static: crash, crash, crash, LAP, LAP  (raceline spawn rescued 2/5 seeds).
# Our harness's seed->spawn mapping differs (RNG draw order), so we reproduce the
# AFTER failure (5/5 crash at the same hairpin) but not their before-bimodality.
def outcome_rows(d, policy, reset):
    return [r for r in d["results"][policy] if r["reset"]==reset]

groups = []
for name, d in (("Gap\nSpielberg", sp),):
    for reset, tag in (("rl_grid_static","before"), ("cl_grid_static","after")):
        rows = outcome_rows(d, "gap", reset)
        groups.append((f"{name}\n{tag}", float(np.mean([r['laps'] for r in rows])),
                       sum(1 for r in rows if r["outcome"]=="crash"), "ours"))
# Silverstone: before from team CSV, after from our aligned repro
rows_after = outcome_rows(si, "gap", "cl_grid_static")
groups.append(("Gap\nSilverstone\nbefore", 0.4, 3, "teamCSV"))   # 3 crash + 2 lap = mean 0.4 laps
cr_after = sum(1 for r in rows_after if r["outcome"]=="crash")
groups.append(("Gap\nSilverstone\nafter", float(np.mean([r['laps'] for r in rows_after])), cr_after, "ours"))
rl = json.load(open(HERE_ABS/"results"/"rl_before_after_PPO.json"))
for name, track in (("PPO\nSpielberg","Spielberg"), ("PPO\nSilverstone","Silverstone")):
    for reset, tag in (("rl_grid_static","before"), ("cl_grid_static","after")):
        rows = [r for r in rl["results"][track] if r["reset"]==reset]
        groups.append((f"{name}\n{tag}", float(np.mean([r['laps'] for r in rows])),
                       sum(1 for r in rows if r["outcome"]=="crash"), "ours"))
gx = np.arange(len(groups))
bars = axs[1].bar(gx, [g[1] for g in groups], 0.6,
                  color=[RED if "before" in g[0] else GREEN for g in groups])
for g, b in zip(groups, bars):
    note = " (team CSV)" if g[3]=="teamCSV" else ""
    axs[1].text(b.get_x()+b.get_width()/2, b.get_height()+0.02,
                f"{g[1]:.2f}\n{g[2]}/5 crash{note}", ha="center", fontsize=8)
axs[1].set_xticks(gx); axs[1].set_xticklabels([g[0] for g in groups], fontsize=9)
axs[1].set_ylabel("laps reached (mean, 5 seeds)")
axs[1].set_title("Silverstone: raceline spawn was RESCUING 2/5 seeds; centreline removes the lottery", fontsize=11)
axs[1].grid(axis="y", alpha=.25)
plt.tight_layout()
out = HERE_ABS / "slides_assets" / "fig_spawn_outcomes.png"
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=150); plt.close()
print("saved", out)
