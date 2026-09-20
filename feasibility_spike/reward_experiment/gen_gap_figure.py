"""Generalization-gap chart: baselines vs our RL policies per circuit, under the
IDENTICAL protocol (centreline spawn, seeds 0-4, 15000-step cap). Reads every
results/eval_*.json present, so it picks up models as their evals land.

    python gen_gap_figure.py
"""
import json, os, pathlib, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
R = HERE / "results"
NAVY="#0E1A2B"; LIGHT="#C3CAD6"; GREEN="#10B981"; BLUE="#3B82F6"; RED="#FF6B6B"; ORANGE="#F59E0B"; GREY="#2A3B54"
plt.rcParams.update({
    "figure.facecolor":NAVY, "axes.facecolor":NAVY, "savefig.facecolor":NAVY,
    "axes.edgecolor":"#3A4A63", "axes.labelcolor":LIGHT, "axes.titlecolor":"#FFFFFF",
    "xtick.color":LIGHT, "ytick.color":LIGHT, "text.color":LIGHT, "grid.color":GREY,
    "font.size":12, "axes.titleweight":"bold"})

SERIES = [  # (key-substring, label, color)
    ("random", "Random", GREY),
    ("gap",    "Gap-follower", ORANGE),
    ("ppo",    "PPO (ours)", BLUE),
    ("sac",    "SAC (ours)", GREEN),
]
data = {}
for fname in sorted(os.listdir(R)):
    if not (fname.startswith("eval_") and fname.endswith(".json")):
        continue
    d = json.load(open(R / fname))
    key = fname[len("eval_"):-len(".json")]
    for track, eps in d.get("results", {}).items():
        laps = [e["laps"] for e in eps]
        data.setdefault(key, {})[track] = (float(np.mean(laps)), float(np.std(laps)))

tracks = []
for key in data:
    for t in data[key]:
        if t not in tracks: tracks.append(t)
tracks = sorted(tracks, key=lambda t: ("real" in t or t in ("Spielberg","Silverstone"), t))
tracks = ["synthetic_track_0", "synthetic_track_1", "synthetic_track_2", "Spielberg", "Silverstone"]
tracks = [t for t in tracks if any(t in data[k] for k in data)]

x = np.arange(len(tracks)); n = len([s for s in SERIES if s[0] in data]); w = 0.8 / max(n, 1)
fig, ax = plt.subplots(figsize=(11.5, 5.0))
off = -(n - 1) / 2
for sub, label, color in SERIES:
    key = next((k for k in data if sub in k), None)
    if key is None: continue
    means = [data[key].get(t, (np.nan, 0))[0] for t in tracks]
    stds  = [data[key].get(t, (0, 0))[1] for t in tracks]
    ax.bar(x + off * w, means, w, label=label, color=color, yerr=stds, capsize=3, error_kw=dict(ecolor=LIGHT, alpha=.6))
    off += 1
ax.axhline(1.0, ls="--", c="#8892a3", alpha=.6); ax.text(-0.45, 1.03, "1 lap", color="#8892a3", fontsize=10)
ax.set_xticks(x); ax.set_xticklabels([t.replace("synthetic_track_", "syn-") for t in tracks], fontsize=11)
ax.set_ylabel("laps reached (mean over 5 spawn seeds, \u00b1std)")
ax.set_title("Zero-shot vs baselines, identical protocol \u2014 RL wins in-distribution, baseline wins on unseen real circuits", fontsize=12)
ax.legend(facecolor=NAVY, edgecolor="#3A4A63", labelcolor=LIGHT); ax.grid(axis="y", alpha=.25)
plt.tight_layout()
out = HERE / "slides_assets" / "fig_gen_gap.png"
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=150); plt.close()
print("saved", out)
for k in data: print(f"  {k}: tracks={list(data[k].keys())}")
