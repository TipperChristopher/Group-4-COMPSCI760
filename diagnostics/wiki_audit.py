import io, re, glob, os, json, subprocess
import pandas as pd
W = "team_repo_heldout/feasibility_spike/wiki/"; R = "team_repo_heldout/"
pages = [q.replace(chr(92), "/") for q in sorted(glob.glob(W + "*.md"))] + [W + "raw-sources/index.md"]
names = {os.path.basename(p) for p in glob.glob(W + "*.md")}
txt = {p: io.open(p, encoding="utf-8").read() for p in pages}
issues = []
def add(sev, where, msg): issues.append((sev, where, msg))
# B. markdown links
for p, s in txt.items():
    for m in re.finditer(r"\]\(([^)#\s]+)(#[^)]+)?\)", s):
        tgt = os.path.normpath(os.path.join(os.path.dirname(p), m.group(1)))
        if not m.group(1).startswith("http") and not os.path.exists(tgt):
            add("BLOCKER", p, f"broken link {m.group(1)}")
    for m in re.finditer(r"\]\(#([^)]+)\)", s):
        heads = [re.sub(r"[^\w\- ]", "", h.lower()).strip().replace(" ", "-") for h in re.findall(r"^#+ (.+)$", s, re.M)]
        if m.group(1) not in heads: add("WARNING", p, f"anchor #{m.group(1)} not found")
# backtick page references must exist (wiki pages or known repo files)
known_repo = {"CONTEXT.md", "README.md", "RESUME.txt"}
for p, s in txt.items():
    for m in set(re.findall(r"`([a-z0-9\-]+\.md)`", s)):
        if m not in names and m not in known_repo and not glob.glob(R + "**/" + m, recursive=True) and not glob.glob("team_repo/**/" + m, recursive=True):
            add("WARNING", p, f"reference to non-existent `{m}`")
# index lists every page
idx = txt[W + "index.md"]
for n in names:
    if n not in ("index.md",) and n not in idx: add("BLOCKER", "index.md", f"page {n} not listed")
# frontmatter
for p, s in txt.items():
    b = os.path.basename(p)
    if b in ("index.md", "log.md", "SCHEMA.md") or "raw-sources" in p: continue
    fm = s.split("---")[1] if s.startswith("---") else ""
    for k in ("title:", "type:", "updated:", "sources:"):
        if k not in fm: add("BLOCKER", p, f"frontmatter missing {k}")
    if "updated: 2025" in fm: add("BLOCKER", p, "updated year 2025")
# orphans + see-also reciprocity
for n in names:
    inbound = [os.path.basename(q) for q, s in txt.items() if os.path.basename(q) != n and n in s]
    if not inbound: add("WARNING", n, "orphan (no inbound reference)")
for p, s in txt.items():
    if "## See also" not in s: 
        if os.path.basename(p) not in ("index.md", "log.md", "SCHEMA.md") and "raw-sources" not in p:
            add("WARNING", p, "no See also section")
        continue
    sa = s.split("## See also")[1]
    for tgt in set(re.findall(r"`([a-z0-9\-]+\.md)`", sa)):
        if W + tgt in txt and os.path.basename(p) not in txt[W + tgt]:
            add("INFO", p, f"See-also → {tgt} not reciprocated")
# retracted phrases still asserted without a correction on the same page
retracted = {
 "second ceiling": "corner sharpness refuted", "Stability comes from the algorithm": "γ correction",
 "--seed 0`": "wrong track seed", "Rerun command #3": "stale resume", "paused at 12/20": "sac 2M finished",
 "never modify the team's": "D3 amended", "ruled out": "overfitting wording",
 "γ is the lever": "interaction", "amplifies": "interaction"}
for p, s in txt.items():
    for ph, why in retracted.items():
        for m in re.finditer(re.escape(ph), s):
            window = s[max(0, m.start()-1200): m.end()+1200]
            if not re.search(r"Correction|Amended|Stale|refuted|REFUTED|correction|historical|~~|wrong|WRONG|retract|superseded|2026-10", window):
                add("WARNING", p, f"retracted claim '{ph}' ({why}) without nearby correction")
# lifecycle words
for p, s in txt.items():
    for m in re.finditer(r"\b(is running|in progress|not yet run|pending|paused)\b", s):
        line = s[:m.start()].count("\n") + 1
        add("INFO", f"{os.path.basename(p)}:{line}", f"lifecycle word '{m.group(1)}' — verify still true")
# B2 ground-truth spot checks
rs = pd.read_csv(R + "results/ppo_diagnosis/runs_summary.csv").set_index("run")
checks = [
 ("G999 last200k m", rs.loc["PPO_1tracks_s0_G999_gamma", "last200k_mean_m"], 234.8),
 ("G999 lap rate", round(100*rs.loc["PPO_1tracks_s0_G999_gamma", "last200k_lap_rate"]), 65),
 ("V4 last200k m", rs.loc["PPO_1tracks_s0_V4_nsteps", "last200k_mean_m"], 68.4),
 ("V0 last200k m", rs.loc["PPO_1tracks_s0_V0_baseline", "last200k_mean_m"], 36.4),
 ("P40 last200k m", rs.loc["PPO_1tracks_s0_P40_penalty40", "last200k_mean_m"], 71.7),
 ("G999L99 lap rate", round(100*rs.loc["PPO_1tracks_s0_G999L99_full", "last200k_lap_rate"]), 59),
]
curve = []
for st in [400000,600000,800000,1000000,1200000,1400000,1600000,1800000,2000000]:
    d = pd.read_csv(R + f"results/ppo_diagnosis/eval_g999_curve/ck{st}.csv"); curve.append(int(d[d.track.str.startswith("test_")].completed_lap.sum()))
checks.append(("G999 curve", curve, [0,0,10,0,6,5,4,0,0]))
for f, exp in [("oursac1200k_A-test-1.47m", 29), ("oursac1200k_B-narrowA-1.07m", 0), ("oursac2000k_A-test-1.47m", 27)]:
    checks.append((f, int(pd.read_csv(R + f"results/ppo_diagnosis/eval_width_oursac/{f}.csv").completed_lap.sum()), exp))
geo = json.load(open(R + "results/track_geometry/summary.json"))
gs = json.dumps(geo)
checks.append(("geometry has 0.0095 real sharper share", "0.0095" in gs or "0.00949" in gs or "0.0094" in gs, True))
for name, got, exp in checks:
    ok = (abs(got-exp) < 0.05) if isinstance(exp, float) else got == exp
    if not ok: add("BLOCKER", "ground-truth", f"{name}: wiki says {exp}, file says {got}")
    else: add("PASS", "ground-truth", f"{name} = {got}")
# report
from collections import Counter
c = Counter(i[0] for i in issues)
print("COUNTS", dict(c))
for sev in ("BLOCKER", "WARNING", "INFO", "PASS"):
    for i in issues:
        if i[0] == sev: print(f"{sev:8} {os.path.basename(i[1]) if '/' in i[1] else i[1]}: {i[2]}")
