"""Rebuild ONLY our inserted slides in Group4_ProjectUpdate_v2.pptx with the
corrected, smoother flow. All original slides are kept and never modified;
our old (wrong scoring-flip) slide is replaced by accurate ones."""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
AST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "slides_assets")
SRC = os.path.join(ROOT, "_v2_build_src.pptx")
OUT = os.path.join(ROOT, "Group4_ProjectUpdate_v3_latest.pptx")

WHITE=RGBColor(0xF8,0xFA,0xFC); GREEN=RGBColor(0x10,0xB9,0x81); BLUE=RGBColor(0x3B,0x82,0xF6)
BODY=RGBColor(0xCB,0xD5,0xE1); MUTED=RGBColor(0x94,0xA3,0xB8)
TFONT="Montserrat"; BFONT="Inter"

prs = Presentation(SRC)
BLANK = min(prs.slide_layouts, key=lambda L: len(L.placeholders))

def new_slide():
    s = prs.slides.add_slide(BLANK)
    for ph in list(s.placeholders):
        ph._element.getparent().remove(ph._element)
    s.shapes.add_picture(os.path.join(AST, "bg.png"), 0, 0, width=prs.slide_width, height=prs.slide_height)
    return s

def title(s, txt, size=28):
    tb = s.shapes.add_textbox(Inches(0.62), Inches(0.55), Inches(12.1), Inches(1.0))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; r = p.add_run(); r.text = txt
    r.font.name = TFONT; r.font.size = Pt(size); r.font.bold = True; r.font.color.rgb = WHITE

def bullets(s, items, top=1.65, left=0.65, width=12.0, size=14.5, gap=9):
    tb = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(5.3))
    tf = tb.text_frame; tf.word_wrap = True
    for i, (lead, txt) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.space_after = Pt(gap)
        r1 = p.add_run(); r1.text = "›  "; r1.font.name = BFONT; r1.font.bold = True; r1.font.size = Pt(size); r1.font.color.rgb = GREEN
        if lead:
            rl = p.add_run(); rl.text = lead; rl.font.name = BFONT; rl.font.bold = True; rl.font.size = Pt(size); rl.font.color.rgb = WHITE
        r2 = p.add_run(); r2.text = txt; r2.font.name = BFONT; r2.font.size = Pt(size); r2.font.color.rgb = BODY

def caption(s, txt, top=5.75, size=13, left=0.7, width=12.2, color=BODY):
    tb = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(1.5)); tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; r = p.add_run(); r.text = txt; r.font.name = BFONT; r.font.size = Pt(size); r.font.color.rgb = color

def img(s, name, left, top, width):
    path = os.path.join(AST, name)
    if os.path.exists(path): s.shapes.add_picture(path, Inches(left), Inches(top), width=Inches(width))

def notes(s, t): s.notes_slide.notes_text_frame.text = t

# N1: Evaluation Protocol refinements (methodology)
s = new_slide(); title(s, "Evaluation Protocol \u2014 Our Refinements")
bullets(s, [
 ("Splits: ", "TRAIN synthetic \u00b7 VALIDATION held-out synthetic (checkpoint choice) \u00b7 TEST 23 real circuits (zero-shot)."),
 ("Metric: ", "graded fractional laps (not 0/1 completion, a 0% floor) + completion + lap time + crash rate."),
 ("Selection: ", "BEST checkpoint on validation \u2014 PPO's final policy degrades to 0.10 after peaking at 0.34, so the final checkpoint is a noisy estimator."),
 ("Spawn: ", "eval on the centreline + 5 FIXED spawn seeds (0-4), averaged per circuit \u2014 identical for every policy & baseline (matches training; reproducible, verified through VecNormalize)."),
 ("Frozen for both algos: ", "reward, crash penalty, VecNormalize (obs-only), SB3 defaults. Compute: PPO ~40 min/1M, SAC ~3.5-4 h/1M (CPU)."),
])
notes(s, "OURS. INSERT after 'Datasets & Evaluation Protocol'. WHY best-checkpoint: PPO degrades 0.34->0.10; final policy is a poor estimator. RUBRIC: methodology + dataset + experimental design.")

# N2: Spawn fix (bug 4) + before/after figure
s = new_slide(); title(s, "Bug 4 \u2014 Train/Eval Spawn Mismatch (Our Fix)")
bullets(s, [
 ("Root cause: ", "eval spawned on the raceline (~0.8 m off-centre, ~0.3 m from the kerb); training used the centreline."),
 ("What it actually did (Silverstone): ", "the raceline spawn was accidentally RESCUING the reactive baseline on 2 of 5 seeds. The centreline fix removes that lottery \u2014 it now fails deterministically at the same 81 m hairpin, every seed."),
 ("Spielberg: ", "stable 5/5 either way (open start straight)."),
 ("So the fix ", "does not rescue a weak policy \u2014 it exposes the TRUE, reproducible performance. That is what a fair protocol is for."),
], top=1.6, width=5.9, size=13.5)
img(s, "fig_spawn_outcomes.png", 6.9, 2.35, 6.2)
notes(s, "OURS. HONEST: Silverstone-before bar uses the TEAM's harness CSV (crash,crash,crash,LAP,LAP) because the seed->spawn mapping is harness-dependent (RNG draw order); our aligned re-run reproduces the AFTER failure (5/5 crash at 80.7-80.9 m) and Spielberg 5/5 both. Also note: the ~1 m start line yields only 3-4 DISTINCT spawns, so '5 seeds' weights ~3 positions, not 5. RUBRIC: changes (3pts).")

# N3: PPO vs SAC results + figure
s = new_slide(); title(s, "Pilot Results: PPO vs SAC (single track, 1M steps)")
img(s, "fig_ppo_vs_sac.png", 0.85, 1.5, 11.6)
caption(s, "Same reward, budget, obs-normalization: SAC completes 2 laps and gets faster (lap time 40.3 \u2192 24.7 s); PPO peaks at 0.34 laps then degrades to 0.10, reward oscillating +37..\u221219. (Pilots n=1 \u2014 diagnostic, not the grid.)")
notes(s, "OURS. RESULTS. INSERT after baselines. WHAT WE DID: segmented training + deterministic per-checkpoint eval; reproducible. Single-track capability, not the generalization result. RUBRIC: results (3pts).")

# N4: Why PPO + how to fix (ties to the survey's prediction)
s = new_slide(); title(s, "Our Analysis: Why PPO Can't Hold a Policy \u2014 and the Fix")
bullets(s, [
 ("The survey predicted this: ", "under a FIXED step budget PPO and SAC 'reuse experience differently' \u2014 SAC's replay buffer learns more per env-step; PPO's on-policy updates discard data after one use."),
 ("What that looks like: ", "PPO peaks at 0.34 laps then forgets it (no replay memory); SAC accumulates and completes 2 laps."),
 ("Parallelism is NOT a free fix: ", "with steps fixed, more envs add no data \u2014 updates drop from ~976 (1 env) to ~122 (8 envs). It only trades update count for lower variance."),
 ("Real levers under the frozen protocol: ", "dense reward shaping that rewards braking (clearance / racing-line term), and reporting best-checkpoint rather than final."),
 ("Takeaway: ", "a fixed-budget comparison structurally favours replay-buffer algorithms \u2014 exactly what the proposal anticipated, and a fair property of the study."),
])
notes(s, "OURS. ANALYSIS. Say: this result CONFIRMS the survey's prediction rather than surprising us. Q&A-proof: 'why didn't you give PPO more envs?' -> fixed budget: more envs = fewer updates, no extra data. RUBRIC: results + Q&A depth.")

# N5: Generalization gap vs baseline
s = new_slide(); title(s, "Results vs Baseline: The Generalization Gap")
bullets(s, [
 ("In-distribution (train track): ", "SAC completes it (24.7 s/lap); gap-follower and random both crash."),
 ("Unseen real circuit (Spielberg): ", "the no-learning gap-follower completes it (67 s/lap); our PPO-5-tracks zero-shot crashes at 0.11 laps."),
 ("Unseen real circuit (Silverstone): ", "both crash \u2014 the baseline dies at the same hairpin every seed."),
 ("The story: ", "RL wins in-distribution; the reactive baseline wins out-of-distribution. Closing that gap is exactly what the 1/5/20/100 diversity sweep is for."),
])
notes(s, "OURS. RESULTS. HONEST labels: SAC numbers = single-track pilot; real-circuit rows = PPO_5tracks zero-shot. SAC's first-ever unseen-track eval runs as soon as its save-run finishes. RUBRIC: results (3pts).")

# N6: Conclusions & next steps
s = new_slide(); title(s, "Conclusions & Next Steps")
bullets(s, [
 ("De-risked: ", "FOUR bugs found & fixed, reward + spawn protocol frozen, baselines set, eval protocol formalized."),
 ("Pilots: ", "SAC completes & is stable; PPO peaks early then forgets (no replay memory) \u2014 with a concrete fix path."),
 ("Gap: ", "zero-shot RL still loses to the reactive baseline on unseen circuits \u2014 the diversity grid is the test."),
 ("Next: ", "freeze ONE setup (merge reward, freeze crash penalty + 2M budget + spawn), run 2\u00d74\u00d7\u22653-seed grid with bootstrap CIs."),
 ("Honest status: ", "current numbers are pilots (n=1); the grid is not yet run."),
])
notes(s, "OURS. CONCLUSION. INSERT after Timeline. RUBRIC: timeline + honesty + progress.")

# ---- rebuild order: originals kept, ours re-placed ----
sldIdLst = prs.slides._sldIdLst
ids = list(sldIdLst)
o = ids[:13]; ns = ids[13:]
# o: 0 Title 1 Motiv 2-4 Lit 5 RQ 6 Datasets 7 Bugs12 8 Baselines 9 RewardCrawl 10 Timeline 11 Roles 12 ImgSrc
# ns: [N1 protocol, N2 spawn, N3 ppovssac, N4 whyfix, N5 gap, N6 concl]
desired = (o[0:7] + [ns[0]] + o[7:8] + [o[9]] + [ns[1]] + [o[8]] + [ns[2], ns[3], ns[4]] + [o[10]] + [ns[5]] + o[11:13])
for e in ids: sldIdLst.remove(e)
for e in desired: sldIdLst.append(e)

prs.save(OUT)
print("saved:", OUT, "| total slides:", len(prs.slides._sldIdLst), "(13 originals + 6 ours)")
